"""Ground truth for the VAD-lines stream (`test_e2e_vad_line_stream.cpp`, ADR-127): Moonshine's own streaming
loop for a non-streaming recogniser (moonshine-ai/moonshine `core/transcriber.cpp` + `core/voice-activity-
detector.cpp`, main as of 2026-11-14), ported line for line, over the same Silero VAD (`silero_vad.onnx`
sha256 1a153a22..., the file Moonshine pins and the zoo's silero-vad export was made from).

The loop, per `transcribe_stream` (called after every `add_audio_to_stream` push):
  * the pushed audio waits until at least `transcription_interval` (0.5 s, compared in float) has gathered,
    or the stream has stopped;
  * the VAD takes it in hops of 512 samples (a remainder waits for the next call; at stop it is dropped):
    Silero's probability per hop (64 samples of context, state carried), averaged over a ring of the last 16
    (`ceil(0.5 s * 16000 / 512)`, summed in float in ring order), scaled down linearly over the last third of
    a 15 s line, and `> 0.5` is voice;
  * voice starting opens a line with the last 8192 samples (the hop included); voice continuing appends the
    hop; voice ending appends the hop and closes the line; stopping closes an open line;
  * every line touched by that call is transcribed from scratch -- open ones too (`decode_incomplete_lines`).

The recogniser is a HF checkpoint run by transformers (`--asr hf:<dir>`), greedy, with Moonshine's own token
budget `ceil(seconds * 6.5)` in float when `--budget moonshine` (its transcriber's `max_tokens_per_second`);
or `--asr none`, lines only.

Per clip and push size, in `<out>/<clip>.<push>/`:
    probs.npy     (H,)    f32 -- Silero's probability per hop (the stream's whole clip, hop by hop)
    smoothed.npy  (H,)    f32 -- what was compared to the threshold, per hop
    updates.txt               -- per transcribe call that ran: `call <i> <samples pushed>` then a line per
                                 touched segment: `line <k> <first sample> <n samples> <complete>`, then a
                                 tab and its text (nothing after the tab with `--asr none`)
    lines.txt                 -- the final lines: `<k> <first sample> <n samples> <complete>`, a tab, the text
    ids/<call>_<k>.npy        -- the recogniser's ids for that line at that call (with an `--asr`)

    python scripts/vad_lines_reference.py <silero_vad.onnx> <out> --asr hf:<dir>|none [--budget moonshine]
        --push <samples>[,...] <clip.npy> [...]
"""
import argparse
import math
from pathlib import Path

import numpy as np

RATE = 16000
HOP = 512
CONTEXT = 64
THRESHOLD = np.float32(0.5)
WINDOW = int(math.ceil((0.5 * RATE) / HOP))  # vad_window_size_from_duration(0.5f, 512), float math
LOOK_BEHIND = 8192
MAX_SEGMENT = int(np.float32(15.0) * RATE)  # vad_sample_count_from_duration(15.0f)
INTERVAL = np.float32(0.5)


class Silero:
    def __init__(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)

    def predict(self, hop):
        x = np.concatenate([self.context, hop]).astype(np.float32)[None, :]
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": np.array(RATE, np.int64)})
        self.context = x[0, -CONTEXT:].copy()
        return np.float32(out.reshape(-1)[0])


class Vad:
    """`VoiceActivityDetector`, its state and its three events."""

    def __init__(self, silero):
        self.silero = silero
        self.silero.reset()
        self.window = [np.float32(0.0)] * WINDOW
        self.window_index = 0
        self.segment = np.zeros(0, np.float32)
        self.look_behind = np.zeros(LOOK_BEHIND, np.float32)
        self.remainder = np.zeros(0, np.float32)
        self.previous = False
        self.processed = 0
        self.segments = []  # dicts: start, audio, complete, updated
        self.probs, self.smoothed = [], []

    def process(self, audio):
        for s in self.segments:
            s["updated"] = False
        buf = np.concatenate([self.remainder, audio.astype(np.float32)])
        while len(buf) >= HOP:
            self._hop(buf[:HOP])
            buf = buf[HOP:]
        self.remainder = buf

    def _hop(self, hop):
        self.processed += HOP
        self.look_behind = np.concatenate([self.look_behind[HOP:], hop])
        p = self.silero.predict(hop)
        self.probs.append(p)
        self.window[self.window_index] = p
        self.window_index = (self.window_index + 1) % WINDOW
        acc = np.float32(0.0)
        for v in self.window:  # std::accumulate in float, ring order
            acc = np.float32(acc + v)
        smoothed = np.float32(acc / np.float32(WINDOW))
        fade = (MAX_SEGMENT * 2) // 3
        if MAX_SEGMENT and len(self.segment) > fade:
            factor = np.float32(np.float32(len(self.segment) - fade) / np.float32(fade))
            smoothed = np.float32(smoothed * factor)
        self.smoothed.append(smoothed)
        voice = bool(smoothed > THRESHOLD)
        if voice and not self.previous:
            n = min(LOOK_BEHIND, self.processed)
            self.segment = self.look_behind[LOOK_BEHIND - n:].copy()
            self.segments.append({"start": self.processed - len(self.segment), "audio": self.segment.copy(),
                                  "complete": False, "updated": True})
        elif not voice and self.previous:
            self.segment = np.concatenate([self.segment, hop])
            self._update(complete=True)
            self.segment = np.zeros(0, np.float32)
            # `look_behind_audio_buffer.resize(look_behind_sample_count, 0.0f)` is a no-op: same size.
        elif voice and self.previous:
            self.segment = np.concatenate([self.segment, hop])
            self._update(complete=False)
        self.previous = voice

    def _update(self, complete):
        s = self.segments[-1]
        s["audio"], s["complete"], s["updated"] = self.segment.copy(), complete, True

    def stop(self):
        if self.previous:
            self._update(complete=True)
            self.previous = False


class HfAsr:
    def __init__(self, model_dir, budget):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, dtype=torch.float32).eval()
        self.budget = budget

    def __call__(self, audio):
        inputs = self.processor(audio, sampling_rate=RATE, return_tensors="pt")
        kwargs = {"do_sample": False, "num_beams": 1}
        if self.budget == "moonshine":
            # `ceil(audio_duration * max_tokens_per_second)` in float, decode steps after the start token.
            kwargs["max_new_tokens"] = int(math.ceil(np.float32(np.float32(len(audio)) / np.float32(RATE))
                                                     * np.float32(6.5)))
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, **kwargs)[0].tolist()
        text = self.processor.batch_decode([ids], skip_special_tokens=True)[0]
        return ids, text


def run(silero, asr, audio, push):
    vad = Vad(silero)
    pending = np.zeros(0, np.float32)
    calls, texts = [], {}
    pushed = 0
    pieces = [audio[i:i + push] for i in range(0, len(audio), push)]
    for i, piece in enumerate(pieces + [None]):
        stopped = piece is None
        if not stopped:
            pending = np.concatenate([pending, piece])
            pushed += len(piece)
        long_enough = np.float32(len(pending)) / np.float32(RATE) >= INTERVAL
        if not ((long_enough or stopped) and len(pending) > 0):
            if stopped:
                vad.stop()  # marks lines complete; nothing is re-transcribed
            continue
        if stopped:
            vad.process(pending)
            vad.stop()
        else:
            vad.process(pending)
        pending = np.zeros(0, np.float32)
        touched = []
        for k, s in enumerate(vad.segments):
            if not s["updated"]:
                continue
            ids, text = asr(s["audio"]) if asr else ([], "")
            texts[k] = (s["start"], len(s["audio"]), text)
            touched.append((k, s["start"], len(s["audio"]), int(s["complete"]), ids, text.strip()))
        calls.append((len(calls), pushed, touched))
    # Lines closed by a stop with nothing new keep the text they had.
    texts = {k: (s["start"], len(s["audio"]), texts.get(k, (0, 0, ""))[2], s["complete"])
             for k, s in enumerate(vad.segments)}
    return vad, calls, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("silero")
    ap.add_argument("out")
    ap.add_argument("--asr", required=True)
    ap.add_argument("--budget", default="model")
    ap.add_argument("--push", required=True)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    silero = Silero(a.silero)
    asr = None if a.asr == "none" else HfAsr(a.asr.removeprefix("hf:"), a.budget)
    for clip in a.clips:
        audio = np.load(clip).reshape(-1).astype(np.float32)
        for push in (int(p) for p in a.push.split(",")):
            vad, calls, texts = run(silero, asr, audio, push)
            d = Path(a.out) / f"{Path(clip).stem}.{push}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "probs.npy", np.asarray(vad.probs, np.float32))
            np.save(d / "smoothed.npy", np.asarray(vad.smoothed, np.float32))
            (d / "ids").mkdir(exist_ok=True)
            with open(d / "updates.txt", "w") as f:
                for i, pushed, touched in calls:
                    f.write(f"call {i} {pushed}\n")
                    for k, start, n, complete, ids, text in touched:
                        f.write(f"line {k} {start} {n} {complete}\t{text}\n")
                        if asr:
                            np.save(d / "ids" / f"{i}_{k}.npy", np.asarray(ids, np.float32))
            with open(d / "lines.txt", "w") as f:
                for k in sorted(texts):
                    start, n, text, complete = texts[k]
                    f.write(f"{k} {start} {n} {int(complete)}\t{text.strip()}\n")
            margin = float(np.min(np.abs(np.asarray(vad.smoothed) - THRESHOLD))) if vad.smoothed else 0.0
            print(f"{d.name}: {len(vad.segments)} lines, {len(calls)} calls, threshold margin {margin:.2e}")
            for k in sorted(texts):
                start, n, text, _ = texts[k]
                print(f"  [{start / RATE:6.2f} +{n / RATE:5.2f}] {text}")


if __name__ == "__main__":
    main()
