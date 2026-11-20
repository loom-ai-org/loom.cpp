"""The native engine's answers, for the web gates to compare against (ADR-139).

Runs each door through loom-py (the native engine, LuaJIT) and writes what came back:

    python web/test/make_reference.py OUT_DIR \
        --chat gemma-3-270m-it.gguf --asr moonshine-tiny-q8_0.gguf \
        --tts pocket-tts-2m-f32.gguf:voices/alba.gguf --phoneme-tts paradee-8m-f32.gguf:af_heart

OUT_DIR/reference.json holds the texts and the case list, each waveform OUT_DIR/<name>.f32 (raw
little-endian float32). The JS gates (`node web/test/node_gate.mjs`, `web/test/browser_gate.mjs`)
read the same cases.

Greedy text must match token for token. A waveform must have the native sample count and be within its
case's `min_snr_db`: LuaJIT and PUC Lua give the same numbers and the seeded draws are the same on every
platform (ADR-141), but WASM SIMD and x86 sum products in different orders.
"""
import argparse
import array
import json
import sys
import wave
from pathlib import Path

import loom

ROOT = Path(__file__).resolve().parents[2]
JFK = ROOT / "samples" / "jfk.wav"
CHAT = [{"role": "user", "content": "Write one sentence about the sea."}]
SPEECH = "Hello from loom, running in a browser."
PHONEME_TEXT = "Hello from loom."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--chat")
    p.add_argument("--asr")
    p.add_argument("--tts", help="MODEL.gguf[:VOICE_FILE.gguf]")
    p.add_argument("--phoneme-tts", help="MODEL.gguf:VOICE_NAME, a phoneme-input TTS (phonemised here)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = []

    if args.chat:
        m = loom.Model.from_file(args.chat)
        text = m.chat(CHAT, max_new_tokens=24, temperature=0)
        cases.append({"name": "chat", "door": "text2text", "model": str(Path(args.chat).resolve()),
                      "messages": CHAT, "max_new_tokens": 24, "text": text})
        print("chat:", repr(text), file=sys.stderr)
        del m

    if args.asr:
        m = loom.Model.from_file(args.asr)
        with wave.open(str(JFK)) as w:
            raw = w.readframes(w.getnframes())
        samples = [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
        text = m.transcribe(samples).text
        cases.append({"name": "asr", "door": "speech2text", "model": str(Path(args.asr).resolve()),
                      "audio": str(JFK), "text": text})
        print("asr:", repr(text), file=sys.stderr)
        del m

    for spec, phonemes in ((args.tts, False), (args.phoneme_tts, True)):
        if not spec:
            continue
        path, _, voice = spec.partition(":")
        m = loom.Model.from_file(path)
        case = {"name": "phoneme_tts" if phonemes else "tts", "door": "text2speech",
                "model": str(Path(path).resolve()),
                # Give F32 files: every matmul then runs in f32 on both builds (WebGPU: ggml-0065), and the
                # web output is within 78-92 dB of native. Paradee is one pass; Pocket-TTS draws its latents
                # frame by frame, so differences compound (its sample count, every frame's stop decision,
                # must still match exactly). A quantised TTS on a GPU differs by more than this, by design:
                # its matmuls round activations to f16 where the CPU quantises them to Q8 (ADR-139).
                "min_snr_db": 60.0 if phonemes else 50.0}
        kwargs = {}
        if voice and voice.endswith(".gguf"):
            vpath = (Path(path).parent / voice).resolve()
            kwargs["voice"] = m.voice(str(vpath))
            case["voice_file"] = str(vpath)
        elif voice:
            kwargs["voice"] = voice
            case["voice"] = voice
        if phonemes:
            # The G2P is loom-py's (Python only); the web door takes its output as `phonemes`.
            from loom import phonemizers
            lang = (m.contract.get("voice_languages") or ["en-US"])[m.voices.index(voice)] if voice else "en-US"
            ipa = phonemizers.phonemize_styled(PHONEME_TEXT, m.text2speech._phoneme_style(),
                                               alphabet=m.contract.get("phoneme_alphabet") or "ipa", language=lang)
            audio = m.text2speech.infer(phonemes=ipa, **kwargs)
            case["phonemes"] = ipa
        else:
            audio = m.text2speech.infer(SPEECH, **kwargs)
            case["text"] = SPEECH

        array.array("f", audio.samples).tofile(open(out / f"{case['name']}.f32", "wb"))
        case["samples"] = len(audio.samples)
        case["sample_rate"] = audio.sample_rate
        cases.append(case)
        print(f"{case['name']}: {len(audio.samples)} samples at {audio.sample_rate}", file=sys.stderr)
        del m

    (out / "reference.json").write_text(json.dumps({"cases": cases}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
