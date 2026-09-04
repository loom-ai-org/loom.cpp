#!/usr/bin/env python3
"""The BACK HALF of the ASR oracle: transcribe a synthesised WAV and say whether it said the words.

    ~/.venvs/piper/bin/python scripts/asr_oracle.py out.wav --expect "hey can you shut down the computer"

`scripts/tts_synth.cpp` is the front half. Together they are the only trustworthy verdict on a TTS or
codec-LM change, and the reason is [Retro-006](../docs/retros/retro-006-kokoro-shipped-noise.md):
Kokoro once matched PyTorch at cosine 0.996 and shipped unintelligible audio. A correlation, an exact
integer match against `transformers`, and a plausible peak are all compatible with noise. Words are
not.

**A standard ASR model from `transformers`, not loom's own whisper**, deliberately: the front half is
the thing under test, and grading it with the same engine makes a shared bug invisible. Resampling is
a linear interpolation for the same reason — an oracle should have as little of the system under test
in it as possible. `~/Dev/models/whisper-small` is the default because it is the smallest checkpoint
that reliably transcribes 22–44 kHz synthetic speech; `whisper-tiny` invents words on quiet clips.

It prints the transcript and, with `--expect`, the fraction of expected words that appear. It does not
decide a threshold: what counts as passing depends on the sentence and the family, and a script that
returned a verdict would be inventing one. `test_model_cards.py` is where a per-model threshold lives.
"""
import argparse
import re
import sys
import wave

import numpy as np


def read_wav(path: str):
    with wave.open(path) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{path} is not 16-bit mono; tts_synth writes that and this reads it")
        rate, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, rate


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    """Linear resample. An oracle, not a codec -- good enough to recognise words by, and the same
    approach loom-py's model-card gate takes for the same reason."""
    if rate == 16000:
        return pcm
    m = int(round(len(pcm) * 16000 / rate))
    return np.interp(np.linspace(0, len(pcm) - 1, m), np.arange(len(pcm)), pcm).astype(np.float32)


def words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--model", default="/home/flavio/Dev/models/whisper-small")
    ap.add_argument("--expect", default=None, help="the sentence the synthesiser was asked for")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    pcm, rate = read_wav(args.wav)
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    rms = float(np.sqrt((pcm ** 2).mean())) if len(pcm) else 0.0
    # Printed before the transcript because they are the two numbers that explain an empty one: real
    # speech lands near ±0.3, and a peak at 1.0 is clipping rather than loudness.
    print(f"{args.wav}: {len(pcm)} samples at {rate} Hz = {len(pcm)/rate:.2f} s, "
          f"peak {peak:.4f}, rms {rms:.5f}")
    if not len(pcm):
        raise SystemExit("no audio")

    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    feats = proc(to_16k(pcm, rate), sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats, language=args.language, task="transcribe", max_new_tokens=128)
    transcript = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    print(f"TRANSCRIPT: {transcript}")

    if args.expect:
        want, got = words(args.expect), set(words(transcript))
        hit = [w for w in want if w in got]
        print(f"MATCHED {len(hit)}/{len(want)} expected words: {hit}")
        print(f"MISSED: {[w for w in want if w not in got]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
