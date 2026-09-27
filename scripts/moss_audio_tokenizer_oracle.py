"""MOSS-Audio-Tokenizer-v2's GGUF against the reference's own decode, on real codes (ADR-049/050).

    ~/.venvs/piper/bin/python scripts/moss_audio_tokenizer_oracle.py \
        --checkpoint /home/flavio/Dev/models/moss-audio-tokenizer-v2 \
        --gguf moss-audio-tokenizer-v2.gguf --wav speech.wav [--seconds 30]

Encodes the clip with the reference's own encoder (48 kHz stereo; a mono clip is duplicated into a
slightly delayed right channel so the channels differ), then decodes the codes three ways: the
reference whole-sequence, the reference's `chunk_duration=8` stream (what MOSS-TTS calls), and the
engine. It checks all 32 codebooks and the 12-codebook prefix MOSS-TTS emits (the other 20 columns
filled with the declared absent id), each with a sabotage arm that has to be far away.

Recorded 2026-09-25 on 30 s of speech: 32 codebooks, relative RMS 1.2e-06 (max 2.6e-06), and the
same against the stream; the 12-codebook prefix is 1.3e-06. The sabotage arms are 1.43 (one-frame
shift) and 0.36 (all 32 codebooks for the prefix).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "loom-exporter"))
sys.path.insert(0, str(HERE.parent.parent / "loom-py"))


def rel(a, b):
    return float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    import loom
    from loom_exporter import moss_audio_tokenizer_export as moss
    import transformers

    ref = transformers.AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True,
                                                 dtype=torch.float32).eval()
    ref.set_attention_implementation("sdpa")
    ref.set_compute_dtype("fp32")                 # the checkpoint's bf16 would autocast on CPU

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.T.copy())
    if sr != 48000:
        wav = torchaudio.functional.resample(wav, sr, 48000)
    if wav.shape[0] == 1:
        wav = torch.cat([wav, torch.nn.functional.pad(wav, (96, 0))[:, :wav.shape[1]] * 0.8])
    wav = wav[:, :int(args.seconds * 48000)]
    with torch.no_grad():
        enc = ref.encode(wav.unsqueeze(0), return_dict=True)
        codes = enc.audio_codes[:, :, :int(enc.audio_codes_lengths[0])]            # [32, 1, n]
        whole = ref.decode(codes, return_dict=True).audio[0].numpy().T.reshape(-1)
        stream = ref.decode(codes, return_dict=True, chunk_duration=8).audio[0].numpy().T.reshape(-1)
    n = codes.shape[-1]
    m = loom.Model.from_file(args.gguf)
    absent = m.hparam("codec.absent_code", "u32")
    got = np.asarray(m.infer(codes=[float(c) for c in codes[:, 0].T.reshape(-1)]), np.float32)
    print(f"32 codebooks, {n} frames: max|d|={np.abs(got - whole).max():.3e} "
          f"rel_rms={rel(got, whole):.2e}; vs the 8 s stream {rel(got, stream):.2e}; "
          f"sabotage (one-frame shift) {rel(got, np.roll(whole, 7680)):.2e}")

    k = min(n, 101)
    with torch.no_grad():
        want12 = ref.decode(codes[:12, :, :k], return_dict=True).audio[0].numpy().T.reshape(-1)
    rows = np.full((k, 32), absent)
    rows[:, :12] = codes[:12, 0, :k].T.numpy()
    got12 = np.asarray(m.infer(codes=[float(c) for c in rows.reshape(-1)]), np.float32)
    all32 = np.asarray(m.infer(codes=[float(c) for c in codes[:, 0, :k].T.reshape(-1)]), np.float32)
    print(f"12-codebook prefix, {k} frames: max|d|={np.abs(got12 - want12).max():.3e} "
          f"rel_rms={rel(got12, want12):.2e}; sabotage (all 32) {rel(all32, want12):.2e}")


if __name__ == "__main__":
    main()
