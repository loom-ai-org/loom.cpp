"""Produces the hand-rolled-reference artifacts test_e2e_whisper_encoder_reference.cpp compares against:
a real forward pass of Whisper's actual AudioEncoder (the installed `openai-whisper` package's real
model.py classes, loaded from the real checkpoint) -- NOT hand-derived. Mirrors every other
reference_forward_*.py's role in this project.

Deterministic end to end (no sampling anywhere in the encoder), so this is a plain, exact numerical
check -- no fixed-noise-injection machinery needed (unlike VITS's SDP/flow, which needed torch.randn
monkeypatching).

Usage: python reference_forward_whisper_encoder.py <model.pt> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.audio import log_mel_spectrogram, N_SAMPLES

from whisper_common import mel_hparams, pad_reflect


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(ckpt_path)
    model.eval()

    rng = np.random.RandomState(0)
    audio = (rng.randn(N_SAMPLES).astype(np.float32) * 0.1)  # already exactly 30s, no pad/trim needed

    with torch.no_grad():
        mel = log_mel_spectrogram(torch.from_numpy(audio), n_mels=model.dims.n_mels)  # (n_mels, n_frames)
        xa = model.encoder(mel.unsqueeze(0))  # (1, n_audio_ctx, n_audio_state)

    hp = mel_hparams(model.dims.n_mels)
    waveform_padded = pad_reflect(audio, hp["reflect_pad"])  # what the loom-engine graph's own input needs

    np.save(out_dir / "ref_waveform_padded.npy", waveform_padded)
    np.save(out_dir / "ref_mel.npy", mel.numpy())
    # engine's own convention is channel-first ne=[n_state, n_ctx] (C=ne[0], n_state fastest); xa[0]'s
    # NATIVE (untransposed) PyTorch shape is (n_ctx, n_state) -- row-major, n_state fastest -- which is
    # ALREADY byte-identical to ggml's ne=[n_state,n_ctx] once the batch dim is dropped (numpy (rows,
    # cols) -> ggml ne=[cols,rows], same rule VITS's z_p/z/wav reference dumps rely on -- do NOT
    # transpose here, that was a real, confirmed bug the first time this rule came up in VITS).
    np.save(out_dir / "ref_xa.npy", xa[0].numpy())  # (n_ctx, n_state)
    print(f"mel {mel.shape}, xa {xa.shape}, waveform_padded {waveform_padded.shape}")


if __name__ == "__main__":
    main()
