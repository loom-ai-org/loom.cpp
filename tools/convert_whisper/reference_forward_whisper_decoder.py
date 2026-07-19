"""Produces the hand-rolled-reference artifacts test_e2e_whisper_decoder_reference.cpp compares against:
a real, ONE-SHOT teacher-forced forward pass of Whisper's actual TextDecoder (`model.decoder(tokens, xa,
kv_cache=None)` -- no incremental caching in the reference either, n_past=0/n_tokens=T covers the whole
causal triangle in one call, exactly matching what the engine's own persistent-KvCache self-attention
computes for a first (n_past=0) call). Deterministic end to end -- no sampling anywhere in the decoder
itself (only the ARGMAX-based greedy sampling loop, which lives in loom::WhisperDriver, is stochastic-
adjacent, and that's not exercised by this reference -- it's a plain forward-pass logits check).

Uses the real encoder's own output (xa) as the cross-attention source, computed by re-running
reference_forward_whisper_encoder.py's own logic here rather than depending on its saved .npy (keeps
this script self-contained and independent of run order).

Usage: python reference_forward_whisper_decoder.py <model.pt> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.audio import log_mel_spectrogram, N_SAMPLES


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(ckpt_path)
    model.eval()

    rng = np.random.RandomState(0)
    audio = (rng.randn(N_SAMPLES).astype(np.float32) * 0.1)

    # Arbitrary but VALID (< n_vocab) token ids -- teacher-forcing a fixed causal-mask/cross-attention
    # correctness check doesn't need semantically meaningful tokens, same "any real checkpoint tensor
    # exercises the real math" spirit as every other reference script's arbitrary-but-valid test inputs.
    tokens = [50257, 50362, 1770, 13, 2264, 346, 353, 318]
    assert max(tokens) < model.dims.n_vocab

    with torch.no_grad():
        mel = log_mel_spectrogram(torch.from_numpy(audio), n_mels=model.dims.n_mels)
        xa = model.encoder(mel.unsqueeze(0))
        logits = model.decoder(torch.tensor([tokens]), xa, kv_cache=None)  # (1, T, n_vocab)

    np.save(out_dir / "ref_dec_xa.npy", xa[0].numpy())  # (n_audio_ctx, n_state), see encoder ref's own note
    np.save(out_dir / "ref_dec_tokens.npy", np.array(tokens, dtype=np.int32))
    np.save(out_dir / "ref_dec_logits.npy", logits[0].numpy())  # (T, n_vocab)
    print(f"tokens={tokens}, xa={xa.shape}, logits={logits.shape}")


if __name__ == "__main__":
    main()
