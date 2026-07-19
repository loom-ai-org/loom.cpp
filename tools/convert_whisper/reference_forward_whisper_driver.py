"""Produces the reference greedy-decoded token sequence test_e2e_whisper_driver.cpp compares
loom::WhisperDriver::transcribe's own output against: a real, PLAIN greedy argmax decode loop built
directly from Whisper's own encoder/decoder forward passes (NOT `model.decode()`, which applies extra
logic -- suppress_tokens, timestamp constraints, temperature fallback, etc. -- that loom::WhisperDriver
deliberately does not implement; this reference mirrors the driver's own simpler loop exactly, one full
`model.decoder(...)` call per step, no incremental KV-cache optimization needed for a short reference
run). Both are deterministic (argmax, no sampling), so this is a plain exact-token-sequence comparison.

Usage: python reference_forward_whisper_driver.py <model.pt> <out_dir> [n_new_tokens]
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
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir> [n_new_tokens]", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    n_new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(ckpt_path)
    model.eval()
    tok = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)
    prompt = list(tok.sot_sequence) + [tok.no_timestamps]

    rng = np.random.RandomState(0)
    audio = (rng.randn(N_SAMPLES).astype(np.float32) * 0.1)
    hp = mel_hparams(model.dims.n_mels)
    waveform_padded = pad_reflect(audio, hp["reflect_pad"])

    with torch.no_grad():
        mel = log_mel_spectrogram(torch.from_numpy(audio), n_mels=model.dims.n_mels)
        xa = model.encoder(mel.unsqueeze(0))

        tokens = list(prompt)
        generated = []
        for _ in range(n_new_tokens):
            logits = model.decoder(torch.tensor([tokens]), xa, kv_cache=None)
            next_tok = int(logits[0, -1].argmax().item())
            generated.append(next_tok)
            tokens.append(next_tok)
            if next_tok == tok.eot:
                break

    np.save(out_dir / "ref_driver_waveform_padded.npy", waveform_padded)
    np.save(out_dir / "ref_driver_prompt.npy", np.array(prompt, dtype=np.int32))
    np.save(out_dir / "ref_driver_generated.npy", np.array(generated, dtype=np.int32))
    print(f"prompt={prompt}, generated={generated}, eot={tok.eot}")


if __name__ == "__main__":
    main()
