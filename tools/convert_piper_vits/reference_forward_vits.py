"""Produces the hand-rolled-reference artifacts test_e2e_vits_stats_reference.cpp,
test_e2e_vits_flow_vocoder_reference.cpp, and test_e2e_vits_logw_reference.cpp compare against: real
forward passes of piper's own TextEncoder/StochasticDurationPredictor/ResidualCouplingBlock/Generator,
loaded directly from the real checkpoint (bypassing piper's own ONNX export/runtime entirely), NOT
hand-derived. Mirrors tools/convert_nemo/reference_forward_parakeet_tdt.py's role for that model.

Real text -> phoneme -> token-id sequence uses piper_phonemize (the same in-process espeak-ng binding
piper's own runtime calls -- see BACKLOG.md for the exact phonemization/BOS-blank-interleave/EOS
convention this reproduces byte-for-byte, confirmed against voice.py's own `phonemes_to_ids`).

StochasticDurationPredictor and the coupling flow/vocoder are both fed FIXED, externally-injected noise
(a `torch.randn` monkeypatch for SDP's own internal `z = torch.randn(...)` call; a hand-constructed z_p
for the flow/vocoder) rather than letting them sample -- both are genuinely stochastic in the real model,
so an exact match against a HOST-side driver's own (different) RNG stream is only possible by pinning the
noise itself, not by seeding two different RNG algorithms identically.

Usage: python reference_forward_vits.py <checkpoint.ckpt> <config.onnx.json> <out_dir>
"""
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, "/home/flavio/Dev/piper/src/python")
from piper_train.vits.models import TextEncoder, StochasticDurationPredictor, ResidualCouplingBlock, Generator
from piper_phonemize import phonemize_espeak

from vits_common import load_piper_checkpoint


def phonemes_to_ids(phonemes, id_map):
    """Real algorithm, piper's own `voice.py::Voice.phonemes_to_ids` (not hand-derived) --
    [BOS, p1, blank, p2, blank, ..., pn, blank, EOS], no blank right after BOS.
    """
    ids = list(id_map["^"])
    for p in phonemes:
        if p not in id_map:
            print(f"missing phoneme from id map: {p!r}", file=sys.stderr)
            continue
        ids.extend(id_map[p])
        ids.extend(id_map["_"])
    ids.extend(id_map["$"])
    return ids


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <model.ckpt> <config.onnx.json> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, cfg_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = json.load(f)
    id_map = cfg["phoneme_id_map"]
    voice = cfg["espeak"]["voice"]

    text = "Hello world, this is a test."
    sentences = phonemize_espeak(text, voice)
    token_ids = phonemes_to_ids(sentences[0], id_map)
    T = len(token_ids)
    print(f"text={text!r} -> {T} tokens")
    with open(out_dir / "ref_token_ids.json", "w") as f:
        json.dump(token_ids, f)

    full_sd = load_piper_checkpoint(ckpt_path)
    sd = {k[len("model_g."):]: v for k, v in full_sd.items() if k.startswith("model_g.")}

    # --- TextEncoder (deterministic) -> m_p/logs_p ---
    enc_p = TextEncoder(256, 192, 192, 768, 2, 6, 3, p_dropout=0.1)
    missing, unexpected = enc_p.load_state_dict({k[len("enc_p."):]: v for k, v in sd.items() if k.startswith("enc_p.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    enc_p.eval()

    x_ids = torch.tensor([token_ids], dtype=torch.long)
    x_lengths = torch.tensor([T], dtype=torch.long)
    with torch.no_grad():
        x_cond, m_p, logs_p, x_mask = enc_p(x_ids, x_lengths)
    np.save(out_dir / "ref_m_p.npy", m_p.numpy())       # (1, 192, T)
    np.save(out_dir / "ref_logs_p.npy", logs_p.numpy()) # (1, 192, T)
    print("m_p[0,:5,0]:", m_p[0, :5, 0].tolist())

    # --- StochasticDurationPredictor (reverse), fixed injected noise -> logw ---
    dp = StochasticDurationPredictor(192, 192, 3, 0.5, 4, gin_channels=0)
    missing, unexpected = dp.load_state_dict({k[len("dp."):]: v for k, v in sd.items() if k.startswith("dp.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    dp.eval()

    rng = np.random.RandomState(7)
    z_noise_np = (rng.randn(1, 2, T).astype(np.float32) * 0.8)  # noise_scale_w=0.8 baked in directly
    z_noise_t = torch.from_numpy(z_noise_np)
    real_randn = torch.randn

    def fake_randn(*args, **kwargs):
        if tuple(args) == (1, 2, T):
            return z_noise_t.clone()
        return real_randn(*args, **kwargs)

    with torch.no_grad(), mock.patch("torch.randn", side_effect=fake_randn):
        logw = dp(x_cond, x_mask, g=None, reverse=True, noise_scale=1.0)  # already scaled above
    # ggml's [T,2] convention (T=ne[0], fastest) is byte-identical to numpy's native (2,T) row-major
    # layout -- save WITHOUT transposing (same lesson recorded in BACKLOG.md for z_p below).
    np.save(out_dir / "ref_sdp_z_noise.npy", z_noise_np[0]) # (2, T)
    np.save(out_dir / "ref_sdp_logw.npy", logw[0, 0].numpy()) # (T,)
    print("logw[0,0,:10]:", logw[0, 0, :10].tolist())

    # --- Coupling flow (reverse) + HiFi-GAN vocoder, fixed hand-picked z_p -> waveform ---
    flow = ResidualCouplingBlock(192, 192, 5, 1, 4, n_flows=4, gin_channels=0)
    missing, unexpected = flow.load_state_dict({k[len("flow."):]: v for k, v in sd.items() if k.startswith("flow.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    flow.eval()
    dec = Generator(192, "2", (3, 5, 7), ((1, 2), (2, 6), (3, 12)), (8, 8, 4), 256, (16, 16, 8), gin_channels=0)
    missing, unexpected = dec.load_state_dict({k[len("dec."):]: v for k, v in sd.items() if k.startswith("dec.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    dec.eval()

    torch.manual_seed(123)
    Tp = 8
    z_p = torch.randn(1, 192, Tp) * 0.5
    with torch.no_grad():
        y_mask = torch.ones(1, 1, Tp)
        z = flow(z_p, y_mask, g=None, reverse=True)
        wav = dec(z, g=None)
    # Same "don't transpose" rule: ggml's [T,C] flow/vocoder convention is byte-identical to PyTorch's
    # native (1, C, T) tensor's own (C, T) memory layout once the batch dim is dropped.
    np.save(out_dir / "ref_z_p.npy", z_p[0].numpy())  # (192, Tp)
    np.save(out_dir / "ref_wav.npy", wav[0, 0].numpy())  # (Tp*256,)
    print("wav rms:", wav.pow(2).mean().sqrt().item(), "max abs:", wav.abs().max().item())


if __name__ == "__main__":
    main()
