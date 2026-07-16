#!/usr/bin/env python3
"""Independent numpy re-implementation of the toy flow-matching vector field's forward-Euler ODE
integration (see toy_ode_common.py), used as the ground truth test_e2e_toy_ode.cpp compares
OdeStepper::integrate()'s C++ output against. Mirrors OdeStepper's loop exactly: same dt, same per-step
t, same scalar-broadcast timestep/conditioning treatment.

Requires: pip install numpy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as enc
import toy_ode_common as common


def vector_field(latent: np.ndarray, t: float, conditioning: np.ndarray, weights: dict) -> np.ndarray:
    """latent: (n_tokens, n_embd) natural layout. t: scalar. conditioning: (n_embd,). Returns
    d(latent)/dt, same shape as latent."""
    cur = latent + t + conditioning  # both broadcasts match the engine's ADD(latent,timestep)/ADD(_,conditioning)
    data = cur.T[np.newaxis, :, :]  # (n_tokens,n_embd) -> (1, n_embd, n_tokens) for conv1d's (N,IC,IL)
    hidden = enc.conv1d(data, weights["vf_conv1.weight"], stride=1, padding=common.PADDING)
    hidden = enc.gelu_erf(hidden)
    out = enc.conv1d(hidden, weights["vf_conv2.weight"], stride=1, padding=common.PADDING)
    return out[0].T  # (1,n_embd,n_tokens) -> (n_tokens, n_embd)


def integrate(n_steps: int, t_start: float, t_end: float) -> np.ndarray:
    weights = common.generate_weights()
    conditioning = common.generate_conditioning()
    latent = common.generate_initial_latent().T.copy()  # gguf-ready (C,T) -> natural (T,C)

    dt = (t_end - t_start) / n_steps
    for step in range(n_steps):
        t = t_start + step * dt
        velocity = vector_field(latent, t, conditioning, weights)
        latent = latent + dt * velocity
    return latent.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_latent = integrate(args.n_steps, args.t_start, args.t_end)
    # final_latent is (n_tokens, n_embd) in numpy's natural C-order (n_embd fastest) -- the OPPOSITE flat
    # byte order from the engine's ne=[n_tokens,n_embd] ggml convention (n_tokens fastest), which is what
    # the C++ side's flat host buffer is in throughout (it never transposes, just round-trips whatever
    # order "initial_latent.data" was baked in). Dumping the transpose's C-order flat bytes reproduces
    # that same n_tokens-fastest order -- see toy_ode_common.py's generate_initial_latent() docstring for
    # the matching reasoning on the input side.
    final_latent.T.tofile(out_dir / "expected_final_latent.bin")


if __name__ == "__main__":
    main()
