#!/usr/bin/env python3
"""Independent numpy reference for StyleTTS2's `KarrasSchedule` + `ADPM2Sampler` (real source:
Modules/diffusion/sampler.py, read in full off the real `yl4579/StyleTTS2` repo before writing this --
NOT re-derived from memory). Uses a TOY denoiser (a fixed per-element affine map, not the real
Transformer1d network) so this fixture isolates the discretization math (schedule + midpoint-update +
ancestral-noise-injection) from the real network weights, matching this project's usual "verify the
mechanism independently first" discipline (e.g. test_hifigan_generator's own synthetic-weights
precedent). The ancestral noise draws are pre-generated here (fixed seed) and written out so
test_style_diffusion_sampler.cpp can replay the EXACT SAME sequence instead of relying on its own RNG --
decouples "is the discretization math right" from "do two independently-implemented RNGs agree" (they
never will, by design, so an exact bit-for-bit comparison of the whole pipeline would be meaningless
otherwise).

KarrasSchedule (Karras et al. 2022 eq. 5):
    sigmas[i] = (sigma_max**(1/rho) + i/(num_steps-1) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    for i in [0, num_steps), then a final 0.0 appended (len = num_steps+1).

ADPM2Sampler (rho=1.0, the real class default) per real source:
    sigma_up   = sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
    sigma_down = sqrt(sigma_next**2 - sigma_up**2)
    sigma_mid  = (sigma + sigma_down) / 2          # ((sigma**(1/r)+sigma_down**(1/r))/2)**r at r=1
    d      = (x - fn(x, sigma)) / sigma
    x_mid  = x + d * (sigma_mid - sigma)
    d_mid  = (x_mid - fn(x_mid, sigma_mid)) / sigma_mid
    x      = x + d_mid * (sigma_down - sigma)
    x_next = x + noise * sigma_up                  # noise ~ N(0,1), fresh draw per step
`forward(noise, fn, sigmas, num_steps)`: x = sigmas[0]*noise, then num_steps-1 calls to `step` over
sigmas[0..num_steps-1] (the trailing appended 0 from KarrasSchedule is NEVER actually reached -- real
source, not a bug in this reference).

Usage: python3 reference_style_diffusion_sampler.py <out_dir>
Requires: pip install numpy
"""
import sys
from pathlib import Path

import numpy as np


def karras_schedule(num_steps, sigma_min, sigma_max, rho):
    rho_inv = 1.0 / rho
    steps = np.arange(num_steps, dtype=np.float64)
    denom = max(num_steps - 1, 1)
    sigmas = (sigma_max ** rho_inv + (steps / denom) * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
    return np.concatenate([sigmas, [0.0]])


def toy_denoise_fn(x, sigma, a, b):
    """A fixed per-element affine map standing in for the real KDiffusion-preconditioned Transformer1d
    network -- deliberately sigma-independent (the SAMPLER's own math already handles all sigma
    dependence; the "network" itself just needs to be SOME deterministic function of x for this fixture
    to exercise the loop correctly)."""
    return a * x + b


def adpm2_step(x, fn, sigma, sigma_next, noise):
    sigma_up = np.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = np.sqrt(sigma_next ** 2 - sigma_up ** 2)
    sigma_mid = (sigma + sigma_down) / 2.0
    d = (x - fn(x, sigma)) / sigma
    x_mid = x + d * (sigma_mid - sigma)
    d_mid = (x_mid - fn(x_mid, sigma_mid)) / sigma_mid
    x_next = x + d_mid * (sigma_down - sigma)
    x_next = x_next + noise * sigma_up
    return x_next


def adpm2_sample(noise0, fn, sigmas, num_steps, step_noises):
    x = sigmas[0] * noise0
    trace = [x.copy()]
    for i in range(num_steps - 1):
        x = adpm2_step(x, fn, sigmas[i], sigmas[i + 1], step_noises[i])
        trace.append(x.copy())
    return x, trace


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("style_diffusion_sampler_ref")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(1234)
    dim = 8
    num_steps = 5
    sigma_min, sigma_max, rho = 1e-4, 3.0, 9.0

    sigmas = karras_schedule(num_steps, sigma_min, sigma_max, rho).astype(np.float32)

    a = rng.normal(loc=0.9, scale=0.05, size=dim).astype(np.float32)
    b = rng.normal(scale=0.1, size=dim).astype(np.float32)

    noise0 = rng.normal(size=dim).astype(np.float32)
    step_noises = rng.normal(size=(num_steps - 1, dim)).astype(np.float32)

    fn = lambda x, sigma: toy_denoise_fn(x, sigma, a, b)
    x_final, trace = adpm2_sample(noise0, fn, sigmas, num_steps, step_noises)

    sigmas.tofile(out_dir / "sigmas.bin")
    a.tofile(out_dir / "a.bin")
    b.tofile(out_dir / "b.bin")
    noise0.tofile(out_dir / "noise0.bin")
    step_noises.tofile(out_dir / "step_noises.bin")
    x_final.astype(np.float32).tofile(out_dir / "expected_x_final.bin")
    np.stack(trace).astype(np.float32).tofile(out_dir / "expected_trace.bin")

    print(f"dim={dim}, num_steps={num_steps}, sigmas={sigmas}, x_final={x_final}")


if __name__ == "__main__":
    main()
