---
type: retro
date: 2026-09-24
domain: exporter
tags: [exporter, weight-norm, verification, oracle, family-9, voxcpm2]
---

# Retro-056: A Weight-Norm Fold Checked After the Reference Ran Checks Nothing

## The Issue

VoxCPM2's AudioVAE decoder is weight-normed throughout. The export's `VAEDecodePhase` reads each
convolution's `weight` directly (`F.conv1d(x, conv.weight, ...)`), after a `fold_weight_norm` that
removed torch's weight-norm **parametrizations**. The wrapper check compared it with
`audio_vae.decode` on random latents: **max |d| 0.000e+00**.

An exact zero on a fold is suspicious. It usually means both sides read the same tensor. They did.

## Root Cause

This AudioVAE uses the **old** weight norm, `torch.nn.utils.weight_norm`, not the parametrization.
The old one registers `weight_g` and `weight_v` as parameters and keeps `weight` as a plain tensor
attribute, which a **forward pre-hook** recomputes on every call. The fold matched only
parametrizations, so it removed nothing. `conv.weight` was whatever the hook last wrote:

* in the check, the reference's `decode` ran first, so the hook had refreshed every `weight` from
  the loaded `g` and `v`, and the wrapper read correct values;
* in the export, nothing runs the reference's forward before the trace. `weight` still holds the
  value computed when `weight_norm` was applied, **from the initialisation, before the checkpoint
  was loaded**.

The export would have traced a decoder of random convolutions. The check could not see that,
because running the reference is what repaired the state it was checking.

## The Fix

`fold_weight_norm` handles both spellings: it calls `remove_weight_norm` on any module carrying a
`WeightNorm` hook, which computes `weight` from `g` and `v` once and drops the hook. The check now
folds a **freshly loaded** AudioVAE that has never run forward and compares it with the reference.
It is 0.0 again, now for the right reason. `tests/ci/test_voxcpm2_export.py` pins it on a tiny random
model, and fails when the old-style branch is disabled (sabotaged and confirmed). The broken fold
never reached an export: the suspicion came before the first one ran.

`copy.deepcopy` of the reference's VAE was the first attempt at a fresh copy. It fails outright on
old-style weight norm, because `weight` is a non-leaf tensor, so the fresh module is built from
`audiovae.pth` instead.

## Takeaways

* **Check a module rewrite on a module that has never run forward.** A hook that repairs state on
  every call makes any check run after the reference's own forward unable to fail.
* **An exact zero where rounding should show is a sign of shared state, not of precision.**
* **Weight norm has two spellings in torch, and they fail differently.** A parametrization recomputes
  `weight` on every access; the old hook recomputes it only on forward.
