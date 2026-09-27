---
type: retro
date: 2026-09-24
domain: exporter
tags: [exporter, kv-cache, multi-phase, oracle, family-9, voxcpm2]
---

# Retro-057: Two Cached Stacks Wrote One Cache's First Layers, and the Speech Was Still Right

## The Issue

VoxCPM2 has two KV-cached transformers, a 28-layer base LM and an 8-layer residual LM, exported as two
phases. The first engine build synthesised "The quick brown fox jumps over the lazy dog." and the Whisper
ASR oracle transcribed it **exactly**. The teacher-forced gate then disagreed with the reference from the
second patch on, by O(1), on every patch.

Bisecting by phase through an ad-hoc probe (`LoomLuaBridge::load_script` beside the driver):

| check | max \|Δ\| |
|---|---|
| prefill, both LMs and the guided DiT (patch 0) | 4.5e-06 |
| `feat_encode` at T=3 and T=1, `dit_step` with a random `cond` | ≤ 1.5e-05 |
| one `base_lm` decode step, residual LM prefill **not** run first | 0 (vs the explicit-input run) |
| the same step with the residual LM's prefill run first | **0.82** |

## Root Cause

Every cached module that is not a private stream gets the SAME `KvCache` (`register_topologies`), and
`_kv_cache_geometry` sized it for all 36 blocks. But `fuse_loom_attention` numbers blocks densely per
PROGRAM, so each phase started at layer 0. The residual LM's eight blocks wrote cache slots 0-7, which
were the base LM's. Nothing failed:

* the prefill matched, because a prefill attends over the K/V it computes in the same call;
* the declared geometry (36 slots) was right, so nothing was out of range;
* the audio was intelligible, because a transformer whose first eight layers read another stack's
  history still produces plausible hidden states. An FSQ bottleneck and a DiT sit between them and
  the waveform.

Every earlier export has ONE cached phase (plus private alias streams), so the collision could not
happen before.

Two false leads cost time and are worth knowing. The FSQ bottleneck was suspected first, because the
prefill multiplies it by `audio_mask = 0` and so never exercised it. Then the `{from=}` retained-output
edge, because passing the same input as an explicit array "fixed" it. That probe had also dropped the
residual LM's prefill, which was the real variable.

## The Fix

`decomposition.offset_cached_attention_layers`: each converted phase's cached ATTENTION nodes are shifted
past the blocks every earlier phase fused, so the slots are disjoint inside the one cache the geometry
already declared. A no-op for every export with a single cached phase, whose bytes do not move.
`tests/ci/test_cached_phase_layers.py` pins it.

## Takeaways

* **An ASR oracle passes a model with a corrupted cache.** Intelligible speech says the pipeline is
  plausible, not that it is right. Only the tensor comparison found this.
* **A declared total and per-part indices must be reconciled where the parts are joined.** The cache
  size was the sum over phases; the indices were per phase. Each was right on its own.
* **When bisecting, change one variable per probe.** The explicit-input probe also removed the
  residual prefill, and it pointed at the wrong mechanism until a probe varied that alone.
