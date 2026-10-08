---
type: retro
date: 2026-10-08
domain: backends
tags: [backends, metal, norm, ggml, upstream, kokoro, oracle]
---

# Retro-075: Upstream Had Already Fixed the Norm

## The Issue

`kokoro-82m.gguf` on Metal (M1 Pro, `MTL0`) produced audio of the right length that peaked at 3.02
against the CPU's 0.27. Whisper heard "(gasps)" where the CPU output reads "Halo World". It passed on
Vulkan and CUDA, and the rc14 engine failed identically. The 11x peak pointed at the obvious
large-magnitude suspects: the iSTFT, the sine generator, a convolution.

## Root Cause

The first divergence was small and early: a `NORM` over rows of 66 elements (an AdaIN instance norm
in the decoder) whose rows no longer summed to zero (abs-sum off by 0.3%). ggml-metal sized the norm's
threadgroup as `min(nth, ne00_t)` = 66 threads: two full simdgroups and one with 2 lanes. The kernel
finishes its row sum with a cross-simdgroup step in which each lane of a simdgroup reads one
simdgroup's partial. The 2-lane simdgroup read only partials 0 and 1, so outputs 64 and 65 of every
row got a wrong mean and variance. The vocoder's AdaIN chain turned that into the 11x peak.

**Upstream had fixed it two months earlier**: llama.cpp `a194a75b7e` (#26708, 2026-08-07) rounds the
thread count up to whole simdgroups. Our ggml pin is v0.19.0, which predates it.

## The Fix

`ggml-0025` is that one line, backported. After it, Kokoro on `MTL0` peaks at 0.2714 (CPU 0.2710) and
Whisper hears "Halo World". It can be deleted once the pin passes `a194a75b7e`
([UPSTREAM.md](../../cmake/patches/UPSTREAM.md) PR 22).

## Takeaway

**When a pinned dependency's op is wrong, search upstream's history for that op before writing a
fix.** `gh api "repos/ggml-org/llama.cpp/commits?path=<file>"` takes seconds and returned the fix, its
reasoning and its test cases. And **rank divergences by first, not by largest**: the defect was a 0.3%
error in a norm; the 1100% error was its consequence. The per-node CPU-oracle diff of
[Retro-074](retro-074-a-gpus-f32-matmul-was-half-precision.md#the-fix) found it on its first run because
it reports the earliest node over tolerance, not the worst.
