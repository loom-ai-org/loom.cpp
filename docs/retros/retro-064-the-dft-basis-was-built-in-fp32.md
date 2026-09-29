---
type: retro
date: 2026-09-29
domain: exporter
tags: [exporter, coremltools, stft, precision, f64, family-10, qwen3-tts]
---

# Retro-064: The DFT Basis Was Built in fp32

## The Issue

Once the speaker encoder stopped aborting ([Retro-063](retro-063-an-expand-as-was-lowered-as-an-identity.md)),
its x-vector matched the reference at **cosine 0.9999998**, max |Δ| 7.5e-4 on values up to 7.57. That
reads like float noise for a 40-convolution stack. At f64, though, the reference sits **1.3e-6** from
its own f32 run. So loom's gap was about 570x the reference's float spread: a defect.

## Root Cause

A bisect helped here. Truncated copies of the wrapper (mel, block 0, MFA, ASP, FC) were each exported
as a standalone GGUF, run through the engine with a phase probe, and compared with torch at f64. The
gap was already whole at the **log-mel**: 0.40 on a bin at −11.43, next to the clamp floor. Binned by
level, the error was about 1e-5 of each frame's energy, spread over the frame's other bins. That is
tiny in loud bins (3e-6) and dominant in near-silent ones. An f32 DFT computed in numpy by the same
algorithm (a matmul against the basis) lands at 7.8e-4, and torch's FFT at 1.1e-3. So the algorithm
was not the cause; the constant was.

`torch.stft` reaches the graph through coremltools' `lower_complex_dialect_ops`, and its
`_calculate_dft_matrix` builds the basis as `cos(outer(k, n) * 2π / N)` **with every step in fp32**.
`k·n` reaches (N/2)(N−1), which is 523,776 at N=1024. Times 2π, that product's ulp is **0.25 rad**,
and dividing by N afterwards cannot recover it. The folded basis in the GGUF was off by **1.4e-4**,
against the 6e-8 that f32 rounding alone allows.

## The Fix

`torch_patches` patch 4 replaces `_calculate_dft_matrix`. It reduces `k·n` mod N in integers, takes
the cosine at f64, and casts to f32 (a non-constant `n_fft` still goes to the original). The basis is
now within 6e-8. The x-vector is **1.5e-6** from f64, the same as the reference's own f32 run. Greedy
codes through `waveform=` match `transformers` **624/624**, in both x-vector and ICL mode.
`TestDftBasisPrecision` fails without the patch (2.0e-4).

**Every export that traces `torch.stft` carried this.** A scan of the rc11 tree found it in
conformer-ctc-small, parakeet-rnnt, parakeet-tdt and granite-speech (5.3e-5 at N=512), and in
F5-TTS's mel and Qwen3-TTS (1.4e-4 at N=1024). Paraformer and SenseVoice build their basis on the host
and are exact. Those six GGUFs change on re-export; see the hub.

## Takeaway

**Difference a folded constant against numpy before trusting anything built on it.** This is SNAC's
`reciprocal` epsilon (`torch_patches` patch 3) again: a coremltools helper computed a constant
imprecisely, it folded into a weight, and downstream it looked exactly like float noise. Both times,
f64 was what told noise from a defect ([[feedback-run-the-reference-at-f64]]). The bisect was
possible without re-exporting the model, because a truncated wrapper exports in seconds.

## Related

* [Retro-063](retro-063-an-expand-as-was-lowered-as-an-identity.md)
* [Retro-062](retro-062-an-f32-wrapper-check-could-not-tell-a-spelling-from-a-defect.md): the same f64 rule
* [Epic-03](../epics/epic-03-model-coverage.md) family 10, Qwen3-TTS
