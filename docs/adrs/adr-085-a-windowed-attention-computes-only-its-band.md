---
type: adr
status: accepted
date: 2026-10-23
domain: performance
tags: [attention, window, mimi, pocket-tts, armv6, pi-zero, exporter]
---

# ADR-085: A Windowed Attention Computes Only Its Band

Serves [Epic-05](../epics/epic-05-edge-performance.md) (P4.41) and
[Epic-03](../epics/epic-03-model-coverage.md) (family 9's Pocket-TTS). Builds on
[ADR-084](adr-084-pocket-ttss-small-models-ship-their-own-head-and-v4s-text-rules-as-data.md).

## Context

Pocket-TTS's Mimi decoder runs a causal transformer at 200 Hz with a window of 50 positions
(`context`): 4.8 s of audio is 960 positions, and each attends to its last 50. The export builds that
window as a dense `[T, T]` additive mask from the positions (`MimiDecoderPhase.window_mask`), and the
attention ran as plain ops: `Q @ K^T` over all 960 keys, a softmax over all of them, `A @ V` over all of
them. 95% of every row is masked. On a Pi Zero W, profiled per node, that was **6.4 s of the 2m model's
18.3 s synthesis** (4 layers; 18m and 7m have 12).

## Options

1. **ggml's `flash_attn_ext` with a `-inf` F16 mask.** Its CPU kernel skips a key whose mask is `-inf`.
   Rejected: the mask must be F16 and padded, one node walks all 960 keys per row to find the 50, the
   scalar kernel's per-key `expf` and dot are not ARMv6's tuned paths, and GPU backends take F16 K/V.
2. **The band as a MIL rewrite in the exporter** (blocks of 16 positions, keys gathered from shifted
   slices). Rejected: the sequence length is a symbol (`16 * n_codes`), and a slice whose extent is
   derived from a shape is exactly what breaks the export's shape algebra.
3. **Mimi decoded streaming, a step at a time against a 50-position cache**, as the reference does.
   Not now: conv states for every SEANet layer and a ring KV cache, for a model decoded once per chunk.
4. **An attribute on the engine's `ATTENTION`, declared by the export.** Chosen.

## Decision

* **`ATTENTION` takes `window` (int, `kv_cache=false` only)**: a promise that the mask is `-inf` wherever
  `q - window < k <= q` does not hold. The op then computes blocks of queries against only the keys their
  rows can see, `[q0 - window + 1, q0 + nq)`, with the same three ops as the dense form (`mul_mat`,
  `soft_max_ext` with the block's corner of the mask, `mul_mat`) and a pairwise `concat` of the blocks.
  Inside the band the mask is still applied, so it can carry anything the window allows.
* **The block is 32 queries.** Pi Zero W, the 2m synthesis: blocks of 16 / 24 / 32 / 48 / 64 ran
  9.56-9.73 / 9.55-9.62 / 9.61 / 9.67-9.70 / 9.73-9.75 s. `$LOOM_ATTENTION_WINDOW_BLOCK` overrides it and
  `$LOOM_ATTENTION_WINDOW=0` runs the node densely.
* **Each Q block is copied contiguous.** A batched `mul_mat` reaches tinyBLAS only with a contiguous
  `src1`; the strided view ran ggml's per-element dot at 33 MMAC/s, against 280 for `A @ V`, and made the
  first board run 11.2 s instead of 9.6 ([Retro-092](../retros/retro-092-a-view-chose-the-kernel.md)).
* **The export declares the window, per phase**: `ExportPhase.attention_window`. It fuses the phase's
  SDPA blocks into uncached `ATTENTION` nodes carrying it, writes no cache geometry for them, and raises
  if the phase also declares a cache or fused nothing. The wrapper that builds the mask is the only thing
  that knows it is banded; Pocket-TTS's test checks `window_mask` against the declaration.

## Consequences

* **Measured**, Pi Zero W, 4.8 s of audio (60 frames, EOS disarmed), one process per arm, two rounds with
  the order flipped, together with ggml-0050-0052 (Epic-05 P4.41): **2m 18.51-18.57 s -> 9.61-9.63 s
  (1.93x, RTF 3.9 -> 2.0)**, **7m 44.6-45.0 s -> 22.4-22.5 s (1.99x)**. Mimi's attention went 6.4 s ->
  0.58 s in the node profile. The waveforms move 2.8e-6 (2m) and 1.2e-6 (7m) against main's, peaks 0.68
  and 0.62.
* **Numerics.** The softmax rows are the dense form's (an out-of-band column contributes an exact 0);
  `A @ V` regroups its sums. Against an f64 SDPA: 1.1e-7 (`test_primitive_registry`). On ARMv6, whose sums
  run in order, the banded and dense forms are byte-identical; the Pocket gate's teacher-forced error is
  unchanged on all three models (2.5e-5 / 2.3e-5 / 6.3e-5).
* **`LOOM_ATTENTION_WINDOW=0` is not today's dense graph.** The fused node's dense form reads Q as a
  strided view, which on ARMv6 is the slow dot above: 26.9 s against the unfused graph's 18.3. Every
  dense `ATTENTION` has that view; changing it moves every attention model's numerics, so it is a hub
  item, not part of this.
* **An older engine reads a windowed node correctly but slowly**: it ignores `window` and takes the dense
  fused form above, 26.9 s for the 2m model on the Pi Zero against 18.3 for the file exported without the
  window. A Pocket-TTS file exported with this needs the release that carries it; the small models' cards
  already require 1.0.0rc19 for their text rules (ADR-084).
