---
type: retro
date: 2026-10-10
domain: engine
tags: [engine, exporter, get-rows, views, silent-failure, sanotts, tts]
---

# Retro-080: A Gather Read a View as Rows

## The Issue

sanoTTS's piperlite voices (2026-10-10) expand per-token states to frames: the acoustic net's token
stack produces `[channels, tokens]`, and each frame takes the row of the token it belongs to. The export
spells that as `index_select(states.transpose(0, 1), 0, frame_index)`, a GET_ROWS over a PERMUTE.

The first export of `amy-en-1p46m` ran without an error, produced the right number of samples, and
correlated **0.002** with upstream's torch path. The durations were right (508 frames on both sides),
which pointed away from the driver.

## Root Cause

ggml-cpu's `get_rows` copies `ne0` contiguous elements from the start of each selected row and never
reads the table's element stride `nb[0]`. A PERMUTE is a view: its rows are strided. So every gathered
row was assembled from the wrong elements, with no shape mismatch to catch it.

Every gather before this one read a weight table, which is always packed, so the assumption had never
been tested. A stage-by-stage bisect (export the `synth` phase truncated after each stage, probe it,
diff against torch at f64) showed the token stack exact to 4e-7 and the gather wrong from element 1.

## What Changed

* **Exporter:** a GET_ROWS whose table is produced by PERMUTE or TRANSPOSE gets a CONT first
  (`_pack_gathered_views`, beside `_materialize_view_outputs`). This makes the file right on every
  engine that can load it, rc16 included. No published GGUF had a gather on a view (all of
  `hf-models/` scanned), so every other export is unchanged.
* **Engine:** `op_get_rows` packs a non-contiguous table (`ensure_packed`). A packed table costs
  nothing extra, so embedding lookups are unaffected.
* Tests on both sides, each sabotage-checked: a toy conv-then-gather through the real compiler, and a
  transposed-table GET_ROWS primitive test.

## Takeaway

**A ggml op that takes a tensor does not mean it takes any layout of that tensor.** When a lowering
reaches an op with a new KIND of operand (an activation where it only ever saw weights), check that
op's stride handling before trusting a passing shape check. Fix it at the producer, so old engines
also get correct files, and at the op, so the next model cannot hit it.

Related: [ADR-031](../adrs/adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md) (retained references), `ensure_packed`'s own note on why
`ggml_is_contiguous` alone is not a sufficient guard.
