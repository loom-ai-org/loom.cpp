---
type: retro
date: 2026-10-10
domain: exporter
tags: [exporter, reference, transformers, attention-mask, asr, nemotron, oracles]
---

# Retro-079: The Reference's Default Path Was the Broken One

## The Issue

Nemotron 3.5 ASR (2026-10-10) is NVIDIA's multilingual streaming recogniser. On `samples/jfk.wav` its
`transformers` reference transcribed:

> Your American do you for your country?

That is not a checkpoint NVIDIA reports at 7.9% English WER. Every exporter here loads its reference
with `attn_implementation="eager"`, because eager is what traces to plain ops, and the reference
script did the same. With `"sdpa"` the same checkpoint, audio and processor produce JFK's sentence
word for word. Eager was wrong at every lookahead (0, 3, 6 and 13). At 13 it returned nothing at all.

Building on the eager reference would have produced an export graded as **exact against a broken
oracle**: tokens identical, transcript garbage, and every check green.

## Root Cause

A transformers 5.14 bug: an **additive** mask fed to code written for a **boolean** one. The
encoder builds its chunked-limited attention mask through `create_bidirectional_mask`. That function
returns a boolean mask under sdpa, and an additive float mask (`0` = attend, `-inf` = masked) under
eager. The attention layer then applies it the boolean way:

    matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))

`logical_not(0.0)` is `True`, so under eager this blanks exactly the positions that should be visible
and keeps the masked ones. Nothing fails, and the model emits a plausible short sentence.

## What Changed

* **The reference uses sdpa** (`attn_implementation="sdpa"`), and every Nemotron oracle was produced
  that way.
* **The export does not inherit the bug from either path.** `nemotron_asr_export._NemotronEncoderWrapper`
  builds the chunked-limited mask itself, as a boolean (`_chunked_limited_mask`, checked element for
  element against `transformers`' own predicate), and calls the layers directly with it. The layers
  then compute the sdpa semantics whatever implementation is traced.
* **The gate test asks a question a broken mask cannot answer.** `test_e2e_nemotron_asr_mil_export`
  checks transformers' 48 tokens and also that a *wrong* language prompt gives an empty transcript,
  as the checkpoint does.

## Two Smaller Ones, Found the Same Day

* **The engine's `RANGE_1D` returns −1 elements for a negative step.** `op_range_1d` forces
  `end > start` to keep `ggml_arange` from asserting, so `arange(T - 1, -T, -1)` (the reference's
  relative positions) became `arange(start, start + 1, -1)`. That surfaced only at a later `RESHAPE`.
  The export spells the same values as `-arange(1 - T, T)`, which runs on released wheels. The engine
  fix is on the hub.
* **A multi-axis constant pad was refused.** The causal subsampling `Conv2d` pads frequency and then
  time in two `F.pad` calls, and coremltools merges them into one pad on two axes.
  `passes.transpose_pad_to_last_axis` now splits such a pad into one-axis pads. For a constant pad
  that is exact. Any other mode on several axes is still refused.

## Takeaway

**Check a reference's output for plausibility before grading anything against it.** An oracle exists
to be trusted, and a plausible-but-wrong oracle converts every downstream check into a check of
agreement with a defect. The prompt here was the model card: a published WER against an absurd
transcript is a contradiction, and contradictions get resolved before code is written.

The narrower lesson is about which implementation counts as the reference. **When a reference ships
more than one implementation of the same layer, they are two references, and they can disagree.**
"Use eager because it traces" picks an implementation for the trace's convenience. Comparing eager
against sdpa on one real input costs a minute and would have shown the bug first.
