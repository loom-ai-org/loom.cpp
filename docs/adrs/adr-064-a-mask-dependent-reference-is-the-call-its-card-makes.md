---
type: adr
status: accepted
date: 2026-10-02
tags: [asr, moonshine, encoder, attention, reference, decoding, family-2]
supersedes: []
---

# ADR-064: A Mask-Dependent Reference Is the Call Its Card Makes

## Context

Moonshine Streaming (`moonshine-ai/moonshine-streaming-tiny`/`-small`, [Epic-03](../epics/epic-03-model-coverage.md))
has an encoder whose layers attend through a sliding window: `(16, 4)` -- sixteen frames back, three
ahead -- on the first two and last two layers, `(16, 0)` between. It was trained that way; the
architecture's point is bounded latency.

transformers (5.14) applies those windows **only when the call carries an `attention_mask`**
(`MoonshineStreamingEncoder.forward` builds the per-layer masks inside `if attention_mask is not None`).
Without one, every layer attends to the whole clip. So "the reference" is two different functions
depending on an argument:

* the model card's usage passes `**processor(...)`, which includes the mask -- windows on;
* transformers' own docstring example passes `input_values` alone -- windows off.

The processor's mask has a second effect: it zero-pads a clip to whole 80-sample frames and **masks
the partial last frame**, zeroing its embedding. Without the mask the partial frame's real samples
are normalised and embedded like any other.

The card also caps decoding: `max_length = int(n_samples * 6.5 / 16000)`, "to avoid hallucination
loops". transformers' `generate` has no such default (it would stop at 20 tokens).

## Options

1. **The docstring's call** (no mask): one bidirectional attention per layer, simplest graph. Not the
   model the card describes, and not what it was trained as.
2. **Both, chosen by an input**: a flag the driver turns into two mask shapes. Two behaviours to verify
   and document for one checkpoint, and nothing upstream asks for the second.
3. **The card's call, entirely**: windows always, the partial frame zeroed, the card's token cap as the
   driver's default budget.

## Decision

Option 3. The card is the authority on how the checkpoint is meant to be run, and where transformers'
behaviour depends on an argument, the export takes the value the card passes.

* The windows are built in the graph from the frame count (one topology for every length).
* The partial frame needs no mask: an all-zero frame embeds to exactly zero (per-frame CMVN of zeros is
  zero, the projection has no bias, `silu(0) = 0`), so the driver zeroes the partial frame's real
  samples and pads it whole. The engine already pads to whole frames for files declaring
  `loom.samples_per_chunk` (family 3's contract), so the driver touches at most 79 samples.
* The budget is `floor(audio_samples * 6.5 / 16000) - 1` new tokens (`max_length` counts the start
  token), computed exactly. The card computes it in float32, which rounds up by one at exactly one
  length the file accepts (1,284,923 samples); the driver header records that.

## Consequences

* loom and the card's call agree in token ids on jfk.wav and on all 73 LibriSpeech-dummy utterances,
  for both sizes; the encoder at f64 equals the card's call to 1.2e-14.
* A clip under ~0.3 s gets a budget of zero tokens and returns no text, as the card's call does.
* `max_new_tokens` still overrides the budget, as for every ASR driver.
* The same rule applies to the next checkpoint whose transformers forward branches on a mask or flag:
  name the branch, and take the card's.
