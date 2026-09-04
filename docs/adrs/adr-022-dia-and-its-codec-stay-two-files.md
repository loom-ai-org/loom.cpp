---
type: adr
status: accepted
date: 2026-09-03
domain: model-coverage
tags: [model-coverage, family-10, family-11, codec, packaging, composition]
---

# ADR-022: Dia and Its Codec Stay Two Files

## Context

Dia emits codec tokens and no audio. Its output kind is `audio_codes`
([ADR-020](adr-020-audio-codes-is-its-own-modality.md)), its interface is `text2codes`, and what turns
those tokens into a waveform is DAC — family 11's first leaf, exported and verified separately. The
pair is the point of family 10: text → Dia → nine delayed code streams → realign → DAC → waveform.

**How that pair ships was left open on purpose.** ADR-020 named the modality and said explicitly that
"whether family 10 ships as one GGUF or two is a separate decision this does not make". This is that
decision.

It is not a free choice, because loom has a stated property pulling the other way: **the model is one
file.** A GGUF carries its own topologies, its own driver and its own vocabulary precisely so that a
host needs no sidecars, and `multi_phase_export` already packs several traced phases into one file —
Dia itself is three. Merging DAC in would be a fourth phase and about twenty more lines of driver, and
the result would declare `text2speech` and hand a caller audio.

## Options

**A. One GGUF, DAC merged.** ~6.6 GB. The file declares `text → audio`, resolves to `text2speech`, and
a host calls one door. Keeps "the model is one file" literally true for this family.

**B. Two GGUFs, chained by the host.** 6.4 GB + 217 MB. Dia declares `text2codes`, DAC declares
`codes2speech`, and the composition is two calls with a frame-major array of integers between them.

## Decision

**Two files (B).** Three reasons, in the order they decide it:

* **The codec is shared and the LM is not.** Dia, Parler, CSM and Orpheus all decode through DAC —
  Dia's own `audio_tokenizer_config.json` names `descript/dac_44khz`, which is the checkpoint family 11
  exported. Merging puts the same 217 MB inside every one of them, and a user who has two of those
  models has downloaded the codec twice. The asymmetry is the argument: one codec serves ~20 LMs, so
  the file that should be shared is the one that would be duplicated.
* **Merging makes the codes unreachable.** They are the useful intermediate, not an implementation
  detail — they are what a caller caches, edits, streams a frame at a time, or feeds to a different
  codec at a different rate. A merged file can only return audio, so the `text2codes` interface would
  exist and have no model behind it.
* **ADR-020 already paid for this.** `audio_codes` was made its own modality so that the two halves
  would *compose* rather than be spelled as one thing; a merged file spends that and gets nothing the
  host could not do itself in two calls.

Against: "the model is one file" now reads "the model is one file, and a pipeline is a pipeline". That
is the honest statement, and the same one already holds for every host that runs a phonemizer in front
of a TTS model.

## Consequences

* **The join is not covered by either file's own tests, so it needs its own.**
  `tests/gate/test_e2e_dia_dac_composition.cpp` is that test: it drives the real Dia GGUF, feeds its
  codes into the real DAC GGUF, and compares both the codes and the waveform against `transformers`
  running the identical pipeline, at **two clip lengths**. Two, because family 11's own failure was a
  decoder that returned one frame's worth of audio for every input and raised nothing — a single clip
  length cannot see that, since any constant is consistent with itself. Observed at HEAD: exact
  integer equality on every code, and max |diff| 5.3e-6 on a waveform peaking at 0.009.
* **`loom.codec.n_codebooks` is the contract between the files**, written on both sides under the same
  key — Dia from its channel count, DAC from its quantizer count. That is what lets a host check that
  a pair fits before running 33 decoder steps into a shape error, and the composition gate asserts a
  host could.
* **The reference is a script, not a paste.** `scripts/dia_dac_reference.py` chains both HF models and
  writes the `.npy` fixtures, so the oracle is re-derivable — the thing family 2's gate does not have.
  It decodes greedy and CFG-free to match the driver, and has to learn sampling on the same commit the
  driver does.
* **loom-py needs a `Text2Codes` door** for the first half; `Codes2Speech` already exists for the
  second. The composition in Python is then two `infer` calls, which is what a host writes.
* **The delay pattern stays with the LM.** It always did — DAC knows nothing about it — but the choice
  here is what keeps that true: a merged file would have had a place to hide it.

## See also

* [ADR-020](adr-020-audio-codes-is-its-own-modality.md) — why codec tokens are their own modality
* [ADR-021](adr-021-dias-decoder-resolves-two-dynamic-axes.md) — Dia's second dynamic axis
* [ADR-013](adr-013-one-door-per-task.md) — one door per task
* [Epic-03 §2](../epics/epic-03-model-coverage.md) — families 10 and 11
