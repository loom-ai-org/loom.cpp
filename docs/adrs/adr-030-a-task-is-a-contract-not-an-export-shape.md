---
type: adr
status: accepted
date: 2026-09-12
tags: [exporter, model-coverage, family-11, registry]
---

# ADR-030: A Task Is a Contract, Not an Export Shape

## Context

`audio-codec` declared `AudioCodecExportConfig` as its base class, and `TaskRegistry.register()`
checks every family registered under a task against it. That held while the task had one family:
DAC and then SNAC are `Flattened` exports — codes in, waveform out, one traced graph.

EnCodec is the same contract and a different shape. Its decoder contains a 2-layer LSTM over the time
axis, and a topology is a pure dataflow graph: no node in one can carry state across timesteps. So it
is three phases — the RVQ sum and first convolution, the recurrence, the upsampling stack — with a
host-side loop between them, which makes it a `BaseMultiPhaseModelExportConfig` and not an
`AudioCodecExportConfig` at all.

## Decision

**`audio-codec` declares the root `LoomExportConfig`**, and the registry check for this task becomes
the weak one. The recognizer stays in `audio_codec_export.py` — detection is a property of the family
— and routes `model_type == "encodec"` to `encodec_export.py`.

## Why

The alternative readings were worse in ways worth naming:

* **Widen `AudioCodecExportConfig` to cover both shapes.** It would have to become a multi-phase
  config that DAC and SNAC use one phase of, which is fitting the simple case into the general one for
  the sake of a type check.
* **Give EnCodec its own task.** It has the same input kind, the same output kind, the same four
  hparams and the same caller-facing door. A task the caller cannot distinguish is not a task.
* **Leave the declaration wrong.** `register()` would reject the family outright.

`automatic-speech-recognition` reached the same answer before this, on the same evidence: three
families under it (NeMo encoders, transducers, Whisper) build three config classes, and its entry
argues that the only true common base is the root "when its families genuinely do not share an export
shape". This is the second instance, which is what turns that argument into a rule: **the task fixes
the contract; the decomposition and the driver are the family's own.**

## Consequences

* The check that `config_class` is right for a task no longer bites for `audio-codec`. The real fix is
  the one already recorded at the ASR entry — move `config_class` onto the RECOGNIZER, where the build
  actually happens, so each family is checked against its own base rather than against a shared one.
  Two tasks now want it.
* `CodecFamily` lost its `ENCODEC` member and the two branches carrying its `decode` signature and
  config spellings, which were written against the export it turned out not to need. A live branch
  selecting an unusable path is the failure that enum's own docstring warns about.
* `DriverInputs` gains nothing, but the driver side gains `RecurrentCall` — the caller `RecurrentPhase`
  never had. Any future family with an LSTM now costs one phase and one component per layer.
