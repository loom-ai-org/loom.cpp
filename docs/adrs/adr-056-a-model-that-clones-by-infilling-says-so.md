---
type: adr
status: accepted
date: 2026-09-30
tags: [hosts, loom-py, f5-tts, family-9, voice-cloning, driver-inputs, exporter]
supersedes: []
---

# ADR-056: A Model That Clones by In-Filling Says So

## Context

F5-TTS has no voice of its own. It clones the voice in a clip by IN-FILLING one spectrogram whose first
frames are the clip, so a call needs three inputs a text door does not have: the clip (`waveform`),
the clip's transcript joined to the text to speak (`text_ids`), and where the join is (`n_ref_text`).
`loom_cli --ref-text` built them from the start. loom-py's `Text2Speech.infer(text)` could not, so the
published card called `model.infer(...)` with the join spelled out.

The driver made the gap worse. The exporter reads every caller input as `inputs.<name> or
inputs.tokens`, which lets a host send a model's primary input under the canonical name. F5 has two
inputs of different kinds and neither is primary, so a bare `text2speech.infer("hello world")` handed
the sentence's ids to the mel front end **as audio samples**, and the call died further in with an
error about neither.

loom-py holds no per-architecture code, so the door cannot key on the model being F5-TTS, and a
model that does not in-fill would take a `waveform` and ignore it, speaking in its own voice. That is
a wrong answer, not an error.

## Options

* **Detect it in the host** (the tokenizer tag `f5`, or `n_ref_text` in the driver source). Rejected.
  It is per-architecture knowledge in the host, which every repo's CLAUDE.md forbids.
* **A contract field read by `ModelContract`.** Rejected. It does not choose the door, since the door
  is `text2speech` either way. It would cost an engine change and a loom-py pin bump for what is a
  host-side assembly rule.
* **An hparam the export writes, `loom.tts.reference`.** Chosen. By [ADR-020](adr-020-audio-codes-is-its-own-modality.md)'s
  split, hparams are what a host reads to BUILD a door's inputs, and `hparam_str` already exists.

## Decision

* **`loom.tts.reference = "infill"`** declares the shape: the transcript, then a space when it does
  not end in one, then the text; `n_ref_text` counts the transcript's ids including that space.
  The join is `loom_cli --ref-text`'s, which is the reference's own `infer_batch_process`.
  The exporter's `hparams()` writer accepts `str` for it (it took only int and float before).
* **loom-py: `Text2Speech.infer(text, reference=, reference_text=)`** builds the three inputs and
  returns `Audio` at the declared rate. It refuses: one half without the other, `tokens=`/`phonemes=`
  with a reference (the join is text-level), an `Audio` at the wrong rate (resampling is the caller's),
  and any model that does not declare `infill`, naming its voice files when it has some. Given text
  alone, an `infill` model says what to pass instead of reaching the driver.
* **A new binding kind, `REQUIRED`**, reads a caller input by its own name only and fails by name
  without it. F5-TTS's `waveform` and `text_ids` use it; every other model keeps `CALLER`'s alias.
* **The card moves to the door only once a released `loom-py-rt` carries it**, since a card must run
  on the published runtime. Until then it keeps its `model.infer(...)` call, which the new file still
  accepts (the names did not change).

## Consequences

* A GGUF exported before the key is refused by the door, which says to re-export. The Hub's F5-TTS
  was republished with it on 2026-09-30; the driver and one KV changed and the tensors did not. Its
  door output is bit-identical to the old card's `model.infer(...)` on the old file (max |d| 0).
* The next model that clones from a clip plus its transcript declares its own value if its inputs
  differ. Qwen3-TTS's ICL mode is one such case, but it takes the transcript as a separate
  `ref_tokens` and lives on `text2codes`.

## Related

* [ADR-013](adr-013-one-door-per-task.md): one door per task, declared by the file
* [ADR-020](adr-020-audio-codes-is-its-own-modality.md): the contract/hparam split
* [ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md): F5-TTS's sampler
* [ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md): voices as files, the other way in
* [Epic-06](../epics/epic-06-high-level-api-and-hosts.md): the hosts
