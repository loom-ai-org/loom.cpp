---
type: adr
status: accepted
date: 2026-10-01
tags: [high-level-api, asr, translation, contract, engine, canary]
supersedes: []
---

# ADR-061: The Language Written Is Its Own Role

## Context

ASR had six canonical roles ([HIGH-LEVEL-API §4](../HIGH-LEVEL-API.md#canonical-input-names)), two of
them about language: `language` (what the audio is in) and `task` (`transcribe` or `translate`).
They were enough for Whisper, whose only translation is into English, so `translate` names its output
language as well as its operation.

Canary-1b-v2 ([Epic-03](../epics/epic-03-model-coverage.md)) writes any of its 25 languages from any
of them: its `canary2` prompt carries a source-language token and a target-language token, and
English-to-French is as ordinary as French-to-English. `task` cannot say which target; a model-specific
knob would put the most common thing a caller wants from this model behind `infer`.

Canary was also the first dynamic-length ASR export with a prompt a run-time argument can reach, and
`transcribe` used to hand such a file only the waveform and its length.

## Options

1. **Overload `task`** with `translate:fr`-style names. Keeps the role count, but `task` stops being
   an operation, and every host has to parse it.
2. **A per-model knob** (`target_lang` through `infer`). No API change, and the common case is not
   reachable from `transcribe`.
3. **A seventh role, `target_language`**, resolved by the engine against a table the file declares.

## Decision

Option 3. The file declares the languages it can WRITE as `loom.asr.target_language_names` /
`_ids` -- parallel arrays, like the source table, and separate from it because the two sets need not
match. `TranscribeOptions::target_language` (loom-py `transcribe(target_language=)`, `loom_cli
--target-language`) is resolved against that table and passed to the driver as an id; omitted, it is
omitted, and the driver applies its own default.

**Canary's default target is English** (the user's decision, 2026-10-01), even for English audio,
where it is simply the transcript. The driver's precedence is `target_language`, then `task`
(`transcribe` = the source language, `translate` = English), then English.

Two engine rules follow:

* A **declared table reaches the driver on either decode path**. `transcribe`'s dynamic-length branch
  passes `language`, `task` and `target_language` when the file declares the tables; a file with none
  is called exactly as before, so parakeet's literal `<|en|>` piece still selects nothing.
* A `target_language` on a file that declares no target table **throws**, with no
  "ignored, with a warning" branch like `language` has: naming an output language is always a request
  for output in it, and a model that cannot choose would answer in its own language.

## Consequences

* The high-level API's ASR roles are seven. A future family that chooses its output language costs no
  API change.
* Whisper declares no target table, so `target_language="en"` on it throws and points at
  `task="translate"`. Mapping one to the other would make the role mean two things.
* `task` keeps its meaning for Canary: the export maps `transcribe` to 0 ("the source") and
  `translate` to `<|en|>`.
* Pinned by `tests/ci/test_transcribe_no_selection.cpp` against `dynamic_asr_languages.gguf`, whose
  target table is deliberately a SUBSET of its source table, so the test can tell them apart.
