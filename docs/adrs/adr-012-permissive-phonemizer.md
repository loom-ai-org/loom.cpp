---
type: adr
status: accepted
date: 2026-08-15
tags: [licensing, phonemizer, tts, text-frontend, submodule]
---

# ADR-012: A Permissively-Licensed Phonemizer, With Rules in the Engine

## Context

VITS, Kokoro, StyleTTS2 and Matcha-TTS all take phoneme ids. None of them has a real text→phoneme
front end, so none has a usable text door. (SupertonicTTS is the exception in this family: its
`TextVectorizer` is a licence-free unicode codepoint lookup table.)

`espeak-ng`-based phonemization was confirmed to work numerically via the external `piper_phonemize`
package — but **both stock espeak-ng and the piper-phonemize fork are GPL-3**, incompatible with this
repository's MIT licence. Vendoring was rejected on those grounds and the task was blocked.

## Options Considered

1. **Vendor espeak-ng / piper-phonemize.** Numerically proven, licence-incompatible. Rejected.
2. **Shell out to an installed espeak-ng.** Moves the licence question to the user and makes the text
   door depend on a system package an edge deployment may not have.
3. **Embed the rules per-GGUF**, preserving "the model is one file".
4. **A C++ port of [`orthography2ipa`](https://github.com/TigreGotico/orthography2ipa)** (TigreGotico,
   same org as phoonnx), vendored as a submodule — verified **Apache-2.0**, permissive, compatible with
   MIT. Rule-based transduction with no weights: ~900 language JSON specs plus one language-agnostic
   engine (tokenizer, beam search, allophone rules, stress, sandhi) — the same interpreter/data split
   this project argues for everywhere else.

## Decision

**Option 4, with the rules and data living in the engine, one copy.**

Embedding per-GGUF was considered and rejected despite preserving "the model is one file": it trades
that for a re-export corollary, which is the wrong side for data expected to keep improving. Single
source of truth, improvable for every model at once by bumping the submodule, no re-export.

The file declares only what it needs — `loom.text.phoneme_alphabet`, `loom.text.languages`, its own
symbol→id table — plus `loom.phonemizer.ruleset`, the version it was validated against, so a rule
change that alters output is **attributable**. It **WARNS, never fails**.

**The Python door comes first, and it is not a stopgap:** it is the oracle the C++ port is verified
against, the same relationship `fixture_gen/`'s reference forwards have to the exported graphs.

## Consequences

* **Positive:** the licence blocker is gone and the repository stays MIT.
* **Positive:** the task splits in two, and only the second half needs the port. The phoneme symbol
  table is data already sitting in each checkpoint and simply not exported; exporting it as a
  vocabulary family gives four TTS models a real `model.tokenizer` with no licence question involved.
* **Risk (measurable before any C++) — the fold-down.** `orthography2ipa` is a *superset* of the union
  of espeak-ng, Epitran and others. Feeding a checkpoint richer IPA than its training front end
  produced is a low **quality** risk — these models degrade gracefully. What the superset *does*
  require is a fold-down into each checkpoint's fixed symbol→id table, because a symbol outside it has
  no id at all: a lookup with no answer, not a degradation.
* **Risk — tie-breaking.** The beam search returns ranked lattices, and the tie-break must be pinned or
  the CLI and `loom-py` drift. That is exactly the failure [ADR-013](adr-013-one-door-per-task.md)
  exists to stop.

## Related

* Epic: [Epic-07: Text Front-Ends and Tokenizers](../epics/epic-07-text-frontends-and-tokenizers.md)
* Design detail: [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) §5
* Ledger record, verbatim:


VITS, Kokoro, StyleTTS2, and Matcha-TTS drivers all still take raw token-id/demo text input — none of them
do real text→phoneme conversion. Real `espeak-ng`-based phonemization was confirmed to work numerically
(via the external `piper_phonemize` Python package) but vendoring it was rejected: both stock espeak-ng and
the piper-phonemize fork are GPL-3, incompatible with this repo's permissive licensing (see
`[[loom_engine_licensing_phonemizer]]` memory). SupertonicTTS is the one model in this family that needs
no phonemizer at all — its `TextVectorizer` is a license-free unicode codepoint lookup table.

**The licence blocker is gone.** The plan is now a C++ port of
[`orthography2ipa`](https://github.com/TigreGotico/orthography2ipa) (TigreGotico, same org as phoonnx),
vendored as a **submodule**: verified **Apache-2.0**, which is permissive and compatible with this repo's
MIT. It is rule-based transduction with no weights — ~900 language JSON specs plus one language-agnostic
engine (tokenizer, beam search, allophone rules, stress, sandhi), which is the same interpreter/data
split this repo argues for everywhere else. `src/text/phonemize.cpp` + `include/loom/text/phonemize.h`
remains the intended shape.

**Decided with it (user direction, 2026-08-15; docs/HIGH-LEVEL-API.md §5):**

* **Rules and data live in the engine, one copy** — single source of truth, improvable for every model
  at once by bumping the submodule, no re-export. Embedding per-GGUF was considered and rejected
  despite preserving "the model is one file": it trades that for the re-export corollary, which is the
  wrong side for data expected to keep improving. The file declares only what it needs
  (`loom.text.phoneme_alphabet`, `loom.text.languages`, its own symbol→id table) plus
  `loom.phonemizer.ruleset`, the version it was validated against, so a rule change that alters output
  is *attributable*. It WARNS, never fails.
* **Python door first, the CLI's native one scoped as the target.** The Python path is not a stopgap:
  it is the oracle the C++ port is verified against, the same relationship `fixture_gen/`'s reference
  forwards have to the exported graphs.

**Task #79 splits in two, and only the second half needs the port.** The phoneme symbol table is data
already sitting in the checkpoint and simply not exported; exporting it as a vocabulary family gives
four TTS models a real `model.tokenizer` and a working `synthesize(phonemes=...)` with no licence
question involved at any point.

**Two risks, both measurable before any C++ (§5).** `orthography2ipa` is a superset of the union of
espeak-ng, Epitran and others, harmonized across references and literature-validated, so feeding a
checkpoint richer IPA than its training front end produced is a low *quality* risk — these models
degrade gracefully, Piper managing in some languages with graphemes substituted outright. What the
superset does require is a **fold-down** into each checkpoint's fixed symbol→id table, since a symbol
outside it has no id at all and that is a lookup with no answer rather than a degradation. Second: the
beam search returns ranked lattices, and the tie-break must be pinned or the CLI and loom-py drift —
the exact failure P5.0 exists to stop.

**Corrected 2026-08-11:** that entry used to call Supertonic "fully closed", which was true of the engine
and false of everything a user touches. `loom::SupertonicTextVectorizer` existed and was gate-verified,
but nothing was wired to it: the MIL export never wrote the KVs, and neither `loom_cli` nor loom-py
dispatched on the tag. Now done end to end — the export carries `tokenizer.ggml.supertonic.*`, both hosts
read it, and `model.tokenize("hello world")` returns the same ids the real Python `TextVectorizer` does.
The export is otherwise byte-identical (snapshot diff: three added KVs, every topology, the driver and all
683 tensors unchanged).

