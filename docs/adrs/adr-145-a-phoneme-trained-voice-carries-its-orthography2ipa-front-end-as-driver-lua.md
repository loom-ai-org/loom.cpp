---
type: adr
status: proposed
date: 2026-11-27
domain: model-coverage
tags: [tts, vits, phonemizer, orthography2ipa, text-front-end, lua, driver]
supersedes: []
---

# ADR-145: A Phoneme-Trained Voice Carries Its orthography2ipa Front End as Driver Lua

Serves [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). Amends
[ADR-012](adr-012-permissive-phonemizer.md) (orthography2ipa as the permissive G2P, rules in the engine as one copy).
Follows [ADR-002](adr-002-embedded-lua-drivers.md) (orchestration is an embedded Lua driver) and
[ADR-142](adr-142-kale-tts-runs-its-javascript-front-end-in-the-engine.md) (a model's front end runs in the engine
from its own file).

## Context

ADR-012 chose orthography2ipa as the G2P for phoneme-trained voices, through a C++ port with the rules and data in
the engine as one copy. Part 2 of Epic-07's phonemization plan, the port, is still open. Until it lands, a VITS
voice has a text door only in loom-py, through the Python package, and no text door in C++, the web build or any
other host.

Two facts changed since ADR-012:

1. **orthography2ipa is a compiler and two small interpreters.** Each language spec compiles to a versioned IR, a
   JSON document whose revision is a hash of its content. A Python and a Lua interpreter run it, and the rules
   engine is gone (orthography2ipa #2526, #2550). The Lua interpreter is about 3,600 lines and needs nothing from
   loom except the driver's environment. `orthography2ipa/export_to_loom.py` writes it, with the IR of each
   language, as a driver script.
2. **The Lua interpreter equals Python.** On loom, 3,881 of 3,881 conformance vectors and 6,369 of 6,369 gold
   sentences are byte-identical to `orthography2ipa.transcribe` (orthography2ipa #2756, in review).

A voice also depends on the rules it was trained with. Its training phonemes came from one revision of the spec
data. A later rule change can give a symbol sequence that the voice never heard, and ADR-012's single copy changes
every voice at once.

## Options

* **Keep ADR-012: wait for the C++ port.** One copy of the rules, improvable for every voice at once. But the text
  door stays absent outside loom-py until the port lands, and the port is a second implementation that must track
  the Python one rule by rule.
* **The front end in loom-py only.** Works today. Every other host still has no text door.
* **A separate `text2phonemes` GGUF beside each voice.** One file per language, shared by voices. The host must
  load two files and know which goes with which, which is ADR-013's one-door-per-task problem moved to the host.
* **The voice's GGUF carries the o2i driver in front of its own.** Chosen.

## Decision

A phoneme-trained voice may carry its orthography2ipa front end in its own GGUF, as driver Lua:

* **The driver** is the o2i interpreter with the compiled IR of the voice's language(s), as
  `export_to_loom.driver_script()` writes it, followed by the voice's own driver. The o2i block runs in its own
  function scope, and its `infer` becomes a local.
* **`infer({text = <codepoints>, ...})`** runs o2i, maps the IPA with the voice's own phoneme table
  (`tokenizer.ggml.tokens` and the `tokenizer.ggml.phoneme.*` framing), and calls the voice's driver.
  `infer({tokens = ...})` is unchanged. **`phonemize({text})`** returns the IPA as codepoints, so a host can check
  the front end against Python.
* **The file declares the front end:** `loom.text.frontend = "o2i"`, `loom.phonemizer.ruleset`, and `o2i.version`,
  `o2i.language`, `o2i.spec_revision`. The revision is the IR hash that orthography2ipa computes, never one that
  loom computes.
* **The exporter writes it,** as an option of the VITS export config (`frontend="o2i:<lang>"`), from the
  orthography2ipa package installed at export time. The engine gets no code.

ADR-012's licence decision and its fold-down risk stand. Its rejection of per-GGUF rules is replaced for voices that
carry this driver. A voice exported without the option keeps the loom-py door.

## Measured (2026-11-27)

[Phoonnx/phoonnx_kab-DZ_miro_o2i](https://huggingface.co/Phoonnx/phoonnx_kab-DZ_miro_o2i), a VITS trained on o2i
kab-DZ IPA (162 symbols), exported with loom-exporter #101 and the o2i driver put in front:
[TigreGotico/kab-DZ-miro-loom](https://huggingface.co/TigreGotico/kab-DZ-miro-loom). loom-py-rt 1.0.0 from PyPI,
Common Voice Kabyle sentences:

| Check | Result |
| :--- | :--- |
| `phonemize` against Python `orthography2ipa.transcribe` (7.133.1a3), 2,000 sentences | 2,000 identical |
| `infer(text)` against `infer(tokens)` with phoonnx's own ids, same seed, 50 sentences | 50 sample-identical |
| ASR WER, NVIDIA Kabyle conformer, 50 sentences | 0.360 (the phoonnx ONNX voice: 0.372) |
| Real-time factor, laptop CPU | 0.039 |
| File size | 62.9 MB, of which 0.2 MB is the front end |

## Consequences

* **Positive:** a voice has a text door on every host loom runs on, the web build included, with no engine code.
* **Positive:** the voice keeps the rules it was trained with. A rule change in orthography2ipa does not change the
  voice until it is re-exported, and the recorded revision says which rules it has.
* **Positive:** there is no second implementation of the rules to keep in step. The C++ port that Epic-07's part 2
  plans is not needed for these voices.
* **Negative:** a rule fix reaches a voice only through a re-export. This is the corollary that ADR-012 named. The
  recorded revision makes the re-export a mechanical step.
* **Negative:** each voice carries its own copy of the interpreter (129 KB) and of its IR (76 KB for kab-DZ). A
  multi-voice host loads the same rules more than once.
* **Limit:** only a language whose text needs no normalisation has its whole front end in o2i. Kabyle is one. A
  language that needs number and date expansion first still needs that step outside the file.
