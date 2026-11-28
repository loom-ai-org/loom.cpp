---
type: adr
status: proposed
date: 2026-11-28
domain: model-coverage
tags: [tts, phonemizer, orthography2ipa, text-front-end, lua, tokenizer]
supersedes: []
---

# ADR-145: A Phoneme Voice Carries Its Rule-Based Phonemizer, and the Engine Runs It Behind One Interface

Serves [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). Amends
[ADR-012](adr-012-permissive-phonemizer.md) (orthography2ipa as the permissive G2P, rules and data in the engine as
one copy). Follows [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md) (a front end's rules ship as data,
its shape is code), [ADR-013](adr-013-one-door-per-task.md) (one door per task) and
[ADR-003](adr-003-per-model-complexity-in-the-exporter.md) (per-model complexity in the exporter).

## Context

ADR-012 chose orthography2ipa as the G2P for phoneme voices (VITS, Kokoro, Matcha, StyleTTS2), through a C++ port
with the rules and data in the engine. Epic-07's phonemization part 2, the port, is open. Until it lands, a phoneme
voice has a text door only in loom-py, through the Python package (`loom-py-rt[phonemes]`).

Three facts changed since ADR-012:

1. **orthography2ipa is a compiler and two small interpreters.** Each language spec compiles to a versioned IR, a
   JSON document whose revision is a hash of its content. A Python and a Lua interpreter run it, and the rules
   engine is gone (orthography2ipa #2526, #2550).
2. **The Lua interpreter equals Python, on both of loom's Lua VMs.** On loom, 3,881 of 3,881 conformance vectors
   and 6,369 of 6,369 gold sentences are byte-identical to `orthography2ipa.transcribe` (orthography2ipa #2756).
   On the Kabyle measurement below, the same holds under LuaJIT 2.1 and under PUC Lua 5.1.5 with LuaBitOp
   (`-DLOOM_LUA=puc`, the web build's VM, ADR-139).
3. **A voice depends on the rules it was trained with.** Its training phonemes came from one revision of the spec
   data. A later rule change can give a sequence the voice never heard, so a newer rule set is not always an
   improvement for that voice. ADR-012's single copy changes every voice at once.

ADR-041 puts a front end's rules in the file and its shape in the engine. kale-tts follows it (ADR-142): its front
end is C++ (`KaleTtsVocab`, `JsRegex`), and only its data comes from the file. **This ADR puts the shape in the
file too**: the interpreter that runs the rules is a Lua script in the GGUF. That difference is the decision.

## Options

* **Keep ADR-012: wait for the C++ port.** One copy of the rules, improvable for every voice at once. But the text
  door stays absent outside loom-py until the port lands, and the port is a second implementation that must track
  the Python one rule by rule.
* **The o2i interpreter in front of the voice's driver** (the first Kabyle file, loom.cpp #146). It works on 1.0.0,
  but it sets `loom.text.frontend = "o2i"`, which no host door knows, so `text2speech.infer(text)` fails. It also
  copies the symbol table and Piper's framing into the driver, and leaves the interpreter's globals in the driver's
  state. Rejected.
* **A declared driver function** (`loom.phonemizer.entry = "phonemize"`), which each host's `phonemes` path calls.
  No engine change, but the phonemizer shares the driver's Lua state, and each host carries the branch.
* **A rule-based phonemizer interface in the engine, with its script and data from the file.** Chosen.

## Decision

### The interface

`loom::RuleBasedPhonemizer` (`include/loom/text/phonemize.h`, Epic-07's planned slot) sits beside the tokenizers,
with `phonemize(text, lang) -> IPA`. Its first implementation runs a Lua script the file carries:

* in **its own Lua state**, not the driver's, opened without `io` and `os`: a phonemizer is a pure function;
* through a fixed **script contract**, so the engine never knows which phonemizer it runs:
  * `phonemize(codepoints, lang) -> codepoints, dropped`. `lang` is the 0-based index into `loom.text.languages`.
    `dropped` is optional: the codepoints the script could not read, in text order;
  * optionally `set_lexicon(lang, entries)`, a run-time overlay of `word -> IPA` entries.
* through one **host function** in that state, defined before the script runs: `phonemizer_data(lang) -> string`,
  the value of `loom.phonemizer.data.<lang>` for index `lang`, or `nil`. The script pulls a language's data the first
  time it needs it and caches what it builds, so a file with many languages decodes only the ones a call uses.

The engine exposes it the way it exposes `tokenize`. loom-py, `loom_cli` and the web JS API each make one generic
call in their `phonemes` path: the file's phonemizer when the file declares one, an installed provider otherwise.
IPA to ids stays in the voice's `PhonemeVocab`, with its framing and ADR-071's phoneme style and fold.
`loom.phonemizers.set_lexicon` routes to the file's phonemizer when there is one.

### The keys

| Key | Value |
| :--- | :--- |
| `loom.text.frontend` | `"phonemes"`, unchanged |
| `loom.text.languages` | the phonemizer's languages, in index order |
| `loom.phonemizer.kind` | `"lua"` |
| `loom.phonemizer.script` | the script that implements the contract |
| `loom.phonemizer.data.<lang>` | the data of one language (for orthography2ipa, the compiled IR JSON), outside the script |
| `loom.phonemizer.ruleset` | the version and the revision of each language's data |
| `loom.phonemizer.phonemes.<lang>` | optional: the phoneme set the data can emit |
| `loom.phonemizer.identity` | optional: JSON, the phonemizer identity of each language, as a voice config records it |
| `loom.phonemizer.unsupported` | optional: the rules of the source phonemizer that the file does not run |

The data stays apart from the script: it keeps ADR-041's split, a host can list the languages without running the
script, and a later C++ implementation of the same interface could read the same keys with no script.

With the rules in the file, `loom.phonemizer.ruleset` describes the rules the file carries. It is no longer "the
version this export was validated against", and a host does not warn on a mismatch with an installed package.

### Text the script cannot read

The script reports and does not refuse or fold. A codepoint it cannot read is left out of the IPA and returned in
`dropped`, and the host warns, as the Python package warns (`UnmappedGraphemeWarning`). Digits are such text: number
and date expansion is not part of a phonemizer. A symbol of the IPA that is not in the voice's table is
`PhonemeVocab`'s case and follows its rule.

### The exporter

One option shared by the phoneme TTS export configs writes the `loom.phonemizer.*` keys from the phonemizer package
installed at export time (for orthography2ipa, `orthography2ipa.export_to_loom`). The voice's driver stays as the
config writes it. The exporter does not edit generated Lua, and the host does not merge two GGUFs.

ADR-012's licence decision and its fold-down risk stand. Its rejection of rules in each GGUF is replaced for files
that carry `loom.phonemizer.script`. A file without one keeps the installed provider.

### Epic-07 part 2

The C++ port of orthography2ipa is not needed for files that carry their phonemizer. It stays open only for phoneme
voices exported without one, and a later C++ implementation would be a second implementation of
`RuleBasedPhonemizer` that reads the same keys.

## Measured

[Phoonnx/phoonnx_kab-DZ_miro_o2i](https://huggingface.co/Phoonnx/phoonnx_kab-DZ_miro_o2i), a VITS trained on
orthography2ipa kab-DZ IPA (162 symbols), with the interpreter and the kab-DZ IR in the file:
[TigreGotico/kab-DZ-miro-loom](https://huggingface.co/TigreGotico/kab-DZ-miro-loom), Common Voice Kabyle sentences.
The file uses the rejected driver layout; the script, the IR and the ids are the same in the chosen one.

| Check | Result |
| :--- | :--- |
| IPA against Python `orthography2ipa.transcribe` (7.133.1a3), 2,000 sentences, loom-py-rt 1.0.0 | 2,000 identical |
| The same, loom.cpp `main` with LuaJIT 2.1, a native host | 2,000 identical, median 112 µs per sentence |
| The same, loom.cpp `main` with `-DLOOM_LUA=puc` (PUC Lua 5.1.5 + LuaBitOp 1.0.3) | 2,000 identical, median 475 µs per sentence |
| Audio from text against audio from phoonnx's own ids, same seed, 50 sentences | 50 sample-identical |
| ASR WER, NVIDIA Kabyle conformer, 50 sentences | 0.360 (the phoonnx ONNX voice: 0.372) |
| Size of the phonemizer in the file | 205 KB: interpreter 129 KB, kab-DZ IR 76 KB |

## Consequences

* **Positive:** a phoneme voice has a text door on every host loom runs on, through the door each host already has
  (`text2speech.infer(text)`), with no per-model engine code (ADR-003).
* **Positive:** the voice keeps the rules it was trained with. A rule change in orthography2ipa does not change the
  voice until it is re-exported, and `loom.phonemizer.ruleset` says which rules it has.
* **Positive:** the interface serves any phoneme family and any rule-based phonemizer that implements the contract.
* **Negative:** a rule fix reaches a voice only through a re-export. This is the corollary that ADR-012 named.
* **Negative:** each file carries its own copy of the script and its data. A host with several voices loads the same
  rules more than once.
* **Limit:** the phonemizer does no number or date expansion. Text with numbers needs that step before
  `text2speech.infer`, or the digits are dropped and reported.
