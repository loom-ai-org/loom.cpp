---
type: adr
status: accepted
date: 2026-09-26
tags: [engine, tokenizer, exporter, family-9, voxtral, mistral]
supersedes: []
---

# ADR-054: A tiktoken Vocabulary Is Merged by Rank, in the Shared BPE

## Context

Voxtral-4B-TTS encodes its text with **Tekken**, Mistral's tokenizer since Nemo. `tekken.json` is a
tiktoken vocabulary: 1000 markers, then byte sequences in RANK order, and a regex. `mistral_common`
builds a `tiktoken.Encoding` from it, and three things about that encoding differ from every byte-level
BPE the engine had:

* **The regex splits a letter run at a case change.** `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}...]+`
  makes "HelloWorld" two words. No `BpeShape` could express it, and `tokenizer_detect` listed "tekken"
  among the regexes it could not ("case-transition/camelCase shapes").
* **Merging is by the rank of the MERGED bytes**, and there is no merge list: tiktoken joins the
  adjacent pair whose concatenation has the lowest rank.
* **No normalization and no marker split.** tiktoken encodes the string as given (no NFC), and it is
  built with `special_tokens={}`, so a typed "[AUDIO]" is seven characters of text.

## Options

1. **A per-model vocabulary class** (`VoxtralVocab`), as VoxCPM2 and CosyVoice3 got. Rejected: nothing
   here is Voxtral's. The same file is every Mistral model's tokenizer, and a class named after one
   model would be copied for the next.
2. **Rebuild a merge list from the ranks and use the existing merge loop.** This is what llama.cpp's
   and HF's converters do: for each token, BPE its bytes with the lower ranks until two parts remain,
   and record that pair. Rejected: the result equals tiktoken's rule only when each token's canonical
   split is the pair that meets during encoding. It is also 130k strings in the file and a minute of
   Python at export.
3. **A new shape in `BpeVocab`, merged by rank.** Chosen.

## Decision

`BpeShape::kTekken`, selected by `tokenizer.ggml.pre = "tekken"` under `tokenizer.ggml.model = "gpt2"`:

* **The regex is scanned by hand**, like the other shapes. The two letter alternatives overlap (Lm, Lo
  and M are in both classes), so `match_tekken_letters` writes out the regex's backtracking: the
  optional prefix is tried first, and the greedy `A*` gives back one character at a time until `B+`
  can start. Two new Unicode tables (Lu+Lt and Ll) come from the same generator as the others, at the
  same UCD version (14.0).
* **Merging asks the rank.** Ids past the markers are the ranks in order, so the id of the
  concatenation is the merge priority. No merges are written or read.
* **No NFC, and no added-token split** for this shape. The markers are still typed CONTROL, for
  `decode` and `is_control`.
* **The piece table keeps the rank when a marker shares its spelling.** It is filled from the end, so
  that text reaches the rank and not the marker.

The exporter writes it from `tekken.json` directly (`tekken_tokenizer_export`). The markers come
first, then the kept ranks in GPT-2's byte spelling. The file's regex is checked against the one the
shape scans, and a different one raises.

## Consequences

* **12000/12000** texts over twelve classes (camelCase, NFD, Arabic and Hindi, whitespace, paths,
  typed markers, ...) encode to `Tekkenizer.encode`'s ids. An NFC sabotage arm drops it to 1095/1200.
  `tests/ci/test_tekken_vocab.cpp` pins each rule on a hand-traced fixture, and two sabotages of the
  shape turn it red.
* **A TTS with a plain BPE front end needs no per-model host code.** `loom_cli` gained one branch
  (task text-to-speech + tag "gpt2"), and loom-py's `text2speech` door worked unchanged.
* **Any Mistral LM's tokenizer is now within reach** of the causal-LM path. None has been exported yet.
* **An rc10 engine refuses the file by name** ("unimplemented pretokenizer family 'tekken'"), rather
  than tokenizing it some other way.
* The Unicode tables stay at 14.0, where tiktoken's regex engine uses a newer UCD. Characters assigned
  after 14.0 can split differently. None of the gated classes has one.

## Related

* [Epic-03](../epics/epic-03-model-coverage.md) family 9, Voxtral-4B-TTS
* [ADR-033](adr-033-a-decode-only-table-is-still-a-vocabulary-family.md): when a tokenizer DOES get its
  own tag
* [Retro-059](../retros/retro-059-a-shared-tokenizers-whitespace-was-ascii.md): the shared `\s` this
  shape reuses
