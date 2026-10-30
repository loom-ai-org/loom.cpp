---
type: adr
status: accepted
date: 2026-10-25
domain: text-frontends
tags: [text-frontends, phonemes, arpabet, cmudict, g2p, loom-py, auroratts]
---

# ADR-088: ARPAbet Is a Phoneme Style the Door Folds To

Serves [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md) and
[Epic-03](../epics/epic-03-model-coverage.md) (AuroraTTS-100K). Extends
[ADR-071](adr-071-a-phoneme-table-declares-its-conventions.md), whose styles (`espeak`, `misaki`) are
IPA conventions; this one is another alphabet.

## Context

AuroraTTS-100K was trained on g2p_en's output: ARPAbet with stress digits (`DH AH0`, `K W IH1 K`), from
CMUdict plus a neural guesser for unknown words, with `" "` between words and punctuation as tokens.
Its 79-symbol table is ARPAbet. The text door's G2P (orthography2ipa plus a lexicon) produces IPA, and
nothing in loom maps one to the other. Two smaller facts shape the encoding:

* ARPAbet phonemes are several letters long, and the engine's phoneme vocabulary matches the longest
  symbol at each position: concatenated, `D HH` ("adhere") reads as `DH` and an orphan `H`, and `N G`
  ("engage") as `NG`. Phonemes have to be written apart, and upstream's word boundary is itself a space.
* g2p_en is Apache-2.0 but pulls nltk, a neural model and a tagger; CMUdict, what it looks words up in,
  is BSD-2-Clause and is data.

## Options

1. **Card guidance only**: the card shows g2p_en producing ids outside loom. No text door.
2. **An `arpabet` alphabet with its own provider** (a CMUdict lookup in loom-py, beside the IPA one).
   Two G2P paths to keep in step, and out-of-vocabulary words would still need IPA rules.
3. **An `arpabet` style**: the door phonemizes to IPA as for every model, and the fold rewrites it to
   ARPAbet tokens; CMUdict ships as an IPA lexicon written so the fold returns it exactly.

## Decision

Option 3 (the user, 2026-10-25).

* **The export declares `loom.tts.phoneme_style = "arpabet"`** and spells upstream's word boundary `|`
  in the table (the ids are upstream's). A phoneme string is CMUdict-shaped: `"DH AH0 | K W IH1 K"`. The
  engine skips the spaces (no symbol holds one), so no engine change is needed to encode it.
* **The fold** (`loom.phonemizers`, `_fold_arpabet`) reads every IPA spelling the door's lexicons and
  rules produce (ipa-dict's, misaki's compressed symbols, orthography2ipa's, espeak's length marks). A
  vowel takes the digit of the stress mark before it and 0 without one; a schwa is AH0 whatever marks
  it (orthography2ipa stresses every monosyllable, `ˈðə`, and CMUdict's "the" is `DH AH0`); an
  r-coloured vowel's `ɹ` before another vowel is that syllable's onset (`AH0 R EY1`, "arabia").
* **The lexicon is CMUdict rewritten as IPA** (loom-exporter `tools/cmudict_lexicon.py`): each word's
  first pronunciation, the one g2p_en uses, with stress marks before their vowels and a `.` wherever two
  phones would otherwise spell a third (`AO IH0` against `OY`, `AH0 R` against `ER0`, `T SH` against
  `CH`). The fold reads `.` as a syllable break. **Every one of CMUdict's 135,166 entries comes back
  exactly** (commit `7479086`); without the breaks 181 did not.

## Consequences

* Measured through upstream's own model (Whisper-small WER, 30 LibriSpeech sentences): g2p_en's tokens
  8.1%, the door with the CMUdict lexicon 13.4% over four noise draws (misaki's 11.7%; Epic-03 has the
  others and why: contractions are split before the lookup, a defect of the door, not of the fold).
* **The door drops punctuation**, which g2p_en keeps as tokens; orthography2ipa does not emit it. It
  affects every phoneme model's door alike and is left as it is: on this model, g2p_en's tokens
  without their punctuation score 8.9% against 8.1%, a small share of the door's gap.
* A file declaring the style fails on an older loom-py with "unknown phoneme style 'arpabet'" at the
  text door; `phonemes=` and `tokens=` still work. Its card says which release it needs.
* When orthography2ipa's C++ port lands the fold moves into the engine with the others (ADR-071).
