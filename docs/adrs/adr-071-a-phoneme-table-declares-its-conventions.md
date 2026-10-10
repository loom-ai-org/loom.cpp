---
type: adr
status: accepted
date: 2026-10-10
tags: [text-frontends, phonemes, g2p, orthography2ipa, lexicon, sanotts, loom-py]
supersedes: []
---

# ADR-071: A Phoneme Table Declares the Conventions It Was Trained In

Serves [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). It builds on
[ADR-012](adr-012-permissive-phonemizer.md) (orthography2ipa as the permissive G2P) and on the
"fold-down" that ADR-012 named as the cost of a G2P whose output is a superset of a checkpoint's.

## Context

A phoneme vocabulary in a GGUF says which SYMBOLS a checkpoint knows. It does not say which
CONVENTIONS its training data wrote them in. The text door's G2P (orthography2ipa plus a lexicon such as
ipa-dict, as the VITS card recommends) and espeak (what every Piper voice was trained on) disagree on
several things. orthography2ipa:
* writes stress before a syllable's onset (`ˈkwɪk` vs `kwˈɪk`);
* stresses every monosyllable (`ˈðə`);
* leaves English vowels unmarked for length (`i`, `ɑ`, `ɔ` vs `iː`, `ɑː`, `ɔː`);
* writes `ɝ` and `ɫ`, which espeak never does and the table may not hold.

A large model hears through that: Piper's VITS is word-perfect through the door either way. sanoTTS's
1.46M-parameter amy is not. Whisper WER on 30 LibriSpeech sentences, through the text door:

| front end | amy (piperlite, espeak style) | heart-nano (nano, misaki style) |
|---|---|---|
| upstream's own G2P (the ceiling) | 8.5% | 8.1% |
| ipa-dict lexicon, as phonemized | 66.8% | 81.3% |
| ipa-dict lexicon, folded | 11.1% | 15.7% |
| misaki lexicon, as phonemized | 30.2% | 11.5% |
| misaki lexicon, folded | 12.3% | 11.9% |

One lexicon serves both lines folded (misaki's, Apache-2.0, bundled in the sanoTTS repo); the
differences inside 11-12% are a few words in about 370.

## Options

1. **Card guidance only.** Tell readers to bring espeak-style phonemes. The built-in door then stays
   unintelligible on the voices that need it most.
2. **Per-model lexicons only.** A better-matched lexicon (misaki) helps, but 30% is far from 12%.
   Out-of-vocabulary words still come from the rules in the wrong conventions.
3. **The model declares its style, the host folds to it.** The fold is per STYLE, not per model, and
   the style is a fact about the training data that the export knows.

## Decision

Option 3 (the user, 2026-10-10). An export declares `loom.tts.phoneme_style` (`espeak`, `misaki`).
This is an hparam rather than a contract field, by ADR-020's split, like `tts.reference`: it says how
the host builds the door's input. loom-py's text door folds the G2P's output to that style
(`loom.phonemizers.fold`) before the model's table encodes it.
* Only text the door phonemized is folded. `phonemes=` from a caller is taken as given.
* Language-independent rules apply to every language: stress before the vowel, `ɫ`→`l`, `r`→`ɹ`, and
  misaki's compressed symbols to espeak's. English-only rules apply only to `en-*`: length, stressed
  `ə`→`ʌ`, and `ɝ`/`əɹ` by stress. Only English has been measured.
* An unknown style is refused, never passed through.

## Consequences

* sanoTTS declares `espeak` (piperlite) and `misaki` (nano). VITS does not yet: switching it changes
  published audio, so it is a hub item with an A/B, beside the Piper framing question
  ([Retro-081](../retros/retro-081-piper-has-two-framings.md)). Kokoro (misaki) is the other candidate.
* The rules live in Python today. When orthography2ipa's C++ port lands, the fold moves with it into
  the engine, under the same declared key.
* A remaining gap: function-word stress (`ðˈʌ` for "the"). A per-word rule needs the text, which the
  fold does not see. It is one unmeasured part of what still separates ~12% from ~8%.
