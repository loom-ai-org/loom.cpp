---
type: adr
status: accepted
date: 2026-10-01
tags: [tokenizer, text-frontend, speecht5, family-9b, exporter, engine]
supersedes: []
---

# ADR-059: A Number Speller Ships as Data Beside the Vocabulary

## Context

SpeechT5's vocabulary is 75 characters with no digits, so `2026` reached the model as one `<unk>`
([ADR-057](adr-057-a-char-sentencepiece-model-ships-as-unigram.md)). The model cannot be given
digit ids: its embedding table has no rows for them, and it was never trained on any. What the
reference offers is `SpeechT5Tokenizer(normalize=True)`, which runs transformers'
`EnglishNumberNormalizer` on the text before tokenizing. That function strips thousands separators and
spells each number, with its sign, currency, percent and up to two decimals, as English words.

## Options

* **Spell numbers in the host** (loom-py). Rejected. It is per-model code in a host, and `loom_cli`
  would not get it.
* **A `speecht5` vocabulary tag with its own C++ reader.** Rejected. The segmenter is plain Unigram,
  and the speller is not SpeechT5-specific: any SentencePiece model can declare one.
* **The speller's shape in C++, its data in the file** ([ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md)'s
  split), run by `loom::Vocab` when the file declares one. Chosen.

## Decision

`tokenizer.ggml.numbers.scheme = "english_number_normalizer"` makes `loom::Vocab::encode` run
`loom::NumberSpeller` on the text before SentencePiece's own normalisation, which is where the
reference runs it. The exporter (`number_normalizer_export`) writes the rest from the reference:

* its words (`ones`, `teens`, `tens`, `scales`) and currency names, in the reference's dict order;
* the pattern's optional symbol chain, in the PATTERN's order, which is not the dict's. The exporter
  reads it out of the reference's own source and refuses a mismatch;
* Python's `\d` and `\w`, the two classes the reference's regexes use, as codepoint tables computed
  from `unicodedata`: the zero of each Nd run of ten, and `isalnum() or '_'` as ranges (66 runs,
  734 ranges).

SpeechT5's export turns it on (`normalize_numbers=True`), where upstream defaults it off. Off, every
number is one `<unk>`, so the only inputs that change are ones that could not be spoken.

**Two deliberate deviations, both where the reference drops words:** an all-zero integer part is
"zero" (upstream spells it as nothing, so `0` vanished and `0.5` became " point five"), and every
thousands separator goes (upstream's comma pass never rescans, so `1,000,000` became `1000,000` and
was spoken "one thousand" and then nothing). Where the reference raises (two currency symbols in one
number, more than 36 digits), the text is left unchanged.

**Verified differentially**, against the reference with those two fixes patched in: 40,046 random
strings (digits, separators, every currency, Unicode digits such as `٣`, NBSP, `_`) produce identical
spelled text and identical ids. `tests/ci/test_number_speller.cpp` pins 28 cases from that reference.
Whisper reads back loom's audio for `$15,000.50`, `3%`, `May 3, 2026` and `-5` as written.

## Consequences

* The differential found an older engine divergence that this ADR's decision did not need, but
  exposed: `loom::Vocab`'s Unigram encoder did not fuse unknown runs
  ([Retro-067](../retros/retro-067-unigram-did-not-fuse-unknown-runs.md)).
* **Wheels first.** An engine before this change ignores the keys, so a re-exported SpeechT5 file
  runs on rc11 as today, with numbers as `<unk>`. The card says numbers are spoken only from the next
  release, and the Hub copy is not republished until then.
* A second language is a second set of word tables under the same scheme if its grammar is this
  function's shape. A different shape is a new scheme name, which older engines refuse by name.
