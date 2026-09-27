---
type: retro
date: 2026-09-25
domain: engine
tags: [engine, tokenizer, bpe, unicode, family-9, cosyvoice3]
---

# Retro-059: A Shared Tokenizer's Whitespace Was ASCII, and Only a New Model's Diff Looked

## The Issue

CosyVoice3's text front end ([ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md))
was diffed against the reference over 900 generated texts. The first run had two differences, both in
the "exotic" class and neither in the new normalisation code:

```
'…9¹¹\xa0.²é…'   ref … 4102 25847 …      got … 4102 13 14154 …
'…ß\xa0\xa0٣\t'  ref … 4102 4102 27856   got … 9238 27856
```

## Root Cause

`BpeVocab`'s pre-tokenizer implements the Qwen2/GPT-2 split regex by hand, and its `is_ws` (the
regex's `\s`) was ASCII's six characters. `tokenizers` evaluates `\s` as Unicode's White_Space: NBSP,
U+3000, U+2000-200A and eight others. So an NBSP before punctuation joined a `[^\s\p{L}\p{N}]` run
("\xa0." as one pre-token) where the reference split it off. Two NBSPs became one run that a merge then
fused (`9238`) where the reference gives two tokens. The pre-token boundary moved and the ids changed.

This affected **every model tagged `gpt2`** (the Qwen2/Qwen3 and LFM2 causal LMs and Whisper among
them; the SentencePiece-style byte-fallback shape has no regex and was not affected), on any text with a
non-ASCII space. No earlier diff found it because none generated one: the causal-LM and ASR oracles
use prose.

## The Fix

`is_ws` is now White_Space exactly. Python's extra U+001C-001F are NOT included: they are
`str.isspace()`, which is `is_python_space` and a different set. Every one of the 32 whitespace
characters, in 8 contexts each, now matches `tokenizers` (256/256), and 9000/9000 texts through the
whole front end match. `test_cosyvoice3_vocab.cpp` pins the NBSP pair: with the old `is_ws` it goes red.

## Takeaway

**A diff harness for one model's front end is also a harness for the shared code under it.** Generate
the characters nobody types (non-ASCII spaces, other scripts' digits, superscripts) in every such diff,
because those are exactly where a hand-written regex and the real one part ways. The exotic class was 2
texts in 900, and those 2 were the only finding.

## Related

* [ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)
* [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md)
