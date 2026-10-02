---
type: adr
status: accepted
date: 2026-10-02
tags: [tokenizer, bpe, sentencepiece, vocab, moonshine, engine]
supersedes: []
---

# ADR-065: A Converted SentencePiece BPE Declares Its Dummy Prefix

## Context

Moonshine Streaming ships `tokenizer.json` only: a SentencePiece BPE that transformers' converter
turned into a fast tokenizer -- literal UTF-8 pieces, U+2581 for a space, `<0xNN>` byte fallback, merges
in the order SentencePiece's scores implied (the merged piece's id rises along the merge list, one
inversion in 61,249). Its normalizer is `Prepend("▁")` then `Replace(" ", "▁")`; its decoder
ends `Fuse` then `Strip(" ", 1, 0)`. That pair is SentencePiece's `add_dummy_prefix`.

loom.cpp already reads this family as `BpeShape::kSpmByteFallback` -- Gemma 3's, whose normalizer has
the `Replace` and no `Prepend`. Exported without a name, the file was detected as byte-level BPE
(its llama.cpp chkhsh is in no table) and every U+2581 decoded to nothing:
"Andso,myfellowAmericans".

## Options

1. **Rebuild a SentencePiece "llama" vocabulary** (`loom::Vocab`'s BPE) from `tokenizer.json`, with
   scores reconstructed from the merge order. Its encode merges by score, which equals HF's merge
   rank only where the order is exactly SentencePiece's (one inversion here is enough to doubt it),
   and that path refuses byte fallback, so it would also need an encode change.
2. **Write the vocabulary without the prefix** and strip a leading space on the host. Decodes right;
   encodes every first word differently from transformers, silently.
3. **Declare the prefix on the existing shape**: `tokenizer.ggml.add_space_prefix`, llama.cpp's own
   key, read by `BpeVocab` on `kSpmByteFallback` only.

## Decision

Option 3. On that shape the key prepends U+2581 to each segment of an encode (HF normalizes each
stretch between added tokens on its own) and strips one leading space from a decode. The key on any
other shape is a load error: a byte-level vocabulary spells a space as a byte, and a prefix it does not
implement would be dropped silently. The exporter writes the key only when the normalizer prepends
U+2581, and refuses a prefix whose decoder does not strip exactly one space back off. A loom-only
table name, `spm-byte-fallback`, names the shape for a converted tokenizer llama.cpp has no `pre`
name for (llama.cpp reads such a file as an SPM vocabulary instead).

## Consequences

* Moonshine's transcripts are transformers' character for character (73/73 LibriSpeech utterances,
  both sizes).
* Merges stay HF's own, by rank: no reconstruction to argue about.
* Gemma 3's file carries no key and is byte-identical; its tokenizer gate
  (`test_e2e_spm_byte_fallback_tokenizer`) passes unchanged against the published GGUF.
* Needs an engine with this (1.0.0-rc14). An older engine ignores the key and decodes with a leading
  space -- a file that runs, only not exactly; it is published with rc14.
* Pinned by `tests/ci/test_spm_prefix_vocab.cpp` (encode, decode, byte fallback, the refusal) and the
  exporter's `tests/ci/test_moonshine_export.py` (the key written, not written, refused).
