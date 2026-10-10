---
type: retro
date: 2026-10-10
domain: model-coverage
tags: [asr, nemo, citrinet, wordpiece, tokenizer, reference, oracle]
---

# Retro-082: The Reference's Tokenizer Was Another Checkpoint's

## The Issue

Citrinet-256-ls (2026-10-10) matched NeMo's own forward id for id on 73/73 LibriSpeech clips, and
NeMo's transcripts scored 14.35% WER. The model card claims 3.78% on test-clean. NeMo's own
`transcribe()` printed `festi [SEP] season`, `wh [MASK]her`, `clas [UNK]s` and `slap [CLS] the`, and
wrapped every utterance in `[CLS] ... [SEP]`.

## Root Cause

The checkpoint's tokenizer is WordPiece (`tokenizer.type: wpe`), a 256-line `vocab.txt` whose first five
lines are `[PAD] [UNK] [CLS] [SEP] [MASK]`. NeMo builds it as
`AutoTokenizer("bert-base-cased", vocab_file=...)`. Under transformers 4.57.6 that tokenizer keeps
**bert-base-cased's special-token ids**, `[UNK]`=100 to `[MASK]`=103, over a vocabulary where 100 to 103
are `##se`, `on`, `##ve` and `##et`. So `ids_to_tokens` maps both 1-4 and 100-103 to the specials.
Every `##ve` is decoded as `[SEP]`.

The model is right. It emits id 2 first and id 3 last on all 73 clips (it was trained on
`[CLS] ... [SEP]` targets) and 100-103 as ordinary pieces. Decoded by its own `vocab.txt`, which is what
the engine's `WordPieceVocab` does, the same ids score **9.39%**.

## What Changed

* The engine decodes WordPiece in `transcribe` and drops the five specials by the vocabulary's own ids.
  Its transcript is the one the vocabulary file spells, not the one NeMo's runtime prints.
* The oracle is the ids, from `m.forward`, and the WER is scored from those ids through `vocab.txt`.
  Both arms were first scored through NeMo's decode, so the 14.35% hid the problem on both sides at
  once.

## Takeaway

**Compare ids, and decode them with the checkpoint's own vocabulary file.** A reference runtime's
tokenizer object is code, and here it was assembled from a second checkpoint's name. A transcript that
looks plausible except for a few bracketed words is a decode problem, not a model problem: check what
`ids_to_tokens` says about the ids the model actually emits before reading a WER.
