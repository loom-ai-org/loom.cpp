---
type: retro
date: 2026-09-03
domain: text-frontend
tags: [tokenizer, tts, phonemes, host-wiring, tag-dispatch, test-coverage]
---

# Retro-029: A Vocabulary Only Two Of Three Hosts Could Read

## The Issue

Task #79 part 1 — export the phoneme symbol table these four TTS checkpoints were already carrying —
shipped on **2026-08-15**, in two commits on the same day. The exporter wrote the KVs, `PhonemeVocab`
read them back, `loom-py` encoded through it, and all four published GGUFs
(`vits-piper-en-gb-miro`, `kokoro-82m`, `styletts2-ljspeech`, `matcha-tts-ljspeech`) carry a real
`tokenizer.ggml.tokens` today.

The hub still listed it as **the next thing to pick up**, two and a half weeks later. It was neither
done nor open, and the reason it was not done is not the reason it was still listed:

* **`loom_cli` had never learned the `"phonemes"` tag.** A VITS GGUF printed no `tokenizer:` line at
  all, and `--prompt` with an IPA string fell through to the LM path, where `bpe_vocab` is null for
  any non-`"gpt2"` tag, so `parse_token_ids` found no integers in it and the run died on
  `error: --prompt produced no token ids` — with the model's own 159-symbol table sitting unread in
  the same file.
* **`PhonemeVocab` had no test anywhere in this tree.** Every sibling vocabulary here has one.

## Root Cause

Adding a vocabulary family is **four edits, and the fourth has no list**. The exporter writes the KVs,
the engine adds the class, and then *each host* dispatches on the tag. The first three are what the
feature feels like, because together they are what makes the data reachable at all; the fourth is
per-host, and nothing in the tree enumerates the hosts.

This is the second time, not the first. ADR-012's own record carries the Supertonic instance verbatim,
from three weeks earlier: *"`loom::SupertonicTextVectorizer` existed and was gate-verified, but nothing
was wired to it."* The branch that correction added to `loom_cli` even states the trap in its comment —
that falling through would have let a non-`"gpt2"` prompt be parsed as literal token ids. The next
family added walked into exactly that, past a comment describing it.

**Why the suite could not catch it.** The four families' gate tests drive them with synthetic ids
(`{5, 42, 7, 88, ...}`) on purpose — real phonemes are what `scripts/tts_ids.py` exists to produce, for
the ASR oracle — so no test in `loom.cpp` has ever gone through the table. `loom-exporter`'s
`tests/ci/test_tts_text_door.py` holds the write side and `loom-py`'s tests hold its own read; the
engine's class was the one link with a test on neither side of it.

## Takeaway

**A vocabulary is shipped when every host dispatches on its tag, not when the file carries it.** The
export is the half that is easy to see finished, because it is where the interesting work is. Count the
hosts as part of the item — today that is `loom_cli` and `loom-py` — and check the tag reaches each
one, on a real published file rather than a fixture.

**When a branch's comment names a bug it exists to prevent, read it as a class.** The Supertonic branch
did not say "Supertonic would have fallen through"; it said any non-`"gpt2"` tag does. That is a
statement about the *next* family, and the next family was added without it being applied.

**A model whose gate tests feed it synthetic ids has no coverage of its text door, by construction.**
That is the right call for the audio path — driving TTS with real phonemes is what makes an ASR oracle
possible, and it belongs in the gate — but it means the vocabulary needs its own hermetic test rather
than the confidence a green end-to-end run would otherwise buy.

## The Record

The reproduction, on the published GGUF, before the fix:

```
$ loom_cli --model vits-piper-en-gb-miro.gguf --prompt "hˈeɪ wˈɜːld"
loaded 'vits-piper-en-gb-miro.gguf'
  architecture: vits_mil
  graph_topology: Multi-topology file (Lua driven), 2 sub-modules
  weights: 412 tensors
error: --prompt produced no token ids
```

and after, with the assembly printed beside the table because it is declared per checkpoint and the
`-1` in three of the four files is a sentinel rather than an id:

```
  tokenizer: phoneme symbols (phonemes), 159 tokens
  assembly: bos=1 eos=2 blank=0, interleaved between every phoneme
  encode("hˈeɪ wˈɜːld") -> [1, 20, 0, 120, 0, 18, 0, 74, 0, 3, 0, 35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0, 2]
```

All four families were checked against `loom-py`'s `model.tokenize()` on the same string and agree id
for id — which is the property [ADR-013](../adrs/adr-013-one-door-per-task.md) exists to hold, and the
one that had no way of being false before, because only one host could answer.

`tests/ci/test_phoneme_vocab.cpp` covers what does not need a checkpoint: the longest-match scan (a
table holding both `a` and `aɪ`), the per-checkpoint assembly, the `-1` sentinel, the UTF-8-aware skip
for a symbol outside the inventory, first-id-wins for a duplicate, and the two inputs `load` must
refuse — another family's tag, and its own tag with no table. Verified able to fail: capping the scan
at one byte breaks three of its assertions.

Decision and split: [ADR-012](../adrs/adr-012-permissive-phonemizer.md).
Domain: [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).
