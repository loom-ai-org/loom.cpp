---
type: retro
date: 2026-08-29
domain: text-frontend
tags: [tokenizer, testing, oracle, case-selection, p4.23]
---

# Retro-021: Nine Verbatim-Oracle Cases, And Not One Of Them Was A Marker

## The Issue

`BpeVocab` shipped unable to encode a special token. `tokenize("<|im_start|>")` returned seven literal
ids where it should return one, `tokenize("<start_of_turn>")` eight where it should return `[105]` —
so a chat template was not merely un-applied, it was **unrepresentable**, and every instruction-tuned
LM in the fleet was being run outside the distribution it was trained on (P4.23).

It shipped past a gate written specifically to hold this tokenizer to the reference implementation.
`test_e2e_spm_byte_fallback_tokenizer` had **nine cases, every expectation
`AutoTokenizer.encode(text)` verbatim, all of them exact and all of them round-tripping.**

## Root Cause Analysis

The nine cases were chosen to hit each **structural difference** the SPM-byte-fallback family had from
every other shape — a bare first word, a multi-space run, combining characters, CJK, an astral emoji,
literal tab/newline, digits, the empty string. That is a good list, and it was the right list *for the
thing being added at the time*, which was a new `BpeShape`.

**None of the nine contained a special token**, because special tokens are not a property of the shape.
They are orthogonal to it: every BPE family has them, and none of the four shapes handles them
differently. So a case list organised by "what is different about this family" had no slot for the one
class of input that was broken in every family at once.

The oracle was never wrong. `AutoTokenizer.encode("<start_of_turn>")` would have said `[2, 105]` on the
day the test was written, exactly as it says now. **It was simply never asked.**

Two smaller things kept the gap invisible:

* `detokenize` worked. A special token's *spelling* is in the vocabulary like any other entry, so
  `decode([105])` returned `<start_of_turn>` and every round-trip assertion passed. The asymmetry was
  the whole defect and the test's round-trip direction was the half that worked.
* The file did not carry the information either — `tokenizer.ggml.token_type` was never written — so
  even a correct `encode` could not have known which ids were added. A test could have failed here and
  the fix would still have been in two repositories.

## Takeaway

**A verbatim-oracle test is only as good as its case selection, and case selection is where the
thinking that produced the code leaks into the thing that is supposed to check it.** Copying the
reference's own outputs removes every chance of being wrong about an answer and none of being wrong
about the question.

So when adding a component under a reference oracle, enumerate cases along **two** axes, not one:

1. what is different about this implementation (the axis that gets written naturally, because it is
   the axis you were just thinking along);
2. what classes of input the component's *interface* admits at all — here: ordinary text, and control
   tokens, which every tokenizer has and no shape treats specially.

The second axis is the one that catches a defect shared by every variant, which is exactly the kind
that no amount of per-variant testing finds.

**And check both directions of an invertible pair separately.** `decode(encode(s)) == s` held for all
nine cases while `encode` was wrong, because the two failures were not symmetric: decode reads a table
the vocabulary already has, encode has to *decide* something. A round-trip is not a test of either
half.

The gate now carries nine more cases on the second axis — the markers, a marker inside prose, a
non-special added token, and the two the added set must **not** swallow (`<bos>`, which is added, and
`<0x41>`, which is a byte-fallback entry and must stay literal text). It was verified to fail against
the artifact that shipped, which is the only thing that makes any of the above a claim rather than a
hope.
