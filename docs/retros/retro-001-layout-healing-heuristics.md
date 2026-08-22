---
type: retro
date: 2026-08-03
domain: inference-engine
tags: [correctness, broadcasting, ggml-primitives, silent-wrong-answer]
---

# Retro-001: Causal LMs Were Wrong Whenever `n_tokens == n_head`

## The Issue

Qwen3-0.6B agreed with HuggingFace to `max|Δ| ≈ 2e-5` at every prompt length from 2 to 32 **except 8
and 16**, where it was off by 13.7 and 22.9 logits and the argmax changed. The failure was
length-dependent, not content-dependent: five different 8-token prompts all produced the wrong top-1.
It predated the KV-cache thread entirely — a pre-session export from 2026-07-30 is bit-identical to a
fresh unfused one and failed the same way.

## Root Cause Analysis

`op_add` / `op_mul` / `op_mul_mat` (`src/ops/primitives_basic.cpp`) carry "dynamically heal transposed
layouts" heuristics that infer an operand's intended layout **from its sizes**. Size-based inference is
ambiguous the moment two axes are equal, and a transformer makes that collision happen for real: RoPE
broadcasts `cos [head_dim, n_tokens, 1]` into `q [head_dim, n_tokens, n_head]`, so a branch testing
`n_head == n_tokens` fires exactly at the collision and turns per-token rotation into per-head rotation.
A GQA model failed at **two** lengths, `n_head` and `n_head_kv`.

## Resolution & Lesson Learned

The heuristics were guarded on `ggml_can_repeat` / `can_mul_mat`: none may run when the operands are
already compatible, because at that point there is nothing to fix and a guess can only corrupt.

* **Actionable takeaway 1 — layout is the exporter's to know, not the engine's to guess.** The engine
  sees sizes; the exporter sees the true layout. Every size-guessing branch is a silent-wrong-answer
  generator with no error path. See [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md).
* **Actionable takeaway 2 — argmax-level tests cannot see this class of bug.** The only numeric gate on
  the path used 3- and 7-token prompts, both lengths that happen to pass. What found it was
  `tools/debug/compare_logits.cpp` driving `GraphBuilder` directly. What now prevents it is
  `tests/test_broadcast_axis_collision.cpp`, which sweeps `n_tokens` *across* the head count at the
  primitive level in milliseconds — and was confirmed to fail with the guards reverted.
* **Still open:** the guards make the guessing unreachable for correct graphs; they do not remove it.
  `op_repeat` remains unguarded. Tracked in
  [the backlog](../backlog/active-index.md#engine--correctness).

---

## Full record (verbatim from the ledger)


Found while measuring something else — the fused-vs-unfused logit comparison KV-CACHE.md stage 2 asked
for. **It predated the KV-cache thread entirely**: the pre-session `qwen3_0.6b_mil_monolithic.gguf`
(2026-07-30) is bit-identical to a fresh unfused export (`max|Δ| = 0.000e+00` over the whole 151k-token
logit vector) and failed the same way.

**Symptom.** Qwen3-0.6B agreed with HF to `max|Δ| ≈ 2e-5` at every prompt length from 2 to 32 **except
8 and 16**, where it was off by 13.7 and 22.9 logits and the argmax changed. Length-dependent, not
content-dependent — five different 8-token prompts all gave the wrong top-1, and `"A B C D E F G H"`
predicted 425 instead of `" I"`.

**Root cause.** `op_add`/`op_mul`/`op_mul_mat` (`src/ops/primitives_basic.cpp`) carry "dynamically heal
transposed layouts" heuristics that infer an operand's intended layout **from its sizes**. Those are
ambiguous the moment two axes are equal, and a transformer makes that happen for real:

* RoPE multiplies `cos [head_dim, n_tokens, 1]` into `q [head_dim, n_tokens, n_head]`. `op_mul`'s
  branch `a->ne[2] == b->ne[1] && b->ne[2] == 1` reads `n_head == n_tokens` — true only at the
  collision — and permuted `cos` to `[head_dim, 1, n_head]`, turning **per-token rotation into
  per-head rotation**.
* `op_mul_mat`'s `a->ne[1] == b->ne[2] && a->ne[2] == b->ne[1]` does the same to attention's own
  `q`/`k`, both `[head_dim, n_tokens, n_head]`.

A GQA model therefore failed at **two** lengths (`n_head` and `n_head_kv`), because k/v carry the
smaller head count. Confirmed by construction: tiny models with `(n_head_kv, n_head)` of (2,4) and
(3,6) failed at exactly {2,4} and {3,6}.

**Fix.** None of these heuristics may run when the operands are *already* compatible — at that point
there is nothing to fix and a guess can only corrupt. Guarded on `ggml_can_repeat`/`can_mul_mat`
respectively. This is the same reasoning the NOTE in `op_add` already recorded for a sibling branch
that had been deleted for "re-corrupting already-correct tensors"; the remaining branches had the
identical flaw. The standing principle is in that note: *the real fix belongs at the exporter, which
knows the true layout instead of guessing from ambiguous shapes* — these guards make the guessing
unreachable for correct graphs, but the heuristics themselves are still layout-guessing and should
eventually go.

**Verification.** Qwen3-0.6B now matches HF at every length 2–32 (`max|Δ| ≈ 2e-5`), fused and unfused;
all three tiny models have no failing lengths; LFM2's HF-token gate 8/8; Whisper 13/13; `ctest` 142/142.

**Why nothing caught it, and what changed.** `test_e2e_lfm2_mil_export` was the only numeric gate on
this path and its prompts are 3 and 7 tokens — both lengths that happen to pass.
`tests/test_broadcast_axis_collision.cpp` now sweeps `n_tokens` *across* the head count at the
primitive level (no checkpoint, milliseconds), and was confirmed to fail with the guards reverted.
`tools/debug/compare_logits.cpp` is the harness that found it: it drives `GraphBuilder` directly,
because the driver's `infer` entry returns an argmax — the resolution at which this bug is invisible.

---

## Still open: who depends on these heuristics?

Follow-up to the fixed bug above, and the reason it is not fully closed. The guards added in `6c170a1`
make the size-guessing *unreachable for graphs that are already correct* — they do not remove the
guessing. Any graph that genuinely fails the compatibility check still gets a layout inferred from
ambiguous sizes, and **nobody currently knows which models that is**. The heuristics were added
empirically, one model at a time, and `op_add`'s own NOTE records the principle they violate: *the real
fix belongs at the exporter, which knows the true layout instead of guessing from ambiguous shapes.*

**Remaining surface** (`src/ops/primitives_basic.cpp`):

| site | state |
|---|---|
| `op_mul_mat` (2 branches) | guarded by `can_mul_mat` |
| `op_add` (2 branches) | guarded by `ggml_can_repeat` |
| `op_mul` (3 branches) | guarded by `ggml_can_repeat` |
| `op_repeat` (2 branches) | **UNGUARDED** |

`op_repeat`'s second branch is the same bug, still live:

```cpp
if (a->ne[0] == ne[1] && a->ne[1] == ne[0]) {   // fires on ANY square target, correct or not
    a = ggml_permute(pc.ctx, a, 1, 0, 2, 3);
}
```

It transposes an already-correct tensor whenever the repeat target happens to be square in its first
two dims — exactly the `n_tokens == n_head` shape of the fixed bug, in a primitive no causal LM
exercises heavily. It was left alone in `6c170a1` because the fix there was verified against causal
LMs, and `op_repeat`'s documented consumer is StyleTTS2's diffusion `Transformer1d` (broadcasting a
style vector `[channels]` to `[channels, T]`) — a path with no elementwise numeric gate to catch a
regression. Guarding it needs its own verification, not a copy-paste.

**The work, in order.** (1) Instrument each branch with a counter naming the op and the shapes, run all
11 models, and record which branches fire for which model — that converts "some model probably needs
these" into a list. (2) For each real firing, fix the layout at the *exporter* so the operands arrive
correct. (3) Delete the branch. A branch that fires for nothing can be deleted immediately, which is
the cheap half and may well be most of them.

**Why this matters beyond tidiness:** every one of these is a silent-wrong-answer generator with no
error path, and the fixed bug shows the failure is invisible to argmax-level tests. Until step 1 is
done, the honest statement about any of them is "we do not know whether it is load-bearing."
