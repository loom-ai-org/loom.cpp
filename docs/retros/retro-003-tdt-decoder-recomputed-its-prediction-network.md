---
type: retro
date: 2026-08-06
domain: inference-engine
tags: [asr, transducer, caching, redundant-compute]
---

# Retro-003: The TDT Decoder Recomputed Its Prediction Network on Every Blank Frame

## The Issue

`TdtDecoder::decode_greedy` ran the whole LSTM stack once per inner iteration of the decode loop.

## Root Cause Analysis

The prediction network's output is a pure function of `(last_label, h, c)`, and all three change **only
when a token is emitted**. On a blank the loop discarded `h_new`/`c_new`, advanced the frame, and
recomputed bit-identical values from identical inputs on the next pass. Most frames of real audio are
blank, so that was the bulk of what the decoder spent its time on.

## Resolution & Lesson Learned

The recomputation was removed; the decoder itself later left C++ entirely for Lua (see
[ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md)).

**Actionable takeaway:** in an autoregressive loop, ask which inputs actually changed since the last
iteration. A loop whose body is a pure function of state that only changes on one branch is
recomputing on every other branch.

---

## Full record (verbatim from the ledger)


The second finding from the same scoping pass, and independent of the Lua migration, so it lands on
its own.

`TdtDecoder::decode_greedy` ran the whole LSTM stack once per inner iteration. Its output is a pure
function of `(last_label, h, c)`, and all three change **only when a token is emitted** — on a blank the
loop discarded `h_new`/`c_new`, advanced the frame, and recomputed bit-identical values from identical
inputs next time round. Most frames of real audio are blank, so that was the bulk of what the decoder
did. Caching it is what NeMo's own implementation does; the equivalence argument is that the discarded
recompute could not have differed.

The state now commits where it is produced rather than on emission, which is the same condition said
the other way round: reaching the run at all means the state is about to become current.

**Measured on 11s of real speech (`samples/jfk.wav`), parakeet-tdt-0.6b: the prediction stack runs 37
times instead of ~140** — once per emitted token plus the initial one, against once per frame. Decode
wall-clock moves less than that ratio suggests, median **~1.25s → ~1.04s over five runs each**, because
the joint network is the widest matmul in the loop (1024 → 8197) and still runs every frame. This
machine's run-to-run spread is wide enough that the timing is worth little; the call-count is exact.

**Gate — an A/B on the real models, because the existing tests cannot reach the branch.** The
`parakeet-rnnt` reference fixture decodes to an *empty* token list, so `test_e2e_parakeet_rnnt` would
have compared empty against empty and passed either way. Instead, the same binary ran both checkpoints
over `samples/jfk.wav` before and after the change:

  * parakeet-tdt: 36 tokens, identical ids and identical frame indices.
  * parakeet-rnnt: 26 tokens, identical — and this is the branch that matters most, since plain RNN-T
    forces `skip = 0` on emission and therefore re-enters the inner loop, the one path where the cache
    must invalidate rather than persist.

`test_e2e_parakeet_tdt` also still reproduces `reference_forward_parakeet_tdt.py`'s own tokens and
frame indices exactly (16/16, "Yeah.").

