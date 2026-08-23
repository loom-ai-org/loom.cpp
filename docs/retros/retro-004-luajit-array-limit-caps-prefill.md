---
type: retro
date: 2026-08-05
domain: inference-engine
tags: [lua, marshalling, large-vocab, scaling-limit]
---

# Retro-004: The Lua Boundary Capped Prefill Length for Large-Vocab Models

## The Issue

Found while gating P4.0.11a. `infer` raised `table overflow` on long prompts — for Gemma 3 at roughly
**512 prompt tokens**, for Qwen3 near 880, for LFM2 near 2048.

## Root Cause Analysis

`MonolithicCall` returned the topology's whole `[n_vocab, n_tokens]` logits tensor across the Lua
boundary so that `ArgmaxEpilogue` could argmax a single row, and LuaJIT's array part tops out near
2^27 elements. For Gemma 3's 262144-wide vocabulary a 600-token prefill is 157M doubles. Nothing on
the roadmap had a vocabulary large enough to hit it before.

## Resolution & Lesson Learned

`loom.run_subgraph_argmax(module, axes, inputs, row)` — the argmax happens on the C++ side and only the
result crosses.

**Actionable takeaway:** the host-language boundary has capacity limits that scale with the model, not
with the code. A marshalling shape that is fine for eleven models can be a hard ceiling for the
twelfth; move the reduction to the side that owns the memory.

---

## Full record (verbatim from the ledger)


Found while gating P4.0.11a. `MonolithicCall` returns the topology's whole `[n_vocab, n_tokens]` logits
tensor across the Lua boundary so `ArgmaxEpilogue` can argmax a row — and LuaJIT's array part tops out
near 2^27 elements. For Gemma 3's 262144-wide vocab that is **~512 prompt tokens**, past which `infer`
raises `table overflow`; a 600-token prefill is 157M doubles. Qwen3 (151936) caps near 880, LFM2
(65536) near 2048, so nothing on the roadmap has hit it before.

**Fixed on the flattened path** by `loom.run_subgraph_argmax(module, axes, inputs, row)`: the module ran
identically and returned one number, the argmax of the requested row, read from the tensor with `nb[1]`
as the row stride so the other rows were never touched. Nothing crossed the boundary but the answer —
the Lua boundary stays a *per-step* boundary rather than a per-logit one, the same reasoning
`KV-CACHE.md` §1.1 gives for not driving attention from Lua.

Gated on KV-cached topologies only, so the blast radius was the causal LMs: the vocab is what makes the
cap reachable and only that family has one, so no ASR/TTS driver text moved. Gemma's 600-token prefill,
which raised `table overflow`, returned HF's own top-1; Qwen3 and LFM2 agreed with iterated `infer`
22/22 each.

**Then fixed on the modular path too, and the fused call retired — P4.0.14 (2026-08-06).** That entry is
where this ends: the modular chain's last stage retains like every other one, both builders reduce with
`loom.argmax_row(module, row)`, and `run_subgraph_argmax` no longer exists. The one-decision-in-two-
components property survives the move in a stronger form — `MonolithicCall.retained` /
`ArgmaxEpilogue.retained_module` are checked by `driver_ir.check_subgraph_calls`, which knows what a
module is, rather than by `validate` noticing an absent local. The cap is now gated by a test that
prefills past it (`test_e2e_prefill_past_marshalling_ceiling`) rather than by a number in this table.

