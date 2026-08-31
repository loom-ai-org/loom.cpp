---
type: epic
status: active
domain: host-api
last_updated: 2026-08-29
---

# Epic-06: The High-Level API and Its Hosts

## 1. Context and Scope

Two hosts consume the engine: `loom_cli` and `loom-py`. This epic covers the API they present, the
layer each piece of behaviour belongs in, and what a GGUF must declare for a host to dispatch on it
without knowing the architecture.

The authority is [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md); this epic is the orientation.

## 2. Architectural Overview

**One door per task, chosen by the modality pair the file declares** —
[ADR-013](../adrs/adr-013-one-door-per-task.md). X2Y interfaces come off `loom.task` plus the declared
input/output modalities, with canonical input names. A knob with no role in the declared contract stays
`infer`-only.

**The layering rule:** in the **FILE** when it is a property of the checkpoint; in the **ENGINE** when
it is a property of the task; in the **HOST** when it needs the host's ecosystem. Corollary, which
settles the hard cases: anything shipped inside a GGUF can only be fixed by re-exporting every model,
so an evolving policy must not be baked into files even when Lua could express it.

Net: **per-task code may live in any layer; per-architecture code only in the exporter.**

### What the engine owns

| | |
|---|---|
| `include/loom/core/model_contract.h` | the one place that knows the declared KV names. Every reader is absence-tolerant; `declared()` separates a file that states its contract from one a caller must know about |
| `include/loom/core/text_generate.h` | `loom::text::generate` — one LM loop, both driver shapes, the file's own EOS **set** (`eos_token_ids`, plural since P4.23) and its own decode rule (P4.24) |
| `include/loom/core/chat_template.h` | `loom::ChatTemplate` — a conversation to the prompt text a checkpoint was tuned on, from role tags the EXPORTER reduced its Jinja to ([ADR-018](../adrs/adr-018-chat-template-as-role-tags.md)) |
| `include/loom/core/session.h` | topologies registered and caches attached once, owned in an order that cannot dangle |
| `transcribe.cpp` | reads the declared ASR table; Whisper's spellings survive only as a flagged legacy fallback |

### Resolution order for anything a model can infer itself

**Optional argument → autodetect from the file → default.** Language is the worked example. Capability
is declared by the **export**, not by the host.

### Compatibility

Pre-contract GGUFs still work through the legacy fallback, and **the fallback must stay tested** — a
declared-only test would stay green while the path every GGUF on disk depends on rotted. The `v4`
fixture set is kept for exactly this reason; `v5` is the contract-declaring set.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Authority | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) |
| Decisions | [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md), [ADR-018](../adrs/adr-018-chat-template-as-role-tags.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| Active tasks | [Backlog → Host API](../backlog/active-index.md#host-api) |

## 4. The Record

### P4.24 — the engine could not sample: `argmax_row` was the only decode rule — DONE (2026-08-29)

**One sentence.** There was no temperature, top-k, top-p or multinomial draw anywhere in `src/` or
`include/`; every causal LM decoded greedily, and a checkpoint whose own `generation_config.json` says
`"do_sample": true` was run in a mode its authors did not choose.

`loom.sample_row(module, row, {temperature =, top_k =, top_p =})` now sits beside `loom.argmax_row`,
reducing ON the tensor and drawing from the bridge's shared `rng_`.

#### The shape, and the two decisions inside it

**Greedy is `temperature = 0`, not a separate flag**, and that is what makes the two bindings incapable
of disagreeing: at `temperature <= 0` or `top_k == 1`, `sample_tensor_row` returns
`argmax_tensor_row`'s own answer, by calling it. Two ways to get a token out of one forward pass that
can differ is the failure P4.0.14 already retired once.

**The knobs' path is `optional argument → autodetect from the file → default`** (Epic-06 §2). The
exporter writes the checkpoint's `generation_config.json` sampling defaults as `loom.sampling.*` KVs
and as the driver's own `inputs.temperature or <default>` fallbacks; `GenerateOptions`' three knobs are
`std::optional` and only a SET one is passed. Unset therefore means "what the file declared", and a
file declaring nothing means greedy — a host default filled in anywhere along that path would silently
overrule every checkpoint for every caller who named nothing.

The order is `transformers`' own processor order — temperature, then top-k, then top-p, then the draw —
because the reference this is defined against is `generate` under the checkpoint's own generation
config. Any other order gives a different distribution from the same three numbers.

#### Why it is a bridge builtin at all, against the criterion that seems to forbid it

`lua_bridge.h`'s binding criterion lists "a sampler" among the things that do NOT belong in C++. That
line meant an orchestration — a LOOP over steps, like the ADPM2 diffusion sampler, which is still Lua —
and it now says so. `loom.sample_row` is a **reduction over one row**, the same shape as `argmax_row`:
it reads no model config, every knob is a call argument, two unrelated families could use it unchanged,
and its input cannot cross the boundary at all — `run_subgraph` marshals every element into a Lua table
and LuaJIT's array part tops out near 2^27, so a 262144-wide vocab overflows at ~512 prompt tokens
([Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md)). Sampling in Lua would reinstate
that ceiling at exactly the vocabulary size that found it. The decode loop AROUND it is still driver
Lua.

Temperature + top-k + top-p covers the whole fixture set and exactly one model needs any of it
(gemma-3-270m-it: `top_k 64, top_p 0.95`). Min-p, the repetition penalties, Mirostat and beam search are
not implemented and nothing asks for them.

#### Gates

`tests/ci/test_sample_row.cpp`, over the 6-class toy fixture — which is what makes "every class is
eventually drawn" a statement a test can make:

* every greedy spelling equals the argmax — an empty options table, `temperature = 0`, `top_k = 1`, and
  a `top_p` below the best token's own probability (the "at least one candidate survives" clause);
* one seed gives the same 40 draws twice; a different seed differs;
* at a flattening temperature all six classes are reached, which no argmax ever is;
* `top_k = 2` returns exactly two distinct ids, and the argmax is among them;
* a negative `top_k` and a `top_p` outside `(0, 1]` raise by name.

Sabotaged to confirm it fails: forcing the greedy branch takes the test red.

**Greedy stayed the default and it was verified rather than asserted** — a re-exported greedy
checkpoint decodes the same ids, and every fixture in `v5` carries a driver that calls `argmax_row`, so
nothing already on disk changed path at all.

