---
type: epic
status: active
domain: host-api
last_updated: 2026-08-22
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
| `include/loom/core/text_generate.h` | `loom::text::generate` — one LM loop, both driver shapes, the file's own EOS |
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
| Decisions | [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| Active tasks | [Backlog → Host API](../backlog/active-index.md#host-api) |

## 4. The Record

### P4.24 — the engine cannot sample: `argmax_row` is the only decode rule — SCOPED, NOT STARTED

**One sentence.** There is no temperature, top-k, top-p or multinomial draw anywhere in `src/` or
`include/`; every causal LM decodes greedily, and a checkpoint whose own `generation_config.json` says
`"do_sample": true` is run in a mode its authors did not choose.

Split out of [P4.23](epic-07-text-frontends-and-tokenizers.md) because it is a **capability gap, not a
bug** — nothing here is behaving other than as written. It is half of P4.23's reported symptom, and it
is the half that is not about tokenization.

#### What it costs today

`google/gemma-3-270m-it` ships `{"do_sample": true, "top_k": 64, "top_p": 0.95}`. Run greedily, the
model card's own snippet loops:

```python
model.text2text.infer("The capital of France is", max_new_tokens=14)
#  -> ' Paris.\n\nThe capital of France is Paris.\n\nThe capital of'
```

Greedy decoding on a 270M model is what turns a missing chat template (P4.23) into a repetition **loop**
rather than merely a wrong answer, and it is an **independent** reason any output differs from
`transformers`, whose default for this checkpoint is sampled. Fixing P4.23 alone will not make this
snippet match HF; fixing this alone will not make it correct either. **They are two causes of one
symptom and both are needed.**

#### The constraint that decides the design, and it is already established

**A sampler must run engine-side, as a bridge builtin.** `loom.argmax_row` exists for exactly this
reason and the comment at `src/core/lua_bridge.cpp:283` is the argument: `run_subgraph` marshals every
output element into a Lua table, LuaJIT's array part tops out near 2^27 entries, and a 262144-wide
vocab (Gemma 3) therefore overflows at ~512 prompt tokens — so a driver whose only use for the logits
is one reduction would pay a 157M-element table to compute one integer
([Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md)). **Sampling in Lua would
reinstate that ceiling.** The reduction stays on the tensor; only the chosen id crosses the boundary.

**And the randomness half already exists.** `LoomLuaBridge` owns a `std::mt19937 rng_` seeded by
`loom.seed_rng`, shared by `loom.gaussian_array` and `loom.uniform_array` (Kokoro and StyleTTS2 draw
from both). A sampler joins that stream and that seeding door — with the ordering caveat the comment at
`lua_bridge.cpp:856` already documents, which is that a shared stream makes DRAW ORDER part of the
contract.

So the shape is `loom.sample_row(module, row, {temperature=, top_k=, top_p=, ...})` beside
`argmax_row`, not a new subsystem.

#### Sizing it against the checkpoints actually in the set

| checkpoint | `generation_config.json` |
|---|---|
| gemma-3-270m-it | `do_sample: true, top_k: 64, top_p: 0.95` |
| qwen3-0.6b-base | `do_sample: false` |
| smollm2-360m-it | no sampling keys (greedy) |
| lfm2-350m | no sampling keys (greedy) |

**Temperature + top-k + top-p covers the entire current fixture set, and exactly one model needs any of
it.** Nothing asks for min-p, repetition/frequency penalties, Mirostat or beam search. Scope to the
three, and let the next checkpoint that needs a fourth be the one that adds it.

#### Where the knobs come from — the one decision to take first

Epic-06 §2 already states the rule: **optional argument → autodetect from the file → default**, with
capability declared by the export rather than the host. Applied here:

* the **exporter** writes the checkpoint's `generation_config` sampling defaults as KV (it already
  reads that file for other purposes — `granite_speech_export.py:437`, `qwen3_asr_export.py:363`);
* `GenerateOptions` gains the three knobs, defaulted to "unset";
* unset means "use what the file declared"; the file declaring nothing means **greedy**.

**Greedy must remain the default when nothing is declared, and that is not conservatism — it is what
keeps every byte-identity baseline valid.** A sampled default would move every gate that compares
tokens or audio against a reference, and none of those movements would mean anything. Verify it rather
than assert it, the way P4.22 did: the ids for every existing fixture must be bit-identical before and
after.

#### Acceptance

* `loom.sample_row` beside `argmax_row`, reducing on the tensor, drawing from `rng_`.
* **Invariants that are cheap and strong**: `temperature -> 0` equals `argmax_row`; `top_k = 1` equals
  `argmax_row`; the same seed gives the same ids twice; two different seeds differ. A sampler that
  passes those three is doing the right arithmetic even before any distribution is checked.
* Gemma 3, correctly templated (P4.23) **and** sampled at the checkpoint's own `top_k`/`top_p`, stops
  looping and produces an answer comparable to `transformers` under the same seed and settings — the
  reference this item is defined against.
* Every existing gate baseline **bit-identical** under the greedy default, verified rather than assumed.
* Older GGUFs, whose drivers call `argmax_row`, keep working untouched — the `v4` fixture set is the
  test that says so (Epic-06 §2, Compatibility).

#### What NOT to do

* **Do not marshal logits into Lua and sample there.** It reinstates the prefill ceiling Retro-004 is
  about, at the exact vocabulary size that found it.
* **Do not make sampling the default.** See the byte-identity note above.
* **Do not implement beam search, repetition penalty or Mirostat.** Nothing in the set asks for them,
  and each is a decode-loop change rather than a reduction change — a different item if it ever lands.
* **Do not put the sampler in `text_generate.cpp`.** The decode loop for a KV-cached driver lives in
  the driver's own Lua (`infer_with_past`); a sampler in the C++ loop would only reach the one-token
  driver shape and would silently do nothing for every model that matters here.
