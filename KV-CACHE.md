# KV cache on the MIL path — what is missing and in what order

This document is the working record of a review session (2026-08-02) that started from one question:
*how should attention layers manage a KV cache — wrap them the way `recurrent.py` wraps LSTM/GRU (an
SDPA kernel in C++ with projections, masking, residual and cache management orchestrated from Lua), or
make a composite ggml op that owns the cache itself?*

The answer measured from the tree is that **the second design is already built and shipping**, and that
neither option is where the gap is. What is missing is three concrete things, none of them a kernel.

Relationship to the other documents:

* [`EXPORT-PREPARATION.md`](EXPORT-PREPARATION.md) — §4's second bullet is this thread's origin. It
  filed the work as "a real gap and a real capability item — it belongs in P4/P5, not in preparation",
  and its measured note (2026-08-01) correctly identified `FuseLoomAttention` as the blocker. This
  document supersedes that bullet's *four-step* decomposition (see §2: one of the four is not needed)
  and schedules the work **before stage D**, at the author's direction.
* [`BACKLOG.md`](BACKLOG.md) — P4.0.9 is this document's backlog row.

Everything below that cites a file or a count was checked against the tree during the session.

---

## 1. Findings

### 1.1 The composite-op design is not a choice to make; it is the status quo

`src/ops/primitives_attention.cpp:29-101` is a composite SDPA — the torch reference's
`softmax(q @ k^T * scale + mask) @ v`, line for line — with the cache seam already in it:

* the cache is **pre-allocated persistent storage outside the compute graph**: `KvCache` owns its own
  `ggml_context` and backend buffer (`include/loom/core/kv_cache.h:50-52`), which is the "Model Context"
  half of the two-context paradigm. It is deliberately *not* a graph op, so `ggml-alloc` never sees it.
* `op_attention` appends this step's K/V at cells `[n_past, n_past + n_tokens)` and reads back
  `[0, n_kv)` (`:65-76`), routing the writes through `side_effects` because they have no data-dependency
  edge to the read.
* `n_kv` is derived automatically from `n_tokens + n_past` (`graph_builder.cpp:129`).
* the cache binds per registered module and survives across calls (`lua_bridge.cpp:145`, `:520`), so a
  Lua driver gets persistence with **no address ever crossing the scripting boundary**.
* `attrs.kv_cache=false` covers the non-autoregressive case, which is how the same op serves Whisper's
  cross-attention (`convert_whisper/convert_whisper_decoder.py:90`).

Two consequences worth stating plainly, because both were live questions:

* **Wrapping attention the way `recurrent.py` wraps LSTM would be a regression.** The Lua boundary is a
  *per-step* boundary, not a *per-layer* one. Driving projections/mask/residual from Lua means crossing
  it once per layer per token, marshalling every intermediate through Lua doubles and building one graph
  per layer. Note that even `run_recurrent` (`lua_bridge.cpp:191-273`) is a **C++ loop**, not a Lua one:
  the per-timestep round trip was already too expensive to expose.
* **LSTM should not become a composite ggml op for symmetry.** `recurrent.py`'s docstring has the
  reason: ggml has no LSTM kernel, and the trip count is a data-dependent sequence length. Attention
  gets to be one node because `n_kv` is a known symbol at graph-build time.

### 1.2 The whole orchestration already runs from Lua, for a real model

`tools/convert_whisper/whisper_driver.lua` prefills at `{n_tokens = n_prompt, n_past = 0}` and then
decodes at `{n_tokens = 1, n_past = t}`, in ~55 lines, with no C++ driver logic. Cross-attention AR
decode with a KV cache is a **solved orchestration shape in this engine**. `EXPORT-PREPARATION.md`
§1.4's table lists it as "not started", which is true only of the *MIL* path.

### 1.3 A MIL-exported causal LM has zero `ATTENTION` nodes

Measured on `qwen3_0.6b_mil_monolithic.gguf` (2318 nodes):

```
ADD 450, MUL 450, MUL_MAT 254, RESHAPE 228, PERMUTE 142, VIEW 140,
POW/REDUCE_SUM/SCALE/RSQRT 113 each, CONCAT 57, REPEAT 56, CONT 29,
SOFTMAX 28, SILU 28, GET_ROWS 1, CAST 1, COS 1, SIN 1
```

28 `SOFTMAX` = 28 layers of attention arriving fully expanded. `exporter.py:125` maps
`loom_fused_attention → ATTENTION` and `dialect.py:258` registers a `FuseLoomAttention` pass — but its
`_fuse_blocks` body is `pass`, a documented placeholder (`dialect.py:268-270`). **The op is registered
and never produced.**

The pattern it must match is crisp and stable. Around the first `SOFTMAX`, in the exported (ne-order)
topology:

```
MUL      [query_1, _21]            -> mul_0        # scale folded onto q, not onto the scores
MUL_MAT  [key_1, mul_0]            -> matmul_0     # q @ k^T
ADD      [matmul_0, attention_mask_3] -> add_0
SOFTMAX  [add_0]                   -> softmax_0
PERMUTE/CONT [value_1]             -> ..._mm_y_cont
MUL_MAT  [..._mm_y_cont, softmax_0] -> attn_output_1
PERMUTE  [attn_output_1]           -> _264         # back to [head_dim, n_head, n_tokens]
RESHAPE  [_264]                    -> _267         # flatten to [n_embd, n_tokens]
MUL_MAT  [o_proj.weight, _267]     -> ...
```

`key_1`/`value_1` are the *post*-GQA-repeat vars: `fuse_gqa_repeat_kv` (`passes.py:48`) has already
normalized HF's `repeat_kv()` into a `reshape → tile → reshape` triple, so the pre-repeat K/V (8 heads,
not 16) is reachable by walking back through it. Qwen3's qk-norm sits upstream of the window and is
invisible to the match.

### 1.4 The cache geometry is declared nowhere

`tests/test_e2e_whisper_lua_driver.cpp:141` constructs
`KvCache(cfg.n_text_layer, cfg.n_text_state, cfg.n_text_state, cfg.n_text_ctx, backend)` from a
hardcoded C++ `WhisperConfig`. So Whisper's "self-contained one-GGUF model" **is not**: the host still
needs a per-model C++ struct to size the cache. Every other host would need one too.

This is the piece of the "pre-allocate the cache with a special ggml storage op" intuition that is
genuinely missing — but it belongs as *declared metadata plus host allocation*, not as a graph op.

---

## 2. What `EXPORT-PREPARATION.md` §4 got wrong, and it makes the work smaller

That bullet's step 2 was "**trace with past-KV semantics**, so a decode step computes K/V for the new
token only. This is precisely `use_past`/`decoder_with_past`." **That step is not needed.**

Once the SDPA subgraph is replaced by an `ATTENTION` node with `kv_cache=true`, the engine supplies the
past itself: the traced graph computes K/V for whatever tokens it is handed, and `op_attention` appends
them and attends over `[0, n_kv)`. A decode step is then just a call at `n_tokens = 1` — no second trace,
no `decoder_with_past` graph, no merged decoder.

What *is* real from step 3 is narrower than "change the exported input contract": exactly one input
needs to change. `attention_mask` traces as `["n_tokens", "n_tokens", "1", "1"]` because
`causal_lm_export` deliberately shares one `ct.RangeDim` across `tokens`/`cache_position`/
`attention_mask`, and it must become `["n_kv", "n_tokens"]` — the shape the bespoke converter already
declares (`convert_qwen3/convert_qwen3.py:64`).

**Two independent `RangeDim`s cannot be traced.** With no cache, HF computes scores of shape
`[1, h, s, s]`, and adding a `[1, 1, s, kv]` mask to that fails type inference at conversion time. The
fix is to **retype that input at export time**, which is legitimate for one specific reason: after
fusion the mask is consumed *only* by the fused node, so no other node's shape derives from it. That is
a checkable property, not an assumption — see step 3.2's link.

So §4's four pieces become: fusion (expensive), metadata (small), mask retyping + the loop (moderate).

---

## 3. Decisions

1. **`infer` is the entry point for every model driver.** `main` (6 synthesized models), `synthesize`
   (5 TTS families, bespoke *and* MIL) and `transcribe` (Whisper) all become `infer`. One generic name,
   applied to every driver rather than to the new one only.
2. **`main_topo` becomes `main_topology`.** Including the GGUF key
   `model.graph_topology.main_topo` → `.main_topology`.
3. **`infer_with_past` owns the loop.** Prefill, then decode until `max_new_tokens` or `eos_token`,
   returning a token array. The alternative — a single cached step at a caller-supplied `n_past` — puts
   the loop back in the host, which is the per-model C++ driver shape the architecture is retiring.
   `infer` keeps its current behaviour exactly (one prefill, argmax the last row, one token).
4. **The fusion pass is opt-in, off by default.** Anchored on `softmax`, the pattern would also match
   VITS/Kokoro/StyleTTS2 self-attention — which is non-autoregressive and must never acquire a cache —
   and firing there would change five TTS models that have nothing to do with this thread. Only the
   causal-LM family sets the flag.
5. **Geometry is declared; *which* topologies need a cache is derived.** `n_embd_k`/`kv_size` are model
   facts a GGUF must carry. "Does this topology use the cache" is exactly "does it contain an
   `ATTENTION` node with `kv_cache` true", which the parsed topology answers precisely — declaring it
   separately would create a second authority that can disagree with the graph.

---

## 4. Implementation plan

Four stages. Each numbered step is one commit; "touches" names the models a step can affect, which is
its per-commit gate. The three standing practicalities from `EXPORT-PREPARATION.md` §6 apply unchanged
(`TMPDIR=` *and* `--basetemp=` under `/home/flavio/.claude/tmp/`; baseline snapshots from a `git
worktree`, not from the tree's stale `.gguf` files; `cd` into the tree being measured).

The **negative gate** applies here more than anywhere: three of these steps add checks, and a check that
never runs is indistinguishable from a passing one on the output side. Where a step claims a check now
runs, break the thing being checked and record the message in the commit.

### Stage N — naming (before any KV work)

*First because `infer_with_past` only makes sense once `infer` exists, and because doing it later would
mix a mechanical rename into commits whose diffs need to be about numerics.*

**N.1 — `main`/`synthesize`/`transcribe` → `infer`.** `DriverBuilder.entry_name`,
`MultiPhaseDriverBuilder.entry_name`, `RawLuaDriver.entry`, the five hand-written `.lua` drivers,
Whisper's, and the 14 `bridge.call` sites in `tests/`. *Touches: all 11 + Whisper. Gate: a
`compare_snapshots.py` diff showing the entry line and nothing else, per model.*

**N.2 — `main_topo` → `main_topology`.** 38 occurrences in 12 files, including the GGUF topology key.
*Touches: Qwen3, LFM2 ×2, Conformer-CTC, Parakeet ×2. Gate: same, one key rename and nothing else.*

### Stage 1 — declare the cache in the GGUF

**1.1 — the geometry hparams.** *Measured while doing it, and it makes the step smaller than written:*
`convert_whisper_all.py:81-85` **already writes four of the five** — `loom.n_layer`, `loom.n_head_kv`,
`loom.n_embd_head_k`, `loom.n_embd_head_v` — under a comment saying they are for "KvCache sizing", and
**nothing has ever read them**. Only the capacity was missing. So this is one added KV
(`loom.kv_cache_size`), not a new namespace: a parallel `loom.kv_cache.n_layer` would be a second
spelling of a fact already declared, and two spellings that can disagree is the failure this project
keeps removing elsewhere. *Touches: Whisper (bespoke converter only).*

**1.2 — engine-side allocation from the model.** `GraphTopology::uses_kv_cache()` (does any node have an
`ATTENTION` with `kv_cache` true — decision 5, and note the attr **defaults to true**, so reading it as
absent-means-false would report exactly the models that need a cache as not needing one) plus
`make_kv_cache(model, backend)`, which composes the flat per-token width as `n_head_kv * n_embd_head_k`
— never `n_head`, since the cache stores the un-repeated K/V. A model whose topology reports
`uses_kv_cache()` but whose file omits a geometry key raises naming the key *and* saying the fix is on
the converter side.

**1.3 — delete `WhisperConfig`'s cache fields from the Lua test.** This is the step's real acceptance
test and the reason stage 1 exists: `test_e2e_whisper_lua_driver.cpp` allocates from the GGUF alone, and
the transcription stays token-identical to the C++ oracle. *Negative gate: remove one KV and confirm the
load fails naming it.*

### Stage 2 — `FuseLoomAttention`

**2.1 — the pass, opt-in, GQA repeat left in place.** Match the §1.3 window anchored on `softmax`; emit
`loom_fused_attention` with `layer` (from `TORCHSCRIPT_MODULE_NAME` scope, falling back to occurrence
order) and `scale` (the constant folded onto q, else 1.0). Absorb the trailing `permute`+`reshape` so
the op's declared output is `[batch, seq, n_head*head_dim]` — matching what `op_attention` actually
returns, rather than q's shape.

**2.2 — the topology rule.** Lower `loom_fused_attention` to an `ATTENTION` node, transposing q/k/v into
the engine's `[head_dim, n_head, n_tokens]` layout (`convert_qwen3.py:76-78` is the reference).

**2.3 — strip the GQA repeat.** Walk back through `fuse_gqa_repeat_kv`'s `reshape → tile → reshape`
triple to reach the true `n_head_kv` K/V, halving the cache for Qwen3 and letting `op_attention`'s own
`ggml_mul_mat` broadcast do the GQA. Correctness does not depend on this — attending with the repeat
already present is numerically identical, just twice the cache — so it is its own commit with its own
numeric check.

*Touches: Qwen3, LFM2 ×2 and the generic HF fallback. **Byte-identity is deliberately abandoned here**
— the topology changes by construction. The gate is each model's numeric reference test, plus a node
histogram in the commit message showing 28 `SOFTMAX` become 28 `ATTENTION`. Every other model must be
byte-identical, which is what decision 4 buys.*

### Stage 3 — `infer_with_past`

**3.1 — `N_KV` in the axis vocabulary** (`axes.py`), and `_validate_input_axes` taught that a second
dynamic axis named `n_kv` is legal — it is precisely the case P4.0.2's check was written to catch.

**3.2 — retype the fused mask input.** Declare it `["n_kv", "n_tokens"]`, with a link asserting the
property that makes it sound: the mask var's only consumers are fused-attention nodes. Break it (feed
the mask somewhere else) and confirm the export fails.

**3.3 — the `PrefillDecodeLoop` component and the second entry.** `DriverScript` grows a second entry
function; `PrefillArgmaxBuilder` gains a sibling that emits prefill → `While` → argmax → array return.
`driver_ir` already has `While`/`Break`, and `transpile_operation` already emits
`loom.causal_mask(n_tokens, n_past)` over real variables.

**3.4 — the e2e test.** Generate N tokens from a real checkpoint through `infer_with_past` and compare
against the same N tokens produced by calling `infer` N times with a growing prompt. That comparison is
the whole claim of this document — the cache is only correct if the two agree.

### Stage gate

Full 11-model sweep. `infer` byte-identical for the seven non-causal-LM models, numerically identical
for the four causal ones; `infer_with_past` agreeing with iterated `infer`; and the negative-gate probes
from 1.3, 2.1 and 3.2 recorded with their messages.
