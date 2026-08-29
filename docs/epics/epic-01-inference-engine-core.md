---
type: epic
status: active
domain: inference-engine
last_updated: 2026-08-22
---

# Epic-01: The Data-Driven ggml Inference Engine Core

## 1. Context and Scope

`loom.cpp` is the runtime: the `ggml` engine, its primitive registry, its Lua bridge, its caches and
its tokenizers. It targets edge devices, and the property that decides most of its design is that
**it hardcodes no model**.

In scope: graph construction and reuse, the primitive registry, memory/allocator behaviour, KV and
conv-state caches, per-*task* C++ (tokenizers, CTC decode, generation), the Lua bridge, and the pinned
`ggml` dependency.

Out of scope: anything per-*model*. That belongs to [Epic-02](epic-02-mil-exporter-and-compiler.md) —
see [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md).

## 2. Architectural Overview

A model is one GGUF carrying its own **graph topologies** as JSON metadata and its own **driver
script** as embedded Lua, alongside the weights those describe. The engine parses them and builds the
`ggml` graph at run time.

* **Two contexts.** A persistent *model context* holds the weights, memory-mapped from the GGUF. An
  ephemeral *compute context* holds the DAG and its activations.
* **Symbol table.** `name → ggml_tensor*`, seeded with the weights and extended as nodes compute, so
  downstream nodes reference producers by name.
* **Primitive registry.** JSON op string → `ggml` call. A primitive may query the backend and emit
  either a native op or an exactly-equivalent composition
  ([ADR-007](../adrs/adr-007-backend-capability-negotiation.md)).
* **Dynamic axes, not dynamic shapes.** `ggml` has no dynamic dimensions, so a new sequence length is a
  new graph. `GraphBuilder` **retains the last graph** and returns it unchanged when the axes repeat —
  so an ODE solver's steps, an LSTM's timesteps or a decode loop build once, not once per iteration.
* **Allocation** uses `ggml_gallocr`'s two-pass strategy rather than hand-estimated byte sizes, and the
  compute buffer is released when a build stops needing it.
* **Caches.** `KvCache` and `ConvStateCache` are attached per session;
  [ADR-016](../adrs/adr-016-kv-cache-shape.md) covers the KV cache's shape and its limits.
* **Orchestration is Lua** ([ADR-002](../adrs/adr-002-embedded-lua-drivers.md)); every model reaches
  inference through `infer`.

**A gotcha worth knowing before changing anything here:** graph reuse and length-dependent constants
interact. A constant folded into a retained graph at one length is wrong at another, and the failure is
silent. Anything derived from an axis must be an input, not a baked constant.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Full design | [`docs/SPECIFICATION.md`](../SPECIFICATION.md) |
| KV cache | [`docs/KV-CACHE.md`](../KV-CACHE.md), [ADR-016](../adrs/adr-016-kv-cache-shape.md) |
| Foundational | [ADR-001](../adrs/adr-001-data-driven-gguf-topologies.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md), [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| Verification | [ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md) |
| Retros | [Retro-001](../retros/retro-001-layout-healing-heuristics.md), [Retro-003](../retros/retro-003-tdt-decoder-recomputed-its-prediction-network.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| Active tasks | [correctness](../backlog/active-index.md#engine--correctness) · [performance](../backlog/active-index.md#engine--performance) |

## 4. Standing Scope Limitations

These are deliberate boundaries, not defects. Each names what would have to change.


- **`KvCache` is single-sequence.** No ring buffer, no multi-stream/multi-sequence support. The
  `ggml_set_rows` index-tensor indirection this entry used to list alongside them **exists since
  P4.0.15** — writes are addressed by a cell-index tensor, and `KvCache::fill_cell_index` is the single
  place a second addressing policy would go. What is still missing is a *policy* that uses it: only the
  contiguous append `[n_past, n_past + n_tokens)` is ever written, reads are still a plain view over
  `[0, n_kv)`, and there is one `kv_size` for every layer.
- **Decoding is greedy, always.** `loom.argmax_row` is the only decode rule the bridge offers; there is
  no temperature, top-k, top-p or multinomial draw in the engine. The RNG to build one on already
  exists (`rng_`, `loom.seed_rng`, shared with `gaussian_array`/`uniform_array`), and the reduction
  must stay on the tensor for the reason `argmax_row` itself exists. Scoped as **P4.24**,
  [Epic-06 §4](epic-06-high-level-api-and-hosts.md).
- **KV cache storage is always F32.** No quantized cache types (`Q8_0` etc.). Weight quantization is
  handled per-model by the MIL exporter's `quantize=` kwarg (LFM2, Qwen3) — KV-cache quantization is a
  separate, still-untouched runtime concern (different mechanism, different point in the inference
  pipeline; check how the KV cache is currently allocated/typed before assuming it's a trivial extension).
- **Sampling is greedy argmax only** (`Generator::argmax` in `src/core/generation.cpp`). No temperature,
  top-k, top-p, or repetition penalty.
- **Only one level of `repeat_for` nesting** is supported in the JSON graph schema
  (`include/loom/core/graph_topology.h`'s `RepeatBlock::nodes` is a flat `vector<TopologyNode>`, not
  recursive).
- **`GgufModel::hparam_env()` only surfaces numeric scalar KV types** into the `SymbolEnv`; string, bool,
  and array-typed `loom.*` KVs are silently skipped.
- **No chunked/windowed inference for long Conformer-CTC audio** — cost grows O(n²) with length (relative
  position attention over the whole clip at once), per an explicit prior choice to defer this. Would need
  window size/overlap selection and stitching per-window CTC token sequences at the boundaries.
- **Only the small (16-layer, `d_model=176`) Conformer-CTC checkpoint has been verified.** Larger variants
  should work unmodified (topology is generated entirely from `model_config.yaml`) but this is unexercised.
- **Remaining BPE pretokenizer families** beyond the ~40 already in `bpe_vocab.cpp`'s `pre_spec_table()`
  (CJK-script splitters, case-transition/camelCase shapes, `byte_encode=false` SPM-style-BPE families like
  gemma4/sarvam-moe) raise a named error rather than being supported — bounded, add one when a real model
  needs it, per `pre_spec_table()`'s own comment.
- **General multi-scheme quantization tool.** `tools/quantize/quantize_gguf_q8_0.py` is Q8_0-only,
  single-model-shaped. A model-agnostic tool covering more of ggml's quant families, plus a
  per-tensor-role policy (skip norm weights/embeddings) rather than blanket quantization, is unbuilt.
- **Attention-variant primitives beyond `ATTENTION`/`REL_POS_ATTENTION`/`REL_POS_ATTENTION_SHAW`** (e.g.
  a dedicated flash-attention op) — add only when a real model needs one, not speculatively.


## 5. The ggml Pin

The dependency is pinned in `cmake/GgmlPin.cmake` and carries loom's own patches
([ADR-014](../adrs/adr-014-patch-ggml-rather-than-write-kernels.md)). A bump is real work — v0.16.0 →
v0.19.0 was 154 upstream commits — and is filed as its own item so a behavioural change never rides
along with a correctness fix.


`cmake/GgmlPin.cmake` held `v0.16.0` (`524f974b`, 2026-07-10). Upstream master `8846b79e` (2026-08-12)
was **154 commits ahead**. Filed as its own item deliberately: it was discovered while asking whether
upstream had fixed the NPU device-type question (it had not — P4.8e), and **nothing in the gap changes
that design**, so tangling a pin bump into that work would have mixed a behavioural change into a
correctness fix.

**The target is a TAG, not master.** Three tags had shipped since the pin — v0.17.0, v0.18.0/1, v0.19.0
— and `v0.19.0` (`30bf8685`, tagged 2026-08-07) carries **153 of the 154 commits**, master being one
commit ahead of it. So the whole gap is available without pinning a moving branch, which the file's own
convention (a tag plus its commit, so a backend package cannot drift from the base) requires anyway.

What is known about the gap, from reading rather than building:

* **One new backend directory, `ggml-et`** — an accelerator platform requiring a proprietary SDK at
  `/opt/et` (`aifoundry-utils`), with a `GGML_ET_SYSEMU` mode that runs against an emulator instead of
  hardware. It returns `GGML_BACKEND_DEVICE_TYPE_GPU`, which is the third data point in P4.8e's
  argument. Unbuildable here; listed so the next backend survey does not rediscover it.
* **The device-type enum is byte-identical**, comments included, so `primary_rank` needs nothing.
* Not surveyed: op coverage changes, which is the part that could matter to P4.7d's support matrix and
  to the two remaining splits in the zoo (`ATAN`, Whisper's 400-wide reflect pad).

### The result: nothing moved, to every digit

Baselines were captured on the old pin first, because the thing to detect here does not fail a test —
a backend kernel that stops claiming an op raises the split count and only shows up as lost speed.

| | `ci` | `gate` | conformer | causal_lm_kv |
|---|---|---|---|---|
| CPU (dev box) | 58/58 | 82/82, 8 ran | — | — |
| Vulkan (dev box) | 58/58 | 82/82 | 2104 / 0, **1 split**, rel 3.520e-03, rms 4.292e-02 | **1 split**, 2034 / 0 |
| CUDA (workstation) | 58/58 | 82/82 | 2104 / 0, **1 split**, rel 3.492e-03, rms 4.188e-02 | **1 split**, 2034 / 0 |

Every one of those equals its pre-bump baseline exactly — same max delta, same element index `[9147]`,
same rms, same node and split counts, on both backends. That is stronger than "no regression": the
kernels compute **bit-identically** across 153 commits, so nothing changed a reduction order or a
kernel selection underneath us. The build itself was the other half of the evidence and passed
silently: no API breakage in the engine, no new warning from any file this repo owns.

**What this does NOT cover, since "82 tests" overstates the reach.** Only **8** gate tests actually
run: `v4` holds GGUFs, and most gate tests want PyTorch reference directories that are not in it. So
the zoo-wide split comparison this item asked for is really two models on two backends. The breadth
claim rests on the build and the ci suite; the numerical claim is narrow and should be quoted as such.

**`ggml-et`, the one new backend directory, was not built** — it needs a proprietary SDK at `/opt/et`.
It is read about in P4.8e, where its unconditional `GGML_BACKEND_DEVICE_TYPE_GPU` is the third data
point in that argument.

### A build trap that cost a wasted cycle, and it is about this LAN rather than about ggml

The first CUDA rebuild died with `ninja: error: manifest 'build.ninja' still dirty after 100 tries,
perhaps system time is not set`. The cause is that **the workstation's clock is ~290 s behind the dev
box**, and `rsync -a` preserves mtimes — so every synced file lands with an mtime in the workstation's
*future*. Editing `GgmlPin.cmake` made CMake re-run, and ninja then regenerated `build.ninja` in a loop
because its input was permanently newer than its output.

Same root cause as the stale-`.o` trap recorded in P4.8e, from the opposite direction. **The rule for
this LAN: `touch` the tree on the workstation after every rsync**, before building.

A second, unrelated self-inflicted wound worth naming because it hid the first: `pgrep -f "cmake
--build build-cuda"` run over ssh **matches its own shell**, whose command line contains the pattern.
It reported "still building" for a build that had already failed. Bracket the pattern (`[c]make`), or
watch the log for a terminal marker rather than watching for a process.

