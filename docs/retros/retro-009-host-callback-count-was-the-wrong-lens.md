---
type: retro
date: 2026-08-13
domain: backends
tags: [gpu, graph-splits, profiling, wrong-metric, exporter-passes]
---

# Retro-009: Counting Host Callbacks Was the Wrong Lens

## The Issue

P4.7 put the engine on a GPU and measured Qwen3-0.6B at **0.95x** — *slower than the CPU*. Matcha came
in at 0.84x. The named cause was 226 `ggml_map_custom` nodes: host callbacks that no backend but the
CPU can dispatch, each one cutting the graph. Removing them worked, twice. Then the metric ran out.

## Root Cause Analysis

Three findings, in the order they arrived:

1. **The custom ops were an exporter blind spot, not an engine limit.** All 226 were one thing — an RMS
   norm the exporter had never been taught to recognise, emitted as a five-op chain and lowered to a
   host callback. A `fuse_rms_norm` pass collapsed it. `lower_pow` then took 149 `POW` nodes and
   Matcha's 32 `RSQRT` the same way.
2. **`ggml_map_custom` was never the real predicate.** `PAD_1D_REFLECT` is not a custom op — it is a
   *real ggml op that `ggml-vulkan` does not implement*, so every reflect pad was still a CPU node and
   nothing about the custom-op count said so.
3. **The gaps do not line up between backends.** CUDA has `PAD_REFLECT_1D` but no `POOL_1D`; Vulkan has
   `POOL_2D` but neither; OpenCL, OpenVINO and Hexagon have none of the three. So there is no static
   answer at all — which is what made this an engine decision rather than an export one.

## Resolution & Lesson Learned

Exporter passes removed the recognisable patterns; the engine gained a primitive that **asks the
backend what it can run** and emits either the native op or an exactly-equivalent composition
([ADR-007](../adrs/adr-007-backend-capability-negotiation.md)). The one case with no exact composition
anywhere got the project's first accepted approximation
([ADR-008](../adrs/adr-008-atan-approximation.md)).

* **Actionable takeaway 1 — measure the thing you care about, not its proxy.** The cost was *graph
  splits*. `ggml_map_custom` count correlated with splits until it didn't, and the residue was
  invisible for as long as the proxy was trusted.
* **Actionable takeaway 2 — "which ops are missing" has no portable answer.** Deciding it at export
  time compiles every artifact for the least capable backend. One GGUF may be run by any backend, so
  the decision belongs where the backend is known: run time.
* **Actionable takeaway 3 — the CPU baseline was measured wrong twice** before anything interleaved the
  runs. Interleave A/B arms; do not trust two separately-timed runs on a thermally noisy box.

---

## Full record (verbatim from the ledger)

### P4.7a — RMS norm reaches its primitive, and the GPU story changes


P4.7 measured Qwen3-0.6B at **0.95x** on a GPU — slower than the CPU — and named the cause: 226
`ggml_map_custom` nodes, host callbacks no backend but the CPU can dispatch, each one cutting the graph.
All 226 were one thing, an RMS norm the exporter never recognised. It is recognised now.

`loom_exporter/passes.py`'s **`fuse_rms_norm`** collapses the five-op chain PyTorch's RMSNorm traces to —
`pow(x,2) → reduce_mean(-1) → add(eps) → rsqrt → mul(x, ·)` — into one `loom_rms_norm`, which
`topology_ops.py` lowers to the engine's `RMS_NORM`. The primitive was already there and had never once
been emitted.

### What it did

| | splits | device / CPU nodes | CPU | GPU | |
|---|---|---|---|---|---|
| qwen3-0.6b, decomposed | 453 | 2879 / 339 | 794.5 ms | 797.5 ms | 1.00x |
| qwen3-0.6b, **fused** | **1** | **2201 / 0** | 755.9 ms | **275.5 ms** | **2.74x** |
| lfm2-350m, decomposed | 181 | 1145 / 135 | 453.6 ms | 257.2 ms | 1.76x |
| lfm2-350m, **fused** | **1** | **875 / 0** | 419.9 ms | **189.1 ms** | **2.22x** |

**One split means the whole graph ran on the device** — not one node fell back. Qwen3's topology drops
from 2066 nodes to 1501, LFM2's from 820 to 595, and both go from "the GPU is not worth using" to the
range the Conformer was already in.

**The CPU path is unchanged, and that is the point of checking it.** 794.5 → 755.9 ms and 453.6 → 419.9
ms are within this machine's run-to-run variance; a fused `ggml_rms_norm` and five vectorized ops cost
about the same on a CPU. Nobody pays for this.

### Correctness

* **Numerically**: max relative difference between the two exports' logits is **4.1e-07** on the CPU
  (fp32 rounding — `ggml_rms_norm` accumulates its sum in double where the decomposed chain's
  `REDUCE_SUM` does not), with **0/8 argmax disagreements**. On the GPU, 3.1e-04 with the same 0/8.
* **A model that must NOT change does not.** VITS has no RMS norm, and re-exporting it with the pass in
  place produced a **byte-identical** GGUF to the one before. That is the check the sweep recipe demands
  — a fusion that cannot be shown to leave the rest alone is a fusion nobody can trust.
* The gate suite passes on both a CPU-only and a Vulkan build (82/82, 8–10 tests doing real work),
  including `test_e2e_causal_lm_infer_with_past` on the re-exported model and both device-parity tests.
  The KV-cache parity test now reports `splits=1, device=2034, cpu=0` — a whole cached decode on the
  device, same twelve tokens as the CPU.
* Eight unit tests in `tests/ci/test_passes.py`, and seven of them are **negative**: a multiply that
  feeds back a different tensor, a mean over another axis, an exponent that is not 2, a shared
  intermediate, keep_dims=False. That is where a fusion pass does its damage — emitting `RMS_NORM` for a
  chain that normalizes something else is silently wrong arithmetic no shape check downstream catches.

### Two things worth keeping

**MIL's `rsqrt` carries an epsilon of its own** (default 1e-12) and computes `1/sqrt(x + epsilon)`, so
the value the engine must add to the mean square is the SUM of that and the traced `variance + self.eps`
— measured, not assumed: the fused op comes out at `1.000001e-06` for a model written with `eps=1e-6`.
Dropping the term would be a wrong answer nothing downstream would flag, which is why it has its own
test.

**Matcha was not fused, and should not have been.** It has 38 `POW` and 32 `RSQRT`, which looks like the
same pattern and is not: its chain starts with `SUB(x, mean)` and reduces ne axis 1. That is a
hand-rolled **LayerNorm** over a non-`ne[0]` axis, and `ggml_rms_norm` is neither mean-centred nor able
to reduce any axis but `ne[0]`. The pass's axis guard refuses it. Fusing that one is a separate item —
it would need `LAYER_NORM` plus a transpose, and `LAYER_NORM` has the same ne[0]-only restriction.

### The one it does not fix

Kokoro and StyleTTS2 carry 50 `POW` each with **no** `RSQRT` — a real `x**p`, not a normalization — so
they still split. `POW(x,2) → ggml_sqr` is the item for those, priced under P4.7 above.


### P4.7b — the last of the host callbacks: SQR and a hand-rolled LayerNorm

P4.7a took RMS norm out of the causal LMs and left two things behind: 149 `POW` nodes spread across
every family, and Matcha's 32 `RSQRT`. Both are gone now, and with them essentially every reason a
scheduler had to cut a graph.

**`lower_pow`** rewrites `pow(x, 2)` to MIL's own `square`, which `exporter.py` already mapped to the
engine's `SQR`. That it was worth a whole pass is a measurement, not a guess: **every `pow` in every
model is a square** — 149 of them, exponent 2.0 in every single one (Kokoro 50, StyleTTS2 50, Matcha 38,
the NeMo encoders 3 each, GigaAM and Whisper 1 each). There was never a general `pow` to preserve, only
a squaring op nobody had recognised. `pow(x, 0.5)` is deliberately NOT handled: no traced model has
produced one, and this repo adds a path when a model needs it.

**`fuse_layer_norm`** recognises the four-op statistic a model writes when it normalizes a channel axis
itself instead of calling `torch.nn.LayerNorm`, and emits MIL's `layer_norm` — transposed into ne[0] and
back when the axis is not already trailing, since `ggml_norm` normalizes ne[0] and nothing else. It is a
separate pass from `fuse_rms_norm` and deliberately so: the two differ by exactly the mean-centring, and
a matcher that treated `sub(x, mean)` as optional would emit `RMS_NORM` for a layer norm the moment a
`sub` failed to match for an unrelated reason.

### What is left, across all thirteen fixture models

| | before P4.7 | now |
|---|---|---|
| `POW` | 149 | **0** |
| `RSQRT` | 258 | **0** |
| `ATAN` / `ATAN2` / `SHAPE` | 2 | **2** |

**Two `ggml_map_custom` nodes remain in the entire model zoo** — one `ATAN` each in Kokoro's and
StyleTTS2's STFT phase computation, which has no ggml counterpart and no composition that avoids the
transcendental. Everything else the engine ever pushed onto a host callback is now a real ggml op.

**That count was the wrong lens, which P4.7c below discovered.** A graph also splits on real ggml ops
whose BACKEND kernel is missing — `PAD_REFLECT_1D` and `POOL_1D` are both unimplemented in ggml-vulkan —
and no amount of counting `ggml_map_custom` reveals them. Kokoro's remaining seven splits were four
reflect pads and two `ATAN`, not two `ATAN`.

### What it did (AMD Vega 3 iGPU vs 4 CPU threads, best of three)

| module | splits before | splits now | GPU vs CPU |
|---|---|---|---|
| qwen3-0.6b `main_topology` | 453 | **1** | 2.82x |
| lfm2-350m `main_topology` | 181 | **1** | 2.22x |
| matcha `encoder_mu` | 61 | **1** | 3.65x (was **0.84x** — slower than the CPU) |
| kokoro `decoder_vocoder` | 107 | **7** | 4.45x |
| styletts2 `decoder_vocoder` | ~107 | **7** | 4.28x |
| conformer-ctc `main_topology` | 5 | **1** | 2.56x |

### The measurement that was wrong twice, and what it cost

The obvious objection to transposing an axis into ne[0] is that `ggml_norm` calls `ensure_packed`, so
each norm pays two full copies. A best-of-three on Matcha's `encoder_mu` said that cost **31% of CPU
throughput** — and on the strength of it the transpose was replaced with `div(centered, sqrt(var+eps))`,
which removes the same host callback while moving nothing. A second best-of-three then said the
transpose was the *fastest* of the three. Both were noise: this module swings **33–85 ms between runs of
the same binary** on this machine.

Six interleaved rounds of twenty runs each, taking the minimum per arm, settled it:

    unfused chain               cpu 50.5 ms   gpu 45.7 ms
    transpose + layer_norm      cpu 44.2 ms   gpu 12.1 ms      <- what shipped
    div(centered, sqrt(...))    cpu 54.5 ms   gpu 12.8 ms

**Transposing is the fastest of the three on the CPU as well**, because `ggml_norm` is one fused pass
where the chain it replaces is eight, and that buys more than two copies cost. The division form — which
avoids the copies entirely — is the slowest. The lesson is not about layer norms: a best-of-three on a
40 ms module on a thermally-throttled laptop is not a measurement, and it produced two contradictory
"findings" before anything interleaved them.

### Correctness

* **Byte-identity on both sides of the line, which is the check that makes the rest trustworthy.** The
  four models with nothing to fuse (VITS, Supertonic, LFM2 monolithic and modular) re-export
  **byte-identical**; the four with something to fuse (Kokoro, StyleTTS2, Matcha, Parakeet) all differ.
  Naming which must move before diffing is the discipline BACKLOG.md §6 records, and this is it applied.
* The gate suite passes on a CPU-only and a Vulkan build alike (82/82, 10 tests doing real work),
  including all three MIL Lua-driver end-to-end tests, whose models are exactly the ones that changed.
* 542 exporter CI tests, 15 of them new across the two passes, and most of the new ones negative: an
  exponent that is not 2, a non-constant exponent, two means over different axes, a missing centring
  (that graph is an RMS norm and belongs to the other pass), a shared intermediate.


### P4.7c — reflect padding, composed rather than fallen back on

P4.7b left two `ggml_map_custom` nodes in the zoo and I called that the end of the fallbacks. It was
not. **`PAD_1D_REFLECT` is not a custom op — it is a real ggml op that `ggml-vulkan` does not
implement**, so every reflect pad was a node the scheduler had to run on the CPU, and nothing about the
custom-op count said so. Looking only at `ggml_map_custom` was the wrong lens.

### How much it was worth, measured before deciding

Substituting device-supported stand-ins of identical shape into Kokoro's `decoder_vocoder`, so each
fallback's cost could be priced separately:

| | splits | CPU nodes |
|---|---|---|
| as exported | 7 | 4 |
| if the `ATAN` were device-native | 5 | 3 |
| if the reflect pads were device-native | **3** | 1 |
| if both were | 1 | 0 |

The two reflect pads cost **four** of the six removable splits — twice what the remaining `ATAN` costs.
And unlike the `ATAN`, this one has an **exact** fix, because reflect padding is not a transcendental:
it is a slice and a concatenation.

### The composition

`topology_ops.py`'s `pad` rule now emits, for `mode="reflect"`, one one-element `VIEW` per padded
element plus the `CONCAT`s that join them — the left block being elements `lp0..1` and the right block
`T-2` down to `T-1-rp0`, which is exactly torch's "reflect" convention (edge element excluded:
`[a,b,c,d]` with (1,1) → `[b,a,b,c,d,c]`). `VIEW` inherits the parent's strides, so a `[1, *ne_rest]`
view at byte offset `k*4` selects element `k` along ne[0] for every row and channel — correct for a
rank-2 `[T, C]` tensor and not only for the effectively-1-D waveform that motivated it.

**Bit-identical, and checked as such rather than argued.** Same module, two GGUFs, identical inputs:
**0 of 38400 outputs differ in any bit**, on the CPU and on the device, for Kokoro and StyleTTS2 alike.
No arithmetic happens in a slice, so there is nothing to round.

| module | splits | CPU nodes | GPU |
|---|---|---|---|
| kokoro `decoder_vocoder` | 7 → **3** | 4 → **1** | 1507 → **1274 ms** |
| styletts2 `decoder_vocoder` | 7 → **3** | 4 → **1** | 1452 → **1309 ms** |

The CPU path is unaffected (6709 → 6428 ms and 6578 → 6170 ms, i.e. no worse, and within this machine's
noise band — see P4.7b for how wide that is).

*(**Superseded by P4.7e**, which moved this into the engine. P4.7d's support matrix showed the
composition belongs there: CUDA, Metal, SYCL and CANN all implement `PAD_REFLECT_1D`, so only Vulkan
needs it, and an export cannot know which backend will run it. Everything below about WHAT the
composition is and why it is exact still stands — it is the same composition, in `op_pad_1d_reflect`
now. What changed is who decides to use it.)*

### The width guard, and Whisper

The composition costs `2 * (lp0 + rp0)` nodes, because `ggml_concat` is two-input. Above
`_REFLECT_PAD_COMPOSE_LIMIT = 32` the primitive is kept instead. That is not hypothetical tidiness:
**Whisper's STFT centre-framing pads 200 either side**, and composing it would emit **800 nodes** into a
503-node topology. It keeps `PAD_1D_REFLECT`, exactly as before, and re-exports bit-identical.

Whisper also shows why this item does not claim to have finished the job. Its encoder still reports 4
splits, and removing the reflect pad would not change that: it also uses **`POOL_1D`**, which
`ggml-vulkan` does not implement either. Which is the general shape of what is left — not custom ops,
but real ggml ops with missing backend kernels, and the answer for those is upstream (a shader) rather
than here (a composition). The tool that priced this item (substitute a stand-in, re-count splits) is
the one to reach for before writing any of them.

### What remains, per module

* kokoro / styletts2 `decoder_vocoder`: 3 splits, 1 CPU node — the `ATAN`. An exact mapping does not
  exist (ggml has no inverse trig at all, and `atan` has no closed form over the real ops available); a
  minimax rational on `[-1,1]` with `atan(x) = π/2 - atan(1/x)` range reduction would be ~15-20 native
  nodes at ~1e-7 relative error. That would be **the first approximation of a transcendental this
  project has accepted**, and it is worth 2 splits, so it is priced here and not taken.
* whisper `encoder`: 4 splits — `POOL_1D` and a 400-wide reflect pad. **`POOL_1D` is DONE as P4.7d
  below**, taking this to 2.


### P4.7d — POOL_1D, spelled as the POOL_2D that backends actually implement

The last op P4.7c left on the CPU. **`GGML_OP_POOL_2D` is the only pooling op every GPU backend
implements** — and a 1-D pool is a 2-D pool with a one-tall window, so the engine spells it that way
**where it has to**. (As shipped this entry substituted unconditionally; **P4.7e put it behind
`backend_can_run`** along with the reflect pad, so Metal and SYCL — which do implement `POOL_1D` — and a
CPU-only build all keep the native op. Everything below about the equivalence and its one exception is
unchanged; what moved is when the substitution happens.)

*(The first version of this entry said "ggml-vulkan implements POOL_2D but not POOL_1D", which was true
and undersold it: **CUDA has no `POOL_1D` either**. Only Metal and SYCL do. See the support matrix
below, which was checked after the fact and changed what this item is worth — it is not a workaround for
one backend, it is the spelling that works on all of them, and CUDA is what P4.8 does next.)* `ggml_pool_2d` sizes its output `[calc(ne0,k0,s0,p0), calc(ne1,k1,s1,p1), ne2, ne3]`
against `ggml_pool_1d`'s `[calc(ne0,k0,s0,p0), ne1, ne2, ne3]`, and `calc(ne1, 1, 1, 0) == ne1`.

**Engine-side, not exporter-side**, unlike P4.7a–c. Nothing per-MODEL is involved: it is one
primitive's lowering, and a topology that says `POOL_1D` should keep meaning what it says.

### The thing that was nearly shipped wrong

The two spellings are **not** interchangeable, in one combination, and it is invisible in every
interior window: **an average pool with padding divides by different numbers.** `ggml_pool_1d` divides
by `count`, the in-bounds elements the window actually covered; `ggml_pool_2d` divides by `ka = k0*k1`,
the whole kernel, treating padded cells as zeros it still counts. The shapes agree, so nothing structural
catches it — only the values at the windows that overhang an edge differ, 8 of 256 in the case that found
it.

That case was found by running the comparison **before** shipping the lowering, not after. The predicate
is now `op == MAX || p0 == 0` (a max is indifferent to padding; with no padding no window overhangs, so
`count == k0 == ka`), and everything else keeps `ggml_pool_1d` — a correct CPU fallback beats a fast
wrong answer.

`tests/ci/test_pool_1d_lowering.cpp` pins both halves: seven parameter combinations that must be
bit-identical, and the padded average that must NOT be, asserted against the exposed predicate. A test
that only covered the equivalent cases would still pass with the guard deleted, which is why the
negative case is there.

### What it bought, and what it did not

Whisper's encoder, the only user in the zoo:

| | splits | CPU nodes | GPU |
|---|---|---|---|
| before | 4 | 4 | 2967 ms |
| after | **2** | **3** | 3014 ms |

**The splits halved and the wall clock did not move** — 2967 vs 3014 ms is inside this machine's noise.
That is not a disappointment to explain away, it is the measurement: Whisper's `POOL_1D` is a *global
max*, `k0 = s0 = 240000` over a flattened tensor, so its output is **one scalar** and the round trip the
split was costing was one float. A split is only worth what crosses it.

The value is therefore structural rather than in this number: pooling as an op class now runs on a device
backend at all, for any model that pools something bigger than a scalar. Whisper happens to be the worst
possible advertisement for its own fix.

### What is left in the zoo

* whisper `encoder`: **2 splits** — the 400-wide reflect pad P4.7c's width guard correctly declines to
  compose (800 nodes into a 503-node topology).
* kokoro / styletts2 `decoder_vocoder`: **3 splits** — the `ATAN`, priced in P4.7c and not taken.
* everything else: **1 split**, nothing falling back.

Both remaining items are now upstream questions rather than exporter or engine ones: a `pad_reflect_1d`
shader and an `atan` op. Which is the natural boundary — the three passes and this lowering took every
case where loom could express the same thing in ops a backend already has, and stopped where that
stopped being true.

### The support matrix, and where this kind of work belongs

Checked against each backend's own `supports_op`, not by grepping for mentions (v0.16.0):

| op | CPU | CUDA | Metal | Vulkan | SYCL | CANN | OpenCL | OpenVINO | Hexagon |
|---|---|---|---|---|---|---|---|---|---|
| `PAD_REFLECT_1D` | yes | **yes** | **yes** | no | yes | yes | no | no | no |
| `POOL_1D` | yes | **no** | yes | no | yes | no | no | no | no |
| `POOL_2D` | yes | yes | yes | yes | yes | yes | no | no | no |
| `atan` | *ggml has no atan op at all — not in the unary enum, not in any backend* |

Three things follow, and the third is the one worth keeping.

1. **This item is worth more than its own first paragraph claimed** — CUDA lacks `POOL_1D` too, so the
   lowering is what makes pooling run on a device at all for the backend P4.8 goes to next.
2. **The `ATAN` gap is universal, not Vulkan's.** Those two splits would be splits on CUDA and Metal
   alike. That cuts both ways: an approximation would pay off everywhere rather than on one backend,
   which strengthens the case for it somewhat — it is still the first approximation of a transcendental
   this project would accept.
3. **P4.7c is in the wrong repo, and this matrix is what shows it.** `PAD_REFLECT_1D` exists on CUDA,
   Metal, SYCL and CANN; only Vulkan lacks it. So composing it in the EXPORTER bakes a Vulkan-shaped
   workaround into an artifact that four other backends would have run in one node. The exporter emits
   one GGUF for every backend and cannot know the target. The engine knows exactly which backend it has,
   and ggml exposes `ggml_backend_supports_op(backend, op)` to ask it.

   Which gives the line these four items were groping for:

   * **The exporter says what the model MEANS.** The three fusions belong there and would be right even
     if every backend implemented every op: fewer nodes, a fused kernel, faster on the CPU too.
   * **The engine says it in ops THIS BACKEND has.** A portability lowering is neither per-model (so not
     the exporter's, by the standing rule) nor per-task — it is per-BACKEND, and the backend is a thing
     only the engine ever sees. One branch per gap, no permanent tax on every artifact, and the topology
     keeps saying `PAD_1D_REFLECT` instead of open-coding it in forty nodes.

   This entry already works that way; P4.7c did not. **P4.7e below is that move**, generalized into a
   mechanism (`PrimitiveContext` carries the `Backends`; `backend_can_run` asks) rather than a branch in
   one primitive.

