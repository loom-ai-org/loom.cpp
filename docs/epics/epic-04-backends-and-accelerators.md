---
type: epic
status: active
domain: backends
last_updated: 2026-08-31
---

# Epic-04: Backends and Accelerators

## 1. Context and Scope

The engine takes a device backend plus a CPU fallback and schedules across them. This epic covers
device resolution, graph scheduling, per-backend op coverage, and which accelerators are reachable
today.

The constraint that shapes all of it: **one GGUF may be run by any backend**, so nothing about which
ops are available can be decided at export time.

## 2. Architectural Overview

* **`loom::Device` resolves a spec against `ggml`'s device registry**, not against any backend name the
  engine knows. A CUDA build's `CUDA0` is selectable by code that has never heard of CUDA.
* **Generic specs rank by kind**, three tiers, discrete before integrated within a rank, derived at run
  time from what the device reports — [ADR-010](../adrs/adr-010-device-selection-by-kind.md). `"npu"`
  always throws: `ggml` has no NPU device type to resolve it to.
* **`GraphBuilder` built against a device plus its CPU fallback uses `ggml_backend_sched`**; a CPU-only
  build still uses the plain `ggml_gallocr` it always did. A default build is unchanged, byte for byte
  and allocator for allocator.
* **Primitives negotiate op coverage with the backend**
  ([ADR-007](../adrs/adr-007-backend-capability-negotiation.md)), so the same file builds 1692 `ggml`
  nodes on a CPU and 1732 on Vulkan.
* **Backends are dynamic libraries and separate packages**
  ([ADR-009](../adrs/adr-009-backends-as-dynamic-libraries.md)) — one engine binary, and the deployment
  decides which accelerator files travel with it.

### What decides the speedup

Not the device — **how many times the scheduler has to cut the graph.** Measured on an AMD Vega 3 iGPU
against 4 CPU threads, one forward each:

| model | splits | GPU vs CPU |
|---|---|---|
| conformer-ctc-small | 1 | **2.56×** |
| lfm2-350m | 1 | **2.22×** |
| qwen3-0.6b | 1 | **2.82×** |
| matcha `encoder_mu` | 1 | **3.65×** |
| kokoro `decoder_vocoder` | 3 | **4.62×** |

Those splits used to be 453, 181, 61, 107 and 5, which left Qwen3 at 0.95× and Matcha at 0.84× —
*slower than the CPU*. None of it was the engine; see
[Retro-009](../retros/retro-009-host-callback-count-was-the-wrong-lens.md) for what it was and why the
obvious metric was the wrong one.

**One caveat on every number above:** that GPU reports `uma: 1`, so it shares memory with the host and
a split costs a synchronisation rather than a transfer. These are a lower bound on a discrete card over
PCIe.

**Retained inter-module outputs are not what a device charges for.** LFM2's 20-module modular export
cost 183 splits against the monolithic export's 181 — measured, against the prediction.

### Verified today

| backend | state |
|---|---|
| CPU | always; the fallback for every split |
| Vulkan | verified, incl. on discrete NVIDIA hardware |
| CUDA | verified on an RTX 5090; both device-parity gates pass against `CUDA0` and **ran** rather than skipped |
| Metal | not started — needs macOS wheels first, see [Epic-08](epic-08-packaging-and-release.md) |
| NPU | open; no NPU registers as a `ggml` device type the engine can resolve |

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-007](../adrs/adr-007-backend-capability-negotiation.md), [ADR-008](../adrs/adr-008-atan-approximation.md), [ADR-009](../adrs/adr-009-backends-as-dynamic-libraries.md), [ADR-010](../adrs/adr-010-device-selection-by-kind.md) |
| Retros | [Retro-007](../retros/retro-007-gpu-chose-the-integrated-gpu.md), [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md), [Retro-009](../retros/retro-009-host-callback-count-was-the-wrong-lens.md) |
| Active tasks | [Backlog → Backends](../backlog/active-index.md#backends--accelerators) |

## 4. The Record

### P4.7 — the engine runs on a GPU — DONE (2026-08-12)


Roadmap item 1 in the README, and the one thing it named as blocking: the engine talked to a single
`ggml_backend_t` through a plain `ggml_gallocr` and used no `ggml_backend_sched` at all. It now takes a
**pair** of backends and schedules across them. A default build is unchanged, byte for byte and
allocator for allocator; a build configured with `-DGGML_VULKAN=ON` (or `GGML_CUDA`/`GGML_METAL`/...)
gets a device.

**The type that carries it is `loom::Backends` (`include/loom/core/backend.h`): two non-owning handles,
`primary` and `fallback`.** It is implicitly constructible from a bare `ggml_backend_t`, which is why
this landed without touching a single one of the 110 test files that construct one — a bare backend
still means "one backend, no scheduler", and `hybrid()` is false for it. `loom::Device` is the RAII
owner and the thing that resolves a spec: argument, then `$LOOM_DEVICE`, then autodetection (the
project's standing order for anything the machine can answer for itself). `"auto"` prefers a device and
falls back to the CPU; `"gpu"` **throws** when there is none, because a caller who spelled it out is
asking a question about the machine and a silent CPU run is how a large slowdown goes unnoticed.

**Two allocation strategies, chosen once per builder by `Backends::hybrid()`** (`graph_builder.h`):

* CPU-only keeps the plain `ggml_gallocr`, the shrink policy P4.0.13/P4.0.15 measured, and everything
  else exactly as it was. A single-backend graph has nothing to schedule.
* A device plus its CPU fallback uses `ggml_backend_sched`. `GraphBuilder::compute()` exists because the
  two need different calls; every compute site in the engine goes through it now.

**Graph reuse survives the scheduler, which was the thing most at risk.** `ggml_backend_sched` keeps
`is_alloc` set until the next `reset`, so a retained graph is re-run without being re-split or
re-allocated: a fixed-shape loop still reports `builds()==1`, and a decode loop keeps both its graph and
its split plan. The scheduler is sized from the BUILT graph (`n_nodes` plus the distinct tensors
reachable through `src`/`view_src`, an upper bound on `n_nodes + n_leafs`, which is what
`ggml_backend_sched_alloc_graph` asserts on) rather than from `estimate_graph_size()`'s deliberately
generous 8x — at that bound the scheduler's own context buffer, `capacity *
GGML_SCHED_MAX_SPLIT_INPUTS * 2` tensor structs, would be hundreds of megabytes for a graph needing
tens, on an engine whose target is edge devices.

**Three host-pointer reads had to go first.** `t->data` is a device address on a device backend, and
`op_range_1d`'s bounds and `op_fill`'s shape both dereferenced it at graph-BUILD time.
`primitive_registry.h` now carries `is_materialized`/`read_tensor_prefix`/`scalar_value_or`, which go
through `ggml_backend_tensor_get`. `op_fill` additionally wrote its result through `dst->data` on a
tensor freshly created in the builder's `no_alloc` context, where `data` is ALWAYS null — a
null-pointer write on any backend, never noticed because nothing exports `FILL`. It is now
`clamp(arange, v, v)`, in-graph.

### What was measured

RADV/GFX9 (AMD Radeon Vega 3, an iGPU) against 4 CPU threads on the same machine, one forward per
figure, graph built once and reused, `ggml` v0.16.0:

| model | ggml nodes | splits | device / CPU nodes | CPU | GPU | |
|---|---|---|---|---|---|---|
| conformer-ctc-small | 2104 | **5** | 2086 / 18 | 144.8 ms | 50.8 ms | **2.85x** |
| lfm2-350m monolithic | 1204 | **181** | 844 / 360 | 465.7 ms | 302.3 ms | 1.54x |
| qwen3-0.6b monolithic | 3050 | **453** | 2146 / 904 | 776.1 ms | 818.2 ms | **0.95x** |

**The split count is the whole story, and it is set by the EXPORT, not by the engine.** Every split is
a device→host→device round trip. What forces them is `ggml_map_custom`: a C function pointer, which no
backend but the CPU can dispatch. And the reason Qwen3 has 453 of them is that the MIL compiler lowers
**RMS norm** to `POW → REDUCE_SUM → SCALE → ADD → RSQRT → MUL → MUL`, of which `POW` and `RSQRT` are
custom ops — 113 of each, exactly 28 layers x 4 norms plus the final one. `SUM_ROWS` and `SCALE` are
then dragged onto the CPU with them because they sit between two CPU nodes.

**The engine has a native `RMS_NORM` primitive, and not one exported model uses it.** Counted across all
thirteen fixtures (`v2/`, exporter HEAD): `RMS_NORM` appears **zero** times, while `POW` appears in ten
of them and `RSQRT` in three.

| fixture | RMS_NORM | POW | RSQRT | |
|---|---|---|---|---|
| causal_lm_kv (qwen3-0.6b) | 0 | 113 | 113 | the 453-split case |
| lfm2, monolithic and modular | 0 | 45 | 45 | |
| matcha | 0 | 38 | 32 | |
| kokoro, styletts2 | 0 | 50 | 0 | `POW` without `RSQRT` — not RMS norm, some other power |
| conformer, parakeet ×2 | 0 | 3 | 0 | |
| gigaam, whisper | 0 | 1 | 0 | |
| supertonic, vits | 0 | 0 | 0 | already free of both |

`exporter.py` DID map `"rms_norm" → "RMS_NORM"`, and that was worse than a missing translation: it
mapped an op **MIL does not have** (checked — coremltools' only `*norm*` core ops are
batch/instance/l2/layer/local_response), so it could never fire, and would have emitted an `RMS_NORM`
node with no `eps` attr — which the engine raises on — if a future coremltools ever added one. The entry
is gone; the primitive is reached by a fusion pass instead. **See P4.7a below: this is now DONE.**

The two smaller variants this entry priced — `POW(x, 2)` and the hand-rolled LayerNorm — are **DONE as
P4.7b below**, which took the remaining 149 `POW` and 32 `RSQRT` nodes out of the zoo and left exactly
two `ggml_map_custom` nodes in it.

**The KV cache works on a device**, which the first version of this entry listed as untested because no
export on hand had one. Re-exporting Qwen3-0.6B-Base against loom-exporter HEAD produced a fused causal
LM with 28 cached `ATTENTION` nodes and an `infer_with_past` entry; twelve greedily-decoded tokens
through the device's cache agree with twelve full CPU prefills over a growing prompt. That covers the
cache in a device buffer, `KvCache::fill_cell_index` rewriting the cell-index tensor between steps on an
already-split graph, and the retained graph outliving a moving `n_past` under a split plan.

**Fidelity.** Frame-wise and token-wise decisions are identical on every model tried; the elementwise
gap is fp32 reduction order, except where a model amplifies it:

* qwen3-0.6b logits: max relative 7.5e-4, rms 2.1e-3, **0/8 argmax disagreements**.
* conformer-ctc: max relative 3.4e-3, **0/17 frame argmax disagreements**.
* lfm2-350m, monolithic and modular: same next token as the CPU. The modular export runs 20 modules
  through the Lua bridge with retained outputs crossing between them, which is the arrangement P4.0.12
  built and the one a device makes expensive — 183 splits against the monolithic export's 181, so
  decomposing a model costs essentially nothing extra here.

**The conformer's 3.4e-3 was bisected, not accepted.** Truncating the topology's node list and comparing
each prefix: exact through node 19, then the STFT's `CONV_1D` pair introduces 8.3e-5, and node 33 — the
log-mel's `LOG` — turns that into 1e-2, because `d(log x) = dx/x` and a near-silent mel bin has a tiny
`x`. Everything downstream inherits it. Giving the test waveform a 1e-3 noise floor, so no mel bin sits
at the zero that does the amplifying, takes the output gap from 2.5e-2 to 3.4e-3 — the same arithmetic,
better conditioned. This is a property of that model's front end, not a measure of backend error, which
is why `tests/gate/test_e2e_device_parity.cpp` compares the argmax exactly on top of the tolerance.

### The fixtures this was verified against

Every artifact on hand predated this work by enough to get in its way: drivers with a `main` entry point
rather than `infer`, a Conformer export with no `tokenizer.ggml.*` KVs at all, a Qwen3 export that
re-prefills instead of caching. Twelve models were re-exported against loom-exporter HEAD into a
`LOOM_FIXTURES` directory — conformer-ctc, kokoro, matcha, vits, styletts2, supertonic, lfm2
(monolithic and modular), parakeet-tdt, parakeet-rnnt, gigaam, whisper, plus the fused Qwen3-0.6B-Base
that gave this entry its `causal_lm_kv.gguf`. With those present, `ctest -L gate` runs 10 real tests
rather than skipping everything, and all 10 pass on the CPU and Vulkan builds alike.

The rest of the gate suite still skips: those tests want reference `_DIR` fixtures — PyTorch forward
dumps from `fixture_gen/` — and a GGUF alone does not satisfy them. Producing all of those is a
separate, much larger job, and the tests that do run are the ones that exercise a whole model through
the Lua bridge, which is the path this change touched.

### Tests

* `tests/ci/test_device_selection.cpp` — what a spec resolves to and what it refuses. Hermetic, so it
  names only the CPU and states the rest as invariants ("auto" resolves to a device iff one exists).
* `tests/ci/test_scheduled_graph.cpp` — the scheduler path driven by `Backends{cpu_a, cpu_b}`: two CPU
  backends, an arrangement no host would ask for and the only one that exercises the machinery with no
  GPU present. Parity is exact, reuse counters hold, a rebuild releases the previous allocation.
  **Its limit is recorded in the file:** swapping `ggml_backend_sched_graph_compute` for the plain
  `ggml_backend_graph_compute` does NOT turn it red, because two CPU backends put every buffer in host
  memory. The parity comparison itself is live (feeding the scheduled run different tokens fails it).
* `tests/gate/test_e2e_device_parity.cpp` — CPU vs device on the real Conformer-CTC encoder. Skips (77)
  on a missing fixture AND on a build with no device backend, so it is green everywhere it cannot speak.
  Asserts the device ran the majority of nodes, which is the failure mode "it runs on the GPU now" hides
  best.

* `tests/gate/test_e2e_device_parity_kv.cpp` — the other half, and the one that answers what the first
  version of this entry listed as unverified: a KV-cached decode on the device against N full prefills
  on the CPU. It is the only test that exercises the cache living in a device buffer, the cell-index
  tensor being rewritten between steps on a graph the scheduler has already split, and P4.0.15's graph
  reuse holding while `n_past` moves underneath a split plan. Twelve greedily-decoded tokens agree, on
  a graph the scheduler cuts 453 times.

Sharing `LOOM_CONFORMER_CTC_MIL_GGUF` with `test_vocab` turned up a **pre-existing crash in that test**,
fixed here: `LOOM_CHECK` records a failure and carries on, so a `Vocab::load` returning null — which is
what an export predating the embedded SentencePiece vocab (P4.0.17 step 3) gives, since it has no
`tokenizer.ggml.*` KVs at all — was dereferenced on the next line and took the process out with SIGSEGV.
It now reports and returns. Nothing about a device is involved; pointing that variable at an old
artifact on disk is all it took, and "this test is broken" was the wrong message for "this fixture is
too old".

### The build tools, and `cmake/VulkanToolchain.cmake`

Neither of Debian bookworm's two relevant packages is new enough for `ggml` v0.16.0's Vulkan backend,
and both fail in ways that name neither cause:

* `glslc` 2023.2 answers ggml's `GL_KHR_cooperative_matrix` probe as though it supported it — the probe
  greps stderr for "extension not supported", which this version does not emit — so coopmat shader
  variants get generated and the build dies in `conv2d_mm.comp` with `'coopmat': undeclared
  identifier`. `-DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT=OFF` does not help: ggml's CMake overwrites it from
  the probe.
* Vulkan-Headers 1.3.239 lack `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` (1.3.264+) and
  `vk::LayerSettingEXT` (1.3.272+). Header-only, and the loader is ABI-stable, so newer headers against
  the system `libvulkan.so.1` is a supported arrangement rather than a workaround.

`cmake/VulkanToolchain.cmake` makes `-DGGML_VULKAN=ON` work on such a machine without anyone having to
know the above. **FetchContent with pinned tags, not submodules** — the same answer this repo already
gives for ggml, nlohmann_json and LuaJIT, and it gives the "bump it when we need to" property a
submodule would without a `--recursive` clone every consumer has to remember.

Three decisions in it are worth keeping:

* **Both probes test the failure, not a version number.** The glslc probe runs ggml's own feature-test
  shader and accepts two answers — it compiles, or it refuses in the words ggml greps for; anything
  else is a glslc that will lie to ggml's probe. The header probe compiles the two declarations
  `ggml-vulkan.cpp` uses. A version comparison would be a proxy that goes stale in both directions, and
  a backported distro package is a real case.
* **glslc is built at CONFIGURE time, not as an ExternalProject.** ggml runs glslc during ITS configure,
  five times, and those runs decide which shader variants are generated and which compile definitions
  are set. A binary that does not exist yet makes all five fail to launch, which ggml reads as
  "supported" and acts on — a wrong answer from a process that never ran. The cost is a slow first
  configure on a machine that needed it; it is idempotent per build directory.
* **A generated `SPIRV-HeadersConfig.cmake`.** ggml does `find_package(SPIRV-Headers CONFIG REQUIRED)`
  and then never links the target, so SPIRV-Headers has to be satisfied twice over and FetchContent's
  own redirect covers neither: the package config is absent (SPIRV-Headers defines its config through
  `install(EXPORT)`, which produces nothing in a build tree that never installs) and the include path is
  absent (nothing links the target that carries it). That is the `'spv' has not been declared` error.
  The module writes a three-line config and an `include_directories`.

### What this does NOT cover

* **NPUs.** The device layer resolves `GGML_BACKEND_DEVICE_TYPE_ACCEL` alongside GPU/iGPU, so an
  accelerator backend would be selected; none has been built or run.
* **Only Vulkan, only one device.** CUDA, Metal, SYCL and the rest are compile-time switches that were
  never flipped here. Multi-GPU is not attempted: `Device` initializes exactly one device backend, and
  `ggml_backend_sched` is handed two backends, not N.
* **Quantized weights on a device** (Q8_0 exports) were not run.
* **`loom-py` ships CPU-only wheels.** The binding takes `device=` and forwards it; a wheel with a
  device backend compiled in does not exist yet.
* **Flash attention** is still not a primitive. The README named a GPU as what would make
  `ggml_flash_attn_ext`'s forced F16 K/V cast worth its precision cost; that trade is now possible to
  make and has not been made.


### Holding more than two backends — DONE (2026-08-14)

`Backends` holds N. It keeps `primary` and `fallback` exactly as they were -- so the implicit
`ggml_backend_t` conversion and every existing call site are untouched -- and gains `assists`, the
backends that sit BETWEEN the primary and the CPU. `schedule_order()` is the single place that knows
the ordering ggml requires, and `GraphBuilder`'s hardcoded `backends[2]` is gone.

The rule: **a primary with its own memory attaches every host-memory accelerator; nothing else
attaches anything.** A host-memory primary gets no assist (there is nothing between it and the CPU
worth having), and a second discrete device is never attached, because ggml has no general
peer-to-peer path -- a copy between two discrete backends goes through host memory both ways, four
transfers where falling back to the CPU costs two. An assist that fails to initialize is skipped
rather than fatal: the graph is correct without it, since the CPU can run everything.

Verified on the three-device DL build:

```
spec auto -> primary Vulkan0  assists=1  order: Vulkan0 BLAS CPU
spec gpu  -> primary Vulkan0  assists=1  order: Vulkan0 BLAS CPU
spec BLAS -> primary BLAS     assists=0  order: BLAS CPU
spec cpu  -> primary CPU      assists=0  order: CPU
```

**And it costs nothing where it gains nothing, which is the measurement that mattered.** The worry was
that a third backend would perturb split planning -- the thing P4.7a-d spent four items driving down.
Same binary, same models, `--device Vulkan0`, before and after:

| model | splits before | splits after | device / fallback nodes |
|---|---|---|---|
| `lfm2_monolithic` | 1 | 1 | 876 / 0 (unchanged) |
| `lfm2_modular` (aux, prefix) | 1, 1 | 1, 1 | 13 / 0, 2 / 0 (unchanged) |
| `causal_lm_kv` | 1 | 1 | 2202 / 0 (unchanged) |

Identical, as predicted from BLAS's op set being a subset of every GPU backend's -- it claims nothing
the GPU had already claimed. The probe above is what makes that a real result rather than a vacuous
one: it confirms the assist IS in the chain while the numbers stay flat, so this is "costs nothing",
not "did nothing".

Where it is expected to pay is untested here and honestly so: a primary with thin op coverage and
large matmuls falling back -- an NPU. That measurement needs the 285K.

**Not done, deliberately:** `LoomLuaBridge::device_report()` still buckets every node as either
"device" or "fallback" by comparing against the primary's name, so a node that ran on an assist counts
as fallback. That is not wrong (it did not run on the primary) but it will under-report an assist
doing useful work, which is exactly the case the NPU measurement will need to see. Fix it when there
is a backend where the distinction is visible.

Worth noting for the NPU work specifically: BLAS is a usable local proxy for the ACCEL *selection*
path, and is nothing at all as a proxy for NPU throughput. No performance number should ever be taken
from it.


### P4.8c — CUDA, on the workstation: the tier-1 claim stops being a claim (2026-08-14)

The scoping above says of every backend ggml ships that **"the engine needs no work"**, and that the
honest first step was "CUDA on a box that has an NVIDIA GPU". That step is taken, on the RTX 5090
workstation, and it is worth being blunt about what was and was not done: **not one line of C++ was
written or changed.** The tree was rsynced across, configured with two flags, and built.

```sh
cmake -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
```

438 targets in **~5 minutes** on 24 cores, against the 20–40 the same suite costs on the dev box, with
no warning from any file this repo owns (the one that appears is upstream's `ggml-cpu/repack.cpp`).
`ctest -L ci`: **58/58 in 5.45 s** — CUDA compiled in changes nothing hermetic, which is the null
result that had to be checked before any of the rest meant anything.

One thing to know before repeating it: **ggml rewrites the architecture.** `-DCMAKE_CUDA_ARCHITECTURES=120`
configures as `120a`, the arch-*specific* Blackwell feature set, and the native variant as `120a-real`.
That is upstream's doing rather than a typo to correct, and the resulting binary is what the numbers
below were measured on.

**What the probes say, and the first one is a second sighting of the P4.8b trap.**
`tools/debug/probe_tiers` on the workstation:

    device     type   buft_is_host   host_buffer
    CUDA0      GPU    false          true
    CPU        CPU    true           false

`caps.host_buffer` is **true** for CUDA and `buft_is_host` **false** — exactly the inversion measured on
Vulkan, now confirmed on an unrelated backend. Had the ranking been built on `caps.host_buffer`, as its
name invites, CUDA would have ranked as a host-memory accelerator. `probe_selection` resolves
`"auto"` → `CUDA0` and `"gpu"` → `CUDA0`; `probe_chain npu` throws the intended message. `assists=0`,
correctly — there is no host-memory accelerator in this build for a discrete primary to attach.

**Both device-parity gates pass against CUDA0, and they ran rather than skipped** (the fixtures were
copied over; `ctest` reports Skipped for exit 77 and reported Passed):

| | |
|---|---|
| `test_e2e_device_parity` (Conformer-CTC) | 2104 device nodes, **0 CPU fallback, 1 split** |
| | max relative diff **3.492e-03**, 0 argmax disagreements over 17 frames |
| `test_e2e_device_parity_kv` (Qwen3-0.6B) | 12/12 tokens identical to the CPU's iterated-prefill oracle |
| | `main_topology`: **1 split**, 2034 device / 0 CPU |

**The 3.492e-03 is the most informative number here, and not because it is small.** That test's header
argues the tolerance is a property of *this model's front end* — the log-mel's `LOG` amplifying an
8.3e-5 `CONV_1D` difference, bisected node by node — rather than a measure of how wrong a backend is
allowed to be. It was measured at 3.4e-3 on RADV/GFX9. A second backend, different vendor, different
memory system, different reduction order, landing at 3.49e-3 is what turns that argument into evidence.
If the number had been a Vulkan artefact there is no reason CUDA should agree to two digits.

**Throughput, whole process including load, Qwen3-0.6B, 64 tokens, interleaved:**

| device | runs | |
|---|---|---|
| CPU (Core Ultra 9 285K, 24 cores) | 22.50 / 21.67 / 20.85 s | |
| CUDA0 (RTX 5090) | 1.46 / 1.42 / 1.48 s | **≈ 14.7x** |

Two caveats that keep this honest: it is wall clock for the whole process, so the 2.3 GB weight upload
is *inside* the CUDA number rather than excluded from it, and the CPU arm is a 24-core desktop part,
not a weak baseline. The generated text is character-identical on both — 64 greedy tokens agreeing is
the same amplifier `test_e2e_device_parity_kv` relies on, run longer.

**Sizing, and it corrects the scoping above.** `libggml-cuda.so` stripped, single arch `120a`: **59 MB**,
against Vulkan's 46.5 MB — so a CUDA deployment is **≈ 62 MB**, and the engine is still 2% of it. The
scoping predicted CUDA "clears [PyPI's 100 MB ceiling] outright". **For one architecture it does not**,
which matters for `loom-py-rt-cuda`: the thing that blows the ceiling is the multi-arch fat binary a
general-purpose wheel ships (sm_75 through sm_120), not CUDA as such. A per-arch or narrow-arch backend
package is therefore a live option rather than a lost cause, and which arches to carry becomes a real
decision instead of a foregone one.

## What P4.8c did NOT establish

Four things, each of which someone could mistake this entry for having covered.

1. **The NPU is still untested, and the hardware being present is why that is worth saying.** The Intel
   NPU does not appear in this registry at all — no OpenVINO backend was compiled in, so
   `/dev/accel/accel0` sits there unregistered. Every rank-1 question the scoping raises (does a discrete
   NPU register as ACCEL with non-host memory?) is exactly as open as it was.
2. **This is a LINKED build, not a `GGML_BACKEND_DL` one** — so it is not the configuration the wheels
   ship, and the 109-file prerequisite P4.8b found still stands between the gate suite and that
   configuration. It should be cleared before a `loom-py-rt-cuda` is trusted.
3. **Quantized weights on a device remain unverified.** Both fixtures here are F32/F16; the `q8_0` gates
   run on the CPU. That gap predates CUDA and is untouched by it.
4. **No `loom-py-rt-cuda` package exists.** `packaging/common/BackendPackage.cmake` is the reusable
   half (P4.8a) and a CUDA package is that directory with two strings changed, but it has not been
   built, so the `==`-pin and `$ORIGIN` story is still verified only for Vulkan.

The workstation tree lives at `/home/flavio/loom/loom.cpp` (rsynced, not cloned — see the note in
P4.8's toolchain section), fixtures at `/home/flavio/loom/fixtures`, and the probes are hand-compiled
into `/home/flavio/loom/probes` per `tools/debug/README.md`.


### P4.8d — the NPU answers, and the answer invalidates rank 1 (2026-08-14)

`tools/debug/README.md` names the first question to point a probe at: **does a discrete NPU register as
`GGML_BACKEND_DEVICE_TYPE_ACCEL` with a non-host buffer type?** — the assumption rank 1 rests on, which
no hardware had ever tested. The Intel NPU in the workstation now answers it, and it is the branch the
scoping listed as the bad one: **it is not ACCEL at all**, so `Device::open("npu")` never resolves and
that spec is dead code. Not because of the hardware — because of the backend.

## Getting there, which is nearly root-free and worth writing down

* **`pip install openvino` is the whole toolkit install.** The 2026.3.0 wheel ships
  `OpenVINOConfig.cmake` *and* `libopenvino_intel_npu_plugin.so`, so `-DOpenVINO_DIR=<site-packages>/openvino/cmake`
  is all `ggml-openvino/CMakeLists.txt`'s `find_package(OpenVINO REQUIRED)` needs. No apt, no archive.
* **The build dies on `CL/cl2.hpp` and it is not obvious why.** OpenVINO's `intel_gpu/ocl/ocl_wrapper.hpp`
  includes the *deprecated* OpenCL C++ bindings, which `opencl-headers` does not carry. conda-forge's
  `clhpp` is the missing piece; `ocl-icd` + `opencl-headers` alone configure fine and then fail to compile.
* **The NPU user-mode driver needs no root either.** `intel/linux-npu-driver` v1.35.0 ships `.deb`s;
  `dpkg -x` into a prefix and `ZE_ENABLE_ALT_DRIVERS=<prefix>/…/libze_intel_npu.so.1` is enough for the
  system's `libze_loader` (1.20.6) to find it. The firmware was already in `/lib/firmware/intel/vpu`.
* **One thing does need root, and it is the whole blocker:** `/dev/accel/accel0` is `root:render` 0660.
  Without the `render` group the driver says `Failed to detect any VPU device` and OpenVINO reports
  `['CPU']` — which looks exactly like a machine with no NPU. `sudo usermod -aG render <user>`, then a
  fresh login. After it: `['CPU', 'NPU']`, `NPU -> Intel(R) AI Boost`.

## What the probes measure, with the NPU genuinely engaged

    OpenVINO: using device NPU
    device     type   buft_is_host   host_buffer
    OPENVINO0  GPU    false          false
    CPU        CPU    true           false

    "auto" -> OPENVINO0    "gpu" -> OPENVINO0
    "npu"  -> throws: no NPU/accelerator device with its own memory is available

**Every one of those rows is byte-identical to the run where the backend targeted the CPU.** That is the
finding, and it is worse than P4.8b's BLAS defect rather than another instance of it:

1. **`ggml_backend_openvino_device_get_type` returns `GGML_BACKEND_DEVICE_TYPE_GPU` unconditionally**
   (`ggml-openvino.cpp:751`), and the device buffer type leaves `.is_host` null, so it reads as discrete
   memory. The backend reports the family it belongs to, not the device it drives.
2. **Which hardware it actually drives is an environment variable the engine never sees.**
   `GGML_OPENVINO_DEVICE` defaults to `"CPU"`; asking for one that is absent prints
   `device NPU is not available, fallback to CPU` and carries on, registry entry unchanged. So `"gpu"`
   can resolve to a backend executing on the CPU, and **no registry property distinguishes the three
   cases**. BLAS at least declared ACCEL, which is what let a rank partition exclude it. There is no
   analogous repair here: a ranking cannot correct a backend that misreports its own type.
3. **`"npu"` throws on the one machine in this project with a real NPU, actively in use.** Not a
   hypothetical about exotic hardware — the exact configuration the spec was written for.

The WARN is visible at all only because P4.8a's log callback keeps WARN/ERROR while dropping INFO. This
is precisely the failure that decision was made for, arriving sooner than expected.

## And then it does not run the graph at all, for a reason op coverage never predicted

P4.8's item 3 warns that **"a first NPU benchmark will not measure the NPU; it will measure how much of
the graph fell back"**, and points at P4.7d's support matrix. That worry was aimed at the wrong thing.
Qwen3-0.6B on `--device OPENVINO0` with `GGML_OPENVINO_DEVICE=NPU`:

    GGML OpenVINO backend std::exception: stoi
    error: GraphBuilder::compute: the graph failed to run on OPENVINO (ggml_status -1)

`ggml-decoder.cpp:304`, `extract_layer_from_name`: find `"_l"` anywhere in a tensor's name, take
everything up to the next space, `std::stoi` it — llama.cpp's `cache_k_l0` convention. It is called
through `compute_llm_params` (`utils.cpp:185`, `:424`), and at two call sites via `.value()` on the
optional. So a name with `_l` followed by anything non-numeric throws, and a graph with no `_l` at all
throws differently.

> **Correction (2026-08-14, same day).** This entry first said `ggml-openvino` "is a llama.cpp-shaped
> backend, not a general ggml one" that "reconstructs a transformer from tensor names rather than
> executing the ggml graph it was handed". **That is wrong and unfair to the backend**, and the
> upstream doc (`llama.cpp/docs/backend/OPENVINO.md`) plus the source say why. It IS a general
> translator: `openvino/op/` holds 33 per-op translator files, `op_table.cpp` maps 39 GGML ops onto
> them, and `translate_session.cpp` builds an `ov::Model` through OpenVINO's frontend API — the doc's
> own words are that it "walks the GGML graph and identifies inputs, outputs, weights, and KV cache
> tensors" and then "translates the GGML operations into an `ov::Model`". There is even a general
> escape hatch: `is_naive(cgraph)` sends any graph of **fewer than 20 compute nodes** to
> `naive_compute`, which never touches the LLM machinery at all.
>
> The accurate statement is narrower and more useful: **it is a general translator with a mandatory
> LLM-shaped parameter-inference step for any graph of 20 nodes or more**, and that step identifies the
> KV cache and per-layer tensors by parsing llama.cpp's naming. loom's Qwen3 graph is 2202 nodes, so it
> takes that path and dies there — before op coverage is ever reached. Everything measured above stands;
> only the explanation of it was overstated.

Note the trap in the combination, stated carefully: `supports_op` is an ordinary per-op table, so the
scheduler is told node-by-node that these ops are supported — and above the 20-node threshold the
execution path imposes a further STRUCTURAL precondition that `supports_op` never mentions. It is not
that the backend declines to honour its own op table; it is that op support is necessary and not
sufficient, which no part of the device API can express. That is a second limit on what
`ggml_backend_supports_op` can be trusted to answer, alongside P4.7e's.

## Forcing the naive path: measured, and it does NOT capture everything

The obvious follow-up to the correction is whether `naive_compute` — the sub-20-node escape hatch that
skips all LLM machinery — would run loom's graphs if the threshold simply did not stop it. It is a
one-constant experiment (`naive_graph_size_threshold = 20` → a million, in a build tree, reverted
after), and the answer is no. **Three distinct barriers, all hit within one afternoon on two models:**

1. **The name parser is reachable from the naive path too**, which was the surprise. `naive_compute`
   binds inputs through `get_ov_input_tensor` → `try_make_kv_sliced_tensor` → `extract_layer_from_name`
   (`utils.cpp:743`, `:806`), so the causal-LM still died on `stoi` with the LLM path bypassed entirely.
   The cause is loom's own naming: `extract_layer_from_name` does an **unanchored** `name.find("_l")`,
   and **308 of the causal-LM's 316 tensors** are `model_model_layers_0_...`, where `_layers_...` parses
   as the layer index and throws. Not one loom tensor matches llama.cpp's `_l<digit>` convention.
2. **`IM2COL` dynamic-dim propagation asserts a stride-1 convolution** (`ggml-decoder.cpp:1529`,
   `node->src[1]->ne[src_dyn] == node->ne[...]`, i.e. `IW == OW`). Conformer-CTC's convolutional
   subsampling is strided, so it fails there on the dynamic path.
3. **On the static/NPU path a different wall**: `broadcast_merge_into` failing inside OpenVINO's own
   eltwise shape inference — a translation gap rather than a workload assumption.

Barriers 2 and 3 should not be read as defects. Upstream's doc scopes the backend to "a subset of GGML
ops and **text-only models**"; Conformer-CTC and the whole TTS half of the zoo are outside that scope,
and hitting walls there is the documented behaviour rather than a surprise.

## What that leaves, and what it rules out

**Ruled out: porting a forced-naive backend into this repo.** The idea was that naive is a general
translator held back by a threshold; it is not — it is the small-graph escape hatch, and outside
text-only models it fails immediately for reasons a threshold does not touch. Vendoring a 250 KB
actively-developed backend to reach that is the relicensing-and-maintenance decision `Dependencies.cmake`
exists to avoid, for a path measured not to work.

**Barrier 1 is on OUR side of the fence — so it was removed, to see what was behind it.** The engine
renamed every graph tensor `_l` → `_L` just before compute (the parser's `find` is case-sensitive, so
this defeats it and changes nothing else), env-gated, in a build tree, reverted after. Two rounds,
because the first did not land where it looked:

* **With the rename alone**, `stoi` is gone and the model **translates and compiles** — real progress.
  It then fails binding an OUTPUT: model `[1,1,5,?]` against loom's `[1,8,5,128]`. A head count of
  **one**. The forced naive threshold was not in effect, because the dynamic path gates naive on
  `!is_model_splitted(cgraph)` and loom's graph trips that heuristic — so it was on the LLM path with
  `compute_llm_params` having recognised no attention pattern and left the head count at its default.
* **With `is_model_splitted` forced false as well**, genuinely on the naive path, it fails at the very
  first INPUT: model `[1,8,5,128]` against loom's `[1,5,8,128]`. **Two middle dimensions transposed.**

So the answer to "does it work at all" is no, and the reason is now specific and is neither naming nor
op coverage: **the backend's shape and layout derivation disagrees with loom's tensors.** loom builds
attention out of permuted views, and the translator's idea of a parameter's shape is not that view's
shape. That is a deeper incompatibility than the parse, and it sits in the part of the backend built
around llama.cpp's graph conventions.

What that settles: **the naming work is not worth doing.** Re-exporting the zoo to dodge an unanchored
substring search buys a translation that then disagrees about dimension order on input 0. The upstream
ask stays worth filing and stays two lines — anchor the search, or return `nullopt` on parse failure
rather than throwing, so a foreign graph gets a clean "unsupported" instead of a `stoi` — but it should
be filed as a robustness fix, not as something that unblocks loom.

**And on pre-applying the optimizations the naive path lacks** — the idea that if we know what the LLM
path does, loom can do it first: mostly true, with one exception that matters. Static shapes and KV
slicing are things loom is unusually well placed to supply, since it already buckets shapes (P4.0.15)
and owns its own cache. But `naive_compute` calls `core.compile_model()` on **every** `graph_compute`
with no `decoder_cache`, so a forced-naive decode would recompile a 2200-node OpenVINO model per token.
That is not a graph optimization that can be applied earlier; it is a caching layer inside the backend,
and no amount of preparation on loom's side substitutes for it.

## What this leaves

* **Rank 1 has no *possible* inhabitant** — stronger than it looked when this was written, and P4.8e
  below has the enum comment that settles it. It was designed for "an accelerator with its own memory";
  ggml defines `ACCEL` as the BLAS/AMX co-processor role instead, so the tier was never going to fill.
* **The tie-break-within-a-rank measurement is still unavailable**, and the reason changed: the
  workstation was supposed to be the box with a rank-0 GPU *and* a rank-1 NPU at once. It has both
  physically and only rank 0 in the registry, so a CUDA + OpenVINO build produces two rank-0 devices
  separated by registration order — which is the tie-break case, but not the one that was wanted.
* **No throughput number exists, and none should be quoted.** Nothing ran. The NPU's speed on a loom
  graph is exactly as unknown as it was this morning.
* **Hexagon: read, not built — and it clears the charge OpenVINO does not** (2026-08-14, below).

The OpenVINO build tree is `/home/flavio/loom/loom.cpp/build-openvino` (tests off), its probes are in
`/home/flavio/loom/probes-ov`, and the extracted NPU driver is at `/home/flavio/loom/npu-umd/prefix` —
usable only with `ZE_ENABLE_ALT_DRIVERS` and `LD_LIBRARY_PATH` pointed at it.

## P4.8d(ii) — `ggml-hexagon`, read rather than built, and rank 1 becomes a pattern

There is no Qualcomm hardware here and `ggml-hexagon/CMakeLists.txt` demands `HEXAGON_SDK_ROOT` — a
registration-walled proprietary SDK that cross-builds DSP-side "skel" libraries and optionally
code-signs them (`HEXAGON_HTP_CERT`). So this is a read of 4351 lines, which is the right cost for the
question: does the other in-tree NPU backend make P4.8d's mistakes?

**On the charge that matters, it is innocent — and note the charge itself was narrowed by the
correction above.** `ggml_backend_hexagon_graph_compute` walks `graph->nodes[i]`, remaps each node to
an HTP opcode, fuses neighbours where it can, and submits. There is **no tensor-name parsing anywhere**
— every `->name` is logging, every `atoi` is env-var option handling — and the string `llama` does not
occur in the file. It executes the graph it is handed, whatever shape that graph is, with no
LLM-parameter step and so no node-count threshold above which a structural assumption switches on.
That is the difference from `ggml-openvino`: not translator versus reconstructor, since both translate,
but whether anything is assumed about what the graph MEANS.

**On the ranking it makes the same claim, and that is the finding.**

```c
static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);          // unreachable -- same tell as ggml-openvino
}
```

**Two independent NPU backends, written by different vendors, both report GPU.** One is an accident;
two is the convention. Rank 1 — "an accelerator with its own memory, a discrete NPU" — is not a tier
ggml's backends populate, and the `"npu"`/`"accel"` spec has no reachable inhabitant in the pinned
revision. That is now a statement about ggml rather than about one backend, and it is what should
decide whether the spec stays.

**And its host/discrete answer is an environment variable.** `ggml_backend_hexagon_buffer_type_is_host`
returns `opt_hostbuf` — `GGML_HEXAGON_HOSTBUF`, **default 1, i.e. host memory** — and
`props->caps.host_buffer` is set from the same global. So the probe P4.8b made load-bearing for the
rank-1/rank-2 split is, for this backend, *runtime-configurable*: same silicon, either answer, chosen
by the environment. Tallying every backend measured or read so far:

| backend | `caps.host_buffer` | `buft_is_host` |
|---|---|---|
| Vulkan, CUDA | true | false |
| BLAS | false | true |
| OpenVINO | false | false |
| Hexagon | `opt_hostbuf` | `opt_hostbuf` (same) |

Inverted, inverted the other way, both false, and identical-but-configurable. **Neither field carries
the same meaning across backends**, which is a stronger version of the trap P4.8b recorded: the fix was
to prefer `buft_is_host`, and this says `buft_is_host` is merely the less bad of two unreliable inputs.

**Its op set is aimed at llama.cpp's workload even though its architecture is not**, and for loom's zoo
that is the binding constraint. The 39 supported ops are LLM-shaped — `MUL_MAT`, `MUL_MAT_ID`, `ROPE`,
`FLASH_ATTN_EXT`, `RMS_NORM`, `SOFT_MAX`, `GLU`, `SSM_CONV`, `GATED_DELTA_NET` — and contain **no
convolution of any kind**: no `CONV_1D`, `CONV_2D`, `CONV_TRANSPOSE_1D`, not even `IM2COL`, alongside
the `POOL_1D`/`POOL_2D`/`PAD_REFLECT_1D` gaps P4.7d already tabulated. Every ASR encoder and every TTS
vocoder in the zoo would fall back essentially whole; the causal-LM family is the only part with a
plausible story. So P4.8's "budget for the coverage work before drawing any conclusion from a number"
holds for Hexagon — it is just that the missing class is convolution, not the three exotic ops the
matrix pointed at.


## 5. Planned: Metal (P4.11) — SCOPED, strictly after macOS wheels

**Ordering, both halves of it.** This cannot start before P4.10: there is no macOS base wheel for a
backend package to attach to, and P4.10's blocker 4 — ggml's DL loader searching for `.so` where CMake
wrote `.dylib` — is the *same code path* that would discover `libggml-metal`, so starting here would
mean debugging that twice. But unlike P4.10 this does **not** block P5, and the reason is worth stating
so the sequence is not read as one long Apple project: P4.10 decides what a Mac can run **at all**,
which multiplies every family P5 adds; P4.11 decides how **fast** one platform runs, which multiplies
nothing. If P5 is ready first, P5 goes first.

**The shape is already decided and is not the work.** `loom-py-rt-metal` is `packaging/rt-vulkan/`
with the four strings `packaging/README.md` names, plus `archs = "arm64"` (cibuildwheel's macOS
spelling) instead of `"x86_64 aarch64"`. **Intel Macs get no Metal wheel**: `packaging/README.md`
already scopes Metal as arm64-only, and P4.10 explains why Apple Intel is a target one row wide.

**What it costs elsewhere: a version bump becomes TEN strings across FOUR files**, up from seven
across three — the root pyproject gains a `metal` extra pin, and the new package carries its own
`version` plus its `loom-py-rt ==` pin. That circular exact-pin set is the ggml-ABI agreement, so the
new file joins it rather than sitting beside it.

**Four questions to answer before any workflow is written.** The Vulkan and CUDA packages made every
one of these a Linux answer, and none of them carries over:

1. **Discovery** — P4.10 blocker 4, unchanged and shared. Settle it there; this inherits the answer.
2. **Linking is `install_name`/`@rpath`, not soname.** The build-side half of the `==` pin
   (`cmake/GgmlPin.cmake`, read by both builds so the revision cannot drift) is unaffected, but the
   mechanism that binds a backend to its base library is different on macOS, and P4.8g is the record of
   what a mismatch looks like: it loads without error, registers nothing, and shows up only as an
   accelerator missing from `loom.devices()`.
3. **`GGML_METAL_EMBED_LIBRARY` — ANSWERED 2026-08-31 by reading, and the answer is better than
   expected.** It defaults to `${GGML_METAL}` (`ggml/CMakeLists.txt:241`), so it is already ON whenever
   Metal is. What it embeds is the **`.metal` SOURCE**, not a compiled metallib: two `sed` passes
   inline `ggml-common.h` and `ggml-metal-impl.h`, and the result goes into the binary through
   `.incbin` in a `__DATA,__ggml_metallib` section, compiled by the Metal framework at run time. Two
   consequences. **(a)** The lone-`.dylib` trap does not arise — nothing looks for `default.metallib`
   beside a `python` executable. **(b)** **Full Xcode is NOT required to build it.** `xcrun -sdk macosx
   metal` is invoked at `src/ggml-metal/CMakeLists.txt:95`, inside the `else()` of
   `if (GGML_METAL_EMBED_LIBRARY)` — the **whole** non-embed branch, not just its `SHADER_DEBUG` arm,
   which only picks flags. So with EMBED on, nothing in the build ever calls it. Do not let a missing
   `metal` compiler be read as "cannot build Metal here"; it means "do not turn EMBED off".

   **And it is not a Homebrew formula, so do not go looking for one.** `xcrun` itself is present
   (`/usr/bin/xcrun`); what is absent on a Command-Line-Tools-only install is the `metal` *utility* it
   dispatches to, which ships with **full Xcode** — App Store, ~10-15 GB, then
   `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, and on Xcode 16+ the Metal
   compiler is a further downloadable component (`xcodebuild -downloadComponent MetalToolchain`).
   **The recommendation is not to**: it buys only the non-embed path, which a DL-loaded backend must
   not use anyway.
4. **Size, measured rather than guessed.** `libggml-vulkan.so` is 46.5 MB because 44 MB of it is
   compiled SPIR-V; Metal ships shader *source* or a metallib and should be far smaller, which would
   make it the first backend package that is small for a reason other than restraint. The number belongs
   in `packaging/README.md`'s table once it exists — do not write it before.

**What to expect when it does run, and the trap in expecting it.** Metal implements both
`PAD_REFLECT_1D` and `POOL_1D` natively (P4.7d's support matrix; only Vulkan lacks the pad, and only
Metal and SYCL have the 1-D pool), and P4.7e put the engine's substitutions behind
`backend_can_run` — so on Metal those lowerings stay *off* and the graph should split less than
Vulkan's does. That is a prediction, not a result. P4.8's standing warning applies unchanged: a first
benchmark measures how much of the graph fell back, not the backend, so read the split count before
reading the timing.

**Device selection needs no new work**, and that is a claim to check rather than assume: ggml-metal
registers as a GPU-kind device, so P4.8e's hierarchy should rank it without a new tier and
`device="gpu"` should find it. `loom.devices()` naming it is the check.

**A quantized convolutional model is already safe on Metal, and here is why so it is not re-derived.**
P4.29 gave a folded, block-quantized conv kernel a CPU-only lowering
(`ggml_conv_2d_direct_packed`, geometry in `op_params` 6..8) and gated it on `backend_can_run`.
Metal's `supports_op` for `GGML_OP_CONV_2D` (`ggml-metal-device.m`) requires
`src[0]->type == F16 || F32`, so it declines the node and the graph keeps the `im2col` +
`mul_mat_kernel_first` lowering — the same path Vulkan runs with zero fallback nodes. **Checked by
reading, not run**, like everything else in this section.

The one thing to watch: Metal declines on the TYPE test alone, where Vulkan also fails a geometry test
(`cout == op->ne[2]`) and CUDA was given an explicit `op_params[6] == 0` guard because it had neither.
Today that is enough, because loom only ever builds a packed node for a *quantized* kernel. **If a
packed node is ever built from an F32 folded kernel, Metal needs CUDA's guard**, or it will read a
`[IC*K, OC]` tensor as a `[KW, KH, IC, OC]` one and return a wrong answer rather than an error.

**Done means something stricter than P4.10's bar, because this failure is silent.** P4.10 can be
called done on "built and imported in CI"; a backend cannot, since one that fails to load is
indistinguishable from a slow CPU run. Done here is: `loom.devices()` listing Metal on a real Mac, a
model producing output that matches the CPU path, and a timing pair for both.

**The hardware now exists — `fdemelo@macbook-pro`, an Apple M1 Pro on macOS 15.6.1 (Epic-08 §4 has the
full inventory)** — so this item is no longer blocked on borrowing a machine, and the P4.8c standard
becomes reachable rather than aspirational: CUDA stopped being a claim only when it ran on the
workstation, and Metal should stop being one only when it runs on this laptop. **Read the split count
before the timing** — a first benchmark measures how much of the graph fell back, not the backend.

**Non-goals:** CoreML and the Neural Engine (not Metal, no ggml backend targets it, and out-of-tree
options were already turned down on licensing); Intel Macs; a `universal2` backend wheel.

