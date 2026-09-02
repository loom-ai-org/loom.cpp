---
type: epic
status: active
domain: backends
last_updated: 2026-09-02
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
| Retros | [Retro-007](../retros/retro-007-gpu-chose-the-integrated-gpu.md), [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md), [Retro-009](../retros/retro-009-host-callback-count-was-the-wrong-lens.md), [Retro-026](../retros/retro-026-three-nodes-were-half-the-runtime.md) |
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


## 5. Metal (P4.11) — SHIPPED 2026-08-31, run on an Apple M1 Pro

`loom-py-rt-metal` builds, installs beside the base wheel, registers as a device, and produces
output an ASR model reads back identically to the CPU's. **The interesting part is not that it
works — it is that whether it is faster depends entirely on the model, by 14x.**

### 5.1 The four scoped questions, answered

1. **Discovery — inherited from P4.10's blocker 4, and there was nothing to fix.** ggml builds every
   backend as a CMake `MODULE`, and Darwin gives module libraries the `.so` suffix. So the shipped
   file is `loom_rt_metal/libggml-metal.so`, not `.dylib`, and ggml's extension-less-of-`__APPLE__`
   loader finds it. Full reasoning in [Epic-08 §4.1](epic-08-packaging-and-release.md).
2. **Linking is `install_name`/`@rpath`, and it DID need work.** The backend records
   `@rpath/libggml-base.dylib` as its `LC_LOAD_DYLIB`, and `@rpath` expands against the `LC_RPATH` of
   *the binary doing the loading* — this library. With none set there are no candidate paths at all.
   That is the real difference from ELF, where a `NEEDED libggml-base.so` resolves against whatever
   is already loaded in the process (and it always is, pulled in by `libloom_engine` long before any
   backend is dlopened) — which is why the Vulkan and CUDA packages need no rpath and have none.
   `packaging/common/BackendPackage.cmake` now sets `INSTALL_RPATH "@loader_path/../loom"` under
   `if(APPLE)`: `site-packages/loom_rt_metal/` → `site-packages/loom/`, siblings by construction.
   Verified with `otool -l` on the shipped wheel and then by loading it.
3. **`GGML_METAL_EMBED_LIBRARY` — confirmed by building.** It defaults on with Metal and embeds the
   `.metal` **source** through `.incbin`, compiled by the Metal framework at run time. **Full Xcode
   was not required**: `xcrun metal` appears only in the non-embed branch, and this machine has
   Command Line Tools only. Do not read a missing `metal` compiler as "cannot build Metal here".
4. **Size — measured, and it is the smallest backend by a wide margin.**

   | backend | shipped library | wheel |
   |---|---|---|
   | Vulkan | 46.5 MB (44 MB of it compiled SPIR-V) | ~46 MB |
   | CUDA | larger still (fat binaries, cubins per SM) | 88 MB |
   | **Metal** | **0.87 MB** | **0.18 MB zipped** |

   Metal ships shader *source*, not compiled kernels, so it is small for a structural reason rather
   than out of restraint.

### 5.2 It works, and the correctness bar was met

`pip install loom_py_rt_metal-...whl` beside the base wheel, from a clean venv:

```
devices: [('BLAS','Accelerate'), ('CPU','Apple M1 Pro'), ('MTL0','Apple M1 Pro')]
device="gpu" -> MTL0
```

So **device selection needed no new work**, as predicted: ggml-metal registers as a GPU-kind device
and P4.8e's hierarchy ranks it without a new tier.

Output agreement, on VITS: same sample count, **cosine 0.99999250**, max abs difference 1.55e-3 — and
because [correlation is not the test for a TTS family](../retros/retro-006-kokoro-shipped-noise.md),
whisper-small transcribed both waveforms and returned **byte-identical text**.

**P4.29 reconciles with Metal, and this was RUN rather than read.** The prediction was that
ggml-metal's `supports_op` would decline a folded, block-quantized `CONV_2D` on its
`src[0]->type == F16 || F32` test and keep the `im2col` lowering, as Vulkan does. It does. The 2x2 of
{F32, Q4_0} x {CPU, Metal} all four transcribe identically:

```
f32   cpu   peak=0.1416 -> ' This is a test of balloon energy running on Apple Silicon.'
f32   MTL0  peak=0.1418 -> ' This is a test of balloon energy running on Apple Silicon.'
q4_0  cpu   peak=0.1381 -> ' This is a test of balloon energy running on Apple Silicon.'
q4_0  MTL0  peak=0.1380 -> ' This is a test of balloon energy running on Apple Silicon.'
```

Q4_0 carries the *same* 27/56 split profile as F32 — the fallback is the `PAD` nodes, not the
quantization — and is faster on both devices (CPU 87.4 ms vs 94.1 ms; Metal 348.2 ms vs 493.3 ms).
**The standing caveat is unchanged and still untested:** Metal declines on the type test ALONE, where
CUDA was given an explicit `op_params[6] == 0` guard. That is sufficient only because loom builds a
packed node for a *quantized* kernel and never an F32 one. If that ever changes, Metal needs CUDA's
guard or it will read a `[IC*K, OC]` tensor as `[KW, KH, IC, OC]` and return a wrong answer rather
than an error.

### 5.3 The result that matters: faster or slower depending on the model

Measured on the M1 Pro, F32, best-of-n, same input both sides. **The split count is listed first
because it is where the graph went** — a first benchmark on a new backend measures how much of it
fell back. It is *not* where the time went; §5.4 and §5.7 are both about how far apart those two
questions turned out to be.

| model | shape | splits (CPU / total) | CPU | Metal | |
|---|---|---|---|---|---|
| whisper-small | encoder is large matmuls | **0 / 4** | 1764.3 ms | **654.9 ms** | **2.69x faster** |
| VITS (f32) | 1-D convolution throughout | **27 / 56** | 94.1 ms | 493.3 ms | **5.24x slower** |

**Both columns have since moved twice, and §5.8 carries the current pair.** The CPU side because
P4.30b made the default thread count this machine's eight physical cores rather than ggml's four; the
Metal side because P4.30a found what the 5.24x was and `ggml-0014` removed half of it, and then
P4.30d rewrote the convolution kernel as `ggml-0015`. Re-measured at HEAD on 2026-09-02, VITS f32 is
**54.6 ms on the CPU and 97.6 ms on Metal**, and whisper-small — a whole transcription of
`samples/jfk.wav` through `scripts/bench_asr_loom.cpp`, not the encoder alone — is **1.326 s on the
CPU and 0.752 s on Metal**. The shape of the finding is unchanged even though the gap is now a fifth
of what it was: **it depends on the model, and for a convolutional one Metal still loses.**

**The cause of VITS's fallback is one op.** Of 1308 nodes, only 48 land on the CPU — but they are
scattered, so each is a round trip. By op: **21 `PAD`**, 12 `CONT`, 6 `SUB`, 6 `SCALE`, 3 `DIV`.
`ggml-metal`'s `supports_op` accepts `GGML_OP_PAD` only when the **leading** pads are zero
(`op_params` 0, 2, 4 and 6), i.e. trailing padding only. A convolution pads both sides, so every one
of VITS's declines on that test. Where do they come from? The exported topology declares
`12 x PAD_1D {lp0: 1, rp0: 1}` (which fall back) and `6 x PAD_1D {lp0: 0, rp0: 1}` (which do not);
the remaining 9 are the rational-quadratic spline's own `ggml_pad_ext(..., 1, 0, ...)` and
`(..., 1, 1, ...)` in `primitives_spline.cpp`. **Every one of them has `lp0 == 1`** — a single
leading element, not an arbitrary pad.

**And then the fix was prototyped, which is the only reason that paragraph does not end in the wrong
conclusion.** See §5.4 — collapsing the fallback entirely is worth **1.8%**, not 5x.

### 5.4 The PAD fix, prototyped and measured: it is a correctness item, not a performance one

**Tracked as P4.30c step 5**, alongside the open question of where VITS's 5.24x actually comes from
(**P4.30a**) — which this is not.

The `supports_op` test is easy to relax and the kernel change is small, so rather than estimate the
payoff it was **built as a throwaway** on the M1 Pro (hand-edited in the build tree, measured,
reverted — nothing of it is in the repo). Four edits: four `lp0..lp3` fields on
`ggml_metal_kargs_pad`, filled from `op_params` 0/2/4/6 in `ggml_metal_op_pad`; `kernel_pad_impl`
mapping `dst[i0] <- src[i0 - lp0]` per dimension instead of `src[i0]`, copying when every index is in
range and zero-filling otherwise; and `supports_op` returning `true`. The `_4` vectorised variant
needs no thought because `is_c4` is already hardcoded `false` upstream ("note: this is slower").

The result:

| | splits (CPU / total) | Metal | digest |
|---|---|---|---|
| stock | 27 / 56 | 493.3 ms | `bbc58397d238efde` |
| with leading-pad support | **0 / 2** | **484.7 ms** | `bbc58397d238efde` |

**The hypothesis was right and the conclusion drawn from it was wrong.** `PAD` really is the only op
in VITS's graph that Metal declines: the other 27 CPU nodes (12 `CONT`, 6 `SUB`, 6 `SCALE`, 3 `DIV`)
are all supported for these shapes and were scheduler collateral, assigned to the CPU to avoid
copying around an unsupported neighbour. Remove the one op and the whole graph moves — 56 splits
become 2, none on the CPU. The identical digest also says the patched kernel is numerically right.

**And it buys 1.8%.** So the 27 round trips were nearly free, and the 5.24x is not the fallback.

**Where the 5.24x actually is: work, not overhead.** VITS's node count is fixed by its architecture,
so scaling the utterance scales the work inside each node while leaving the graph shape alone. If the
gap were per-dispatch overhead, Metal's cost per output sample would fall sharply as the utterance
grows. It does not:

| utterance | CPU µs/sample | Metal µs/sample |
|---|---|---|
| x1 (23.5k samples) | 1.355 | 7.247 |
| x2 | 1.358 | 7.013 |
| x4 | 1.341 | 6.846 |
| x8 (136k samples) | 1.702 | 6.693 |

Metal is flat within 8% across a 5.8x change in work. It is **compute-bound and simply ~5x more
expensive per unit of work on this graph**, which is a statement about kernels and lowering, not
about scheduling.

**The likeliest reason, written here as a hypothesis and since MEASURED WRONG in both halves — see
§5.7.** The hypothesis was: VITS is convolution-dominated (the vocoder profile puts the conv path at
92%), loom's CPU convolution carries **seven hand-written ggml patches** — `ggml-0004` through
`ggml-0007`, plus P4.13's kernel fold and P4.29's quantized direct convolution — "while its Metal path
runs stock `im2col` + `mul_mat`". Both halves of that last clause are false. At F32 ggml-metal does
not lower a convolution through `im2col` at all: it has a native `CONV_2D` kernel and runs it. And
where it *does* take the `im2col` route — the Q4_0 export, whose folded kernel it declines on its type
test — that route is the **faster** of the two. What was right is only the contrast with whisper-small,
which wins because it is large matrix multiplications, and the instruction that closes the paragraph:
**confirming it means a per-op profile on Metal, not more reasoning.**

**So the follow-up split in two**, and neither was a release blocker:

* The **PAD kernel** — worth doing for cleanliness and upstreamability (it is the same shape of gap
  P4.7d recorded for Vulkan's `PAD_REFLECT_1D`), and it removes a fallback that would matter more on
  a discrete GPU where a round trip crosses PCIe. On Apple it is worth 1.8%. Do not scope it as a
  performance fix. **Still open, as P4.30c step 5.**
* **Why Metal's convolution is 5x the CPU's here** — the item with the payoff in it, and the one
  that had to start with a per-op profile. **That is P4.30a, and §5.7 is its answer.**

### 5.5 The finding that was not scoped: `device=""` now picks the slower device

`device=""` means "decide for me", and on this laptop it resolves to `MTL0` — because the hierarchy
ranks a GPU above a CPU. That rule was written for a discrete GPU across a PCIe bus, where the ranking
is a proxy for "has its own fast memory". **Apple Silicon is unified memory**, so the proxy no longer
carries the thing it stood for: the win of moving work to the GPU no longer includes escaping a slow
bus, and what is left is per-dispatch overhead plus, here, 27 round trips.

The consequence is concrete: with Metal present, a default-device VITS run is **5.24x slower** than
the same call without it. (**That figure is P4.11's and is kept here because the rest of this section
reasons from it; it is 1.79x at HEAD** — `ggml-0014` and `ggml-0015` fixed two convolution kernels,
§5.7 and §5.8. The conclusion below is unchanged, which is the point: a smaller ratio in the same
direction still is not a rule.) This is a statement about the hierarchy on a unified-memory part, not
about Metal — and it is **why Metal is NOT in the base macOS wheel** even though 0.87 MB would otherwise be
a very cheap accelerator to ship by default. `loom-py/CMakeLists.txt` sets `GGML_METAL OFF` for the
base wheel explicitly, since ggml defaults it ON whenever `APPLE`; without that line the base wheel
and the backend package would each carry a copy, which is two on one `sys.path` with no rule about
which loads. Shipping it as an extra applies the trade to somebody who asked for a GPU.

Ranking on unified memory is on the hub as its own item. If it stops preferring a GPU whose memory is
the CPU's, folding Metal into the base wheel is worth revisiting.

**A related non-problem, checked rather than assumed:** ggml also defaults `GGML_BLAS` on for Apple,
so the base wheel ships a 59 KB `libggml-blas.so` (Accelerate) and `device=""` resolves to `BLAS`.
That splits VITS's graph 50 ways and costs **nothing measurable** — 94.3 ms against the CPU's
94.0 ms — because BLAS shares host memory, so a split is not a copy. It stays.

### 5.6 What is verified, and what is not

Verified on hardware: build, install, registration, `device="gpu"`, output agreement under an ASR
oracle, and timings on two model families. That is the P4.8c standard — CUDA stopped being a claim
when it ran on the workstation, and Metal stops being one here.

Not verified: **the wheel has not been published**, and **`cibuildwheel`'s `delocate` repair step has
not been run** — cibuildwheel refuses to system-install a framework CPython outside CI, so the local
wheels were built with `python -m build`. The `repair-wheel-command` in `packaging/rt-metal/
pyproject.toml` carries `--exclude libggml-base --ignore-missing-dependencies`, reasoned from how
delocate resolves an `@rpath` that points outside the wheel, and **first exercised by CI**.
A fourth PyPI project also needs its own trusted publisher before `publish-pypi`'s Metal step can
succeed. **Both were settled by the 1.0.0-rc7 release (2026-08-31):** `delocate` ran in CI for the
first time, and `loom-py-rt-metal` published its first release, so all four projects now sit at
`1.0.0rc7`.

**Non-goals, unchanged:** CoreML and the Neural Engine (not Metal, no ggml backend targets it, and
out-of-tree options were turned down on licensing); Intel Macs; a `universal2` backend wheel.

### 5.7 P4.30a — where the gap actually is: two convolution kernels, one of them a one-line bug

**SHIPPED 2026-09-02**, on the M1 Pro, at `7782a30`. The charter was "start with a per-op profile,
not with more reasoning", and the profile put **97% of Metal's VITS time in two ops** — one of which
has **three nodes in the whole graph** and was costing 45% of the run.

#### How to read a device profile, because the raw table cannot be read at face value

`$LOOM_PROFILE` on a scheduler costs a `ggml_backend_synchronize` per node (see
`include/loom/core/profile.h`), and on Metal that is not a rounding error: a profiled f32 VITS run is
925 ms against a 494 ms un-profiled one. **The calibration is free and exact, because the graph
contains its own control.** `RESHAPE`, `VIEW`, `PERMUTE` and `TRANSPOSE` compute nothing at all, so
whatever the report charges them *is* the per-node overhead: 611 `RESHAPE` at 122.13 ms is
0.1998 ms each, 167 `VIEW` at 32.17 ms is 0.1926, 112 `PERMUTE` at 23.64 ms is 0.2111. Subtract
0.196 ms x calls from every bucket and those three land on zero, and **the remaining buckets sum to
the un-profiled wall clock within 1%**. Do that before quoting a device profile; do not do it on the
CPU at more than one thread, where the overhead is a thread-pool barrier that lands unevenly and the
same subtraction drives no-op rows negative.

#### The profile, f32 VITS, calibrated

| op | calls | Metal, stock | Metal, `+ggml-0014` | CPU @ 1 thread |
|---|---:|---:|---:|---:|
| `CONV_2D` | 117 | **241.0 ms** | 241.3 ms | ~212 ms |
| `CONV_TRANSPOSE_1D` | **3** | **224.7 ms** | **7.8 ms** | ~21.6 ms |
| everything else | 2078 | ~28 ms | ~30 ms | ~62 ms |
| **wall (un-profiled)** | | **494.1 ms** | **278.7 ms** | **296.0 ms** |

**Three nodes were 45% of the run**, and nothing upstream of a profile would have said so: the op is
fully supported by ggml-metal, so it produces no split, no fallback and no warning. Ranking by node
count or by split count both miss it completely — which is the same lesson §5.4 learned from the
`PAD` prototype, one level further in. The hypothesis this replaced, and why it survived two weeks of
being right about the outcome and wrong about every fact in it, is
[Retro-026](../retros/retro-026-three-nodes-were-half-the-runtime.md).

#### `CONV_TRANSPOSE_1D`: one thread per threadgroup

`ggml_metal_op_conv_transpose_1d` dispatched `(OL, OC, 1)` threadgroups of **`(1, 1, 1)` threads**.
Every threadgroup therefore occupied a single lane of a 32-wide SIMD group, and 31 of every 32 lanes
on the GPU were idle by construction. Against `scripts/conv_census.py`'s arithmetic for the reference
utterance — VITS's three vocoder upsamples are **1.50 GFLOP** of the graph's 18.39 —

| | time | rate | of this part's 5.31 TFLOP/s |
|---|---:|---:|---:|
| Metal, stock | 224.7 ms | 6.7 GFLOP/s | **0.13%** |
| Metal, `+ggml-0014` | 7.8 ms | 193 GFLOP/s | 3.6% |
| **one** CPU core (`ggml-0008`/`0009`) | 21.6 ms | 69 GFLOP/s | — |

so the stock kernel had the entire GPU losing **10x to a single Firestorm core**.
`cmake/patches/ggml-0014-metal-conv-transpose-1d-threadgroup.patch` gives the x axis `nth` threads,
divides the grid by the same factor and bounds-checks the tail — the shape `ggml_metal_op_conv_2d`
directly above it already uses. Written up for upstream as PR 14 in
[`cmake/patches/UPSTREAM.md`](../../cmake/patches/UPSTREAM.md).

**Verified three ways.** `test-backend-ops test -o CONV_TRANSPOSE_1D` is **116/116 on MTL0** against
the CPU reference; the VITS waveform digest is **bit-identical** to stock on both exports
(`bbc58397d238efde` f32, `c5f02103027bcaee` q4_0); and the re-profile is a clean control — the
`CONV_TRANSPOSE_1D` bucket collapses 224.7 -> 7.8 ms while `CONV_2D` sits unchanged at 241 ms.

**It is not a VITS-only fix.** `scripts/conv_census.py` over the zoo: Kokoro **4**, StyleTTS2 **4**,
Matcha **5**, VITS **3**. Every vocoder-bearing TTS family hits it; the ASR families and Supertonic
do not use the op at all.

#### `CONV_2D`: 1.3% of peak, and Metal's own matmul proves the headroom

What is left is one op. After `ggml-0014`, **`CONV_2D` is 86% of Metal's VITS time** — 241 ms for the
other 16.89 GFLOP, i.e. **70 GFLOP/s, 1.3% of the part's peak, and slower than one CPU core.** The
kernel explains it without a microbenchmark: `kernel_conv_2d` is one thread per output element
looping `IC*KH*KW` with **two global loads per FMA and no reuse at all** — no register tile, no
threadgroup staging, no vectorisation. The occupancy is fine; the arithmetic intensity is 0.25
FLOP/byte.

**The Q4_0 export is the natural experiment that says this is fixable**, because there Metal declines
the folded block-quantized kernel on its type test (§5.2) and loom's `im2col` + `mul_mat` lowering
runs instead. Calibrated the same way:

| path | | |
|---|---:|---:|
| native `CONV_2D` (f32) | 117 calls | **241 ms** for 16.89 GFLOP — 14.3 ms/GFLOP |
| `IM2COL` + `MUL_MAT` (q4_0) | 126 + 150 calls | **99.7 + 10.5 ms** for ~15.7 GFLOP — 7.0 ms/GFLOP |

Two things fall out. **The arithmetic is not the problem** — `MUL_MAT` does the whole graph's
convolution in **10.5 ms, about 1.49 TFLOP/s, 28% of peak** (with Q4_0 weights, so an F32 matmul
would pay more weight traffic; `IM2COL` touches only F32 activations and carries no such caveat).
And **the data movement is** — `im2col`'s materialisation alone costs 10x the matmul it feeds.
Even so, the im2col route is ~2x better than the native kernel, which is why q4_0 on Metal is
149.7 ms against f32's 278.7 ms.

**The ceiling, stated as a ceiling** ([Retro-011](../retros/retro-011-chasing-the-gemm-and-convolution-gap.md)):
if `CONV_2D` ran at the rate Metal's own `MUL_MAT` already demonstrates, 16.89 GFLOP would take
~11 ms and VITS on Metal would land near the 8-thread CPU's 55 ms. That is the size of the prize and
it is an upper bound, not an estimate. The two candidate shapes are the two loom already knows: a
tiled implicit-GEMM kernel with threadgroup staging (what the CPU's `ggml-0004`/`0006` do), or lower
to `im2col` + `mul_mat` and accept the expansion traffic. **Neither is what worked — see §5.8, which
closes this as `ggml-0015`, and note before reading it that the 5.31 TFLOP/s every ratio above is
divided by is a SPEC number the part does not deliver.**

#### What this decides for the device hierarchy

It does not rescue it. With `ggml-0014` in, a default-device VITS run on this laptop is still
**5.07x slower** than the same call without Metal (f32; 2.91x at Q4_0), because the residual is a
kernel-quality gap and not a scheduling one. §5.5's conclusion stands unchanged — Metal ships as an
extra, `GGML_METAL OFF` stays in `loom-py/CMakeLists.txt` — and the unified-memory ranking item stays
open on the hub with a smaller, better-understood number attached to it. **§5.8 shrinks that number
again without changing the conclusion.**

### 5.8 P4.30d — the convolution kernel, and the register that was an address

**SHIPPED 2026-09-02**, on the M1 Pro, as
`cmake/patches/ggml-0015-metal-conv-2d-register-tile.patch`. §5.7 left one op at 86% of a VITS
synthesis and named two candidate shapes for fixing it. **Neither is what worked**, and the thing
that carried half the win is not a tile at all.

#### First, throw away the denominator

Every ratio in §5.7 is quoted against the M1 Pro's **5.31 TFLOP/s** spec. That number is not
reachable by any kernel here. A pure FMA loop with sixteen independent chains and no memory traffic
at all measures **2.11 TFLOP/s** on this part (`scripts/bench22.mm` prints it before anything else,
alongside a 180 GB/s streaming-read figure against a 200 GB/s spec). So the stock kernel's "1.3% of
peak" is really **3.6% of what a kernel can get**, and the headroom was always 28x, not 77x. This is
[Retro-011](../retros/retro-011-chasing-the-gemm-and-convolution-gap.md)'s rule applied one level up
— divide by a measured roofline before scoping — and a GPU is where it is easiest to skip, because
the spec sheet is published and looks authoritative.

#### The harness, because the model was the wrong loop

Iterating on a Metal kernel through a VITS synthesis costs a ggml rebuild — the Metal library is
embedded — plus a profiled run, per idea. `scripts/bench22.mm` runs **the same work** against a bare
Metal device instead: the 31 distinct convolution shapes `scripts/conv_census.py` reports for the
utterance `bench_vits_loom.cpp` pins, each at the multiplicity the graph issues it. **117 nodes,
16.884 GFLOP** — the `CONV_2D` row of §5.7's profile table exactly. It reproduces the stock kernel at
**222.3 ms / 76 GFLOP/s** against the model's 241 ms / 70, so a ratio it reports is a prediction about
the model, and a variant is a recompile of one `.mm`.

Every variant is checked before it is timed, against a two-level oracle: the stock kernel is the
reference for all 31 shapes, and the stock kernel is itself checked against a scalar CPU reference on
every shape small enough to afford one (a `L=73216 IC=32 K=7 OC=32` reference is half a billion MACs
and there are six of that size).

#### What the kernel does now

One thread accumulates **eight output channels at one output position**. Each activation it loads
feeds eight FMAs instead of one; the eight weights beside it sit at an address every lane in the
threadgroup shares, so one cache line serves the SIMD group. Loads per FMA: 2 -> 1.125. The grid
becomes the output's own shape — x over columns, y over blocks of eight channels, z over rows and
batch — and `KW` becomes a ggml function constant so the tap loop unrolls for the widths that occur.

#### The half of it that is not the tile

The same tile, written with one `device const TK *` per output channel, is **68.8 ms**. Written with
32-bit element indices off a single base pointer it is **45.6 ms**. Nothing about the arithmetic
changed; sixteen registers of pure address stopped competing with the accumulators.

`maxTotalThreadsPerThreadgroup` is the only register-pressure signal Metal exposes without a GPU
capture, and it reads the story directly: **704 threads for the pointer version, 896 for the index
version**, against 1024 for a kernel under no pressure. On this GPU occupancy *is* latency hiding,
and the kernel is latency-bound throughout — it issues 17 instructions per 8 FMAs and reaches 40% of
what that mix allows.

**That probe explains this gap and does not catch the next one.** Push past 32 accumulators per
thread — 8 positions x 8 channels — and the kernel runs **302.7 ms, 0.73x, slower than the stock
kernel it was meant to beat**, while `maxTotalThreadsPerThreadgroup` still reads 896. The compiler
spilled to scratch rather than raise the register count, and the probe reports register *pressure*,
not spilling. Only the clock finds that one.

Because 32-bit indices cannot address every tensor, the fast path declines — at threadgroup
granularity — a stride that is not a whole number of elements or a tensor past 4G elements, and falls
into the original 64-bit loop. That loop is kept as the general path and also carries the padded
edges and partial channel tiles, so declining is always *correct*, not merely safe.

#### The sweep, in the order the wins landed

All rows from one run, because a table assembled from several is not a table:

| | total | rate | occupancy | |
|---|---:|---:|---:|---:|
| stock | 222.3 ms | 76 GFLOP/s | 1024 | 1.00x |
| + register tile, 64-bit pointers | 68.8 ms | 245 GFLOP/s | 704 | 3.23x |
| + 32-bit element indices | 45.6 ms | 370 GFLOP/s | 896 | 4.88x |
| + `KW` as a function constant | 44.8 ms | 377 GFLOP/s | 896 | 4.96x |
| + 128 threads per threadgroup | **43.6 ms** | **387 GFLOP/s** | 896 | **5.10x** |

**5.10x on the op set, and 18% of the measured roofline** against the stock kernel's 3.6%.

#### What did not work, which is the part worth keeping

* **Threadgroup staging — the shape §5.7 named first — is worse at every activation length.**
  Staging the weight tile into threadgroup memory and reading it as a broadcast gives **55.2 ms**
  against 43.6. The uniform global loads were already being served by one cache line per SIMD group,
  so there was nothing to save, and the barriers are not free.
* **More output positions per thread loses, despite a much better ratio — and that is the trap in
  reasoning about loads per FMA at all.** A 4-position x 4-channel tile has **0.5 loads per FMA
  against the shipped kernel's 1.125**, less than half, and runs **69.7 ms**: it gives back more
  occupancy than the ratio buys. The exception is real but small: on the two longest activations a
  2-position x 8-channel tile runs 6.56 and 7.38 ms against 7.71 and 8.69, about 6% of the whole set.
  It is **left on the table deliberately**, because taking it costs a second pipeline family and a
  threshold on `OW`.
* **The `im2col` + `mul_mat` route is no longer the fast one.** §5.7's natural experiment ran it at
  7.0 ms/GFLOP against the native kernel's 14.3; this kernel is at 2.6.

#### End to end, and the control

| op | calls | Metal `+ggml-0014` | Metal `+ggml-0015` | CPU @ 1 thread |
|---|---:|---:|---:|---:|
| `CONV_2D` | 117 | 241.3 ms | **60.1 ms** | ~212 ms |
| `CONV_TRANSPOSE_1D` | 3 | 7.8 ms | 7.7 ms | ~21.6 ms |
| everything else | 2078 | ~30 ms | ~22 ms | ~62 ms |
| **wall (un-profiled)** | | **278.1 ms** | **97.6 ms** | 296.0 ms |

The re-profile is the same clean control §5.7 used: the patch touched one op and one bucket moved.
Both profile columns are calibrated against the graph's own no-op nodes (§5.7's method; the per-node
overhead here is 0.180 ms, from 872 `RESHAPE`/`VIEW`/`PERMUTE` executions that agree to 0.02 ms).
The calibrated buckets sum to 89.4 ms against a 97.6 ms wall — a looser fit than §5.7's 1%, because
the run is now short enough that a 378 ms profiling overhead dominates what is being subtracted.

**Correctness.** `test-backend-ops test -o CONV_2D` is **2026/2026 on MTL0** against the CPU
reference — F16 and F32 kernels, `KH > 1`, stride, dilation, padding and batch, none of which VITS
exercises. The VITS waveform digest is **bit-identical** to stock on both exports (`bbc58397d238efde`
f32, `c5f02103027bcaee` q4_0). `ctest -L ci` is 75/75.

#### It is not a VITS-only fix, and it moves whisper too

whisper-small's encoder opens with two convolutions over 3000 frames, and it gains without anything
else changing: **0.874 -> 0.752 s**, so Metal goes from 1.52x to **1.76x faster than the CPU** on
that model.

#### What this decides for the device hierarchy — still not rescued, and a new inversion

A default-device VITS run on this laptop is **1.79x slower** than the same call without Metal at f32,
down from 5.07x. Smaller, same conclusion: §5.5 stands, Metal ships as an extra, `GGML_METAL OFF`
stays in `loom-py/CMakeLists.txt`.

**But the Q4_0 column is now the slow one, and that is new.** Metal declines loom's folded
block-quantized convolution kernel on its type test (§5.2), so a Q4_0 export lowers through
`im2col` + `mul_mat` — untouched by this patch, and now **1.53x slower than the same model at f32**
(149.1 ms against 97.6). Before `ggml-0015` the ordering was the other way round, at 149.7 against
278.7. Quantizing a convolutional model for Metal now costs time as well as accuracy, which is a
reason to look at that type test rather than at the quantization. Filed on the hub.

#### Current numbers, for anyone re-running this

M1 Pro (16-core GPU), macOS 15.7.9, `7167822` + `ggml-0015`, `scripts/bench_vits_loom.cpp` median of
9 and `scripts/bench_asr_loom.cpp` median of 5, interleaved ABBA with a settle between arms.
`ctest -L ci` on this build is 75/75. Achievable rooflines on this part, measured by
`scripts/bench22.mm`: **2.11 TFLOP/s** F32 FMA, **180 GB/s** streaming read.

| | CPU (8 threads) | Metal, stock | `+ggml-0014` | `+ggml-0015` |
|---|---:|---:|---:|---:|
| VITS f32 | **54.6 ms** | 494.1 ms (9.05x) | 278.1 ms (5.09x) | **97.6 ms (1.79x)** |
| VITS q4_0 | **51.4 ms** | 350.2 ms (6.81x) | 149.7 ms (2.91x) | 149.1 ms (2.90x) |
| whisper-small, `jfk.wav` | 1.326 s | — | 0.874 s (1.52x faster) | **0.752 s (1.76x faster)** |
