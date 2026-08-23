---
type: adr
status: accepted
date: 2026-08-13
tags: [packaging, ggml-backend-dl, wheels, leanness, accelerators]
---

# ADR-009: Backends Are Dynamic Libraries and Separate Packages, Not Build Variants

## Context

Sixteen backend directories ship in the pinned `ggml` — CUDA, Metal, SYCL, OpenCL, HIP, OpenVINO,
Hexagon, CANN among them. Because `loom::Device` resolves a spec against `ggml`'s *device registry*
rather than against any backend name it knows, a CUDA build's `CUDA0` is already selectable by code
that has never heard of CUDA. Adding one costs a build matrix and a test run, not C++.

The problem is what that does to the artifact. This engine exists to be lean enough for edge devices,
and compiling every backend in would end exactly the property it is for. The sizes make it a constraint
rather than a preference: a backend library is tens of megabytes against an engine measured in
single-digit ones.

## Options Considered

1. **Compile all backends into one binary.** One artifact, and it ends the leanness the engine is for.
2. **A wheel matrix** — one wheel per (platform, accelerator) combination. Combinatorial, and it makes
   the accelerator a property of the base install rather than of the deployment.
3. **`GGML_BACKEND_DL`**: each backend is a shared library `ggml` discovers at run time; one engine
   binary serves every accelerator, and the deployment decides which files travel with it.

## Decision

**`GGML_BACKEND_DL`, with accelerators as separate installable packages.** The base wheel carries the
engine and its CPU backend; `loom-py-rt-cuda`, `loom-py-rt-vulkan` and friends carry one backend each
and pin the base with `==`. Extras compose, and **an extra that finds no device should fail loudly** —
the same decision `Device::open("gpu")` already makes in raising rather than silently falling back.

This works through `loom::Device` unchanged.

## Consequences

* **Positive:** one engine binary, any accelerator, and the deployment chooses. Measured cost of a
  backend is 46–59 MB — carried only by installs that want it.
* **Positive:** a new tier-1 backend really is a CMake flag plus a test run.
* **Negative:** backend packages are **arch-specific too**, so the matrix does not disappear — it moves
  off the base wheel onto the accelerator packages.
* **Negative:** DL loading is a real code path with real failure modes. `import loom` started printing
  to stderr because `ggml` logs each backend it loads at INFO (now filtered to WARN/ERROR); a wheel is
  a zip and **a zip cannot carry a symlink**, which `ggml`'s versioned `.so` names assume; and the DL
  loader searches for `.so` where CMake writes `.dylib` on macOS.
* **Negative:** the gate suite must run against the DL configuration, not just a static build — see
  [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md).

## Related

* Epic: [Epic-04: Backends and Accelerators](../epics/epic-04-backends-and-accelerators.md)
* Epic: [Epic-08: Packaging and Release](../epics/epic-08-packaging-and-release.md)
* Ledger record, verbatim:


P4.7 got the engine onto a GPU and stopped at the one device this machine has: an AMD iGPU through
Vulkan. That was a hardware accident, not a decision. What follows is the shape of the rest — CUDA
next, then NPUs — and, because "compile in every backend" and "the engine targets edge devices" cannot
both hold, how a build stays as small as the box it ships to.

**Nothing here is started. Every number in it is a count of what upstream ships, not a measurement.**

### The good news first: tier 1 costs a CMake flag

`ggml` v0.16.0, the revision this repo already pins, ships **sixteen** backend directories: `ggml-cuda`,
`ggml-metal`, `ggml-vulkan`, `ggml-sycl`, `ggml-opencl`, `ggml-hip`, `ggml-musa`, `ggml-blas`,
`ggml-rpc`, `ggml-webgpu`, `ggml-cann` (Ascend), `ggml-hexagon` (Qualcomm), `ggml-openvino`,
`ggml-zdnn`, `ggml-zendnn`, `ggml-virtgpu`.

Two of the four NPU targets named for this item are already in there — **OpenVINO** outright, and
**Qualcomm** as `ggml-hexagon` rather than as the out-of-tree `ggml-qnn`. Which of those two is the
right Qualcomm path is a real question and is NOT answered here; in-tree costs nothing to try, so it
should be tried first.

**And for every one of them, the engine needs no work.** That is what P4.7's device layer bought, and it
is worth being explicit about why: `loom::Device` resolves a spec against the ggml *device registry*,
never against a backend name it knows. A CUDA build's `CUDA0` is selectable by `--device cuda0` today,
by code that has never heard of CUDA. `GGML_BACKEND_DEVICE_TYPE_ACCEL` — what an NPU registers as — is
already in `is_offload_device()`. The work in tier 1 is a build matrix and a test run, not C++.

### Tier 2: out of tree, and priced accordingly

**CoreML**, **RKNPU2** (`ggml-rknpu2`, and the `rk-llama.cpp` fork), and `ggml-qnn` if it beats
`ggml-hexagon` are not in the pinned ggml. Each would mean vendoring a backend or carrying a ggml fork,
and this repo's dependency policy (`Dependencies.cmake`) is pinned FetchContent of upstream, precisely
so it never owns somebody else's tree. Before any of them: **check the licence.** The project is MIT and
has already turned down a dependency over exactly this (Task #79, espeak-ng's GPL-3). A vendored
backend that cannot be shipped under MIT is not a cheap dependency, it is a relicensing decision.

Note also that CoreML is not Metal. Metal is the GPU and is in tier 1; reaching the **Neural Engine**
means CoreML, and no ggml backend targets it. That is a bigger piece of work than the others, not a
sibling of them.

### The leanness answer, and it is already verified

`GGML_BACKEND_DL`. Each backend becomes a shared library that ggml discovers at RUN time, so one engine
binary serves every accelerator and the deployment decides which `.so` files travel with it. Measured on
this machine (`-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON`):

* it works through `loom::Device` **unchanged** — `Device::open` already calls `ggml_backend_load_all()`,
  and `loom_cli --list-devices` reports a dynamically loaded backend exactly as it reports a linked one;
* the CPU becomes a plugin too, and splits into per-microarchitecture variants (`libggml-cpu-haswell.so`,
  `-zen4`, `-sapphirerapids`, …) with the best picked at load time — which is a second, unrelated win:
  one artifact stops being compiled for one `-march`;
* discovery order is `GGML_BACKEND_DIR` (compile-time), the executable's directory, the current
  directory, and `$GGML_BACKEND_PATH` (a specific file);
* **with no `.so` found, the registry is EMPTY — there is no CPU either.** Every spec, `"cpu"` and
  `"auto"` included, fails. `loom::Device` now says so in as many words rather than reporting "ggml
  reports no CPU device", because a deployment that forgot to ship its backends needs to be told that is
  what happened.

So the guard this item asks for is mostly not a new invention: it is `GGML_BACKEND_DL` plus a decision
about what each artifact carries. What IS still needed on the engine side is small and namable:

1. **A `Backends` that holds more than two.** `Device` initializes exactly one device backend and hands
   `ggml_backend_sched` a pair. Two GPUs, or a GPU *and* an NPU, needs a list — `ggml_backend_sched`
   takes N backends already (CPU last), so this is `loom::Backends` growing a vector and `GraphBuilder`
   passing it through, not a redesign.
2. **A device-selection story for more than one match.** `"gpu"` means "the first GPU/iGPU/accelerator
   registered". With two accelerators of different kinds in a box that stops being a sensible default.
3. **Op coverage decides whether an accelerator is worth using at all, and the NPU backends are the
   sharp end of it.** P4.7 measured 453 splits costing Qwen3 its entire speedup on a GPU that supports
   nearly every op; P4.7a–d cleared them and the zoo now runs at 1–3 splits. The support matrix in
   P4.7d is the warning for this item: of `PAD_REFLECT_1D`, `POOL_1D` and `POOL_2D`, **OpenCL, OpenVINO
   and Hexagon implement none — not even `POOL_2D`**, which every GPU backend has. So a first NPU
   benchmark will not measure the NPU; it will measure how much of the graph fell back. Budget for
   the coverage work before drawing any conclusion from a number.

   P4.7d's closing section also settles where that work goes: **portability lowerings in the engine,
   keyed on `ggml_backend_supports_op`; op-recognition fusions in the exporter.** An NPU backend will
   need many of the former, and doing them the exporter way would mean an artifact compiled for the
   least capable backend anyone might run it on.

### `loom-py`: profiles, and why a wheel matrix is the wrong first instinct

The axes are architecture (x86-64, arm64) × libc/OS (manylinux, macOS, Windows) × accelerator (none,
CUDA, Metal, Vulkan, OpenVINO, Hexagon, RKNPU2, CoreML). The full cross product is not a plan, and two
of the combinations people ask for collapse on inspection: **Metal is Apple-only**, so "Arm + Metal" and
"Apple Silicon + Metal" are one profile; **Arm + CUDA** is real but means Jetson/Grace, which is a
distinct manylinux variant rather than a flag.

PyPI wheel tags encode architecture and libc but have **no accelerator dimension**, so an accelerator
has to be expressed as either a package-name suffix (torch's `cu121` shape — a full wheel per
accelerator, which is the combinatorial matrix) or as something loaded at run time. `GGML_BACKEND_DL`
makes the second possible: **one arch-tagged base wheel, plus small `loom-py-rt-<backend>` packages that
drop a `.so` where ggml looks.** `pip install loom-py-rt[cuda]` then means "also fetch that backend",
`Model(..., device="auto")` finds it, and a Raspberry Pi installs nothing extra and gets the CPU
variants it already had.

**The sizes make this a constraint rather than a preference**, which is the part that was argued from
combinatorics before it was measured. Release builds, stripped, on this machine:

| | |
|---|---|
| `libloom_engine.so` | **1.2 MB** |
| `libggml-base.so` + `libggml-cpu.so` | 1.7 MB |
| a CPU-only deployment | **≈ 3 MB** |
| `libggml-vulkan.so` | **46.5 MB**, of which 44 MB is `.rodata` — 1785 compiled SPIR-V shaders |
| the same deployment with Vulkan | **≈ 50 MB** |

Two things follow. First, **compiling backends in never threatens the leanness this repo means by the
word**: `CLAUDE.md`'s leanness is about CODE — per-model complexity belongs in the exporter — and
`libloom_engine.so` is byte-identical whether ggml ships one backend or nine. It is 3% of a Vulkan
deployment. What grows is the ARTIFACT, and every byte of that growth is somebody else's precompiled
kernels. Second, ~50 MB sits against PyPI's 100 MB default per-file ceiling before anything else is
added, and CUDA — whose fat binaries carry cubins per SM architecture — clears it outright (not measured
here; the mechanism is well known and is why torch hosts `cu121` off-index). A wheel matrix is therefore
not merely inelegant, it does not fit.

### The backend packages are arch-specific too

A backend package ships native code, so the combinatorics do not vanish — they **factor**. The base is
`arch × os`; a backend is `arch × os × backend`. The artifact COUNT is comparable; what changes is that
each artifact is 1–50 MB rather than a full duplicate of everything, each builds independently, and a
new backend does not force a re-release of the base wheel on every platform.

And the practical matrix is far sparser than the cross product, because most backends exist on one or
two architectures at all:

| backend | worth building for |
|---|---|
| Vulkan | x86-64, aarch64 — the broadest |
| CUDA | x86-64, aarch64 (Jetson/Grace) |
| Metal | arm64 macOS only |
| Hexagon / QNN | aarch64 |
| RKNPU2 | aarch64 |
| OpenVINO | x86-64 |

Roughly nine backend wheels, not backends times every platform. **No tag abuse is involved**: the
package NAME carries the accelerator dimension and the wheel TAG carries architecture, which is what
each was designed for, and `pip` resolves the right arch itself.

### Extras compose, and one of them should be allowed to fail

`pip install "loom-py-rt[hub,vulkan]"` is ordinary PEP 508 and works. But the two extras are different
in kind and the difference is worth being deliberate about rather than discovering:

* `[hub]` is a pure-Python feature toggle (`huggingface_hub`) — architecture-independent, always
  resolvable, additive.
* `[vulkan]` is a hardware toggle — native, architecture-specific, and **may not exist for the
  platform at all**.

Which decides one case: `pip install loom-py-rt[metal]` on Linux. There is no Metal wheel for manylinux,
so pip fails to resolve it. The alternative — environment markers, so `[metal]` quietly resolves to
nothing off macOS — lets the install succeed and hands back no Metal.

**Take the loud failure.** It is the same decision `Device::open("gpu")` already makes in raising rather
than falling back to the CPU (P4.7): a caller who spelled out the accelerator is asserting something
about the machine, and finding out at install time beats finding out from an unexplained performance
number later. Install-time loudness matching run-time loudness.

Two corollaries: there should be **no `[all]` extra** — that is the wheel matrix wearing a hat — and
`[vulkan,cuda]` together is legitimate, because with `GGML_BACKEND_DL` the registry picks at run time.

### Two costs to price before adopting any of this

1. **The base wheel has to go SHARED, reversing a deliberate decision.** `GGML_BACKEND_DL` refuses to
   configure without `BUILD_SHARED_LIBS=ON`, and loom-py currently forces it OFF on purpose: `_loom.so`
   statically folds in the engine and ggml so that its only external dependencies are
   libc/libstdc++/libgomp/libm. Its CMakeLists comment explains why — a shared build "produces an import
   that fails the moment the wheel leaves this build tree". The answer is `$ORIGIN` RPATH with
   `libggml-*.so` shipped beside `_loom.so`, which is routine wheel practice (auditwheel does exactly
   this), but that comment records a real failure and needs rewriting with the new reason rather than
   deleting.
2. **Backend packages need an exact `==` pin on the base version.** A backend `.so` links
   `libggml-base.so.0`, and ggml offers no ABI guarantee across versions — so any loom-py release that
   bumps its ggml pin invalidates every previously published backend wheel. A compatible-release range
   would silently pair mismatched libraries.

Either way the *engine* side is the same work, which is why this is one item and not two.

### What would make this item startable

A machine with the hardware, or CI that has it. Every claim above about tier 1 is "upstream ships a
directory"; none of it is "we ran it". The honest first step is CUDA on a box that has an NVIDIA GPU —
it is the most used, it is in tier 1, and `tests/gate/test_e2e_device_parity{,_kv}.cpp` are written
against "the first non-CPU device" and would run against it unmodified. That test passing on a second
backend is the evidence that the device layer generalizes; until then, it is a claim.

### That machine now exists, and its toolchain is ready (2026-08-14)

The condition above is met. A **workstation on the LAN** (`ssh 192.168.1.100`, passwordless from the
dev box) carries an **RTX 5090** (32 GB, driver 580.105.08) *and* an **Intel NPU** (Arrow Lake,
`intel_vpu`, `/dev/accel/accel0`), on 24 cores against this dev box's 4 — so a full suite build drops
from 20–40 minutes to a few, which is reason enough to build there for ordinary work and not only for
CUDA.

Toolchain, installed and verified into `/opt/mamba/envs/py-3.12` (micromamba; note the envs are in
`/opt/mamba/envs`, not `~/micromamba/envs`): **CUDA 12.8**, CMake 4.4.2, Ninja 1.13.2.
`nvcc -arch=sm_120` compiles — 12.6, which was there before, answers `Value 'sm_120' is not defined`,
because Blackwell needs 12.8+. Build with `-DCMAKE_CUDA_ARCHITECTURES=120`: it is the only GPU there,
and a multi-arch build is larger for nothing.

Three things worth knowing before spending time on them:

* **Invoke through `micromamba run -p ...`, always.** Conda's `cuda-nvcc` ships an activation script
  setting `NVCC_PREPEND_FLAGS=-ccbin=<env>/bin/x86_64-conda-linux-gnu-c++`, which points nvcc at the
  env's own gcc 12.4. Calling nvcc by absolute path skips it, picks up the system gcc 14.2, and fails
  with "gcc versions later than 13 are not supported" — a false alarm that looks like a broken
  toolchain.
* **`/home` there is 98% full (~34 GB).** micromamba's package cache is `/home/flavio/.conda/pkgs`, on
  that filesystem, even though the envs live on `/` — so installs eat the tight one. `micromamba
  clean -t` reclaims tarballs safely. The `v4` fixture set is 13 GB; copy only the GGUFs a given gate
  needs rather than the tree.
* **CMake 4 turned out to be fine.** `nlohmann_json` declares `cmake_minimum_required(VERSION
  3.1...3.14)` and CMake 4 removed <3.5 compatibility, but the range's upper bound governs policy, so
  it configures. A full build under CMake 4 is still untested; `cmake=3.31` or
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` if it misbehaves.

**And the first measurement to take there is not CUDA.** P4.8b's rank 1 — "an accelerator with its own
memory" — rests on an assumption no hardware has ever tested: that a discrete NPU registers as
`GGML_BACKEND_DEVICE_TYPE_ACCEL` with a non-host buffer type. The only two devices ever measured are
BLAS (ACCEL, host memory) and Vulkan (IGPU, device memory). If the NPU reports host memory it belongs
in rank 2; if it is not ACCEL at all, `Device::open("npu")` never resolves and that spec is dead code.
`tools/debug/probe_tiers.cpp` answers it in a minute and could send the ranking back to the drawing
board, so it is worth answering before building anything on top of it.

### P4.8a — the packaging half, built and verified (2026-08-14)

The half of P4.8 that needed no hardware is done, and it is the half CUDA blocks on: a CUDA wheel is
impossible until the base wheel is shared, and that reversal has nothing to do with which accelerator
ships. **`loom-py` now builds with `GGML_BACKEND_DL`**, and the pilot accelerator package exists.

What the engine gained is one function. `ggml`'s own backend search looks in the executable's
directory and the current directory, and **inside a Python interpreter the executable is `python`** --
so a DL-built wheel would have found no backends at all, the CPU included, and every device spec
including `"cpu"` would have failed. `loom::add_backend_search_path()` (`core/backend.h`) is how a host
says where its backends actually are; host directories are swept before ggml's defaults, and sweeping
is repeatable rather than once-only because ggml dedupes on the registration pointer. The pre-existing
`ensure_backends_loaded` became that sweep. `tests/ci/test_device_selection.cpp` gained the invariant
that matters -- a stale, blank or twice-added path never costs the registry a device.

On the loom-py side the base wheel is shared, `$ORIGIN`-RPATH'd, and ships `libloom_engine.so` plus the
ggml family beside `_loom.so`; `loom/__init__.py` registers `$LOOM_BACKEND_DIR`, then the package
directory, then every `loom_rt_*` package on `sys.path` (a directory scan, not an import, so a broken
accelerator package cannot take the base package down). `packaging/rt-vulkan/` is the pilot, and
`packaging/common/BackendPackage.cmake` is the part CUDA reuses -- a CUDA package is that directory
with two strings changed. `cmake/GgmlPin.cmake` is new here: the ggml revision now lives alone in a
file so a backend package builds against the base's exact revision with **no second copy of the tag to
drift**, which is the build-side half of the `==` version pin.

**Verified end to end, which was the point.** The old static build's CMake comment recorded a shared
build producing "an import that fails the moment the wheel leaves this build tree" -- so the test is
exactly that: a real wheel, a clean venv, a neutral working directory. It imports, finds its backends,
and runs 49/49 CI tests from site-packages. The CPU arrives as a per-microarchitecture plugin chosen at
load time (`libggml-cpu-haswell.so` on this Zen+ box), which is the second, unrelated win --
one artifact stops being compiled for one `-march`.

Two things this turned up that were not in the scoping.

**Going DL made `import loom` print to stderr.** ggml logs each backend it loads at INFO, and a static
build had nothing to load, so this was new noise introduced by the packaging change rather than
something the engine always did. The binding now installs a log callback that drops INFO/DEBUG and
**keeps WARN/ERROR** -- those are the messages explaining a backend that loaded and then found no
usable device, which is the exact failure `loom.devices()` (also new) exists to make visible.

**A wheel is a zip, and a zip cannot carry a symlink.** ggml's libraries are built with
VERSION/SOVERSION, so `libggml-base.so -> .so.0 -> .so.0.16.0` materialised as three byte-identical
copies of an 880 KB library, and `libggml.so` as three more; 2.6 MB of duplication, which the zip's
per-entry compression turned into 737 KB of wheel (7.79 MB -> 7.06 MB, measured both ways).
The properties are now unset for the packaged build, which is
correct on the merits too: a soname exists to let versions coexist for independently built consumers,
and nothing here is independently built -- the libraries ship in one directory and a backend package
pins the base with `==` precisely because ggml's ABI will not tolerate the mixing a soname would allow.
Worth recording the trap: `set_target_properties(... VERSION "")` does NOT clear it. An empty version
is still a version, the library comes out named `libggml-base.so.` with a trailing dot, and the wheel
gains a fourth copy instead of losing two. `set_property(TARGET x PROPERTY VERSION)` with no value is
the spelling that unsets.

