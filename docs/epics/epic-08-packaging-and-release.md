---
type: epic
status: active
domain: packaging
last_updated: 2026-09-09
---

# Epic-08: Packaging and Release

## 1. Context and Scope

Three repositories, one published Python distribution, and a set of optional accelerator packages.
This epic covers repository layout, wheel building, the accelerator package family, and the release
procedure.

## 2. Architectural Overview

### Repositories

| repo | what it holds |
|---|---|
| `loom.cpp` | the `ggml` engine, its primitives, its Lua bridge |
| `loom-exporter` | turns a PyTorch checkpoint into a GGUF this engine runs |
| `loom-py` | Python bindings; vendors `loom.cpp` as a submodule at `vendor/loom.cpp` |

All three under `github.com/loom-ai-org`, side by side on disk.
**The ledger and knowledge base stay in `loom.cpp` and cover all three** —
[ADR-011](../adrs/adr-011-three-repositories.md).

### Wheels

`loom-py` ships a `GGML_BACKEND_DL` base wheel: `manylinux_2_28` on x86-64 and aarch64, CPython
3.10–3.13. Accelerators are **separate packages** that pin the base with `==` —
[ADR-009](../adrs/adr-009-backends-as-dynamic-libraries.md). A backend costs 46–59 MB, carried only by
installs that ask for it.

**Two things that cost real time and will again:**

* **A wheel is a zip, and a zip cannot carry a symlink** — `ggml`'s versioned `.so` names assume one.
* **A "two strings changed" second package is not verified until it is built.** The Vulkan pilot
  declared a `readme` that did not exist and failed metadata generation before ever reaching CMake;
  the end-to-end verification that was cited for it had actually been of the *base* wheel.

The CUDA package settled the toolkit version rather than the other way round: `nvidia-*-cu13` does not
exist, which chose CUDA 12.9. The wheel fits PyPI's 100 MB per-file ceiling at **88 MB with more
architectures** than the version that did not fit.

### Release

A version bump is **seven version strings across three files**. The publish path is proven end to end
to PyPI.

**Wheels published before 2026-08-14 silently require AVX2** — worth knowing before debugging an
illegal-instruction report from an older install.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-011](../adrs/adr-011-three-repositories.md), [ADR-009](../adrs/adr-009-backends-as-dynamic-libraries.md), [ADR-025](../adrs/adr-025-armv6-is-built-in-its-own-emulated-userland.md), [ADR-026](../adrs/adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) |
| Retros | [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md), [Retro-024](../retros/retro-024-a-blocker-read-from-one-half-of-an-agreement.md), [Retro-033](../retros/retro-033-a-shared-library-links-clean-without-its-symbols.md), [Retro-034](../retros/retro-034-the-boards-own-libstdcxx.md), [Retro-035](../retros/retro-035-the-emulator-said-it-was-an-arm10e.md), [Retro-036](../retros/retro-036-one-switch-two-decisions.md), [Retro-037](../retros/retro-037-ps-said-six-percent-top-said-fifty.md), [Retro-038](../retros/retro-038-two-panels-on-one-cache-way.md) |
| Active tasks | [Backlog → Packaging](../backlog/active-index.md#packaging--release) |

## 4. macOS wheels (P4.10) — SHIPPED 2026-08-31, verified on an M1 Pro

Apple Silicon and Apple Intel wheels are wired into `wheels.yml`, and the arm64 one has been built,
installed from a clean venv and used to run two models on real hardware. **What follows is the
record, not a plan** — the scoping that preceded it is in §5.

**Why it went before P5.** P5 adds model families, and a family lands in `loom-py` for free — a model
the bindings have never heard of works the day the exporter can produce it. That is true only on a
platform that exists, so every model P5 adds was unreachable from a Mac until this landed. It
multiplies P5's value while P5 does nothing for it.

**The two Apple targets.** Apple Silicon tags `macosx_14_0_arm64` (`arm64` is Apple's name for the
ISA Linux calls `aarch64`); Apple Intel tags `macosx_14_0_x86_64`. **Two native wheels, not
`universal2`** — a fat binary doubles the download for everyone to serve one half, and that half is
ending. Apple Intel remains **CI-only**: "supported" there means built and imported by a runner, and
the platforms table says so rather than implying parity.

**The variant ladder needed no work, as predicted.** ggml's `GGML_CPU_ALL_VARIANTS` has an
`elseif (APPLE)` arm giving `apple_m1` (DOTPROD) / `apple_m2_m3` (+I8MM) / `apple_m4` (+SME). Every
Apple Silicon part has dotprod, so the lowest rung is not a compromise baseline the way Linux-ARM's
`armv8.0_1` is. All three build, and `tests/ci/test_cpu_variants.py` now carries an `arm64` row and
confirms **`libggml-cpu-apple_m1.so` is the one an M1 Pro selects**.

### 4.1 The four scoped blockers: three were real, and the fourth did not exist

| # | scoped as | what it actually was |
|---|---|---|
| 1 | LuaJIT's Makefile hard-errors on Darwin without `MACOSX_DEPLOYMENT_TARGET` | **REAL.** First thing the build hit. Fixed in `cmake/Dependencies.cmake`. |
| 2 | `loom-py`'s install rule matches `*.so*`, so a macOS wheel ships `_loom.so` alone | **REAL**, and half of it. Fixed by matching both suffixes. |
| 3 | `$ORIGIN` is ELF-only; macOS needs `@loader_path` | **REAL.** Fixed in `loom-py/CMakeLists.txt`. |
| 4 | ggml's DL loader searches for `.so` where CMake wrote `.dylib` → zero devices, silently | **DID NOT EXIST.** See below. |

**Blocker 4 is the one worth reading, because the reasoning that produced it was sound and the
conclusion was still wrong.** `ggml-backend-reg.cpp`'s `backend_filename_extension()` really does
return `.dll` on `_WIN32` and `.so` otherwise, with no `__APPLE__` case — that was read correctly at
the pin. The missing half was **what kind of target a backend is**. ggml builds each one with
`add_library(${backend} MODULE ...)`, and on Darwin CMake gives *module* libraries the suffix `.so`
(`CMAKE_SHARED_MODULE_SUFFIX`) while only *shared* libraries get `.dylib`
(`CMAKE_SHARED_LIBRARY_SUFFIX`). So the loader and the build already agree, and they agree on `.so`.

Confirmed by building it. The runtime directory of a macOS build:

```
libggml-base.dylib   libggml.dylib   libloom_engine.dylib      <- SHARED, linked, .dylib
libggml-cpu-apple_m1.so   libggml-cpu-apple_m2_m3.so           <- MODULE, dlopened, .so
libggml-cpu-apple_m4.so   libggml-blas.so   libggml-metal.so
```

and `loom.devices()` reports a CPU. **The takeaway is that a loader and a build are two halves of one
agreement, and reading one half predicts nothing.** No `SUFFIX ".so"` override was needed, in our
build or upstream; the fix that was scoped would have been a no-op papering over nothing. It is also
why blocker 2 is *asymmetric* and not merely "add `.dylib`": the linked libraries are `.dylib` and
the backends beside them are `.so`, so the install rule has to match both.

### 4.2 Three blockers nobody predicted, and one wrong wheel tag

None of these are exotic; all three are invisible on Linux for a structural reason.

1. **The vendored LuaJIT `make` inherits the outer make's jobserver and dies.** `make[3]: /bin/sh: Bad
   file descriptor`, then `write jobserver: Bad file descriptor`. GNU make advertises its jobserver
   through `MAKEFLAGS` but only passes the descriptors to a recipe it recognises as recursive — one
   spelled `$(MAKE)`. Ours is deliberately plain `make`, because the outer generator may not be make
   at all. **Not a macOS bug**: it is any `Unix Makefiles` build on any platform, and it had never
   been seen because everything that matters (CI, scikit-build-core) uses Ninja. Fixed by clearing
   `MAKEFLAGS`/`MAKELEVEL` for that one command, which is generator-independent.
2. **`loom.LoomError` silently became `RuntimeError`.** The engine throws `loom::Error` from
   `libloom_engine.dylib`; the `catch` that translates it lives in `_loom.so`. Those classes are
   header-only, so their typeinfo is a weak symbol in both binaries — and pybind11 builds every
   extension module `-fvisibility=hidden`, which under Mach-O's two-level namespace stops the two
   copies coalescing. **Apple's libc++ compares `type_info` by address**, so the catch clause was
   simply skipped. ELF's flat namespace merges them, which is why Linux never saw it. Fixed by giving
   the exception classes explicit default visibility in `loom_errors.h`. The symptom was not a crash
   — it was `except loom.LoomError` catching nothing, i.e. the documented way to tell "your GGUF is
   wrong" from "this binding is wrong" quietly ceasing to work.
3. **`/tmp` and `/var` are symlinks on macOS.** `test_backend_discovery.py` compared an
   `os.path.abspath` against a path the package had `resolve()`d, which follows symlinks. On Linux
   they agree; on macOS a venv under either standard temporary root compares `/tmp/.../loom` against
   `/private/tmp/.../loom` and fails on a path that is the same directory. **This would have failed
   in CI too** — cibuildwheel's test step runs from a temp directory. The production code is right to
   resolve; the test now does the same.

**And the wheel tag comes from the environment, not from CMake.** `CMAKE_OSX_DEPLOYMENT_TARGET`
decides what the Mach-O binaries require; **scikit-build-core decides the filename**, and it reads
`MACOSX_DEPLOYMENT_TARGET`. With only the CMake half set, a hand build produced binaries good for
11.0 inside a wheel named `macosx_15_0_arm64` — which pip then refuses to install on the macOS 12 it
would have run on perfectly. Both halves are now set: a pre-`project()` default in
`loom-py/CMakeLists.txt` and `[tool.cibuildwheel.macos].environment` in `pyproject.toml`.

### 4.3 The floor is 14.0, and the first macOS CI run is what said so

Scoped as cibuildwheel's defaults, **11.0 on arm64 and 10.13 on x86-64**. Both are wrong, and the
reason is a backend nobody was thinking about.

ggml compiles its BLAS backend with `ACCELERATE_NEW_LAPACK` and `ACCELERATE_LAPACK_ILP64`
(`src/ggml-blas/CMakeLists.txt`, unconditional for Apple, with **no deployment-target gating**),
which selects Accelerate's *new* BLAS interface — `cblas_sgemm$NEWLAPACK$ILP64`, a symbol that
exists only on **macOS 13.3+**. The first `macos-15` CI run said it in as many words:

```
ggml-blas.cpp:142:13: warning: 'cblas_sgemm' is only available on macOS 13.3 or newer
```

An 11.0-tagged wheel would therefore install on macOS 11 or 12 and **quietly have no BLAS**: the
backend is `dlopen`ed, the symbol is missing, ggml logs at a level the binding drops, and the
accelerator is simply absent — the silent shape P4.8a already cost a build cycle to find.

**Dropping BLAS was the obvious fix and is the wrong one.** It is **1.80x on whisper-small** (990 ms
against the CPU's 1778 ms) on an M1 Pro. It is neutral on VITS, which is what an early, VITS-only
measurement said — and would have been the wrong basis for removing it. A BLAS is for large matrix
multiplication; the model that shows it is the ASR encoder, not the TTS vocoder.

So the floor rose, on both architectures -- Accelerate's new BLAS lands on 13.3 for Intel too -- which
removed the `overrides` block along with the mismatch. The alternatives were a tag that overpromises, or a fourteenth ggml patch dropping
the two defines so ggml-blas uses classic Accelerate — the latter keeps 11.0 and is the better
long-term answer, but it needs its own measurement (classic is LP64, and the new interface may not be
the same speed) and did not belong in a release.

**And the floor is 14.0 rather than the 13.3 the requirement actually is, because 13.3 cannot be
spelled.** Since macOS 11 a wheel's platform tag carries only the MAJOR version: `packaging` generates
`macosx_15_0`, `macosx_14_0`, `macosx_13_0`, and nothing with a non-zero minor, so `macosx_13_3_arm64`
matches nothing pip offers. scikit-build-core knows this and zeroes the minor -- checked rather than
assumed, `normalize_macos_version("13.3", arm=True)` returns `13.0` -- so setting 13.3 would ship
binaries needing 13.3 inside a wheel tagged `macosx_13_0`: installable on 13.0, 13.1 and 13.2, and
silently BLAS-less there. That is the same defect one major version smaller. **14.0 is the lowest
floor that is both expressible and true**, and the extra cost over 13.3 is macOS 13.x.

**The general point is that this cost nothing to find because a macOS row exists in `ci.yml`.** It
was invisible on every local build, which used the same 11.0 floor and never read the warnings.

### 4.4 The runner labels are not the ones this was scoped with

`macos-13` (Intel) has been **retired** and `macos-14` is **deprecated**. The matrix uses
**`macos-15`** for arm64 and **`macos-15-intel`** for x86_64 — the GA pair. This is the first thing
to check if the job one day fails to schedule rather than fails to build.

The matrix also gained an `artifact` key, because `arch` stopped being unique: Linux x86_64 and Apple
Intel are both `x86_64`, and two `upload-artifact` steps sharing a name is an error rather than a
merge. `gpu-smoke-test` installs `dist/*manylinux*.whl` for the same reason — its `wheels-*-x86_64`
glob now also matches an Apple wheel, and pip treats an explicit filename as a request, not a
preference.

### 4.5 What was verified, and on what

On **`fdemelo@macbook-pro`** — Apple M1 Pro, macOS 15.6.1, the **`apple_m1` rung**, which is the
lowest of ggml's three Apple rungs and therefore the one every `macosx_14_0_arm64` wheel must serve.
An M4 runner could not have stood in for it. This is the analogue of `raspberry-pi-check` that the
item was scoped as lacking.

* `loom_py_rt-cp311-cp311-macosx_14_0_arm64.whl` — **2.23 MB zipped, 5.64 MB unpacked**.
* Installed into a clean venv, imported from a neutral cwd: `loom.devices()` → `BLAS (Accelerate)`,
  `CPU (Apple M1 Pro)`.
* **`pytest tests/ci`: 80 passed, 1 skipped** (the skip is `orthography2ipa`, absent by choice), both
  against the installed wheel and against the source tree. The same suite is green on Linux.
* **Two models actually run**: VITS synthesised speech, and whisper-small transcribed it back —
  the ASR oracle, because correlation is not the test for a TTS family.

**`cibuildwheel` cannot be run locally on macOS** and this is not a defect: it refuses to
system-install python.org CPython outside CI, which is the right call on someone's laptop. The local
builds went through `python -m build` instead, so **the `delocate` repair step is exercised only in
CI** — worth knowing when reading a first macOS CI run.

**One piece of friction that is not ours**: `nlohmann/json` is fetched as a **full ~290 MB clone** for
a header-only library, and it failed twice on this link before succeeding. `GIT_SHALLOW` on that
`FetchContent_Declare` is an obvious improvement and is deliberately left out of this item.

### 4.6 What "done" means here, and what is still open

Done: wheels for both Apple architectures wired into the release workflow; `import loom` and
`loom.devices()` reporting a CPU; `pytest tests/ci` green; blocker 4 answered **in writing, and
answered "no"**; and — the bar this item was raised to once the hardware existed — the arm64 wheel
installed and a model run on a real M1 Pro.

Still open, and deliberately not folded in:

* **Publishing.** Nothing here has been uploaded to PyPI. The macOS rows produce artifacts on the
  next release run; the first Apple wheels reach the index when someone publishes them.
* **The `loom-py` submodule pointer** still names a `loom.cpp` commit that predates P4.20–P4.29, and
  `loom.cpp/main` is many commits ahead of its origin. Every macOS build here used the working tree.
  Bumping the pointer is a release-time act and belongs with the rc7 push, not here.
* **Windows** stays out of scope and stays behind this.


## 5. The Record

### P4.8g — `loom-py-rt-cuda`, and what "two strings changed" cost — DONE (2026-08-14)

P4.8a said a second accelerator package would be `packaging/rt-vulkan/` with two strings changed.
The three files are indeed that small — `CMakeLists.txt` naming `cuda`/`GGML_CUDA`, a `pyproject.toml`
with the `==` pin, and an `__init__.py` holding nothing importable — and the claim was still wrong,
because **the pilot had never been built**. `packaging/rt-vulkan/pyproject.toml` declares
`readme = "README.md"` and no such file exists, so it fails metadata generation before reaching CMake.
P4.8a's end-to-end verification was of the BASE wheel, which ships its own per-microarchitecture CPU
plugins and never needed a backend package to prove itself.

So CUDA was the first backend package ever built here, and it found **four defects, every one of them
in shared code**:

1. **No `README.md`** — `rt-vulkan` has the identical bug.
2. **`FetchContent_MakeAvailable(ggml)` brought ggml's own `install()` rules into the project.** The
   first wheel was 297 MB and held `libggml-cuda.so` TWICE — once where this package installs it, once
   where ggml's rule does — plus a `lib/libggml-base.so` that `BackendPackage.cmake`'s own comment
   forbids, because two of those on one `sys.path` have no rule about which loads. Vulkan escaped it by
   accident: it needed `Populate` + `add_subdirectory` for an unrelated glslc reason, and that spelling
   happens to be the correct one. Both paths now share it.
3. **`EXCLUDE_FROM_ALL` then excluded the target we wanted** — `ninja: no work to do`, and the install
   step failed looking for a library nothing had compiled. Introduced by fixing (2); fixed by putting
   exactly one target back with `set_target_properties(... EXCLUDE_FROM_ALL FALSE)`.
4. **A soname mismatch, and it fails SILENTLY.** The base wheel ships `libggml-base.so` with no
   version chain, because P4.8a unset VERSION/SOVERSION when it found a zip cannot carry a symlink and
   was paying 2.6 MB for three copies of each library. That decision never reached
   `BackendPackage.cmake`, so the backend built with ggml's defaults and recorded
   `NEEDED libggml-base.so.0`. Nothing provides that name. The dlopen fails, ggml logs it at a level
   the binding drops, and **the entire symptom is an accelerator missing from `loom.devices()`** — no
   error, no warning, no traceback. Found with `ctypes.CDLL` on the shipped file.

Item 4 is the one to carry forward. A backend package has exactly one job, and its failure mode is
indistinguishable from not having installed it.

### Verified from a clean venv, on the workstation

```
   CPU    | Intel(R) Core(TM) Ultra 9 285K
   CUDA0  | NVIDIA GeForce RTX 5090

cpu     21.19s  ' Paris. The capital of Germany is Berlin. ...'
CUDA0    1.28s  ' Paris. The capital of Germany is Berlin. ...'
```

Two wheels, a venv built from nothing, a neutral working directory, character-identical output and
~16.6x wall clock including load. That is the packaging claim end to end for the first time.

### Two things left explicitly undone, both real

* **The wheel does not fit PyPI, and stripping cannot help.** Three architectures (`80;89;120`) is
  **112 MB packed**, against a 100 MB per-file ceiling; the default arch list is far worse at 297 MB.
  `strip` changes nothing because the payload is `.nv_fatbin` cubins rather than symbols. So the arch
  set is a release decision with a hard constraint attached, and P4.8c's "a narrow-arch package is a
  live option" now has numbers: roughly two architectures fit, three do not.
* **The built library has `RPATH /opt/mamba/envs/py-3.12/lib`** — the build machine's conda prefix, so
  `libcudart`/`libcublas` resolve from a path no user has. Fine for a local artifact and wrong for a
  published one, which needs the CUDA runtime bundled (auditwheel) or declared as `nvidia-*` pip
  dependencies. Not fixed here because it is a CI-shape decision, not a packaging bug.


### P4.8h — the CUDA wheel fits, and neither lever cost coverage — DONE (2026-08-14)

P4.8g left the CUDA wheel at 112 MB against PyPI's 100 MB per-file ceiling, with the arch set called a
release decision. It is settled, and the answer is better than the trade it looked like: **88 MB with
MORE architectures than the version that did not fit.**

| build | toolkit | configuration | wheel |
|---|---|---|---|
| A | 12.8 | `80;89;120`, FA on | 112 MB |
| B | 12.8 | ggml's default list, FA on | 148 MB |
| C | 12.8 | ggml's default list, **FA off** | 101 MB |
| D | 13.1 | **no list**, FA off | 135 MB — ten cubin sets, no PTX |
| **E/F** | **13.1** | **explicit list, FA off** | **88 MB** |

### Compression was already maximal, which killed the obvious idea first

`nvcc` gained `--compress-mode` in 12.8 and it looked like free savings. It is not available: **ggml
already sets `GGML_CUDA_COMPRESSION_MODE` to `"size"` by default** and applies it whenever the toolkit
is 12.8+. The measurement that proved it is worth keeping — build A was byte-identical to the earlier
wheel, because the flag both failed to reach `nvcc` and would have changed nothing. There is no
compression lever left.

### FlashAttention is unreachable code, and it was a third of the binary

`ggml_flash_attn_ext` appears NOWHERE in the engine. The attention primitive builds the composite path
— `mul_mat` -> `soft_max_ext` -> `mul_mat`, with `mul_mat_set_prec` on the QK product — so nothing loom
emits can ever dispatch to a FlashAttention kernel. `GGML_CUDA_FA=OFF` removed **47 MB** (148 -> 101)
with no functional change of any kind.

Revisit the day the engine grows a primitive that emits `FLASH_ATTN_EXT` — scoped, not built. Two
things that work would need re-checking: the device/CPU parity tolerance, since FA changes reduction
order and internal precision, and the property that the composite path runs identically on every
backend.

### "Take the default" is not portable across toolkits, and D is the proof

ggml chooses its architecture list only `if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`. **Under CUDA 13
CMake defines it first**, so ggml's careful list is skipped entirely and every architecture the toolkit
knows gets a real cubin: ten sets, no PTX at all, 135 MB — *larger* than the 12.8 default it was meant
to improve on. The list is now explicit in `packaging/rt-cuda/CMakeLists.txt` rather than inherited.

### What the shipped list covers, and what it drops

Real cubins for **8.6, 8.9, 12.0a, 12.1a** and PTX for **7.5, 8.0, 9.0**:

* RTX 30x (8.6) and Jetson Orin (8.7, by binary compatibility from 8.6);
* RTX 40x (8.9); RTX 50x (12.0a); DGX Spark (12.1a);
* Turing, A100 and Hopper JIT from PTX rather than finding nothing.

Dropped deliberately: Maxwell 5.0, Pascal 6.1, Volta 7.0, which CUDA 13 no longer supports. A real
loss of the cheapest hobby hardware, accepted on the grounds that those cards cannot run current models
usefully and that tier is better served by Metal on Apple silicon — a different backend package
entirely, unaffected by any of this.

**12.1a requires CUDA >= 12.9 and there is no 12.x substitute.** ggml rewrites every `12X` to `12Xa`
because Blackwell's FP4 tensor-core instructions are not forward-compatible and cannot be branched on
in host code, and an `a` cubin runs only on its exact architecture. So Spark coverage is a toolkit
version, not a flag.

### The toolkit upgrade required installing nothing

`py-3.13` on the workstation already carried **CUDA 13.1.0**. The constraint was to leave `py-3.12`'s
torch/Lightning alone; nothing was installed in either environment, and the invariants were checked
after: `py-3.12` still reports `2.8.0+cu128 / True` with `nvcc 12.8`, `py-3.13` still `2.9.0+cu130 /
True`. `~/.local/bin/ninja` supplied the generator, since `py-3.13` has none.


### P4.8i — the runtime comes from NVIDIA's wheels, and the GPU list follows the CPU — DONE (2026-08-14)

P4.8g left two things unfixed and named them: the built library carried an RPATH into the build
machine's conda prefix, and the architecture set was a release decision. Both are settled, and the
first one settled the toolkit version rather than the other way round.

### `nvidia-*-cu13` does not exist, and that chose CUDA 12.9

The plan was to depend on NVIDIA's own runtime wheels rather than bundle `libcublas` — a machine with
an NVIDIA GPU is not surprised to be asked for NVIDIA's runtime, and bundling it would dwarf a 72 MB
wheel. Checking the names first turned out to matter:

| package | what is actually on PyPI |
|---|---|
| `nvidia-cublas-cu13` | a **1.4 KB placeholder sdist**, version 0.0.1 |
| `nvidia-cublas-cu12` | a real manylinux wheel, **12.9.2.10** |
| `nvidia-cuda-runtime-cu12` | a real manylinux wheel, **12.9.79** |

So a CUDA 13 build has no distributable runtime at all. The `cu12` line reaches 12.9 — and **12.9 is
exactly the minimum for `121a-real`**. That makes 12.9 the only version that gets both DGX Spark
coverage and a runtime pip can install; 13.1 gets the first and loses the second. P4.8h had reached
13.1 for the coverage alone, and this reverses that for a reason it could not have seen.

### The RPATH, and why it is not ctypes preloading

`nvidia-cuda-runtime-cu12` unpacks to `nvidia/cuda_runtime/lib/`, a sibling of `loom_rt_cuda/` in
site-packages, so `$ORIGIN/../nvidia/cuda_runtime/lib` reaches it with nobody executing anything.
That property is the point: `loom/__init__.py` finds accelerator packages by SCANNING `sys.path` and
never imports them, precisely so a broken accelerator cannot take the base package down. torch's
approach — preloading its CUDA libraries with `ctypes` at import — would require giving that up.

Verified by removing the escape route rather than trusting the ordering. The build still emits conda's
own `-Wl,-rpath,$PREFIX/lib` ahead of ours (conda's compiler wrappers inject it, and it is absent in a
manylinux container, which is where CI builds), so the installed library was rewritten with `patchelf`
to the `$ORIGIN` entries alone before testing:

```
libcudart.so.12   -> .../loom_rt_cuda/../nvidia/cuda_runtime/lib/libcudart.so.12
libcublas.so.12   -> .../loom_rt_cuda/../nvidia/cublas/lib/libcublas.so.12
libcublasLt.so.12 -> .../loom_rt_cuda/../nvidia/cublas/lib/libcublasLt.so.12
```

and then end to end: `pip install` resolved `nvidia-cublas-cu12 12.9.2.10` and friends, `loom.devices()`
listed `CUDA0`, and a real model generated correct text in 1.16 s.

**And the repair step needs excludes, or it undoes all of this.** cibuildwheel runs `auditwheel repair`
by default, whose entire job is to copy external libraries INTO the wheel and repoint RPATHs at its own
`.libs`. Left alone it would bundle `libcublas.so.12` — cancelling the dependency and adding hundreds
of megabytes. `libggml-base.so` needs excluding for a different reason: it is genuinely absent at
repair time, being the base wheel's to ship, and auditwheel treats a library it cannot find as an
error. Both packages now carry a `repair-wheel-command` saying so.

### The GPU list follows the CPU architecture

pip already selects by platform tag, so a per-CPU list costs the user nothing and keeps the aarch64
wheel — going to the most constrained devices — from carrying desktop kernels:

| CPU | real cubins | PTX | why |
|---|---|---|---|
| x86_64 | 8.6, 8.9, 12.0a | 7.5, 8.0, 9.0 | RTX 30x/40x/50x. No 12.1a: GB10 is an ARM part |
| aarch64 | 8.7, 12.1a | 8.0, 9.0 | Orin and DGX Spark. No Ada or RTX 50x board exists on an ARM host |

Orin gets its own `87-real` rather than leaning on 8.6 binary compatibility, because it is a
first-class target on that side and the wheel has room once the desktop kernels are gone. Measured:
the x86_64 wheel is **72 MB**, down from 84 MB when it carried `121a` it could never use.

### Python floor moved to 3.10, and the backends are NOT multiplied by it

3.9 is past EOL. 3.10 stays, and the reason is a target rather than a preference: **JetPack 6 ships
Ubuntu 22.04, whose system Python is 3.10**, so a 3.11 floor would push Jetson users into a venv before
they could install anything.

The backend packages are **one wheel per platform, not per interpreter**. They are `py3-none-<platform>`
because the payload is a plain shared library that ggml dlopens and Python never imports; their
`build = "cp310-*"` names which interpreter runs the build, not which the wheel serves. Only the base
wheel, carrying `_loom.so`, needs one build per Python. Worth stating because the natural reading of a
wheel matrix is that everything multiplies by everything, and here three quarters of that product is
the same file.

### Not verified

**The aarch64 arch list has never been compiled.** There is no ARM machine here, so that row of the
table is a CI-only path — the strings are right in principle and untested in fact.


## 6. 32-bit ARM Linux — ARMv6 shipped (P7)

**Shipped 2026-09-04, on the board rather than in a container**: a Raspberry Pi Zero W synthesises
speech from the `linux_armv6l` wheel this section describes. What §6 said before this was a scope
from a source read, with ARMv6 named a non-goal; the read was accurate about the mechanics and wrong
about the cost, which turned out to be one CMake guard and no C++ at all.

### 6.1 Three tiers, and only one of them was ever a port

The question is usually asked as "Raspberry Pi Zero", which is two different machines.

| board | ISA | status |
|---|---|---|
| **Zero 2 W** (BCM2710, Cortex-A53) on **64-bit** Raspberry Pi OS | ARMv8.0-A | **a supported target since aarch64 shipped.** A53 is ARMv8.0 with no dotprod/FP16/SVE — the same profile `wheels.yml`'s `raspberry-pi-check` already gates on its A72 rows, selecting the same `libggml-cpu-armv8.0_1.so`. |
| Zero 2 W on the **32-bit** image, Pi 2/3 on 32-bit | ARMv7-A + NEON | **builds, unverified.** The guard below is on pointer width, so armv7 takes the same path; no board here runs it and no wheel is published for it. |
| **Zero / Zero W**, Pi 1 (BCM2835, ARM1176JZF-S) | ARMv6 + VFPv2, **no NEON** | **shipped** — §6.3 |

### 6.2 The port was one guard, and the predicted blockers mostly were not

`GGML_CPU_ALL_VARIANTS` was the only hard failure, and it is a configure-time one rather than a
compile-time one: `loom-py/CMakeLists.txt` FORCEd it ON, and ggml's ARM ladder
(`src/CMakeLists.txt`) is aarch64-only — every rung is emitted as `-march=armv8.x-a[+dotprod…]` at a
compiler that has no such architecture. `armv6l` and `aarch64` both resolve to `GGML_SYSTEM_ARCH ==
"ARM"`, so a 32-bit build walks straight into it. The fix is `CMAKE_SIZEOF_VOID_P EQUAL 8` around
the FORCE, and a 32-bit wheel is therefore one un-split `libggml-cpu.so`. Nothing is lost: every rung
of that ladder distinguishes dotprod / FP16 / SVE / i8mm, none of which exists below ARMv8.

**One blocker was not on the list at all, and it is the only code change this item needed outside
that guard: `libggml-base.so` references `__atomic_fetch_add_8`.** `ggml_graph_next_uid()` (ggml.c)
is a `__atomic_fetch_add` on a static `uint64_t`; ARMv6 has `LDREX`/`STREX` and no `LDREXD`, so GCC
cannot inline a 64-bit atomic and emits a libatomic call that nothing in this tree links.
`cmake/Dependencies.cmake` now probes for it with `check_c_source_compiles` — which links, not just
compiles — and puts libatomic on `ggml-base` PUBLIC when the toolchain needs it. **How it fails is
the reason it is worth reading about**: every shared library in the build links clean, because a
shared object is allowed to carry undefined symbols; the first *executable* is what fails, and a
build that links no executable — the wheel build — succeeds and fails at `import loom`, in `dlopen`,
on the user's board. See [Retro-033](../retros/retro-033-a-shared-library-links-clean-without-its-symbols.md).

The rest of the predicted list came out better than the read expected:

* **LuaJIT — the "one genuine unknown" — needed nothing.** Its ARM port covers ARMv5TE up
  (`lj_arch.h` sets `LJ_ARCH_VERSION 60` for `__ARM_ARCH_6__`), and because the build is native
  inside an emulated target userland, the `minilua`/`buildvm` bootstrap that makes cross-compiling it
  hard never has a host/target split to reconcile. A 703 KB `libluajit.a`, unpatched.
* **This engine still has no ISA-specific code.** `grep -rE '__aarch64__|__ARM_NEON|immintrin|AVX'
  src include tools` returns nothing; every byte of it is in ggml and in `cmake/patches/`, which is
  [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md) paying out on an axis it was not
  written for.
* **ggml compiles clean.** `__ARM_NEON` is simply undefined, every `arm_neon.h` include is guarded by
  it, `arch/arm/quants.c` takes its 33 scalar `#else` arms and `arch/arm/repack.cpp` gates its bodies
  on `__aarch64__`. Our own patches degrade rather than fail: `ggml-0006`/`0007` have generic arms,
  `ggml-0010` falls back to scalar `erff`, and `ggml-0001`/`0002`/`0011`/`0012` go inert.
* **Blocker 4 (`VECTOR_REGISTERS 32` on any `__ARM_NEON`) does not arise at all here** — there is no
  NEON on this rung. It is still live for armv7.

### 6.3 The wheel is built in an emulated Raspbian, and does not go to PyPI

Both halves are forced rather than chosen, and [ADR-025](../adrs/adr-025-armv6-is-built-in-its-own-emulated-userland.md)
records why: no provider has a 32-bit ARM runner; cross-compiling gets the wheel tag, the default
`-march` and *libgcc* wrong; and PyPI accepts only `manylinux*`/`musllinux*` tags for Linux, whose
floor is armv7.

`.github/docker/Dockerfile.armv6` is the build environment — Raspbian bookworm (glibc 2.36, the
board's own), `docker run --platform linux/arm/v6`, `qemu-arm` under binfmt. `wheels.yml`'s
`build-armv6-wheel` job builds it on every release and attaches it to the GitHub release; the
artifact is deliberately **not** named `wheels-base-*`, because `publish-pypi` globs that into one
upload and one refused tag fails all of it.

**`QEMU_CPU=arm1176` is what makes the emulated check a check**, and it is worth stating because the
failure it catches is invisible without it. qemu-user runs whatever CPU model it is given; under a
permissive one (`cortex-a15`, `max`) an ARMv7 build imports, runs and passes, and then takes SIGILL
on the board. Measured: a program containing one `movw`/`movt` pair — ARMv6T2, which an ARM1176 is
not — prints its answer under `QEMU_CPU=cortex-a15` and dies with "Illegal instruction" under
`QEMU_CPU=arm1176`.

### 6.4 What it does, and how slowly

Measured on the board — a Raspberry Pi Zero W (BCM2835, one ARM1176JZF-S at 1 GHz, 427 MB usable),
Raspbian bookworm, Python 3.11.2, the `linux_armv6l` wheel installed from a file. VITS at Q4_0,
11.7 MB, given the phonemes for *"hello world, this is a Raspberry Pi Zero"*:

| | |
|---|---|
| model load | 1.2 s warm, 4.5 s cold |
| synthesis | **237–265 s for 3.19 s of audio** — 74x to 83x slower than real time |
| peak RSS | 90 MB, of which 11.7 MB is the model |
| wheel | 1.49 MB (one `libggml-cpu.so`; the x86-64 wheel ships fourteen variants) |

**The spread is thermal, and [Retro-025](../retros/retro-025-the-arm-that-ran-second-paid-for-the-first.md)
is why it is reported as a spread.** One run on a board that had been idle is 236.7 s; three
back-to-back are 266.4 / 263.5 / 265.0 s. A Zero W has no heatsink and this is four and a half
minutes of solid floating point, so the second and third runs are paying for the first.

It scales linearly rather than falling off a cliff — on the same idle board, 0.43 s of audio costs
30.8 s, 0.98 s costs 71.7 s, 2.19 s costs 165.8 s and 3.19 s costs 236.7 s — so ~74x is the shape of
the thing and not an artifact of one utterance length. For comparison, the same file and the same
phonemes on the x86-64 dev box take 1.18 s: the board is **200x slower than a laptop**, and the Pi 4
columns in the README's benchmark tables transfer to it in no way at all.

**It is deterministic, and that was checked rather than assumed** — all three runs produced
byte-identical output (`sha256 cf88b7ae…` over the 16-bit PCM), as did the earlier one-off.

**Q4_0 buys size here, not speed, and the difference is worth stating because the name implies
otherwise.** ABBA on the board, 60 s settle between arms, normalised to seconds of compute per second
of audio (the two files disagree on duration — the predictor is quantization-sensitive — so raw
seconds are not comparable):

| | ms of audio per run | A | B | mean |
|---|---|---|---|---|
| VITS **Q4_0**, 11.7 MB | 3.19 s | 78.8x | 82.5x | **80.7x** |
| VITS **F32**, 62.8 MB | 3.33 s | 79.8x | 83.5x | **81.7x** |

**1.2% apart, inside the thermal drift** — and the drift is visible in the table, both B arms being
slower than both A arms. What Q4_0 is worth on this board is 5.4x of file, on 427 MB of RAM.

**And 96% of the run is convolution.** `LOOM_PROFILE`, one thread (which is all there is), Q4_0:

```
by op            calls        ms       %
CONV_2D            117  217234.6   84.4%
CONV_TRANSPOSE_1D    3   29587.4   11.5%
ADD                298    4180.3    1.6%
MUL_MAT             36    2577.5    1.0%
```

`MUL_MAT` is **1.0%**, which is why it costs nothing that tinyBLAS declines this target outright:
`llamafile/sgemm.cpp`'s F32 case has no arm for a machine with neither AVX, NEON, VXE, MMA nor RVV
and falls through to `return false`. The engine's ~2x-on-x86 GEMM is simply not on the critical path
of a vocoder here. Any future work on this rung is convolution work.

**The ceiling is the board, not the engine.** Measured with the same eight-chain scalar FMA loop at
`-O2` on each machine, one core: the Zero W does **87 MFLOP/s** and the x86 dev box **~8.9 GFLOP/s**
— 100x per core, ~400x against its four. loom's measured end-to-end gap is **210x**, so the ARMv6
build is doing *better* than the raw floating-point ratio, not worse. There is no order of magnitude
sitting on the floor waiting for a patch; there is an ARM1176 with VFPv2, no SIMD of any kind, and
one of it.

**And it is CORRECT, which is the part that had to be checked rather than assumed.** The board's
waveform against the dev box's, same model, same seed, sample for sample:

```
n 70400 70400
samples differing: 102 (0.14%)
max abs diff (of 32767): 1
cosine: 0.999999999
```

Every difference is one least-significant bit of 16-bit PCM — floating-point summation in a different
order, which is what two ISAs are expected to disagree by and nothing more. The duration predictor
produced the same 70400 samples on both, and the ASR oracle ([Retro-006](../retros/retro-006-kokoro-shipped-noise.md)'s
rule: cosine agreement is not intelligibility) transcribes the board's own file back as
*"Hello world, this is a Raspberry Pi Zero."*

**Both hermetic suites are green on this rung.** loom-py's `pytest tests/ci` is **95 passed /
4 skipped on the board itself** and identically so in the emulated Raspbian under `QEMU_CPU=arm1176`;
the engine's own `ctest -L ci` is **81/81** in 150 s emulated (it needs a build tree, which the board
has no room for).

`ctest` found exactly one 32-bit defect, and it was in a fixture rather than in the engine:
`tests/fixtures/reference_duration_aligner.py` passed `np.int64` as `np.repeat`'s `repeats`, and
numpy casts that argument to the platform index type under the `safe` rule — which on a 32-bit
interpreter is `int32`, so it raises rather than truncating. `np.intp` is the type that was meant and
is correct on both. Nothing in `src/` or `include/` needed a line.

**One caveat that is the board's and not the wheel's, because it cost an hour to establish that.**
The reference Pi segfaults at backend discovery — intermittently, in a way that looks exactly like a
32-bit memory bug — and it is not one: a third-party package had installed its own
`libstdc++.so.6.0.32` into `/usr/local/lib/arm-linux-gnueabihf`, which `/etc/ld.so.conf.d` puts ahead
of Raspbian's 6.0.30, so `std::filesystem::directory_iterator` frees its shared state under one
libstdc++ and releases it under another. Fifteen lines of C++ that never mention this project
reproduce it, and `LD_PRELOAD` of the distribution's own copy fixes it 5/5 —
[Retro-034](../retros/retro-034-the-boards-own-libstdcxx.md). Worth knowing because the emulated gate
cannot see it: the container has exactly one libstdc++, which is the point of building there.

**Which models are worth putting on this board, measured rather than reasoned.** Every output below
is identical to the same file's on x86-64, and all three are Q4_0:

| model | file | peak RSS | task | on the board |
|---|---|---|---|---|
| `distilbert-ner` | 115 MB | 131 MB | 17-token sentence → a label per token | **8.7–12.0 s** |
| `conformer-ctc-small` | 44.6 MB | 84 MB | 3 s of speech → transcript | 61.9 s = **20.6x** real time |
| `vits-piper-en-gb-miro` | 11.7 MB | 90 MB | 3.19 s of speech synthesised | 245–265 s = **80x** real time |
| both ASR and NER, one process | 160 MB | 185 MB | audio → transcript → entities | 65.5 s end to end |

**And Q4_0 is worth 3.8x on the transformer and nothing on the vocoder**, which is the rule this rung
turns out to have. ABBA on the board, DistilBERT: Q4_0 **12.04 / 8.74 s** against F32 **39.60 /
39.90 s**, and 131 MB of RSS against 262 MB. Against VITS's 1.2%, that is the same knob giving two
opposite answers, and the profile above says why: VITS is 96% `CONV_2D`/`CONV_TRANSPOSE_1D`, whose
kernels are float, while DistilBERT is essentially all `MUL_MAT`, where Q4_0 × Q8_0 is an **integer**
dot product — and an ARM1176's integer path is far better than its VFPv2 scalar float. The
transformer sustains roughly 140 MFLOP-equivalent/s against the board's measured 87 MFLOP/s *floating
point* ceiling, which is only possible because it is not doing floating point.

So the selection rule for an ARMv6 board is **matmul-shaped models, quantized** — encoders, token
classifiers, CTC ASR — and not convolutional vocoders, where the quantization pays for itself in file
size alone. A causal LM is a separate no: decode costs 2 FLOPs per parameter per token, so even
`gemma-3-270m-it` at Q4_0 projects to several seconds per token before the 512 MB question is asked.

### 6.5 What INT8 would and would not buy, and where the headroom actually is

The obvious idea on a board with no floating-point SIMD is to get the arithmetic into integers.
Measured, on the board:

| MAC throughput, one core, eight independent chains, `-O2` | MMAC/s | vs VFP |
|---|---|---|
| `float` VFP multiply-add | 43.4 | 1.00x |
| `uint32_t` scalar multiply-add | 70.0 | 1.61x |
| `__smlad` — ARMv6's dual 16x16 MAC on a GPR pair | **156.7** | **3.61x** |

**THE 3.61x IN THAT TABLE IS WRONG, and `scripts/bench30.c` is what disproved it.** Its `float` arm
measures the loop rather than the FPU: a real 2x2-tiled F32 GEMM on the same board reaches **153.7
MMAC/s**, against the tiled *quantized* kernel's 167.9 — within 10% of each other. `__smlad` is
genuinely emitted (192 of them in the shipped `libggml-cpu.so`, checked with `objdump`) and genuinely
does two MACs per instruction; the advantage is simply spent on the `uxtb16`/`ssub16`/`sxtb16` that
unpack Q4_0 nibbles and Q8_0 bytes around it — twelve setup instructions per eight MACs, where an F32
`vmla` needs none. **On this core the win is the register tile, not the integer arithmetic**, and
quantization's value is memory: eight times less weight through a 16 KB cache. The rest of this
section is kept for the instruction inventory, which is accurate.

ARMv6 does have SIMD; it is just not NEON. `__ARM_FEATURE_SIMD32` and `__ARM_FEATURE_DSP` are both
defined by the stock Raspbian compiler, `<arm_acle.h>` exposes `__smlad`, `__smlald`, `__sxtb16` and
the rest, and they work on an ARM1176 — two 16-bit MACs per instruction on the general-purpose
registers, with `__sxtb16` unpacking two int8s into two int16s in one more. **3.61x is the ceiling for
any "make it integer" idea on this core.**

**It does not reach VITS by changing the quantization type, and the code says why before a stopwatch
does.** `ggml_conv_2d_direct_packed` (`cmake/patches/ggml-0013`) "dequantizes it once per call into
the scratch buffer its direct sweep already repacks an F32 kernel into, so every lowering below that
point is the F32 lowering". Q4_0 and Q8_0 differ in what is stored, not in what is computed —
confirmed by ABBA on the board: **80.6x / 81.1x / 81.7x** real time for Q4_0 / Q8_0 / F32, a 1.4%
spread. And the other lowering is not the answer either: `GGML_CPU_DISABLE_CONV_HEURISTICS=1`, which
sends the convolutions to im2col + `MUL_MAT`, is **1.63x slower** (416 s against 256 s). Both are in
[Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md).

Getting integers into a convolution therefore needs an **int8 x int8 -> int32 direct-conv kernel**,
which ggml has for no architecture — quantized dot products exist only under `MUL_MAT`. §6.7 measures
what that is actually worth, and the answer is far below the 3.61x ceiling above: most of VITS's
convolution time is not in the direct sweep at all.

**The headroom worth having is under `MUL_MAT`, which serves both models this rung should actually
run (§6.4).** That was scoped from the table above and then measured properly — see §6.6 for the
prototype, and for the correction: the projection made here from these microbenchmark numbers was
**2.2x and the measured answer is 1.47x**, because the incumbent kernel was never at the naive scalar
bound this table describes.

### 6.6 The ARMv6 kernels, built and measured on the board

> **The seconds in this section are about a third high.** They were taken while
> `bluetooth-km-switch` was spinning on half of this board's single core — `ps` reported it at 5.7%,
> which is an average over its lifetime rather than a rate. The **ratios are ABBA'd and hold**: the
> 1.80x below re-measures at 1.837x on a quiet board. For a clean anchor, the pre-kernel VITS
> synthesis is **181.8 s**, not the 259 s recorded here.
> [Retro-037](../retros/retro-037-ps-said-six-percent-top-said-fifty.md).


Three patches, `cmake/patches/ggml-0017` to `0019`. Two ship on by default and one does not.

**What shipped, measured on a Pi Zero W by same-session ABBA** — baseline wheel against kernel wheel,
`pip install --force-reinstall` between arms, 30 s settle, four arms:

| | baseline A / B | kernels A / B | |
|---|---|---|---|
| `conformer-ctc`, 3 s of audio | 66.2 / 67.6 s | **14.3 / 17.0 s** | **4.28x** — 22.3x -> **5.2x** real time |
| VITS, 3.2 s of audio | 259.9 / 258.2 s | **141.3 / 146.6 s** | **1.80x** — 81x -> **44x** real time |
| `distilbert-ner`, 17 tokens | 9.54 / 8.74 s | 9.57 / 6.29 s | 1.15–1.62x; too noisy at 10 s to pin |

Every transcript correct. VITS's waveform **changes** — `sha 1e5fa69a` rather than the baseline's
`cf88b7ae`, identical across both kernel arms — because the convolution route now quantizes its
activations to Q8_0, which `ggml-0013`'s own comment warned is less accurate than dequantize-then-F32.
The ASR oracle is what clears it: the board's own audio still transcribes as *"Hello world, this is a
Raspberry Pi Zero."* ([Retro-006](../retros/retro-006-kokoro-shipped-noise.md)'s rule — cosine
agreement is not intelligibility, and here there is not even a cosine to appeal to, since the duration
predictor lands on 3.24 s instead of 3.19 s.)

**distilbert-ner is where the measurement is weakest and this should say so.** At ~10 s a run it swings
±40% on this board; the kernel-arm readings across three ABBAs are 6.27, 6.31, 6.37, 6.51 and then
9.57, and the last is as likely to be the machine as the code. conformer at 66 s and VITS at 260 s are
the signals worth quoting.

**Three changes got it there, and the last is an ARMv6 arm of a kernel that already existed:**

| | what | measured in isolation |
|---|---|---|
| `0018` `gemm44` | the F32 GEMM's 4x4 tile written out longhand | 173.9 vs 110.6 MMAC/s |
| `0019` conv route | `k` was passed in elements where `llamafile_sgemm` wants blocks | fixes a wrong answer |
| `0019` conv tile | an ARMv6 arm for `ggml_conv_1d_direct_tile_impl` | 51.8 vs 22.5 MMAC/s, **2.30x** |

**How to size a tile on this architecture**How to size a tile on this architecture, which took three benchmarks to get right.** The first
attempt at 4x4 measured **27.2 MMAC/s** against 2x2's 153.7 and was written off as register pressure.
It was not: it was the `av[RM]`/`bv[RN]` arrays in the inner loop going to the stack, plus
`A[lda*(ii+i)+l]` recomputing addresses every iteration. Rewritten with explicit scalar accumulators
and post-incremented pointers, the same 4x4 tile reaches **226.7 MMAC/s** — the best F32 result on the
board and 1.45x over the shipped 2x2.

**And the two kernels then top out at different shapes, for a reason worth remembering**
(`scripts/bench31.c`, `bench32.c`):

| tile | F32 | int16 `__smlad` |
|---|---|---|
| 2x2 | 155.9 | 228 |
| 4x2 | — | **270** |
| 4x4 | **226.7** | 217 (spills) |

**Size the tile by which register file the accumulators land in.** F32 accumulates in VFP, which has
32 single-precision registers, so 4x4's sixteen accumulators fit and the pointers use the GPR file
independently. Integer accumulates in the *same* 14-register GPR file the pointers are in, so 4x2 —
eight accumulators and six pointers — is the wall. The Q4_0 kernel is stricter again: its unpack needs
four live registers per row and four per column on top, which is why it stays at 2x2.

That table also settles what `__smlad` is worth. At its best tile it is 270 against F32's best 226.7 —
**1.19x**, with int16 operands and no unpacking at all. Real, but a fraction of what the tile is worth,
and it would need an int16 type ggml does not have.

**Two things went wrong on the way and both are recorded**, because both were the kind of error that
passes review: the tile order was first fixed to what one convolution shape wanted, which made
conformer 5% slower ([Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)), and
the emulated `ctest` that pronounced the first version green had compiled the kernels *out*
([Retro-035](../retros/retro-035-the-emulator-said-it-was-an-arm10e.md)). The suite is green now on a
build whose `sgemm.cpp.o` is `Tag_CPU_arch: v6` and whose `libggml-cpu.so` carries ten ARMv6 symbols
— checked with `nm` before the suite was believed.

**Emulated builds of this repo must pass `-DGGML_NATIVE=OFF`.** Under QEMU `gcc -mcpu=native` answers
`-mcpu=arm10e -march=armv5te+fp`, one rung below the target, and everything still builds and passes.

### 6.7 How the work was scoped, and where the scoping was wrong

*Recorded as [ADR-026](../adrs/adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md).*

**Two earlier versions of this section argued against ARMv6-specific kernels on carry cost — code ggml
would never take, maintained here forever. That is not an objection on this project's terms**: ARMv6
is the declared floor (below it is microcontrollers, which have none of the machinery this engine
assumes), so a kernel for it serves a permanent tier. Once that is granted the answer changes, because
a *dedicated* kernel can do what a drop-in `vec_dot` replacement cannot — pick its own loop order and
hold a register tile — and that turns out to be where most of the win is.

**Where the convolution time goes.** Attributing `LOOM_PROFILE`'s buckets to the topology's real
shapes gives the rate each lowering achieves on the board:

| bucket | ms | MMAC | MMAC/s | lowering |
|---|---|---|---|---|
| L=275, flow + vocoder pre (41 calls) | 92195 | 2041 | **22.1** | direct path **declines** -> F32 im2col |
| L=94, text encoder (57 calls) | 33166 | — | — | same shape class |
| L=2200 / L=17600 / L=70400 resblocks | 91874 | 5422 | 43.0 / 50.8 / 90.2 | direct sweep |

`ggml_conv_1d_direct_ok` refuses the top row twice over — weights 1.47 MB against a 512 KB budget, and
`OL/4 = 68 < OC/4 = 96` — so `ggml_compute_forward_conv_2d_impl` dequantizes the kernel and runs a
batched **F32** im2col GEMM at 22 MMAC/s. **49% of the runtime is convolutions that threw their
quantization away.**

**What a dedicated kernel reaches at that exact shape** (IC*K=960, OC=384, OL=276 —
`scripts/bench27.c`, `bench28.c`, `bench29.c`; every arm verified `memcmp`-identical to the first):

| | MMAC/s | vs today |
|---|---|---|
| F32 GEMM — what runs today | 29.7 | 1.00x |
| q4_0 x q8_0, shipped `vec_dot` | 66.3 | 2.23x |
| q4_0 x q8_0, `__smlad` `vec_dot` (the drop-in of §6.6) | 78.3 | 2.63x |
| order A, 1x4 column tile | 97.0 | 3.27x |
| order B — weights stream, not activations | 106.9 | 3.60x |
| **order B + 2x2 register tile** | **167.9** | **5.65x** |

Two things in that table are worth more than the `__smlad` instruction itself. **The loop order**:
`for oc { for ol }` keeps the weight row in L1 and sweeps all 281 KB of activations once per output
row — 108 MB per call. The other order sweeps 207 KB of weights per column, 57 MB. **And the 2x2
tile**: the nibble unpack is ~10 instructions and the activation unpack ~6, so 1x1 pays 20
instructions per 8 MACs and 2x2 pays 48 per 32 — 2.50 down to 1.50. At 167.9 the kernel is past the
isolated dot product's 149.6 and past the pure-`__smlad` microbenchmark's 156.7, because the tile
amortises the overheads those still carried.

**Neither is expressible as a `vec_dot`.** `vec_dot` computes one output element from one row and one
column; the loop order and the tile belong to whatever calls it. That is the technical reason a
dedicated ARMv6 `MUL_MAT` is worth more than a patched kernel, and it is measured rather than argued.

**Everything below is the SCOPING, kept because its reasoning produced the patches and because two of
its numbers turned out wrong in instructive ways — §6.6 is what was actually built and measured.** The
shape-level benchmark said the dedicated GEMM was **5.65x** against the F32 conv fallback and **2.53x**
against the shipped quantized path; end to end on the board it is **1.62x** on distilbert-ner and
**1.07x** on conformer-ctc. Both are real wins and both are far below what a single-shape benchmark
promised, which is [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)'s whole
reason for existing.

**Order to build:**

1. **A dedicated ARMv6 `MUL_MAT` for q4_0 x q8_0** — order B, 2x2 tile, `__smlad`, behind
   `__ARM_FEATURE_SIMD32 && !__aarch64__`. **2.53x** over the shipped quantized path. Amdahl on the two
   models this rung is good at (§6.4): distilbert-ner is 96.6% `MUL_MAT` -> **2.40x** (9.1 s -> 3.8 s),
   conformer-ctc 82.2% -> **1.99x** (61.8 s -> 31 s, 20.6x -> 10.3x real time).
2. **fp16 scales by arithmetic instead of the 256 KB table** — 1.18x measured, bit-identical, not even
   ISA-specific (§6.6), and worth relatively more once the arithmetic is 2.5x faster.
3. **Route the convolutions the direct sweep declines through it** rather than dequantizing to F32.
   **5.65x on 49% of VITS**: 80x -> **48x** real time.
4. **An int8 `__smlad` direct convolution** for the buckets the sweep accepts — still last, but no
   longer marginal now that there is a 167.9 MMAC/s target to aim at against the sweep's 59 average.
   It keeps its reason to exist: at L=70400 an im2col materialises 15.8 MB, which a 427 MB board should
   not pay. If it reached the same 167.9, VITS lands at **29x** real time.

**What became of that order.** Steps 1 and 2 were built, measured and shipped (§6.6), and they are the
two that mattered — though at 1.62x and 1.07x rather than the 2.53x projected here, and distilbert-ner
landed at 6.3 s rather than the 3.8 s this section predicted. Step 3 was built and is **off by
default**: correct on 1-D convolutions and worth 2.6% there rather than 5.65x, and *wrong* on 2-D
ones. Step 4 was **not** built, and step 3 is the reason — the 1.14x that made it look like an
afterthought came from the same shape-level model that overstated steps 1 and 3, so what an int8
direct convolution is worth here is not known until step 3's 2-D case is understood. That is the next
task, and it is debugging rather than kernel writing.

### 6.8 Where the remaining ARMv6 performance is

Ordered by evidence, not by appeal. Everything here was measured on the board with the kernels of
§6.6 installed.

**1. SHIPPED — the direct-convolution predicate was reading a cache size from another machine.**
`ggml_conv_1d_direct_budget()` decides whether a convolution's weights are small enough for the direct
sweep to pay, and it asks `sysconf`. An ARM1176 reports **0 for every level**, so the function fell to
its 512 KB floor — on a core with a 16 KB L1 data cache and no L2 it can use, the BCM2835's 128 KB L2
belonging to the GPU. Thirty-two times too generous, and silently.

That matters more here than the arithmetic does. The direct sweep re-reads the WHOLE weight tensor
once per position block, and a block is four positions on this arm, so its weight traffic is **one
byte per MAC whatever the shape**. Inside L1 that byte is free; outside it the sweep is DRAM-bound and
a GEMM, which blocks both operands, wins by whatever the miss rate is. Per shape, one VITS synthesis,
`LOOM_PROFILE_NODES=1` at one thread, same wheel both arms:

| dst shape (OL,1,OC,1) | calls | weights | direct sweep | im2col + GEMM | |
|---|---|---|---|---|---|
| 70400,1,32,1  | 6  |  28 KB | **22529.7 ms** | 23714.6 ms | sweep by 5% |
| 17600,1,64,1  | 6  | 112 KB | 35290.3 ms | **22806.2 ms** | GEMM 1.55x |
|  2200,1,128,1 | 6  | 448 KB | 26598.5 ms | **11837.2 ms** | GEMM 2.25x |
|   275,1,384,1 | 28 | 1.47 MB | 17520.1 ms | 17854.1 ms | declines in both |
|    94,1,192,1 | 44 | 147 KB | 3800.7 ms | 3833.6 ms | declines in both |

The turn is between 28 KB and 112 KB, so `ggml-0019` gives the function an ARMv6 arm returning
**32 KB** — twice the L1, because this core's replacement policy is not LRU and a working set of about
twice the cache still takes roughly half its reads out of it, which is what the 28 KB row is.
`ggml-0006` grows **`GGML_CPU_CONV1D_BUDGET`** alongside it, a byte count that overrides the detection
entirely; that is what made every number below measurable.

**End to end on the board**, one wheel, one session, an idle board, VITS on 3.24 s of audio:

| | VITS | |
|---|---|---|
| before P7 — no ARMv6 kernels at all | 181.49 / 182.14 s | 57.0x real time |
| P7's kernels, `CONV1D_BUDGET=524288` | 98.96 s | |
| **+ the 32 KB budget, this item** | **80.96 / 80.87 / 80.88 s** | **1.223x** |

conformer-ctc and distilbert-ner have no convolution in this budget's range and do not move. The ASR
oracle transcribes the waveforms either side of the change alike as *"Hello world, this is a Raspberry
Pi Zero."*, 8/8 words.

**These numbers replace a set that was 25% high**, and the ones in §6.6 are high by about a third for
the same reason — a service on the board had entered a spin and was taking half the core, while `ps`
reported it at 5.7% because that column is an average over the process's lifetime. Ratios taken by
ABBA survived it (§6.6's 1.80x for VITS re-measures at 1.837x); seconds and real-time factors did not.
[Retro-037](../retros/retro-037-ps-said-six-percent-top-said-fifty.md).

**One thing this corrects about the switch it came from.** `GGML_CPU_DISABLE_CONV_HEURISTICS` gates
*two* decisions — this predicate and the im2col patch-batch budget — so the 1.27x it was first
measured at could not be attributed to either, and the "**15.8 MB of im2col at L=70400**" this section
used to warn about is a property of the switch's *other* half rather than of declining the direct
path: the fallback is `ggml-0004`'s **batched** im2col. It is still not a `return false`, but for the
opposite reason to the one recorded — at OC=32 the sweep wins.
[Retro-036](../retros/retro-036-one-switch-two-decisions.md).

**2. SHIPPED — the im2col patch-batch budget was sized against a cache eight times too big.**
`ggml-0004` caps a batch at 512 KB so the patches are still resident when the GEMM reads them back,
and that number is half of a Cortex-A72's 1 MB L2. This core's last level is a 16 KB L1. `ggml-0019`
gives it **64 KB**, and `ggml-0004` grows **`GGML_CPU_CONV2D_PATCH_BUDGET`** (bytes; 0 = no cap) so
that the cap can be moved without also moving the predicate.

Swept on an idle board, one wheel, one session, this cap the only variable:

| cap | 16 KB | 32 KB | **64 KB** | 256 KB | none | 512 KB |
|---|---|---|---|---|---|---|
| VITS | 80.58 s | 76.68 s | **76.37 / 76.07 / 76.35 s** | 77.16 s | 77.57 / 77.52 s | 80.92 / 79.71 / 80.06 / 80.88 s |

**1.061x**, and the shape is the interesting part rather than the size. There is a **floor as well as a
ceiling**: at 16 KB — the size that would actually fit this L1, which is what the rule's own reasoning
asks for — it is as bad as 512 KB, because a batch that small leaves the GEMM eight columns wide and
what is won in residency is lost in shape. So the constant is swept, not derived, and it is not
`l1 / 2`. **Uncapping entirely is not the answer either**, though it was the obvious guess and was
what this section previously predicted: 77.5 s, a full percent worse than 64 KB.

That prediction came from reading 1.149x off the old kill switch, on the contaminated board, for a
different set of buckets. Measured directly, on its own knob, on a quiet board, it is 1.061x.

**Where P7.1 leaves the board.** VITS **181.8 -> 76.2 s** across P7 and P7.1 together, **57.0x ->
23.5x real time**; conformer-ctc 10.9 s for 3 s of audio (3.62x real time) and distilbert-ner 5.07 s
for 17 tokens.

**3. SHIPPED — `CONV_TRANSPOSE_1D` was packing its two GEMM panels onto the same cache way.** The op
is **6.5%** of a synthesis rather than the 11.5% this list used to claim, and it holds no accumulator
array: `ggml-0009` already made the compute a GEMM. What was wrong is where the operands sit.

`gemm44` walks eight live streams — four rows of each operand, `lda`/`ldb` apart — and
`ggml_call_mul_mat_ldc` pins `lda = ldb = k`. This op packs its kernel at `wdata` and its transposed
activation at `wdata + nk`, and for the first node **`nk` is 524288 floats, exactly 2 MB** — a whole
number of this core's 4 KB L1 ways. The two panels are placed on top of each other by construction, so
the a-streams and the b-streams share sets four ways deep and evict each other on every access.
`ggml-0009` now skews the second panel by 16 floats, with the matching reservation in
`ggml_graph_plan`:

| | before | after | |
|---|---|---|---|
| K=16 Cout=128 Cin=256 L=275 | 1638.6 ms | **596.2 ms** | 2.75x |
| K=16 Cout=64 Cin=128 L=2200 | 1521.2 ms | 1199.9 ms | 1.27x |
| K=8 Cout=32 Cin=64 L=17600 | 1653.4 ms | 1439.7 ms | 1.15x |
| the op | 4813.1 ms | **3235.7 ms** | **1.49x** |
| VITS, ABBA on one board | 75.82 / 76.04 s | 74.69 / 75.20 s | **1.013x** |

**Bit-identical** — it is an address, not arithmetic — checked by hash on x86 across a whole synthesis
and on the board across all four ABBA arms. **No measurable change on x86** (min-of-7 on the op, 44.0
against 45.1 ms, inside that box's spread): the win is where the L1 is small, and the patch is inert
elsewhere rather than a speedup everywhere.

Two things this cost that are worth not repeating, both in
[Retro-038](../retros/retro-038-two-panels-on-one-cache-way.md). The axis looked like `n` — the tile
constant `GGML_CONV_TRANSPOSE_1D_TILE / mk` gives that node only 32 columns, and a cache constant
sized for a bigger machine is exactly what §6.8's first two items were — and **raising `n` makes it
worse**, 91.5 MMAC/s at n=32 down to 72.1 at n=272. And nodes 2 and 3 were read as "92% and 97% of the
226.7 MMAC/s roofline, effectively done", then got 27% and 15% faster: that roofline was measured at a
different shape and was never a property of the machine.

**4. MEASURED OUT — the OC=32 direct sweep, from both directions.** It is the last bucket taking the
sweep and the second largest in the graph (15505 ms of a 72 s synthesis), and neither obvious move
helps. **Routing it to im2col is 12% slower** end to end — 74.58 s against 83.58 s at
`GGML_CPU_CONV1D_BUDGET=0`, so the sweep is worth **1.12x** there rather than the 5% previously
recorded, and that is worth knowing because the arm it beats had since got faster. **Phase-major loses
on five of the six convolutions** and wins 1.08x on the sixth, which is 261 ms.

The rate does track what it looked like it tracked -- the tap window's byte span, `(k-1)*dilation`,
because a tile touches `IC * ceil(span/32)` of this core's 512 lines:

| span | 8 B | 16 B | 32 B | 72 B | 96 B | 288 B |
|---|---|---|---|---|---|---|
| MMAC/s, in the model | 207.7 | 198.1 | 154.1 | 136.3 | 135.8 | **107.9** |

-- but the **dense ceiling is only 165-191 MMAC/s** (`scripts/bench38.c`, arm B), so a layout fix
cannot reach the GEMM's rate on this core and the de-interleave costs 138-396 ms per call. Both entries
are in [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md).

**5. MEASURED OUT — `CONV_TRANSPOSE_1D`'s remaining data movement.** 545 ms of the op's 3236, split
overlap-add 389, transposes 120, kernel repack 35. A gather instead of the scatter-add is worth
**22 ms on a whole synthesis, 0.03%** (`scripts/bench39.c`), and that is the optimistic arm -- a
correct one has to carry the previous tile's last row across the boundary. The other two are
structural: the GEMM wants position-major and the tensor is channel-major, and ggml has no per-node
persistent scratch in which to cache a repack of a constant kernel.

**6. SHIPPED — the patch gather moved one element at a time down a straight line.** The im2col
gather computes, per ELEMENT, two strided coordinates, a four-way bounds test and a three-term
address: about fifteen instructions to move one float. On a 1-D convolution over a contiguous source
that is work spent walking a straight line — for a fixed `(ic, kx)` the elements successive patches
contribute are a **contiguous run** of the input, and only the destination stride varies. `ggml-0020`
takes the run at a time, resolving the bounds to two ends and a body once per run.

| | gather only, per element | across a synthesis |
|---|---|---|
| element at a time | 33-75 ns | 2.17 s |
| **run at a time** | **18-32 ns** | **0.98 s** |

| VITS, ABBA | 74.44 / 75.13 s | 69.15 / 69.05 s | **1.082x** |
|---|---|---|---|

**Bit-identical** — the same values to the same places in a different order — checked by hash on x86
across a whole synthesis and on the board across all four arms. conformer-ctc (10.8 s) and
distilbert-ner (5.17 s) do not move.

**And it is worth nearly five times what the isolated bench said**, which is the opposite of this
project's usual correction. `scripts/bench41.c` measures the gather alone at 1.19 s saved; end to end
it is 5.69 s, and `CONV_2D` as a whole drops 62697 -> 58159 ms. The element-at-a-time gather was
evicting the cache for the GEMM that reads its output immediately afterwards, so fixing its locality
pays twice. [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) exists because an
isolated number usually overstates; this is the case where it understated, and for a reason worth
recognising — a phase that interleaves with another in a tight loop is not separable by construction.

**7. MOSTLY MEASURED OUT — cache-blocking the GEMM's reduction.** `tinyBLAS`'s `mnpack` is a 4x4
REGISTER tile with no cache blocking, so the patch panel is re-read once per 4-column tile — 16x, 32x
and 96x at the three buckets' shapes. Blocking `k` so the panel slice stays resident wins on two of
seven shapes (2200 k=7 at **1.55x**, k=3 at 1.16x) and loses on five, for **1.76 s** available. A gate
derived from the L1 rather than fitted — block when the tile's eight operand strips exceed it, i.e.
`k > 512` — captures 1.58 s of that and classifies six of the seven correctly. Not taken for now: it
reorders the reduction, so unlike everything else in this section it would not be bit-identical, and
it needs accumulate variants of every tile kernel. `scripts/bench40.c`.

**THE ESTIMATE THIS SECTION CARRIED FOR THIS WORK WAS WRONG, and instructively.** It said 16-21 s was
in the im2col path, computed by dividing the buckets' MACs by 226.7 and 285.6 MMAC/s. Those are
rooflines measured at *other shapes* — the exact error
[Retro-038](../retros/retro-038-two-panels-on-one-cache-way.md) had just been written to record. At
these shapes the flat GEMM already runs at 114-262 MMAC/s and there was never 16 s in it. What was
really there: 5.7 s in the gather, 1.8 s in blocking, and the permute below.

**8. MEASURED OUT — the permute-back.** After the GEMM the result is scattered into `dst` with a
stride of `dst_w*dst_h` floats: a cache line per output element, 11.4 M of them across a synthesis,
and on a 128-set L1 the 70400-byte row pitch puts 64 output channels into 16 sets. Walking channels on
the outside instead makes the write contiguous and the read strided, which is **3.2x faster measured
on its own** — 69.6 -> 21.7 ms at the largest shape, 0.63 -> 0.19 s across a synthesis
(`scripts/bench42.c`).

**End to end it is 0.24 s SLOWER**: 68.98 / 69.00 s against 69.22 / 69.24 s, with brackets 0.02 s
wide. The `CONV_2D` buckets do improve by 239 ms and something outside them gets slower by more. Not
shipped. Bit-identical at one and four threads, correct, and not worth having.

Also measured and worse than both: letting the GEMM write straight into `dst` with
`ldc = dst_w*dst_h` — which `ggml-0004`'s own header proposes and the `defer` path uses — because the
bias pass that then remains has to read-modify-write the destination, 43.6 ms against 21.7. So
`defer`'s unreachability for the high-dilation convolutions (it needs
`patches_per_batch > (knl_w-1)*dilation`, 72 wanted against the 64 KB cap's 32) turns out not to
matter: the path it cannot take is not the one to want.

**THE PATTERN THIS SECTION KEEPS PRODUCING, now three times.** A phase of this op measured on its own
does not predict what it is worth in the graph, and **the sign is not predictable either**: the gather
understated by 5x (1.19 s alone, 5.69 s in place), this one overstates past zero (0.44 s alone, -0.24 s
in place), and the GEMM's cache blocking wins on two shapes of seven. These phases evict each other's
working sets, so the only number that decides anything here is an ABBA on the model.
[Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) carries all three.

**9. An int16 `__smlad` path is worth 1.19x and needs a type ggml does not have.** 270 MMAC/s against
the best F32 tile's 226.7 (`scripts/bench32.c`). Real, small, and a numerics change; last.

**Two things that look like opportunities and are not.** Padding conformer's `d_model` from 176 to 192
so its weights could be block-quantized is now a **pessimisation**: the F32 kernel reaches 226.7 where
the q4_0 one reaches 167.9, so quantizing those weights would make them slower, not faster. And
threading is not a lever at all -- the board has one core.

**One that is not FLOPs but raises the ceiling.** `distilbert-ner` carries **90.9 MB of F32
embeddings** out of its 131 MB RSS, declined by the exporter's op gate rather than by any shape rule
(`GET_ROWS` is not in `PACKED_WEIGHT_FIRST_OPS`). Quantizing it would cut the model's footprint about
2.5x, which on 512 MB is the difference between what fits and what does not.

### 6.9 What is still not there

* **No PyPI install.** `pip install <release-asset URL>`, and README says so rather than letting pip
  fall through to building an sdist on a 1 GHz core.
* **No armv7l wheel.** The build path is the same and no board here runs it.
* **RAM is the ceiling, not the ISA.** Both Zeros have 512 MB, and `src/core/gguf_model.cpp` parses
  with `no_alloc=true` and then reads weights into a backend buffer — there is no mmap path. The
  0.6B ASR and LM models do not fit at any quantization this repo ships
  ([ADR-017](../adrs/adr-017-no-k-quants.md)).
* **The README's Pi 4 columns do not transfer.** Every optimisation in
  [Epic-05](epic-05-edge-performance.md) is `__aarch64__`- or AVX2-guarded, so this rung runs at the
  generic-C level.
