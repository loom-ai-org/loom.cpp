---
type: epic
status: active
domain: packaging
last_updated: 2026-08-31
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
| Decisions | [ADR-011](../adrs/adr-011-three-repositories.md), [ADR-009](../adrs/adr-009-backends-as-dynamic-libraries.md) |
| Retros | [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md), [Retro-024](../retros/retro-024-a-blocker-read-from-one-half-of-an-agreement.md) |
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

