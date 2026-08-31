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
| Retros | [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md) |
| Active tasks | [Backlog → Packaging](../backlog/active-index.md#packaging--release) |

## 4. Planned: macOS wheels (P4.10) — SCOPED, lands before new model families


**Why before P5 rather than after.** P5 adds model families, and a family lands in `loom-py` for free
— a model the bindings have never heard of works the day the exporter can produce it. That is true
only on a platform that exists. Every model P5 adds is unreachable from a Mac until this is done, so
this multiplies P5's value while P5 does nothing for it. It is also the item most visible from
outside, since it is what `pip install loom-py-rt` can resolve to.

**What already ships (2026-08-16):** `manylinux_2_28` on x86-64 and aarch64, CPython 3.10–3.13. The
aarch64 wheel serves Raspberry Pi 4 and 5 on 64-bit Raspberry Pi OS — see `loom-py`'s
`raspberry-pi-check` job, which executes the Cortex-A72 rung under QEMU because no CI runner is that
old. 32-bit Raspberry Pi OS (`armv7l`) is deliberately NOT supported: `GGML_CPU_ALL_VARIANTS`'
Linux-ARM ladder is AArch64-only, so it is an unbuildable configuration rather than a slow one.

**The two Apple targets, and what they are called.** Apple Intel is plain `x86_64`
(`macosx_10_13_x86_64`). Apple Silicon is `arm64` in Apple's naming — the same ISA Linux calls
`aarch64` — and tags as `macosx_11_0_arm64`.

**The variant ladder is already solved for both, which is the one thing that needs no work.** ggml's
`GGML_CPU_ALL_VARIANTS` has an `elseif (APPLE)` arm: `apple_m1` (DOTPROD), `apple_m2_m3` (+I8MM),
`apple_m4` (+SME). Every Apple Silicon part has dotprod, so the lowest rung covers M1 and there is no
baseline hole of the kind that made every pre-2026-08-14 wheel require AVX2. Apple Intel resolves to
`GGML_SYSTEM_ARCH == "x86"`, whose ladder starts at a true `x64` baseline.

**THERE IS NOW A MAC, as of 2026-08-31: `fdemelo@macbook-pro`, passwordless ssh.** Apple **M1 Pro**,
arm64, macOS **15.6.1** (24G90), 10 cores, L1d 64 KB / L2 4 MB / no L3. That is the **`apple_m1` rung**
— DOTPROD, no I8MM, no SME — which is the *lowest* of ggml's three Apple rungs and therefore the right
verification target: it is the baseline every `macosx_11_0_arm64` wheel has to serve, and an M4 runner
could never have stood in for it. It does **not** cover Apple Intel, which stays a CI-only row.

**What is on it, because this decides what can be attempted before installing anything.** Homebrew
**6.0.18** at `/opt/homebrew`, with **cmake 4.4.2** and git; **mambaforge** at `~/mambaforge`, base
Python 3.10, envs `env-py-3.11` (3.11.12, arm64 — **use this one**), `env-py-3.13`, `env-py-3.9`,
`hummingbot`. Apple clang **17.0.0**, and **Command Line Tools only — no full Xcode**
(`xcode-select -p` is `/Library/Developer/CommandLineTools`), so `xcodebuild` and `xcrun metal` both
error; see Epic-04 §5, where that turns out **not** to block Metal. Missing and likely wanted:
**`ninja`** and **`pkg-config`**, both a `brew install` away.

**The homebase is `/Users/fdemelo/loom`, and it is already staged** (2026-08-31) — the same layout as
`~/Dev/loom` here, so a path in one repo's docs resolves on both:

```
/Users/fdemelo/loom/{loom.cpp, loom-exporter, loom-py}           # clean trees at the current HEADs
/Users/fdemelo/loom/hf-models/vits-{f32,q4_0}-dyn.gguf           # 62.8 MB / 11.7 MB, post-P4.28
/Users/fdemelo/loom/hf-models/whisper-small/whisper-small.gguf   # the ASR oracle, 16 kHz only
```

1.1 GB of 207 GB free, and **no build directories** — they are Linux objects and would only confuse a
macOS configure. Rosetta 2 is installed, which matters only if the `macosx_10_13_x86_64` wheel is ever
smoke-tested locally rather than in CI.

**Three rsync traps, all already paid for here.** (1) The Mac's clock is **241 s behind** this host, so
freshly-synced files carry future mtimes and a build system will loop on stale objects — `touch`
everything newer than now after any sync, exactly as the workstation precedent requires. (2) macOS ships
**openrsync** ("rsync version 2.6.9 compatible"), which rejects `--info=stats2` and other modern flags:
keep to `-a --stats`. (3) `--exclude 'build-*'` is a **glob, not a directory match**, and it silently
swallowed `loom-py/.github/workflows/build-image.yml` on the first pass.

**THE SSH INVOCATION GOTCHA, which cost one wrong inventory already.** `ssh macbook-pro 'cmd'` runs a
**non-login, non-interactive** shell, so it sources neither `~/.zprofile` (Homebrew's `shellenv`) nor
the conda init block: `brew`, `cmake` and `conda` all report as absent and the bare `python3` resolves
to the system **3.9.6**. Use **`ssh macbook-pro 'zsh -lc "..."'`** for anything on the Homebrew path,
and note that **`conda` is not on the login PATH either** — reach the interpreter by absolute path,
`~/mambaforge/envs/env-py-3.11/bin/python`, or activate explicitly via `~/mambaforge/bin/conda`.

**Four blockers, in the order a build hits them.** All four were established by reading the pinned
sources on 2026-08-16 and **re-verified against the current pins on 2026-08-31**; none has yet been
observed on the Mac, which is now a thing that can be done rather than a standing limitation.

1. **LuaJIT stops the build outright.** `luajit-src/src/Makefile:321-322` is a hard
   `$(error missing: export MACOSX_DEPLOYMENT_TARGET=XX.YY)` on Darwin, and `cmake/Dependencies.cmake`
   invokes `make -C ... BUILDMODE=static XCFLAGS=-fPIC` with no environment at all. Engine-side fix,
   and the cheap one: the same variable cibuildwheel needs anyway to tag the wheel (`11.0` on arm64,
   `10.13` on x86-64), so one export does both jobs.
2. **`loom-py`'s install rule ships nothing.** `CMakeLists.txt`'s
   `install(DIRECTORY ... FILES_MATCHING PATTERN "*.so*")` — on macOS the engine and ggml libraries are
   `.dylib`, so the wheel would contain `_loom.so` alone. That revives the exact
   `cannot open shared object file` failure the same file's comment records as already fixed, through a
   different door, and only at import time.
3. **`$ORIGIN` is ELF-only.** `loom-py/CMakeLists.txt` sets it on both install and build RPATH; macOS
   needs `@loader_path`. There is no `if(APPLE)` anywhere in either repo's CMake — grepped, not assumed.
4. **The one that would burn a day of CI: ggml's DL loader looks for the wrong extension on macOS.**
   `ggml-backend-reg.cpp`'s `backend_filename_extension()` returns `.dll` on `_WIN32` and `.so`
   otherwise — there is no `.dylib` case, and its only `__APPLE__` block is `get_executable_path()`.
   CMake emits `.dylib` for shared libraries on macOS and nothing in ggml's build overrides `SUFFIX`.
   So a macOS `GGML_BACKEND_DL` build plausibly produces backends its own loader will never find by
   name, which with DL means **zero devices, including no CPU** — the failure mode P4.8a already
   showed is silent. Confirm on a Mac first; the fix is `SUFFIX ".so"` on the backend targets in our
   build, or upstream. Do not start the macOS work by writing a workflow — start by settling this.

**The wheel shape: two wheels, not `universal2`.** cibuildwheel builds each natively (`macos-14` for
arm64, `macos-13` for x86-64), a fat binary doubles the download for everyone to serve one half, and
the half it serves is ending — macOS 26 is Apple's last Intel release, and an Apple Silicon Mac can
resolve the x86-64 wheel under Rosetta 2 anyway. **Do Apple Silicon first and treat Intel as one extra
matrix row**, not as an equal target.

**Metal is a separate package, and is NOT part of this item — it is P4.11, below.** `loom-py-rt-metal` is
`packaging/rt-vulkan/` with four strings changed (`packaging/README.md`), and it is the only
accelerator that would ever apply to a Mac — CoreML is not Metal, and no ggml backend targets the
Neural Engine. Blocker 4 has to be settled before a `[metal]` package can mean anything, since a
backend `.dylib` is discovered by the same code path. Open question to answer then, not now: whether
`GGML_METAL_EMBED_LIBRARY` is required for a DL-loaded Metal backend that travels without a bundle.

**Verification: CI for both architectures, plus a real M1 Pro for arm64.** The CI checks are the wheel
test step running `pytest tests/ci` on both runner architectures, and `tests/ci/test_cpu_variants.py`
gaining `arm64 → libggml-cpu-apple_m1.*` in its baseline table (the extension follows blocker 4's
resolution). **The `macbook-pro` above is the analogue of `raspberry-pi-check` that this item was
scoped as lacking** — it is a genuine `apple_m1` part, so it answers the question an M4 runner cannot:
does the lowest rung actually run. **Apple Intel remains CI-only**, and for that row "supported" still
means "built and imported in CI" — say so in the platforms table rather than implying parity.

**Done means:** wheels for `macosx_11_0_arm64` and `macosx_10_13_x86_64` published for 3.10–3.13;
`import loom` and `loom.devices()` reporting a CPU on both; `pytest tests/ci` green on both;
`loom-py`'s *Supported platforms* table extended; blocker 4 answered in writing either way; and — new,
now that the hardware exists — **the arm64 wheel installed from the index on `macbook-pro` and a model
run on it**, which is a stronger bar than this item was originally scoped with and should not be
dropped back to the CI one. Windows stays out of scope and stays behind this.


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

