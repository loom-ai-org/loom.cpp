---
type: adr
status: accepted
date: 2026-11-06
tags: [packaging, windows, msvc, luajit, loom-py, ggml-patches]
supersedes: []
---

# ADR-096: Windows Builds With MSVC, and LuaJIT Is Built by CMake There

## Context

Windows was out of scope until 2026-11-06 (loom-py's `pyproject.toml` said so). The engine itself had
few POSIX dependencies: `backend.cpp` was already `__linux__`-guarded, and nothing else in `src/` calls
POSIX. The build did have them:

1. **LuaJIT is built by its own Makefile** (`cmake/Dependencies.cmake`, `make -C src`). There is no
   `make` on a Windows build machine, and LuaJIT's documented Windows route, `src/msvcbuild.bat`, has two
   problems of its own. It refuses to run outside a developer prompt (`if not defined INCLUDE goto
   :FAIL`), and under the Visual Studio generator a custom command does not inherit one. That is
   the generator scikit-build-core and cibuildwheel use on Windows. And its `static` mode compiles with
   no CRT flag, i.e. `/MT`, while everything else is `/MD`. The library ends up inside `loom_engine.dll`,
   so the DLL would carry two C runtimes, and the linker reports it only as warning LNK4098.
2. **ggml-0006 includes `<unistd.h>` unconditionally** in `ops.cpp`, for the conv1d budget's `sysconf`.
3. **A Windows DLL exports only what is marked**, and neither `loom_engine` nor ggml's private
   `ggml_graph_view` (used by `profile.cpp`) is marked.
4. Smaller: a GCC `format` attribute in `profile.cpp`, `M_PI` without `_USE_MATH_DEFINES`, UTF-8 sources
   read in the system codepage, and loom-py's wheel rule matching `*.so`/`*.dylib` only.

## Options

**1. MinGW-w64 / llvm-mingw.** LuaJIT's Makefile works there. Rejected: CPython on Windows is built
with MSVC. A pybind11 extension built against another C runtime is a supported-but-fragile
arrangement, and every Windows wheel on PyPI that a user mixes ours with is MSVC.

**2. clang-cl.** It is what llama.cpp's Windows releases use for CPU speed. But it still needs the MSVC
headers and libraries, and it does not remove the LuaJIT question. Left open as a later
speed item: it changes only the compiler, not anything decided here.

**3. MSVC, with `msvcbuild.bat`.** Rejected for the two reasons in Context.

**4. MSVC, with LuaJIT's bootstrap transcribed into CMake targets.**

## Decision

Option 4. `cmake/LuaJitMsvc.cmake` is `msvcbuild.bat`'s x64 path as targets: `minilua` → DynASM
(`vm_x64.dasc`, the .bat's flags) → `genversion.lua` → `buildvm` (`peobj` plus the five tables) → a
static `luajit_static` from `lj_*.c lib_*.c` and `lj_vm.obj`. It uses CMake's compiler and runtime, so
it works under every generator and is `/MD` like everything else. It defines the same `luajit::luajit`,
so nothing downstream changes. x64 only; ARM64 needs a cross-built host toolchain and nobody has asked.

The rest, each where it belongs:
* `ggml-0059-windows-msvc.patch`: the `<unistd.h>` guard, a `GetLogicalProcessorInformation` arm for
  the conv1d cache probe, and `GGML_API` on `ggml_graph_view`, which its neighbours in `ggml-impl.h`
  already have for the same cross-DLL reason. Upstream, the first two belong in PR 6.
* `loom_engine`: `WINDOWS_EXPORT_ALL_SYMBOLS` (inert unless SHARED), `/utf-8`, `_USE_MATH_DEFINES`.
* `backend.cpp`: a Windows physical-core count (`GetLogicalProcessorInformationEx`, process affinity,
  highest EfficiencyClass only), the Linux and Apple rules in Windows' vocabulary.
* loom-py: `$<0:>` on the output directories, so the VS generator does not append `Release/` (without
  it, the install rule copies `runtime/` and the wheel ships no DLLs), and `*.dll` in that rule.

## Consequences

* **Verified on a Celeron N4100 (Zotac ZBOX PI335, Windows 10 22H2, MSVC 19.44, CMake 4.4)**:
  `python -m build --wheel` with no developer prompt builds `loom_py_rt-…-win_amd64.whl` (5.1 MB, nine CPU
  variants) in 16 min and `loom_py_rt_vulkan-…-py3-none-win_amd64.whl` in 13 min. loom-py `tests/ci`: 182
  passed, 2 skipped, against the installed wheel. Exactly one CPU variant stays resident (`sse42` there), as
  on Linux. The Vulkan wheel finds the SDK's own headers (`VulkanToolchain.cmake`'s probe now looks where
  `find_package(Vulkan)` does) rather than fetching them.
* **Output matches Linux.** On 73 LibriSpeech clips, Citrinet-256 transcribes identically to Linux at F32 and
  Q8_0_PC, and Zipformer small identically at F32 (69/73 at Q8_0_PC, same WER). Speed on the N4100, 4 threads,
  mean of 3 interleaved launches (ASR: 81 s of audio; TTS: 3 sentences after a warm-up):

  | RTF, F32 / Q8_0_PC | CPU | Vulkan (UHD 600) | Vulkan speed-up |
  |---|---|---|---|
  | Citrinet-256 | 0.179 / 0.109 | 0.089 / 0.082 | 2.02x / 1.34x |
  | Zipformer small | 0.520 / 0.449 | 0.303 / 0.337 | 1.72x / 1.33x |
  | VITS (miro) | 0.917 / 0.936 | 1.713 / 0.886 | 0.54x / 1.06x |
  | Matcha | 13.26 / 13.03 | 6.41 / 6.96 | 2.07x / 1.87x |

  VITS F32 on Vulkan is slow on two specific sentences of the three timed: 2.2-2.8x their neighbours, in every
  launch, and on that arm only. Over sentences 4-10 its RTF is 0.89. That is a hub item. Q8_0_PC gains little on
  the GPU, and is slower than F32 there on Zipformer and Matcha.
* **MSVC's x86 ladder is nine rungs, not fourteen** (no zen4, sapphirerapids, ivybridge, piledriver,
  cooperlake). It starts at the same `x64` baseline.
* **loom's own x86 kernels are AVX2-only.** On SSE4.2 the conv1d direct tile is scalar
  (`GGML_CONV1D_LANES 1`) and depends on auto-vectorisation, which MSVC does less of than GCC. Measured on
  Linux, a build limited to the SSE4.2 variant costs 1.5x (Zipformer) to 2.5x (VITS). A clang-cl build (Option
  2) is the open speed item.
* **Seeded output is not portable across C++ standard libraries.** The Lua bridge draws through
  `std::normal_distribution` / `std::uniform_real_distribution`, whose algorithms the standard leaves
  to the implementation. The same seed gives a different VITS duration on Windows than on Linux, and the
  TTS oracle WERs differ by one or two words in 80 for that reason alone. This predates the port (macOS's
  libc++ is a third implementation) and is tracked in the hub.
* **Vulkan builds and runs on the UHD 600, and its F32 output matches the CPU's** (VITS SNR >= 91.9 dB,
  Matcha >= 44.2 dB, ASR transcripts identical). At Q8_0_PC its ASR WER stays within 0.1 point of the CPU's
  (Citrinet 70/73, Zipformer 70/73 transcripts identical to Linux) and its TTS WER within one word in 80.
  It costs RAM: the peak working set is 60-390 MB above the CPU's for the same file, on a 4 GB box.
* **Vulkan on that box is stable on a correct supply.** On the first adapter, which did not match the box's
  5 V 4 A rating, sustained Vulkan load froze or switched off the machine seven times. That was independent of
  precision, heat, the display, memory, the GPU timeout and the driver version (31.0.101.2112, .2115). On a
  5 V 4 A supply (2026-11-10, driver .2140), all 38 Vulkan launches completed over about an hour of GPU load.
  The board peaked at 50 °C, and Windows applied no thermal limit.
* A runner job (`windows-latest`) in loom-py's `wheels.yml` is what makes this a shipped platform.
  Until then the wheel is built by hand.

## Related

* [Epic-08](../epics/epic-08-packaging-and-release.md) — packaging, and the macOS port this follows.
* `cmake/patches/UPSTREAM.md`, PR 6 — where ggml-0059's first half belongs upstream.
