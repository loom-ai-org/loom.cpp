---
type: reference
domain: build
tags: [build, cmake, ggml-patches, arm, gpu, vulkan, testing, fixtures]
last_updated: 2026-10-24
---

# Building and testing loom.cpp

Building the engine from source, the patched `ggml` it builds against, GPU backends, and the two test
classes. Moved here from the README on 2026-10-24, unchanged. Packaging and wheels are
[Epic-08](epics/epic-08-packaging-and-release.md).

## Building

```sh
cmake -B build
cmake --build build -j"$(nproc)"
```

Dependencies (`ggml`, `nlohmann_json`, LuaJIT) are fetched by CMake; nothing else is needed to build
and run the hermetic suite. The fetched `ggml` is patched at configure time from `cmake/patches/` —
fifty diffs at present ([`UPSTREAM.md`](../cmake/patches/UPSTREAM.md) describes each one). The first eleven fix GCC's code generation for ggml's ARM F32 GEMM (1.6x), the matmuls it
declined to accept at all — by row count and by contraction length — a fused convolution that batched
its work too coarsely to stay in cache, a direct 1-D convolution for long activations with small
weights, the elementwise nodes a vocoder's resblock wraps around every convolution — its bias, its
leaky ReLU and its residual — none of which now costs a pass over memory of its own, an exact-erf GELU
that was a scalar libm call per element, and `conv_transpose_1d`'s single-threaded prologue and
dot-product-at-a-time compute; see `cmake/GgmlPatches.cmake` for the rules such a patch has to meet.
**`ggml-0011` is why `UPSTREAM.md` now requires a number from an x86 box AND one from the Pi before a
patch here is called done** — it was measured on one ISA, shipped to both, and regressed the other by
1.3-1.75x.

Two of them are heuristics tuned on measured hardware, so they carry a run-time escape:
`GGML_CPU_DISABLE_CONV_HEURISTICS=1` declines both, the way ggml's own `GGML_CPU_DISABLE_FUSION`
declines its fusions.

**32-bit ARM (ARMv6, ARMv7) builds from this checkout with no flags of its own.** There is no
ISA-specific code in this repo — `grep -rE '__aarch64__|__ARM_NEON|immintrin|AVX' src include tools`
returns nothing — so `__ARM_NEON` simply goes undefined, ggml takes its scalar arms, and the patches
above take their generic ones. Two things are worth knowing before you do it. `GGML_NATIVE` is off in
a wheel build and this path passes no `-march`, so the baseline is **the compiler's own default**:
build in the userland the binary is for, or a Debian armhf toolchain's `armv7-a+neon` will produce
something that dies on a Pi Zero. And `GGML_CPU_ALL_VARIANTS` cannot be used at all here — ggml's ARM
variant ladder is AArch64-only, so a 32-bit build is one un-split `libggml-cpu.so`. `loom-py` handles
both for you; [Epic-08 §6](epics/epic-08-packaging-and-release.md) is the whole story, including
the emulated Raspbian the `linux_armv6l` wheel is built in.

There is one build option of this repo's own, `-DLOOM_TINYBLAS=OFF`, which drops ggml's blocked GEMM
(`GGML_LLAMAFILE`) back out again. It exists to make GEMM measurements A/B-able and defaults **on**,
where it is worth ~2x on x86-64 and 1.6x on aarch64 at convolutional shapes ([Epic-05](epics/epic-05-edge-performance.md)).

## Running on a GPU

A default build is CPU-only. Compiling a device backend in is a `ggml` option, passed straight through
— this repo adds no option of its own here, because there is nothing per-backend for it to decide:

```sh
cmake -B build -DGGML_VULKAN=ON     # or -DGGML_CUDA=ON, -DGGML_METAL=ON, -DGGML_SYCL=ON, ...
cmake --build build -j"$(nproc)"

build/tools/loom_cli/loom_cli --list-devices
build/tools/loom_cli/loom_cli --device gpu --model model.gguf --wav audio.wav
```

`--device` takes `auto` (the default, and what `$LOOM_DEVICE` sets), `cpu`, `gpu`, or a device name
such as `Vulkan0`. **`auto` prefers a device and falls back to the CPU; `gpu` is an error when there
is none**, because a caller who spelled it out is asking a question about the machine, and answering it
with a silent CPU run turns "no GPU here" into an unexplained performance number.

**The Vulkan build tools sort themselves out.** `glslc` and the Vulkan headers on a stable distribution
are likely too old for `ggml`'s Vulkan backend, and both fail in ways that name neither cause — so
`cmake/VulkanToolchain.cmake` probes for those two failures specifically and, when it finds them,
fetches pinned Vulkan-Headers and builds `glslc` from a pinned `shaderc` into the build directory. That
costs several minutes on the first configure of a machine that needed it, and nothing at all on one that
did not. `-DLOOM_VULKAN_FETCH_TOOLCHAIN=OFF` turns the diagnosis into an error naming what to install
instead, which is the right setting for an image that provides its own toolchain.

**A primitive can choose its own lowering.** Some ops are host callbacks through `ggml_map_custom` (a C
function pointer, so there is nothing for a GPU to dispatch) and others are real ggml ops a given backend
happens not to implement. Either way a primitive builds what the topology asked for, asks
`ggml_backend_supports_op`, and either keeps it or emits an equivalent — so the same GGUF lowers
differently per backend and the file on disk keeps saying what the model does. Every device run still
carries a CPU backend behind it for whatever is left. The CLI prints where each module actually ran;
`ctest -L gate -R device_parity` checks that a device gets the same answer as the CPU.

## Testing

Two classes of test, and a test's own directory is which class it is in.

```sh
ctest --test-dir build -L ci      # hermetic: builds its own fixtures, seconds, what CI runs
ctest --test-dir build -L gate    # real exported checkpoints; skips cleanly without them
```

`tests/ci/` needs nothing but this repo, a toolchain and `gguf`+`numpy` — every fixture it reads is
generated from `tests/fixtures/*.py` by a ctest step that runs first. `tests/gate/` compares against
real models: gigabytes that cannot live in git and hours that cannot run in CI, so each of those tests
exits 77 (Skipped) when its fixture is absent, and a developer with none of them still gets a green
suite that means *nothing hermetic broke*.

Point the gate suite at its fixtures with one variable:

```sh
export LOOM_FIXTURES=~/loom-fixtures
scripts/fixtures.py status    # what the gate tests want, and what you have
scripts/fixtures.py fetch     # from the published fixture repo
```

Every fixture also still honours the per-test variable it always had
(`LOOM_KOKORO_MIL_GGUF`, …), which wins over the root — pointing one test at one artifact you just
rebuilt is what you do while working on it.
