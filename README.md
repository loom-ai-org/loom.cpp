<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-inline-dark.svg">
    <img src="assets/logo-inline.svg" alt="" width="52" align="middle">
  </picture>
  &nbsp;loom.cpp
</h1>

An inference engine built on [`ggml`](https://github.com/ggml-org/ggml) that hardcodes no model. A
model is a GGUF that carries its own **graph topologies** as JSON metadata and its own **driver
script** as embedded Lua, alongside the weights those describe; the engine parses them and builds the
compute graph at run time. Adding a model architecture is an export, not a C++ patch.

That is the whole architectural bet, and it is why this repo stays small: it targets edge devices, so
every per-model decision belongs in the exporter where it costs a Python change instead of a
specialized C++ driver. See [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## What the engine offers a host, and what it leaves alone

The engine hardcodes no model, but it does own the loops every host would otherwise write. Those are
**per-task, not per-model** — one CTC decoder covers every CTC model — and they live here because the
copies hosts wrote had already drifted apart from each other:

| | |
|---|---|
| `loom::text::generate` | the causal-LM decode loop, both driver shapes, the file's own stop token |
| `loom::audio::transcribe` | long-form ASR: windowing, segment splitting, and the seek to where the model closed its last segment |
| `loom::Session` | topologies registered and caches attached, in an order that cannot dangle |

Underneath them the low-level surface is unchanged and stays raw: `LoomLuaBridge::call` invokes the
driver with the driver's own arguments. The split is the same rule the whole tree is built on —
**in the file when it is a property of the checkpoint, in the engine when it is a property of the
task, in the host when it needs the host's ecosystem** — and
[`docs/HIGH-LEVEL-API.md`](docs/HIGH-LEVEL-API.md) is where it is argued.

A model says what it is, so a host never has to recognise one:

```cpp
const loom::ModelContract contract = loom::ModelContract::read(model);
contract.task;             // "automatic-speech-recognition"
contract.interface_name(); // "speech2text" -- the modality pair a host offers a door for
```

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | this one — the runtime, its primitives and its Lua bridge |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | turns a PyTorch checkpoint into a GGUF this engine runs |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | Python bindings, with the engine as a submodule |

## Supported models

Seventeen, published at [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org). Each is a
single GGUF carrying its own topologies, driver and — where the architecture has one — its vocabulary,
so the engine runs all of them without a line of per-model C++.

### Language models

| Model | Exported from |
|---|---|
| [`loom-ai-org/qwen3-0.6b-base-loom`](https://huggingface.co/loom-ai-org/qwen3-0.6b-base-loom) | [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| [`loom-ai-org/lfm2-350m-monolithic-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-monolithic-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/lfm2-350m-modular-loom`](https://huggingface.co/loom-ai-org/lfm2-350m-modular-loom) | [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) |
| [`loom-ai-org/smollm2-360m-instruct-loom`](https://huggingface.co/loom-ai-org/smollm2-360m-instruct-loom) | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| [`loom-ai-org/gemma-3-270m-it-loom`](https://huggingface.co/loom-ai-org/gemma-3-270m-it-loom) | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) |

The two LFM2 entries are the *same checkpoint exported two ways* — one fused graph against one topology
per layer — which is how the engine's two decomposition paths stay honest about producing the same
model.

### Speech recognition

| Model | Exported from |
|---|---|
| [`loom-ai-org/whisper-small-loom`](https://huggingface.co/loom-ai-org/whisper-small-loom) | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) |
| [`loom-ai-org/conformer-ctc-small-loom`](https://huggingface.co/loom-ai-org/conformer-ctc-small-loom) | [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) |
| [`loom-ai-org/parakeet-tdt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-tdt-0.6b-loom) | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| [`loom-ai-org/parakeet-rnnt-0.6b-loom`](https://huggingface.co/loom-ai-org/parakeet-rnnt-0.6b-loom) | [`nvidia/parakeet-rnnt-0.6b`](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) |
| [`loom-ai-org/gigaam-v3-rnnt-loom`](https://huggingface.co/loom-ai-org/gigaam-v3-rnnt-loom) | [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3) |
| [`loom-ai-org/qwen3-asr-0.6b-loom`](https://huggingface.co/loom-ai-org/qwen3-asr-0.6b-loom) | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| [`loom-ai-org/granite-speech-4.0-1b-loom`](https://huggingface.co/loom-ai-org/granite-speech-4.0-1b-loom) | [`ibm-granite/granite-4.0-1b-speech`](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) |

Each takes a raw waveform: the mel frontend is inside the graph, not in front of it.

### Speech synthesis

| Model | Exported from |
|---|---|
| [`loom-ai-org/kokoro-82m-loom`](https://huggingface.co/loom-ai-org/kokoro-82m-loom) | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| [`loom-ai-org/matcha-tts-ljspeech-loom`](https://huggingface.co/loom-ai-org/matcha-tts-ljspeech-loom) | [Matcha-TTS (LJSpeech checkpoint)](https://github.com/shivammehta25/Matcha-TTS) |
| [`loom-ai-org/supertonic-2-loom`](https://huggingface.co/loom-ai-org/supertonic-2-loom) | [`Supertone/supertonic-2`](https://huggingface.co/Supertone/supertonic-2) |
| [`loom-ai-org/vits-piper-en-gb-miro-loom`](https://huggingface.co/loom-ai-org/vits-piper-en-gb-miro-loom) | [`OpenVoiceOS/pipertts_en-GB_miro`](https://huggingface.co/OpenVoiceOS/pipertts_en-GB_miro) |
| [`loom-ai-org/styletts2-ljspeech-loom`](https://huggingface.co/loom-ai-org/styletts2-ljspeech-loom) | [`yl4579/StyleTTS2-LJSpeech`](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) |

**Only Supertonic takes text.** It encodes graphemes itself and its GGUF carries the codepoint table;
the other four consume *phoneme* ids that a phonemiser produces outside the engine, so their files
embed no vocabulary at all. That is a real limitation of those checkpoints, not a missing feature here.

## Performance against onnxruntime

The reference question for an edge runtime: **the same checkpoint, on the same machine, at the same
thread count — how does loom compare to onnxruntime?** Three tasks, because no single one is
representative: an all-convolutional vocoder, an encoder-decoder ASR model, and an autoregressive LM.

**`>1.00x` means loom is faster.** Measured 2026-08-24, **both engines run back to back on the machine
in the row** — see [what these numbers are not](#what-these-numbers-are-not).

| machine | arch | threads | TTS<br>VITS (piper en-GB) | LM<br>Qwen3-0.6B | ASR<br>whisper-small |
|---|---|---|---|---|---|
| Intel Core Ultra 9 285K | x86-64 | 4 | **1.03x** | **1.02x** | 0.71x |
| Intel Core Ultra 9 285K | x86-64 | 24 (all) | **1.17x** | **1.03x** | **1.29x** |
| AMD Ryzen 3 3250U | x86-64 | 4 (all) | **1.03x** | **1.05x** | 0.69x |
| Raspberry Pi 4B | aarch64 | 4 (all) | 0.98x | **1.08x** | 0.57x |

**TTS and the LM are at parity; ASR is still behind at four threads.** The caveats below matter as
much as the numbers.

* **TTS is at or just above parity everywhere.** That is what P4.14/P4.15 was for: a built-in F32 GEMM
  micro-kernel, four convolution patches to the pinned `ggml`, and a duplicated text encoder removed
  from the export.
* **The LM is at parity**, a few percent ahead everywhere. It only just became so — until 2026-08-23
  the engine called every causal-LM driver's `infer` rather than its `infer_with_past`, so the host
  re-fed a growing prompt and each token recomputed the whole sequence. That was worth 2.83x. Note that
  **its 24-thread figure is the LM at a thread count that does not suit it**: a decode step's `mul_mat`
  has `ne1 = 1`, so this task peaks at 8 threads and plateaus after. That shows up as spread rather than
  as a loss — loom ranges 24.5-28.6 tok/s across nine runs there where onnxruntime holds 27.1-27.6.
* **ASR is the one still behind, and it is now 1.4–1.8x rather than 2.4–3.6x.** Whisper's exported
  driver used to hand the decoder the raw encoder output every step, so cross-attention K/V was
  re-projected over all 1500 encoder frames per token — `MUL_MAT 768x1500` ran 684 times for an
  11-second clip where the encoder itself needs 48, **57.7% of ASR runtime**, where onnxruntime computes
  those tensors once. Exporting the projected K/V as its own topology is worth **2.4–2.6x on every
  machine here** (on the Pi, 90.0 s to 37.4 s for the same clip). What is left is a genuine gap at four
  threads and a **win at 24**, which is the clearest sign that what remains is thread-scaling headroom
  rather than a second defect of the same kind.
* **loom used to stop scaling at 8 threads and go backwards; that is fixed, and the 24-thread row is
  the fix.** VITS on the 285K was 0.080 s at 8 threads and **0.191 s at 24 — the same as at one
  thread**. The cause was not in this repository: `ggml` defaults to OpenMP, so `ggml_barrier` was
  `#pragma omp barrier` — one after every non-empty graph node, 2520 of them in a synthesis — and
  libgomp's default wait policy **slept every thread on a futex at each one** (334,609 voluntary
  context switches per 5 syntheses, against 160 now). Building `ggml` against its own threadpool, whose
  spin is bounded, made the curve monotonic again and is worth **4.8x at 24 threads**
  ([Retro-017](docs/retros/retro-017-libgomp-slept-at-every-graph-node.md)).
* **The engine still defaults to 4 threads whatever the machine has**, which is `ggml`'s default and is
  why the 285K has two rows. `$LOOM_N_THREADS` overrides it. Measured across both x86 parts, the best
  default would be the **physical core count** — 24 on the 285K, but **2 rather than 4 on the
  two-core-plus-SMT Ryzen**, where 4 is slower than 2 on all three tasks. That change is not made here:
  every figure on this page is one inference at a time on an idle machine, and a host running several
  loom instances concurrently is exactly the case `ggml`'s conservative 4 suits.

### What these numbers are not

**The baseline is `pip install onnxruntime` 1.28.0**, the same distribution channel `loom-py` ships
through. That choice flatters loom: at the *identical* version, the conda-forge build synthesises VITS
in 0.065 s where the PyPI wheel takes 0.120 s — **1.86x apart, same machine, same script**. Against
conda-forge, the x86 TTS wins above become losses. Whichever baseline a comparison uses, it should say
which.

**Each pair is checked for equal work, not merely equal wall time.** TTS pins VITS's three scales so
both engines emit the same 73216 samples, and both harnesses print that count; ASR compares the
transcripts, which are identical; the LM runs the same prompt to the same token budget greedily on both
sides, and both emit the same tokens. **Model load is outside every timer on both sides.**

**Both sides use the same estimator**, which is not a detail: every harness here warms up and reports
a best-or-median over repeated runs in one process. `bench_lm_loom.cpp` used to time a single cold
generation instead, and comparing that against a warmed-up onnxruntime moved the LM column by 5-7% —
enough to flip its sign on all three machines.

**Reproducing it:** loom's side is `scripts/bench_{vits,lm,asr}_loom.cpp`, onnxruntime's is
`scripts/bench_onnx_tasks.py`. The latter drives `onnxruntime` directly rather than through `optimum`,
which added roughly 2x of its own overhead to whisper and mis-derives Qwen3's `head_dim`.

**The Pi throttles, and it will lie if allowed to.** It goes 55 -> 84 C during a single whisper run and
caps the ARM clock at 1580 MHz; two back-to-back measurements once came out 87.1 s and 115.8 s, 33%
apart. Its row is taken with a cooldown before every measurement and the two engines interleaved, so
both meet the same clock.

## Building

```sh
cmake -B build
cmake --build build -j"$(nproc)"
```

Dependencies (`ggml`, `nlohmann_json`, LuaJIT) are fetched by CMake; nothing else is needed to build
and run the hermetic suite. The fetched `ggml` is patched at configure time from `cmake/patches/` —
nine diffs at present, fixing GCC's code generation for ggml's ARM F32 GEMM (1.6x), the matmuls it
declined to accept at all, a fused convolution that batched its work too coarsely to stay in cache, a
direct 1-D convolution for long activations with small weights, the elementwise nodes a vocoder's
resblock wraps around every convolution — its bias, its leaky ReLU and its residual — none of which now
costs a pass over memory of its own, and `conv_transpose_1d`'s single-threaded prologue and
dot-product-at-a-time compute; see `cmake/GgmlPatches.cmake` for the rules such a patch has to meet.

Two of them are heuristics tuned on measured hardware, so they carry a run-time escape:
`GGML_CPU_DISABLE_CONV_HEURISTICS=1` declines both, the way ggml's own `GGML_CPU_DISABLE_FUSION`
declines its fusions.

There is one build option of this repo's own, `-DLOOM_TINYBLAS=OFF`, which drops ggml's blocked GEMM
(`GGML_LLAMAFILE`) back out again. It exists to make GEMM measurements A/B-able and defaults **on**,
where it is worth ~2x on x86-64 and 1.6x on aarch64 at convolutional shapes ([Epic-05](docs/epics/epic-05-edge-performance.md)).

### Running on a GPU

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

## Documentation

The project's knowledge base is a four-tier hub-and-spoke set under [`docs/`](docs/), covering all
three repos.

| | |
|---|---|
| [`docs/backlog/active-index.md`](docs/backlog/active-index.md) | **the hub** — open work only, one line each, linked to its context |
| [`docs/epics/`](docs/epics/) | what each domain is and how it works (engine, exporter, models, backends, performance, host API, text front-ends, packaging) |
| [`docs/adrs/`](docs/adrs/) | why a technical choice was made, what was considered, and what it cost |
| [`docs/retros/`](docs/retros/) | what broke, the root cause, and the takeaway |

The specifications those tiers refer back to:

| | |
|---|---|
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | the data-driven design and why the engine hardcodes nothing |
| [`docs/KV-CACHE.md`](docs/KV-CACHE.md) | how a cached attention block reaches the engine from an export |
| [`docs/LOOM_PROCEDURAL_GENERALIZATION.md`](docs/LOOM_PROCEDURAL_GENERALIZATION.md) | the embedded-Lua orchestration blueprint |
| [`docs/HIGH-LEVEL-API.md`](docs/HIGH-LEVEL-API.md) | one door per task, what each layer owns, and what a file must declare for a host to dispatch |

[`BACKLOG.md`](BACKLOG.md) was the single-file ledger until it reached ~9,000 lines. It is now a
redirect carrying a map from every old section to its new home, so a code comment citing an item by
number (`P4.3e`, `P4.15b`) still resolves. Item numbers were not changed.

## Roadmap

**1. GPUs — done; NPUs, not yet.** The engine takes a device backend and a CPU fallback and schedules
across them ([Epic-04](docs/epics/epic-04-backends-and-accelerators.md)); see [Running on a GPU](#running-on-a-gpu) above. Measured on an AMD
Vega 3 iGPU against 4 CPU threads, one forward each:

| model | splits | GPU vs CPU |
|---|---|---|
| conformer-ctc-small | 1 | **2.56×** |
| lfm2-350m | 1 | **2.22×** |
| qwen3-0.6b | 1 | **2.82×** |
| matcha `encoder_mu` | 1 | **3.65×** |
| kokoro `decoder_vocoder` | 3 | **4.62×** |

What decides that number is how many times the scheduler has to cut the graph, and what forces a cut is
`ggml_map_custom` — a host callback, so there is nothing for a device to dispatch. Those splits used to
be 453, 181, 61, 107 and 5, which left Qwen3 at 0.95× and Matcha at 0.84× — *slower than the CPU*. None
of it was the engine: it was three patterns the exporter emitted as host callbacks because it had never
been taught to recognise them — an RMS norm (`POW`+`RSQRT`), a squaring (`POW`), and a hand-rolled
LayerNorm. **Across all thirteen exported models there are now exactly two `ggml_map_custom` nodes
left** — one `ATAN` each in Kokoro's and StyleTTS2's STFT phase, which has no ggml counterpart.
[Retro-009](docs/retros/retro-009-host-callback-count-was-the-wrong-lens.md) has the numbers, including a CPU measurement that came out wrong twice before
anything interleaved the runs.

**Counting host callbacks turned out to be the wrong lens**, which is worth knowing before optimizing
anything here: a graph splits just as readily on a *real* ggml op whose backend kernel is missing, and
the gaps do not line up between backends — CUDA has `PAD_REFLECT_1D` but no `POOL_1D`, Vulkan has
`POOL_2D` but neither, and the NPU backends have none of the three.

So **a primitive asks the backend what it can run** ([ADR-007](docs/adrs/adr-007-backend-capability-negotiation.md)) and emits either the native op
or an exactly-equivalent composition. That decision belongs in the engine rather than the export: one
GGUF may be run by any backend, so deciding it at export time compiles every artifact for the least
capable one. The same Kokoro file builds 1692 ggml nodes on a CPU and 1732 on Vulkan, and its topology
says `PAD_1D_REFLECT` either way.

`ATAN` had no exact composition anywhere — ggml has no inverse trigonometry in any backend — so it gets
the one **approximation** in the engine: range reduction, a degree-8 minimax polynomial and a branchless
reconstruction, measured at **1.81 ULP** and confined to backends that cannot dispatch the host callback,
so a CPU build still gets libm. **Eleven of the twelve models now run a whole module on the GPU with
nothing falling back at all.** The twelfth is Whisper, whose 400-wide reflect pad is cheaper to fall back
on than to compose — and which CUDA, Metal and SYCL run natively regardless.

One caveat on every speedup on this page: the GPU measured here reports `uma: 1`, so it shares memory
with the host and a split costs a synchronisation rather than a transfer. These numbers are a lower bound
on what a discrete card over PCIe would show.

Of the two decisions the earlier version of this item said were waiting on a GPU, one was answered and
one is still open. Retained inter-module outputs turn out not to be what a device charges for — measured
before the fusion above, LFM2's 20-module modular export cost 183 splits against the monolithic
export's 181, so decomposing a model into modules was never the expensive part. `FLASH_ATTENTION` is
still unbuilt: a GPU makes `ggml_flash_attn_ext`'s forced F16 K/V cast worth considering, but what
stands in the way is the gate suite's exact-fp32 comparisons, not the hardware.

**What is next is CUDA, then NPUs** — see [Epic-04](docs/epics/epic-04-backends-and-accelerators.md). Sixteen backend directories already
ship in the pinned ggml — CUDA, Metal, SYCL, OpenCL, HIP, OpenVINO, Hexagon (Qualcomm), CANN (Ascend)
among them — and because `loom::Device` resolves a spec against ggml's *device registry* rather than
against any backend name it knows, a CUDA build's `CUDA0` is already selectable by code that has never
heard of CUDA. Those cost a build matrix and a test run, not C++. CoreML (the Neural Engine, which
Metal is not) and RKNPU2 are out of tree and cost more, licence check included.

Compiling all of them in would end the leanness this engine is for, so the answer is `GGML_BACKEND_DL`:
each backend becomes a shared library ggml discovers at run time, one engine binary serves every
accelerator, and the deployment decides which files travel with it. That already works through
`loom::Device` unchanged. See [ADR-009](docs/adrs/adr-009-backends-as-dynamic-libraries.md) and the [backlog](docs/backlog/active-index.md#backends--accelerators) for what is still missing (a `Backends` that holds more than two, and
the custom-op fusion above, which on an NPU stops being an optimization and becomes a prerequisite).

**2. Builds for more platforms.** Linux x86-64 is what is built and tested today. Next: macOS on Intel,
macOS on Apple Silicon, and Linux on ARM — the last of which is the one that matters most for an engine
whose stated target is edge devices.

**3. More models — [Epic-03](docs/epics/epic-03-model-coverage.md)**, ordered by coverage per unit of effort: BERT token classifiers
(the smallest possible template, and the first non-audio task) → codec decoders → CNN+CTC and SANM
encoders → the remaining TTS families → text encoder-decoders → small classifiers → music. Each is an
export, so the measure of the design is that none of them should need engine work.

**4. The follow-ups the docs already name.** [`docs/backlog/active-index.md`](docs/backlog/active-index.md)
is the ledger and the authority; the ones worth knowing about from here are the `KvCache` memory redesign (deferred with its reasons),
KV-cache addressing policies beyond the contiguous append `ggml_set_rows` already permits, quantized KV
cache, a permissively-licensed phonemiser so the phoneme-input TTS models get a text door, and
generalizing the grapheme front-end out of C++ once a second such model exists.

## Licence

MIT — see [`LICENSE`](LICENSE).
