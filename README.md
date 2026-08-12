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

## Building

```sh
cmake -B build
cmake --build build -j"$(nproc)"
```

Dependencies (`ggml`, `nlohmann_json`, LuaJIT) are fetched by CMake; nothing else is needed to build
and run the hermetic suite.

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
scripts/fixtures.py status    # what the 77 gate tests want, and what you have
scripts/fixtures.py fetch     # from the published fixture repo
```

Every fixture also still honours the per-test variable it always had
(`LOOM_KOKORO_MIL_GGUF`, …), which wins over the root — pointing one test at one artifact you just
rebuilt is what you do while working on it.

## Documentation

| | |
|---|---|
| [`BACKLOG.md`](BACKLOG.md) | the project ledger: every work item, what was measured, and what was found |
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | the data-driven design and why the engine hardcodes nothing |
| [`docs/KV-CACHE.md`](docs/KV-CACHE.md) | how a cached attention block reaches the engine from an export |
| [`docs/LOOM_PROCEDURAL_GENERALIZATION.md`](docs/LOOM_PROCEDURAL_GENERALIZATION.md) | the embedded-Lua orchestration blueprint |

## Roadmap

**1. GPUs and NPUs.** The engine talks to a single `ggml_backend_t` through a plain `ggml_gallocr` and
uses no `ggml_backend_sched` at all — fine for CPU, and the one thing that has to change before a
second device can hold part of a graph. Two existing decisions are waiting on it: `ATTENTION` always
takes the composite `MUL_MAT`→`soft_max_ext`→`MUL_MAT` path because `ggml_flash_attn_ext` forces an F16
K/V cast that fights exact fp32 verification, and a `FLASH_ATTENTION` primitive becomes worth adding
once a GPU makes that trade pay; and retained inter-module outputs are a latent win today that becomes
a paid one the moment a marshalled edge would mean a device→host→device round trip per step.

**2. Builds for more platforms.** Linux x86-64 is what is built and tested today. Next: macOS on Intel,
macOS on Apple Silicon, and Linux on ARM — the last of which is the one that matters most for an engine
whose stated target is edge devices.

**3. More models — P5 in the ledger**, ordered by coverage per unit of effort: BERT token classifiers
(the smallest possible template, and the first non-audio task) → codec decoders → CNN+CTC and SANM
encoders → the remaining TTS families → text encoder-decoders → small classifiers → music. Each is an
export, so the measure of the design is that none of them should need engine work.

**4. The follow-ups the docs already name.** [`BACKLOG.md`](BACKLOG.md) is the ledger and the authority;
the ones worth knowing about from here are the `KvCache` memory redesign (deferred with its reasons),
KV-cache addressing policies beyond the contiguous append `ggml_set_rows` already permits, quantized KV
cache, a permissively-licensed phonemiser so the phoneme-input TTS models get a text door, and
generalizing the grapheme front-end out of C++ once a second such model exists.

## Licence

MIT — see [`LICENSE`](LICENSE).
