# loom in the browser

The engine compiled to WebAssembly, with ggml's WebGPU backend or its CPU backend, and a JavaScript API
over the same doors loom-py has. Any loom GGUF runs, because the file carries its own graphs and its own
driver ([ADR-139](../docs/adrs/adr-139-the-engine-runs-in-a-browser-on-webgpu-or-the-cpu.md)).

```js
import { init } from "loom-web";            // or "./dist/loom.mjs"

const loom = await init();                  // WebGPU where the browser has it, else the CPU
const lm = await loom.loadModel("https://huggingface.co/loom-ai-org/smollm2-360m-instruct-loom/resolve/main/smollm2-360m-instruct.gguf");
for await (const piece of lm.text2text.chatStream([{ role: "user", content: "Hi!" }])) console.log(piece);

const asr = await loom.loadModel(".../moonshine-tiny-q8_0.gguf");
const { text } = await asr.speech2text.transcribe(await decodeAudio(file, asr.sampleRate));

const tts = await loom.loadModel(".../pocket-tts-2m-q8_0_pc.gguf");
const voice = await tts.loadVoice(".../voices/alba.gguf");
const { samples, sampleRate } = await tts.text2speech.infer("Hello.", { voice });
```

## Two builds

| build | backend | runs in | calls |
|---|---|---|---|
| `loom-webgpu` | WebGPU (and the CPU for what it does not support) | Chrome/Edge 137+ (WebGPU and JSPI) | return Promises (JSPI) |
| `loom-cpu` | the CPU, WASM SIMD | any browser, Node 20+ | the same API, awaited alike |

`init()` picks `webgpu` when the page has a WebGPU adapter and JSPI, else `cpu`; `init({flavour})` names
one. Both run one thread: threads need a cross-origin-isolated page (COOP/COEP headers), which a library
cannot assume.

## The doors

* `model.text2text`: `generate`, `stream`, `chat`, `chatStream` (the file's chat template), `ids`.
  Options: `maxNewTokens`, `temperature`, `topK`, `topP`, `seed`. A file exported before token streams
  answers `stream` in one piece.
* `model.speech2text.transcribe(samples, {language, task, targetLanguage, timestamps})`: the engine's
  whole loop, as `Model.transcribe` in loom-py. `decodeAudio(file, model.sampleRate)` decodes and
  resamples with the browser's own decoder.
* `model.text2speech.infer(text, {voice, language, seed, steps, inputs})` and `.stream(...)`. A file with
  a vocabulary front end takes text. A phoneme-input file (Kokoro, VITS, Paradee) needs `phonemes: "..."`
  or `phonemize: async (text, language) => ipa`, because loom-py's phonemisers are Python. `voice` is a
  name from `model.voices`, or `await model.loadVoice(url)` for a model that takes voice files.
* `model.call(fn, inputs)`: the raw driver call, as `model.infer` in loom-py.

## Building

```sh
source ~/emsdk/emsdk_env.sh    # built and tested with emscripten 7.0.0
web/build.sh                   # both builds into web/dist/; `web/build.sh cpu` for one
```

## Testing against the native engine

```sh
python web/test/make_reference.py /tmp/webref --chat gemma-3-270m-it.gguf \
    --asr moonshine-tiny-q8_0.gguf --tts pocket-tts-2m-q8_0_pc.gguf:voices/alba.gguf \
    --phoneme-tts paradee-8m-q8_0_pc.gguf:af_heart
node web/test/gate.mjs /tmp/webref cpu
LOOM_GATE_SABOTAGE=1 node web/test/gate.mjs /tmp/webref cpu   # must fail every case
web/test/browser_gate.sh /tmp/webref                          # headless Chromium, the WebGPU build
```

The demo page is `web/demo/index.html`; serve the `web/` directory (`python -m http.server -d web`) and
open `/demo/`.
