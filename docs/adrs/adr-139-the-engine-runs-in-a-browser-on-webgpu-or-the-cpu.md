---
type: adr
status: accepted
date: 2026-11-19
domain: backends
tags: [webassembly, webgpu, emscripten, browser, javascript, lua, host-bindings]
supersedes: []
---

# ADR-139: The Engine Runs in a Browser, on WebGPU or the CPU

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (a new backend),
[Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (a JavaScript host) and
[Epic-08](../epics/epic-08-packaging-and-release.md) (an npm package). The user asked for it before 1.0.0:
"add WebGPU / WebAssembly backend from upstream (ggml) to run models in a browser".

## Context

ggml v0.26.0 (our pin) has a WebGPU backend (`ggml-webgpu`, WGSL shaders) that builds under emscripten
against Dawn's `emdawnwebgpu` port. It waits on the GPU with either JSPI or Asyncify. Its CPU backend
has WASM SIMD kernels. What stood between the engine and a browser was not ggml:

* **LuaJIT has no WebAssembly target.** Its interpreter is hand-written assembly per architecture.
* **Every host had its own tokenizer dispatch** (loom-py's binding, `loom_cli`), and a third would have
  been written for JavaScript.
* **The seeded noise differs on libc++.** That is [ADR-141](adr-141-seeded-draws-are-the-same-numbers-on-every-standard-library.md),
  found by this work's own gate.

## Decision

1. **`LOOM_LUA=puc`: PUC Lua 5.1.5 plus LuaBitOp 1.0.3** (`cmake/LuaPuc.cmake`, both MIT, fetched by
   hash). It is the default under emscripten and available natively. Lua 5.1 is the dialect LuaJIT
   implements, and the C API `lua_bridge.cpp` already used. LuaBitOp is the `bit` library LuaJIT has
   built in, by the same author, with the same semantics. The drivers use `bit` in `aten_randn`,
   Pocket-TTS, SpeechT5 and Soprano, and no other LuaJIT-only feature (no `ffi`, `jit`, `goto`).
2. **`loom::Tokenizer`** (`include/loom/core/tokenizer.h`) is loom-py's dispatch over the 14 vocabulary
   families, moved into the engine unchanged. The web binding uses it. loom-py and `loom_cli` switching
   to it is a hub item.
3. **`web/`** in this repo (not a fourth repo):
   * `src/loom_web.cpp` is an embind binding running the same sequence as loom-py's.
   * `js/loom.mjs` is the doors: `text2text` (`chat`, `chatStream`, ...), `speech2text.transcribe`,
     `text2speech.infer`/`stream`, voices, `call`.
   * `build.sh` builds two flavours into `dist/`, and `package.json` is the `loom-web` npm package.
     `webgpu` (ggml-webgpu, JSPI: graph-running calls return Promises) needs WebGPU and JSPI
     (Chrome/Edge 137+). `cpu` (WASM SIMD, no JSPI) runs anywhere. `init()` picks between them.
4. **One thread.** Threads need SharedArrayBuffer, which needs a cross-origin-isolated page (COOP/COEP
   headers), and a library cannot assume one. `default_cpu_thread_count` returns 1 without pthreads.
5. **A phoneme-input TTS takes `phonemes` or a `phonemize` callback in the browser.** loom-py's
   phonemisers are Python. A vocabulary front end (Pocket-TTS, Soprano, Supertonic, ...) takes text as
   it does in Python.

## Options not taken

* **LuaJIT's interpreter via `LJ_TARGET_*` porting**, or a Lua compiled to JavaScript (fengari). The
  first is a new LuaJIT port. The second would change the numbers and run every driver through a second
  implementation, where PUC Lua is the reference implementation of the same dialect.
* **Asyncify instead of JSPI** for the WebGPU build. It would run in every browser but grow the wasm
  and slow every call. The CPU build already covers browsers without JSPI.
* **A fourth repo for the JS package.** The build is the engine's CMake and the JS layer is thin. The
  user chose `web/` in this repo.

## Consequences

* **PUC Lua natively:** `ctest -L ci` 160/160. Gates on PUC: the drivers that use `bit` (Pocket-TTS,
  SpeechT5, Soprano, tiny-voca) pass, and so do Kokoro, StyleTTS2, Whisper, Kyutai STT (106 s against
  109 s on LuaJIT), LFM2.5-Audio, PLaMo, OmniVoice (84 517 checks) and Voxtral-4B-TTS (64 checks, 31 min,
  its usual time). So does Paradee: 66.3 and 66.6 dB
  from upstream, with its seed-only draw 6e-08 from the pinned one, so `aten_randn` is still torch's
  stream.
* **The web gate** (`web/test/gate.mjs` in Node, `browser_gate.mjs` in Chromium 155) compares against
  the native engine's own answers. Greedy chat (Gemma-3-270M, 24 tokens) and transcription (Moonshine
  tiny) are identical. Pocket-TTS 2M has the native sample count and is 28.3 dB from native (a
  free-running latent loop, where WASM and x86 sums diverge). Paradee is 41.8 dB. Both builds pass in
  Node and in headless Chromium, and a sabotaged reference fails every case. CPU wasm, one thread:
  Moonshine 11 s of audio in 7.6 s, Pocket-TTS 2.9 s in 1.8 s.
* **WebGPU on a real GPU is not yet measured.** Headless Chromium on Linux offers only SwiftShader,
  which lacks `shader-f16`. ggml-webgpu then declines the adapter, and the WebGPU build falls back to
  its CPU. That path is gated, and correct. A run on a GPU adapter is a hub item.
* **CI:** a `puc-lua` job runs the suite on PUC Lua. A `web` job builds both flavours with emscripten
  7.0.0 and runs `web/test/smoke.mjs`, a GRU and LSTM driver against float64 references, plus its
  sabotage.
