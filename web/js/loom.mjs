// loom for the browser and Node (ADR-139): the doors loom-py offers, over the engine compiled to
// WebAssembly.
//
//   import { init } from "./loom.mjs";
//   const loom = await init();                        // WebGPU where the browser has it, else the CPU
//   const lm = await loom.loadModel("https://.../smollm2-360m-instruct.gguf");
//   for await (const piece of lm.text2text.chatStream([{ role: "user", content: "Hi!" }])) ...
//
// Like loom-py, this file holds no per-model code: a model's file carries its graphs and its driver, and
// each door is chosen by the contract the file declares (`model.contract.interface`).

const HERE = new URL(".", import.meta.url);
const IS_NODE = typeof process !== "undefined" && !!process.versions?.node && typeof window === "undefined";

/** Whether this runtime can run the WebGPU build: a WebGPU adapter, and JSPI to wait on it. */
export async function webgpuAvailable() {
  if (typeof WebAssembly.Suspending !== "function") return false;
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  try {
    return (await navigator.gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

/**
 * Load the engine. `flavour` is "webgpu", "cpu" or "auto" (the default): WebGPU when
 * `webgpuAvailable()`, else the CPU build, which runs anywhere WebAssembly does. `base` is where the
 * `loom-<flavour>.mjs/.wasm` pair lives (this file's directory by default).
 */
export async function init({ flavour = "auto", base = HERE } = {}) {
  if (flavour === "auto") flavour = (await webgpuAvailable()) ? "webgpu" : "cpu";
  if (flavour !== "cpu" && flavour !== "webgpu") throw new Error(`flavour is "auto", "cpu" or "webgpu", not ${flavour}`);
  const url = new URL(`loom-${flavour}.mjs`, base);
  const { default: createLoomModule } = await import(url.href);
  const module = await createLoomModule({
    locateFile: (path) => new URL(path, base).href,
    print: (s) => console.log(s),
    printErr: (s) => console.warn(s),
  });
  return new Loom(module, flavour);
}

let fileCounter = 0;

async function bytesOf(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (typeof Blob !== "undefined" && source instanceof Blob) return new Uint8Array(await source.arrayBuffer());
  if (typeof source === "string" || source instanceof URL) {
    const s = String(source);
    if (IS_NODE && !/^(https?|data|blob):/.test(s)) {
      const { readFile } = await import("node:fs/promises");
      return new Uint8Array(await readFile(s.startsWith("file:") ? new URL(s) : s));
    }
    const response = await fetch(s);
    if (!response.ok) throw new Error(`fetching ${s}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  }
  throw new TypeError("a model is a URL, a path (Node), a Uint8Array, an ArrayBuffer or a Blob");
}

export class Loom {
  constructor(module, flavour) {
    this.module = module;
    /** "cpu" or "webgpu": which build is running. */
    this.flavour = flavour;
    // One call in the engine at a time: under JSPI a call suspends while the GPU works, and the engine
    // is not re-entrant, so a second call waits for the first rather than running inside it.
    this._queue = Promise.resolve();
    try {
      module.FS.mkdir("/loom");
    } catch {}
  }

  /** The ggml devices this build registered: [{name, description, type}]. */
  devices() {
    return JSON.parse(this.module.devices());
  }

  _run(fn) {
    const result = this._queue.then(fn);
    this._queue = result.catch(() => {});
    return result;
  }

  _write(bytes, suffix) {
    const path = `/loom/${++fileCounter}${suffix}`;
    this.module.FS.writeFile(path, bytes);
    return path;
  }

  /**
   * Load a loom GGUF from a URL (or a path in Node, or the bytes). `device` is the engine's spec: ""
   * picks the best device (the WebGPU adapter on the WebGPU build), "cpu" pins the CPU.
   */
  async loadModel(source, { device = "" } = {}) {
    const bytes = await bytesOf(source);
    const path = this._write(bytes, ".gguf");
    try {
      const handle = await this._run(() => this.module.loadModel(path, device));
      return new Model(this, handle, typeof source === "string" ? source : "");
    } finally {
      // The weights are in the backend's buffers now; the file's copy in the heap is not needed.
      this.module.FS.unlink(path);
    }
  }
}

// A sentence ends at a run of terminal marks (with any closing quote or bracket) followed by white
// space, at a CJK terminal mark, or at a line break: loom-py's `split_sentences`, for a TTS whose driver
// does not cut its own pieces.
const SENTENCE_END = /[.!?…]+["')\]”’]*(?=\s)|[。！？]+|\n+/g;

export function splitSentences(text) {
  const out = [];
  let start = 0;
  for (const m of text.matchAll(SENTENCE_END)) {
    const piece = text.slice(start, m.index + m[0].length).trim();
    if (piece) out.push(piece);
    start = m.index + m[0].length;
  }
  const tail = text.slice(start).trim();
  if (tail) out.push(tail);
  return out.length ? out : [text];
}

/** A waveform as a 16-bit PCM WAV file. */
export function toWav(samples, sampleRate, channels = 1) {
  const data = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const ascii = (offset, s) => [...s].forEach((c, i) => data.setUint8(offset + i, c.charCodeAt(0)));
  ascii(0, "RIFF");
  data.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  data.setUint32(16, 16, true);
  data.setUint16(20, 1, true);
  data.setUint16(22, channels, true);
  data.setUint32(24, sampleRate, true);
  data.setUint32(28, sampleRate * channels * 2, true);
  data.setUint16(32, channels * 2, true);
  data.setUint16(34, 16, true);
  ascii(36, "data");
  data.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    data.setInt16(44 + i * 2, Math.trunc(s * 32767), true);
  }
  return new Uint8Array(data.buffer);
}

/**
 * Browser only: an audio file (any format the browser decodes) as mono samples at `sampleRate`, the
 * model's own (`model.sampleRate`). The browser's resampler, so the same choice a page playing it makes.
 */
export async function decodeAudio(source, sampleRate) {
  const bytes = await bytesOf(source);
  const probe = new OfflineAudioContext(1, 1, sampleRate);
  const decoded = await probe.decodeAudioData(bytes.slice().buffer);
  const ctx = new OfflineAudioContext(1, Math.ceil(decoded.duration * sampleRate), sampleRate);
  const node = ctx.createBufferSource();
  node.buffer = decoded;
  node.connect(ctx.destination);
  node.start();
  return (await ctx.startRendering()).getChannelData(0);
}

export class Model {
  constructor(loom, handle, source) {
    this._loom = loom;
    this._h = handle;
    this.source = source;
    /** What the file says it is (the engine's ModelContract, as loom-py's `model.contract`). */
    this.contract = JSON.parse(handle.contract());
    this.text2text = new Text2Text(this);
    this.text2speech = new Text2Speech(this);
    this.speech2text = new Speech2Text(this);
  }

  get task() { return this.contract.task; }
  get interface() { return this.contract.interface; }
  get sampleRate() { return this.contract.sample_rate; }
  get architecture() { return this._h.architecture(); }
  get device() { return { name: this._h.deviceName(), description: this._h.deviceDescription() }; }
  get driverSource() { return this._h.driverSource(); }
  /** The voices the file carries by name (a model taking voice files loads them with `loadVoice`). */
  get voices() { return [...(this.contract.voices || [])]; }
  get chatRoles() { return [...this._h.chatRoles()]; }
  get eosTokenIds() { return [...this._h.eosTokenIds()]; }
  get hasTokenizer() { return this._h.tokenizerKind() !== ""; }

  tokenize(text, lang = "") { return this._h.encode(text, lang); }
  detokenize(ids) { return new TextDecoder().decode(this._h.decodeBytes(Int32Array.from(ids))); }
  applyChatTemplate(messages, addGenerationPrompt = true) {
    const list = typeof messages === "string" ? [{ role: "user", content: messages }] : messages;
    return this._h.applyChatTemplate(list.map((m) => ({ role: m.role, content: m.content })), addGenerationPrompt);
  }

  /** A voice file (ADR-045) for this model, from a URL, a path or its bytes: pass it as `voice`. */
  async loadVoice(source) {
    const path = this._loom._write(await bytesOf(source), ".gguf");
    try {
      return this._h.loadVoice(path);
    } finally {
      this._loom.module.FS.unlink(path);
    }
  }

  /** The raw driver call: `inputs` are numbers and arrays of numbers by name. */
  call(fn, inputs = {}) { return this._loom._run(() => this._h.call(fn, inputs)); }

  _hparam(key) {
    try {
      return this._h.hparamStr(key);
    } catch {
      return "";
    }
  }

  _require(door) {
    if (this.interface !== door) {
      throw new Error(`this model is ${this.interface || "undeclared"} (${this.task || "no task"}), not ${door}`);
    }
  }

  /** Frees the engine's copy of the model. */
  dispose() {
    this._h.delete();
  }
}

function generateOptions(o = {}) {
  const out = { max_new_tokens: o.maxNewTokens ?? 64 };
  for (const [js, c] of [["temperature", "temperature"], ["topK", "top_k"], ["topP", "top_p"], ["seed", "seed"], ["eosToken", "eos_token"]]) {
    if (o[js] != null) out[c] = o[js];
  }
  if (o.inputs) out.inputs = o.inputs;
  return out;
}

class Text2Text {
  constructor(model) { this.m = model; }

  /** Continue `prompt` (text, or token ids); the generated text. */
  async generate(prompt, options = {}) {
    this.m._require("text2text");
    return this.m.detokenize(await this.ids(this._prompt(prompt), options));
  }

  /** The generated ids, without the encode/decode either side. */
  async ids(tokens, options = {}) {
    this.m._require("text2text");
    return [...(await this.m._loom._run(() => this.m._h.generate(Int32Array.from(tokens), generateOptions(options))))];
  }

  /** `generate`, a piece of text at a time as it is drawn (or in one piece for a file without stream functions). */
  async *stream(prompt, options = {}) {
    this.m._require("text2text");
    const tokens = Int32Array.from(this._prompt(prompt));
    if (!this.m._h.canStreamTokens()) {
      yield this.m.detokenize(await this.ids(tokens, options));
      return;
    }
    const run = this.m._loom._run.bind(this.m._loom);
    const stream = await run(() => this.m._h.openTokenStream(tokens, generateOptions(options)));
    try {
      const decoder = new TextDecoder();
      let seen = 0;
      while (!stream.finished()) {
        await run(() => stream.next(options.perPull ?? 1));
        // The bytes of everything so far, new ones fed to a streaming decoder: a byte-level BPE token
        // can end inside a character, which waits for the next token to complete it.
        const bytes = this.m._h.decodeBytes(stream.tokens());
        const piece = decoder.decode(bytes.subarray(seen), { stream: !stream.finished() });
        seen = bytes.length;
        if (piece) yield piece;
      }
    } finally {
      stream.delete();
    }
  }

  /** Ask an instruction-tuned model, in its own chat template: `messages` is a string or [{role, content}]. */
  async chat(messages, options = {}) {
    return this.generate(this.m.applyChatTemplate(messages), { maxNewTokens: 256, ...options });
  }

  chatStream(messages, options = {}) {
    return this.stream(this.m.applyChatTemplate(messages), { maxNewTokens: 256, ...options });
  }

  _prompt(prompt) { return typeof prompt === "string" ? [...this.m.tokenize(prompt)] : [...prompt]; }
}

class Speech2Text {
  constructor(model) { this.m = model; }

  /**
   * Transcribe mono samples in [-1, 1] at the model's rate (`model.sampleRate`; `decodeAudio` gets
   * them from a file). The engine's whole loop: windows, timestamps, the seek. `language`, `task`,
   * `targetLanguage` are names, as loom-py takes them.
   */
  async transcribe(samples, { language, task, targetLanguage, timestamps = false } = {}) {
    this.m._require("speech2text");
    const options = { timestamps };
    if (language) options.language = language;
    if (task) options.task = task;
    if (targetLanguage) options.target_language = targetLanguage;
    const raw = await this.m._loom._run(() => this.m._h.transcribe(Float32Array.from(samples), options));
    return JSON.parse(raw);
  }
}

class Text2Speech {
  constructor(model) { this.m = model; }

  /**
   * Speak `text`: `{samples: Float32Array, sampleRate}`.
   *
   * `text` needs a front end the file declares: a vocabulary (`contract.text_frontend == "vocab"`), or a
   * phoneme table plus `phonemize(text, language)`, an async function from your own G2P (loom-py's
   * phonemisers are Python). `phonemes` (an IPA string) and `tokens` (ids) skip that step. `voice` is one
   * of `model.voices`, or a voice from `model.loadVoice(...)`. `inputs` go to the driver as they are.
   */
  async infer(text, options = {}) {
    this.m._require("text2speech");
    const inputs = await this._inputs(text, options);
    const samples = await this.m._loom._run(() => this.m._h.call("infer", inputs));
    return { samples: Float32Array.from(samples), sampleRate: this.m.sampleRate };
  }

  /**
   * `infer`, a piece at a time: the driver's own pieces where it streams, else a sentence at a time
   * (loom-py's rule: the pieces joined then differ from one `infer` at the joins).
   */
  async *stream(text, options = {}) {
    this.m._require("text2speech");
    const run = this.m._loom._run.bind(this.m._loom);
    if (this.m._h.canStreamSpeech()) {
      const inputs = await this._inputs(text, options);
      const stream = await run(() => this.m._h.openSpeechStream(inputs));
      try {
        while (!stream.finished()) {
          const samples = await run(() => stream.next());
          if (samples.length) yield { samples, sampleRate: this.m.sampleRate };
        }
      } finally {
        stream.delete();
      }
      return;
    }
    const pieces = typeof text === "string" && options.phonemes == null && options.tokens == null ? splitSentences(text) : [text];
    for (const piece of pieces) yield await this.infer(piece, options);
  }

  async _inputs(text, { phonemes, tokens, voice, language, seed = 0, steps, inputs = {}, phonemize } = {}) {
    const c = this.m.contract;
    const out = {};
    if (language != null && c.text_frontend === "vocab" && (c.languages || []).length) {
      const at = c.languages.indexOf(language);
      if (at < 0) throw new Error(`language ${language} is not one this model declares: ${c.languages.join(", ")}`);
      out.language = at + 1;
      language = undefined;
    }
    if (voice != null) {
      if (typeof voice === "string") {
        const names = c.voices || [];
        if (this.m._h.takesVoiceFiles() || !names.length) {
          throw new Error(`this model takes voice files: pass voice: await model.loadVoice(url)`);
        }
        const row = names.indexOf(voice);
        if (row < 0) throw new Error(`this model has no voice ${voice}; its voices are ${names.join(", ")}`);
        out.voice = row;
        if (language == null && row < (c.voice_languages || []).length) language = c.voice_languages[row];
      } else {
        Object.assign(out, voice.inputs);
      }
    }
    Object.assign(out, inputs);
    out.tokens = Float64Array.from(await this._ids(text, phonemes, tokens, language, phonemize));
    if (steps != null) out.n_steps = steps;
    else if (c.default_steps) out.n_steps = c.default_steps;
    out.seed = seed;
    return out;
  }

  async _ids(text, phonemes, tokens, language, phonemize) {
    const named = [["text", text], ["phonemes", phonemes], ["tokens", tokens]].filter(([, v]) => v != null);
    if (named.length !== 1) throw new TypeError(`give exactly one of text, phonemes or tokens (got ${named.map((n) => n[0]).join(", ") || "none"})`);
    if (tokens != null) return [...tokens];
    if (phonemes != null) return typeof phonemes === "string" ? [...this.m.tokenize(phonemes)] : [...phonemes];
    const frontend = this.m.contract.text_frontend;
    if (frontend === "phonemes") {
      if (!phonemize) {
        throw new Error(`this model reads ${this.m.contract.phoneme_alphabet || "ipa"} phonemes: pass phonemize: ` +
          `async (text, language) => ipa from your own G2P, or phonemes: "..."`);
      }
      const lang = language || (this.m.contract.languages || ["en"])[0];
      return [...this.m.tokenize(await phonemize(text, lang))];
    }
    if (frontend !== "vocab") throw new Error("this model declares no text front end: pass tokens or phonemes");
    return [...this.m.tokenize(text, language || "")];
  }
}
