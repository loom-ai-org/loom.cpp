// The web build against the native engine (ADR-139), case by case from `make_reference.py`'s output.
//
//   node web/test/gate.mjs REF_DIR [cpu|webgpu]          # Node: the CPU build (webgpu: JSPI, no GPU)
//
// The same checks run in a browser through `browser_gate.html`, which imports `runGate` from here.
//   text      the native greedy text, character for character
//   speech    the native waveform's length, and its samples within the case's `min_snr_db`
// A check that cannot fail proves nothing: `LOOM_GATE_SABOTAGE=1` flips each comparison's reference
// (one character of the text, the waveform's sign) and every case must then fail.

export async function runGate(loom, ref, { read, log = console.log, sabotage = false, dump = null } = {}) {
  const results = [];
  for (const c of ref.cases) {
    const t0 = performance.now();
    const model = await loom.loadModel(await read(c.model));
    const loaded = performance.now();
    let ok = false;
    let detail = "";
    try {
      if (c.door === "text2text") {
        const text = await model.text2text.chat(c.messages, { maxNewTokens: c.max_new_tokens, temperature: 0 });
        const want = sabotage ? c.text.slice(0, -1) + "#" : c.text;
        ok = text === want;
        detail = JSON.stringify(text) + (ok ? "" : ` != ${JSON.stringify(want)}`);
      } else if (c.door === "speech2text") {
        const samples = wavSamples(await read(c.audio));
        const r = await model.speech2text.transcribe(samples);
        const want = sabotage ? c.text + "#" : c.text;
        ok = r.text === want;
        detail = JSON.stringify(r.text) + (ok ? "" : ` != ${JSON.stringify(want)}`);
      } else if (c.door === "text2speech") {
        const options = {};
        if (c.voice) options.voice = c.voice;
        if (c.voice_file) options.voice = await model.loadVoice(await read(c.voice_file));
        const audio = c.phonemes != null
          ? await model.text2speech.infer(null, { ...options, phonemes: c.phonemes })
          : await model.text2speech.infer(c.text, options);
        if (dump) await dump(`${c.name}.${loom.flavour}.f32`, audio.samples);
        const want = new Float32Array((await read(c.name + ".f32", true)).buffer.slice(0));
        if (sabotage) want.forEach((v, i) => (want[i] = -v));
        const snr = snrDb(want, audio.samples);
        ok = audio.samples.length === want.length && snr >= c.min_snr_db;
        detail = `${audio.samples.length} samples (native ${want.length}), ${snr.toFixed(1)} dB from native`;
      }
    } catch (e) {
      detail = `threw: ${e.message}`;
    }
    const done = performance.now();
    results.push({ name: c.name, ok, detail, device: model.device.name, load_ms: loaded - t0, run_ms: done - loaded });
    log(`${ok ? "PASS" : "FAIL"} ${c.name} [${model.device.name}] ${detail} (load ${(loaded - t0).toFixed(0)} ms, run ${(done - loaded).toFixed(0)} ms)`);
    model.dispose();
  }
  return results;
}

export function snrDb(want, got) {
  let signal = 0;
  let error = 0;
  const n = Math.min(want.length, got.length);
  for (let i = 0; i < n; i++) {
    signal += want[i] * want[i];
    error += (got[i] - want[i]) ** 2;
  }
  return 10 * Math.log10(signal / Math.max(error, 1e-30));
}

// 16-bit PCM mono, which is what samples/jfk.wav is.
export function wavSamples(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 12;
  while (pos + 8 <= bytes.length) {
    const tag = String.fromCharCode(...bytes.subarray(pos, pos + 4));
    const size = view.getUint32(pos + 4, true);
    if (tag === "data") {
      const out = new Float32Array(size / 2);
      for (let i = 0; i < out.length; i++) out[i] = view.getInt16(pos + 8 + i * 2, true) / 32768;
      return out;
    }
    pos += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}

// Node entry point.
if (typeof process !== "undefined" && import.meta.url === new URL(process.argv[1], "file://").href) {
  const { readFile } = await import("node:fs/promises");
  const path = await import("node:path");
  const [refDir, flavour = "cpu"] = process.argv.slice(2);
  const { init } = await import(new URL("../dist/loom.mjs", import.meta.url).href);
  if (flavour === "webgpu") {
    const { installNodeWebGPU } = await import("./node_webgpu.mjs");
    console.log(`WebGPU in Node: ${(await installNodeWebGPU()) ?? "none (npm install in web/ for Dawn)"}`);
  }
  const loom = await init({ flavour });
  const ref = JSON.parse(await readFile(path.join(refDir, "reference.json"), "utf8"));
  const read = async (p) => new Uint8Array(await readFile(path.isAbsolute(p) ? p : path.join(refDir, p)));
  console.log(`loom web gate: ${loom.flavour} build, devices ${(await loom.devices()).map((d) => d.name).join(", ")}`);
  const sabotage = process.env.LOOM_GATE_SABOTAGE === "1";
  // LOOM_GATE_DUMP=DIR writes each case's waveform there (<case>.<flavour>.f32), for listening or an ASR check.
  const dumpDir = process.env.LOOM_GATE_DUMP;
  const { writeFile } = await import("node:fs/promises");
  const dump = dumpDir ? (name, samples) => writeFile(path.join(dumpDir, name), Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength)) : null;
  const results = await runGate(loom, ref, { read, sabotage, dump });
  const failed = results.filter((r) => !r.ok).length;
  console.log(`${results.length - failed}/${results.length} passed${sabotage ? " (sabotaged: all should FAIL)" : ""}`);
  process.exit(sabotage ? (failed === results.length ? 0 : 1) : failed ? 1 : 0);
}
