// The web build in CI, with no model to download (ADR-139): tests/fixtures/make_gru_sweep_gguf.py's file,
// whose driver sweeps a GRU and an LSTM through the engine, against the float64 numpy references the
// file carries -- the engine, the PUC Lua VM, embind and the JS layer, end to end.
//
//   python tests/fixtures/make_gru_sweep_gguf.py /tmp/gru_sweep.gguf
//   node web/test/smoke.mjs /tmp/gru_sweep.gguf [cpu|webgpu]     # webgpu: node --experimental-wasm-jspi
// LOOM_GATE_SABOTAGE=1 perturbs every reference, and every check must then fail.
import { init } from "../dist/loom.mjs";

const [file, flavour = "cpu"] = process.argv.slice(2);
const sabotage = process.env.LOOM_GATE_SABOTAGE === "1";
const loom = await init({ flavour });
const model = await loom.loadModel(file);
const ref = (key) => JSON.parse(model._h.kvStr(key)).map((v) => (sabotage ? v + 1e-3 : v));
const sequence = ref("test.sequence").map((v) => (sabotage ? v - 1e-3 : v));
const sequence2 = JSON.parse(model._h.kvStr("test.sequence2"));
const cases = [
  ["forward", { sequence }, "test.forward"],
  ["backward", { sequence }, "test.backward"],
  ["bidirectional", { sequence }, "test.bidirectional"],
  ["paired", { sequence, sequence2 }, "test.paired"],
  ["lstm_forward", { sequence }, "test.lstm_forward"],
];
let failed = 0;
for (const [fn, inputs, key] of cases) {
  const got = await model.call(fn, inputs);
  const want = ref(key);
  let worst = got.length === want.length ? 0 : Infinity;
  for (let i = 0; i < Math.min(got.length, want.length); i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
  const ok = worst < 1e-5;
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"} ${fn}: ${got.length} values, worst ${worst.toExponential(2)} from float64`);
}
console.log(`loom web smoke (${loom.flavour}, ${(await loom.devices()).map((d) => d.name).join(", ")}): ` +
  `${cases.length - failed}/${cases.length} passed${sabotage ? " (sabotaged: all should FAIL)" : ""}`);
process.exit(sabotage ? (failed === cases.length ? 0 : 1) : failed ? 1 : 0);
