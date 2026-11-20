// The web gate in a real browser (ADR-139): headless Chromium with WebGPU, the same cases and checks as
// `gate.mjs` in Node.
//
//   node web/test/browser_gate.mjs REF_DIR [webgpu|cpu] [--chromium PATH] [--headful]
//
// Serves web/ and the reference's files on 127.0.0.1, opens `browser_gate.html` in Chromium with WebGPU
// enabled, and waits for the page to post its results. Exit 0 when every case passes
// (with LOOM_GATE_SABOTAGE=1: when every case fails).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const refDir = path.resolve(args[0] ?? "");
const flavour = args[1] && !args[1].startsWith("--") ? args[1] : "webgpu";
const chromium = args.includes("--chromium") ? args[args.indexOf("--chromium") + 1] : "chromium";
const headful = args.includes("--headful");
const sabotage = process.env.LOOM_GATE_SABOTAGE === "1";
const TYPES = { ".mjs": "text/javascript", ".js": "text/javascript", ".html": "text/html", ".wasm": "application/wasm", ".json": "application/json" };

let finish;
const done = new Promise((resolve) => (finish = resolve));

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method === "POST" && url.pathname === "/result") {
      let body = "";
      for await (const chunk of req) body += chunk;
      res.end("ok");
      finish(JSON.parse(body));
      return;
    }
    if (req.method === "POST" && url.pathname === "/log") {
      let body = "";
      for await (const chunk of req) body += chunk;
      console.log(body);
      res.end("ok");
      return;
    }
    // /ref/<path>: a reference file (absolute, or relative to REF_DIR); anything else: under web/.
    const file = url.pathname.startsWith("/ref/")
      ? path.resolve(refDir, decodeURIComponent(url.pathname.slice(5)))
      : path.join(web, decodeURIComponent(url.pathname));
    await stat(file);
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch (e) {
    res.writeHead(404);
    res.end(String(e));
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
const page = `http://127.0.0.1:${port}/test/browser_gate.html?flavour=${flavour}&sabotage=${sabotage ? 1 : 0}`;

const flags = [
  ...(headful ? [] : ["--headless=new"]),
  // Headless Chromium on Linux offers WebGPU through SwiftShader (a CPU Vulkan): what this checks there is
  // the browser path -- JSPI, the WebGPU JS API, the shaders -- not GPU speed. With a desktop session,
  // --headful uses the real adapter. $LOOM_CHROMIUM_FLAGS adds flags (e.g. --enable-features=Vulkan).
  "--enable-unsafe-webgpu", ...(process.env.LOOM_CHROMIUM_FLAGS ?? "").split(/\s+/).filter(Boolean),
  "--no-first-run", "--no-default-browser-check", `--user-data-dir=${path.join(refDir, ".chromium-profile")}`,
  page,
];
const browser = spawn(chromium, flags, { stdio: ["ignore", "ignore", "pipe"] });
browser.stderr.on("data", (d) => {
  const s = String(d);
  if (/webgpu|vulkan|dawn/i.test(s) && /error|fail/i.test(s)) process.stderr.write(s);
});
const timeout = setTimeout(() => finish({ error: "timed out after 30 minutes" }), 30 * 60 * 1000);
const outcome = await done;
clearTimeout(timeout);
browser.kill();
server.close();

if (outcome.error) {
  console.log(`browser gate: ${outcome.error}`);
  process.exit(1);
}
const failed = outcome.results.filter((r) => !r.ok).length;
console.log(`${outcome.results.length - failed}/${outcome.results.length} passed in Chromium (${outcome.flavour} build, ` +
  `devices ${outcome.devices.join(", ")})${sabotage ? " (sabotaged: all should FAIL)" : ""}`);
process.exit(sabotage ? (failed === outcome.results.length ? 0 : 1) : failed ? 1 : 0);
