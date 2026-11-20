// WebGPU in Node for the web tests (ADR-139): Dawn's bindings (`npm install` in web/ installs the `webgpu`
// dev dependency) provide `navigator.gpu` and the GPU* globals a browser has, on the machine's real
// adapter (Vulkan on Linux). The WebGPU build also needs JSPI: `node --experimental-wasm-jspi`.
// Returns the adapter's description, or null when Dawn is not installed or finds no adapter.
export async function installNodeWebGPU() {
  let dawn;
  try {
    dawn = await import("webgpu");
  } catch {
    return null;
  }
  const { create, globals } = dawn.default ?? dawn;
  Object.assign(globalThis, globals);
  const gpu = create([]);
  if (typeof navigator === "undefined") globalThis.navigator = {};
  Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
  const adapter = await gpu.requestAdapter();
  return adapter ? `${adapter.info.vendor} ${adapter.info.architecture} (${adapter.info.description})` : null;
}
