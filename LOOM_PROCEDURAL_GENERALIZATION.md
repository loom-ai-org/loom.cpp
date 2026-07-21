# Architectural Blueprint: Procedural Generalization of Loom via Embedded Lua & PyTorch Export

This document provides a highly detailed, actionable specification for generalizing the multi-modular orchestration layer of the Loom Inference Engine. The goal is to entirely eliminate the need for writing bespoke C++ drivers (e.g., `WhisperDriver`, `VitsDriver`) for new models, shifting the orchestration, host-side math, and control-flow logic to an embedded scripting engine driven by an offline compilation toolchain.

---

## 1. Context and Problem Statement

Loom is a high-performance `ggml` engine targeting resource-constrained edge devices. It utilizes a data-driven paradigm where single-module networks are defined via a JSON-based `graph_topology` metadata key embedded in the GGUF model file. At runtime, Loom’s `GraphBuilder` parses this topology to dynamically compile and execute the underlying `ggml` computation graph.

However, advanced architectures (e.g., speech-to-text, TTS, diffusion) are multi-modular. They depend on "broken graphs" and require active orchestration. Currently, this is handled by custom C++ drivers. These drivers manage:
1. **Dynamic Pipeline Routing:** Transferring output buffers of one subgraph as inputs to subsequent subgraphs.
2. **Autoregressive Control Flow:** While-loops for sequence decoding (Whisper) or integration steps (ODE solvers).
3. **Host-Side Data Wrangling:** Complex math, indexing, and scheduling operations (e.g., Whisper's causal attention-mask generation, VITS's column replication, Gaussian noise injection, and dynamic relative-position embedding padding/cropping).

Writing bespoke C++ files for each new model restricts the portability and extensibility of the engine. We require a generalized, data-driven orchestration layer.

---

## 2. Industry Precedents: How Other Engines Handle It

To solve this, we analyze how leading edge-inference frameworks decouple model definition from runtime compilation:

### A. TensorFlow Lite (TFLite)
* **Approach:** Uses **MLIR (Multi-Level Intermediate Representation)**.
* **Mechanism:** Source models are converted to the MLIR TF Dialect, optimized, and lowered to the TFLite Dialect. This representation is serialized into a FlatBuffer schema (`.tflite`).
* **Control Flow:** TFLite FlatBuffers support structured control-flow operators (`While`, `If`) pointing to nested execution subgraphs inside the file.
* **Critique:** Powerful, but relies heavily on the tightly coupled Google compiler ecosystem, making it highly complex to implement in a lightweight, independent engine.

### B. CoreML (`coremltools` by Apple)
* **Approach:** Uses **MIL (Model Intermediate Language)** and in-memory AST parsing.
* **Mechanism:** Apple has no control over PyTorch or TensorFlow, so their Python-side compiler (`coremltools`) ingests the source model using PyTorch’s in-memory export formats (previously TorchScript, now `torch.export`). The compiler parses the PyTorch Abstract Syntax Tree (AST) and lowers it to MIL (Apple's proprietary IR). MIL runs device-specific optimization passes and compiles down to the Apple Neural Engine (ANE) protobuf format.
* **Critique:** **This is the gold standard for third-party framework ingestion.** It cleanly separates the complex development-time compilation frontend (written in Python) from the ultra-fast runtime (written in C++).
* **Reference:** The code for translating Pytorch to MLIR is openly available (https://github.com/apple/coremltools) and perhaps this could serve as the perfect starting point for a Pytorch -> Loom (GGUF, graph_topology, Lua driver).

### C. ONNX Runtime
* **Approach:** Universal serialization standard.
* **Mechanism:** Models are exported to ONNX protobuf graphs. Dynamic loops are represented using high-level ONNX loop/branch attributes containing nested subgraphs.
* **Critique:** Excellent as an intermediate compiler format, but running ONNX Runtime at the edge is heavy. It lacks optimized bare-metal compilation for micro-architectures, has massive binary size overhead, and lacks native support for the revolutionary block-quantized formats (like `Q4_K` or `Q2_K`) native to GGML.

---

## 3. The Proposed Architecture: "The CoreML Way" for Loom

To preserve GGML's blazing performance, bare-metal optimizations, and native quantization, Loom will mimic Apple CoreML’s compilation pipeline. We will use **ONNX or PyTorch FX purely as an offline development-time IR** to extract weight and topology structures. The output is a single self-contained GGUF file containing:
1. **Block-Quantized Weights** (e.g., `Q8_0`, `Q4_K` formats).
2. **Static JSON topologies** for each individual submodule (e.g., `encoder_topo`, `decoder_topo`).
3. **An embedded Lua orchestration script** that handles pipeline routing, host-side math, and loops.

```
Offline Dev-Time Compilation:
[PyTorch Model] ──► [torch.export / ONNX] ──► [Loom Python Compiler] ──► [model.gguf] (Tensors + JSON Topos + Lua Script)

Edge Target Runtime:
[Loom C++ Engine] ── (Loads GGUF via mmap) ──► Spins up Lua State VM ──► Binds GGML and executes model.
```

---

## 4. The Embedding Engine: Lua vs. LuaJIT

We select **LuaJIT** as the embedded runtime engine.
* **License:** Distributed under the highly permissive **MIT License**, making it fully compatible with commercial and open-source releases with no copyleft restrictions.
* **Performance:** With **LuaJIT**, execution speeds approach native C. Additionally, LuaJIT contains a built-in **FFI (Foreign Function Interface)** library, allowing Lua to directly access C-level structures and pointers (like raw GGML tensor buffers) with zero serialization overhead.
* **Portability:** If targeting highly restrictive platforms (sandboxed iOS, bare-metal microcontrollers, or WebAssembly with JIT disabled), Standard Lua (PUC-Rio Lua 5.4) can be compiled with any standard ANSI C compiler, taking up under 200 KB of binary space.

---

## 5. Step-by-Step Implementation Guide for the Next Agent

The following tasks outline the exact steps required to transition Loom to a procedurally generalized architecture.

### Task 1: Design the GGUF Schema Extension
Modify the GGUF writing script to package the generalized modules.
* Embed each sub-graph topology JSON string under namespaced keys: `model.graph_topology.<module_name>`.
* Embed the Lua orchestration script as a string under the key: `model.driver_script`.

### Task 2: Implement Python-to-Lua AST Transpiler
Write the compiler frontend in Python (`tools/codegen/compile_pytorch_to_loom.py`).
1. **Model Export:** Call `torch.export.export()` on individual heavy submodules to obtain their flat, static graphs.
2. **Weight Quantization:** Extract weights, convert them to GGUF format, and quantize them (e.g., to `Q4_K` or `Q8_0`).
3. **Ast Transpiler:** To compile the orchestration code (the driver logic containing loops/ifs), parse the developer’s orchestration python function using Python’s built-in `ast` module. Map Python syntax directly to Lua:
   * `while loop_condition:` $\rightarrow$ `while loop_condition do ... end`
   * `list.append(item)` $\rightarrow$ `table.insert(list, item)`
   * `run_module("encoder", waveform)` $\rightarrow$ `loom.run_subgraph("encoder", {waveform = waveform})`

### Task 3: C++ Engine Integration (Embedding Lua)
Integrate the Lua VM into the C++ Loom engine.
1. Add Lua/LuaJIT to `cmake/Dependencies.cmake`.
2. Implement `LoomLuaBridge` in `src/core/lua_bridge.cpp`. This class is responsible for initializing the Lua state and registering custom C++ functions into the Lua VM context:

```cpp
// Conceptual Lua C++ Bindings inside Loom Engine
int lua_run_subgraph(lua_State* L) {
    const char* module_name = luaL_checkstring(L, 1);
    
    // Parse input tables passed from Lua
    std::unordered_map<std::string, std::vector<float>> inputs = parse_lua_table(L, 2);
    
    // Retrieve the pre-registered GraphBuilder for this module
    GraphBuilder& builder = get_builder(module_name);
    
    // Construct the dynamic shape arguments based on the input shapes
    GraphBuilder::BuildResult r = builder.build(inputs.at("tokens").size(), n_past);
    
    // Copy data from Lua directly to the backend input tensors
    for (auto& [name, data] : inputs) {
        ggml_backend_tensor_set(r.input_tensors.at(name), data.data(), 0, data.size() * sizeof(float));
    }
    
    // Run the high-performance optimized GGML kernels
    ggml_backend_graph_compute(backend, r.graph);
    
    // Retrieve outputs
    std::vector<float> outputs(ggml_nelements(r.output));
    ggml_backend_tensor_get(r.output, outputs.data(), 0, outputs.size() * sizeof(float));
    
    // Push output vector back onto Lua stack as a table
    push_float_array_to_lua(L, outputs);
    return 1;
}
```

3. Register essential host-side math functions to avoid Lua performance bottlenecks (e.g., `loom.host_argmax`, `loom.generate_causal_triangle_mask`, `loom.generate_gaussian_noise`).

### Task 4: Porting whisper and VITS Drivers to Lua
To validate the generalized architecture, migrate the two existing drivers into GGUF-embedded Lua scripts:

#### Whisper Driver Port:
```lua
function transcribe(inputs)
    -- Run Encoder
    local enc_r = loom.run_subgraph("encoder", { waveform = inputs.waveform })
    
    local n_past = 0
    local generated = {}
    local current_tokens = inputs.prompt_tokens
    
    while #generated < inputs.max_new_tokens do
        -- Build & compute decoder subgraph
        local logits = loom.run_subgraph("decoder", {
            tokens = current_tokens,
            positions = loom.range(n_past, n_past + #current_tokens - 1),
            kq_mask = loom.causal_triangle(#current_tokens, n_past),
            xa = enc_r
        })
        
        local next_token = loom.host_argmax(logits)
        table.insert(generated, next_token)
        if next_token == inputs.eot_token then break end
        
        n_past = n_past + #current_tokens
        current_tokens = { next_token }
    end
    
    return generated
end
```

### Task 5: End-to-End Validation
* Load the compiled model `.gguf`.
* Extract the `model.driver_script` metadata key.
* Load and execute the Lua script inside the embedded Lua interpreter.
* Run existing tests (`test_e2e_whisper_driver.cpp` and `test_e2e_vits_driver.cpp`) and verify that results match the original hardcoded C++ implementations to 6 decimal places.

---

## 6. Automated vs. Advanced Export Profiling Specification

To balance the absolute maximum out-of-the-box developer simplicity with extreme, granular hardware-level memory and cache control, Loom establishes a clear split between the **Automatic (Transparent) Exporting** path and the **Advanced (Bespoke) Exporting** path.

### 6.1 Compiler Profile Selection (`profile` parameter)
For the **Automatic Exporting** path, the `LoomGGUFBackend` supports a `profile` parameter during export (passed via kwargs to the backend or as a standard conversion argument):

```python
# Monolithic Profile (Default): Ingests the PyTorch model and compiles the largest possible static sub-graphs
ct.convert(model, convert_to="loom", profile="monolithic", output_path="model.gguf")

# Atomic Profile: Attempts automated segmentation of the model into highly compressed atomic layers
ct.convert(model, convert_to="loom", profile="atomic", output_path="model.gguf")
```

### 6.2 The Automatic Monolithic Path (`profile="monolithic"`, Default)
* **Core Philosophy**: Prioritize graph simplicity and execution speed by maximizing the coverage of static GGML sub-graphs (`graph_topology` metadata). We intentionally allocate and keep intermediate activations in RAM/VRAM all at once, as the extreme simplicity of the resulting driver makes the orchestration of complex models fully out-of-the-box and transparent.
* **Orchestration Boundary**: The LuaJIT driver script is auto-generated by the compiler and acts solely as high-level "glue" to tie together those massive model components that *cannot* be structurally monolithic due to physical boundaries (such as the Encoder/Decoder boundary in Whisper, or Text-Encoder/Diffusion-Sampler/Vocoder boundaries in StyleTTS2).
* **Auto-Generation of Driver**: The backend dynamically maps inputs, schedules the sequential execution of these comprehensive sub-graphs, and synthesizes the standard `function main(inputs)` Lua driver automatically, requiring zero developer intervention.

### 6.3 The Automatic Atomic Path (`profile="atomic"`)
* **Core Philosophy**: Highly optimized, low-VRAM edge-device execution. The compiler backend automatically attempts to scan, partition, and segment the flat incoming MIL model graph into a sequence of atomic sub-graphs (e.g., individual decoder layers) and dynamically synthesizes the matching Lua loop-driver.
* **Graceful Fallback Strategy**: If the compiler backend cannot automatically perform this atomic segmentation for any reason (e.g., highly unusual custom branching or un-traceable dynamic data dependencies), it **must dynamically and gracefully fall back to the default `profile="monolithic"` path**, logging a clean warning message rather than halting compilation.

### 6.4 The Advanced / Bespoke Exporting Workflow
* **Core Philosophy**: Maximum custom hardware and state-handling control for elite systems optimization. The developer bypasses automated profiling altogether—**no `profile` parameter is used**. 
* **User-Defined Interfaces**: The user is fully responsible for manually splitting the model into custom, arbitrary sub-graphs, defining their public interfaces, naming them, and hand-writing the custom embedded LuaJIT driver script to run them (such as the manually partitioned Whisper and StyleTTS2 drivers).
* **Documentation**: We explicitly document this advanced partitioning workflow in `LOOM_MIL_CONVERSION.md` as the reference path for extreme embedded target optimizations.

