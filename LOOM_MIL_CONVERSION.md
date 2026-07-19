# Specification: Procedural Generalization of Loom via CoreML MIL Custom Backend

This document outlines the detailed architectural decisions, data structures, and implementation specifications for building an offline compilation toolchain. The compiler ingests PyTorch models, translates them into the **Model Intermediate Language (MIL)** IR of `coremltools`, and outputs a self-contained GGUF file containing block-quantized weights, static JSON sub-graph topologies, and an embedded Lua orchestration script for the Loom C++ engine.

---

## 1. Architectural Design & Licensing Decisions

### 1.1 Decoupled Plugin Architecture (Option A)
To preserve the permissive **MIT License** of the `loom.cpp` runtime and avoid license contamination/disclaimers from `coremltools`'s **BSD 3-Clause** license, the compiler frontend is designed as a **standalone, runtime-registered plugin**. 
* The compiler codebase resides in an independent repository or subdirectory.
* At runtime, it dynamically imports `coremltools` and registers its custom backend with `coremltools`'s global converter registry.
* This avoids physical codebase modifications to the core `coremltools` repository, keeping dependencies clean and easily upgradeable.

```
+------------------------------------------------------------+
| Developer's Compiling Script (Python)                      |
|                                                            |
|  1. Imports coremltools                                    |
|  2. Imports custom loom_backend (MIT)                      |
|  3. Dynamically registers LoomGGUFBackend with registry    |
|  4. Ingests PyTorch Model -> Runs MIL Passes -> Emits GGUF |
+------------------------------------------------------------+
```

---

## 2. Core Mapping Specifications (MIL to GGUF + Lua)

A `coremltools.converters.mil.Program` represents the complete computation as a set of named functions (`Program.functions`). This structure maps beautifully to Loom’s multi-modular target:

| MIL Program Concept | Loom Engine Target Asset | Serialization Strategy |
| :--- | :--- | :--- |
| **`main` Function** | **Embedded LuaJIT Driver Script** | Iterate over the flattened SSA operations in the `main` block and transpile control-flow (`while_loop`, `cond`) and tensor routings directly to Lua. |
| **Heavy Submodule Functions** (e.g., `"encoder"`, `"decoder"`) | **Static GGUF Sub-graphs & Block-Quantized Weights** | Convert sequential MIL operations into static JSON graph topologies. Extract weight matrices from `const` operations, format as tensors, and write them to GGUF. |
| **Small Constant / Attribute** | **Lua Inline Locals / Primitives** | Small variables (shapes, indices, scales) are serialized inline as Lua constants. |
| **Host-side / Coordinate Math** | **Lua Host Math Bindings** | Map generic MIL mathematical operations (like `argmax`, `range`) to custom high-performance C++ bindings exposed to Lua (e.g., `loom.host_argmax`, `loom.range`). |

### 2.1 Comprehensive 150+ MIL Ops Mapping Strategy

To ensure the engine is fully generic, the goal is to support the full specification of 150+ MIL operations. The official ground-truth schemas for these operations are located within `coremltools/converters/mil/mil/ops/defs/iOS15/` (and subsequent iOS version folders).

**Implementation Strategy for the Exporter:**
1. **Dynamic Op Mapping:** The `LoomGGUFExporter` will feature a dynamic lookup table mapping MIL `op.op_type` directly to their equivalent `ggml` C++ primitive signatures (or Lua host math bindings for control flow / non-tensor ops).
2. **Unsupported Op Trapping:** During the graph traversal (`generate_graph_topology`), if an operation is not yet implemented in the lookup table, the exporter MUST raise a `NotImplementedError(f"MIL op '{op_type}' is missing a ggml mapping.")`.
3. **Iterative Coverage:** The implementing agent will run the conversion on a diverse set of test models (e.g., ResNet, Whisper, VITS). For every trapped `NotImplementedError`, the agent will consult the corresponding definition in `coremltools/converters/mil/mil/ops/defs/` to understand the inputs/attributes and add the corresponding mapping to the exporter.

**Key MIL to GGML Core Mapping Categories:**
* **Math/Linear:** `matmul` -> `ggml_mul_mat`, `add` -> `ggml_add`, `mul` -> `ggml_mul`
* **Activations:** `relu` -> `ggml_relu`, `gelu` -> `ggml_gelu`, `softmax` -> `ggml_soft_max`
* **Convolutions:** `conv` -> `ggml_conv_1d` / `ggml_conv_2d`
* **Transforms:** `reshape`, `transpose` -> `ggml_transpose`, `concat` -> `ggml_concat`
* **Reductions:** `reduce_mean` -> `ggml_mean`

### 2.2 Specialized Loom Primitives & Custom Graph Passes (Fusion Pipeline)

Since Loom contains highly specialized, pre-fused, and domain-specific primitives (which form a superset of standard MIL operations), you can bridge standard PyTorch models and specialized Loom primitives using **Custom Graph Passes** and **Custom Dialect Operations**:

#### 1. Define Custom Dialect Operations
You can register custom specialized operations under a `"loom"` namespace dynamically at runtime in your plugin. This defines the schema (inputs, outputs, types) of your specialized primitives:

```python
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op

@register_op(namespace="loom")
class loom_fused_attention(Operation):
    """
    Specialized Loom dynamic attention primitive.
    """
    input_spec = InputSpec(
        q=TensorInputType(),
        k=TensorInputType(),
        v=TensorInputType(),
        mask=TensorInputType(optional=True)
    )

    def type_inference(self):
        return self.q.sym_type
```

#### 2. Implement a Custom Fusion Pass (`LoomFusionPass`)
Write a custom `AbstractGraphPass` that scans the standard MIL program block, identifies patterns of standard elementwise operations (e.g., `matmul + transpose + add + softmax`), and replaces them with your high-level custom Loom Dialect operation:

```python
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb

@register_pass(namespace="loom")
class FuseLoomAttention(AbstractGraphPass):
    def apply(self, prog):
        for f in prog.functions.values():
            self._fuse_blocks(f)

    def _fuse_blocks(self, block):
        # Scan block operations, locate patterns of standard operations 
        # and replace them using enclosing_block.try_replace_uses_of_var_after_op
        # to construct: %y = loom.loom_fused_attention(q=%q, k=%k, v=%v)
        pass
```

#### 3. Register in the Pass Pipeline
Inject your custom fusion pass into the custom pass pipeline prior to backend execution:

```python
pipeline = ct.PassPipeline.DEFAULT
# Inject your custom fusion pass
pipeline.insert_pass(index=0, pass_name="loom::FuseLoomAttention")
```

#### 4. Serialization
When the `LoomGGUFExporter` encounters `op.op_type == "loom_fused_attention"`, it directly serializes your specialized `loom_fused_attention` primitive in the static JSON graph topology. This matches perfectly with Loom's pre-compiled high-performance C++ execution kernels.

---

## 3. Custom Backend Implementation Blueprint

The following skeleton defines the dynamic registration and the traversal code for the next agent to implement.

### 3.1 Dynamic Backend Registration (`loom_backend/register.py`)
```python
import numpy as np
import coremltools as ct
from coremltools.converters.mil.converter import ConverterRegistry
from coremltools.converters.mil.mil import Program

@ConverterRegistry.backend
class LoomGGUFBackend:
    name = "loom"
    alias_names = ["gguf"]

    def __call__(self, program: Program, **kwargs):
        """
        Compiler Backend Entry Point.
        Ingests the highly optimized MIL Program and lowers it to Loom assets.
        """
        exporter = LoomGGUFExporter(program, **kwargs)
        return exporter.export()
```

### 3.2 Program Exporter & Transpiler (`loom_backend/exporter.py`)
```python
import json
import numpy as np
from coremltools.converters.mil.mil import Block, Function, Operation, Var

class LoomGGUFExporter:
    def __init__(self, program: Program, **kwargs):
        self.program = program
        self.weights = {}
        self.topologies = {}
        self.lua_lines = []
        self.output_path = kwargs.get("output_path", "model.gguf")

    def export(self):
        # 1. Traversal and Segmentation
        for func_name, func in self.program.functions.items():
            if func_name == "main":
                # Generate the embedded LuaJIT orchestration script
                self.transpile_to_lua(func)
            else:
                # Compile heavy neural network layers to GGUF static topologies
                self.topologies[func_name] = self.generate_graph_topology(func)

        # 2. Transpiled Script Consolidation
        driver_script = "\n".join(self.lua_lines)

        # 3. Serialization Phase
        # TODO: Integrate with a GGUF writing library to package:
        #   - Tensors (block-quantized self.weights)
        #   - Topologies (self.topologies as JSON metadata strings)
        #   - driver_script (as string metadata "model.driver_script")
        self.write_gguf(driver_script)

        return self.output_path

    def transpile_to_lua(self, func: Function):
        self.lua_lines.append(f"function {func.name}(inputs)")
        
        # Unpack incoming inputs
        for name in func.inputs.keys():
            self.lua_lines.append(f"    local {name} = inputs.{name}")
            
        # Transpile operations inside function block
        self.transpile_block(func)
        
        # Unpack and return outputs
        outputs_str = ", ".join([v.name for v in func.outputs])
        self.lua_lines.append(f"    return {outputs_str}")
        self.lua_lines.append("end")

    def transpile_block(self, block: Block):
        for op in block.operations:
            self.transpile_operation(op)

    def transpile_operation(self, op: Operation):
        op_type = op.op_type
        outputs_str = ", ".join([v.name for v in op.outputs])

        # A. Constant Serialization
        if op_type == "const":
            val = op.val.val
            # If the tensor is large, treat as a weight matrix for GGUF serialization
            if isinstance(val, np.ndarray) and val.size > 100:
                self.weights[op.outputs[0].name] = val
                self.lua_lines.append(f"    -- Weight {op.outputs[0].name} packaged in GGUF")
            else:
                self.lua_lines.append(f"    local {outputs_str} = {self.format_lua_val(val)}")

        # B. Autoregressive / Loop Control Flow
        elif op_type == "while_loop":
            # MIL structure: cond block is op.blocks[0], body block is op.blocks[1]
            loop_vars = [v.name for v in op.inputs["loop_vars"]]
            cond_block = op.blocks[0]
            body_block = op.blocks[1]

            self.lua_lines.append("    while true do")
            
            # Inline condition evaluation block
            self.transpile_block(cond_block)
            cond_var = cond_block.outputs[0].name
            self.lua_lines.append(f"        if not {cond_var} then break end")
            
            # Inline body execution block
            self.transpile_block(body_block)
            self.lua_lines.append("    end")

        # C. Conditional Branches
        elif op_type == "cond":
            true_block = op.blocks[0]
            false_block = op.blocks[1]
            pred_var = op.inputs["pred"].name
            
            self.lua_lines.append(f"    if {pred_var} then")
            self.transpile_block(true_block)
            self.lua_lines.append("    else")
            self.transpile_block(false_block)
            self.lua_lines.append("    end")

        # D. Submodule Dispatch (Runs optimized C++ GGUF graphs)
        elif op_type in self.program.functions:
            inputs_tbl = ", ".join([f"{k} = {v.name}" for k, v in op.inputs.items() if isinstance(v, Var)])
            self.lua_lines.append(f"    local {outputs_str} = loom.run_subgraph('{op_type}', {{{inputs_tbl}}})")

        # E. Fast Host Math Mapping
        elif op_type == "argmax":
            self.lua_lines.append(f"    local {outputs_str} = loom.host_argmax({op.inputs['x'].name})")
        elif op_type == "range":
            start = op.inputs["start"].name
            end = op.inputs["end"].name
            self.lua_lines.append(f"    local {outputs_str} = loom.range({start}, {end})")

        # F. Fallback for generic SSA math fusions
        else:
            self.lua_lines.append(f"    -- Fallback: host math implementation for {op_type}")
            # Write a translator for basic arithmetic if required, e.g., local %y = %a + %b

    def format_lua_val(self, val):
        if isinstance(val, (int, float, bool)):
            return str(val).lower()
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, np.ndarray):
            return "{" + ", ".join(map(str, val.flatten())) + "}"
        return "nil"

    def generate_graph_topology(self, func: Function) -> str:
        """
        Walks a heavy submodule's MIL graph and serializes it to static JSON topology.
        """
        nodes = []
        for op in func.operations:
            nodes.append({
                "name": op.name,
                "op": op.op_type,
                "inputs": [v.name for v in op.inputs.values() if isinstance(v, Var)],
                "outputs": [v.name for v in op.outputs]
            })
        return json.dumps({"nodes": nodes}, indent=2)

    def write_gguf(self, driver_script: str):
        # TODO: Implement writing of weights, topologies, and driver_script 
        # using Python GGUF library (e.g. `gguf.GGUFWriter`)
        pass
```

---

## 4. Concrete Code Generation / Transpilation Examples

### 4.1 Conditional Evaluation
**PyTorch/MIL Input Pattern:**
```python
# MIL SSA Ops:
# %pred_val = cond_op(x, y)
# cond(%pred_val) {
#   %res_t = true_fn()
# } {
#   %res_f = false_fn()
# }
```
**Transpiled Lua Output:**
```lua
if pred_val then
    local res_t = true_fn()
else
    local res_f = false_fn()
end
```

### 4.2 Multi-Moduler Subgraph Execution
**PyTorch/MIL Input Pattern:**
```python
# Function call representation:
# %xa = encoder(waveform=%input_wave)
```
**Transpiled Lua Output:**
```lua
local xa = loom.run_subgraph('encoder', {waveform = input_wave})
```

---

## 5. Execution Backlog for the Implementing Agent

1. **Setup Plugin Boilerplate:** Create a standalone python repository `loom_mil_compiler`. Setup environment loading dependencies (`coremltools`, `numpy`, `gguf`).
2. **Implement `LoomGGUFExporter`:** Fully implement the class to map intermediate variables to Lua-safe variables (replacing invalid characters like `%`, `.`, or `/` in MIL SSA names with safe characters like `_`).
3. **Register Dynamic Backend:** Verify registration by running a minimal converted model using `convert_to="loom"`.
4. **Develop GGUF Serializer:** Use standard `gguf` python package to write `model.graph_topology.<submodule_name>` keys, the `model.driver_script` key, and the block-quantized weights extracted from `exporter.weights`.
5. **Add Unit/Integration Tests:** Mock a multi-modular PyTorch model containing a loop, run the compilation script, assert the validity of the generated GGUF file, and verify the correct structure of the transpiled Lua code.
