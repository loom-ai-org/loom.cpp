# Architectural Blueprint: Data-Driven ggml Inference Engine

## 1. Overview and Core Philosophy
This document outlines the design for a lightweight, data-driven inference engine built on `ggml`, specifically targeting modular Text-to-Speech (TTS) architectures (e.g., integrating components like Zipformer encoders, FlowMatchingNet ODE solvers, and VAE decoders). 

Instead of hardcoding the neural network topology in C++, the engine parses a structured definition file (JSON) to construct the `ggml` computation graph at runtime. 

### The Two-Context Paradigm
The engine must manage two distinct `ggml_context` life cycles:
1.  **The Model Context (Persistent):** Created once upon loading the `.gguf` file. It holds the neural network weights, which are typically memory-mapped (`mmap`) directly from disk to save RAM.
2.  **The Compute Context (Ephemeral):** Created and destroyed (or reset) for every forward pass. It holds the Directed Acyclic Graph (DAG) structures and intermediate activations.

## 2. The Dynamic Graph Engine Architecture
To bridge the JSON definition and the C++ `ggml` backend, the engine requires three core components:

### A. The Symbol Table
A dictionary (e.g., `std::unordered_map<std::string, ggml_tensor*>`) that maps string names to memory pointers. 
*   **Initialization:** Load persistent weights from the GGUF Model Context and register them in the table.
*   **Execution:** As new nodes are computed, their output `ggml_tensor*` is registered here so downstream nodes can reference them by name.

### B. The Primitive Registry
A mapping system that translates JSON operation strings into `ggml` C functions. This acts similarly to ATen-to-GGML or ONNX-to-GGML primitive mapping. 
*   `"MUL_MAT"` -> `ggml_mul_mat(ctx, a, b)`
*   `"ADD"` -> `ggml_add(ctx, a, b)`
*   `"SILU"` -> `ggml_silu(ctx, a)`

### C. GGUF Embedding
To ensure weights and graph definitions are never out of sync, the JSON definition should be serialized as a string and stored directly within the `.gguf` file as Key-Value metadata (e.g., under the key `model.graph_topology`).

## 3. Schema Design
The JSON schema should closely mirror `ggml`'s literal operations to minimize complex C++ translation logic.

    {
      "name": "encoder_layer1_matmul",
      "op": "MUL_MAT",
      "inputs": ["input_embeddings", "encoder.layer1.weight"],
      "outputs": ["layer1_hidden_states"],
      "attributes": {
        "transpose_b": false
      }
    }

## 4. Handling Dynamic Sequence Lengths
Because `ggml` does not support dynamic tensor dimensions (e.g., shape `[-1, 768]`), dynamic sequence lengths are handled by **rebuilding the compute graph from scratch whenever the length changes**.

### Execution Strategy
1.  Read the exact length of the incoming sequence (e.g., text tokens).
2.  Initialize the Compute Context sized specifically for that sequence length.
3.  Parse the JSON array, injecting the exact dimensions into the input tensors.
4.  Execute the graph.
5.  Retain the graph until a *different* length asks for a different one.

Step 5 was originally "destroy the graph (free the activations)", i.e. a rebuild per forward pass rather than per length. A `GraphBuilder` now keeps the last graph it built and returns it unchanged when called again with the same axes, so a loop that re-runs one module at a fixed shape — an ODE solver's steps, an LSTM's timesteps, a chained module in a decode loop — builds once instead of once per iteration (see BACKLOG.md P4.0.13). Nothing about the dynamic-length story changes: a new length is still a new graph.

### Control Flow (The TTS Catch)
Standard DAGs are static, but TTS architectures often require autoregressive loops or continuous ODE solver steps. 
*   **Solution:** Use the JSON to define the static *sub-graphs* (e.g., one step of a vector field estimator). The C++ engine handles the `while` loop, repeatedly executing the sub-graph while updating the timestep and noisy latent input tensors in-place between iterations.

## 5. Memory Allocation: The "Dry Run" Strategy
Estimating tensor byte sizes manually for dynamic sequence lengths is brittle. The engine must use the modern `ggml` two-pass allocation strategy (`ggml_gallocr`) to find the exact optimal memory footprint automatically.

### Step 1: Metadata Allocation (no_alloc = true)
Create a compute context strictly for the C-struct metadata. Setting `no_alloc` to true ensures that `ggml` calculates the math for output shapes but allocates zero memory for the underlying data arrays.

    struct ggml_init_params params = {
        .mem_size   = 10 * 1024 * 1024, // 10 MB for structs
        .mem_buffer = NULL,
        .no_alloc   = true,             // <--- Critical flag
    };
    struct ggml_context * ctx0 = ggml_init(params);

### Step 2: Build the Ghost Graph
Iterate through the JSON definition and build the DAG. The tensors will have correct shapes and types, but null `data` pointers. Call `ggml_build_forward_expand(graph, final_tensor)`.

### Step 3: Graph Allocation and Execution
Pass the graph to the Graph Allocator. It will traverse the DAG, perform liveness analysis to reuse memory where possible (e.g., overwriting Layer 1's memory with Layer 3's output), and allocate the absolute minimum contiguous block on the target backend (CPU, Metal, CUDA).

    // 1. Create the allocator for the target backend
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    
    // 2. Measure graph, allocate optimal buffer, assign physical addresses
    ggml_gallocr_alloc_graph(allocr, graph);
    
    // 3. Execute optimized kernels
    ggml_backend_graph_compute(backend, graph);
