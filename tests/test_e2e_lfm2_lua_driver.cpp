// Validates the compiled LFM2-350M GGUF model via the LoomLuaBridge.
// Loads the 100% dynamic GGUF multi-topology model, registers its 18 sub-graphs,
// loads the embedded master Lua driver script, and runs generation on the CPU.

#include "test_util.h"
#include "loom/loom.h"
#include <ggml-cpu.h>
#include <cstdio>
#include <vector>

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // Load our newly compiled LFM2 GGUF model
    auto model = loom::GgufModel::load("lfm2_350m.gguf", backend.get());
    LOOM_CHECK(model != nullptr);

    // Extract the Master Lua generation driver script
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    // Initialize our C++ Lua JIT virtual machine bridge
    loom::LoomLuaBridge bridge(backend.get());
    
    // Register the 18 static sub-graph modules dynamically from the GGUF file metadata
    std::printf("Registering LFM2-350M sub-modules from GGUF...\n");
    bridge.register_module("embedding", *model, loom::GraphTopology::parse(model->topology_json("embedding")), nullptr);
    bridge.register_module("output_head", *model, loom::GraphTopology::parse(model->topology_json("output_head")), nullptr);
    
    for (int i = 0; i < 16; ++i) {
        std::string layer_name = "layer_" + std::to_string(i);
        bridge.register_module(layer_name, *model, loom::GraphTopology::parse(model->topology_json(layer_name)), nullptr);
    }

    // Load the Master driver script into the VM
    bridge.load_script(driver_script);

    // Call the master orchestration entry point "main" with starting prompt token IDs: [10, 20, 30, 40]
    std::printf("Invoking 'main' function inside the embedded Lua script...\n");
    const std::vector<double> prompt_tokens = {10.0, 20.0, 30.0, 40.0};
    
    loom::LoomLuaBridge::Value result = bridge.call("main", {
        {"tokens", prompt_tokens}
    });

    double next_tok_val = std::get<double>(result);
    std::printf("\nSUCCESS! LFM2-350M compiled model generated next token ID: %d\n\n", static_cast<int32_t>(next_tok_val));

    LOOM_TEST_REPORT_AND_RETURN();
}
