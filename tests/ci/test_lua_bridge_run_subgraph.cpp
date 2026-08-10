// Exercises loom.run_subgraph itself (unlike test_lua_bridge.cpp, which deliberately bypasses it --
// see that file's own top comment) against a real registered multi-output module: P2's driver-side
// plumbing (EXPORT-ROADMAP.md / BACKLOG.md's implementation sequence) generalizes run_subgraph to
// return every declared output's DATA (in declared order), then every declared output's SHAPE (in that
// same order) -- for a single-output module that's still exactly the (data, shape) pair the binding
// always returned. Reuses the same attention-free toy-model GGUF fixture as test_graph_builder_shapes.cpp
// (make_builder_test_gguf.py) so real weight lookups (token_embd.weight, output.weight, blk.0.*) work.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

namespace {

// Two co-equal outputs ("y", "z") from a shared "cur", plus extra same-shape nodes computed AFTER "z"
// that don't depend on it -- the same shape as test_graph_builder_shapes.cpp's own multi-output
// GraphBuilder test, run here through loom.run_subgraph instead of GraphBuilder::build() directly.
const char* kMultiJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "outputs": ["y", "z"],
  "nodes": [
    {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
    {"op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["y"]},
    {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["normed"], "attrs": {"eps": "$rms_norm_eps"}},
    {"op": "MUL", "inputs": ["normed", "blk.0.norm.weight"], "outputs": ["z"]},
    {"op": "MUL_MAT", "inputs": ["blk.0.ffn.weight", "z"], "outputs": ["ffn_out"]},
    {"op": "ADD", "inputs": ["ffn_out", "cur"], "outputs": ["dummy_unused"]}
  ]
})JSON";

const char* kSingleJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "output": "logits",
  "nodes": [
    {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
    {"op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["logits"]}
  ]
})JSON";

} // namespace

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("multi", *model, loom::GraphTopology::parse(kMultiJson));
    bridge.register_module("single", *model, loom::GraphTopology::parse(kSingleJson));

    bridge.load_script(R"lua(
        -- Captures only the DATA outputs (the common case: an N-output module's caller that isn't
        -- interested in shapes at all) -- Lua silently discards the trailing shape return values.
        function run_data_only(inputs)
            local y, z = loom.run_subgraph('multi', {n_tokens = #inputs.tokens, n_past = 0}, {tokens = inputs.tokens})
            local out = {}
            for i = 1, #y do out[#out + 1] = y[i] end
            for i = 1, #z do out[#out + 1] = z[i] end
            return out
        end

        -- Captures every data output AND every shape -- must line up (data1, data2, shape1, shape2).
        function run_with_shapes(inputs)
            local y, z, sy, sz = loom.run_subgraph('multi', {n_tokens = #inputs.tokens, n_past = 0}, {tokens = inputs.tokens})
            return {sy[1], sy[2], sz[1], sz[2]}
        end

        -- Single-output module: unchanged (data, shape) convention.
        function run_single(inputs)
            local out, shape = loom.run_subgraph('single', {n_tokens = #inputs.tokens, n_past = 0}, {tokens = inputs.tokens})
            local combined = {}
            for i = 1, #out do combined[#combined + 1] = out[i] end
            combined[#combined + 1] = shape[1]
            combined[#combined + 1] = shape[2]
            return combined
        end
    )lua");

    const std::vector<double> tokens = {1, 3, 4};

    // "single"'s "logits" is the exact same MUL_MAT(output.weight, GET_ROWS(...)) computation as
    // "multi"'s "y" -- an independent oracle for the data-only capture below.
    auto single_result = bridge.call("run_single", {{"tokens", tokens}});
    const auto& single_arr = std::get<std::vector<double>>(single_result);
    const size_t y_n = single_arr.size() - 2; // last two entries are the shape
    const double y_ne0 = single_arr[y_n];
    const double y_ne1 = single_arr[y_n + 1];

    auto data_result = bridge.call("run_data_only", {{"tokens", tokens}});
    const auto& data_arr = std::get<std::vector<double>>(data_result);

    // y (N_VOCAB=6 x n_tokens=3 = 18) then z (N_EMBD=4 x n_tokens=3 = 12).
    LOOM_CHECK(data_arr.size() == 18 + 12);
    for (size_t i = 0; i < y_n; ++i) {
        LOOM_CHECK(data_arr[i] == single_arr[i]);
    }

    auto shapes_result = bridge.call("run_with_shapes", {{"tokens", tokens}});
    const auto& shapes_arr = std::get<std::vector<double>>(shapes_result);
    LOOM_CHECK(shapes_arr.size() == 4);
    // y: [N_VOCAB=6, n_tokens=3]; z: [N_EMBD=4, n_tokens=3] -- matches the "single" oracle's own shape.
    LOOM_CHECK(shapes_arr[0] == y_ne0);
    LOOM_CHECK(shapes_arr[1] == y_ne1);
    LOOM_CHECK(shapes_arr[2] == 4.0);
    LOOM_CHECK(shapes_arr[3] == 3.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
