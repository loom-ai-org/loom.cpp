// Exercises GraphTopology::parse() against small embedded JSON literals -- no GGUF file involved. Covers
// the happy path (inputs, a repeat_for block, plain trailing nodes) and the malformed-input error paths.

#include "test_util.h"

#include "loom/loom.h"

namespace {

const char* kValidTopology = R"JSON({
  "version": 1,
  "inputs": [
    { "name": "tokens", "dtype": "i32", "shape": ["n_tokens"] }
  ],
  "output": "logits",
  "nodes": [
    { "op": "GET_ROWS", "name": "inp_embd", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"] },
    { "repeat_for": "$n_layer", "index_var": "i", "nodes": [
      { "op": "RMS_NORM", "inputs": ["cur"], "outputs": ["normed"], "attrs": { "eps": "$rms_norm_eps" } },
      { "op": "MUL_MAT", "inputs": ["blk.{i}.attn_q.weight", "normed"], "outputs": ["q"] }
    ]},
    { "op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["logits"] }
  ]
})JSON";

void test_valid_parse() {
    loom::GraphTopology topo = loom::GraphTopology::parse(kValidTopology);

    LOOM_CHECK(topo.version == 1);
    LOOM_CHECK(topo.output == "logits");
    LOOM_CHECK(topo.inputs.size() == 1);
    LOOM_CHECK(topo.inputs[0].name == "tokens");
    LOOM_CHECK(topo.inputs[0].dtype == "i32");
    LOOM_CHECK(topo.inputs[0].shape == std::vector<std::string>{"n_tokens"});

    LOOM_CHECK(topo.items.size() == 3);

    LOOM_CHECK(!topo.items[0].is_repeat);
    LOOM_CHECK(topo.items[0].node.op == "GET_ROWS");
    LOOM_CHECK(topo.items[0].node.name == "inp_embd");
    LOOM_CHECK((topo.items[0].node.inputs == std::vector<std::string>{"token_embd.weight", "tokens"}));
    LOOM_CHECK((topo.items[0].node.outputs == std::vector<std::string>{"cur"}));

    LOOM_CHECK(topo.items[1].is_repeat);
    LOOM_CHECK(topo.items[1].repeat.count_symbol == "$n_layer");
    LOOM_CHECK(topo.items[1].repeat.index_var == "i");
    LOOM_CHECK(topo.items[1].repeat.nodes.size() == 2);
    LOOM_CHECK(topo.items[1].repeat.nodes[0].op == "RMS_NORM");
    // Symbol expressions are NOT evaluated at parse time -- they stay as raw strings in `attrs`.
    LOOM_CHECK(topo.items[1].repeat.nodes[0].attrs.at("eps").get<std::string>() == "$rms_norm_eps");
    LOOM_CHECK(topo.items[1].repeat.nodes[1].op == "MUL_MAT");
    // {i} substitution also hasn't happened yet -- that's GraphBuilder's job during a build() call.
    LOOM_CHECK(topo.items[1].repeat.nodes[1].inputs[0] == "blk.{i}.attn_q.weight");

    LOOM_CHECK(!topo.items[2].is_repeat);
    LOOM_CHECK(topo.items[2].node.op == "MUL_MAT");
}

// P2 (EXPORT-ROADMAP.md / BACKLOG.md's implementation sequence): a topology may declare more than one
// co-equal output via a plural "outputs" JSON array instead of the singular "output" string.
const char* kMultiOutputTopology = R"JSON({
  "version": 1,
  "inputs": [
    { "name": "x", "dtype": "f32", "shape": ["n_tokens"] }
  ],
  "outputs": ["y", "z"],
  "nodes": [
    { "op": "MUL_MAT", "inputs": ["w", "x"], "outputs": ["y"] },
    { "op": "ADD", "inputs": ["x", "b"], "outputs": ["z"] }
  ]
})JSON";

void test_multi_output_parse() {
    loom::GraphTopology topo = loom::GraphTopology::parse(kMultiOutputTopology);

    LOOM_CHECK(topo.version == 1);
    // `output` (singular) stays populated as the FIRST declared output -- back-compat for any reader
    // that only ever looked at "the" output.
    LOOM_CHECK(topo.output == "y");
    LOOM_CHECK((topo.outputs == std::vector<std::string>{"y", "z"}));
    LOOM_CHECK(topo.items.size() == 2);
}

void test_single_output_still_populates_outputs_vector() {
    loom::GraphTopology topo = loom::GraphTopology::parse(kValidTopology);
    LOOM_CHECK((topo.outputs == std::vector<std::string>{"logits"}));
}

void test_empty_outputs_array_throws() {
    const char* json = R"JSON({"version": 1, "outputs": [], "nodes": []})JSON";
    LOOM_CHECK_THROWS(loom::GraphTopology::parse(json), loom::SchemaError);
}

void test_invalid_json_throws() {
    LOOM_CHECK_THROWS(loom::GraphTopology::parse("{ not valid json "), loom::SchemaError);
}

void test_unsupported_version_throws() {
    const char* json = R"JSON({"version": 2, "output": "x", "nodes": []})JSON";
    LOOM_CHECK_THROWS(loom::GraphTopology::parse(json), loom::SchemaError);
}

void test_missing_output_throws() {
    const char* json = R"JSON({"version": 1, "nodes": []})JSON";
    LOOM_CHECK_THROWS(loom::GraphTopology::parse(json), loom::SchemaError);
}

void test_node_missing_op_throws() {
    const char* json = R"JSON({"version": 1, "output": "x", "nodes": [ { "inputs": [], "outputs": ["x"] } ]})JSON";
    LOOM_CHECK_THROWS(loom::GraphTopology::parse(json), loom::SchemaError);
}

void test_repeat_block_missing_index_var_throws() {
    const char* json = R"JSON({
      "version": 1, "output": "x",
      "nodes": [ { "repeat_for": "$n_layer", "nodes": [] } ]
    })JSON";
    LOOM_CHECK_THROWS(loom::GraphTopology::parse(json), loom::SchemaError);
}

} // namespace

int main() {
    test_valid_parse();
    test_multi_output_parse();
    test_single_output_still_populates_outputs_vector();
    test_empty_outputs_array_throws();
    test_invalid_json_throws();
    test_unsupported_version_throws();
    test_missing_output_throws();
    test_node_missing_op_throws();
    test_repeat_block_missing_index_var_throws();

    LOOM_TEST_REPORT_AND_RETURN();
}
