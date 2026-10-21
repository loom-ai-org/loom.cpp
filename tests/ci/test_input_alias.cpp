// A retained output is READ IN PLACE by the module that consumes it, not copied into its input
// (ADR-073).
//
// Whisper's decoder took 24 retained cross-attention K/V tensors as inputs at every decode step, and the
// copies were 501 ms of an 8.86 s transcription (jfk, Ryzen 3 3250U). An alias removes the copy, and the
// claims that make it safe are what this file states, each against something that would fail:
//
//   1. the aliased chain computes what the marshalled chain computes;
//   2. the source is not written -- its retained output reads back byte-identical after the consumer ran;
//   3. the alias ENDS at the next build(): a reused consumer then fed a plain array computes that array's
//      answer, and the write lands in the consumer's own storage, not in the source it pointed at;
//   4. a reused consumer that is NOT fed an input it last read by alias is refused by name, since its
//      own storage never received that value. That is also the proof the alias path ran at all -- a copy
//      leaves nothing stale;
//   5. a consumer that reads its input THROUGH A VIEW reads the source too. ggml fixes a view's `data` at
//      allocation, so moving only the input left every reshape of it reading the input's own, unfilled
//      storage -- Whisper's decoder transcribed jfk as "[ [ [" and flan-t5 emitted <unk>, while cases 1-4
//      passed, because stage_b uses its input directly.
//
// Shares the attention-free toy GGUF and the two-stage chain with test_lua_bridge_retained_outputs.cpp,
// whose equivalence cases (whole tensor, prefix, single row) now run through the alias as well.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace {

const char* kStageAJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "outputs": ["cur", "normed"],
  "nodes": [
    {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
    {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["normed"], "attrs": {"eps": "$rms_norm_eps"}}
  ]
})JSON";

const char* kStageBJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"hidden","dtype":"f32","shape":["n_embd","n_tokens"]}],
  "output": "logits",
  "nodes": [
    {"op": "MUL_MAT", "inputs": ["output.weight", "hidden"], "outputs": ["logits"]}
  ]
})JSON";

// stage_b again, but reading `hidden` only through views of it: a flatten and an unflatten, both
// `ggml_reshape`s, so the MUL_MAT never sees the input tensor itself.
const char* kStageViewJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"hidden","dtype":"f32","shape":["n_embd","n_tokens"]}],
  "output": "logits",
  "nodes": [
    {"op": "RESHAPE", "inputs": ["hidden"], "outputs": ["flat"], "attrs": {"shape": [-1]}},
    {"op": "RESHAPE", "inputs": ["flat"], "outputs": ["again"], "attrs": {"shape": ["n_embd", "n_tokens"]}},
    {"op": "MUL_MAT", "inputs": ["output.weight", "again"], "outputs": ["logits"]}
  ]
})JSON";

// Every function returns ONE value: LoomLuaBridge::call reads the top of the stack, which for a
// `return logits, shape` would be the shape.
const char* kScript = R"lua(
    function retain_a(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        return {}
    end

    function a_output(inputs)
        local cur = loom.get_output('stage_a')
        return cur
    end

    function b_by_ref(inputs)
        local logits = loom.run_subgraph('stage_b', {n_tokens = inputs.n[1], n_past = 0},
                                          {hidden = {from = 'stage_a'}})
        return logits
    end

    function b_by_value(inputs)
        local logits = loom.run_subgraph('stage_b', {n_tokens = inputs.n[1], n_past = 0},
                                          {hidden = inputs.hidden})
        return logits
    end

    function view_by_ref(inputs)
        local logits = loom.run_subgraph('stage_view', {n_tokens = inputs.n[1], n_past = 0},
                                          {hidden = {from = 'stage_a'}})
        return logits
    end

    function view_by_value(inputs)
        local logits = loom.run_subgraph('stage_view', {n_tokens = inputs.n[1], n_past = 0},
                                          {hidden = inputs.hidden})
        return logits
    end

    function b_without_input(inputs)
        local logits = loom.run_subgraph('stage_b', {n_tokens = inputs.n[1], n_past = 0}, {})
        return logits
    end
)lua";

using Array = std::vector<double>;

Array as_array(const loom::LoomLuaBridge::Value& v) { return std::get<Array>(v); }

}  // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("stage_a", *model, loom::GraphTopology::parse(kStageAJson));
    bridge.register_module("stage_b", *model, loom::GraphTopology::parse(kStageBJson));
    bridge.register_module("stage_view", *model, loom::GraphTopology::parse(kStageViewJson));
    bridge.load_script(kScript);

    const Array tokens = {1, 4, 2, 5, 3};
    const Array n = {static_cast<double>(tokens.size())};

    bridge.call("retain_a", {{"tokens", tokens}});
    const Array a_before = as_array(bridge.call("a_output", {}));
    LOOM_CHECK(!a_before.empty());

    // --- 1. In place equals marshalled. The oracle hands B the very values A retained, through Lua. ---
    const Array by_ref = as_array(bridge.call("b_by_ref", {{"n", n}}));
    const Array oracle = as_array(bridge.call("b_by_value", {{"n", n}, {"hidden", a_before}}));
    LOOM_CHECK(!by_ref.empty());
    LOOM_CHECK(by_ref == oracle);

    // --- 2. The source is untouched by a consumer that read it in place. ---
    LOOM_CHECK(as_array(bridge.call("a_output", {})) == a_before);

    // --- 3. The alias ends at the next build(). Alias, then REUSE the same graph with a different array
    // by value: the answer is that array's, and the source still holds what A computed. Without the
    // restore, the array would be written through the alias into stage_a's retained output. ---
    bridge.call("b_by_ref", {{"n", n}});
    Array other(a_before.size());
    for (size_t i = 0; i < other.size(); ++i) other[i] = 0.25 * static_cast<double>(i % 7) - 0.5;
    const Array other_answer = as_array(bridge.call("b_by_value", {{"n", n}, {"hidden", other}}));
    LOOM_CHECK(as_array(bridge.call("a_output", {})) == a_before);
    LOOM_CHECK(other_answer != by_ref);
    // ...and that answer is the one a fresh graph gives the same array.
    {
        loom::LoomLuaBridge fresh(backend.get());
        fresh.register_module("stage_a", *model, loom::GraphTopology::parse(kStageAJson));
        fresh.register_module("stage_b", *model, loom::GraphTopology::parse(kStageBJson));
        fresh.load_script(kScript);
        LOOM_CHECK(as_array(fresh.call("b_by_value", {{"n", n}, {"hidden", other}})) == other_answer);
    }

    // --- 4. A reused consumer not fed what it last read by alias is refused, by name. ---
    bridge.call("b_by_ref", {{"n", n}});
    bool threw = false;
    try {
        bridge.call("b_without_input", {{"n", n}});
    } catch (const loom::Error& e) {
        threw = true;
        const std::string msg = e.what();
        std::fprintf(stderr, "omitted aliased input -> %s\n", msg.c_str());
        LOOM_CHECK(msg.find("'hidden'") != std::string::npos);
        LOOM_CHECK(msg.find("by reference") != std::string::npos);
    }
    LOOM_CHECK(threw);
    // ...and feeding it again recovers: the refusal left nothing broken behind it.
    LOOM_CHECK(as_array(bridge.call("b_by_ref", {{"n", n}})) == by_ref);

    // --- 5. Through a view. By VALUE first, so the input's own storage holds `other` -- the stale value a
    // view left behind would then read as `other`'s answer rather than as zeros, which could not be told
    // apart from a correct one by accident. By reference, it must be A's answer again, the same one the
    // direct consumer gave. Twice, so the reuse (views already repointed once) is covered too. ---
    LOOM_CHECK(as_array(bridge.call("view_by_value", {{"n", n}, {"hidden", other}})) == other_answer);
    LOOM_CHECK(as_array(bridge.call("view_by_ref", {{"n", n}})) == by_ref);
    LOOM_CHECK(as_array(bridge.call("view_by_ref", {{"n", n}})) == by_ref);
    // ...and the restore moves the views back: by value again, through the same retained graph.
    LOOM_CHECK(as_array(bridge.call("view_by_value", {{"n", n}, {"hidden", other}})) == other_answer);
    LOOM_CHECK(as_array(bridge.call("a_output", {})) == a_before);

    LOOM_TEST_REPORT_AND_RETURN();
}
