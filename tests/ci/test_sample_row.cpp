// P4.24: `loom.sample_row`, the reduction that draws a token instead of maximizing over one.
//
// The invariants here are the cheap, strong ones the item was scoped against, and each is a different
// way for a sampler to be wrong:
//
//   * `temperature -> 0` is the argmax, and so is `top_k = 1`, and so is the empty options table --
//     which is what keeps GREEDY the default and every existing baseline where it is;
//   * one seed gives the same ids twice, two seeds differ -- so the draw is really coming from the
//     shared stream and is really reproducible;
//   * `top_k = k` draws from exactly k distinct ids -- so the truncation is applied rather than
//     computed and discarded;
//   * a temperature high enough to flatten the distribution eventually reaches EVERY class -- so the
//     draw is not quietly the argmax with extra steps.
//
// Reuses the attention-free toy-model fixture (make_builder_test_gguf.py) that
// test_lua_bridge_retained_outputs.cpp already shares: 6 classes wide, which is small enough that
// "every class appears" is a testable statement.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

const char* kStageAJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "outputs": ["cur"],
  "nodes": [
    {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]}
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

const char* kScript = R"lua(
    local function logits(tokens)
        local n = #tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = tokens})
        return loom.run_subgraph_and_retain('stage_b', {n_tokens = n, n_past = 0},
                                             {hidden = {from = 'stage_a'}})
    end

    function greedy(inputs)
        logits(inputs.tokens)
        return loom.argmax_row('stage_b', -1)
    end

    -- The same reduction asked for the other way. `opts` is assembled from whatever the caller set, so
    -- one entry point covers "no options at all", "temperature only" and the full set.
    function sampled(inputs)
        local gen = logits(inputs.tokens)
        return loom.sample_row('stage_b', -1, {temperature = inputs.temperature,
                                                top_k = inputs.top_k,
                                                top_p = inputs.top_p,
                                                generation = gen})
    end

    -- N draws from one seed, so a whole sequence can be compared rather than a single value -- one
    -- draw agreeing is not much evidence about an RNG.
    function draws(inputs)
        local out = {}
        loom.seed_rng(inputs.seed)
        for i = 1, inputs.n do
            logits(inputs.tokens)
            out[i] = loom.sample_row('stage_b', -1, {temperature = inputs.temperature,
                                                      top_k = inputs.top_k,
                                                      top_p = inputs.top_p})
        end
        return out
    end

    local EXPECTED = {
        'it is a count of candidates',
        'cumulative probability',
        'cumulative probability',
        'has no retained outputs',
    }

    function expect_error(inputs)
        local ok, err = pcall(function()
            if inputs.case == 1 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {temperature = 1.0, top_k = -3})
            elseif inputs.case == 2 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {temperature = 1.0, top_p = 0.0})
            elseif inputs.case == 3 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {temperature = 1.0, top_p = 1.5})
            else
                loom.sample_row('stage_never', -1, {temperature = 1.0})
            end
        end)
        if ok then return -1 end
        if string.find(err, EXPECTED[inputs.case], 1, true) then return inputs.case end
        print('case ' .. inputs.case .. ' raised an unexpected message: ' .. err)
        return 0
    end
)lua";

using Args = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("stage_a", *model, loom::GraphTopology::parse(kStageAJson));
    bridge.register_module("stage_b", *model, loom::GraphTopology::parse(kStageBJson));
    bridge.register_module("stage_never", *model, loom::GraphTopology::parse(kStageBJson));
    bridge.load_script(kScript);

    const std::vector<double> tokens = {1, 3, 4};
    const auto greedy = std::get<double>(bridge.call("greedy", {{"tokens", tokens}}));

    // --- 1. Every greedy spelling is the argmax, by the same function rather than by agreement. ---
    LOOM_CHECK(std::get<double>(bridge.call("sampled", {{"tokens", tokens}})) == greedy);
    LOOM_CHECK(std::get<double>(bridge.call(
        "sampled", {{"tokens", tokens}, {"temperature", 0.0}})) == greedy);
    // top_k = 1 with a real temperature: there is one candidate, so the draw cannot land anywhere else.
    LOOM_CHECK(std::get<double>(bridge.call(
        "sampled", {{"tokens", tokens}, {"temperature", 1.0}, {"top_k", 1.0}})) == greedy);
    // A top_p below the best single token's own probability keeps exactly that token, which is the
    // "at least one candidate always survives" clause doing its job rather than selecting nothing.
    LOOM_CHECK(std::get<double>(bridge.call(
        "sampled", {{"tokens", tokens}, {"temperature", 1.0}, {"top_p", 1e-6}})) == greedy);

    // --- 2. Reproducible from a seed, and different seeds really differ. ---
    const Args base = {{"tokens", tokens}, {"n", 40.0}, {"temperature", 3.0}};
    Args seed_a = base; seed_a["seed"] = 1234.0;
    Args seed_b = base; seed_b["seed"] = 1234.0;
    Args seed_c = base; seed_c["seed"] = 9999.0;
    const auto run_a = std::get<std::vector<double>>(bridge.call("draws", seed_a));
    const auto run_b = std::get<std::vector<double>>(bridge.call("draws", seed_b));
    const auto run_c = std::get<std::vector<double>>(bridge.call("draws", seed_c));
    LOOM_CHECK(run_a.size() == 40);
    LOOM_CHECK(run_a == run_b);
    LOOM_CHECK(run_a != run_c);

    // --- 3. It is really SAMPLING: a flat enough distribution reaches every one of the six classes,
    //        which no argmax ever would.
    Args flat = base;
    flat["seed"] = 7.0;
    flat["n"] = 400.0;
    flat["temperature"] = 50.0;
    const auto flat_draws = std::get<std::vector<double>>(bridge.call("draws", flat));
    const std::set<double> flat_ids(flat_draws.begin(), flat_draws.end());
    std::fprintf(stderr, "temperature 50 over 400 draws reached %zu of 6 classes\n", flat_ids.size());
    LOOM_CHECK(flat_ids.size() == 6);

    // --- 4. top_k truncates rather than being read and discarded: k = 2 can only ever return two ids,
    //        and at the same temperature the untruncated run returns more than two.
    Args k2 = flat;
    k2["top_k"] = 2.0;
    const auto k2_draws = std::get<std::vector<double>>(bridge.call("draws", k2));
    const std::set<double> k2_ids(k2_draws.begin(), k2_draws.end());
    LOOM_CHECK(k2_ids.size() == 2);
    LOOM_CHECK(k2_ids.count(greedy) == 1); // the best token is always among the k best

    // --- 5. Every way of getting it wrong that the binding is supposed to catch. ---
    for (double c = 1; c <= 4; ++c) {
        const auto got = std::get<double>(bridge.call("expect_error", {{"case", c}}));
        if (got != c) std::fprintf(stderr, "sample_row error case %g returned %g\n", c, got);
        LOOM_CHECK(got == c);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
