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
// Family 10 added two more knobs to the same options table, and each gets the same treatment:
//
//   * `lo`/`hi` restrict the draw to a half-open id window and return ABSOLUTE ids -- so a channel
//     that may not emit the control tokens at the top of its vocabulary cannot. Dia is why: its
//     `DiaEOSChannelFilterLogitsProcessor` bans them per channel, and an unrestricted sampler on
//     channel 8 emits PAD or BOS, which an argmax never did because the argmax already had a window;
//   * `guidance` combines two modules' logits as `uncond + scale * (cond - uncond)`, checked against
//     that formula computed here from the two rows rather than against itself -- and across a scan of
//     scales, because a guidance implementation that ignored `scale` would pass any single one.
//
// Reuses the attention-free toy-model fixture (make_builder_test_gguf.py) that
// test_lua_bridge_retained_outputs.cpp already shares: 6 classes wide, which is small enough that
// "every class appears" is a testable statement.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
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
    local N_CLASSES = 6

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
        'not a non-empty sub-range',
        'not a non-empty sub-range',
        'which is the one being sampled',
        'guidance needs a `module`',
        'guidance combines two runs of ONE model',
    }

    -- The whole last row, marshalled out, so the test can compute what guidance SHOULD answer instead
    -- of asking the same code twice. Six classes is small enough for that to be free.
    function logits_row(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local out = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                       {hidden = {from = 'stage_a'}})
        local row = {}
        for i = 1, N_CLASSES do row[i] = out[(n - 1) * N_CLASSES + i] end
        return row
    end

    -- One reduction over a restricted window. `lo`/`hi` are passed straight through, so a nil pair is
    -- the whole row -- which is what every driver written before this knob existed passes.
    function windowed(inputs)
        logits(inputs.tokens)
        return loom.sample_row('stage_b', -1, {temperature = inputs.temperature,
                                                top_k = inputs.top_k,
                                                lo = inputs.lo, hi = inputs.hi})
    end

    function windowed_draws(inputs)
        local out = {}
        loom.seed_rng(inputs.seed)
        for i = 1, inputs.n do
            logits(inputs.tokens)
            out[i] = loom.sample_row('stage_b', -1, {temperature = inputs.temperature,
                                                      lo = inputs.lo, hi = inputs.hi})
        end
        return out
    end

    -- Two runs of the same head into two modules, then one guided draw. The unconditional run goes
    -- SECOND and into a different module on purpose: a module has one retained output, so a driver
    -- that ran both through one name would be sampling a row against itself.
    function guided(inputs)
        logits(inputs.tokens)
        local n2 = #inputs.tokens2
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n2, n_past = 0}, {tokens = inputs.tokens2})
        loom.run_subgraph_and_retain('stage_uncond', {n_tokens = n2, n_past = 0},
                                      {hidden = {from = 'stage_a'}})
        return loom.sample_row('stage_b', -1, {temperature = inputs.temperature,
                                                lo = inputs.lo, hi = inputs.hi,
                                                guidance = {module = 'stage_uncond',
                                                            scale = inputs.scale}})
    end

    -- Guidance with a SHORTLIST: the guided logits pick k candidates, the conditional ones are what
    -- gets drawn from. `guidance_top_k` in transformers, and a different operation from `top_k`.
    function guided_shortlist(inputs)
        logits(inputs.tokens)
        local n2 = #inputs.tokens2
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n2, n_past = 0}, {tokens = inputs.tokens2})
        loom.run_subgraph_and_retain('stage_uncond', {n_tokens = n2, n_past = 0},
                                      {hidden = {from = 'stage_a'}})
        return loom.sample_row('stage_b', -1, {lo = inputs.lo, hi = inputs.hi,
                                                guidance = {module = 'stage_uncond',
                                                            scale = inputs.scale,
                                                            top_k = inputs.gtop_k}})
    end

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
            elseif inputs.case == 4 then
                loom.sample_row('stage_never', -1, {temperature = 1.0})
            elseif inputs.case == 5 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {lo = 4, hi = 4})
            elseif inputs.case == 6 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {lo = 0, hi = 99})
            elseif inputs.case == 7 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {guidance = {module = 'stage_b', scale = 3.0}})
            elseif inputs.case == 8 then
                logits({1, 3, 4})
                loom.sample_row('stage_b', -1, {guidance = {scale = 3.0}})
            else
                -- `stage_a`'s output is the EMBEDDING, n_embd wide, not the 6-wide head -- so this is
                -- guidance against a tensor that is not the same distribution.
                logits({1, 3, 4})
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.sample_row('stage_b', -1, {guidance = {module = 'stage_a', scale = 3.0}})
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
    bridge.register_module("stage_uncond", *model, loom::GraphTopology::parse(kStageBJson));
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

    // --- 5. `lo`/`hi`: the reduction happens inside the window, and the id that comes back is
    //        ABSOLUTE. The strong case is the one that excludes the best token, because a window that
    //        is read and discarded gives the same answer as no window at all. ---
    LOOM_CHECK(greedy > 0);   // otherwise "everything below the best" is empty and proves nothing
    const Args full_row = {{"tokens", tokens}};
    LOOM_CHECK(std::get<double>(bridge.call("windowed", full_row)) == greedy);
    Args below = full_row; below["lo"] = 0.0; below["hi"] = greedy;
    const auto best_below = std::get<double>(bridge.call("windowed", below));
    std::fprintf(stderr, "greedy is class %g; restricted to [0, %g) it is %g\n",
                 greedy, greedy, best_below);
    LOOM_CHECK(best_below != greedy && best_below >= 0 && best_below < greedy);
    // A window that still contains the best token gives it back, offset arithmetic and all -- this is
    // what catches an implementation returning an offset into the window instead of an id.
    Args above = full_row; above["lo"] = greedy; above["hi"] = 6.0;
    LOOM_CHECK(std::get<double>(bridge.call("windowed", above)) == greedy);

    // Drawing, not maximizing, and still inside: 400 draws at a flattening temperature over a
    // two-class window land on exactly those two classes.
    Args win_draws = {{"tokens", tokens}, {"n", 400.0}, {"temperature", 50.0}, {"seed", 11.0},
                      {"lo", 2.0}, {"hi", 4.0}};
    const auto win = std::get<std::vector<double>>(bridge.call("windowed_draws", win_draws));
    const std::set<double> win_ids(win.begin(), win.end());
    LOOM_CHECK(win_ids == std::set<double>({2.0, 3.0}));

    // --- 6. Classifier-free guidance: `uncond + scale * (cond - uncond)`, checked against that
    //        formula computed HERE from the two rows. Across a scan of scales, because guidance that
    //        ignored `scale` -- or that combined them the other way round -- would satisfy any single
    //        one of them. ---
    const std::vector<double> tokens2 = {0, 0, 2};
    const auto cond = std::get<std::vector<double>>(
        bridge.call("logits_row", {{"tokens", tokens}}));
    const auto uncond = std::get<std::vector<double>>(
        bridge.call("logits_row", {{"tokens", tokens2}}));
    LOOM_CHECK(cond.size() == 6 && uncond.size() == 6);
    LOOM_CHECK(cond != uncond);   // otherwise every scale gives the same answer and this proves nothing

    std::set<double> guided_answers;
    for (const double scale : {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 8.0}) {
        std::vector<double> expect(6);
        for (size_t i = 0; i < 6; ++i) {
            expect[i] = uncond[i] + scale * (cond[i] - uncond[i]);
        }
        const auto best = static_cast<double>(
            std::max_element(expect.begin(), expect.end()) - expect.begin());
        Args g = {{"tokens", tokens}, {"tokens2", tokens2}, {"scale", scale}};
        const auto got = std::get<double>(bridge.call("guided", g));
        if (got != best) {
            std::fprintf(stderr, "guidance scale %g: expected class %g, got %g\n", scale, best, got);
        }
        LOOM_CHECK(got == best);
        guided_answers.insert(got);
    }
    // The two anchors, stated separately because they are the two ways the formula degenerates and
    // both are things a caller relies on: scale 1 is the conditional untouched, scale 0 is the
    // unconditional one. A sign error passes neither.
    LOOM_CHECK(std::get<double>(bridge.call(
        "guided", {{"tokens", tokens}, {"tokens2", tokens2}, {"scale", 1.0}})) == greedy);
    LOOM_CHECK(std::get<double>(bridge.call(
        "guided", {{"tokens", tokens}, {"tokens2", tokens2}, {"scale", 0.0}})) ==
        static_cast<double>(std::max_element(uncond.begin(), uncond.end()) - uncond.begin()));
    std::fprintf(stderr, "guidance over 7 scales reached %zu distinct classes\n",
                 guided_answers.size());
    LOOM_CHECK(guided_answers.size() >= 2);   // it really depends on the scale

    // Guidance and the window compose: the guided maximum, restricted. Two knobs that each work alone
    // and not together is a real failure mode when both are read from the same table.
    {
        constexpr double kLo = 1.0, kHi = 5.0;
        constexpr double kScale = 3.0;
        std::vector<double> expect;
        for (size_t i = static_cast<size_t>(kLo); i < static_cast<size_t>(kHi); ++i) {
            expect.push_back(uncond[i] + kScale * (cond[i] - uncond[i]));
        }
        const auto best = kLo + static_cast<double>(
            std::max_element(expect.begin(), expect.end()) - expect.begin());
        const auto got = std::get<double>(bridge.call(
            "guided", {{"tokens", tokens}, {"tokens2", tokens2}, {"scale", kScale},
                       {"lo", kLo}, {"hi", kHi}}));
        LOOM_CHECK(got == best);
    }

    // --- 7. `guidance.top_k`: the shortlist comes from the GUIDED logits and the values from the
    //        CONDITIONAL ones. Two different arrays decide two different things, which is the one
    //        property that makes this not a variation on the formula above -- so the check is that
    //        the answer is the conditional argmax OVER THE SHORTLIST, and that it is neither the
    //        unrestricted conditional argmax nor the guided one for at least one k. ---
    {
        constexpr double kScale = 4.0;
        std::vector<double> guided(6);
        for (size_t i = 0; i < 6; ++i) guided[i] = uncond[i] + kScale * (cond[i] - uncond[i]);
        std::set<double> shortlisted_answers;
        for (const int k : {1, 2, 3, 6}) {
            // The k best by the GUIDED logits...
            std::vector<size_t> order(6);
            for (size_t i = 0; i < 6; ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) { return guided[a] > guided[b]; });
            const std::set<size_t> keep(order.begin(), order.begin() + k);
            // ...scored by the CONDITIONAL ones.
            double best = -1;
            for (size_t i = 0; i < 6; ++i) {
                if (keep.count(i) && (best < 0 || cond[i] > cond[static_cast<size_t>(best)])) {
                    best = static_cast<double>(i);
                }
            }
            const auto got = std::get<double>(bridge.call(
                "guided_shortlist", {{"tokens", tokens}, {"tokens2", tokens2},
                                     {"scale", kScale}, {"gtop_k", static_cast<double>(k)}}));
            if (got != best) {
                std::fprintf(stderr, "guidance top_k %d: expected class %g, got %g\n", k, best, got);
            }
            LOOM_CHECK(got == best);
            shortlisted_answers.insert(got);
        }
        // k = 1 is the guided argmax and k = 6 is the conditional one; if those coincided this row
        // would be satisfied by an implementation that ignored one of the two arrays.
        const auto guided_best = static_cast<double>(
            std::max_element(guided.begin(), guided.end()) - guided.begin());
        std::fprintf(stderr, "shortlist k=1 -> guided argmax %g, k=6 -> conditional argmax %g\n",
                     guided_best, greedy);
        LOOM_CHECK(guided_best != greedy);
        LOOM_CHECK(shortlisted_answers.size() >= 2);
    }

    // --- 8. Every way of getting it wrong that the binding is supposed to catch. ---
    for (double c = 1; c <= 9; ++c) {
        const auto got = std::get<double>(bridge.call("expect_error", {{"case", c}}));
        if (got != c) std::fprintf(stderr, "sample_row error case %g returned %g\n", c, got);
        LOOM_CHECK(got == c);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
