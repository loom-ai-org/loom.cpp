// Module-owned output buffers and retrieval addressed by module name (BACKLOG.md P4.0.12).
//
// The claim under test is an EQUIVALENCE, so every case here is written against an oracle computed the
// old way, through Lua tables: a retained chain must produce the same numbers as a marshalled one, a
// retained argmax the same token as a marshalled one, and `loom.get_output` the same array
// `loom.run_subgraph` returns. If retention were merely "also fast", nothing would notice it being
// wrong; run side by side, a stale buffer or a mis-shaped copy is a mismatch.
//
// Reuses the attention-free toy-model GGUF fixture (make_builder_test_gguf.py) that
// test_lua_bridge_run_subgraph.cpp and test_graph_builder_shapes.cpp already share, so real weight
// lookups (token_embd.weight, output.weight, blk.0.*) work.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>

namespace {

// Two declared outputs, so `index` in a retained read/reference is exercised against something where
// getting it wrong is visible: "cur" is the raw embedding, "normed" its RMS-norm.
const char* kStageAJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"tokens","dtype":"i32","shape":["n_tokens"]}],
  "outputs": ["cur", "normed"],
  "nodes": [
    {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
    {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["normed"], "attrs": {"eps": "$rms_norm_eps"}}
  ]
})JSON";

// The consumer half of an inter-module edge: its only input is the [n_embd, n_tokens] tensor stage_a
// produces, which is exactly the intermediate a driver would otherwise round-trip through Lua.
const char* kStageBJson = R"JSON({
  "version": 1,
  "inputs": [{"name":"hidden","dtype":"f32","shape":["n_embd","n_tokens"]}],
  "output": "logits",
  "nodes": [
    {"op": "MUL_MAT", "inputs": ["output.weight", "hidden"], "outputs": ["logits"]}
  ]
})JSON";

const char* kScript = R"lua(
    -- The oracle: A's output marshalled into a Lua table and handed straight back as B's input, which
    -- is what every chained driver did before retained outputs existed.
    function chain_marshalled(inputs)
        local n = #inputs.tokens
        local cur = loom.run_subgraph('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local logits = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0}, {hidden = cur})
        return logits
    end

    -- The same chain with the intermediate never becoming a Lua value at all.
    function chain_retained(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local logits = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                          {hidden = {from = 'stage_a'}})
        return logits
    end

    -- ... and the same chain again with the generation pinned, the runtime half of the staleness
    -- guard. Passing it must not change any value.
    function chain_retained_pinned(inputs)
        local n = #inputs.tokens
        local gen = loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0},
                                                  {tokens = inputs.tokens})
        local logits = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                          {hidden = {from = 'stage_a', index = 1, gen = gen}})
        return logits
    end

    -- A PREFIX of a retained output (BACKLOG.md P4.3d), against the only oracle that isolates the copy
    -- itself: the same rows sliced out of the marshalled table. `cur` is [n_embd, n_tokens] and ne[0]
    -- is the fastest axis, so its first `k` rows are its first `k * n_embd` marshalled elements --
    -- which is exactly the claim `copy_row_prefix` makes about a contiguous view.
    function prefix_marshalled(inputs)
        local n = #inputs.tokens
        local cur = loom.run_subgraph('stage_a', {n_tokens = n, n_past = 0},
                                       {tokens = inputs.tokens})
        -- `#cur / n`, not the shape return: stage_a declares TWO outputs, so run_subgraph's returns are
        -- (cur, normed, shape_cur, shape_normed) and a two-local capture would bind `normed`.
        local head = {}
        for i = 1, inputs.rows * (#cur / n) do head[i] = cur[i] end
        return loom.run_subgraph('stage_b', {n_tokens = inputs.rows, n_past = 0}, {hidden = head})
    end

    function prefix_retained(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        return loom.run_subgraph('stage_b', {n_tokens = inputs.rows, n_past = 0},
                                  {hidden = {from = 'stage_a', rows = inputs.rows}})
    end

    -- `rows` equal to everything the module retained must be the untrimmed copy, not a near-miss: it is
    -- the case every driver hits whose audio happens to fill its last chunk exactly, and the case the
    -- family's own default expression produces.
    function prefix_all_retained(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local trimmed = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                           {hidden = {from = 'stage_a', rows = n}})
        local whole = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                         {hidden = {from = 'stage_a'}})
        local out = {}
        for i = 1, #whole do out[i] = trimmed[i] - whole[i] end
        return out
    end

    -- Retrieval by name, marshalling form: must equal what run_subgraph returns for the same output.
    function get_output_roundtrip(inputs)
        local n = #inputs.tokens
        local cur, normed = loom.run_subgraph('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local got_cur, cur_shape = loom.get_output('stage_a', 1)
        local got_normed = loom.get_output('stage_a', 2)
        local out = {cur_shape[1], cur_shape[2]}
        for i = 1, #cur do out[#out + 1] = got_cur[i] - cur[i] end
        for i = 1, #normed do out[#out + 1] = got_normed[i] - normed[i] end
        return out
    end

    -- `index = 2` must reach the SECOND declared output, not the first: feeding B the norm rather than
    -- the raw embedding has to produce different logits.
    function chain_retained_second_output(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        return loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0},
                                  {hidden = {from = 'stage_a', index = 2}})
    end

    function argmax_marshalled(inputs)
        local n = #inputs.tokens
        local cur = loom.run_subgraph('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local logits, shape = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0}, {hidden = cur})
        return loom.argmax_row(logits, shape[1], n - 1)
    end

    -- The same token id with nothing tensor-shaped crossing the boundary in either direction.
    function argmax_retained(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local gen = loom.run_subgraph_and_retain('stage_b', {n_tokens = n, n_past = 0},
                                       {hidden = {from = 'stage_a'}})
        return loom.argmax_row('stage_b', -1, gen)
    end

    -- A restricted argmax, against a Lua-side slice of the same marshalled row as its oracle. The
    -- window matters here rather than being decoration: the fixture's unrestricted argmax lands
    -- outside [lo, hi), so a binding that ignored the window would return that instead.
    function argmax_range_marshalled(inputs)
        local n = #inputs.tokens
        local cur = loom.run_subgraph('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local logits, shape = loom.run_subgraph('stage_b', {n_tokens = n, n_past = 0}, {hidden = cur})
        local n_vocab = shape[1]
        local base = (n - 1) * n_vocab
        local best = inputs.lo
        for id = inputs.lo, inputs.hi - 1 do
            if logits[base + id + 1] > logits[base + best + 1] then best = id end
        end
        return best
    end

    function argmax_range_retained(inputs)
        local n = #inputs.tokens
        loom.run_subgraph_and_retain('stage_a', {n_tokens = n, n_past = 0}, {tokens = inputs.tokens})
        local gen = loom.run_subgraph_and_retain('stage_b', {n_tokens = n, n_past = 0},
                                                  {hidden = {from = 'stage_a'}})
        return loom.argmax_row_range('stage_b', -1, inputs.lo, inputs.hi, gen)
    end

    -- Runs the SAME module at two different lengths, so the store has to reshape between them, then
    -- reads back the short one. `[n_embd, n_tokens]` changing under a persistent buffer is the case
    -- "whatever the last build produced" would get wrong.
    function reshape_then_read(inputs)
        loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
        loom.run_subgraph_and_retain('stage_a', {n_tokens = 1, n_past = 0}, {tokens = {5}})
        local got, shape = loom.get_output('stage_a', 1)
        local single = loom.run_subgraph('stage_a', {n_tokens = 1, n_past = 0}, {tokens = {5}})
        local out = {shape[1], shape[2]}
        for i = 1, #single do out[#out + 1] = got[i] - single[i] end
        return out
    end

    -- Every way of getting it wrong that the engine is supposed to catch. The expected substring is
    -- matched HERE rather than in C++ because a Lua error message is a string, and the bridge's own
    -- Value variant carries only numbers and arrays -- so the case index comes back on success and 0
    -- (with the real message printed) on a mismatch.
    local EXPECTED = {
        'has no retained outputs',
        "stale read of module 'stage_a'",
        'out of range',
        'must agree exactly',
        "unregistered module 'nope'",
        'is not a non-empty sub-range',
        'is not a non-empty sub-range',
        'so that is not a prefix of it',
        'so that is not a prefix of it',
        'must agree exactly',
    }

    function expect_error(inputs)
        local ok, err = pcall(function()
            if inputs.case == 1 then
                -- Read a module nothing has ever run with retention.
                loom.get_output('stage_never', 1)
            elseif inputs.case == 2 then
                -- Pin a generation, then re-run the module before reading: the classic stale read.
                local gen = loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0},
                                                          {tokens = {1, 3, 4}})
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {2, 2, 2}})
                loom.get_output('stage_a', 1, gen)
            elseif inputs.case == 3 then
                -- An output index past what the module declares.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.get_output('stage_a', 3)
            elseif inputs.case == 4 then
                -- A shape mismatch across an inter-module edge: A retained at 3 tokens, B built for 1.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph('stage_b', {n_tokens = 1, n_past = 0}, {hidden = {from = 'stage_a'}})
            elseif inputs.case == 5 then
                -- A reference to a module that isn't registered at all.
                loom.run_subgraph('stage_b', {n_tokens = 3, n_past = 0}, {hidden = {from = 'nope'}})
            elseif inputs.case == 6 then
                -- An id window running past the end of the row. Silently clamping it would turn a
                -- wrong constant in an export into a plausible answer over the wrong classes.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph_and_retain('stage_b', {n_tokens = 3, n_past = 0},
                                              {hidden = {from = 'stage_a'}})
                loom.argmax_row_range('stage_b', -1, 2, 99)
            elseif inputs.case == 7 then
                -- An empty window (lo == hi). There is no best of nothing.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph_and_retain('stage_b', {n_tokens = 3, n_past = 0},
                                              {hidden = {from = 'stage_a'}})
                loom.argmax_row_range('stage_b', -1, 3, 3)
            elseif inputs.case == 8 then
                -- A prefix longer than what the module retained. Clamping it would hand the decoder a
                -- shorter prompt than the driver's own arithmetic said, which nothing downstream can
                -- notice.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph('stage_b', {n_tokens = 4, n_past = 0},
                                   {hidden = {from = 'stage_a', rows = 4}})
            elseif inputs.case == 9 then
                -- An empty prefix. There is no zero-row segment; a driver that computed one has a
                -- wrong row formula, which is exactly what this is here to surface.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph('stage_b', {n_tokens = 1, n_past = 0},
                                   {hidden = {from = 'stage_a', rows = 0}})
            else
                -- `rows` disagreeing with the axis the consumer was built for. This is the failure the
                -- whole design leans on: PromptSegments feeds ONE local as both, so a formula that is
                -- wrong in the same way in both places is caught by the encoder's own row count, and a
                -- formula wrong in only one place is caught right here.
                loom.run_subgraph_and_retain('stage_a', {n_tokens = 3, n_past = 0}, {tokens = {1, 3, 4}})
                loom.run_subgraph('stage_b', {n_tokens = 3, n_past = 0},
                                   {hidden = {from = 'stage_a', rows = 2}})
            end
        end)
        if ok then return -1 end
        if string.find(err, EXPECTED[inputs.case], 1, true) then return inputs.case end
        print('case ' .. inputs.case .. ' raised an unexpected message: ' .. err)
        return 0
    end
)lua";

std::vector<double> as_array(const loom::LoomLuaBridge::Value& v) {
    return std::get<std::vector<double>>(v);
}

// Every difference the scripts return is `retained - marshalled` on values computed by the identical
// graph, so this is bit-equality in practice; the epsilon only guards against a compiler reordering the
// two runs' accumulations, which would be a legitimate difference and still nowhere near this large.
void check_all_zero(const std::vector<double>& diffs, size_t skip, const char* what) {
    for (size_t i = skip; i < diffs.size(); ++i) {
        if (std::fabs(diffs[i]) > 1e-6) {
            std::fprintf(stderr, "%s: element %zu differs by %g\n", what, i - skip, diffs[i]);
            LOOM_CHECK(false);
            return;
        }
    }
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/builder_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("stage_a", *model, loom::GraphTopology::parse(kStageAJson));
    bridge.register_module("stage_b", *model, loom::GraphTopology::parse(kStageBJson));
    // Registered and never retained, so "read a module that has nothing stored" stays reachable no
    // matter what the cases above have already run.
    bridge.register_module("stage_never", *model, loom::GraphTopology::parse(kStageBJson));
    bridge.load_script(kScript);

    const std::vector<double> tokens = {1, 3, 4};

    // --- 1. An inter-module edge that never becomes a Lua value produces identical numbers. ---
    const std::vector<double> marshalled = as_array(bridge.call("chain_marshalled", {{"tokens", tokens}}));
    LOOM_CHECK(marshalled.size() == 6 * 3); // [N_VOCAB=6, n_tokens=3]

    for (const char* fn : {"chain_retained", "chain_retained_pinned"}) {
        const std::vector<double> retained = as_array(bridge.call(fn, {{"tokens", tokens}}));
        LOOM_CHECK(retained.size() == marshalled.size());
        std::vector<double> diffs(retained.size());
        for (size_t i = 0; i < retained.size(); ++i) diffs[i] = retained[i] - marshalled[i];
        check_all_zero(diffs, 0, fn);
    }

    // --- 2. get_output returns exactly what run_subgraph would have marshalled, for every output. ---
    const std::vector<double> roundtrip = as_array(bridge.call("get_output_roundtrip", {{"tokens", tokens}}));
    LOOM_CHECK(roundtrip[0] == 4.0 && roundtrip[1] == 3.0); // [N_EMBD=4, n_tokens=3]
    LOOM_CHECK(roundtrip.size() == 2 + 12 + 12);
    check_all_zero(roundtrip, 2, "get_output_roundtrip");

    // --- 3. `index` selects among declared outputs rather than being ignored. ---
    const std::vector<double> second = as_array(bridge.call("chain_retained_second_output", {{"tokens", tokens}}));
    LOOM_CHECK(second.size() == marshalled.size());
    bool any_different = false;
    for (size_t i = 0; i < second.size(); ++i) {
        if (std::fabs(second[i] - marshalled[i]) > 1e-6) any_different = true;
    }
    LOOM_CHECK(any_different);

    // --- 4. The reducing read by module name agrees with the marshalled argmax. ---
    const auto argmax_ref = std::get<double>(bridge.call("argmax_marshalled", {{"tokens", tokens}}));
    const auto argmax_got = std::get<double>(bridge.call("argmax_retained", {{"tokens", tokens}}));
    std::fprintf(stderr, "argmax: marshalled %g, retained-by-name %g\n", argmax_ref, argmax_got);
    LOOM_CHECK(argmax_ref == argmax_got);

    // --- 4b. The same reduction restricted to an id window, against a Lua-side slice of the same row.
    // The window is chosen to EXCLUDE the unrestricted winner, so a binding that ignored `lo`/`hi`
    // would agree with check 4 above and disagree here. ---
    {
        const double unrestricted = argmax_got;
        // N_VOCAB is 6 in this fixture; take the half that does not contain the winner.
        const double lo = unrestricted < 3 ? 3 : 0;
        const double hi = unrestricted < 3 ? 6 : 3;
        const std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
            {"tokens", tokens}, {"lo", lo}, {"hi", hi},
        };
        const auto ref = std::get<double>(bridge.call("argmax_range_marshalled", args));
        const auto got = std::get<double>(bridge.call("argmax_range_retained", args));
        std::fprintf(stderr, "argmax_row_range over [%g, %g): marshalled %g, retained %g "
                              "(unrestricted was %g)\n", lo, hi, ref, got, unrestricted);
        LOOM_CHECK(ref == got);
        LOOM_CHECK(got >= lo && got < hi);
        LOOM_CHECK(got != unrestricted);
    }

    // --- 5. A store survives its geometry changing under it, which prefill->decode does every time. ---
    const std::vector<double> reshaped = as_array(bridge.call("reshape_then_read", {{"tokens", tokens}}));
    LOOM_CHECK(reshaped[0] == 4.0 && reshaped[1] == 1.0);
    LOOM_CHECK(reshaped.size() == 2 + 4);
    check_all_zero(reshaped, 2, "reshape_then_read");

    // --- 5b. A PREFIX of a retained output equals the same rows sliced out of the marshalled table
    // (BACKLOG.md P4.3d). This is the capability that lets a family-3 driver hand its decoder only the
    // audio rows its caller's real waveform produced, rather than the whole chunk-padded encoder
    // output -- and the oracle is a Lua slice, so it isolates the copy from everything else. ---
    for (double rows : {1.0, 2.0}) {
        const std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
            {"tokens", tokens}, {"rows", rows},
        };
        const std::vector<double> ref = as_array(bridge.call("prefix_marshalled", args));
        const std::vector<double> got = as_array(bridge.call("prefix_retained", args));
        std::fprintf(stderr, "prefix rows=%g: %zu element(s) each\n", rows, got.size());
        LOOM_CHECK(got.size() == ref.size());
        LOOM_CHECK(got.size() == static_cast<size_t>(6 * rows)); // [N_VOCAB=6, rows]
        std::vector<double> diffs(got.size());
        for (size_t i = 0; i < got.size(); ++i) diffs[i] = got[i] - ref[i];
        check_all_zero(diffs, 0, "prefix_retained");
        // ... and a prefix is genuinely a prefix: it must NOT equal the last rows, which is what a
        // copy that took an offset from the wrong end would produce.
        bool differs_from_tail = false;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::fabs(got[i] - marshalled[marshalled.size() - got.size() + i]) > 1e-6) {
                differs_from_tail = true;
            }
        }
        LOOM_CHECK(rows == static_cast<double>(tokens.size()) || differs_from_tail);
    }

    // `rows` equal to the whole retained output is the untrimmed copy exactly -- the case every
    // chunk-aligned waveform hits, and the one the family's default expression always produces.
    check_all_zero(as_array(bridge.call("prefix_all_retained", {{"tokens", tokens}})), 0,
                   "prefix_all_retained");

    // --- 6. The failure modes are errors that name the real problem, not silence or a crash.
    // The script returns the case index when the message matched, 0 when it raised something else
    // (printing it), and -1 when nothing was raised at all -- the case that would matter most. ---
    for (double case_id = 1; case_id <= 10; ++case_id) {
        const auto verdict = std::get<double>(bridge.call("expect_error", {{"case", case_id}}));
        std::fprintf(stderr, "error case %d: verdict %g\n", static_cast<int>(case_id), verdict);
        LOOM_CHECK(verdict == case_id);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
