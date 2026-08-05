// Sliding-window attention, end to end against the real HF model (BACKLOG.md P4.0.11a).
//
// **The prompt must be longer than the window or this test proves nothing.** Below `sliding_window` a
// banded mask and a full-causal one are elementwise identical, so a windowed model exported with no
// window at all still passes every short-prompt check -- which is exactly why P4.0.11 sat unfinished
// with "no numeric gate" until a windowed checkpoint existed. Gemma-3-270m-it declares
// `sliding_window: 512` over 18 layers in a 5:1 pattern (15 sliding, 3 full at indices 5/11/17), so
// this runs 600 tokens: past the window, and across all three full blocks.
//
// The expectation is the real `Gemma3ForCausalLM`'s own top-1, computed in float32 by
// `gemma_ref.py` and pasted here, the same discipline as test_e2e_lfm2_mil_export's HF-derived tokens.
//
// Set LOOM_SWA_GGUF to a GGUF produced by exporting /home/flavio/Dev/models/gemma-3-270m-it.

#include "test_util.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// The same deterministic sequence gemma_ref.py feeds the reference model: (i*37 + 101) % 60000.
std::vector<double> prompt_ids(int n) {
    std::vector<double> ids;
    ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ids.push_back(static_cast<double>((static_cast<long>(i) * 37 + 101) % 60000));
    }
    return ids;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_SWA_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SWA_GGUF to a GGUF exported from a sliding-window "
                              "checkpoint (gemma-3-270m-it) to run this check\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    // The routing itself, before any numbers: a windowed export declares a SECOND mask input, and the
    // driver has to build it with the window. Checked on the artifact rather than inferred from the
    // result, because a model that silently lost the second mask would still produce *a* number.
    LOOM_CHECK(driver_script.find("loom.causal_mask(#tokens, 0, 512)") != std::string::npos);

    std::unique_ptr<loom::KvCache> kv_cache;
    std::unique_ptr<loom::ConvStateCache> conv_state;
    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : model->topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(name));
        if (topo.uses_kv_cache() && kv_cache == nullptr) {
            kv_cache = loom::make_kv_cache(*model, backend.get());
        }
        loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
        if (topo.uses_conv_state() && conv_state == nullptr) {
            conv_state = loom::make_conv_state_cache(*model, backend.get());
        }
        loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state.get() : nullptr;
        bridge.register_module(name, *model, std::move(topo), cache_for_module, conv_for_module);
    }
    LOOM_CHECK(kv_cache != nullptr);
    bridge.load_script(driver_script);

    // **Forced token-by-token decode, not `infer` and not generation.** Two constraints force this
    // shape, and both are worth stating:
    //
    //  * `infer` cannot prefill 600 tokens here at all. It marshals the whole `[n_vocab, n_tokens]`
    //    logits tensor across the Lua boundary, and Gemma's 262144-wide vocab times 600 rows is 157M
    //    doubles against LuaJIT's ~134M array limit -- it raises "table overflow". For this vocab the
    //    cap lands at ~512 prompt tokens, which is exactly the window, so the prefill path physically
    //    cannot reach past it. That is a real driver limitation (BACKLOG.md), not a window bug.
    //  * Greedy GENERATION past the window is a weak gate on this checkpoint: from a random prompt the
    //    reference model collapses into a repeating `107, 2717`, which a wrong window would likely
    //    reproduce too.
    //
    // Feeding the reference model's own 600 tokens one at a time fixes both: each step returns
    // `[n_vocab, 1]`, and the comparison is against HF's real top-1 at a position 88 past the window,
    // computed over a non-degenerate input.
    bridge.load_script(R"LUA(
        function force_decode(inputs)
            local out, shape
            for i = 1, #inputs.tokens do
                out, shape = loom.run_subgraph('main_topology',
                    {n_tokens = 1, n_past = i - 1},
                    {tokens = {inputs.tokens[i]},
                     cache_position = loom.range(i - 1, 1),
                     attention_mask = loom.causal_mask(1, i - 1),
                     attention_mask_sw512 = loom.causal_mask(1, i - 1, 512)})
            end
            return loom.argmax_row(out, shape[1], 0)
        end
    )LUA");

    const std::vector<double> ids = prompt_ids(600);
    auto result = bridge.call("force_decode", {{"tokens", ids}});
    const double got = std::get<double>(result);
    constexpr double kHfTop1AtPosition599 = 49977.0;
    std::fprintf(stderr, "600-token prompt: expected HF top-1 %d, got %d\n",
                  static_cast<int>(kHfTop1AtPosition599), static_cast<int>(got));
    LOOM_CHECK(got == kHfTop1AtPosition599);

    // A second, SHORTER prompt that stays inside the window, through the ordinary `infer` prefill.
    // It must also match, which is what rules out "the band is applied everywhere, including where it
    // should be a no-op" -- the failure mode opposite to the one above, and equally invisible alone.
    // 64 rows of logits is well inside the marshalling cap.
    const std::vector<double> short_ids = prompt_ids(64);
    auto short_result = bridge.call("infer", {{"tokens", short_ids}});
    const double short_got = std::get<double>(short_result);
    constexpr double kHfTop1AtPosition63 = 2321.0;
    std::fprintf(stderr, "64-token prompt: expected HF top-1 %d, got %d\n",
                  static_cast<int>(kHfTop1AtPosition63), static_cast<int>(short_got));
    LOOM_CHECK(short_got == kHfTop1AtPosition63);

    LOOM_TEST_REPORT_AND_RETURN();
}
