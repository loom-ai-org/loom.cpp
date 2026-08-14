// The claim BACKLOG.md P4.0.14 exists to make: a synthesized causal-LM driver can prefill a prompt
// whose logits tensor is LARGER THAN LUA CAN HOLD.
//
// `loom.run_subgraph` marshals every output element into a Lua table, and LuaJIT's array part tops out
// near 2^27 entries -- so `n_tokens * n_vocab > 2^27` raised `table overflow` and there was no prompt
// length past it, at any speed. The cap is per checkpoint because the vocab is: ~512 prompt tokens for
// Gemma 3's 262144, ~883 for Qwen3's 151936, ~2048 for LFM2's 65536.
//
// **This is a capability check, not a performance one, which is why it belongs in ctest.** Every other
// gate on P4.0.12/P4.0.14 asks whether retained outputs produce the same numbers as marshalled ones
// (`test_lua_bridge_retained_outputs.cpp` runs each case against a marshalled oracle). There can be no
// such oracle here: the marshalled path does not reach this input at all. So the assertion is exactly
// that the call completes and returns a token id in range -- against a baseline tree it does not, and
// the error names the ceiling.
//
// Generic over any causal-LM GGUF with a synthesized driver, like its siblings: the prompt length is
// computed from the file's OWN vocab size, so pointing it at a different checkpoint moves the length
// rather than invalidating the test. LFM2-350M modular is the fixture that matters most -- it is the
// one path P4.0.12 left marshalling, and the reason P4.0.14 is a separate item -- but the flattened
// exports exercise the same driver text through a different builder.
//
// Set LOOM_PREFILL_CEILING_GGUF to a GGUF produced by `loom-export`; skips cleanly if unset.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// LuaJIT's array part tops out near 2^27 entries. A prompt is over the ceiling when
// n_tokens * n_vocab exceeds it -- the product is what overflows, not either factor.
constexpr int64_t kLuaArrayCeiling = 1 << 27;

// The logits width, read off the file rather than claimed about it -- and read off the WEIGHTS because
// no causal-LM export declares it as an hparam (`loom.*` carries n_layer, n_head_kv, the head dims and
// the cache size; the vocabulary is only ever a tensor dimension).
//
// For a causal LM the vocabulary is the largest dimension in the file, and not narrowly: the embedding
// matrix (and the head that is often tied to it) is [n_embd, n_vocab], and every other tensor is sized
// by n_embd, n_ff or a head count -- 65536 against 4096 for LFM2-350M, 262144 against 2048 for
// Gemma 3 270M. An estimate that came out too LARGE would make the prompt shorter than the ceiling and
// weaken the test, which is why the caller is told the number it derived; too small only makes the
// prompt longer than it needs to be.
int64_t widest_tensor_dim(const loom::GgufModel& model) {
    int64_t widest = 0;
    for (const auto& entry : model.weights()) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            widest = std::max(widest, entry.second->ne[d]);
        }
    }
    return widest;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_PREFILL_CEILING_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_PREFILL_CEILING_GGUF to a GGUF produced by "
                              "`loom-export` to run this check\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_env, backend.get());
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    const int64_t n_vocab = widest_tensor_dim(*model);
    LOOM_CHECK(n_vocab > 1000); // a vocabulary, not some incidental head-count dimension

    // Just past the ceiling, not comfortably past it: the logits tensor is n_tokens * n_vocab floats
    // and it is now allocated twice (the graph's own, plus the module's retained copy), so overshooting
    // buys nothing and costs real memory.
    const int64_t n_tokens = kLuaArrayCeiling / n_vocab + 16;
    std::fprintf(stderr, "vocab %lld -> marshalling ceiling at %lld prompt tokens; prefilling %lld\n",
                  static_cast<long long>(n_vocab),
                  static_cast<long long>(kLuaArrayCeiling / n_vocab),
                  static_cast<long long>(n_tokens));
    LOOM_CHECK(n_tokens * n_vocab > kLuaArrayCeiling);

    loom::LoomLuaBridge bridge(backend.get());
    std::unique_ptr<loom::KvCache> kv_cache;
    std::unique_ptr<loom::ConvStateCache> conv_state;
    for (const std::string& mod_name : model->topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(mod_name));
        if (topo.uses_kv_cache() && kv_cache == nullptr) {
            kv_cache = loom::make_kv_cache(*model, backend.get());
        }
        loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
        if (topo.uses_conv_state() && conv_state == nullptr) {
            conv_state = loom::make_conv_state_cache(*model, backend.get());
        }
        loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state.get() : nullptr;
        bridge.register_module(mod_name, *model, std::move(topo), cache_for_module, conv_for_module);
    }
    bridge.load_script(driver_script);

    // Token ids small enough to be valid in any vocabulary. What the model predicts is not the point --
    // that the driver can be asked at all is.
    std::vector<double> prompt(static_cast<size_t>(n_tokens));
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<double>(1 + (i % 100));

    // Caught rather than left to unwind: a driver that still marshals raises `table overflow` from
    // inside Lua, and an uncaught loom::Error aborts the process before the harness can say which check
    // failed or why. This is the one failure this test exists to report, so it reports it.
    int64_t token = -1;
    try {
        token = static_cast<int64_t>(std::get<double>(bridge.call("infer", {{"tokens", prompt}})));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill of %lld tokens FAILED: %s\n  -- a driver whose last call is "
                              "loom.run_subgraph marshals %lld logits into a Lua table, which is past "
                              "LuaJIT's array limit. Retaining and reducing by name is what removes it "
                              "(BACKLOG.md P4.0.14).\n",
                      static_cast<long long>(n_tokens), e.what(),
                      static_cast<long long>(n_tokens * n_vocab));
        LOOM_CHECK(false);
        LOOM_TEST_REPORT_AND_RETURN();
    }
    std::fprintf(stderr, "prefill of %lld tokens returned token id %lld\n",
                  static_cast<long long>(n_tokens), static_cast<long long>(token));
    // In range is the whole assertion: a marshalled driver never gets here, and a reduction reading the
    // wrong tensor (or a row past the end) would not land inside the vocabulary by accident.
    LOOM_CHECK(token >= 0 && token < n_vocab);

    LOOM_TEST_REPORT_AND_RETURN();
}
