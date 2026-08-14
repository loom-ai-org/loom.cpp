// The claim KV-CACHE.md exists to make: a fused causal LM's `infer_with_past` -- prefill, then one
// token at a time against the persistent KV cache -- generates the SAME tokens as calling `infer` N
// times with a growing prompt (stage 3.4).
//
// **Why iterated `infer` is a valid oracle.** Each `infer` call is a fresh full prefill at `n_past = 0`
// over the whole prompt-so-far: it rewrites cache cells `[0, n_tokens)` and attends over exactly those,
// so it never reads a cell an earlier call left behind. It is the slow, obviously-correct way to
// generate, and it is what `tools/loom_cli` did before this entry existed. If the cache were wrong --
// cells written at the wrong offset, a mask that did not span the past, a layer index addressing the
// wrong slot -- the two would diverge, and they diverge LOUDLY: greedy decoding turns one wrong logit
// into a different token and then a different suffix.
//
// Both paths deliberately share ONE bridge and ONE cache, which buys a third property for free: a
// prefill issued after a generation must still be correct, i.e. it must not read the cells that
// generation left past `n_tokens`. That is re-checked at the end.
//
// Generic over any fused causal-LM GGUF rather than pinned to one checkpoint -- what makes a model
// eligible is a property of its graph (a cached ATTENTION node, and no op carrying cross-step state
// nothing holds; see `LoomGGUFExporter._non_cached_sequence_state`), so the test asks the file whether
// it has the entry and skips if not. Qwen3-0.6B and SmolLM2-360M qualify, and since P4.0.10 so does
// LFM2-350M: its ten ShortConv blocks are stateful across steps, but their state now has somewhere to
// live (`SHORT_CONV` + `ConvStateCache`), so the hybrid is eligible rather than excluded. That case is
// the whole reason this test allocates a conv-state store below.
//
// Set LOOM_CAUSAL_LM_KV_GGUF to any GGUF produced by `loom-export <hf-causal-lm-dir>`.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include "cpu_backend.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// Small ids, valid in any vocabulary this could be pointed at. The point of the check is agreement
// between two ways of running the same graph, not what the model says.
const std::vector<double> kPrompt = {1.0, 2.0, 3.0};
constexpr int kNewTokens = 12;

int32_t as_token(const loom::LoomLuaBridge::Value& v) {
    return static_cast<int32_t>(std::get<double>(v));
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CAUSAL_LM_KV_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_CAUSAL_LM_KV_GGUF to a GGUF produced by "
                              "`loom-export <hf-causal-lm-dir>` to run this check\n");
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    if (driver_script.find("function infer_with_past(") == std::string::npos) {
        std::fprintf(stderr, "skipping: '%s' has no infer_with_past entry -- either it was exported "
                              "before KV-CACHE.md stage 3, or its graph carries cross-step state no "
                              "store holds -- an UNFUSED conv/SSM/RWKV op is the real case\n",
                      gguf_env);
        return kSkipReturnCode;
    }

    // Sized from the file's own declared geometry, exactly as the whisper Lua test and loom_cli do --
    // a host needs no per-model struct to allocate it (KV-CACHE.md stage 1).
    std::unique_ptr<loom::KvCache> kv_cache;
    std::unique_ptr<loom::ConvStateCache> conv_state;
    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : model->topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(name));
        if (topo.uses_kv_cache() && kv_cache == nullptr) {
            kv_cache = loom::make_kv_cache(*model, backend.get());
        }
        loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
        // A hybrid's ShortConv blocks carry their own history, which the KV cache does not
        // hold -- allocated from the file's own loom.n_conv_* keys, same as above
        // (BACKLOG.md P4.0.10).
        if (topo.uses_conv_state() && conv_state == nullptr) {
            conv_state = loom::make_conv_state_cache(*model, backend.get());
        }
        loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state.get() : nullptr;
        bridge.register_module(name, *model, std::move(topo), cache_for_module, conv_for_module);
    }
    LOOM_CHECK(kv_cache != nullptr);
    bridge.load_script(driver_script);

    // --- Oracle: N full prefills over a growing prompt ---
    std::vector<int32_t> iterated;
    {
        std::vector<double> grown = kPrompt;
        for (int i = 0; i < kNewTokens; ++i) {
            const int32_t next = as_token(bridge.call("infer", {{"tokens", grown}}));
            iterated.push_back(next);
            grown.push_back(static_cast<double>(next));
        }
    }
    LOOM_CHECK(iterated.size() == static_cast<size_t>(kNewTokens));

    // --- The claim: one prefill plus N-1 cached steps ---
    std::vector<int32_t> cached;
    {
        loom::LoomLuaBridge::Value result = bridge.call("infer_with_past", {
            {"tokens", kPrompt},
            {"max_new_tokens", static_cast<double>(kNewTokens)},
        });
        const auto& raw = std::get<std::vector<double>>(result);
        for (double d : raw) cached.push_back(static_cast<int32_t>(d));
    }

    std::fprintf(stderr, "iterated infer  :");
    for (int32_t t : iterated) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\ninfer_with_past :");
    for (int32_t t : cached) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");

    LOOM_CHECK(cached.size() == iterated.size());
    for (size_t i = 0; i < iterated.size() && i < cached.size(); ++i) {
        LOOM_CHECK(cached[i] == iterated[i]);
    }

    // --- max_new_tokens is honoured, not merely defaulted ---
    {
        loom::LoomLuaBridge::Value result = bridge.call("infer_with_past", {
            {"tokens", kPrompt}, {"max_new_tokens", 3.0},
        });
        LOOM_CHECK(std::get<std::vector<double>>(result).size() == 3);
    }

    // --- eos_token stops the loop, and stops it AFTER emitting the token that matched (the same
    //     convention Whisper's own driver uses for its eot token, where a negative value disables it) ---
    {
        loom::LoomLuaBridge::Value result = bridge.call("infer_with_past", {
            {"tokens", kPrompt},
            {"max_new_tokens", static_cast<double>(kNewTokens)},
            {"eos_token", static_cast<double>(iterated.front())},
        });
        const auto& raw = std::get<std::vector<double>>(result);
        LOOM_CHECK(raw.size() == 1);
        LOOM_CHECK(!raw.empty() && static_cast<int32_t>(raw[0]) == iterated.front());
    }

    // --- A prefill issued after all of that must be unaffected by the cells generation left behind ---
    {
        const int32_t again = as_token(bridge.call("infer", {{"tokens", kPrompt}}));
        std::fprintf(stderr, "prefill after generation: %d (first generated was %d)\n",
                      again, iterated.front());
        LOOM_CHECK(again == iterated.front());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
