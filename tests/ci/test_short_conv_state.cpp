// SHORT_CONV, the conv family's cached primitive (BACKLOG.md P4.0.10).
//
// The claim this file exists to prove is the same one KV-CACHE.md 3.4 makes for attention, one level
// down: **stepping a causal convolution a token at a time against a persistent state produces exactly
// what convolving the whole sequence in one shot produces.** Everything else here is a guard around
// that.
//
// Exercised through PrimitiveRegistry directly rather than through a GGUF, so the property is pinned
// down independently of any exporter change that might also be wrong.

#include "ggml_test_helpers.h"
#include "test_util.h"
#include "cpu_backend.h"

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using loom_test::GgmlScratch;
using loom_test::get_f32;
using loom_test::set_f32;

namespace {

const loom::PrimitiveFn& op(const std::string& name) {
    return loom::PrimitiveRegistry::instance().get(name);
}

constexpr int64_t kKernel = 3;   // LFM2's ShortConv width; state depth is kernel - 1 = 2
constexpr int64_t kChannels = 2;

// One channel-major depthwise kernel, [K, 1, C].
const std::vector<float> kWeights = {0.5f, -1.25f, 2.0f,    // channel 0
                                      1.5f, 0.25f, -0.75f}; // channel 1

// Column-major over [T, C]: all of channel 0's timesteps, then all of channel 1's.
std::vector<float> signal(int64_t total) {
    std::vector<float> v(static_cast<size_t>(total * kChannels));
    for (int64_t c = 0; c < kChannels; ++c) {
        for (int64_t t = 0; t < total; ++t) {
            v[static_cast<size_t>(c * total + t)] = static_cast<float>(t + 1) * (c == 0 ? 0.5f : -0.25f)
                                                     + static_cast<float>(c);
        }
    }
    return v;
}

// A slice of `signal(total)`'s timesteps [begin, begin+len), still column-major over [len, C].
std::vector<float> signal_slice(int64_t total, int64_t begin, int64_t len) {
    const std::vector<float> all = signal(total);
    std::vector<float> v(static_cast<size_t>(len * kChannels));
    for (int64_t c = 0; c < kChannels; ++c) {
        for (int64_t t = 0; t < len; ++t) {
            v[static_cast<size_t>(c * len + t)] = all[static_cast<size_t>(c * total + begin + t)];
        }
    }
    return v;
}

// Runs one SHORT_CONV call and returns its output, expanding the state write first exactly as
// GraphBuilder does -- side-effect roots before the declared output, in push order.
std::vector<float> short_conv_step(ggml_backend_t backend, loom::ConvStateCache* cache,
                                    int64_t n_tokens, uint32_t n_past, uint32_t layer,
                                    const std::vector<float>& data_values) {
    GgmlScratch s(backend);
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kKernel, 1, kChannels);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, n_tokens, kChannels);
    ggml_set_input(data);

    loom::SymbolEnv env;
    env.set("n_tokens", static_cast<double>(n_tokens));
    env.set("n_past", static_cast<double>(n_past));
    std::vector<ggml_tensor*> side_effects;
    loom::PrimitiveContext pc{s.ctx.get(), env, /*kv_cache=*/nullptr, cache, /*kv_cells=*/nullptr, &side_effects};

    nlohmann::json attrs = {{"layer", static_cast<int>(layer)}};
    if (cache == nullptr) attrs["conv_state"] = false;
    ggml_tensor* out = op("SHORT_CONV")(pc, {kernel, data}, attrs)[0];
    LOOM_CHECK(out->ne[0] == n_tokens);
    LOOM_CHECK(out->ne[1] == kChannels);

    ggml_cgraph* gf = ggml_new_graph(s.ctx.get());
    for (ggml_tensor* eff : side_effects) {
        ggml_build_forward_expand(gf, eff);
    }
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_alloc_graph(s.galloc.get(), gf);

    set_f32(kernel, kWeights);
    set_f32(data, data_values);
    s.compute(gf);
    return get_f32(out);
}

// Stateless reference run: no cache, its own backend. Every "one shot" expectation below goes through
// here, so the comparison is against the path the exporter emitted before P4.0.10 existed.
std::vector<float> short_conv_one_shot(int64_t n_tokens, const std::vector<float>& data_values) {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    return short_conv_step(backend.get(), /*cache=*/nullptr, n_tokens, /*n_past=*/0, /*layer=*/0, data_values);
}

bool close(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-5f) return false;
    }
    return true;
}

// The stateless form must reproduce what the exporter used to emit for an LFM2 ShortConv block: a
// depthwise conv padded by kernel-1 on both sides, with the causal prefix kept. If these ever disagree
// the fusion in P4.0.10 would silently change LFM2's prefill numerics, which no other test here covers.
void test_stateless_short_conv_matches_padded_conv_and_slice() {
    constexpr int64_t kT = 6;
    GgmlScratch s;
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kKernel, 1, kChannels);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kT, kChannels);
    ggml_set_input(data);

    loom::SymbolEnv env;
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};
    ggml_tensor* padded = op("CONV_1D_DW")(pc, {kernel, data},
                                            {{"s0", 1}, {"p0", kKernel - 1}, {"d0", 1}})[0];
    ggml_tensor* sliced = op("CONT")(pc, {op("VIEW")(pc, {padded}, {{"shape", {kT, kChannels}}})[0]}, {})[0];

    ggml_cgraph* gf = s.expand(sliced);
    set_f32(kernel, kWeights);
    set_f32(data, signal(kT));
    s.compute(gf);
    const std::vector<float> reference = get_f32(sliced);

    const std::vector<float> stateless = short_conv_one_shot(kT, signal(kT));
    LOOM_CHECK(close(stateless, reference));
}

// The whole point. Prefill 4 tokens, then decode tokens 5, 6 and 7 one at a time, and require the
// concatenation to equal a single 7-token call -- which is what a KV-cached decode loop needs from
// every op that mixes along the token axis.
void test_incremental_decode_matches_one_shot() {
    constexpr int64_t kPrefill = 4;
    constexpr int64_t kTotal = 7;

    const std::vector<float> one_shot = short_conv_one_shot(kTotal, signal(kTotal));

    ggml_backend_ptr backend(loom_test::cpu_backend());
    loom::ConvStateCache cache(/*n_layer=*/1, /*n_state=*/kKernel - 1, kChannels, backend.get());

    std::vector<std::vector<float>> stepped;
    stepped.push_back(short_conv_step(backend.get(), &cache, kPrefill, /*n_past=*/0, /*layer=*/0,
                                       signal_slice(kTotal, 0, kPrefill)));
    for (int64_t t = kPrefill; t < kTotal; ++t) {
        stepped.push_back(short_conv_step(backend.get(), &cache, 1, static_cast<uint32_t>(t), /*layer=*/0,
                                           signal_slice(kTotal, t, 1)));
    }

    // Rebuild the [kTotal, C] column-major output from the per-step [len, C] pieces.
    std::vector<float> joined(static_cast<size_t>(kTotal * kChannels));
    int64_t written = 0;
    for (const std::vector<float>& piece : stepped) {
        const int64_t len = static_cast<int64_t>(piece.size()) / kChannels;
        for (int64_t c = 0; c < kChannels; ++c) {
            for (int64_t t = 0; t < len; ++t) {
                joined[static_cast<size_t>(c * kTotal + written + t)] = piece[static_cast<size_t>(c * len + t)];
            }
        }
        written += len;
    }
    LOOM_CHECK(close(joined, one_shot));
}

// A prompt SHORTER than the state depth still has to work: the history is the concatenated
// [zeros, x] buffer's tail, so a 1-token prefill leaves a zero-padded window rather than reading past
// the front of its own input. Decoding on from there must still track the one-shot answer.
void test_prefill_shorter_than_state_depth() {
    constexpr int64_t kTotal = 4;
    const std::vector<float> one_shot = short_conv_one_shot(kTotal, signal(kTotal));

    ggml_backend_ptr backend(loom_test::cpu_backend());
    loom::ConvStateCache cache(1, kKernel - 1, kChannels, backend.get());

    std::vector<float> joined;
    std::vector<float> first = short_conv_step(backend.get(), &cache, 1, 0, 0, signal_slice(kTotal, 0, 1));
    std::vector<std::vector<float>> pieces{first};
    for (int64_t t = 1; t < kTotal; ++t) {
        pieces.push_back(short_conv_step(backend.get(), &cache, 1, static_cast<uint32_t>(t), 0,
                                          signal_slice(kTotal, t, 1)));
    }
    joined.resize(static_cast<size_t>(kTotal * kChannels));
    for (int64_t c = 0; c < kChannels; ++c) {
        for (int64_t t = 0; t < kTotal; ++t) {
            joined[static_cast<size_t>(c * kTotal + t)] = pieces[static_cast<size_t>(t)][static_cast<size_t>(c)];
        }
    }
    LOOM_CHECK(close(joined, one_shot));
}

// n_past == 0 means "no history" -- so a prefill issued AFTER a generation must not see the window that
// generation left behind. This is what keeps iterated `infer` a valid oracle for `infer_with_past`
// (KV-CACHE.md 3.4) once conv layers are in the graph, and it is the property that would break if
// op_short_conv read the slot unconditionally.
void test_prefill_after_generation_ignores_stale_state() {
    constexpr int64_t kT = 5;
    const std::vector<float> clean = short_conv_one_shot(kT, signal(kT));

    ggml_backend_ptr backend(loom_test::cpu_backend());
    loom::ConvStateCache cache(1, kKernel - 1, kChannels, backend.get());

    // Dirty the slot with an unrelated sequence, then prefill the real one at n_past = 0.
    short_conv_step(backend.get(), &cache, kT, 0, 0, signal(kT));
    short_conv_step(backend.get(), &cache, 1, static_cast<uint32_t>(kT), 0, signal_slice(kT + 1, kT, 1));

    const std::vector<float> after = short_conv_step(backend.get(), &cache, kT, /*n_past=*/0, 0, signal(kT));
    LOOM_CHECK(close(after, clean));
}

// Per-layer slots must not alias: two layers stepped in lockstep with different inputs each have to see
// their own history. A single shared slot passes every test above and fails this one.
void test_layers_have_independent_state() {
    constexpr int64_t kTotal = 5;
    constexpr int64_t kPrefill = 3;
    ggml_backend_ptr backend(loom_test::cpu_backend());
    loom::ConvStateCache cache(/*n_layer=*/2, kKernel - 1, kChannels, backend.get());

    const std::vector<float> l0_one_shot = short_conv_one_shot(kTotal, signal(kTotal));

    short_conv_step(backend.get(), &cache, kPrefill, 0, /*layer=*/0, signal_slice(kTotal, 0, kPrefill));
    // Layer 1 gets a DIFFERENT prefill in between, which would clobber layer 0's window if they aliased.
    short_conv_step(backend.get(), &cache, kPrefill, 0, /*layer=*/1, signal_slice(kTotal + 3, 3, kPrefill));

    std::vector<float> tail;
    for (int64_t t = kPrefill; t < kTotal; ++t) {
        const std::vector<float> step =
            short_conv_step(backend.get(), &cache, 1, static_cast<uint32_t>(t), /*layer=*/0,
                            signal_slice(kTotal, t, 1));
        tail.insert(tail.end(), step.begin(), step.end());
    }
    // tail is [c0_t3, c1_t3, c0_t4, c1_t4]; compare against the one-shot run's same positions.
    bool ok = true;
    for (int64_t t = kPrefill; t < kTotal; ++t) {
        for (int64_t c = 0; c < kChannels; ++c) {
            const float got = tail[static_cast<size_t>((t - kPrefill) * kChannels + c)];
            const float want = l0_one_shot[static_cast<size_t>(c * kTotal + t)];
            if (std::fabs(got - want) > 1e-5f) ok = false;
        }
    }
    LOOM_CHECK(ok);
}

// The negative gate: a topology asking for state without a cache registered must say so, not read
// garbage. Same failure mode op_attention already guards (primitives_attention.cpp:51).
void test_missing_cache_raises() {
    GgmlScratch s;
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kKernel, 1, kChannels);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, kChannels);
    loom::SymbolEnv env;
    env.set("n_tokens", 4);
    env.set("n_past", 0);
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};

    bool threw = false;
    try {
        op("SHORT_CONV")(pc, {kernel, data}, {{"layer", 0}});
    } catch (const loom::SchemaError& e) {
        threw = true;
        LOOM_CHECK(std::string(e.what()).find("no ConvStateCache") != std::string::npos);
    }
    LOOM_CHECK(threw);
}

// A width-1 kernel carries no history at all; emitting it as SHORT_CONV is an exporter bug, and a
// zero-length state slot would otherwise produce an empty concat and a confusing shape error later.
void test_degenerate_kernel_raises() {
    GgmlScratch s;
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, 1, 1, kChannels);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 4, kChannels);
    loom::SymbolEnv env;
    env.set("n_tokens", 4);
    env.set("n_past", 0);
    loom::PrimitiveContext pc{s.ctx.get(), env, nullptr};

    bool threw = false;
    try {
        op("SHORT_CONV")(pc, {kernel, data}, {{"layer", 0}});
    } catch (const loom::SchemaError& e) {
        threw = true;
        LOOM_CHECK(std::string(e.what()).find("position-wise") != std::string::npos);
    }
    LOOM_CHECK(threw);
}

} // namespace

int main() {
    test_stateless_short_conv_matches_padded_conv_and_slice();
    test_incremental_decode_matches_one_shot();
    test_prefill_shorter_than_state_depth();
    test_prefill_after_generation_ignores_stale_state();
    test_layers_have_independent_state();
    test_missing_cache_raises();
    test_degenerate_kernel_raises();

    LOOM_TEST_REPORT_AND_RETURN();
}
