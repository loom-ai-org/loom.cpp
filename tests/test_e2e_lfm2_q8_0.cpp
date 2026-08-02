// Quantized-weight verification for the MIL exporter's `quantize=` kwarg (EXPORT-BACKLOG.md item 6),
// applied to a real model for the first time (test_e2e_qwen3_q8_0.cpp proved the underlying mechanism --
// GgufModel::load/op_mul_mat/op_get_rows need no changes to run genuinely quantized weights -- against
// Qwen3-0.6B-Base; this is the LFM2-specific numerical check that was blocked until EXPORT-BACKLOG.md item
// 3's dynamic-shape/GQA fixes landed).
//
// Compares full logit vectors (not just argmax) between the F32 monolithic LFM2 GGUF
// (export_lfm2_monolithic.py) and a Q8_0-quantized export of the exact same model
// (export_lfm2_monolithic.py's backend() call with quantize="Q8_0" added), at the last token position,
// for the same two prompt lengths test_e2e_lfm2_mil_export.cpp uses. Bypasses the driver's own
// `main(inputs)` entry point (which only returns an argmax'd token id) via a small ad-hoc Lua script that
// calls the same `loom.run_subgraph`/`loom.range`/`loom.causal_mask` bindings directly and returns the raw
// logits tensor plus its vocab-size, so real per-logit differences can be measured here in C++.
//
// Not generated at ctest time (needs the real LFM2-350M checkpoint + coremltools + a quantized export) --
// skips cleanly if the fixtures aren't present, same convention as test_e2e_qwen3_q8_0.cpp /
// test_e2e_lfm2_mil_export.cpp. To (re)generate: export_lfm2_monolithic.py for the F32 reference, and the
// same script with `quantize="Q8_0"` added to its `backend(...)` call (writing to a different
// `output_path`) for the quantized fixture -- or point LOOM_LFM2_MONOLITHIC_GGUF /
// LOOM_LFM2_MONOLITHIC_Q8_0_GGUF at existing copies of each.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// Ad-hoc driver script: identical prologue to the real exported `main(inputs)` (see
// test_e2e_lfm2_mil_export.cpp's dump of `model.driver_script`), except it returns the raw logits array
// (with the vocab size appended as its last element, since LoomLuaBridge::call only returns one flat
// array) instead of an already-argmax'd token id.
const char* kGetLogitsScript = R"(
function get_logits_and_shape(inputs)
    local tokens = inputs.tokens
    local cache_position = loom.range(0, #tokens)
    local attention_mask = loom.causal_mask(#tokens, 0)
    local out, shape = loom.run_subgraph('main_topology', #tokens, 0, {tokens = tokens, cache_position = cache_position, attention_mask = attention_mask})
    out[#out + 1] = shape[1]
    return out
end
)";

// Last-position logits (vocab-sized) plus the vocab size itself.
struct LastRowResult {
    std::vector<double> logits;
    int n_vocab = 0;
};

LastRowResult last_row_logits(const std::string& gguf_path, const std::vector<double>& prompt) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model->has_topology("main_topology"));

    loom::LoomLuaBridge bridge(backend.get());
    bridge.register_module("main_topology", *model, loom::GraphTopology::parse(model->topology_json("main_topology")), nullptr);
    bridge.load_script(kGetLogitsScript);

    loom::LoomLuaBridge::Value result = bridge.call("get_logits_and_shape", {{"tokens", prompt}});
    std::vector<double> flat = std::get<std::vector<double>>(result);

    LastRowResult out;
    out.n_vocab = static_cast<int>(flat.back());
    flat.pop_back();
    LOOM_CHECK(out.n_vocab > 0);
    const size_t n_tokens = prompt.size();
    LOOM_CHECK(flat.size() == n_tokens * static_cast<size_t>(out.n_vocab));

    const auto* row = flat.data() + (n_tokens - 1) * static_cast<size_t>(out.n_vocab);
    out.logits.assign(row, row + out.n_vocab);
    return out;
}

int argmax(const std::vector<double>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

} // namespace

int main() {
    const char* f32_env = std::getenv("LOOM_LFM2_MONOLITHIC_GGUF");
    const char* q8_env = std::getenv("LOOM_LFM2_MONOLITHIC_Q8_0_GGUF");
    const std::string f32_path = f32_env != nullptr ? f32_env : "lfm2_350m_monolithic.gguf";
    const std::string q8_path = q8_env != nullptr ? q8_env : "lfm2_350m_monolithic_q8_0.gguf";

    if (!path_exists(f32_path) || !path_exists(q8_path)) {
        std::fprintf(stderr,
                      "skipping: F32 reference '%s' or quantized fixture '%s' not found (set "
                      "LOOM_LFM2_MONOLITHIC_GGUF / LOOM_LFM2_MONOLITHIC_Q8_0_GGUF, or produce them via "
                      "export_lfm2_monolithic.py -- once plain, once with quantize=\"Q8_0\" added to its "
                      "backend() call and a different output_path)\n",
                      f32_path.c_str(), q8_path.c_str());
        return kSkipReturnCode;
    }

    // Same two prompts as test_e2e_lfm2_mil_export.cpp -- 3 tokens (tight F32 top-1/top-2 margin, 0.135
    // logit units) and 7 tokens (comfortable margin, 2.873 logit units).
    const std::vector<std::vector<double>> prompts = {
        {1, 2, 3},
        {1, 2, 3, 4, 5, 6, 7},
    };

    for (const auto& prompt : prompts) {
        const LastRowResult f32 = last_row_logits(f32_path, prompt);
        const LastRowResult q8 = last_row_logits(q8_path, prompt);
        LOOM_CHECK(f32.n_vocab == q8.n_vocab);

        double max_abs_diff = 0.0;
        bool all_finite = true;
        for (int i = 0; i < f32.n_vocab; ++i) {
            const double d = std::fabs(f32.logits[i] - q8.logits[i]);
            max_abs_diff = std::max(max_abs_diff, d);
            all_finite = all_finite && std::isfinite(q8.logits[i]);
        }
        LOOM_CHECK(all_finite);

        const int f32_top1 = argmax(f32.logits);
        const int q8_top1 = argmax(q8.logits);
        std::fprintf(stderr,
                      "prompt of %zu tokens: max abs logit diff = %f, top-1 f32=%d q8=%d%s\n",
                      prompt.size(), max_abs_diff, f32_top1, q8_top1,
                      f32_top1 == q8_top1 ? "" : " (diverged -- expected when the F32 margin is tighter "
                                                  "than Q8_0's quantization noise, not a correctness bug)");

        // Tolerance measured empirically (see EXPORT-BACKLOG.md item 6), not guessed upfront: real max abs
        // diff across both prompt lengths was 1.52 / 2.68 (Q8_0's ~8-bit per-block quantization of 93
        // matmul-weight tensors compounding through 16 layers of a smaller, ShortConv+GQA architecture --
        // a bigger relative effect than Qwen3-0.6B-Base's own 0.45-0.79, but still small next to typical
        // logit magnitudes of ~10-15). Not LOOM_CHECK'd on argmax match itself (same reasoning as
        // test_e2e_qwen3_q8_0.cpp: Q8_0 is real lossy compression, not a bit-exact reformat) -- the real
        // check is that the divergence stays in this bounded range, not "doesn't crash."
        LOOM_CHECK(max_abs_diff < 4.0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
