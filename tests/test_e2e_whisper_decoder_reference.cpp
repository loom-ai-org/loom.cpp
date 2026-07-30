// Numerical-correctness check for the Whisper TextDecoder topology (tools/convert_whisper/
// convert_whisper_decoder.py) against a real, ONE-SHOT teacher-forced forward pass of OpenAI Whisper's
// own TextDecoder (tools/convert_whisper/reference_forward_whisper_decoder.py). n_past=0/n_tokens=T
// covers the whole causal triangle in a single call, exactly matching the real model's own
// kv_cache=None (non-incremental) path -- fully deterministic, no sampling anywhere in the decoder
// itself. Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_npy_f32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<float> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(float)));
    return data;
}

std::vector<int32_t> read_npy_i32(const std::string& path, std::vector<int64_t>& shape_out) {
    // ref_dec_tokens.npy is saved as np.int32 -- separate reader since read_npy_f32 assumes f32 payload.
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<int32_t> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(int32_t)));
    return data;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_WHISPER_DIR");
    const char* ref_dir_env = std::getenv("LOOM_WHISPER_DECODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_WHISPER_DIR (whisper_decoder.gguf) and "
                              "LOOM_WHISPER_DECODER_REF_DIR (ref_dec_*.npy, produced by "
                              "reference_forward_whisper_decoder.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> tok_shape, xa_shape, logits_shape;
    std::vector<int32_t> tokens = read_npy_i32(ref_dir + "/ref_dec_tokens.npy", tok_shape);
    std::vector<float> xa = read_npy_f32(ref_dir + "/ref_dec_xa.npy", xa_shape);
    std::vector<float> ref_logits = read_npy_f32(ref_dir + "/ref_dec_logits.npy", logits_shape);
    LOOM_CHECK(tok_shape.size() == 1);
    LOOM_CHECK(xa_shape.size() == 2);   // (n_audio_ctx, n_state), native PyTorch, byte-identical to ggml ne=[n_state,n_audio_ctx]
    LOOM_CHECK(logits_shape.size() == 2); // (n_tokens, n_vocab), native PyTorch, byte-identical to ggml ne=[n_vocab,n_tokens]

    const auto n_tokens = static_cast<uint32_t>(tok_shape[0]);
    const auto n_audio_ctx = static_cast<uint32_t>(xa_shape[0]);
    const auto n_state = static_cast<uint32_t>(xa_shape[1]);
    const auto n_vocab = static_cast<uint32_t>(logits_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/whisper_decoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const uint32_t n_layer = model->hparam_u32("n_layer");
    const uint32_t n_embd_k = model->hparam_u32("n_head_kv") * model->hparam_u32("n_embd_head_k");
    const uint32_t n_embd_v = model->hparam_u32("n_head_kv") * model->hparam_u32("n_embd_head_v");
    loom::KvCache kv_cache(n_layer, n_embd_k, n_embd_v, /*kv_size=*/n_tokens, backend.get());

    loom::GraphBuilder builder(topo, *model, backend.get(), &kv_cache);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", n_tokens}, {"n_past", /*n_past=*/0}});

    std::vector<int32_t> tokens_copy = tokens;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) positions[i] = static_cast<int32_t>(i);
    ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));

    // n_past=0, n_kv=n_tokens: plain causal triangle (mirrors Generator::write_inputs' own construction).
    std::vector<float> kq_mask(static_cast<size_t>(n_tokens) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (uint32_t j = 0; j < n_tokens; ++j) {
            kq_mask[static_cast<size_t>(i) * n_tokens + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(r.input_tensors.at("kq_mask"), kq_mask.data(), 0, kq_mask.size() * sizeof(float));

    ggml_backend_tensor_set(r.input_tensors.at("xa"), xa.data(), 0, xa.size() * sizeof(float));
    std::vector<float> xa_mask(static_cast<size_t>(n_audio_ctx) * n_tokens, 0.0f); // no masking on cross-attention
    ggml_backend_tensor_set(r.input_tensors.at("xa_mask"), xa_mask.data(), 0, xa_mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> logits(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, logits.data(), 0, logits.size() * sizeof(float));
    LOOM_CHECK(logits.size() == ref_logits.size());
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == n_vocab);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double d = std::fabs(logits[i] - ref_logits[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(logits.size());
    std::fprintf(stderr, "n_tokens=%u, n_vocab=%u, mean_abs_diff=%g, max_abs_diff=%g\n",
                 n_tokens, n_vocab, mean_abs_diff, max_abs_diff);
    // Same depth-calibrated tolerance rationale as test_e2e_whisper_encoder_reference.cpp (4 layers +
    // GELU/softmax nonlinearities can amplify upstream ULP noise at isolated logit positions without
    // indicating a wiring bug) -- check mean tightly, max loosely.
    LOOM_CHECK(mean_abs_diff < 1e-2);
    LOOM_CHECK(max_abs_diff < 5.0);

    // The actual thing that matters for greedy decoding: does the argmax token match at every position?
    for (uint32_t t = 0; t < n_tokens; ++t) {
        uint32_t best = 0;
        float best_val = logits[static_cast<size_t>(t) * n_vocab];
        uint32_t ref_best = 0;
        float ref_best_val = ref_logits[static_cast<size_t>(t) * n_vocab];
        for (uint32_t v = 1; v < n_vocab; ++v) {
            const float lv = logits[static_cast<size_t>(t) * n_vocab + v];
            if (lv > best_val) { best_val = lv; best = v; }
            const float rv = ref_logits[static_cast<size_t>(t) * n_vocab + v];
            if (rv > ref_best_val) { ref_best_val = rv; ref_best = v; }
        }
        LOOM_CHECK(best == ref_best);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
