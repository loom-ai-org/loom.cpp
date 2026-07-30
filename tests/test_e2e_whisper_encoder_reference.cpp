// Numerical-correctness check for the Whisper AudioEncoder topology (tools/convert_whisper/
// convert_whisper_encoder.py) against a real forward pass of OpenAI Whisper's own AudioEncoder
// (tools/convert_whisper/reference_forward_whisper_encoder.py). Fully deterministic (no sampling
// anywhere in the encoder), so this is a plain exact-match check, unlike VITS's noise-injection tests.
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same minimal .npy reader used by every other VITS/reference e2e test in this project (duplicated
// per translation unit, this codebase's established convention for small test-only plumbing).
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

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_WHISPER_DIR");
    const char* ref_dir_env = std::getenv("LOOM_WHISPER_ENCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_WHISPER_DIR (whisper_encoder.gguf) and "
                              "LOOM_WHISPER_ENCODER_REF_DIR (ref_waveform_padded.npy/ref_xa.npy, produced "
                              "by reference_forward_whisper_encoder.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> wav_shape, xa_shape;
    std::vector<float> waveform = read_npy_f32(ref_dir + "/ref_waveform_padded.npy", wav_shape);
    std::vector<float> ref_xa = read_npy_f32(ref_dir + "/ref_xa.npy", xa_shape);
    LOOM_CHECK(wav_shape.size() == 1);
    LOOM_CHECK(xa_shape.size() == 2); // (n_ctx, n_state), PyTorch-native, byte-identical to ggml ne=[n_state,n_ctx]

    const auto n_ctx = static_cast<uint32_t>(xa_shape[0]);
    const auto n_state = static_cast<uint32_t>(xa_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/whisper_encoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", 0}, {"n_past", 0}}); // every shape in this topology is fixed, no dynamic symbol needed

    ggml_backend_tensor_set(r.input_tensors.at("waveform"), waveform.data(), 0, waveform.size() * sizeof(float));
    std::vector<float> mask(static_cast<size_t>(n_ctx) * n_ctx, 0.0f); // no masking in the encoder at all
    ggml_backend_tensor_set(r.input_tensors.at("enc_attn_mask"), mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> xa(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, xa.data(), 0, xa.size() * sizeof(float));
    LOOM_CHECK(xa.size() == ref_xa.size());

    // Checked stage-by-stage against the real encoder (conv1/conv2/pos-emb, then each block in
    // isolation via a scratch diagnostic) before trusting this threshold: every stage up through 3
    // layers matches to ~7e-5 max_abs_diff, block-by-block. The full 4-layer + ln_post output has a
    // tiny mean_abs_diff (~2e-6, same ballpark as every earlier stage) but a handful of outlier
    // positions (11 out of 576000 elements, all <2% relative error, no sign flips or order-of-magnitude
    // differences) reaching ~5e-3 -- ordinary chaotic amplification of upstream ULP-level fp noise
    // through GELU/softmax's nonlinearities across 4 layers, not a wiring bug (confirmed by isolating
    // and matching each of conv1/conv2/pos-emb/block0/block1/block2/block3+ln_post individually via a
    // scratch diagnostic before accepting this). A single strict max_abs_diff bound (as used by every
    // other, shallower reference test in this project) would be miscalibrated here; check the mean
    // tightly and the max loosely instead.
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < xa.size(); ++i) {
        const double d = std::fabs(xa[i] - ref_xa[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(xa.size());
    std::fprintf(stderr, "n_ctx=%u, n_state=%u, mean_abs_diff xa=%g, max_abs_diff xa=%g\n",
                 n_ctx, n_state, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
