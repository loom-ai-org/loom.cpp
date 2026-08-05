// Numerical-correctness check for Kokoro's `bert_encoder` (plain Linear(768,512) + the real
// `.transpose(-1,-2)`, converted via convert_kokoro_bert_encoder.py), against the real checkpoint's own
// weights and a hand-rolled PyTorch reference. Exercises a genuine axis-convention crossing: CustomAlbert's
// own raw output is TIME-MAJOR (ne=[768,T]), not this project's usual CONV_1D-family [T,C] convention --
// see convert_kokoro_bert_encoder.py's own module docstring. Fully deterministic exact-match check. Skips
// cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_BERT_ENCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_bert_encoder.gguf) and "
                              "LOOM_KOKORO_BERT_ENCODER_REF_DIR (ref_bert_encoder_*.npy) to run this "
                              "numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kHidden = 768;
    constexpr uint32_t kOut = 512;

    std::vector<int64_t> x_shape, out_shape;
    std::vector<float> x = read_npy_f32(ref_dir + "/ref_bert_encoder_x.npy", x_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_bert_encoder_out.npy", out_shape);
    LOOM_CHECK(x_shape.size() == 2 && static_cast<uint32_t>(x_shape[1]) == kHidden);
    const auto T = static_cast<uint32_t>(x_shape[0]);
    LOOM_CHECK(static_cast<uint32_t>(out_shape[0]) == kOut && static_cast<uint32_t>(out_shape[1]) == T);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_bert_encoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    // x (ref: (T,768) row-major, time-major) is byte-identical to ggml ne=[768,T] -- no reordering.
    ggml_backend_tensor_set(r.input_tensors.at("x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == T);
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[1]) == kOut);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    // ref_out (512,T) row-major, channel-first) is byte-identical to ggml ne=[T,512] -- no reordering.
    LOOM_CHECK(out.size() == ref_out.size());

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - ref_out[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "T=%u, mean_diff=%g, max_diff=%g\n", T, mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
