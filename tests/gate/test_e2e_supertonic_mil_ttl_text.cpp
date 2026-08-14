// Numerical-correctness check for the MIL-traced SupertonicTTS "ttl_text" topology
// (supertonic_export.py, part of supertonic_mil.gguf) against the SAME real-module reference fixture
// the bespoke conversion's own test_e2e_supertonic_ttl_text.cpp already uses
// (reference_forward_supertonic_ttl_text.py, T=10) -- valid ground truth directly, no regeneration
// needed.
//
// THIS is the test that answers P4.6's central question, and it is the one that first answered it
// "no". The topology's text axis is `txt_len` wide; the reference is 10 real ids run through the real
// module at T=10 with an all-ones mask, i.e. exactly what the reference implementation does for a
// single utterance (`TextVectorizer.tokenize` pads to the longest string in the batch, so a batch of
// one is never padded). Feeding the same ten ids padded to `txt_len` and comparing the first ten
// columns therefore asks: does padding change the answer?
//
// Measured in PyTorch first, before any of this was exported: with a stock `ConvNextBlock` it does,
// by 1.77 max-abs on a tensor whose own max is 1.82 -- 97% wrong, not a near-miss. The mechanism is
// the block's `F.pad(mode="replicate")`: on a masked tensor the "edge" it replicates is a ZERO column,
// where the unpadded run replicates the last REAL one. supertonic_export.py's `_edge_fill` is what
// closes that gap, and this comparison is what holds it closed. See BACKLOG.md P4.6.
//
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"
#include "fixtures.h"
#include "supertonic_buckets.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::vector<int32_t> read_i32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int32_t> data(static_cast<size_t>(bytes) / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (supertonic_mil.gguf, produced by "
                              "export_supertonic_mil.py) and LOOM_SUPERTONIC_REF_DIR (ttl_*.bin, produced "
                              "by reference_forward_supertonic_ttl_text.py) to run this check\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ttl_txt_ids.bin");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ttl_txt_ids.bin not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    const std::vector<int32_t> txt_ids = read_i32_binary(ref_dir + "/ttl_txt_ids.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/ttl_expected_stl_emb.bin");
    const std::vector<float> expected_txt_emb = read_f32_binary(ref_dir + "/ttl_expected_txt_emb.bin");
    LOOM_CHECK(txt_ids.size() == 10);
    LOOM_CHECK(stl_emb.size() == 50 * 256);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    const uint32_t n_real = static_cast<uint32_t>(txt_ids.size());
    uint32_t t_text = 0;
    const std::string topo_name = loom_test::supertonic_bucket_topology(*model, "ttl_text", n_real,
                                                                        &t_text);
    LOOM_CHECK(!topo_name.empty());
    std::fprintf(stderr, "bucket: %s (%u real ids)\n", topo_name.c_str(), n_real);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(topo_name));
    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", t_text}, {"n_past", 0}});

    // The driver's own padding, done by hand -- see this file's header and the "dp" test's.
    std::vector<int32_t> txt_ids_copy(t_text, 162);
    std::copy(txt_ids.begin(), txt_ids.end(), txt_ids_copy.begin());
    std::vector<float> txt_msk(t_text, 0.0f);
    std::fill(txt_msk.begin(), txt_msk.begin() + n_real, 1.0f);
    std::vector<float> stl_emb_copy = stl_emb;
    ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids_copy.data(), 0, txt_ids_copy.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb_copy.data(), 0, stl_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_msk"), txt_msk.data(), 0, txt_msk.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    std::vector<float> txt_emb(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, txt_emb.data(), 0, txt_emb.size() * sizeof(float));

    // ne=[t_text, 256], T-fast, against a reference that is (256, n_real) row-major -- so the
    // channel stride differs between the two and the comparison has to index rather than memcmp.
    constexpr uint32_t kTxtDim = 256;
    LOOM_CHECK(txt_emb.size() == static_cast<size_t>(t_text) * kTxtDim);
    LOOM_CHECK(expected_txt_emb.size() == static_cast<size_t>(n_real) * kTxtDim);
    double max_abs_diff = 0.0;
    for (uint32_t c = 0; c < kTxtDim; ++c) {
        for (uint32_t t = 0; t < n_real; ++t) {
            const double got = txt_emb[static_cast<size_t>(c) * t_text + t];
            const double want = expected_txt_emb[static_cast<size_t>(c) * n_real + t];
            max_abs_diff = std::max(max_abs_diff, std::fabs(got - want));
        }
    }
    // The padded tail must be exactly zero, not merely small: the real module's last act is
    // `x_t.transpose(1, 2) * txt_msk`, so anything nonzero out there means the mask was not applied.
    double max_abs_pad = 0.0;
    for (uint32_t c = 0; c < kTxtDim; ++c) {
        for (uint32_t t = n_real; t < t_text; ++t) {
            max_abs_pad = std::max(max_abs_pad, std::fabs(static_cast<double>(txt_emb[static_cast<size_t>(c) * t_text + t])));
        }
    }
    std::fprintf(stderr, "txt_emb_max_abs_diff=%g (t_text=%u, n_real=%u), pad_tail_max_abs=%g\n",
                 max_abs_diff, t_text, n_real, max_abs_pad);
    LOOM_CHECK(max_abs_diff < 1e-2);
    LOOM_CHECK(max_abs_pad == 0.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
