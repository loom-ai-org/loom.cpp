// Validates the MIL-compiler-exported Parakeet-RNNT-0.6B encoder GGUF (export_parakeet_rnnt_mil.py)
// against the SAME reference fixture the bespoke-conversion test (test_e2e_parakeet_rnnt.cpp) uses:
// tools/convert_nemo/reference_forward_parakeet_rnnt.py's independent hand-rolled encoder_forward().
// Same shape as test_e2e_parakeet_tdt_mil_export.cpp -- see that file's own comments for why only
// "waveform"/"length" are declared inputs, and why the generic MIL-exporter's auto-generated Lua driver
// script is bypassed in favor of loading the topology directly via GraphBuilder.
//
// Not generated at ctest time (needs the real ~2.4GB checkpoint + coremltools) -- skips cleanly if the
// fixture isn't present. To (re)generate: `~/.venvs/piper/bin/python3 export_parakeet_rnnt_mil.py` from the
// repo root, and reuse the SAME ref/ dir test_e2e_parakeet_rnnt.cpp's own LOOM_PARAKEET_RNNT_DIR produces
// (tools/convert_nemo/reference_forward_parakeet_rnnt.py).

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_PARAKEET_RNNT_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "parakeet_rnnt_encoder_mil_monolithic.gguf";

    const char* dir_env = std::getenv("LOOM_PARAKEET_RNNT_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/parakeet_rnnt_model";
    const std::string ref_dir = dir + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: MIL-exported Parakeet-RNNT GGUF ('%s') or ref fixture ('%s') not found "
                      "(run export_parakeet_rnnt_mil.py and tools/convert_nemo/reference_forward_parakeet_rnnt.py)\n",
                      gguf_path.c_str(), ref_dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("main_topology"));

    constexpr uint32_t kNSamples = 16000;
    constexpr uint32_t kNSubsampled = 13;
    constexpr uint32_t kNEmbd = 1024;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    loom::GraphBuilder::BuildResult result = builder.build({{"n_samples", kNSamples}, {"n_past", 0}});

    ggml_tensor* waveform_t = result.input_tensors.at("waveform");
    ggml_tensor* length_t = result.input_tensors.at("length");

    const std::vector<float> waveform = read_f32_binary(ref_dir + "/waveform.bin");
    LOOM_CHECK(waveform.size() == kNSamples);

    ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
    const int32_t length_val = static_cast<int32_t>(kNSamples);
    ggml_backend_tensor_set(length_t, &length_val, 0, sizeof(int32_t));

    ggml_backend_graph_compute(backend.get(), result.graph);

    std::fprintf(stderr, "MIL-exported encoder output shape: [%ld, %ld]\n",
                 static_cast<long>(result.output->ne[0]), static_cast<long>(result.output->ne[1]));
    LOOM_CHECK(static_cast<uint32_t>(result.output->ne[0]) == kNEmbd);
    LOOM_CHECK(static_cast<uint32_t>(result.output->ne[1]) == kNSubsampled);

    std::vector<float> encoder_out_flat(static_cast<size_t>(kNEmbd) * kNSubsampled);
    ggml_backend_tensor_get(result.output, encoder_out_flat.data(), 0, encoder_out_flat.size() * sizeof(float));

    const std::vector<float> expected_encoder_out = read_f32_binary(ref_dir + "/expected_encoder_output.bin");
    LOOM_CHECK(expected_encoder_out.size() == encoder_out_flat.size());
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < encoder_out_flat.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(encoder_out_flat[i] - expected_encoder_out[i]));
    }
    std::fprintf(stderr, "MIL-exported encoder max abs diff vs. reference_forward_parakeet_rnnt.py = %f\n",
                 static_cast<double>(max_abs_diff));
    // Same tolerance as test_e2e_parakeet_tdt_mil_export.cpp's own -- an earlier version of this test used
    // a much looser 1.3, attributed to xscale=32.0 amplifying coremltools' own STFT fp32-precision noise.
    // That theory was wrong: the real cause was two general exporter bugs (silent FP16-rounding of every
    // constant weight, and a completely dropped conv bias -- see BACKLOG.md), not amplified STFT noise.
    // Once both were fixed this diff dropped from ~1.14 to ~1e-5.
    LOOM_CHECK(max_abs_diff <= 5e-2f);

    LOOM_TEST_REPORT_AND_RETURN();
}
