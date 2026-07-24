// Validates the MIL-compiler-exported Parakeet-TDT-0.6B-v3 encoder GGUF (export_parakeet_tdt_mil.py)
// against the SAME reference fixture the bespoke-conversion test (test_e2e_parakeet_tdt.cpp) uses:
// tools/convert_nemo/reference_forward_parakeet_tdt.py's independent hand-rolled encoder_forward().
// Unlike the bespoke conversion, this GGUF's topology traces the REAL preprocessor+ConformerEncoder
// directly (torch.jit.trace + coremltools), so pos_emb_raw/kq_mask are computed INSIDE the traced graph
// -- only "waveform"/"length" are declared inputs. Loaded directly via GraphBuilder (bypassing the
// generic MIL-exporter's auto-generated Lua driver script, which assumes a causal-LM "argmax the last
// row" convention -- wrong for an encoder's full (n_embd, n_subsampled) tensor output; see BACKLOG.md).
//
// Not generated at ctest time (needs the real ~2.5GB checkpoint + coremltools) -- skips cleanly if the
// fixture isn't present. To (re)generate: `~/.venvs/piper/bin/python3 export_parakeet_tdt_mil.py` from the
// repo root, and reuse the SAME ref/ dir test_e2e_parakeet_tdt.cpp's own LOOM_PARAKEET_TDT_DIR produces
// (tools/convert_nemo/reference_forward_parakeet_tdt.py).

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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
    const char* gguf_env = std::getenv("LOOM_PARAKEET_TDT_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "parakeet_tdt_encoder_mil_monolithic.gguf";

    const char* dir_env = std::getenv("LOOM_PARAKEET_TDT_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/parakeet_tdt_model";
    const std::string ref_dir = dir + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: MIL-exported Parakeet-TDT GGUF ('%s') or ref fixture ('%s') not found "
                      "(run export_parakeet_tdt_mil.py and tools/convert_nemo/reference_forward_parakeet_tdt.py)\n",
                      gguf_path.c_str(), ref_dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("main_topo"));

    constexpr uint32_t kNSamples = 16000;
    constexpr uint32_t kNSubsampled = 13;
    constexpr uint32_t kNEmbd = 1024;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    loom::GraphBuilder::BuildResult result = builder.build(kNSamples, /*n_past=*/0);

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
    std::fprintf(stderr, "MIL-exported encoder max abs diff vs. reference_forward_parakeet_tdt.py = %f\n",
                 static_cast<double>(max_abs_diff));
    // Looser than test_e2e_parakeet_tdt.cpp's bespoke-conversion 5e-2: THIS path's STFT comes from
    // coremltools' own `complex_stft` MIL lowering (lower_complex_dialect_ops.py's `_calculate_dft_matrix`),
    // not this project's own mel_common.py DFT-basis kernels -- it computes the DFT phase matrix
    // `cos/sin(2*pi*i*j/n_fft)` in fp32 arithmetic THROUGHOUT (cast to fp32 before the matmul building the
    // phase angles), with angles up to ~2*pi*(n_fft-1)^2/n_fft (~3200 radians for n_fft=512) -- fp32 has
    // only ~7 significant decimal digits, so cos/sin of an angle that large loses meaningfully more
    // precision than mel_common.py's own kernels (built once in float64 by numpy, rounded to fp32 only at
    // the final GGUF weight write). Isolated by diffing this export's raw preprocessor-only output
    // (export_parakeet_preprocessor_debug_mil.py) against reference_forward_parakeet_tdt.py's
    // compute_mel_features() directly: ~0.01-0.02 per-frame log-mel noise (with the last-frame-zeroing
    // convention matching exactly, confirming the masking itself -- not this precision difference -- was
    // the thing worth being strict about), which a 24-layer encoder can plausibly amplify to the ~0.09
    // observed here. An external (coremltools-side), not loom-engine-side, precision characteristic --
    // same "real, architecture/toolchain-driven precision ceiling" reasoning as StyleTTS2's own widened
    // tolerance (see BACKLOG.md).
    LOOM_CHECK(max_abs_diff <= 0.12f);

    LOOM_TEST_REPORT_AND_RETURN();
}
