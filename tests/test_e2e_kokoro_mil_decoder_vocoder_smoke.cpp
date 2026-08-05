// Structural smoke test for the new MIL-traced Kokoro "decoder_vocoder" phase (export_kokoro_mil.py,
// still in progress -- see BACKLOG.md): loads the GGUF it produces and builds/computes the
// "decoder_vocoder" topology with dummy (zero-filled) inputs, checking only that the graph
// builds/computes without error and the output is finite -- NOT yet a numerical-correctness check
// against a real reference (that's the fuller e2e test, still to come). Mirrors
// test_e2e_vits_smoke.cpp's own role for the equivalent VITS milestone. Skips cleanly (SKIP_RETURN_CODE
// 77) if LOOM_KOKORO_MIL_DECODER_VOCODER_GGUF isn't set.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

std::vector<float> read_tensor(ggml_tensor* t) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

void fill_zero(ggml_tensor* t) {
    std::vector<float> zeros(static_cast<size_t>(ggml_nelements(t)), 0.0f);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size() * sizeof(float));
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_KOKORO_MIL_DECODER_VOCODER_GGUF");
    if (gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_MIL_DECODER_VOCODER_GGUF to the GGUF produced by "
                              "the (in-progress) Kokoro MIL decoder_vocoder export\n");
        return 77;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("decoder_vocoder"));

    constexpr uint32_t kTFrames = 5; // small dummy frame count, purely structural

    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_enc_frames", kTFrames}, {"n_past", 0}});

    fill_zero(r.input_tensors.at("asr"));
    fill_zero(r.input_tensors.at("f0_curve"));
    fill_zero(r.input_tensors.at("n_curve"));
    fill_zero(r.input_tensors.at("s"));
    fill_zero(r.input_tensors.at("rand_ini"));
    fill_zero(r.input_tensors.at("noise_in"));
    // wsum must be strictly positive (it's a real-valued division denominator) -- zero would produce
    // Inf/NaN by construction, not a structural bug this smoke test is meant to catch.
    {
        ggml_tensor* wsum = r.input_tensors.at("wsum");
        std::vector<float> ones(static_cast<size_t>(ggml_nelements(wsum)), 1.0f);
        ggml_backend_tensor_set(wsum, ones.data(), 0, ones.size() * sizeof(float));
    }

    ggml_backend_graph_compute(backend.get(), r.graph);
    auto out = read_tensor(r.output);
    LOOM_CHECK(all_finite(out));
    std::fprintf(stderr, "decoder_vocoder topology: built and computed OK, %zu waveform samples\n", out.size());

    LOOM_TEST_REPORT_AND_RETURN();
}
