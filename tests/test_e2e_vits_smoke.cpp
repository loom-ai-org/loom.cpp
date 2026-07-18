// Structural smoke test for the real VITS conversion (tools/convert_piper_vits/convert_vits.py): loads
// the two GGUF files it produces and builds every declared topology with dummy (zero-filled) inputs,
// checking only that the graph builds/computes without error and that the output is finite -- NOT yet a
// numerical-correctness check against a hand-rolled reference (that's the fuller e2e test, still to
// come). This exists to catch structural bugs (unresolved weight names, primitive attr/shape mismatches)
// early, before the two-phase host driver is built on top of it. Skips cleanly (SKIP_RETURN_CODE 77) if
// LOOM_VITS_DIR isn't set, same convention as every other real-checkpoint-dependent test.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

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
    const char* dir_env = std::getenv("LOOM_VITS_DIR");
    if (dir_env == nullptr) {
        std::fprintf(stderr, "skipping: real VITS GGUF fixture not found (set LOOM_VITS_DIR to a directory "
                              "containing vits_stats.gguf/vits_logw.gguf/vits_flow_vocoder.gguf, produced "
                              "by tools/convert_piper_vits/convert_vits.py)\n");
        return 77;
    }
    const std::string dir = dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    constexpr uint32_t kT = 5; // small dummy token count, purely structural

    // --- Phase 1: TextEncoder (stats) + TextEncoder+SDP (logw) -- two independent GGUF files, each
    //     with its own (partially redundant) TextEncoder weight copy, since GraphTopology supports
    //     only one declared output per topology and GgufModel::load requires exactly one
    //     "model.graph_topology" KV per file. ---
    auto stats_model = loom::GgufModel::load(dir + "/vits_stats.gguf", backend.get());
    LOOM_CHECK(stats_model != nullptr);
    auto logw_model = loom::GgufModel::load(dir + "/vits_logw.gguf", backend.get());
    LOOM_CHECK(logw_model != nullptr);

    loom::GraphTopology stats_topo = loom::GraphTopology::parse(stats_model->topology_json());
    loom::GraphTopology logw_topo = loom::GraphTopology::parse(logw_model->topology_json());

    auto fill_common_inputs = [&](loom::GraphBuilder::BuildResult& r) {
        std::vector<int32_t> tokens(kT, 1);
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, tokens.size() * sizeof(int32_t));
        fill_zero(r.input_tensors.at("attn_mask"));
        for (int i = 0; i < 6; ++i) {
            fill_zero(r.input_tensors.at("emb_rel_k_" + std::to_string(i)));
            fill_zero(r.input_tensors.at("emb_rel_v_" + std::to_string(i)));
        }
    };

    {
        loom::GraphBuilder builder(stats_topo, *stats_model, backend.get());
        loom::GraphBuilder::BuildResult r = builder.build(kT, 0);
        fill_common_inputs(r);
        LOOM_CHECK(r.output->ne[0] == 2 * 192); // stats: [2*out_channels, T]
        LOOM_CHECK(r.output->ne[1] == static_cast<int64_t>(kT));
        ggml_backend_graph_compute(backend.get(), r.graph);
        auto out = read_tensor(r.output);
        LOOM_CHECK(all_finite(out));
        std::fprintf(stderr, "stats topology: built and computed OK, %zu elements\n", out.size());
    }

    std::vector<float> logw_result;
    {
        loom::GraphBuilder builder(logw_topo, *logw_model, backend.get());
        loom::GraphBuilder::BuildResult r = builder.build(kT, 0);
        fill_common_inputs(r);
        fill_zero(r.input_tensors.at("z_noise"));
        LOOM_CHECK(r.output->ne[0] == static_cast<int64_t>(kT));
        ggml_backend_graph_compute(backend.get(), r.graph);
        logw_result = read_tensor(r.output);
        LOOM_CHECK(all_finite(logw_result));
        std::fprintf(stderr, "logw topology: built and computed OK, %zu elements\n", logw_result.size());
    }

    // --- Phase 2: flow + vocoder ---
    auto flow_vocoder = loom::GgufModel::load(dir + "/vits_flow_vocoder.gguf", backend.get());
    LOOM_CHECK(flow_vocoder != nullptr);
    loom::GraphTopology fv_topo = loom::GraphTopology::parse(flow_vocoder->topology_json());

    {
        loom::GraphBuilder builder(fv_topo, *flow_vocoder, backend.get());
        loom::GraphBuilder::BuildResult r = builder.build(kT, 0);
        fill_zero(r.input_tensors.at("z_p"));
        const int64_t expected_wav_len = static_cast<int64_t>(kT) * 8 * 8 * 4; // upsample_rates product
        LOOM_CHECK(r.output->ne[0] == expected_wav_len);
        ggml_backend_graph_compute(backend.get(), r.graph);
        auto wav = read_tensor(r.output);
        LOOM_CHECK(all_finite(wav));
        std::fprintf(stderr, "flow+vocoder topology: built and computed OK, %zu waveform samples\n", wav.size());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
