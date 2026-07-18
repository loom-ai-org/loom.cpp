// End-to-end test of loom::VitsDriver against the real VITS conversion (piper en-GB "miro" checkpoint,
// converted via tools/convert_piper_vits/convert_vits.py): loads the three GGUF files it produces,
// constructs a VitsDriver, and runs a full synthesize() call -- exercising the whole two-phase pipeline
// (TextEncoder -> StochasticDurationPredictor -> host-side generate_path -> coupling flow -> HiFi-GAN
// vocoder) together for the first time, not just each topology in isolation (test_e2e_vits_smoke.cpp
// covers that). Checks the output is finite and non-trivial (not silence, not NaN/Inf) -- NOT yet a
// numerical match against a hand-rolled full-model Python reference (a separate, still-open piece of
// work; see BACKLOG.md). Skips cleanly (SKIP_RETURN_CODE 77) if LOOM_VITS_DIR isn't set.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

    auto stats_model = loom::GgufModel::load(dir + "/vits_stats.gguf", backend.get());
    auto logw_model = loom::GgufModel::load(dir + "/vits_logw.gguf", backend.get());
    auto flow_vocoder_model = loom::GgufModel::load(dir + "/vits_flow_vocoder.gguf", backend.get());
    LOOM_CHECK(stats_model != nullptr && logw_model != nullptr && flow_vocoder_model != nullptr);

    loom::GraphTopology stats_topo = loom::GraphTopology::parse(stats_model->topology_json());
    loom::GraphTopology logw_topo = loom::GraphTopology::parse(logw_model->topology_json());
    loom::GraphTopology flow_vocoder_topo = loom::GraphTopology::parse(flow_vocoder_model->topology_json());

    loom::VitsConfig cfg; // real defaults: hidden_channels=192, n_heads=2, n_text_layers=6, window_size=4
    loom::VitsDriver driver(*stats_model, stats_topo, *logw_model, logw_topo, *flow_vocoder_model,
                             flow_vocoder_topo, cfg, backend.get());

    // The real BOS/blank-interleaved/EOS token-id sequence piper's own runtime produces for the text
    // "Hello world, this is a test." via piper_phonemize + espeak-ng (voice en-gb-x-rp) and the model's
    // own phoneme_id_map -- see BACKLOG.md for the exact phonemization/interleaving convention this
    // reproduces (`[BOS, p1, blank, p2, blank, ..., pn, blank, EOS]`). T=62, long enough to exercise the
    // real emb_rel_k/v pad branch (T > window_size+1=5), unlike shorter arbitrary-token runs.
    const std::vector<int32_t> token_ids = {
        1, 20, 0, 59, 0, 24, 0, 120, 0, 59, 0, 100, 0, 3, 0, 35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0,
        8, 0, 3, 0, 41, 0, 74, 0, 31, 0, 3, 0, 74, 0, 38, 0, 3, 0, 50, 0, 3, 0, 32, 0, 120, 0, 61, 0, 31,
        0, 32, 0, 10, 0, 2};
    std::vector<float> wav = driver.synthesize(token_ids, /*seed=*/42);

    LOOM_CHECK(!wav.empty());
    bool all_finite = true;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float x : wav) {
        if (!std::isfinite(x)) all_finite = false;
        sum_sq += static_cast<double>(x) * x;
        max_abs = std::max(max_abs, std::fabs(x));
    }
    LOOM_CHECK(all_finite);
    const double rms = std::sqrt(sum_sq / static_cast<double>(wav.size()));
    std::fprintf(stderr, "synthesize(%zu tokens) -> %zu samples, rms=%f, max_abs=%f\n", token_ids.size(),
                 wav.size(), rms, static_cast<double>(max_abs));
    // A real tanh-activated vocoder output that's all zeros (or otherwise degenerate) would indicate a
    // wiring bug even without a full numerical reference -- rms should be a real, non-tiny fraction of
    // tanh's own [-1,1] range.
    LOOM_CHECK(rms > 1e-4);
    LOOM_CHECK(max_abs <= 1.0001f); // tanh's own range, with a hair of float slack

    LOOM_TEST_REPORT_AND_RETURN();
}
