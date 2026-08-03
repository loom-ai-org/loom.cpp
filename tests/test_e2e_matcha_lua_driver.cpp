// Validates the procedural-generalization architecture on Matcha-TTS's deterministic Euler-ODE (CFM)
// sampling loop combined with real PER-TOKEN duration expansion (loom.expand_by_duration) -- a second,
// independent proof of the Euler-loop control-flow shape already validated for SupertonicTTS, plus the
// new expand_by_duration binding. Runs the SAME real Matcha-TTS checkpoint through TWO independent
// paths -- the existing hand-written loom::MatchaDriver (C++ control flow, loaded from the OLD
// four-separate-GGUF-file convention) and a LoomLuaBridge running the hand-ported
// tools/convert_matcha/matcha_driver.lua (loaded from the NEW single-file
// convert_matcha_lua_all.py output) -- and asserts they produce numerically matching waveforms. Skips
// cleanly if the required env vars/files aren't present.

#include "test_util.h"

#include "loom/loom.h"
#include "loom/loom_legacy.h" // the pre-MIL C++ driver this test uses as its oracle

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* dir_all_env = std::getenv("LOOM_MATCHA_DIR");
    const char* dir_lua_env = std::getenv("LOOM_MATCHA_LUA_DIR");
    if (dir_all_env == nullptr || dir_lua_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_DIR (matcha_encoder_{mu,logw}.gguf/"
                              "matcha_decoder.gguf/matcha_vocoder.gguf, for the C++ oracle) and "
                              "LOOM_MATCHA_LUA_DIR (a directory with matcha.gguf, produced by "
                              "convert_matcha_lua_all.py) to run this check\n");
        return 77;
    }
    const std::string dir_all = dir_all_env;
    const std::string dir_lua = dir_lua_env;

    const std::vector<int32_t> tokens = {5, 42, 7, 88, 13, 100, 3, 61};
    constexpr uint32_t kNSteps = 10;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: the existing hand-written C++ driver ---
    std::vector<float> ref_wav;
    loom::MatchaConfig cfg;
    {
        loom::MatchaDriver driver(dir_all, cfg, backend.get());
        ref_wav = driver.synthesize(tokens, kNSteps, kSeed);
    }

    // --- New path: LoomLuaBridge running the hand-ported matcha_driver.lua ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_lua + "/matcha.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder_mu", *model, loom::GraphTopology::parse(model->topology_json("encoder_mu")));
        bridge.register_module("encoder_logw", *model,
                                loom::GraphTopology::parse(model->topology_json("encoder_logw")));
        bridge.register_module("decoder", *model, loom::GraphTopology::parse(model->topology_json("decoder")));
        bridge.register_module("vocoder", *model, loom::GraphTopology::parse(model->topology_json("vocoder")));
        bridge.load_script(driver_script);

        const std::vector<double> tokens_d(tokens.begin(), tokens.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"tokens", tokens_d},
            {"n_steps", static_cast<double>(kNSteps)},
            {"seed", static_cast<double>(kSeed)},
            {"n_feats", static_cast<double>(cfg.n_feats)},
            {"mel_mean", static_cast<double>(cfg.mel_mean)},
            {"mel_std", static_cast<double>(cfg.mel_std)},
        });
        const auto& wav_d = std::get<std::vector<double>>(result);
        lua_wav.assign(wav_d.begin(), wav_d.end());
    }

    LOOM_CHECK(!ref_wav.empty());
    LOOM_CHECK(lua_wav.size() == ref_wav.size());

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < ref_wav.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(lua_wav[i] - ref_wav[i])));
    }
    std::fprintf(stderr, "waveform_len=%zu, max_abs_diff=%g\n", ref_wav.size(), max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
