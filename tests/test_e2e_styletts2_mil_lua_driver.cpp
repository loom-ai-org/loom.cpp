// Validates the MIL-traced StyleTTS2 export (export_styletts2_mil.py) end-to-end: runs the real
// yl4579/StyleTTS2-LJSpeech checkpoint through a LoomLuaBridge executing the hybrid orchestration
// (tools/convert_styletts2/styletts2_driver_mil.lua, loaded from export_styletts2_mil.py's combined
// styletts2_mil.gguf), with the three MIL-traced topologies ("albert", "decoder_vocoder", "diffusion")
// registered alongside the LSTM-bound topologies still loaded from the EXISTING bespoke styletts2.gguf
// (convert_styletts2_lua_all.py) -- see styletts2_driver_mil.lua's own module docstring for why those
// pieces stay bespoke (ggml has no native LSTM op).
//
// This does NOT compare against loom::StyleTTS2Driver/styletts2_driver.lua's own oracle waveform the way
// test_e2e_styletts2_lua_driver.cpp does for the all-bespoke Lua port -- deliberately, same reasoning
// test_e2e_kokoro_mil_lua_driver.cpp already documents for Kokoro's own analogous MIL export: the
// decoder_vocoder MIL topology's own numerical-reference test
// (test_e2e_styletts2_mil_decoder_vocoder_reference.cpp) already found and root-caused real, general bugs
// getting to a real precision ceiling (~2e-3 mean/~0.03 max abs, a HiFi-GAN-vocoder amplification ceiling,
// not further fixable), and the diffusion MIL topology carries its OWN small residual too
// (test_e2e_styletts2_mil_diffusion_reference.cpp, ~5e-7) -- both small enough to trust each topology on
// its own, but compounding through this driver's other bespoke stages before ever reaching decoder_vocoder
// means a tight full-pipeline match against the bespoke C++ oracle isn't a meaningful target (StyleTTS2's
// own existing bespoke-vs-bespoke oracle test, test_e2e_styletts2_lua_driver.cpp, already needed a looser
// 5e-3 tolerance for exactly this "style vector -> HiFi-GAN vocoder" sensitivity reason, and this path adds
// two more small, independent sources of residual on top of that one). The per-phase reference tests
// (test_e2e_styletts2_mil_albert_reference.cpp, test_e2e_styletts2_mil_decoder_vocoder_reference.cpp,
// test_e2e_styletts2_mil_diffusion_reference.cpp) are where the real numerical confidence comes from; this
// test's own job is just to confirm the LUA ORCHESTRATION (mixed bespoke+MIL topology registration,
// cross-phase RNG feeding both the diffusion sampler and SineGen, frame expansion) runs end-to-end without
// crashing and produces a sane result.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* dir_lua_env = std::getenv("LOOM_STYLETTS2_LUA_DIR");
    const char* gguf_mil_env = std::getenv("LOOM_STYLETTS2_MIL_GGUF");
    if (dir_lua_env == nullptr || gguf_mil_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_LUA_DIR (a directory with styletts2.gguf, "
                              "produced by convert_styletts2_lua_all.py, for the LSTM-bound bespoke "
                              "topologies) and LOOM_STYLETTS2_MIL_GGUF (styletts2_mil.gguf, produced by "
                              "export_styletts2_mil.py) to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;
    const std::string gguf_mil_path = gguf_mil_env;

    // Same fixture as test_e2e_styletts2_lua_driver.cpp/test_e2e_styletts2_driver.cpp's own oracle test
    // (a single-leading-0-wrapped phoneme sequence -- real StyleTTS2's own convention, NOT Kokoro's
    // leading+trailing one, see styletts2_driver.h's own docstring).
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9};
    constexpr uint32_t kDiffusionSteps = 5;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::StyleTTS2Config cfg; // real defaults: style_dim=128, d_model=512, hidden_per_dir=256, etc.
    std::vector<float> lua_wav;
    {
        auto model_lua = loom::GgufModel::load(dir_lua + "/styletts2.gguf", backend.get());
        LOOM_CHECK(model_lua != nullptr);
        auto model_mil = loom::GgufModel::load(gguf_mil_path, backend.get());
        LOOM_CHECK(model_mil != nullptr);
        const std::string driver_script = model_mil->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());

        // --- MIL-traced (styletts2_mil.gguf) ---
        bridge.register_module("albert", *model_mil, loom::GraphTopology::parse(model_mil->topology_json("albert")));
        bridge.register_module("decoder_vocoder", *model_mil,
                                loom::GraphTopology::parse(model_mil->topology_json("decoder_vocoder")));
        bridge.register_module("diffusion", *model_mil,
                                loom::GraphTopology::parse(model_mil->topology_json("diffusion")));

        // --- Bespoke/LSTM-bound (styletts2.gguf, from convert_styletts2_lua_all.py -- only the subset
        //     NOT superseded by the MIL topologies above is registered). ---
        const char* bespoke_topo_names[] = {
            "bert_encoder", "text_encoder_cnn",
            "text_encoder_lstm_h_fwd", "text_encoder_lstm_c_fwd", "text_encoder_lstm_h_bwd", "text_encoder_lstm_c_bwd",
            "duration_lstm_0_h_fwd", "duration_lstm_0_c_fwd", "duration_lstm_0_h_bwd", "duration_lstm_0_c_bwd",
            "duration_lstm_1_h_fwd", "duration_lstm_1_c_fwd", "duration_lstm_1_h_bwd", "duration_lstm_1_c_bwd",
            "duration_lstm_2_h_fwd", "duration_lstm_2_c_fwd", "duration_lstm_2_h_bwd", "duration_lstm_2_c_bwd",
            "duration_adaln_0", "duration_adaln_1", "duration_adaln_2",
            "top_lstm_h_fwd", "top_lstm_c_fwd", "top_lstm_h_bwd", "top_lstm_c_bwd",
            "duration_proj",
            "f0n_shared_lstm_h_fwd", "f0n_shared_lstm_c_fwd", "f0n_shared_lstm_h_bwd", "f0n_shared_lstm_c_bwd",
            "f0n_f0_block0", "f0n_f0_block1", "f0n_f0_block2",
            "f0n_n_block0", "f0n_n_block1", "f0n_n_block2",
            "f0n_f0_proj", "f0n_n_proj",
        };
        for (const char* name : bespoke_topo_names) {
            bridge.register_module(name, *model_lua, loom::GraphTopology::parse(model_lua->topology_json(name)));
        }

        bridge.load_script(driver_script);

        const std::vector<double> input_ids_d(input_ids.begin(), input_ids.end());
        loom::LoomLuaBridge::Value result = bridge.call("synthesize", {
            {"input_ids", input_ids_d},
            {"diffusion_steps", static_cast<double>(kDiffusionSteps)},
            {"seed", static_cast<double>(kSeed)},
            {"style_dim", static_cast<double>(cfg.style_dim)},
            {"d_model", static_cast<double>(cfg.d_model)},
            {"hidden_per_dir", static_cast<double>(cfg.hidden_per_dir)},
            {"harmonic_num", static_cast<double>(cfg.harmonic_num)},
            {"upsample_scale", static_cast<double>(cfg.upsample_scale)},
            {"gen_istft_n_fft", static_cast<double>(cfg.gen_istft_n_fft)},
            {"gen_istft_hop", static_cast<double>(cfg.gen_istft_hop)},
            {"sigma_min", static_cast<double>(cfg.sigma_min)},
            {"sigma_max", static_cast<double>(cfg.sigma_max)},
            {"rho", static_cast<double>(cfg.rho)},
            {"sigma_data", static_cast<double>(cfg.sigma_data)},
        });
        const auto& wav_d = std::get<std::vector<double>>(result);
        lua_wav.assign(wav_d.begin(), wav_d.end());
    }

    LOOM_CHECK(!lua_wav.empty());

    double sum_sq = 0.0, max_abs = 0.0;
    for (float v : lua_wav) {
        LOOM_CHECK(std::isfinite(v));
        sum_sq += static_cast<double>(v) * v;
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(v)));
    }
    const double rms = std::sqrt(sum_sq / lua_wav.size());
    std::fprintf(stderr, "waveform_len=%zu, rms=%g, max_abs=%g\n", lua_wav.size(), rms, max_abs);
    // Real speech-scale bounds with generous margin (same "catch genuinely broken (silent, NaN/Inf, or
    // wildly exploding), not chase a precise scale" intent as Kokoro's own analogous test) -- this
    // fixture's own synthetic seed/input isn't guaranteed to land in the exact same operating regime as
    // the real inference demo's, and this pipeline's own HiFi-GAN-style vocoder is documented (BACKLOG.md,
    // styletts2_driver.lua's own PRECISION NOTE) to respond to any out-of-distribution style vector with
    // real, sometimes large amplitude swings -- not a bug, just this network class's own behavior.
    LOOM_CHECK(rms > 1e-4 && rms < 10.0);
    LOOM_CHECK(max_abs < 100.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
