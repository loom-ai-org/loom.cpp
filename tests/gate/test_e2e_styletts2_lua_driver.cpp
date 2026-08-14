// Validates the procedural-generalization architecture on StyleTTS2 -- the most complex driver ported
// so far (44 topologies, 6 BiLSTM instances, and the genuinely new style-diffusion sampler: ADPM2 over
// the real Transformer1d denoiser, ported to Lua as plain host arithmetic + loom.run_subgraph calls,
// with loom.gaussian_array/loom.uniform_array supplying every stochastic draw from the shared rng_
// stream in the same order the C++ driver uses). Runs the SAME real yl4579/StyleTTS2-LJSpeech checkpoint
// through TWO independent paths -- the hand-written loom::StyleTTS2Driver, which is RETIRED (P4.0.8,
// E.3), and a LoomLuaBridge running the hand-ported tools/convert_styletts2/styletts2_driver.lua
// (loaded from the single-file convert_styletts2_lua_all.py output). The C++ half is no longer run
// here: its output at these exact inputs is frozen in
// fixtures/legacy_driver_reference/styletts2_driver_waveform.npy, at the same 5e-3 bound the live
// comparison used (observed 3.8e-03 -- this family's own "style vector -> HiFi-GAN vocoder"
// sensitivity, which is why the bound was always looser than the other four). See that directory's
// README.md for the provenance and for why the fixture cannot be regenerated. Skips cleanly if the
// required env vars/files aren't present.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"
#include "tts_driver_inputs.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    namespace cfg = loom_test::tts_inputs::styletts2;
    const char* dir_lua_env = loom_test::fixture_env("LOOM_STYLETTS2_LUA_DIR");
    if (dir_lua_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_LUA_DIR (a directory with styletts2.gguf, "
                              "produced by convert_styletts2_lua_all.py) to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;

    // Same fixture as test_e2e_styletts2_driver.cpp's own oracle test.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9};
    constexpr uint32_t kDiffusionSteps = 5;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav =
        loom_test::read_npy_f32(std::string(LOOM_LEGACY_REF_DIR) + "/styletts2_driver_waveform.npy", ref_shape);

    // --- New path: LoomLuaBridge running the hand-ported styletts2_driver.lua ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_lua + "/styletts2.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        const char* topo_names[] = {
            "albert", "diffusion", "bert_encoder", "text_encoder_cnn",
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
            "decoder_core", "sinegen", "stft_forward", "generator",
        };
        for (const char* name : topo_names) {
            bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
        }
        bridge.load_script(driver_script);

        const std::vector<double> input_ids_d(input_ids.begin(), input_ids.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"input_ids", input_ids_d},
            {"diffusion_steps", static_cast<double>(kDiffusionSteps)},
            {"seed", static_cast<double>(kSeed)},
            {"style_dim", static_cast<double>(cfg::style_dim)},
            {"d_model", static_cast<double>(cfg::d_model)},
            {"hidden_per_dir", static_cast<double>(cfg::hidden_per_dir)},
            {"harmonic_num", static_cast<double>(cfg::harmonic_num)},
            {"upsample_scale", static_cast<double>(cfg::upsample_scale)},
            {"gen_istft_n_fft", static_cast<double>(cfg::gen_istft_n_fft)},
            {"gen_istft_hop", static_cast<double>(cfg::gen_istft_hop)},
            {"sigma_min", static_cast<double>(cfg::sigma_min)},
            {"sigma_max", static_cast<double>(cfg::sigma_max)},
            {"rho", static_cast<double>(cfg::rho)},
            {"sigma_data", static_cast<double>(cfg::sigma_data)},
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
    // Looser than every other ported model's ~1e-6: diagnosed in depth (see styletts2_driver.lua's own
    // "PRECISION NOTE" and BACKLOG.md's dated entry) before accepting this rather than assuming it away.
    // Every individual piece (all weights, every single-shot graph, the diffusion sampler's own 256-float
    // sample in isolation) matches to ~1e-6/1e-7 -- StyleTTS2 is simply the only ported driver whose style
    // vector comes out of an ITERATIVE process (5 ADPM2 diffusion steps) rather than a passthrough or a
    // single affine combination, and that vector then conditions ~50+ sequential layers ending in an
    // adversarially-trained (GAN-style) vocoder -- a network class documented to amplify tiny input
    // perturbations. 5e-3 leaves real margin above the ~3.1e-3 this test actually produces while still
    // being tight enough to catch a genuine regression: every logic/weight bug found while building this
    // port (e.g. the un-namespaced BiLSTM/AdaLN weight collisions) was caught by the conversion script's
    // own content-aware merge() assert or a per-graph isolation check, not by this end-to-end diff --
    // an actual wrong-topology/wrong-weight bug here would be expected to blow up shape checks or produce
    // silence/NaNs long before landing in the 1e-2 neighborhood.
    LOOM_CHECK(max_abs_diff < 5e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
