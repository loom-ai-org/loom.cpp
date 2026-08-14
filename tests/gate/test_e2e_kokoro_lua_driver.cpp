// Validates the procedural-generalization architecture on Kokoro-82M -- the largest driver ported so
// far: 43 topologies (6 BiLSTM instances, each host-stepped as a Lua for-loop over loom.run_subgraph,
// matching loom::BiLstmStepper's own per-timestep mechanics exactly) plus the new loom.uniform_array
// binding (SineGen's rand_ini draws, shared with StyleTTS2Driver's own identical need). Runs the SAME
// real Kokoro-82M checkpoint through TWO independent paths -- the existing hand-written
// loom::KokoroDriver -- which is RETIRED (P4.0.8, E.3) -- and a LoomLuaBridge running the hand-ported
// tools/convert_kokoro/kokoro_driver.lua (loaded from the single-file convert_kokoro_lua_all.py
// output). The C++ half is no longer run here: its output at these exact inputs is frozen in
// fixtures/legacy_driver_reference/kokoro_driver_waveform.npy, at the same 1e-3 bound the live
// comparison used (observed 1.9e-06). See that directory's README.md for the provenance and for why the
// fixture cannot be regenerated. Skips cleanly if the required env vars/files aren't present.

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
    namespace cfg = loom_test::tts_inputs::kokoro;
    const char* dir_lua_env = loom_test::fixture_env("LOOM_KOKORO_LUA_DIR");
    if (dir_lua_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_LUA_DIR (a directory with kokoro.gguf, produced "
                              "by convert_kokoro_lua_all.py) to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;

    // Same fixture as test_e2e_kokoro_driver.cpp's own oracle test.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9, 0};
    std::vector<float> ref_s(256);
    for (size_t i = 0; i < ref_s.size(); ++i) ref_s[i] = 0.05f * std::sin(static_cast<float>(i) * 0.37f);
    constexpr float kSpeed = 1.0f;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav =
        loom_test::read_npy_f32(std::string(LOOM_LEGACY_REF_DIR) + "/kokoro_driver_waveform.npy", ref_shape);

    // --- New path: LoomLuaBridge running the hand-ported kokoro_driver.lua ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_lua + "/kokoro.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        const char* topo_names[] = {
            "albert", "bert_encoder", "text_encoder_cnn",
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
        const std::vector<double> ref_s_d(ref_s.begin(), ref_s.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"input_ids", input_ids_d},
            {"ref_s", ref_s_d},
            {"speed", static_cast<double>(kSpeed)},
            {"seed", static_cast<double>(kSeed)},
            {"style_dim", static_cast<double>(cfg::style_dim)},
            {"d_model", static_cast<double>(cfg::d_model)},
            {"hidden_per_dir", static_cast<double>(cfg::hidden_per_dir)},
            {"harmonic_num", static_cast<double>(cfg::harmonic_num)},
            {"upsample_scale", static_cast<double>(cfg::upsample_scale)},
            {"gen_istft_n_fft", static_cast<double>(cfg::gen_istft_n_fft)},
            {"gen_istft_hop", static_cast<double>(cfg::gen_istft_hop)},
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
