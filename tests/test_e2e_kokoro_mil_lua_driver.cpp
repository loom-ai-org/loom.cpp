// Validates the MIL-traced Kokoro export (export_kokoro_mil.py) end-to-end: runs the real Kokoro-82M
// checkpoint through a LoomLuaBridge executing the hybrid orchestration
// (tools/convert_kokoro/kokoro_driver_mil.lua, loaded from export_kokoro_mil.py's combined
// kokoro_mil.gguf), with the two MIL-traced topologies ("albert_bert_encoder", "decoder_vocoder")
// registered alongside the LSTM-bound topologies still loaded from the EXISTING bespoke kokoro.gguf
// (convert_kokoro_lua_all.py) -- see kokoro_driver_mil.lua's own module docstring for why those pieces
// stay bespoke (ggml has no native LSTM op).
//
// This does NOT compare against loom::KokoroDriver/kokoro_driver.lua's own oracle waveform the way
// test_e2e_kokoro_lua_driver.cpp does for the all-bespoke Lua port -- deliberately, same reasoning
// test_e2e_vits_mil_lua_driver.cpp already documents for VITS: the decoder_vocoder MIL topology's own
// numerical-reference test (test_e2e_kokoro_mil_decoder_vocoder_reference.cpp) found and root-caused two
// real, general bugs getting to a real precision ceiling (~2e-3 mean/~0.025 max abs, see that test's own
// comments for the full amplification reasoning) -- small enough to trust the MIL topology itself, but
// compounding through F0Ntrain/frame-expansion's real (if tiny) differences before EVER reaching
// decoder_vocoder means a tight full-pipeline match against the bespoke oracle isn't a meaningful target.
// The per-phase reference tests (test_e2e_kokoro_mil_albert_bert_encoder_reference.cpp,
// test_e2e_kokoro_mil_decoder_vocoder_reference.cpp) are where the real numerical confidence comes from;
// this test's own job is just to confirm the LUA ORCHESTRATION (mixed bespoke+MIL topology registration,
// cross-phase RNG, frame expansion) runs end-to-end without crashing and produces a sane result.

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
    const char* dir_lua_env = std::getenv("LOOM_KOKORO_LUA_DIR");
    const char* gguf_mil_env = std::getenv("LOOM_KOKORO_MIL_GGUF");
    if (dir_lua_env == nullptr || gguf_mil_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_LUA_DIR (a directory with kokoro.gguf, produced "
                              "by convert_kokoro_lua_all.py, for the LSTM-bound bespoke topologies) and "
                              "LOOM_KOKORO_MIL_GGUF (kokoro_mil.gguf, produced by export_kokoro_mil.py) "
                              "to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;
    const std::string gguf_mil_path = gguf_mil_env;

    // Same fixture as test_e2e_kokoro_lua_driver.cpp/test_e2e_kokoro_driver.cpp's own oracle test.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9, 0};
    std::vector<float> ref_s(256);
    for (size_t i = 0; i < ref_s.size(); ++i) ref_s[i] = 0.05f * std::sin(static_cast<float>(i) * 0.37f);
    constexpr float kSpeed = 1.0f;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::KokoroConfig cfg; // real defaults: style_dim=128, d_model=512, hidden_per_dir=256, etc.
    std::vector<float> lua_wav;
    {
        auto model_lua = loom::GgufModel::load(dir_lua + "/kokoro.gguf", backend.get());
        LOOM_CHECK(model_lua != nullptr);
        auto model_mil = loom::GgufModel::load(gguf_mil_path, backend.get());
        LOOM_CHECK(model_mil != nullptr);
        const std::string driver_script = model_mil->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());

        // --- MIL-traced (kokoro_mil.gguf) ---
        bridge.register_module("albert_bert_encoder", *model_mil,
                                loom::GraphTopology::parse(model_mil->topology_json("albert_bert_encoder")));
        bridge.register_module("decoder_vocoder", *model_mil,
                                loom::GraphTopology::parse(model_mil->topology_json("decoder_vocoder")));

        // --- Bespoke/LSTM-bound (kokoro.gguf) ---
        const char* bespoke_topo_names[] = {
            "text_encoder_cnn",
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
        const std::vector<double> ref_s_d(ref_s.begin(), ref_s.end());
        loom::LoomLuaBridge::Value result = bridge.call("synthesize", {
            {"input_ids", input_ids_d},
            {"ref_s", ref_s_d},
            {"speed", static_cast<double>(kSpeed)},
            {"seed", static_cast<double>(kSeed)},
            {"style_dim", static_cast<double>(cfg.style_dim)},
            {"d_model", static_cast<double>(cfg.d_model)},
            {"hidden_per_dir", static_cast<double>(cfg.hidden_per_dir)},
            {"harmonic_num", static_cast<double>(cfg.harmonic_num)},
            {"upsample_scale", static_cast<double>(cfg.upsample_scale)},
            {"gen_istft_n_fft", static_cast<double>(cfg.gen_istft_n_fft)},
            {"gen_istft_hop", static_cast<double>(cfg.gen_istft_hop)},
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
    // This fixture's `ref_s` is a synthetic sine-wave-based style vector (see above), not a real learned
    // speaker embedding -- genuinely out-of-training-distribution, and real Kokoro's HiFi-GAN-style
    // vocoder responds to it with a real, brief (~40-sample) resonance burst well above typical speech
    // loudness. Confirmed NOT a MIL-export bug: a standalone probe running loom::KokoroDriver (the
    // existing, independently-verified bespoke C++ oracle) on this EXACT fixture produces the same burst
    // at nearly the same sample (rms=1.09, max_abs=21.7 at sample 11656) that this Lua/MIL path produces
    // (rms~0.88, max_abs~16.7 at sample 11651) -- two fully independent implementations agreeing closely
    // enough to confirm this is the real model's own behavior, not an implementation bug in either one.
    // Bounds below are set from that oracle's own observed range, with real margin, not an assumed "quiet
    // speech" range -- still generous enough to catch a genuinely broken (silent, NaN/Inf, or wildly
    // exploding beyond the oracle's own scale) synthesis.
    LOOM_CHECK(rms > 1e-4 && rms < 3.0);
    LOOM_CHECK(max_abs < 30.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
