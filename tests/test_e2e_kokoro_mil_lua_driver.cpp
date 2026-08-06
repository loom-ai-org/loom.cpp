// Validates the MIL-traced Kokoro export end-to-end: runs the real Kokoro-82M checkpoint through a
// LoomLuaBridge executing the orchestration in tools/convert_kokoro/kokoro_driver/, loaded from the
// combined kokoro_mil.gguf -- with EVERY topology it calls coming from that one file.
//
// It used to be a hybrid: two MIL-traced topologies plus 37 LSTM-bound ones from a second GgufModel
// over the pre-MIL kokoro.gguf, because ggml has no LSTM op and those pieces had never been traced.
// P4.0.7 closed that -- the six BiLSTMs export as RecurrentPhases (per-timestep cell topologies plus a
// host-side loop) and the remaining hand-built pieces as ordinary traced phases -- so this test now
// loads exactly one file, which is what the one-GGUF-per-model convention always meant.
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
// this test's own job is just to confirm the LUA ORCHESTRATION (topology registration, cross-phase RNG,
// frame expansion) runs end-to-end without crashing and produces a sane result. Numerical equivalence
// with the bespoke topologies these replaced is test_e2e_kokoro_mil_topology_equivalence.cpp's job.

#include "test_util.h"
#include "tts_driver_inputs.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    namespace cfg = loom_test::tts_inputs::kokoro;
    const char* gguf_mil_env = std::getenv("LOOM_KOKORO_MIL_GGUF");
    if (gguf_mil_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_MIL_GGUF (kokoro_mil.gguf) to run this check\n");
        return 77;
    }
    const std::string gguf_mil_path = gguf_mil_env;

    // Same fixture as test_e2e_kokoro_lua_driver.cpp/test_e2e_kokoro_driver.cpp's own oracle test.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9, 0};
    std::vector<float> ref_s(256);
    for (size_t i = 0; i < ref_s.size(); ++i) ref_s[i] = 0.05f * std::sin(static_cast<float>(i) * 0.37f);
    constexpr float kSpeed = 1.0f;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    std::vector<float> lua_wav;
    {
        auto model_mil = loom::GgufModel::load(gguf_mil_path, backend.get());
        LOOM_CHECK(model_mil != nullptr);
        const std::string driver_script = model_mil->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());

        // Every topology the driver calls, all of them from kokoro_mil.gguf. Until P4.0.7 this
        // registered a mix: the two MIL-traced phases plus 37 LSTM-bound ones loaded from a SECOND
        // GgufModel over the pre-MIL kokoro.gguf, because the MIL export was partial. It no longer is
        // -- the six BiLSTMs are RecurrentPhases and the rest are ordinary traced phases -- so the
        // artifact under test is one self-contained file, which is what the one-GGUF-per-model
        // convention always meant.
        //
        // Numerical equivalence with the topologies these replaced is NOT this test's job and never
        // was (it has no oracle waveform -- see the header). It is
        // test_e2e_kokoro_mil_topology_equivalence.cpp's, which feeds identical inputs to both files'
        // versions of every shared topology.
        for (const std::string& name : model_mil->topology_names()) {
            bridge.register_module(name, *model_mil,
                                    loom::GraphTopology::parse(model_mil->topology_json(name)));
        }
        // Six BiLSTMs, one cell topology per direction. It was 39 until the cell topology gained its
        // second declared output: each BiLSTM was four topologies (`_h_fwd`/`_c_fwd`/`_h_bwd`/`_c_bwd`)
        // whose node lists were identical, so every timestep evaluated the gate stack twice to read
        // each half of the same step (recurrent.py::_lstm_cell_topology).
        LOOM_CHECK(model_mil->topology_names().size() == 27);

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
