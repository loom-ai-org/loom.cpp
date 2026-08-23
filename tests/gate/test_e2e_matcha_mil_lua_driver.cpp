// Validates the MIL-traced Matcha-TTS export (export_matcha_mil.py) end-to-end: runs the real
// matcha_ljspeech.ckpt + generator_v1 checkpoints through a LoomLuaBridge executing the MIL-traced
// orchestration (tools/convert_matcha/matcha_driver/, loaded from the single combined matcha GGUF
// `loom-export` produces) and checks the result matches the hand-written loom::MatchaDriver (the
// bespoke topology's own oracle) -- mirrors test_e2e_matcha_lua_driver.cpp's own bespoke-Lua-vs-oracle
// comparison exactly, just against the MIL-traced topologies instead.
//
// That oracle is RETIRED (P4.0.8, E.3) and is no longer constructed here: its output at these exact
// inputs is frozen in fixtures/legacy_driver_reference/matcha_driver_waveform.npy, at the same 2e-2
// bound and with the same observed 1.04e-02 / rmse 6.78e-04 as the live comparison. See that
// directory's README.md for the provenance and for why it cannot be regenerated. Unlike
// test_e2e_vits_mil_lua_driver.cpp (which deliberately avoids the bespoke oracle after finding a real
// bug in ITS topology at realistic scale), Matcha's bespoke oracle has no known issues -- per-phase
// numerical checks against real-module references (test_e2e_matcha_mil_{text_encoder,decoder,vocoder}
// .cpp) already confirm each MIL topology independently, so a tight full-pipeline comparison here is the
// right level of rigor, not a false-confidence trap.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* mil_gguf_env = loom_test::fixture_env("LOOM_MATCHA_MIL_GGUF");
    if (mil_gguf_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_MIL_GGUF (the matcha GGUF produced by "
                              "`loom-export`) to run this check\n");
        return 77;
    }
    const std::string mil_gguf = mil_gguf_env;

    const std::vector<int32_t> tokens = {5, 42, 7, 88, 13, 100, 3, 61};
    constexpr uint32_t kNSteps = 10;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav =
        loom_test::read_npy_f32(std::string(LOOM_LEGACY_REF_DIR) + "/matcha_driver_waveform.npy", ref_shape);

    // --- New path: LoomLuaBridge running matcha_driver/ over the MIL-traced topologies ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(mil_gguf, backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        // One `encoder`, not `encoder_mu` + `encoder_logw`: P4.15f merged them, because each carried
        // its own copy of the TextEncoder and the driver ran both on the same tokens.
        bridge.register_module("encoder", *model, loom::GraphTopology::parse(model->topology_json("encoder")));
        bridge.register_module("decoder", *model, loom::GraphTopology::parse(model->topology_json("decoder")));
        bridge.register_module("vocoder", *model, loom::GraphTopology::parse(model->topology_json("vocoder")));
        bridge.load_script(driver_script);

        const std::vector<double> tokens_d(tokens.begin(), tokens.end());
        // No n_feats/mel_mean/mel_std: the driver carries them as ExportConstants read off the
        // checkpoint (P4.0.8's first follow-up). That is also this test's check of them -- the frozen
        // reference below was produced with the values tts_driver_inputs.h used to supply, so if the
        // export read anything else off the checkpoint the waveform moves.
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"tokens", tokens_d},
            {"n_steps", static_cast<double>(kNSteps)},
            {"seed", static_cast<double>(kSeed)},
        });
        const auto& wav_d = std::get<std::vector<double>>(result);
        lua_wav.assign(wav_d.begin(), wav_d.end());
    }

    LOOM_CHECK(!ref_wav.empty());
    LOOM_CHECK(lua_wav.size() == ref_wav.size());

    double max_abs_diff = 0.0, sum_sq_diff = 0.0;
    for (size_t i = 0; i < ref_wav.size(); ++i) {
        const double d = std::fabs(lua_wav[i] - ref_wav[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_sq_diff += d * d;
    }
    const double rmse = std::sqrt(sum_sq_diff / static_cast<double>(ref_wav.size()));
    std::fprintf(stderr, "waveform_len=%zu, max_abs_diff=%g, rmse=%g\n", ref_wav.size(), max_abs_diff, rmse);
    // Unlike test_e2e_matcha_lua_driver.cpp's own 1e-3 bound (the SAME bespoke topology executed twice,
    // via the Lua interpreter vs the C++ oracle -- genuinely the same op sequence, should match nearly
    // bit-exact), this compares TWO INDEPENDENTLY DERIVED computation graphs of the same architecture
    // (a real MIL trace of the actual PyTorch modules vs a hand-built ggml topology) -- each already
    // independently verified tight against real-module references per-phase
    // (test_e2e_matcha_mil_{text_encoder,decoder}.cpp: ~1e-4/~4e-4) before ever reaching here. A single
    // Decoder step's own ~4e-4 residual accumulates over `kNSteps=10` sequential Euler updates
    // (`z += v*dt`) directly in mel-space, then that mel-space drift gets amplified through a highly
    // nonlinear HiFi-GAN vocoder -- the same "small independent per-phase residuals compound through a
    // long nonlinear vocoder" pattern already documented for Kokoro's/StyleTTS2's own MIL-vs-bespoke
    // comparisons (BACKLOG.md). 0.02 has real margin above the observed ~0.0104 peak on a raw
    // (-1,1)-range waveform while still catching a genuinely broken (not just precision-limited) driver.
    LOOM_CHECK(max_abs_diff < 0.02);

    LOOM_TEST_REPORT_AND_RETURN();
}
