// Validates the MIL-traced StyleTTS2 export (export_styletts2_mil.py) end-to-end: runs the real
// yl4579/StyleTTS2-LJSpeech checkpoint through a LoomLuaBridge executing the hybrid orchestration
// (tools/convert_styletts2/styletts2_driver/, loaded from export_styletts2_mil.py's combined
// styletts2_mil.gguf), with the three MIL-traced topologies ("albert", "decoder_vocoder", "diffusion")
// registered alongside the LSTM-bound topologies still loaded from the EXISTING bespoke styletts2.gguf
// (convert_styletts2_lua_all.py) -- see styletts2_driver/'s own module docstring for why those
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
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* gguf_mil_env = loom_test::fixture_env("LOOM_STYLETTS2_MIL_GGUF");
    if (gguf_mil_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_MIL_GGUF (styletts2_mil.gguf) to run this "
                              "check\n");
        return 77;
    }
    const std::string gguf_mil_path = gguf_mil_env;

    // Same fixture as test_e2e_styletts2_lua_driver.cpp/test_e2e_styletts2_driver.cpp's own oracle test
    // (a single-leading-0-wrapped phoneme sequence -- real StyleTTS2's own convention, NOT Kokoro's
    // leading+trailing one, see styletts2_driver.h's own docstring).
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9};
    constexpr uint32_t kDiffusionSteps = 5;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    std::vector<float> lua_wav;
    {
        auto model_mil = loom::GgufModel::load(gguf_mil_path, backend.get());
        LOOM_CHECK(model_mil != nullptr);
        const std::string driver_script = model_mil->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());

        // Every topology the driver calls, all of them from styletts2_mil.gguf. Until P4.0.7 this
        // registered a mix: three MIL-traced phases plus 38 more loaded from a SECOND GgufModel over
        // the pre-MIL styletts2.gguf, because the MIL export was partial. It no longer is -- the six
        // BiLSTMs are RecurrentPhases and the rest are ordinary traced phases -- so the artifact under
        // test is one self-contained file.
        //
        // Numerical equivalence with the topologies these replaced is
        // test_e2e_kokoro_mil_topology_equivalence.cpp's job (it covers both families), not this
        // test's: this one has no oracle waveform, by design -- see the header.
        for (const std::string& name : model_mil->topology_names()) {
            bridge.register_module(name, *model_mil,
                                    loom::GraphTopology::parse(model_mil->topology_json(name)));
        }
        // Six BiLSTMs, one cell topology per direction. It was 41 until the cell topology gained its
        // second declared output: each BiLSTM was four topologies (`_h_fwd`/`_c_fwd`/`_h_bwd`/`_c_bwd`)
        // whose node lists were identical, so every timestep evaluated the gate stack twice to read
        // each half of the same step (recurrent.py::_lstm_cell_topology).
        LOOM_CHECK(model_mil->topology_names().size() == 29);

        bridge.load_script(driver_script);

        const std::vector<double> input_ids_d(input_ids.begin(), input_ids.end());
        // None of the eleven model constants: the driver carries them as ExportConstants now
        // (P4.0.8's first follow-up). `diffusion_steps` stays, because it is the knob the real
        // StyleTTS2 repo's own inference entry point exposes.
        //
        // One consequence worth stating: the waveform is no longer bit-comparable with what this test
        // produced before. `sigma_data` used to arrive as a `float` (0.45731624995853165f), and the
        // driver now carries the config.yml value as a double, so the KDiffusion preconditioning
        // differs in the last bits and the ADPM2 loop compounds that. Same as the VITS case; the
        // bounds below are what this test always checked.
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"input_ids", input_ids_d},
            {"diffusion_steps", static_cast<double>(kDiffusionSteps)},
            {"seed", static_cast<double>(kSeed)},
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
