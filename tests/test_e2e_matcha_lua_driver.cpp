// Validates the procedural-generalization architecture on Matcha-TTS's deterministic Euler-ODE (CFM)
// sampling loop combined with real PER-TOKEN duration expansion (loom.expand_by_duration) -- a second,
// independent proof of the Euler-loop control-flow shape already validated for SupertonicTTS, plus the
// new expand_by_duration binding. Runs the real Matcha-TTS checkpoint through a LoomLuaBridge executing
// the hand-ported tools/convert_matcha/matcha_driver.lua (loaded from convert_matcha_lua_all.py's
// single-file output) and asserts the waveform matches the hand-written C++ driver's.
//
// That C++ driver -- loom::MatchaDriver, which this test used to CONSTRUCT and run alongside the Lua
// one -- is retired (P4.0.8, E.3). Its output at these exact inputs is frozen in
// fixtures/legacy_driver_reference/matcha_driver_waveform.npy instead, at the same 1e-3 bound the live
// comparison used (observed 1.4e-05: the same op sequence executed through the Lua interpreter rather
// than through C++ control flow). See that directory's README.md for the provenance and for why the
// fixture cannot be regenerated.
//
// Skips cleanly if the required env vars/files aren't present.

#include "test_util.h"
#include "npy_fixture.h"
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
    namespace cfg = loom_test::tts_inputs::matcha;
    const char* dir_lua_env = std::getenv("LOOM_MATCHA_LUA_DIR");
    if (dir_lua_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_LUA_DIR (a directory with matcha.gguf, produced "
                              "by convert_matcha_lua_all.py) to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;

    const std::vector<int32_t> tokens = {5, 42, 7, 88, 13, 100, 3, 61};
    constexpr uint32_t kNSteps = 10;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav =
        loom_test::read_npy_f32(std::string(LOOM_LEGACY_REF_DIR) + "/matcha_driver_waveform.npy", ref_shape);

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
            {"n_feats", static_cast<double>(cfg::n_feats)},
            {"mel_mean", static_cast<double>(cfg::mel_mean)},
            {"mel_std", static_cast<double>(cfg::mel_std)},
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
