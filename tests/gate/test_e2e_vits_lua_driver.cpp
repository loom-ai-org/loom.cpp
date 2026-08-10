// Validates the procedural-generalization architecture on VITS -- the most involved of the three
// CFM/flow-style Lua ports (dynamic-length relative-position table cropping via loom.get_weight +
// loom.pad_crop_relative_embeddings, combined with Gaussian-noise-interleaved duration expansion). Runs
// the real piper VITS checkpoint through a LoomLuaBridge executing the hand-ported
// tools/convert_piper_vits/vits_driver.lua (loaded from convert_vits_lua_all.py's single-file output)
// and asserts the waveform matches the hand-written C++ driver's.
//
// That C++ driver -- loom::VitsDriver, which this test used to CONSTRUCT and run alongside the Lua one
// -- is retired (P4.0.8, E.3). Its output at these exact inputs is frozen in
// fixtures/legacy_driver_reference/vits_driver_waveform.npy instead, at the same 1e-3 bound the live
// comparison used (observed 3.3e-07: the same op sequence executed through the Lua interpreter rather
// than through C++ control flow, so it should and does match to near bit-exactness). See that
// directory's README.md for the provenance and for why the fixture cannot be regenerated.
//
// Skips cleanly if the required env vars/files aren't present.

#include "test_util.h"
#include "fixtures.h"
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
    namespace cfg = loom_test::tts_inputs::vits;
    const char* dir_lua_env = loom_test::fixture_env("LOOM_VITS_LUA_DIR");
    if (dir_lua_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_LUA_DIR (a directory with vits.gguf, produced by "
                              "convert_vits_lua_all.py) to run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;

    // Same real BOS/blank-interleaved/EOS token-id sequence as test_e2e_vits_driver.cpp -- T=62, long
    // enough to exercise the real emb_rel_k/v pad branch (T > window_size+1=5).
    const std::vector<int32_t> token_ids = {
        1, 20, 0, 59, 0, 24, 0, 120, 0, 59, 0, 100, 0, 3, 0, 35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0,
        8, 0, 3, 0, 41, 0, 74, 0, 31, 0, 3, 0, 74, 0, 38, 0, 3, 0, 50, 0, 3, 0, 32, 0, 120, 0, 61, 0, 31,
        0, 32, 0, 10, 0, 2};
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- Oracle: loom::VitsDriver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav =
        loom_test::read_npy_f32(std::string(LOOM_LEGACY_REF_DIR) + "/vits_driver_waveform.npy", ref_shape);

    // --- New path: LoomLuaBridge running the hand-ported vits_driver.lua ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_lua + "/vits.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("stats", *model, loom::GraphTopology::parse(model->topology_json("stats")));
        bridge.register_module("logw", *model, loom::GraphTopology::parse(model->topology_json("logw")));
        bridge.register_module("flow_vocoder", *model,
                                loom::GraphTopology::parse(model->topology_json("flow_vocoder")));
        bridge.load_script(driver_script);

        const std::vector<double> token_ids_d(token_ids.begin(), token_ids.end());
        const uint32_t k_channels = cfg::hidden_channels / cfg::n_heads;
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"token_ids", token_ids_d},
            {"seed", static_cast<double>(kSeed)},
            {"n_text_layers", static_cast<double>(cfg::n_text_layers)},
            {"window_size", static_cast<double>(cfg::window_size)},
            {"k_channels", static_cast<double>(k_channels)},
            {"inter_channels", static_cast<double>(cfg::inter_channels)},
            {"noise_scale", static_cast<double>(cfg::noise_scale)},
            {"noise_scale_w", static_cast<double>(cfg::noise_scale_w)},
            {"length_scale", static_cast<double>(cfg::length_scale)},
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
