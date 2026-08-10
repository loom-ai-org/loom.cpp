// Validates the procedural-generalization architecture on SupertonicTTS's deterministic Euler-ODE (CFM)
// sampling loop -- the second control-flow shape this architecture needs to prove out, after Whisper's
// autoregressive/KV-cache case. Runs the real SupertonicTTS checkpoint through a LoomLuaBridge
// executing the hand-ported tools/convert_supertonic/supertonic_driver.lua (loaded from
// convert_supertonic_lua_all.py's single-file output) and asserts the waveform matches the hand-written
// C++ driver's.
//
// That C++ driver -- loom::SupertonicDriver, which this test used to CONSTRUCT and run alongside the
// Lua one -- is retired (P4.0.8, E.3). Its output at these exact inputs is frozen in
// fixtures/legacy_driver_reference/supertonic_driver_waveform_<style>.npy instead, at the same 1e-3
// bound the live comparison used (observed 7.5e-07). The fixture is per-voice-style because the style
// vectors are an input; see that directory's README.md for which styles are covered, for the
// provenance, and for why the fixture cannot be regenerated.
//
// Skips cleanly if the required env vars/files aren't present.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"
#include "tts_driver_inputs.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

void flatten_recursive(const nlohmann::json& node, std::vector<float>& out) {
    if (node.is_array()) {
        for (const auto& child : node) flatten_recursive(child, out);
    } else {
        out.push_back(node.get<float>());
    }
}

std::vector<float> load_style_field(const nlohmann::json& j, const char* field) {
    std::vector<float> out;
    flatten_recursive(j.at(field).at("data"), out);
    return out;
}

} // namespace

int main() {
    namespace cfg = loom_test::tts_inputs::supertonic;
    const char* dir_lua_env = loom_test::fixture_env("LOOM_SUPERTONIC_LUA_DIR");
    const char* style_json_env = loom_test::fixture_env("LOOM_SUPERTONIC_VOICE_STYLE_JSON");
    if (dir_lua_env == nullptr || style_json_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_LUA_DIR (a directory with supertonic.gguf, "
                              "produced by convert_supertonic_lua_all.py) and "
                              "LOOM_SUPERTONIC_VOICE_STYLE_JSON (a real assets/voice_styles/*.json) to "
                              "run this check\n");
        return 77;
    }
    const std::string dir_lua = dir_lua_env;

    std::ifstream f(style_json_env);
    LOOM_CHECK(static_cast<bool>(f));
    nlohmann::json style_json;
    f >> style_json;
    const std::vector<float> style_ttl = load_style_field(style_json, "style_ttl");
    const std::vector<float> style_dp = load_style_field(style_json, "style_dp");
    LOOM_CHECK(style_ttl.size() == 50 * 256);
    LOOM_CHECK(style_dp.size() == 8 * 16);

    const std::vector<int32_t> txt_ids = {12, 45, 67, 23, 89, 34, 56, 78, 90, 15};
    constexpr uint32_t kNSteps = 10;
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // The fixture below is per-style; derive the style's name from the JSON path (".../F1.json" -> "F1").
    std::string style_name = style_json_env;
    if (const size_t slash = style_name.find_last_of('/'); slash != std::string::npos) {
        style_name = style_name.substr(slash + 1);
    }
    if (const size_t dot = style_name.find_last_of('.'); dot != std::string::npos) {
        style_name = style_name.substr(0, dot);
    }

    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    // Keyed by voice style, because the style vectors are an INPUT here and a different one produces a
    // different waveform: a fixture exists for the style it was frozen with, and pointing
    // LOOM_SUPERTONIC_VOICE_STYLE_JSON at another one skips rather than silently comparing against the
    // wrong reference. See fixtures/legacy_driver_reference/README.md for which styles are covered.
    const std::string ref_path = std::string(LOOM_LEGACY_REF_DIR) + "/supertonic_driver_waveform_" +
                                  style_name + ".npy";
    if (!std::ifstream(ref_path)) {
        std::fprintf(stderr, "skipping: no frozen reference waveform for voice style '%s' (%s). The "
                              "retired C++ oracle that produced these cannot be re-run; only the styles "
                              "already in fixtures/legacy_driver_reference/ can be checked.\n",
                      style_name.c_str(), ref_path.c_str());
        return 77;
    }
    std::vector<int64_t> ref_shape;
    const std::vector<float> ref_wav = loom_test::read_npy_f32(ref_path, ref_shape);

    // --- New path: LoomLuaBridge running the hand-ported supertonic_driver.lua ---
    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_lua + "/supertonic.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("dp", *model, loom::GraphTopology::parse(model->topology_json("dp")));
        bridge.register_module("ttl_text", *model, loom::GraphTopology::parse(model->topology_json("ttl_text")));
        bridge.register_module("vfe", *model, loom::GraphTopology::parse(model->topology_json("vfe")));
        bridge.register_module("decoder", *model, loom::GraphTopology::parse(model->topology_json("decoder")));
        bridge.load_script(driver_script);

        const std::vector<double> txt_ids_d(txt_ids.begin(), txt_ids.end());
        const std::vector<double> style_ttl_d(style_ttl.begin(), style_ttl.end());
        const std::vector<double> style_dp_d(style_dp.begin(), style_dp.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"txt_ids", txt_ids_d},
            {"style_ttl", style_ttl_d},
            {"style_dp", style_dp_d},
            {"n_steps", static_cast<double>(kNSteps)},
            {"seed", static_cast<double>(kSeed)},
            {"t_text", 10.0},
            {"txt_dim", 256.0},
            {"lat_dim", 144.0},
            {"sample_rate", 44100.0},
            {"base_chunk_size", 512.0},
            {"compression_factor", 6.0},
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
