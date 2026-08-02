// Validates the procedural-generalization architecture on SupertonicTTS's deterministic Euler-ODE (CFM)
// sampling loop -- the second control-flow shape this architecture needs to prove out, after Whisper's
// autoregressive/KV-cache case. Runs the SAME real SupertonicTTS checkpoint through TWO independent
// paths -- the existing hand-written loom::SupertonicDriver (C++ control flow, loaded from the OLD
// six-separate-GGUF-file convert_supertonic_all.py output) and a LoomLuaBridge running the hand-ported
// tools/convert_supertonic/supertonic_driver.lua (loaded from the NEW single-file
// convert_supertonic_lua_all.py output) -- and asserts they produce numerically matching waveforms.
// Skips cleanly if the required env vars/files aren't present.

#include "test_util.h"

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
    const char* dir_all_env = std::getenv("LOOM_SUPERTONIC_ALL_DIR");
    const char* dir_lua_env = std::getenv("LOOM_SUPERTONIC_LUA_DIR");
    const char* style_json_env = std::getenv("LOOM_SUPERTONIC_VOICE_STYLE_JSON");
    if (dir_all_env == nullptr || dir_lua_env == nullptr || style_json_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_ALL_DIR (six-file output of "
                              "convert_supertonic_all.py, for the C++ oracle), LOOM_SUPERTONIC_LUA_DIR "
                              "(a directory with supertonic.gguf, produced by "
                              "convert_supertonic_lua_all.py), and LOOM_SUPERTONIC_VOICE_STYLE_JSON (a "
                              "real assets/voice_styles/*.json) to run this check\n");
        return 77;
    }
    const std::string dir_all = dir_all_env;
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

    // --- Oracle: the existing hand-written C++ driver ---
    std::vector<float> ref_wav;
    {
        loom::SupertonicConfig cfg;
        LOOM_CHECK(txt_ids.size() == cfg.txt_len_fixed);
        loom::SupertonicDriver driver(dir_all, cfg, backend.get());
        ref_wav = driver.synthesize(txt_ids, style_ttl, style_dp, kNSteps, kSeed);
    }

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
