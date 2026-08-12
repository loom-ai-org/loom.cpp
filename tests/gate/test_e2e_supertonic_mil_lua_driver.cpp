// Full end-to-end check for the MIL-traced SupertonicTTS export: runs the real orchestration
// (tools/convert_supertonic/supertonic_driver/, loaded from export_supertonic_mil.py's single
// combined supertonic GGUF `loom-export` produces) and checks the result matches the hand-written
// loom::SupertonicDriver oracle -- same "MIL lua driver vs. bespoke C++ driver" comparison as
// test_e2e_matcha_mil_lua_driver.cpp.
//
// That oracle is RETIRED (P4.0.8, E.3) and is no longer constructed here: its output at these exact
// inputs is frozen in fixtures/legacy_driver_reference/supertonic_driver_waveform_<style>.npy, at the
// same 1e-3 bound and with the same observed 2.08e-06 as the live comparison. The fixture is
// per-voice-style because the style vectors are an input; see that directory's README.md. Per-topology numerical checks against real-module references
// (test_e2e_supertonic_mil_{dp,ttl_text,vfe,decoder}.cpp: ~1e-2 or tighter) already validate each phase
// independently before ever reaching here.
//
// The ten ids below were the WHOLE text axis when this fixture was frozen. They are ten real ids in a
// padded, bucketed axis now (BACKLOG.md P4.6/P4.6a) -- so what this test asks has quietly become much
// sharper than what it was written to ask: the same waveform, out of a graph 32 positions wide with
// 22 of them padding, chosen at run time by the driver. It is the padding-is-inert claim and the
// bucket-selection claim at once, against ground truth that predates both.
//
// Skips cleanly if the required env vars/files aren't present.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

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
    const char* mil_gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* style_json_env = loom_test::fixture_env("LOOM_SUPERTONIC_VOICE_STYLE_JSON");
    if (mil_gguf_env == nullptr || style_json_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (the supertonic GGUF produced by "
                              "`loom-export`) and LOOM_SUPERTONIC_VOICE_STYLE_JSON (a real "
                              "assets/voice_styles/*.json) to run this check\n");
        return 77;
    }

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


    // --- Oracle: the retired C++ driver's own output at these exact inputs, frozen (P4.0.8, E.3) ---
    // Keyed by voice style, because the style vectors are an INPUT here and a different one produces a
    // different waveform: a fixture exists for the style it was frozen with, and pointing
    // LOOM_SUPERTONIC_VOICE_STYLE_JSON at another one skips rather than silently comparing against the
    // wrong reference. See fixtures/legacy_driver_reference/README.md for which styles are covered.
    std::string style_name = style_json_env;
    if (const size_t slash = style_name.find_last_of('/'); slash != std::string::npos) {
        style_name = style_name.substr(slash + 1);
    }
    if (const size_t dot = style_name.find_last_of('.'); dot != std::string::npos) {
        style_name = style_name.substr(0, dot);
    }
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

    // --- New path: LoomLuaBridge running the MIL-traced supertonic_driver/ ---
    std::vector<float> mil_wav;
    {
        auto model = loom::GgufModel::load(mil_gguf_env, backend.get());
        LOOM_CHECK(model != nullptr);
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        // How many txt_ids this export accepts, read from the file rather than from a C++ constant
        // (P4.0.8's first follow-up). Every text-touching topology here was traced at a FIXED text
        // length, so this is not advice -- a caller that sends MORE than this is calling a model that
        // cannot run, and the only thing that said so used to be a literal in tts_driver_inputs.h.
        //
        // It was `==` until P4.6 made the driver pad: the ten ids below are now ten real ids in a
        // `txt_len`-wide axis, and that this still reproduces a waveform frozen when the axis was
        // exactly ten wide is the whole point of the comparison at the bottom of this file.
        LOOM_CHECK(txt_ids.size() <= model->hparam_u32("txt_len"));

        // Every topology in the file, by name off the file. Naming the four by hand worked until the
        // text ones came in buckets (BACKLOG.md P4.6a) -- there are sixteen now, and which of them a
        // call runs is the driver's decision, so a host that enumerates is the only kind that works.
        loom::LoomLuaBridge bridge(backend.get());
        for (const std::string& name : model->topology_names()) {
            bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
        }
        bridge.load_script(driver_script);

        const std::vector<double> txt_ids_d(txt_ids.begin(), txt_ids.end());
        const std::vector<double> style_ttl_d(style_ttl.begin(), style_ttl.end());
        const std::vector<double> style_dp_d(style_dp.begin(), style_dp.end());
        // None of t_text/lat_dim/sample_rate/base_chunk_size/compression_factor: the driver carries
        // them as ExportConstants now (P4.0.8's first follow-up), four of them derived from the real
        // SpeechDecoder. The frozen reference below was produced with the literals that used to be
        // here, so it is what checks the export derived the same numbers.
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"txt_ids", txt_ids_d},
            {"style_ttl", style_ttl_d},
            {"style_dp", style_dp_d},
            {"n_steps", static_cast<double>(kNSteps)},
            {"seed", static_cast<double>(kSeed)},
        });
        const auto& wav_d = std::get<std::vector<double>>(result);
        mil_wav.assign(wav_d.begin(), wav_d.end());

        // --- The same call with NO style at all (BACKLOG.md P4.6b) ---
        // `style_ttl`/`style_dp` are optional now, and the export carries F1's own embeddings as the
        // default. This test's fixture IS F1, so the check writes itself: omitting the style must
        // produce bit-for-bit what passing it produced. Bit-for-bit and not "close" is the point --
        // the driver either reached the same numbers or it reached different ones, and there is no
        // arithmetic in between to blur it.
        //
        // Skipped rather than failed for a style other than F1, and for an older GGUF that carries no
        // default: neither is a regression, and asserting on either would make this test's meaning
        // depend on which fixture the runner happened to point at.
        if (style_name == "F1") {
            bool has_default = true;
            std::vector<double> defaulted;
            try {
                loom::LoomLuaBridge::Value r2 = bridge.call("infer", {
                    {"txt_ids", txt_ids_d},
                    {"n_steps", static_cast<double>(kNSteps)},
                    {"seed", static_cast<double>(kSeed)},
                });
                defaulted = std::get<std::vector<double>>(r2);
            } catch (const loom::Error& e) {
                has_default = false;
                std::fprintf(stderr, "no default style in this GGUF (%s) -- skipping the "
                                      "style-omitted comparison\n", e.what());
            }
            if (has_default) {
                LOOM_CHECK(defaulted.size() == wav_d.size());
                double style_diff = 0.0;
                for (size_t i = 0; i < wav_d.size(); ++i) {
                    style_diff = std::max(style_diff, std::fabs(defaulted[i] - wav_d[i]));
                }
                std::fprintf(stderr, "style omitted vs style passed: max_abs_diff=%g\n", style_diff);
                LOOM_CHECK(style_diff == 0.0);
            }
        }

        // --- ...and a DIFFERENT style must produce different audio ---
        // Without this the two checks above are both satisfied by a driver that ignores
        // `inputs.style_*` entirely and always uses the default: passing F1 would "match" the F1
        // fixture for the wrong reason. So load a sibling voice out of the same directory and require
        // the waveform to actually move. There is no reference for that voice and none is needed --
        // the claim is only that the style is READ, and inequality is the whole of it.
        std::string sibling = style_json_env;
        const size_t slash = sibling.find_last_of('/');
        const std::string dir = slash == std::string::npos ? std::string(".") : sibling.substr(0, slash);
        for (const char* other : {"M1", "F2"}) {
            std::ifstream of(dir + "/" + other + ".json");
            if (!of) continue;
            nlohmann::json other_json;
            of >> other_json;
            const std::vector<float> o_ttl = load_style_field(other_json, "style_ttl");
            const std::vector<float> o_dp = load_style_field(other_json, "style_dp");
            loom::LoomLuaBridge::Value r3 = bridge.call("infer", {
                {"txt_ids", txt_ids_d},
                {"style_ttl", std::vector<double>(o_ttl.begin(), o_ttl.end())},
                {"style_dp", std::vector<double>(o_dp.begin(), o_dp.end())},
                {"n_steps", static_cast<double>(kNSteps)},
                {"seed", static_cast<double>(kSeed)},
            });
            const auto& other_wav = std::get<std::vector<double>>(r3);
            double moved = 0.0;
            for (size_t i = 0; i < std::min(other_wav.size(), wav_d.size()); ++i) {
                moved = std::max(moved, std::fabs(other_wav[i] - wav_d[i]));
            }
            std::fprintf(stderr, "voice %s vs %s: %zu vs %zu samples, max_abs_diff=%g\n",
                         other, style_name.c_str(), other_wav.size(), wav_d.size(), moved);
            // A different voice changes the predicted duration too, so the lengths usually differ --
            // which is already proof the style was read. When they happen to match, the samples must
            // not: 1e-2 is far above the 5e-06 the SAME style reproduces itself to.
            LOOM_CHECK(other_wav.size() != wav_d.size() || moved > 1e-2);
            break;
        }
    }

    LOOM_CHECK(!ref_wav.empty());
    LOOM_CHECK(mil_wav.size() == ref_wav.size());

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < ref_wav.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(mil_wav[i] - ref_wav[i])));
    }
    std::fprintf(stderr, "waveform_len=%zu, max_abs_diff=%g\n", ref_wav.size(), max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
