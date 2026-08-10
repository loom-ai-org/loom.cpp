// Validates the MIL-traced VITS export (export_vits_mil.py) end-to-end: runs the real piper VITS
// checkpoint through a LoomLuaBridge executing the MIL-traced orchestration
// (tools/convert_piper_vits/vits_driver/, loaded from export_vits_mil.py's single combined
// vits_mil.gguf) and checks the result is a well-formed, plausible waveform.
//
// This does NOT compare against the existing hand-written loom::VitsDriver (the bespoke topology's own
// oracle) the way test_e2e_vits_lua_driver.cpp does for the bespoke Lua port -- deliberately. Chasing
// down an earlier version of this test's own large (~0.22 absolute, against a ~0.01-0.02 rms signal)
// mismatch against that oracle traced it to a REAL, previously-uncaught bug in the BESPOKE topology
// itself (vits_flow_vocoder.gguf's hand-built RESIDUAL_COUPLING_LAYER_REVERSE/HiFi-GAN composition),
// not this MIL-traced one: isolated via a standalone probe feeding the exact real end-to-end z_p (T=194,
// values up to +-24, dumped from this very test) into (a) this MIL topology, (b) the bespoke topology,
// and (c) a real PyTorch ResidualCouplingBlock+Generator forward pass -- (a) matched (c) to ~1.2e-6,
// (b) diverged from (c) by ~0.22. The bespoke path's own numerical verification
// (reference_forward_vits.py/test_e2e_vits_flow_vocoder_reference.cpp) only ever exercised a small-scale
// z_p (Tp=8, std=0.5), never a realistic one, so this bug was never caught. See
// test_e2e_vits_mil_flow_vocoder_reference.cpp (against reference_forward_vits_widerange.py's fixture,
// the SAME real-PyTorch ground truth used to root-cause this) for the tight numerical check on the MIL
// topology itself, and BACKLOG.md's VITS MIL-export entry for the full writeup. The bespoke oracle isn't
// a trustworthy comparison target at these input scales, so this test doesn't use it -- the per-phase
// reference tests (test_e2e_vits_mil_flow_vocoder_reference.cpp, plus the standalone probes recorded in
// BACKLOG.md for `stats`/`logw`) are where the real numerical confidence comes from. This test's own job
// is just to confirm the LUA ORCHESTRATION (three-topology packing, cross-phase RNG, generate_path) runs
// end-to-end without crashing and produces a sane result.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* dir_mil_env = loom_test::fixture_env("LOOM_VITS_MIL_DIR");
    if (dir_mil_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_MIL_DIR (a directory with vits_mil.gguf, produced "
                              "by export_vits_mil.py) to run this check\n");
        return 77;
    }
    const std::string dir_mil = dir_mil_env;

    // Same real BOS/blank-interleaved/EOS token-id sequence as test_e2e_vits_driver.cpp -- T=62, long
    // enough to exercise the real dynamic relative-position table beyond window_size+1=5.
    const std::vector<int32_t> token_ids = {
        1, 20, 0, 59, 0, 24, 0, 120, 0, 59, 0, 100, 0, 3, 0, 35, 0, 120, 0, 62, 0, 122, 0, 24, 0, 17, 0,
        8, 0, 3, 0, 41, 0, 74, 0, 31, 0, 3, 0, 74, 0, 38, 0, 3, 0, 50, 0, 3, 0, 32, 0, 120, 0, 61, 0, 31,
        0, 32, 0, 10, 0, 2};
    constexpr uint32_t kSeed = 42;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    std::vector<float> lua_wav;
    {
        auto model = loom::GgufModel::load(dir_mil + "/vits_mil.gguf", backend.get());
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
        // No inter_channels/noise_scale/noise_scale_w/length_scale: the driver carries all four as
        // ExportConstants now (P4.0.8's first follow-up).
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"token_ids", token_ids_d},
            {"seed", static_cast<double>(kSeed)},
        });
        const auto& wav_d = std::get<std::vector<double>>(result);
        lua_wav.assign(wav_d.begin(), wav_d.end());

        // The three scales stay OVERRIDABLE (`inputs.length_scale or LENGTH_SCALE`), because unlike
        // every other constant this export binds they are per-utterance knobs -- length_scale is
        // speaking rate. Passing piper's own defaults explicitly must therefore reproduce the call
        // above exactly, which checks both halves at once: that the `or` fallback is reached when the
        // caller says nothing, and that what it falls back to is the value this test used to have to
        // supply. `infer` re-seeds from `inputs.seed`, so the two calls are directly comparable.
        loom::LoomLuaBridge::Value explicit_result = bridge.call("infer", {
            {"token_ids", token_ids_d},
            {"seed", static_cast<double>(kSeed)},
            // piper's own published synthesis defaults, as doubles. Deliberately NOT
            // tts_driver_inputs.h's `float` copies: `static_cast<double>(0.667f)` is
            // 0.6669999957084656, so comparing against them could only ever be approximate -- and the
            // whole point of this call is that it is exact.
            {"noise_scale", 0.667},
            {"noise_scale_w", 0.8},
            {"length_scale", 1.0},
        });
        const auto& explicit_d = std::get<std::vector<double>>(explicit_result);
        LOOM_CHECK(explicit_d.size() == wav_d.size());
        double override_max_diff = 0.0;
        for (size_t i = 0; i < wav_d.size(); ++i) {
            override_max_diff = std::max(override_max_diff, std::fabs(explicit_d[i] - wav_d[i]));
        }
        std::fprintf(stderr, "defaults-vs-explicit max_abs_diff=%g\n", override_max_diff);
        LOOM_CHECK(override_max_diff == 0.0);
    }

    LOOM_CHECK(!lua_wav.empty());
    // 256 = HiFi-GAN's own total upsample factor (8*8*4) -- the waveform length must be an exact
    // multiple of it (one full frame's worth of samples per duration-expanded z_p column).
    LOOM_CHECK(lua_wav.size() % 256 == 0);

    double sum_sq = 0.0, max_abs = 0.0;
    for (float v : lua_wav) {
        LOOM_CHECK(std::isfinite(v));
        sum_sq += static_cast<double>(v) * v;
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(v)));
    }
    const double rms = std::sqrt(sum_sq / lua_wav.size());
    std::fprintf(stderr, "waveform_len=%zu, rms=%g, max_abs=%g\n", lua_wav.size(), rms, max_abs);
    // Real piper waveforms observed in this whole effort sit in the ~0.005-0.05 rms / <1.0 max-abs
    // range (never clipping/saturating at +-1.0) -- generous bounds, just enough to catch a genuinely
    // broken (silent, exploding, or clipped) synthesis rather than pin an exact expected loudness.
    LOOM_CHECK(rms > 1e-4 && rms < 0.5);
    LOOM_CHECK(max_abs < 1.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
