// End-to-end test of loom::SupertonicDriver against the real femelo/supertonic-tts checkpoint
// (converted via tools/convert_supertonic/convert_supertonic_all.py): loads the full GGUF set it
// produces, constructs a SupertonicDriver, and runs a full synthesize() call -- exercising the whole
// pipeline (DurationPredictor -> get_latent_mask -> TTLTextEncoder -> Euler CFM sampling loop over
// VectorFieldEstimator -> SpeechDecoder) together for the first time, not just each piece in isolation
// (every other test_e2e_supertonic_*.cpp covers those). Checks the output is finite and non-trivial
// (not silence, not NaN/Inf) -- NOT yet a numerical match against a hand-rolled full-pipeline Python
// reference, same scope as Kokoro's/StyleTTS2's own driver tests. Uses a real voice-style JSON asset
// (F1.json) for style_ttl/style_dp -- real precomputed styles, not synthetic. Skips cleanly
// (SKIP_RETURN_CODE 77) if LOOM_SUPERTONIC_ALL_DIR isn't set.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
#include <nlohmann/json.hpp>

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

// Loads a real `assets/voice_styles/*.json` asset's `style_ttl`/`style_dp` fields (each `{"data":
// [[[...]]], "dims": [...]}` -- "data" is NESTED to match "dims", e.g. (1,50,256), not a flat array) as
// flat float vectors, row-major flattened -- already in the exact Layout B (style-index-major,
// channel-minor) order this project's own style-encoder outputs use, no reordering needed.
std::vector<float> load_style_field(const nlohmann::json& j, const char* field) {
    std::vector<float> out;
    flatten_recursive(j.at(field).at("data"), out);
    return out;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_SUPERTONIC_ALL_DIR");
    const char* style_json_env = std::getenv("LOOM_SUPERTONIC_VOICE_STYLE_JSON");
    if (dir_env == nullptr || style_json_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_ALL_DIR (a directory produced by "
                              "convert_supertonic_all.py) and LOOM_SUPERTONIC_VOICE_STYLE_JSON (a real "
                              "assets/voice_styles/*.json, e.g. F1.json) to run this end-to-end test\n");
        return 77;
    }
    const std::string dir = dir_env;

    std::ifstream f(style_json_env);
    LOOM_CHECK(static_cast<bool>(f));
    nlohmann::json style_json;
    f >> style_json;
    const std::vector<float> style_ttl = load_style_field(style_json, "style_ttl");
    const std::vector<float> style_dp = load_style_field(style_json, "style_dp");
    LOOM_CHECK(style_ttl.size() == 50 * 256);
    LOOM_CHECK(style_dp.size() == 8 * 16);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::SupertonicConfig cfg;  // real defaults: txt_len_fixed=10 (see convert_supertonic_all.py), ...
    loom::SupertonicDriver driver(dir, cfg, backend.get());

    // Real vocab_size=163; a handful of arbitrary small ids, length MUST equal cfg.txt_len_fixed (a
    // real, documented scope limitation -- see SupertonicConfig's own docstring). Not a real
    // phonemization (the real license-free TextVectorizer is a separate, still-open integration).
    const std::vector<int32_t> txt_ids = {12, 45, 67, 23, 89, 34, 56, 78, 90, 15};
    LOOM_CHECK(txt_ids.size() == cfg.txt_len_fixed);

    std::vector<float> wav = driver.synthesize(txt_ids, style_ttl, style_dp, /*n_steps=*/10, /*seed=*/42);

    LOOM_CHECK(!wav.empty());
    bool all_finite = true;
    bool non_trivial = false;
    for (float v : wav) {
        if (!std::isfinite(v)) all_finite = false;
        if (std::fabs(v) > 1e-6f) non_trivial = true;
    }
    std::fprintf(stderr, "waveform_len=%zu, all_finite=%d, non_trivial=%d\n", wav.size(), all_finite, non_trivial);
    LOOM_CHECK(all_finite);
    LOOM_CHECK(non_trivial);

    LOOM_TEST_REPORT_AND_RETURN();
}
