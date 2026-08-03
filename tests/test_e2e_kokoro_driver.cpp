// End-to-end test of loom::KokoroDriver against the real Kokoro-82M checkpoint (converted via
// tools/convert_kokoro/convert_kokoro_all.py): loads the full GGUF set it produces, constructs a
// KokoroDriver, and runs a full synthesize() call -- exercising the whole pipeline (CustomAlbert ->
// bert_encoder -> DurationEncoder -> duration prediction/frame-expansion -> F0Ntrain -> Decoder core ->
// SineGen -> forward STFT -> Generator core -> inverse STFT) together for the first time, not just each
// topology in isolation (every other test_e2e_kokoro_*.cpp covers those). Checks the output is finite and
// non-trivial (not silence, not NaN/Inf) -- NOT yet a numerical match against a hand-rolled full-pipeline
// Python reference (a separate, much larger piece of work; see BACKLOG.md), same scope as VITS's own
// test_e2e_vits_driver.cpp. Skips cleanly (SKIP_RETURN_CODE 77) if LOOM_KOKORO_ALL_DIR isn't set.

#include "test_util.h"
#include "npy_fixture.h"

#include "loom/loom.h"
#include "loom/loom_legacy.h" // the pre-MIL C++ driver this test uses as its oracle

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_ALL_DIR");
    if (dir_env == nullptr) {
        std::fprintf(stderr, "skipping: real Kokoro GGUF set not found (set LOOM_KOKORO_ALL_DIR to a "
                              "directory produced by tools/convert_kokoro/convert_kokoro_all.py)\n");
        return 77;
    }
    const std::string dir = dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::KokoroConfig cfg;  // real defaults: style_dim=128, d_model=512, harmonic_num=8, ...
    loom::KokoroDriver driver(dir, cfg, backend.get());

    // Real KModel.forward always wraps with a leading/trailing 0 token ([0, *input_ids, 0]) -- a handful
    // of arbitrary small ids within Kokoro's real vocab_size=178, not a real phonemization (that's a
    // separate, still-open piece of work, task #79) -- this test only exercises the MODEL's own forward
    // pass shape/finiteness, not phoneme-to-audio fidelity.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9, 0};

    // ref_s: 256 floats (ref_s[:128] = decoder style, ref_s[128:] = predictor style, real KModel
    // convention) -- a small fixed synthetic vector (no real reference-speaker embedding on hand), same
    // "checkpoint's own real weights, but a synthetic driving input" scope as this test's own smoke-test
    // precedent (VITS's test_e2e_vits_driver.cpp).
    std::vector<float> ref_s(256);
    for (size_t i = 0; i < ref_s.size(); ++i) ref_s[i] = 0.05f * std::sin(static_cast<float>(i) * 0.37f);

    std::vector<float> wav = driver.synthesize(input_ids, ref_s, /*speed=*/1.0f, /*seed=*/42);

    // P4.0.8/E.3: this driver is being retired, and this test -- its own oracle, at these exact inputs
    // -- is the ONLY thing that can produce the reference waveform its Lua successor compares against
    // once it is gone. `LOOM_DUMP_REF_NPY=<path>` freezes it into tests/fixtures/legacy_driver_reference/.
    // Not part of the check; see that directory's README.md for the recipe and for why the fixture
    // cannot be regenerated after this file is deleted.
    if (const char* dump_path = std::getenv("LOOM_DUMP_REF_NPY")) {
        LOOM_CHECK(loom_test::write_npy_f32(dump_path, wav));
        std::fprintf(stderr, "LOOM_DUMP_REF_NPY: wrote %zu samples to %s\n", wav.size(), dump_path);
    }

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
