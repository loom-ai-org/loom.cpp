// End-to-end test of loom::StyleTTS2Driver against the real yl4579/StyleTTS2-LJSpeech checkpoint
// (converted via tools/convert_styletts2/convert_styletts2_reused.py +
// tools/convert_styletts2/convert_styletts2_diffusion.py): loads the full GGUF set they produce,
// constructs a StyleTTS2Driver, and runs a full synthesize() call -- exercising the whole pipeline
// (CustomAlbert -> style-diffusion sampler -> bert_encoder -> DurationEncoder -> duration
// prediction/frame-expansion -> F0Ntrain -> Decoder core -> SineGen -> forward STFT -> Generator core)
// together for the first time, not just each piece in isolation (every other
// test_e2e_styletts2_*.cpp covers those). Checks the output is finite and non-trivial (not silence, not
// NaN/Inf) -- NOT yet a numerical match against a hand-rolled full-pipeline Python reference (a
// separate, much larger piece of work), same scope as Kokoro's own test_e2e_kokoro_driver.cpp. Skips
// cleanly (SKIP_RETURN_CODE 77) if LOOM_STYLETTS2_DIR isn't set.

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
    const char* dir_env = std::getenv("LOOM_STYLETTS2_DIR");
    if (dir_env == nullptr) {
        std::fprintf(stderr, "skipping: real StyleTTS2 GGUF set not found (set LOOM_STYLETTS2_DIR to a "
                              "directory produced by convert_styletts2_reused.py + "
                              "convert_styletts2_diffusion.py)\n");
        return 77;
    }
    const std::string dir = dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::StyleTTS2Config cfg;  // real defaults: style_dim=128, d_model=512, sigma_data from config.yml, ...
    loom::StyleTTS2Driver driver(dir, cfg, backend.get());

    // Real StyleTTS2 demo wraps with a SINGLE LEADING 0 token (`tokens.insert(0, 0)`, NOT Kokoro's
    // leading+trailing convention) -- a handful of arbitrary small ids within the real vocab_size=178,
    // not a real phonemization (task #79, still open) -- this test only exercises the MODEL's own
    // forward pass shape/finiteness, not phoneme-to-audio fidelity.
    const std::vector<int32_t> input_ids = {0, 50, 62, 24, 83, 16, 44, 71, 9};

    std::vector<float> wav = driver.synthesize(input_ids, /*diffusion_steps=*/5, /*seed=*/42);

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
