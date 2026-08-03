// End-to-end check: loom::MatchaDriver's full synthesize() call (TextEncoder -> duration expansion ->
// Euler CFM sampling loop over the Decoder U-Net -> denormalize -> HiFi-GAN v1 vocoder) against the
// real matcha_ljspeech.ckpt + generator_v1 checkpoints -- finite, non-trivial output, same scope as
// test_e2e_supertonic_driver.cpp/test_e2e_kokoro_driver.cpp. Skips cleanly if the env var isn't set.

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
    const char* dir_env = std::getenv("LOOM_MATCHA_DIR");
    if (dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_DIR (a directory with matcha_encoder_mu.gguf, "
                              "matcha_encoder_logw.gguf, matcha_decoder.gguf, matcha_vocoder.gguf, "
                              "produced by tools/convert_matcha/convert_matcha_*.py) to run this "
                              "end-to-end test\n");
        return 77;
    }
    const std::string dir = dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::MatchaConfig cfg;  // real defaults: n_feats=80, mel_mean/mel_std from the real checkpoint
    loom::MatchaDriver driver(dir, cfg, backend.get());

    // Real n_vocab=178; a handful of arbitrary small ids -- not a real phonemization (the real
    // license-free tokenizer is a separate, still-open integration, same as VITS/Kokoro/StyleTTS2's
    // own established scope for driver smoke tests).
    const std::vector<int32_t> tokens = {5, 42, 7, 88, 13, 100, 3, 61};

    std::vector<float> wav = driver.synthesize(tokens, /*n_steps=*/10, /*seed=*/42);

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
    std::fprintf(stderr, "wav.size()=%zu, all_finite=%d, non_trivial=%d\n", wav.size(), all_finite, non_trivial);
    LOOM_CHECK(all_finite);
    LOOM_CHECK(non_trivial);

    LOOM_TEST_REPORT_AND_RETURN();
}
