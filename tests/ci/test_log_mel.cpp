// tests/support/log_mel.h, the phase-blind waveform distance CosyVoice3's free-running arms are graded by:
// zero against itself, exactly a level change in dB, blind to a phase flip, and not blind to silence.
#include "test_util.h"
#include "log_mel.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int kRate = 24000;
    std::vector<double> x(kRate * 2), flipped(x.size()), half(x.size()), silent(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        const double t = static_cast<double>(i) / kRate;
        // Two partials and a slow vibrato: energy across a few mel bands, like a voiced segment.
        x[i] = 0.3 * std::sin(2 * M_PI * (220 * t + 3 * std::sin(2 * M_PI * 5 * t))) + 0.1 * std::sin(2 * M_PI * 1330 * t);
        flipped[i] = -x[i];
        half[i] = 0.5 * x[i];
    }
    const double self = loom_test::log_mel_distance_db(x, x, kRate);
    const double flip = loom_test::log_mel_distance_db(flipped, x, kRate);
    const double down = loom_test::log_mel_distance_db(half, x, kRate);
    const double quiet = loom_test::log_mel_distance_db(silent, x, kRate);
    std::fprintf(stderr, "self %.3g dB, phase flip %.3g dB, -6 dB %.4f dB, silence %.1f dB\n", self, flip, down, quiet);
    LOOM_CHECK(self == 0.0);
    LOOM_CHECK(flip < 1e-9);                                        // a sign is a phase: invisible
    LOOM_CHECK(std::fabs(down - 20.0 * std::log10(2.0)) < 1e-6);    // a level change is exactly its dB
    LOOM_CHECK(quiet > 60.0);
    LOOM_TEST_REPORT_AND_RETURN();
}
