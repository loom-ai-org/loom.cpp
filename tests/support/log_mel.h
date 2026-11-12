#pragma once

// A log-mel distance between two waveforms: a TTS gate's comparison where the waveform itself is the wrong
// oracle.
//
// A vocoder that drives a sine source from F0 integrates F0 into a phase (CosyVoice3's CausalHiFT reaches
// ~1e6 rad within seconds, where an f32 ulp is 0.06 rad). Two correct float32 implementations then emit
// waveforms that are uncorrelated sample by sample while their spectrograms agree. On CosyVoice3's cloned
// voice the reference's OWN f32 run is rmse 0.249 from its f64 run, worse than silence (0.205). A log-mel
// does not see phase, so it separates builds (~0.5 dB) from silence, a 10 ms shift or a 6 dB level change.
//
// Fixed and simple on purpose, so `scripts/` or a notebook can mirror it exactly: a 2048-point Hann window
// every `hop` samples, zero-padded half a window at both ends (centred); power spectrum; 80 triangular
// HTK-mel filters from 0 to 8 kHz on the magnitude; 20*log10(max(x, 1e-5)). The distance is the mean |dB
// difference| over the bins within 60 dB of the REFERENCE's loudest, so silence in both counts for nothing.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace loom_test {

namespace log_mel_detail {

inline void fft(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / static_cast<double>(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

inline double hz_to_mel(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
inline double mel_to_hz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

} // namespace log_mel_detail

// [frames][80] in dB.
inline std::vector<std::vector<double>> log_mel(const std::vector<double>& wave, int sample_rate, int hop = 480) {
    using namespace log_mel_detail;
    constexpr int kFft = 2048, kMels = 80;
    constexpr double kFmax = 8000.0;
    const int bins = kFft / 2 + 1;
    std::vector<double> window(kFft);
    for (int i = 0; i < kFft; ++i) window[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / kFft);
    // Triangular filters on the HTK mel scale, kMels + 2 edges from 0 to kFmax.
    std::vector<double> edges(kMels + 2);
    for (int m = 0; m < kMels + 2; ++m) edges[m] = mel_to_hz(hz_to_mel(kFmax) * m / (kMels + 1));
    std::vector<std::vector<double>> filters(kMels, std::vector<double>(bins, 0.0));
    for (int m = 0; m < kMels; ++m) {
        for (int k = 0; k < bins; ++k) {
            const double f = static_cast<double>(k) * sample_rate / kFft;
            const double up = (f - edges[m]) / (edges[m + 1] - edges[m]);
            const double down = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            filters[m][k] = std::max(0.0, std::min(up, down));
        }
    }
    const int n = static_cast<int>(wave.size());
    const int frames = n / hop + 1;
    std::vector<std::vector<double>> out(frames, std::vector<double>(kMels));
    std::vector<std::complex<double>> buf(kFft);
    std::vector<double> mag(bins);
    for (int t = 0; t < frames; ++t) {
        const int start = t * hop - kFft / 2;
        for (int i = 0; i < kFft; ++i) {
            const int s = start + i;
            buf[i] = {(s >= 0 && s < n) ? wave[s] * window[i] : 0.0, 0.0};
        }
        fft(buf);
        for (int k = 0; k < bins; ++k) mag[k] = std::abs(buf[k]);
        for (int m = 0; m < kMels; ++m) {
            double e = 0.0;
            for (int k = 0; k < bins; ++k) e += filters[m][k] * mag[k];
            out[t][m] = 20.0 * std::log10(std::max(e, 1e-5));
        }
    }
    return out;
}

// Mean |dB difference| over the reference's bins within 60 dB of its loudest. Equal lengths are the
// caller's check; a length mismatch compares the common frames.
inline double log_mel_distance_db(const std::vector<double>& got, const std::vector<double>& ref, int sample_rate) {
    const auto a = log_mel(got, sample_rate), b = log_mel(ref, sample_rate);
    double peak = -1e300;
    for (const auto& row : b) for (double v : row) peak = std::max(peak, v);
    double sum = 0.0;
    size_t count = 0;
    for (size_t t = 0; t < std::min(a.size(), b.size()); ++t) {
        for (size_t m = 0; m < b[t].size(); ++m) {
            if (b[t][m] < peak - 60.0) continue;
            sum += std::fabs(a[t][m] - b[t][m]);
            ++count;
        }
    }
    return count ? sum / static_cast<double>(count) : 0.0;
}

} // namespace loom_test
