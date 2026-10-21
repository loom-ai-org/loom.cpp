#pragma once

// The engine's FFT (ADR-080). A DFT-basis convolution does an STFT or an iSTFT in N^2 MACs per frame;
// this does it in O(N log N), and on the Pi Zero that took heart's iSTFT from 2.23 s to 0.18 s.
//
// Small on purpose: a mixed-radix complex FFT (radices 4, 2, 3, 5, and a generic odd one for anything
// else) of size N/2, and the split that turns it into a real FFT of size N. Every size the zoo uses
// (16, 20, 256, 400, 512, 1024, 1920) factors into 2, 3 and 5. A plan is built once per size and kept
// for the life of the process, so a graph's custom node can hold a pointer to it.

#include <complex>
#include <cstddef>
#include <vector>

namespace loom::fft {

class RealFft {
public:
    // n must be even and >= 2.
    explicit RealFft(int n);

    int size() const { return n_; }
    int bins() const { return n_ / 2 + 1; }

    // x[n] -> X[k], k = 0..n/2 (unnormalized, e^{-2 pi i k n / N}). `work` holds n/2 complex values.
    void forward(const float* x, float* re, float* im, std::complex<float>* work) const;

    // X[k], k = 0..n/2 -> x[n] = irfft(X), normalized by 1/n -- numpy's irfft. The imaginary parts of
    // the DC and Nyquist bins are ignored, as a real inverse must. `work` holds n/2 complex values.
    void inverse(const float* re, const float* im, float* x, std::complex<float>* work) const;

    size_t work_size() const { return static_cast<size_t>(m_); }   // complex values, either direction

    // The process-wide plan for size n. The pointer stays valid until exit.
    static const RealFft& plan(int n);

private:
    void complex_fft(const std::complex<float>* in, std::complex<float>* out, bool inverse) const;
    void work(std::complex<float>* out, const std::complex<float>* in, size_t fstride, int stage,
              bool inverse) const;

    int n_ = 0, m_ = 0;                             // real size, complex size n/2
    std::vector<int> factors_;                      // (radix, remaining length) pairs, kissfft-style
    std::vector<std::complex<float>> twiddles_;     // e^{-2 pi i j / m}, j < m
    std::vector<std::complex<float>> split_;        // e^{-2 pi i k / n}, k <= m
};

}  // namespace loom::fft
