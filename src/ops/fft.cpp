#include "loom/ops/fft.h"

#include "loom/loom_errors.h"

#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace loom::fft {

namespace {

using cf = std::complex<float>;

// Plain arithmetic rather than std::complex's operator*, which handles inf/nan per C Annex G and
// costs a branch per multiply on every compiler this builds with.
inline cf mul(cf a, cf b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

}  // namespace

RealFft::RealFft(int n) : n_(n), m_(n / 2) {
    if (n < 2 || n % 2 != 0) {
        throw Error("loom::fft::RealFft: size " + std::to_string(n) + " is not an even number >= 2");
    }
    // kissfft's factor order: 4s first, then 2, then odd radices.
    int rest = m_, p = 4;
    while (rest > 1) {
        while (rest % p != 0) {
            p = p == 4 ? 2 : p == 2 ? 3 : p + 2;
            if (p * p > rest) p = rest;
        }
        rest /= p;
        factors_.push_back(p);
        factors_.push_back(rest);
    }
    if (factors_.empty()) {   // m == 1: one complex point, a one-entry "radix-1" stage
        factors_.push_back(1);
        factors_.push_back(1);
    }
    // f64 angles from an integer phase below the size: the precision lesson of the DFT basis
    // (loom-exporter torch_patches.py, item 4), where a float `k*n*2*pi/N` cost 1.4e-4.
    twiddles_.resize(m_);
    for (int j = 0; j < m_; ++j) {
        const double a = -2.0 * M_PI * j / m_;
        twiddles_[j] = cf(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
    split_.resize(m_ + 1);
    for (int k = 0; k <= m_; ++k) {
        const double a = -2.0 * M_PI * k / n_;
        split_[k] = cf(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
}

void RealFft::work(cf* out, const cf* in, size_t fstride, int stage, bool inverse) const {
    const int p = factors_[2 * stage], m = factors_[2 * stage + 1];
    cf* const beg = out;
    cf* const end = out + static_cast<size_t>(p) * m;
    if (m == 1) {
        for (cf* o = out; o != end; ++o, in += fstride) *o = *in;
    } else {
        for (cf* o = out; o != end; o += m, in += fstride) work(o, in, fstride * p, stage + 1, inverse);
    }
    out = beg;
    auto tw = [&](size_t i) { const cf t = twiddles_[i]; return inverse ? std::conj(t) : t; };

    switch (p) {
        case 1:
            break;
        case 2:
            for (int k = 0; k < m; ++k) {
                const cf t = mul(out[k + m], tw(k * fstride));
                out[k + m] = out[k] - t;
                out[k] += t;
            }
            break;
        case 4:
            for (int k = 0; k < m; ++k) {
                const cf s0 = mul(out[k + m], tw(k * fstride));
                const cf s1 = mul(out[k + 2 * m], tw(2 * k * fstride));
                const cf s2 = mul(out[k + 3 * m], tw(3 * k * fstride));
                const cf s5 = out[k] - s1;
                const cf a = out[k] + s1;
                const cf s3 = s0 + s2, s4 = s0 - s2;
                out[k + 2 * m] = a - s3;
                out[k] = a + s3;
                // X1 = s5 - i s4 forward, s5 + i s4 inverse; X3 the other sign
                const cf mi(s4.imag(), -s4.real());   // -i * s4
                out[k + m]     = inverse ? s5 - mi : s5 + mi;
                out[k + 3 * m] = inverse ? s5 + mi : s5 - mi;
            }
            break;
        case 3: {
            const float e = tw(fstride * m).imag();   // Im e^{-+2 pi i / 3}
            for (int k = 0; k < m; ++k) {
                const cf s1 = mul(out[k + m], tw(k * fstride));
                const cf s2 = mul(out[k + 2 * m], tw(2 * k * fstride));
                const cf s3 = s1 + s2;
                const cf s0 = (s1 - s2) * e;
                const cf h = out[k] - s3 * 0.5f;
                out[k] += s3;
                out[k + 2 * m] = cf(h.real() + s0.imag(), h.imag() - s0.real());
                out[k + m]     = cf(h.real() - s0.imag(), h.imag() + s0.real());
            }
            break;
        }
        case 5: {
            const cf ya = tw(fstride * m), yb = tw(2 * fstride * m);
            for (int u = 0; u < m; ++u) {
                const cf s0 = out[u];
                const cf s1 = mul(out[u + m], tw(u * fstride));
                const cf s2 = mul(out[u + 2 * m], tw(2 * u * fstride));
                const cf s3 = mul(out[u + 3 * m], tw(3 * u * fstride));
                const cf s4 = mul(out[u + 4 * m], tw(4 * u * fstride));
                const cf s7 = s1 + s4, s10 = s1 - s4, s8 = s2 + s3, s9 = s2 - s3;
                out[u] = s0 + s7 + s8;
                const cf s5(s0.real() + s7.real() * ya.real() + s8.real() * yb.real(),
                            s0.imag() + s7.imag() * ya.real() + s8.imag() * yb.real());
                const cf s6(s10.imag() * ya.imag() + s9.imag() * yb.imag(),
                            -s10.real() * ya.imag() - s9.real() * yb.imag());
                out[u + m] = s5 - s6;
                out[u + 4 * m] = s5 + s6;
                const cf s11(s0.real() + s7.real() * yb.real() + s8.real() * ya.real(),
                             s0.imag() + s7.imag() * yb.real() + s8.imag() * ya.real());
                const cf s12(-s10.imag() * yb.imag() + s9.imag() * ya.imag(),
                             s10.real() * yb.imag() - s9.real() * ya.imag());
                out[u + 2 * m] = s11 + s12;
                out[u + 3 * m] = s11 - s12;
            }
            break;
        }
        default: {   // any other (odd) radix: a direct DFT of p points per butterfly
            cf scratch[64];
            if (p > 64) throw Error("loom::fft: radix " + std::to_string(p) + " is above the generic butterfly's 64");
            const size_t norig = static_cast<size_t>(m_);
            for (int u = 0; u < m; ++u) {
                for (int q = 0, k = u; q < p; ++q, k += m) scratch[q] = out[k];
                for (int q1 = 0, k = u; q1 < p; ++q1, k += m) {
                    size_t ti = 0;
                    cf acc = scratch[0];
                    for (int q = 1; q < p; ++q) {
                        ti += fstride * static_cast<size_t>(k);
                        if (ti >= norig) ti %= norig;
                        acc += mul(scratch[q], tw(ti));
                    }
                    out[k] = acc;
                }
            }
            break;
        }
    }
}

void RealFft::complex_fft(const cf* in, cf* out, bool inverse) const {
    work(out, in, 1, 0, inverse);
}

void RealFft::forward(const float* x, float* re, float* im, cf* z) const {
    // The n reals as n/2 complex points, evens real and odds imaginary: Z = E + i O.
    complex_fft(reinterpret_cast<const cf*>(x), z, false);
    for (int k = 0; k <= m_; ++k) {
        const cf a = z[k % m_];
        const cf b = std::conj(z[(m_ - k) % m_]);
        const cf e = (a + b) * 0.5f;
        const cf d = (a - b) * 0.5f;
        const cf o(d.imag(), -d.real());   // (a - b) / 2i
        const cf X = e + mul(split_[k], o);
        re[k] = X.real();
        im[k] = X.imag();
    }
}

void RealFft::inverse(const float* re, const float* im, float* x, cf* z) const {
    for (int k = 0; k < m_; ++k) {
        // E = (X[k] + conj X[m-k]) / 2, O = (X[k] - conj X[m-k]) e^{+2 pi i k / n} / 2, Z = E + i O.
        // The DC and Nyquist imaginary parts are dropped here, which is what a real inverse means.
        const cf a(re[k], k == 0 ? 0.0f : im[k]);
        const cf b(re[m_ - k], k == 0 ? 0.0f : -im[m_ - k]);
        const cf e = (a + b) * 0.5f;
        const cf o = mul((a - b) * 0.5f, std::conj(split_[k]));
        z[k] = cf(e.real() - o.imag(), e.imag() + o.real());
    }
    cf* out = reinterpret_cast<cf*>(x);
    complex_fft(z, out, true);
    const float s = 1.0f / static_cast<float>(m_);
    for (int i = 0; i < n_; ++i) x[i] *= s;
}

const RealFft& RealFft::plan(int n) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<RealFft>> plans;
    std::lock_guard<std::mutex> lock(mu);
    auto& slot = plans[n];
    if (!slot) slot = std::make_unique<RealFft>(n);
    return *slot;
}

}  // namespace loom::fft
