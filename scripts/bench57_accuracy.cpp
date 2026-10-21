// ADR-079 P4.32f f2 step 3a (ggml-0039): the ARMv6 activation kernels' arithmetic as scalar C, in the
// kernels' exact operation order (scripts/gen_armv6_act_f32.py), measured against f64 and against glibc
// over every 11th float with |x| <= 90 (x86, IEEE; ~2 minutes).
//   g++ -O2 -std=c++17 -ffp-contract=off bench57_accuracy.cpp -o bench57_accuracy && ./bench57_accuracy
// exp's own row reports inf at |x| >= 88.37 by design (the kernels return inf/0 there, where n would
// reach 128); silu's huge row below x = -88.7 is 1 + e^-x overflowing, the same in glibc.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

static inline uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// e^v for v in any range; the integer pass handles |v| beyond the core's reach
static const float LOG2E = 1.44269504088896341f, MAGIC = 12582912.0f;   // 1.5 * 2^23
static const float LN2HI = 0.693359375f, LN2LO = -2.12194440e-4f;        // Cephes' split
static const float C0 = 1.9875691500e-4f, C1 = 1.3981999507e-3f, C2 = 8.3334519073e-3f,
                   C3 = 4.1665795894e-2f, C4 = 1.6666665459e-1f, C5 = 5.0000001201e-1f;
static inline float exp_core(float v) {
    const float t  = v * LOG2E + MAGIC;
    const float nf = t - MAGIC;
    float r = v - nf * LN2HI;
    r = r - nf * LN2LO;
    float p = C0 * r + C1;
    p = p * r + C2; p = p * r + C3; p = p * r + C4; p = p * r + C5;
    const float r2 = r * r;
    p = p * r2;
    p = p + r;
    p = p + 1.0f;
    // integer pass
    const uint32_t a = fb(v) & 0x7fffffffu;
    if (a > 0x7f800000u) return v;                               // NaN
    if ((fb(v) >> 31) && a >= fb(86.98f)) return 0.0f;  // n >= -125 above this (a sum past p's exponent field's bottom wraps into the sign)
    if (!(fb(v) >> 31) && a >= fb(88.37f)) return INFINITY;  // n <= 127 below this
    const int32_t n = (int32_t) (fb(t) - fb(MAGIC));
    return bf(fb(p) + ((uint32_t) n << 23));
}
static inline float sigm(float x) { return 1.0f / (exp_core(-x) + 1.0f); }
static inline float silu(float x) { return x / (exp_core(-x) + 1.0f); }
static const float T0 = -5.70498872745e-3f, T1 = 2.06390887954e-2f, T2 = -5.37397155531e-2f,
                   T3 = 1.33314422036e-1f, T4 = -3.33332819422e-1f;
static inline float tanh_(float x) {
    const float ax = fabsf(x);
    // small: x + x^3 P(x^2)
    const float z = x * x;
    float s = T0 * z + T1; s = s * z + T2; s = s * z + T3; s = s * z + T4;
    s = s * z; s = s * x; s = s + x;
    // large: 1 - 2 / (e^{2|x|} + 1), sign of x
    const float e = exp_core(ax + ax);
    float l = 2.0f / (e + 1.0f);
    l = 1.0f - l;
    const uint32_t a = fb(x) & 0x7fffffffu;
    if (a > 0x7f800000u) return x;
    if (a < fb(0.625f)) return bf(fb(s) | (fb(x) & 0x80000000u));
    return bf(fb(l) | (fb(x) & 0x80000000u));
}

int main() {
    double me[4] = {0}, ml[4] = {0};
    double ulp_max[4] = {0};
    float worst[4] = {0};
    // every 11th float bit pattern with |x| <= 90 (both signs)
    for (uint64_t u = 0; u <= 0x42b40000ull; u += 11) {
        for (int sgn = 0; sgn < 2; ++sgn) {
            const float x = bf((uint32_t) u | (sgn ? 0x80000000u : 0));
            const double xd = x;
            const double t[4] = { std::exp(xd), 1.0 / (1.0 + std::exp(-xd)), xd / (1.0 + std::exp(-xd)), std::tanh(xd) };
            const float o[4] = { exp_core(x), sigm(x), silu(x), tanh_(x) };
            const float l[4] = { expf(x), 1.f / (1.f + expf(-x)), x / (1.0f + expf(-x)), tanhf(x) };
            for (int k = 0; k < 4; ++k) {
                if (!std::isfinite(t[k]) || std::fabs(t[k]) < 1.2e-38) continue;
                const double ulp = std::ldexp(1.0, std::ilogb((float) t[k]) - 23);
                const double eo = std::fabs(o[k] - t[k]) / ulp, el = std::fabs(l[k] - t[k]) / ulp;
                if (!(eo <= ulp_max[k])) { ulp_max[k] = std::isnan(eo) ? INFINITY : eo; worst[k] = x; }   // NaN counts
                ml[k] = std::max(ml[k], el);
                me[k] = std::max(me[k], std::fabs(o[k] - t[k]) / std::fabs(t[k]));
            }
        }
    }
    const char * nm[4] = { "exp", "sigmoid", "silu", "tanh" };
    for (int k = 0; k < 4; ++k)
        printf("%-8s ours: max %.2f ulp (rel %.2e) at x=%g   glibc: max %.2f ulp\n", nm[k], ulp_max[k], me[k], worst[k], ml[k]);
}
