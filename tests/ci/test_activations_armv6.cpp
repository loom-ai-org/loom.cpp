// The ARMv6 activation kernels (ADR-079 P4.32f f2 step 3a, ggml-0039): GELU_ERF, SIGMOID, SILU, TANH; and
// ELU (P4.41, ggml-0052).
//
// On ARMv6 these ran per element at 160-440 ns on a Pi Zero W: GELU_ERF through ggml-0010's scalar rational
// (no SIMD body exists there), the other three through glibc's expf and tanhf. ggml-0039 runs them on the
// VFP's short vectors. GELU_ERF keeps ggml-0010's arithmetic to the bit; SIGMOID, SILU and TANH use an own
// exp (Cephes' construction) and so differ from glibc at rounding level. Registered twice,
// GGML_CPU_ARMV6_ACT=0 and =1 (read once), and pins:
//
//   1. **The numbers**, in both runs, against f64: GELU_ERF within ggml-0010's measured bound (2.64e-7 of
//      max(|x|, 1), ~2 ULP of scale), TANH within 2 ULP and SIGMOID / SILU within 3 of the true value.
//   2. **The special values**, as the stock code produces them: NaN in, NaN out; sigmoid(+-inf) = 1 / 0,
//      tanh(+-inf) = +-1, silu(+inf) = +inf, and +-0 kept by tanh and silu; elu(-inf) = -1, +-0 kept.
//   3. **GELU_ERF bit-identity**: the two runs' outputs compared byte for byte (whichever runs second).
//
// Rows of 1, 7, 8, 9 and 1003 floats cover the eight-float blocks and the padded tail; TANH and SIGMOID
// also run in place. Elsewhere both registrations are stock ggml and pass the same checks.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool compute(ggml_cgraph* gf) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

bool act_on() {
    const char* e = std::getenv("GGML_CPU_ARMV6_ACT");
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
    return !(e != nullptr && std::atoi(e) == 0);
#else
    (void) e;
    return false;
#endif
}

enum Kind { GELU_ERF, SIGMOID, SILU, TANH, ELU };
const char* kNames[] = { "gelu_erf", "sigmoid", "silu", "tanh", "elu" };

double ref(Kind k, double x) {
    switch (k) {
        case GELU_ERF: return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        case SIGMOID:  return 1.0 / (1.0 + std::exp(-x));
        case SILU:     return x / (1.0 + std::exp(-x));
        case TANH:     return std::tanh(x);
        case ELU:      return x > 0 ? x : std::expm1(x);
    }
    return 0;
}

ggml_tensor* node(ggml_context* ctx, Kind k, ggml_tensor* a, bool inplace) {
    switch (k) {
        case GELU_ERF: return ggml_gelu_erf(ctx, a);
        case SIGMOID:  return inplace ? ggml_sigmoid_inplace(ctx, a) : ggml_sigmoid(ctx, a);
        case SILU:     return ggml_silu(ctx, a);
        case TANH:     return inplace ? ggml_tanh_inplace(ctx, a) : ggml_tanh(ctx, a);
        case ELU:      return inplace ? ggml_elu_inplace(ctx, a) : ggml_elu(ctx, a);
    }
    return nullptr;
}

// one tensor of `rows` rows of n floats through kind k; returns the outputs and the inputs it used
std::vector<float> run(Kind k, int64_t n, int64_t rows, bool inplace, std::vector<float>& in) {
    ggml_init_params ip = { (size_t) 8 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
    float* d = (float*) a->data;
    // row 0 over [-12, 12], the rest dense over [-1, 1], where tanh's two formulas meet (and the
    // large-|x| one cancels: a threshold moved from 0.625 to 0.0625 passed a 4-ULP bound over [-12, 12])
    for (int64_t i = 0; i < n * rows; ++i) {
        const float u = (float) ((i * 7919) % 1009) / 1009.0f;
        d[i] = i < n ? -12.0f + 24.0f * u : -1.0f + 2.0f * u;
    }
    // both ends of the kernels' exp range, and beyond: e^-x wrapped into a NaN for x in (88.03, 88.37) and
    // into a wrong tiny value for x in (86.98, 88.03] before the bound below was made asymmetric
    const float big[] = { 86.9f, -86.9f, 87.0f, -87.0f, 88.0f, -88.0f, 88.2f, -88.2f, 88.3f, -88.3f,
                          88.5f, -88.5f, 100.0f, -100.0f,
                          20.0f, -20.0f, 50.0f, -50.0f, 1e-3f, -1e-3f };
    const int64_t nbig = sizeof big / sizeof big[0];
    for (int64_t i = 0; i < n * rows && i < nbig; ++i) d[(i * 13) % (n * rows)] = big[i];
    in.assign(d, d + n * rows);
    ggml_tensor* y = node(ctx, k, a, inplace);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    LOOM_CHECK(compute(gf));
    std::vector<float> out((float*) y->data, (float*) y->data + n * rows);
    ggml_free(ctx);
    return out;
}

// worst error in the kind's own unit: ULP of the true value, or GELU_ERF's |err| / max(|x|, 1)
double worst(Kind k, const std::vector<float>& in, const std::vector<float>& out) {
    double w = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const double t = ref(k, in[i]);
        double e;
        if (k == GELU_ERF) {
            e = std::fabs(out[i] - t) / std::fmax(std::fabs((double) in[i]), 1.0);
        } else {
            if (std::fabs(t) < 1e-30) { e = std::fabs(out[i]) < 1e-30 ? 0 : 1e9; }   // underflowed: must be ~0
            else e = std::fabs(out[i] - t) / std::ldexp(1.0, std::ilogb((float) t) - 23);
        }
        // NOT fmax: fmax(w, NaN) is w, and a NaN output passed this check once (the kernels' exp wrapped
        // into the sign bit for every x in (86.98, 88.37] and SILU/SIGMOID returned NaN there)
        if (!(e <= w)) w = std::isnan(e) ? INFINITY : e;
    }
    return w;
}

float one(Kind k, float x) {
    ggml_init_params ip = { (size_t) 1 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*) a->data)[0] = x;
    ggml_tensor* y = node(ctx, k, a, false);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    LOOM_CHECK(compute(gf));
    const float r = ((float*) y->data)[0];
    ggml_free(ctx);
    return r;
}

} // namespace

int main() {
    const bool on = act_on();
    std::printf("GGML_CPU_ARMV6_ACT path %s\n", on ? "ON" : "off (stock ggml)");
    std::vector<float> gelu_all;
    for (Kind k : { GELU_ERF, SIGMOID, SILU, TANH, ELU }) {
        double w = 0;
        for (int64_t n : { 1, 7, 8, 9, 1003 }) {
            for (bool inplace : { false, true }) {
                if (inplace && (k == GELU_ERF || k == SILU)) continue;
                std::vector<float> in;
                const std::vector<float> out = run(k, n, 2, inplace, in);
                w = std::fmax(w, worst(k, in, out));
                if (k == GELU_ERF) gelu_all.insert(gelu_all.end(), out.begin(), out.end());
            }
        }
        // stock: tanhf 1.52 ULP, sigmoid/silu 1.7 / 1.4; the kernels 0.75, 1.7, 1.4 (on these inputs).
        // ELU (ggml-0052): glibc's expm1f against the kernel's own exp minus one where |x| > ~0.35.
        const double bound = k == GELU_ERF ? 2.7e-7 : k == TANH ? 2.0 : 3.0;
        std::printf("  %-9s worst %.3g %s (bound %.3g)\n", kNames[k], w, k == GELU_ERF ? "of max(|x|,1)" : "ULP", bound);
        LOOM_CHECK(w <= bound);
    }

    // special values, as the stock code produces them
    const float inf = INFINITY, nan = NAN;
    for (Kind k : { GELU_ERF, SIGMOID, SILU, TANH, ELU }) LOOM_CHECK(std::isnan(one(k, nan)));
    LOOM_CHECK(one(SIGMOID, inf) == 1.0f);
    LOOM_CHECK(one(SIGMOID, -inf) == 0.0f);
    LOOM_CHECK(one(TANH, inf) == 1.0f);
    LOOM_CHECK(one(TANH, -inf) == -1.0f);
    LOOM_CHECK(one(SILU, inf) == inf);
    LOOM_CHECK(one(TANH, 0.0f) == 0.0f && !std::signbit(one(TANH, 0.0f)));
    LOOM_CHECK(std::signbit(one(TANH, -0.0f)));
    LOOM_CHECK(one(SILU, 0.0f) == 0.0f);
    LOOM_CHECK(one(GELU_ERF, inf) == inf);
    LOOM_CHECK(one(ELU, inf) == inf);
    LOOM_CHECK(one(ELU, -inf) == -1.0f);
    LOOM_CHECK(one(ELU, -100.0f) == -1.0f);
    LOOM_CHECK(one(ELU, 0.0f) == 0.0f && !std::signbit(one(ELU, 0.0f)));
    LOOM_CHECK(one(ELU, -0.0f) == 0.0f && std::signbit(one(ELU, -0.0f)));
    LOOM_CHECK(one(ELU, -1e-30f) == -1e-30f);   // expm1 rounds to x below 2^-24
    LOOM_CHECK(one(ELU, 3.5f) == 3.5f);

    // GELU_ERF byte for byte against the other registration
    if (const char* dir = std::getenv("LOOM_TEST_TMPDIR")) {
        char mine[1024], other[1024];
        std::snprintf(mine, sizeof mine, "%s/act_armv6_gelu_%s.f32", dir, on ? "on" : "off");
        std::snprintf(other, sizeof other, "%s/act_armv6_gelu_%s.f32", dir, on ? "off" : "on");
        if (FILE* f = std::fopen(mine, "wb")) {
            std::fwrite(gelu_all.data(), sizeof(float), gelu_all.size(), f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen(other, "rb")) {
            std::vector<float> them(gelu_all.size());
            const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
            std::fclose(f);
            LOOM_CHECK(n == them.size());
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += std::memcmp(&gelu_all[i], &them[i], 4) != 0;
            std::printf("  gelu_erf vs the other registration: %zu of %zu floats differ\n", diff, n);
            LOOM_CHECK(diff == 0);
        }
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
