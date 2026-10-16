// A HiFi-GAN stage's TAIL is folded into its last resblock convolution (ADR-079 P4.32f f2 step 3b,
// ggml-0040).
//
// Piper/VITS-style vocoders end every upsampling stage with
//
//     s = ADD(r0, r1); s = ADD(s, r2)     the resblocks' sum
//     h = LEAKY_RELU(MUL(s, 1/3))         its mean, activated: the next transposed convolution's input
//
// where each r is a convolution with its bias and residual (ggml-0005/0007 fuse those). ggml-0040 lets
// the convolution producing r1 add r0, and the one producing r2 add the running sum, scale it and apply
// the LEAKY_RELU, each in the node's own arithmetic and order. It is part of the convolution epilogue
// (GGML_CPU_CONV_EPILOGUE, on by default on ARMv6 only), so this is registered with it off and on, and pins:
//
//   1. **The numbers**, against a double-precision reference, in both runs.
//   2. **That it fuses**: the residual ADD's own result is poisoned before the run and survives exactly
//      when the epilogue is on (the fused kernel writes only the last node's tensor).
//   3. **Bit-identity**: the two runs' outputs compared byte for byte (whichever runs second, DEPENDS).
//
// Two cases put the whole tail in the other operand's buffer, the exact alias a graph allocator produces
// for an ADD whose operand is dead; the fused kernel then has to read that operand before it stores.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr float kPoison = -123456.0f;

bool compute(ggml_cgraph* gf, int n_threads) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

bool epilogue_on() {
    const char* e = std::getenv("GGML_CPU_CONV_EPILOGUE");
    if (e != nullptr) return std::atoi(e) != 0;
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

bool all_poison(const ggml_tensor* t) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        if (((const float*) t->data)[i] != kPoison) return false;
    }
    return true;
}

void fill(std::vector<float>& v, float a, float b, int mod) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = a + b * (float) ((int) (i % mod) - mod / 2);
}

struct Case {
    const char* name;
    int64_t     kw, ic, oc, il, dil;
    bool        scale;     // * 1/3 after the sum
    bool        leaky;     // then LEAKY_RELU(0.1)
    int         inplace;   // 1: the allocator's way -- the sum, the MUL and the LEAKY share the other
                           // operand's buffer (nodes are not views); 2: ggml_add_inplace, a VIEW node
};

double run_case(ggml_context* ctx, const Case& cs, bool epi, std::vector<float>& got) {
    const int64_t pad = (cs.kw - 1) * cs.dil / 2;
    const int64_t ol  = cs.il + 2 * pad - (cs.kw - 1) * cs.dil;
    std::vector<float> K((size_t) cs.kw * cs.ic * cs.oc), X((size_t) cs.il * cs.ic), B((size_t) cs.oc),
                       R((size_t) ol * cs.oc), O((size_t) ol * cs.oc);
    fill(K, 0.003f, 0.0011f, 53);
    fill(X, 0.01f, 0.013f, 97);
    fill(B, 0.02f, 0.05f, 7);
    fill(R, -0.01f, 0.017f, 31);
    fill(O, 0.005f, -0.021f, 43);
    const float third = 1.0f / 3.0f, slope = 0.1f;

    ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.kw, 1, cs.ic, cs.oc);
    ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.il, 1, cs.ic, 1);
    ggml_tensor* tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, cs.oc, 1);
    ggml_tensor* tr = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ol, cs.oc, 1);
    ggml_tensor* to = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ol, cs.oc, 1);
    ggml_tensor* ts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 1);
    std::memcpy(tk->data, K.data(), K.size() * sizeof(float));
    std::memcpy(tx->data, X.data(), X.size() * sizeof(float));
    std::memcpy(tb->data, B.data(), B.size() * sizeof(float));
    std::memcpy(tr->data, R.data(), R.size() * sizeof(float));
    std::memcpy(to->data, O.data(), O.size() * sizeof(float));
    ((float*) ts->data)[0] = third;

    ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, tx, 1, 1, (int) pad, 0, (int) cs.dil, 1);
    ggml_tensor* yb   = ggml_add(ctx, ggml_reshape_3d(ctx, conv, ol, cs.oc, 1), tb);
    ggml_tensor* y    = ggml_add(ctx, yb, tr);                           // the resblock's residual
    ggml_tensor* s    = cs.inplace == 2 ? ggml_add_inplace(ctx, to, y) : ggml_add(ctx, to, y);
    ggml_tensor* m    = cs.scale ? ggml_mul(ctx, s, ts) : s;
    ggml_tensor* out  = cs.leaky ? ggml_leaky_relu(ctx, m, slope, false) : m;
    if (cs.inplace == 1) {
        // what a graph allocator does with a dead operand: the whole tail in the other operand's buffer,
        // so the fused result IS `to` -- which the kernel must read before it stores
        s->data = m->data = out->data = to->data;
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    for (int i = 0; i < ggml_nelements(y); ++i) ((float*) y->data)[i] = kPoison;
    const bool past_sum = cs.scale && cs.inplace == 0;   // the sum has its own buffer and a MUL after it
    if (past_sum) for (int i = 0; i < ggml_nelements(s); ++i) ((float*) s->data)[i] = kPoison;
    LOOM_CHECK(compute(gf, 2));
    // the residual ADD's own tensor is written only when the tail is NOT fused into the convolution,
    // and the sum's only when the fusion stopped at it
    LOOM_CHECK(all_poison(y) == epi);
    if (past_sum) LOOM_CHECK(all_poison(s) == epi);

    const float* o = (const float*) out->data;
    double err = 0, peak = 0;
    for (int64_t oc = 0; oc < cs.oc; ++oc) {
        for (int64_t t = 0; t < ol; ++t) {
            double acc = B[oc];
            for (int64_t ic = 0; ic < cs.ic; ++ic) {
                for (int64_t k = 0; k < cs.kw; ++k) {
                    const int64_t p = t + k * cs.dil - pad;
                    if (p >= 0 && p < cs.il) acc += (double) K[(oc * cs.ic + ic) * cs.kw + k] * X[ic * cs.il + p];
                }
            }
            const size_t i = (size_t) (oc * ol + t);
            double v = O[i] + (acc + R[i]);
            if (cs.scale) v *= third;
            if (cs.leaky) v = v > 0 ? v : slope * v;
            peak = std::fmax(peak, std::fabs(v));
            err  = std::fmax(err, std::fabs(o[i] - v));
            if (std::isnan(o[i])) err = INFINITY;
        }
    }
    got.assign(o, o + ggml_nelements(out));
    return err / peak;
}

} // namespace

int main() {
    const bool epi = epilogue_on();
    std::printf("GGML_CPU_CONV_EPILOGUE %s\n", epi ? "ON" : "off");
    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);

    const Case cases[] = {
        { "k3 d1 + other",                         3, 32, 32, 301, 1, false, false, 0 },
        { "k3 d3 + other, * 1/3, leaky",           3, 24, 24, 551, 3, true,  true,  0 },
        { "k7 d1, OC % 4 = 2, + other, * 1/3",     7, 8,  6,  203, 1, true,  false, 0 },
        { "k3 d5, all in other's buffer, tail",     3, 16, 16, 301, 5, true,  true,  1 },
        { "k3 d1, all in other's buffer, sum only", 3, 16, 12, 137, 1, false, false, 1 },
        { "k3 d5 + other as a view (add_inplace)",  3, 16, 16, 301, 5, true,  true,  2 },
    };
    std::vector<float> all;
    for (const Case& cs : cases) {
        std::vector<float> got;
        const double err = run_case(ctx, cs, epi, got);
        std::printf("  %-38s rel err %.2e\n", cs.name, err);
        LOOM_CHECK(err < 1e-5);
        all.insert(all.end(), got.begin(), got.end());
    }

    if (const char* dir = std::getenv("LOOM_TEST_TMPDIR")) {
        char mine[1024], other[1024];
        std::snprintf(mine, sizeof mine, "%s/conv_resblock_tail_%s.f32", dir, epi ? "on" : "off");
        std::snprintf(other, sizeof other, "%s/conv_resblock_tail_%s.f32", dir, epi ? "off" : "on");
        if (FILE* f = std::fopen(mine, "wb")) {
            std::fwrite(all.data(), sizeof(float), all.size(), f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen(other, "rb")) {
            std::vector<float> them(all.size());
            const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
            std::fclose(f);
            LOOM_CHECK(n == them.size());
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i) diff += std::memcmp(&all[i], &them[i], 4) != 0;
            std::printf("  vs the other registration: %zu of %zu floats differ\n", diff, n);
            LOOM_CHECK(diff == 0);
        }
    }
    ggml_free(ctx);
    LOOM_TEST_REPORT_AND_RETURN();
}
