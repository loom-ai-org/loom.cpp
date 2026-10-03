// The direct depthwise convolution and what is fused around it.
//
// `cmake/patches/ggml-0021-conv2d-dw-interior-vec.patch` splits each output row of ggml's direct
// depthwise kernel into a SIMD interior (every tap in bounds) and scalar edges; `ggml-0022` fuses the
// causal block CONV_1D_DW lowers to -- PAD(left) -> CONV_2D_DW -> (reshape) -> ADD(bias) -> RELU -- into
// one pass of that kernel. Like test_conv_bias_fusion, this pins:
//
//   1. **The numbers**, against a double-precision reference, for the fused chain and for the bare
//      kernel at stride 1 and 2 with F32 and F16 weights -- the shapes chosen so that every row has a
//      left edge, an interior and a right edge, and one row has NO interior at all.
//   2. **That it fuses.** The PAD's and the convolution's own result tensors are poisoned; a fused run
//      never writes either, an unfused one writes both. Registered twice, with and without
//      `GGML_CPU_DISABLE_FUSION` (ggml reads it once per process), each run asserting its direction.
//   3. **Fused and unfused agree BIT for bit**: the fused epilogue adds the bias and applies the RELU
//      to a finished row in the unfused order. Compared through a file the other registration leaves.
//   4. **An aliased destination**: the graph allocator may give the RELU the block of the PAD's input
//      when that input is dead afterwards, and the taps read BEHIND the output position, so writing a
//      row in place without copying it out first would read outputs as inputs.
//
// Every tensor lives in its own context storage (no graph allocator), so the poison checks mean what
// they say.

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

// out[c][o] = sum_k w[c][k] * x[c][o*s + k*d - pad], zero outside [0, il)
double reference(const std::vector<float>& W, const std::vector<float>& X, int64_t il, int64_t k,
                 int64_t c, int64_t o, int64_t s, int64_t d, int64_t pad) {
    double acc = 0.0;
    for (int64_t kx = 0; kx < k; ++kx) {
        const int64_t sx = o * s + kx * d - pad;
        if (sx < 0 || sx >= il) continue;
        acc += (double) W[c * k + kx] * (double) X[c * il + sx];
    }
    return acc;
}

void fill(std::vector<float>& v, float a, float b, int m) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = a + b * (float) ((int) (i % m) - m / 2);
}

bool all_poison(const ggml_tensor* t) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i)
        if (((const float*) t->data)[i] != kPoison) return false;
    return true;
}

} // namespace

int main() {
    const bool fusion_disabled = [] {
        const char* e = std::getenv("GGML_CPU_DISABLE_FUSION");
        return e != nullptr && std::atoi(e) == 1;
    }();

    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);

    // WakeHuBERT's block, smaller: K=5 at dilation 2 with the causal left pad (K-1)*d, a length that
    // is not a multiple of any SIMD width.
    constexpr int64_t K = 5, D = 2, C = 6, IL = 251, LP = (K - 1) * D;
    std::vector<float> W((size_t) K * C), X((size_t) IL * C), B((size_t) C);
    fill(W, 0.01f, 0.013f, 7);
    fill(X, 0.02f, 0.011f, 97);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 0.05f * (float) ((int) i - 2);   // some rows go negative

    // The fused chain, exactly as src/ops/primitives_conv.cpp and the PAD_1D/ADD/RELU primitives emit
    // it: a [IL, C] activation padded on ne0, reshaped to [W, 1, C, 1], convolved, reshaped back to
    // [OL, C, 1], plus a [1, C, 1] bias, then RELU.
    auto build_chain = [&](ggml_tensor*& pad, ggml_tensor*& conv, ggml_tensor*& x) {
        ggml_tensor* w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, 1, 1, C);
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, IL, C, 1);
        ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, C, 1);
        std::memcpy(w->data, W.data(), W.size() * sizeof(float));
        std::memcpy(x->data, X.data(), X.size() * sizeof(float));
        std::memcpy(b->data, B.data(), B.size() * sizeof(float));
        pad = ggml_pad_ext(ctx, x, (int) LP, 0, 0, 0, 0, 0, 0, 0);
        ggml_tensor* p4 = ggml_reshape_4d(ctx, pad, pad->ne[0], 1, C, 1);
        conv = ggml_conv_2d_dw_direct(ctx, w, p4, 1, 1, 0, 0, (int) D, 1);
        ggml_tensor* add = ggml_add(ctx, ggml_reshape_3d(ctx, conv, conv->ne[0], C, 1), b);
        return ggml_relu(ctx, add);
    };
    auto worst_error = [&](const ggml_tensor* out) {
        double worst = 0.0;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t o = 0; o < IL; ++o) {
                double ref = reference(W, X, IL, K, c, o, 1, D, LP) + (double) B[c];
                ref = ref > 0.0 ? ref : 0.0;
                const double got = (double) ((const float*) out->data)[c * IL + o];
                worst = std::fmax(worst, std::fabs(got - ref) / (std::fabs(ref) + 1e-3));
            }
        return worst;
    };

    // 1-3. the chain: numbers, fused-or-not, and bit-identity across the two registrations
    {
        ggml_tensor *pad, *conv, *x;
        ggml_tensor* out = build_chain(pad, conv, x);
        LOOM_CHECK(out->ne[0] == IL);
        for (ggml_tensor* t : {pad, conv}) for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float*) t->data)[i] = kPoison;

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        LOOM_CHECK(compute(gf, 2));

        const double worst = worst_error(out);
        LOOM_CHECK(worst < 1e-5);
        if (worst >= 1e-5) std::fprintf(stderr, "  fused chain: worst relative error %.3e\n", worst);

        const bool untouched = all_poison(pad) && all_poison(conv);
        LOOM_CHECK(untouched != fusion_disabled);
        if (!fusion_disabled && !untouched) {
            std::fprintf(stderr, "PAD -> CONV_2D_DW -> ADD -> RELU was NOT fused: ggml wrote an "
                                 "intermediate. ggml_cpu_conv_2d_dw_fusion (ggml-0022) no longer matches "
                                 "what the engine emits\n");
        }

        if (const char* dir = std::getenv("LOOM_TEST_TMPDIR")) {
            char path[1024], other[1024];
            std::snprintf(path, sizeof path, "%s/conv_dw_fusion_%s.f32", dir, fusion_disabled ? "unfused" : "fused");
            std::snprintf(other, sizeof other, "%s/conv_dw_fusion_%s.f32", dir, fusion_disabled ? "fused" : "unfused");
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(out->data, 1, ggml_nbytes(out), f); std::fclose(f); }
            if (FILE* f = std::fopen(other, "rb")) {
                std::vector<float> them(ggml_nelements(out));
                const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
                std::fclose(f);
                LOOM_CHECK(n == them.size());
                LOOM_CHECK(std::memcmp(them.data(), out->data, ggml_nbytes(out)) == 0);
            }
        }
    }

    // 4. the fused output written over the PAD's own input
    {
        ggml_tensor *pad, *conv, *x;
        ggml_tensor* out = build_chain(pad, conv, x);
        LOOM_CHECK(ggml_nbytes(out) == ggml_nbytes(x));
        out->data = x->data;                           // the aliasing a graph allocator can produce
        for (ggml_tensor* t : {pad, conv}) for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float*) t->data)[i] = kPoison;

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        LOOM_CHECK(compute(gf, 2));

        const double worst = worst_error(out);
        LOOM_CHECK(worst < 1e-5);
        if (worst >= 1e-5) {
            std::fprintf(stderr, "  aliased destination: worst relative error %.3e -- a row was written "
                                 "over input its later outputs still read\n", worst);
        }
        // only meaningful fused: unfused, the PAD has already copied x out before anything writes it
        LOOM_CHECK((all_poison(pad) && all_poison(conv)) != fusion_disabled);
    }

    // 5. the bare kernel (nothing to fuse): symmetric pad, stride 1 and 2, F32 and F16 weights, and a
    //    row so short against the kernel's reach that it has no interior.
    struct Case { int64_t il, k, s, d, pad; ggml_type wt; };
    for (const Case& cs : { Case{251, 5, 1, 3, 6, GGML_TYPE_F32}, Case{251, 5, 2, 1, 2, GGML_TYPE_F32},
                            Case{97, 3, 1, 1, 1, GGML_TYPE_F16}, Case{9, 5, 1, 4, 8, GGML_TYPE_F32} }) {
        std::vector<float> w((size_t) cs.k * C), xin((size_t) cs.il * C);
        fill(w, 0.03f, 0.017f, 5);
        fill(xin, -0.01f, 0.009f, 31);
        ggml_tensor* tw = ggml_new_tensor_4d(ctx, cs.wt, cs.k, 1, 1, C);
        if (cs.wt == GGML_TYPE_F16) {
            for (size_t i = 0; i < w.size(); ++i) {
                ((ggml_fp16_t*) tw->data)[i] = ggml_fp32_to_fp16(w[i]);
                w[i] = ggml_fp16_to_fp32(((ggml_fp16_t*) tw->data)[i]);   // the reference sees what ran
            }
        } else {
            std::memcpy(tw->data, w.data(), w.size() * sizeof(float));
        }
        ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.il, 1, C, 1);
        std::memcpy(tx->data, xin.data(), xin.size() * sizeof(float));
        ggml_tensor* y = ggml_conv_2d_dw_direct(ctx, tw, tx, (int) cs.s, 1, (int) cs.pad, 0, (int) cs.d, 1);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        LOOM_CHECK(compute(gf, 2));

        const int64_t ol = y->ne[0];
        LOOM_CHECK(ol == (cs.il + 2 * cs.pad - cs.d * (cs.k - 1) - 1) / cs.s + 1);
        double worst = 0.0;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t o = 0; o < ol; ++o) {
                const double ref = reference(w, xin, cs.il, cs.k, c, o, cs.s, cs.d, cs.pad);
                const double got = (double) ((const float*) y->data)[c * ol + o];
                worst = std::fmax(worst, std::fabs(got - ref) / (std::fabs(ref) + 1e-3));
            }
        LOOM_CHECK(worst < 1e-5);
        if (worst >= 1e-5) {
            std::fprintf(stderr, "  bare dw conv il=%lld k=%lld s=%lld d=%lld pad=%lld: worst relative "
                                 "error %.3e\n", (long long) cs.il, (long long) cs.k, (long long) cs.s,
                                 (long long) cs.d, (long long) cs.pad, worst);
        }
    }

    ggml_free(ctx);
    LOOM_TEST_REPORT_AND_RETURN();
}
