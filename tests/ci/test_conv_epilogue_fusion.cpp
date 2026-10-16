// A convolution's EPILOGUE is folded into it: the RELU after its bias (and residual), and the MUL by a
// time mask after that, or after a depthwise convolution (ADR-079 P4.32f, ggml-0036).
//
// NeMo's encoders (Citrinet-256-ls) end every sub-block with
//
//     y = MUL(RELU(ADD(CONV_2D(w, x), bias)), mask)      mask[t] = t < length, broadcast over channels
//
// and multiply every depthwise convolution's output by the same mask. ggml-0005/0007 already fold the
// bias and a residual into the convolution; the RELU and the MUL were two more passes over the
// activation each time, 1.3 s of an 11 s clip on a Pi Zero W. ggml-0036 applies them as the result is
// stored, and also covers the case the older fusion declined: an output that PARTIALLY overlaps the
// input (Citrinet: 53 of 129 convolutions), computed into the convolution's own tensor and carried to
// the output in one pass.
//
// The epilogue is off by default (it was on for ARMv6 until ggml-0037 made the unfused passes faster);
// GGML_CPU_CONV_EPILOGUE=1 selects it. So this is registered twice, =0 and =1 (ggml reads it once), and pins:
//
//   1. **The numbers**, against a double-precision reference, in both runs.
//   2. **That it fuses**: the RELU's (and the depthwise convolution's) own result is poisoned before the
//      run, and survives exactly when the epilogue is on.
//   3. **Bit-identity**: the epilogue applies the RELU and the mask exactly as the unfused nodes do, so
//      the two runs' outputs must be equal byte for byte. Whichever runs second compares (DEPENDS).

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

// Off unless asked for, on every ISA: ARMv6's default went off once ggml-0037 made the unfused passes
// faster than this epilogue (ADR-079 P4.32f f2 step 3b).
bool epilogue_on() {
    const char* e = std::getenv("GGML_CPU_CONV_EPILOGUE");
    return e != nullptr && std::atoi(e) != 0;
}

bool all_poison(const ggml_tensor* t) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        if (((const float*) t->data)[i] != kPoison) return false;
    }
    return true;
}

void poison(ggml_tensor* t) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float*) t->data)[i] = kPoison;
}

void fill(std::vector<float>& v, float a, float b, int mod) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = a + b * (float) ((int) (i % mod) - mod / 2);
}

// A mask that is 1 up to `len` and 0 after: the shape of NeMo's, with a tail that is really masked.
std::vector<float> mask_of(int64_t ol, int64_t len) {
    std::vector<float> m((size_t) ol);
    for (int64_t t = 0; t < ol; ++t) m[t] = t < len ? 1.0f : 0.0f;
    return m;
}

struct Case {
    const char* name;
    int64_t     kw, ic, oc, il, dil;
    bool        residual;
    bool        mask;
    int64_t     overlap;   // > 0: the output starts this many bytes BEFORE the input (partial overlap)
};

// One CONV_1D as loom lowers it -- CONV_2D, a reshape to [OL, OC], the bias ADD -- then an optional
// residual ADD, the RELU, and an optional mask MUL. Returns the worst error against a double reference
// scaled by the result's peak, and leaves the output in `got`.
double run_conv(ggml_context* ctx, const Case& cs, bool epi, std::vector<float>& got) {
    const int64_t pad = (cs.kw - 1) * cs.dil / 2;
    const int64_t ol  = cs.il + 2 * pad - (cs.kw - 1) * cs.dil;
    std::vector<float> K((size_t) cs.kw * cs.ic * cs.oc), X((size_t) cs.il * cs.ic), B((size_t) cs.oc),
                       R((size_t) ol * cs.oc);
    fill(K, 0.003f, 0.0011f, 53);
    fill(X, 0.01f, 0.013f, 97);
    fill(B, 0.02f, 0.05f, 7);
    fill(R, -0.01f, 0.017f, 31);
    const std::vector<float> M = mask_of(ol, ol - 37);

    ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.kw, 1, cs.ic, cs.oc);
    ggml_tensor* tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, cs.oc, 1);
    ggml_tensor* tr = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ol, cs.oc, 1);
    ggml_tensor* tm = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ol, 1, 1);
    // The input lives inside an arena so the output can be placed partially over it.
    const size_t xbytes = X.size() * sizeof(float), obytes = (size_t) ol * cs.oc * sizeof(float);
    ggml_tensor* arena = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) ((xbytes + obytes + 8192) / 4));
    ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cs.il, 1, cs.ic, 1);
    if (cs.overlap > 0) tx->data = (char*) arena->data + 4096 + obytes;
    std::memcpy(tk->data, K.data(), K.size() * sizeof(float));
    std::memcpy(tx->data, X.data(), xbytes);
    std::memcpy(tb->data, B.data(), B.size() * sizeof(float));
    std::memcpy(tr->data, R.data(), R.size() * sizeof(float));
    std::memcpy(tm->data, M.data(), M.size() * sizeof(float));

    ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, tx, 1, 1, (int) pad, 0, (int) cs.dil, 1);
    ggml_tensor* yb   = ggml_add(ctx, ggml_reshape_3d(ctx, conv, ol, cs.oc, 1), tb);
    ggml_tensor* y    = cs.residual ? ggml_add(ctx, yb, tr) : yb;
    ggml_tensor* relu = ggml_relu(ctx, y);
    ggml_tensor* out  = cs.mask ? ggml_mul(ctx, relu, tm) : relu;
    if (cs.overlap > 0) {
        // What Citrinet's allocator did, 4384-4416 bytes ahead of the input, to the bias ADD -- which is
        // what made ggml-0005/0007 decline it -- and, in place, to every node after it. With the epilogue
        // off the whole chain then runs unfused; with it on, through the convolution's own tensor. The
        // two must agree bit for bit, which they would not if an ADD's result were clear of the input:
        // the older fusion would take conv + bias (+ residual) there, with the sums STARTING at the bias
        // (the ARMv6 tile does), a different rounding.
        out->data = (char*) tx->data - cs.overlap;
        yb->data  = out->data;
        y->data   = out->data;
        LOOM_CHECK((char*) out->data + ggml_nbytes(out) > (char*) tx->data);
    }

    poison(conv);
    if (cs.mask) poison(relu);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    LOOM_CHECK(compute(gf, 2));

    got.assign((const float*) out->data, (const float*) out->data + ggml_nelements(out));
    double worst = 0.0, peak = 0.0;
    for (int64_t o = 0; o < cs.oc; ++o) {
        for (int64_t t = 0; t < ol; ++t) {
            double acc = 0.0;
            for (int64_t c = 0; c < cs.ic; ++c) {
                for (int64_t k = 0; k < cs.kw; ++k) {
                    const int64_t s = t + k * cs.dil - pad;
                    if (s < 0 || s >= cs.il) continue;
                    acc += (double) K[(o * cs.ic + c) * cs.kw + k] * (double) X[c * cs.il + s];
                }
            }
            double v = acc + B[o] + (cs.residual ? R[o * ol + t] : 0.0);
            v = v > 0 ? v : 0.0;
            if (cs.mask) v *= M[t];
            peak  = std::fmax(peak, std::fabs(v));
            worst = std::fmax(worst, std::fabs(v - got[o * ol + t]));
        }
    }

    // Fused, the RELU's own tensor is never written (it is the mask's input; without a mask the RELU
    // IS the output). Unfused, it is. In the partial-overlap case the convolution's tensor is the
    // staging buffer and is written either way, so only the RELU says anything there.
    if (cs.mask) {
        LOOM_CHECK(all_poison(relu) == epi);
        if (all_poison(relu) != epi) {
            std::fprintf(stderr, "  %s: RELU intermediate %s -- the epilogue %s\n", cs.name,
                         epi ? "WAS written" : "was NOT written", epi ? "did not fuse" : "fused while off");
        }
    }
    if (cs.overlap == 0 && !cs.mask) {
        // conv + bias (+ residual) fuse with or without the epilogue (ggml-0005/0007); the RELU joins
        // only with it. Either way the convolution's own result is never written.
        LOOM_CHECK(all_poison(conv));
    }
    return peak > 0 ? worst / peak : worst;
}

// CONV_2D_DW, a reshape to [OL, C], then the mask MUL.
double run_dw(ggml_context* ctx, int64_t k, int64_t ch, int64_t il, bool epi, std::vector<float>& got) {
    const int64_t pad = (k - 1) / 2, ol = il;
    std::vector<float> W((size_t) k * ch), X((size_t) il * ch);
    fill(W, 0.01f, 0.007f, 29);
    fill(X, 0.02f, 0.011f, 89);
    const std::vector<float> M = mask_of(ol, ol - 13);

    ggml_tensor* tw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, 1, 1, ch);
    ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, il, 1, ch, 1);
    ggml_tensor* tm = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ol, 1, 1);
    std::memcpy(tw->data, W.data(), W.size() * sizeof(float));
    std::memcpy(tx->data, X.data(), X.size() * sizeof(float));
    std::memcpy(tm->data, M.data(), M.size() * sizeof(float));

    ggml_tensor* conv = ggml_conv_2d_dw_direct(ctx, tw, tx, 1, 1, (int) pad, 0, 1, 1);
    ggml_tensor* out  = ggml_mul(ctx, ggml_reshape_3d(ctx, conv, ol, ch, 1), tm);
    poison(conv);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    LOOM_CHECK(compute(gf, 2));

    LOOM_CHECK(all_poison(conv) == epi);
    if (all_poison(conv) != epi) {
        std::fprintf(stderr, "  depthwise k=%lld: its result %s\n", (long long) k,
                     epi ? "WAS written -- the mask did not fuse" : "was NOT written with the epilogue off");
    }

    got.assign((const float*) out->data, (const float*) out->data + ggml_nelements(out));
    double worst = 0.0, peak = 0.0;
    for (int64_t c = 0; c < ch; ++c) {
        for (int64_t t = 0; t < ol; ++t) {
            double acc = 0.0;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t s = t + j - pad;
                if (s >= 0 && s < il) acc += (double) W[c * k + j] * (double) X[c * il + s];
            }
            acc *= M[t];
            peak  = std::fmax(peak, std::fabs(acc));
            worst = std::fmax(worst, std::fabs(acc - got[c * ol + t]));
        }
    }
    return peak > 0 ? worst / peak : worst;
}

} // namespace

int main() {
    const bool epi = epilogue_on();
    std::printf("GGML_CPU_CONV_EPILOGUE: %s\n", epi ? "on" : "off");

    ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);

    // Citrinet's shapes in miniature: a 1x1 over a length that is no multiple of the ARMv6 block (32)
    // or tile (4); with a residual; a k > 1 dilated one (the vocoders' tile); OC % 4 != 0, which the
    // ARMv6 tile stages; and the partial overlap at Citrinet's own offsets.
    const Case cases[] = {
        { "1x1 + bias + relu + mask",            1, 48, 32, 551, 1, false, true,  0 },
        { "1x1 + bias + residual + relu + mask", 1, 48, 32, 551, 1, true,  true,  0 },
        { "k5 d2 + bias + relu",                 5, 16, 16, 301, 2, false, false, 0 },
        { "k3 + bias + relu + mask, OC % 4",     3, 12, 6,  203, 1, false, true,  0 },
        { "1x1, out 4416 B before in",           1, 48, 32, 551, 1, false, true,  4416 },
        { "1x1, out 4384 B before in, residual", 1, 48, 32, 551, 1, true,  true,  4384 },
    };
    std::vector<float> all;
    for (const Case& cs : cases) {
        std::vector<float> got;
        const double err = run_conv(ctx, cs, epi, got);
        std::printf("  %-38s rel err %.2e\n", cs.name, err);
        LOOM_CHECK(err < 1e-5);
        all.insert(all.end(), got.begin(), got.end());
    }
    const int64_t dws[][3] = { { 5, 32, 551 }, { 21, 24, 138 }, { 13, 8, 61 } };
    for (const auto& d : dws) {
        std::vector<float> got;
        const double err = run_dw(ctx, d[0], d[1], d[2], epi, got);
        std::printf("  depthwise k=%-3lld + mask              rel err %.2e\n", (long long) d[0], err);
        LOOM_CHECK(err < 1e-5);
        all.insert(all.end(), got.begin(), got.end());
    }

    // 3. byte for byte against the other registration's run
    if (const char* dir = std::getenv("LOOM_TEST_TMPDIR")) {
        char mine[1024], other[1024];
        std::snprintf(mine, sizeof mine, "%s/conv_epilogue_%s.f32", dir, epi ? "on" : "off");
        std::snprintf(other, sizeof other, "%s/conv_epilogue_%s.f32", dir, epi ? "off" : "on");
        if (FILE* f = std::fopen(mine, "wb")) {
            std::fwrite(all.data(), sizeof(float), all.size(), f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen(other, "rb")) {
            std::vector<float> them(all.size());
            const size_t n = std::fread(them.data(), sizeof(float), them.size(), f);
            std::fclose(f);
            LOOM_CHECK(n == them.size());
            const bool same = std::memcmp(them.data(), all.data(), all.size() * sizeof(float)) == 0;
            LOOM_CHECK(same);
            if (!same) std::fprintf(stderr, "  the epilogue on and off differ -- it is not the unfused nodes' arithmetic\n");
            else       std::printf("  bit-identical to the other run\n");
        }
    }

    ggml_free(ctx);
    LOOM_TEST_REPORT_AND_RETURN();
}
