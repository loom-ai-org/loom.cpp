// 1-D convolutions (CONV_2D with one row) over the shapes and fusions the vocoders run, against a double
// reference. Every CPU build takes its own paths through here; the ARMv6 build is the one this was
// written for: ggml-0032 (ADR-079 P4.32b) sends every 1-D stride-1 convolution with k = 2..11 to the
// VFP11 short-vector tiles, which have their own blocking (blocks of TB, a last block absorbing a
// remainder under a tile, an overlapping tile for a length not a multiple of 4), a padded last channel
// quad, and three ways out (straight, staged, staged and deferred when the output overlaps the input).
// The shapes below reach each of those.
//
// Three variants per shape:
//   plain     conv alone, every tensor in the context
//   fused     LEAKY_RELU -> conv -> + bias -> + residual, which ggml-0005/0007 fuse into one node
//   resblock  the same through a graph allocator, with the block's input as the residual: the
//             allocator gives the fused result the input's own memory (checked, so the case is real)
//
// ARMv6 is checked in qemu: loom-armv6-build:bookworm builds this directory, and ctest runs it there.
#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr float kSlope = 0.1f;

struct Shape { int64_t ic, oc, k, d, il, pad; };

// The vocoders' (amy, de051, VITS, sanoTTS heart) and the edges of the tiling.
const Shape kShapes[] = {
    {8, 8, 3, 1, 251, 1},       {24, 24, 7, 1, 1000, 3},   {24, 24, 7, 3, 700, 9},   {24, 24, 7, 5, 513, 15},
    {12, 12, 5, 1, 67, 2},      {32, 1, 7, 1, 300, 3},     {3, 6, 2, 1, 64, 0},      {1, 4, 11, 2, 90, 3},
    {64, 64, 7, 1, 130, 3},     {100, 192, 7, 1, 99, 3},   {4, 4, 3, 1, 5, 1},       {96, 96, 5, 1, 299, 2},
    {48, 48, 3, 1, 4, 1},       {16, 16, 9, 1, 200, 4},    {20, 20, 11, 1, 77, 5},   {768, 12, 3, 1, 96, 0},
};

bool compute(ggml_backend_t backend, ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

struct Data { std::vector<float> K, X, B, R; };

Data make(const Shape& s, unsigned seed) {
    Data d;
    d.K.resize((size_t) s.k * s.ic * s.oc); d.X.resize((size_t) s.il * s.ic); d.B.resize(s.oc);
    unsigned st = seed;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };
    for (float& v : d.K) v = u() * 0.2f;
    for (float& v : d.X) v = u();
    for (float& v : d.B) v = u() * 0.5f;
    return d;
}

// Output (o, t) with the input first put through `act` (LEAKY_RELU with kSlope when true); returns the
// value and the scale it is judged against (|bias| + sum |w x| + |res|).
double ref(const Shape& s, const Data& d, bool act, const float* res, int64_t ol, int64_t o, int64_t t, double* scale) {
    double acc = d.B.empty() ? 0.0 : d.B[o], mag = std::fabs(acc);
    for (int64_t c = 0; c < s.ic; ++c)
        for (int64_t j = 0; j < s.k; ++j) {
            const int64_t p = t + j * s.d - s.pad;
            if (p < 0 || p >= s.il) continue;
            double x = d.X[c * s.il + p];
            if (act && x < 0) x *= (double) kSlope;
            const double m = (double) d.K[(o * s.ic + c) * s.k + j] * x;
            acc += m; mag += std::fabs(m);
        }
    if (res) { acc += res[o * ol + t]; mag += std::fabs(res[o * ol + t]); }
    *scale = mag;
    return acc;
}

double worst(const Shape& s, const Data& d, bool act, const float* res, const float* got, int64_t ol) {
    double w = 0;
    for (int64_t o = 0; o < s.oc; ++o)
        for (int64_t t = 0; t < ol; ++t) {
            double scale;
            const double r = ref(s, d, act, res, ol, o, t, &scale);
            w = std::max(w, std::fabs(got[o * ol + t] - r) / (scale + 1e-30));
        }
    return w;
}

} // namespace

int main() {
    ggml_backend_t backend = loom_test::cpu_backend();
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();
    int aliased = 0, resblocks = 0;
    unsigned seed = 7;

    for (const Shape& s : kShapes) {
        const int64_t ol = s.il + 2 * s.pad - (s.k - 1) * s.d;
        for (int threads : {1, 2}) {
            Data d = make(s, ++seed);

            // plain and fused, in the context
            for (int fused = 0; fused < 2; ++fused) {
                ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, false };
                ggml_context* ctx = ggml_init(ip);
                ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, s.ic, s.oc);
                ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.il, 1, s.ic, 1);
                ggml_tensor* tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, s.oc);
                ggml_tensor* tr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ol, s.oc);
                std::memcpy(tk->data, d.K.data(), ggml_nbytes(tk));
                std::memcpy(tx->data, d.X.data(), ggml_nbytes(tx));
                std::memcpy(tb->data, d.B.data(), ggml_nbytes(tb));
                d.R.resize((size_t) ol * s.oc);
                for (size_t i = 0; i < d.R.size(); ++i) d.R[i] = 0.25f * std::sin(0.37f * (float) i);
                std::memcpy(tr->data, d.R.data(), ggml_nbytes(tr));
                ggml_tensor* in = fused ? ggml_leaky_relu(ctx, tx, kSlope, false) : tx;
                ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, in, 1, 1, (int) s.pad, 0, (int) s.d, 1);
                ggml_tensor* out = conv;
                if (fused) {
                    out = ggml_add(ctx, ggml_reshape_2d(ctx, conv, ol, s.oc), tb);
                    out = ggml_add(ctx, out, tr);
                }
                ggml_cgraph* gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                LOOM_CHECK(compute(backend, gf, threads));
                Data dd = d;
                if (!fused) std::fill(dd.B.begin(), dd.B.end(), 0.0f);
                const double w = worst(s, dd, fused, fused ? d.R.data() : nullptr, (const float*) out->data, ol);
                LOOM_CHECK(w < 1e-5);
                if (w >= 1e-5)
                    std::fprintf(stderr, "  %s ic %lld oc %lld k %lld d %lld il %lld pad %lld, %d threads: %.3e\n",
                                 fused ? "fused" : "plain", (long long) s.ic, (long long) s.oc, (long long) s.k,
                                 (long long) s.d, (long long) s.il, (long long) s.pad, threads, w);
                ggml_free(ctx);
            }

            // resblock, through a graph allocator: h -> leaky -> conv -> + bias -> + h
            if (s.ic != s.oc || ol != s.il) continue;
            ++resblocks;
            ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, s.ic, s.oc);
            ggml_tensor* tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.il, s.ic);
            ggml_tensor* tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, s.oc);
            ggml_set_input(tk); ggml_set_input(tx); ggml_set_input(tb);
            ggml_tensor* h = ggml_scale(ctx, tx, 1.0f);          // an intermediate, so its memory is reusable
            ggml_tensor* a = ggml_leaky_relu(ctx, h, kSlope, false);
            ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, ggml_reshape_4d(ctx, a, s.il, 1, s.ic, 1), 1, 1,
                                                    (int) s.pad, 0, (int) s.d, 1);
            ggml_tensor* out = ggml_add(ctx, ggml_reshape_2d(ctx, conv, ol, s.oc), tb);
            out = ggml_add(ctx, h, out);                        // h first: the allocator reuses src[0] in place
            ggml_set_output(out);
            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            LOOM_CHECK(ggml_gallocr_alloc_graph(ga, gf));
            ggml_backend_tensor_set(tk, d.K.data(), 0, ggml_nbytes(tk));
            ggml_backend_tensor_set(tx, d.X.data(), 0, ggml_nbytes(tx));
            ggml_backend_tensor_set(tb, d.B.data(), 0, ggml_nbytes(tb));
            if (out->data == h->data) ++aliased;
            LOOM_CHECK(compute(backend, gf, threads));
            std::vector<float> got((size_t) ol * s.oc);
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            const double w = worst(s, d, true, d.X.data(), got.data(), ol);    // h == X
            LOOM_CHECK(w < 1e-5);
            if (w >= 1e-5)
                std::fprintf(stderr, "  resblock ic %lld k %lld d %lld il %lld, %d threads%s: %.3e\n", (long long) s.ic,
                             (long long) s.k, (long long) s.d, (long long) s.il, threads,
                             out->data == h->data ? " (output over input)" : "", w);
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
    }
    std::printf("resblocks %d, output placed over the input in %d\n", resblocks, aliased);
    LOOM_CHECK(aliased > 0);    // otherwise the deferred flush went untested
    LOOM_TEST_REPORT_AND_RETURN();
}
