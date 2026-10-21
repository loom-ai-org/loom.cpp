// ADR-079 P4.32f f2 step 2: the vocoders' transposed convolutions as POLYPHASE direct convolutions on the
// ggml-0032 short-vector tile, against ggml's CONV_TRANSPOSE_1D (ggml-0009: a GEMM + an overlap-add).
//
// With J = K / s taps per phase, output phase p of a stride-s transposed convolution is an ordinary
// J-tap convolution over the input padded by J - 1 zeros on each side:
//
//     y[oc][q*s + p] = sum_j' sum_ic w[ic][oc][p + (J-1-j')*s] * xpad[ic][q + j'],   xpad[m] = x[m - (J-1)]
//
// so each phase is one call of ggml_armv6_conv_f32_k<J> per quad of output channels, the bias seeds the
// accumulators (the graph's separate bias ADD pass goes away), and every output is written once (no
// overlap-add read-modify-write). The phases are computed per time block into a scratch and interleaved.
//
// Shapes: amy (piper) -- the three upsampling stages. Arms, median ms over reps, one thread:
//   ggml       CONV_TRANSPOSE_1D alone
//   ggml+b     + the bias ADD node (what the model pays today, before the crop)
//   poly       this driver, bias included
// Build (ARMv6, loom-armv6-build image, against the ggml-only static build):
//   c++ -O2 -std=c++17 -I<ggml>/include bench56.cpp <build>/src/libggml{,-cpu,-base}.a -fopenmp -latomic -o bench56
// Run on the board: ./bench56 [reps] [TB]
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct ggml_armv6_conv_args {
    const float * x;
    const float * w;
    const float * b;
    float * y;
    int32_t ystride;
    int32_t n_ic;
    int32_t xrow;
    int32_t nt;
    int32_t d4;
};
extern "C" void ggml_armv6_conv_f32_k1(const ggml_armv6_conv_args *);
extern "C" void ggml_armv6_conv_f32_k2(const ggml_armv6_conv_args *);
extern "C" void ggml_armv6_conv_f32_k3(const ggml_armv6_conv_args *);
extern "C" void ggml_armv6_conv_f32_k4(const ggml_armv6_conv_args *);
typedef void (*conv_fn)(const ggml_armv6_conv_args *);
static const conv_fn kfn[5] = { nullptr, ggml_armv6_conv_f32_k1, ggml_armv6_conv_f32_k2, ggml_armv6_conv_f32_k3,
                                ggml_armv6_conv_f32_k4 };

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_backend_t g_backend;

static double timed_graph(ggml_context * ctx, ggml_tensor * out, int reps) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(g_backend, gf);
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now();
        ggml_backend_graph_compute(g_backend, gf);
        t.push_back((now() - t0) * 1e3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// The driver. x [Cin][L], w as ggml stores a CONV_TRANSPOSE_1D kernel ([Cin][Cout][K]), bias [Cout] or
// null, y [Cout][(L-1)*s + K].
struct Poly {
    int64_t Cin, Cout, L, s, K, J, Q, Cq, TB, TBx, xrow, icc, nch;
    std::vector<float> wp, win, S, bq;   // packed weights, window, phase scratch, bias per quad (padded)
    conv_fn fn;

    Poly(int64_t cin, int64_t cout, int64_t l, int64_t s_, int64_t k, int64_t tb)
        : Cin(cin), Cout(cout), L(l), s(s_), K(k) {
        J   = K / s;
        Q   = L + J - 1;
        Cq  = (Cout + 3) / 4;
        TB  = tb;
        TBx = TB + 4;
        xrow = TB + J - 1 + 4;                         // + the kernel's lookahead
        const int64_t icc_max = std::max<int64_t>(1, 12 * 1024 / (xrow * 4 + 16) - 1);
        nch = (Cin + icc_max - 1) / icc_max;
        icc = (Cin + nch - 1) / nch;
        fn  = kfn[J];
        wp.assign((size_t) s * Cq * Cin * J * 4 + 8, 0.0f);
        win.assign((size_t) (icc + 1) * xrow + 8, 0.0f);
        S.assign((size_t) Cq * s * 4 * TBx + 8, 0.0f);
        bq.assign((size_t) Cq * 4, 0.0f);
    }

    // [p][quad][ic][j'][r] = w[ic][4 quad + r][p + (J-1-j')*s]
    void pack(const float * w, const float * bias) {
        for (int64_t p = 0; p < s; ++p)
            for (int64_t q = 0; q < Cq; ++q)
                for (int64_t ic = 0; ic < Cin; ++ic)
                    for (int64_t j = 0; j < J; ++j)
                        for (int64_t r = 0; r < 4; ++r) {
                            const int64_t oc = 4 * q + r;
                            wp[((((size_t) p * Cq + q) * Cin + ic) * J + j) * 4 + r] =
                                oc < Cout ? w[(ic * Cout + oc) * K + p + (J - 1 - j) * s] : 0.0f;
                        }
        for (int64_t oc = 0; oc < Cq * 4; ++oc) bq[oc] = bias && oc < Cout ? bias[oc] : 0.0f;
    }

    void run(const float * x, float * y) {
        const int64_t OL = (L - 1) * s + K;
        ggml_armv6_conv_args a;
        a.xrow    = (int32_t) (xrow * 4);
        a.d4      = 4;
        a.ystride = (int32_t) (TBx * 4);
        for (int64_t q0 = 0; q0 < Q; q0 += TB) {
            const int64_t nb = std::min(TB, Q - q0);
            const int64_t nt = (nb + 3) / 4;
            for (int64_t ch = 0; ch < nch; ++ch) {
                const int64_t k0 = ch * icc, kc = std::min(icc, Cin - k0);
                for (int64_t i = 0; i < kc; ++i) {     // row = xpad[k0 + i][q0 ..]
                    float *       d  = win.data() + i * xrow;
                    const float * xr = x + (k0 + i) * L;
                    for (int64_t t = 0; t < 4 * nt + J - 1; ++t) {
                        const int64_t m = q0 + t - (J - 1);
                        d[t] = m >= 0 && m < L ? xr[m] : 0.0f;
                    }
                }
                a.x    = win.data();
                a.n_ic = (int32_t) kc;
                a.nt   = (int32_t) nt;
                for (int64_t qd = 0; qd < Cq; ++qd)
                    for (int64_t p = 0; p < s; ++p) {
                        a.w = wp.data() + (((size_t) p * Cq + qd) * Cin + k0) * J * 4;
                        a.b = ch == 0 ? bq.data() + 4 * qd : nullptr;
                        a.y = S.data() + ((size_t) qd * s + p) * 4 * TBx;
                        fn(&a);
                    }
            }
            // interleave: y[oc][(q0 + t)*s + p] = S[quad][p][r][t]
            for (int64_t oc = 0; oc < Cout; ++oc) {
                const int64_t qd = oc / 4, r = oc % 4;
                float * yr = y + oc * OL + q0 * s;
                const float * sp = S.data() + ((size_t) qd * s * 4 + r) * TBx;
                for (int64_t t = 0; t < nb; ++t)
                    for (int64_t p = 0; p < s; ++p) yr[t * s + p] = sp[(size_t) p * 4 * TBx + t];
            }
        }
    }
};

static void fill(float * d, int64_t n, float a, float b, int seed) {
    for (int64_t i = 0; i < n; ++i) d[i] = a + b * (float) (((i + seed) * 7919) % 113 - 56);
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const int tb   = argc > 2 ? atoi(argv[2]) : 32;
    g_backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g_backend, 1);
    struct Shape { const char * name; int64_t cin, cout, l, s, k; };
    const Shape shapes[] = {
        { "amy up0 192>96 s8 k16", 192, 96, 508, 8, 16 },
        { "amy up1  96>48 s8 k16", 96, 48, 4064, 8, 16 },
        { "amy up2  48>24 s4 k8 ", 48, 24, 32512, 4, 8 },
    };
    printf("reps %d, TB %d, median ms; err = max |.| / max |ref| over 4096 sampled outputs\n", reps, tb);
    printf("%-24s %9s %9s %9s %10s %10s\n", "shape", "ggml", "ggml+b", "poly", "err ggml", "err poly");
    int bad = 0;
    for (const Shape & sh : shapes) {
        const int64_t OL = (sh.l - 1) * sh.s + sh.k;
        ggml_init_params ip = { (size_t) 96 << 20, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.k, sh.cout, sh.cin);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.l, sh.cin);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, sh.cout);
        fill((float *) w->data, ggml_nelements(w), 0.0f, 0.002f, 1);
        fill((float *) x->data, ggml_nelements(x), 0.01f, 0.01f, 2);
        fill((float *) b->data, ggml_nelements(b), 0.0f, 0.01f, 3);
        ggml_tensor * ct  = ggml_conv_transpose_1d(ctx, w, x, (int) sh.s, 0, 1);
        const double t_g  = timed_graph(ctx, ct, reps);
        ggml_tensor * ct2 = ggml_conv_transpose_1d(ctx, w, x, (int) sh.s, 0, 1);
        ggml_tensor * ctb = ggml_add(ctx, ct2, b);
        const double t_gb = timed_graph(ctx, ctb, reps);

        Poly P(sh.cin, sh.cout, sh.l, sh.s, sh.k, tb);
        P.pack((const float *) w->data, (const float *) b->data);
        std::vector<float> y((size_t) sh.cout * OL);
        P.run((const float *) x->data, y.data());
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) {
            const double t0 = now();
            P.run((const float *) x->data, y.data());
            t.push_back((now() - t0) * 1e3);
        }
        std::sort(t.begin(), t.end());
        const double t_p = t[t.size() / 2];

        // double reference on sampled outputs
        const float * W = (const float *) w->data, * X = (const float *) x->data, * B = (const float *) b->data;
        const float * G = (const float *) ctb->data;
        double eg = 0, ep = 0, mx = 0;
        for (int smp = 0; smp < 4096; ++smp) {
            const int64_t oc = (smp * 37) % sh.cout;
            const int64_t o  = smp < 64 ? smp : smp >= 4032 ? OL - 1 - (4095 - smp) : ((int64_t) smp * 104729) % OL;
            double acc = B[oc];
            for (int64_t i = 0; i < sh.l; ++i) {
                const int64_t k = o - i * sh.s;
                if (k < 0 || k >= sh.k) continue;
                for (int64_t ic = 0; ic < sh.cin; ++ic) acc += (double) W[(ic * sh.cout + oc) * sh.k + k] * X[ic * sh.l + i];
            }
            mx = std::max(mx, std::fabs(acc));
            eg = std::max(eg, std::fabs(G[oc * OL + o] - acc));
            ep = std::max(ep, std::fabs(y[oc * OL + o] - acc));
        }
        eg /= mx; ep /= mx;
        printf("%-24s %9.1f %9.1f %9.1f %10.2e %10.2e\n", sh.name, t_g, t_gb, t_p, eg, ep);
        if (!(ep < 1e-5)) ++bad;
        ggml_free(ctx);
    }
    printf(bad ? "FAILED %d\n" : "ok\n", bad);
    return bad ? 1 : 0;
}
