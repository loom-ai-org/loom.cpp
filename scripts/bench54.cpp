// ADR-079 P4.32g scoping: the DFT-basis convolutions against an FFT, on the board.
//
// STFT, Citrinet-256-ls's front end (N 512, hop 160, 1101 frames, 257 bins, hann(400) centred in 512):
//   conv   two CONV_2D direct, [512,1,1,257] each (re, im) -- what the engine runs today
//   conv1  one CONV_2D with the 514 outputs stacked
//   mm     the frames as an overlapping view, one MUL_MAT against the [512 x 514] basis
//   mm2    two-stage Cooley-Tukey as MUL_MATs (N = 16 x 32): the window folded into stage 1, the
//          twiddles into stage 2 (one matrix per k1), three CONTs -- expressible by the exporter alone
//   fft    a C real FFT (complex radix-2 of N/2 + the split) in a ggml_custom_4d
// iSTFT, sanoTTS heart's vocoder tail (N 1024, hop 256, 299 frames, 513 bins):
//   convT  three CONV_TRANSPOSE_1D (re, im, the window-square sum) + ADD + DIV -- today
//   ifft   a C inverse real FFT + window + overlap-add + the same normalisation, in a ggml_custom_4d
//
// Every arm is checked against an f64 direct DFT before it is timed (max |err| / max |ref|).
//
// Build (ARMv6, in the loom-armv6-build image, against a ggml-only static build):
//   c++ -O2 -std=c++17 -I<ggml>/include bench54.cpp <build>/src/libggml{,-cpu,-base}.a -fopenmp -latomic -o bench54
// Run on the board, one thread: ./bench54 [reps]
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_backend_t g_backend;

static double timed(ggml_cgraph * gf, int reps) {
    ggml_backend_graph_compute(g_backend, gf);   // warm
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now();
        ggml_backend_graph_compute(g_backend, gf);
        t.push_back((now() - t0) * 1e3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// ---------------------------------------------------------------------------------------------------
// the C real FFT

struct Plan {
    int n = 0, m = 0;                        // real size, complex size n/2
    std::vector<int> rev;                    // bit reversal of m
    std::vector<float> twr, twi;             // e^{-2 pi i j / m}, j < m/2
    std::vector<float> spr, spi;             // e^{-2 pi i k / n}, k <= m
    std::vector<float> win;                  // analysis or synthesis window, n
};

static Plan make_plan(int n, const std::vector<double> & win) {
    Plan p;
    p.n = n; p.m = n / 2;
    int bits = 0;
    while ((1 << bits) < p.m) ++bits;
    p.rev.resize(p.m);
    for (int i = 0; i < p.m; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) if (i >> b & 1) r |= 1 << (bits - 1 - b);
        p.rev[i] = r;
    }
    for (int j = 0; j < p.m / 2; ++j) {
        p.twr.push_back((float) std::cos(2 * M_PI * j / p.m));
        p.twi.push_back((float) -std::sin(2 * M_PI * j / p.m));
    }
    for (int k = 0; k <= p.m; ++k) {
        p.spr.push_back((float) std::cos(2 * M_PI * k / n));
        p.spi.push_back((float) -std::sin(2 * M_PI * k / n));
    }
    for (double w : win) p.win.push_back((float) w);
    return p;
}

// in-place complex FFT of size m on bit-reversed input; sign = -1 forward (twiddles as stored), +1 inverse
static void cfft(const Plan & p, float * re, float * im, bool inverse) {
    const int m = p.m;
    for (int len = 2; len <= m; len <<= 1) {
        const int half = len >> 1, step = m / len;
        for (int i = 0; i < m; i += len) {
            for (int j = 0; j < half; ++j) {
                const float wr = p.twr[j * step], wi = inverse ? -p.twi[j * step] : p.twi[j * step];
                const int a = i + j, b = a + half;
                const float tr = re[b] * wr - im[b] * wi;
                const float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
}

// x[n] (windowed here) -> X[k], k = 0..n/2
static void rfft(const Plan & p, const float * x, float * Xr, float * Xi, float * zr, float * zi) {
    const int m = p.m;
    for (int i = 0; i < m; ++i) {
        const int r = p.rev[i];
        zr[r] = x[2 * i] * p.win[2 * i];
        zi[r] = x[2 * i + 1] * p.win[2 * i + 1];
    }
    cfft(p, zr, zi, false);
    for (int k = 0; k <= m; ++k) {
        const int a = k % m, b = (m - k) % m;
        const float Ar = zr[a], Ai = zi[a], Br = zr[b], Bi = -zi[b];       // B = conj(Z[m-k])
        const float Er = 0.5f * (Ar + Br), Ei = 0.5f * (Ai + Bi);
        const float Or = 0.5f * (Ai - Bi), Oi = -0.5f * (Ar - Br);         // (A - B) / 2i
        Xr[k] = Er + p.spr[k] * Or - p.spi[k] * Oi;
        Xi[k] = Ei + p.spr[k] * Oi + p.spi[k] * Or;
    }
}

// X[k], k = 0..n/2 -> x[n] = irfft(X) (1/n), times the window
static void irfft(const Plan & p, const float * Xr, const float * Xi, float * x, float * zr, float * zi) {
    const int m = p.m;
    for (int k = 0; k < m; ++k) {
        // Z[k] = E + i O, E = (X[k] + conj X[m-k]) / 2, O = (X[k] - conj X[m-k]) e^{+2 pi i k / n} / 2
        const float Ar = Xr[k], Ai = k == 0 ? 0.0f : Xi[k];
        const float Br = Xr[m - k], Bi = (m - k == m) ? 0.0f : -Xi[m - k];
        const float Er = 0.5f * (Ar + Br), Ei = 0.5f * (Ai + Bi);
        const float Dr = 0.5f * (Ar - Br), Di = 0.5f * (Ai - Bi);
        const float cr = p.spr[k], ci = -p.spi[k];                          // e^{+2 pi i k / n}
        const float Or = Dr * cr - Di * ci, Oi = Dr * ci + Di * cr;
        const int r = p.rev[k];
        zr[r] = Er - Oi;
        zi[r] = Ei + Or;
    }
    cfft(p, zr, zi, true);
    const float s = 1.0f / m;
    for (int i = 0; i < m; ++i) {
        x[2 * i]     = zr[i] * s * p.win[2 * i];
        x[2 * i + 1] = zi[i] * s * p.win[2 * i + 1];
    }
}

// ---------------------------------------------------------------------------------------------------
// custom ops

struct StftArgs { const Plan * plan; int hop; };

// dst [F, 2*B] (re rows then im rows, frames fastest -- the conv arm's [OL, OC] layout); src [L]
static void stft_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const StftArgs * a = (const StftArgs *) ud;
    const Plan & p = *a->plan;
    const ggml_tensor * src = dst->src[0];
    const float * x = (const float *) src->data;
    float * y = (float *) dst->data;
    const int F = (int) dst->ne[0], B = p.m + 1;
    std::vector<float> Xr(B), Xi(B), zr(p.m), zi(p.m);
    for (int f = ith; f < F; f += nth) {
        rfft(p, x + (size_t) f * a->hop, Xr.data(), Xi.data(), zr.data(), zi.data());
        for (int k = 0; k < B; ++k) {
            y[(size_t) k * F + f] = Xr[k];
            y[(size_t) (B + k) * F + f] = Xi[k];
        }
    }
}

struct IstftArgs { const Plan * plan; int hop; };

// dst [L]; src0 re [F, B], src1 im [F, B] (frames fastest, as the conv_transpose input)
static void istft_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    if (ith != 0) return;
    (void) nth;
    const IstftArgs * a = (const IstftArgs *) ud;
    const Plan & p = *a->plan;
    const ggml_tensor * re = dst->src[0], * im = dst->src[1];
    const int F = (int) re->ne[0], B = p.m + 1, n = p.n, hop = a->hop;
    float * y = (float *) dst->data;
    const int L = (int) dst->ne[0];
    std::vector<float> wsum(L, 0.0f), Xr(B), Xi(B), fr(n), zr(p.m), zi(p.m);
    std::fill(y, y + L, 0.0f);
    for (int f = 0; f < F; ++f) {
        for (int k = 0; k < B; ++k) {
            Xr[k] = ((const float *) re->data)[(size_t) k * F + f];
            Xi[k] = ((const float *) im->data)[(size_t) k * F + f];
        }
        irfft(p, Xr.data(), Xi.data(), fr.data(), zr.data(), zi.data());
        float * o = y + (size_t) f * hop;
        float * w = wsum.data() + (size_t) f * hop;
        for (int i = 0; i < n; ++i) { o[i] += fr[i]; w[i] += p.win[i] * p.win[i]; }
    }
    for (int i = 0; i < L; ++i) y[i] /= wsum[i];
}

// ---------------------------------------------------------------------------------------------------

static ggml_tensor * new_f32(ggml_context * ctx, std::vector<int64_t> ne, const std::vector<float> & v) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(), ne.data());
    GGML_ASSERT((size_t) ggml_nelements(t) == v.size());
    memcpy(t->data, v.data(), v.size() * sizeof(float));
    return t;
}

// max |a - ref| / max |ref|
static double rel_err(const float * a, const std::vector<double> & ref) {
    double e = 0, m = 0;
    for (size_t i = 0; i < ref.size(); ++i) { e = std::max(e, std::fabs(a[i] - ref[i])); m = std::max(m, std::fabs(ref[i])); }
    return e / m;
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const char * only = getenv("ONLY");
    auto want = [&](const char * name) { return !only || strstr(only, name); };
    g_backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g_backend, 1);

    ggml_init_params ip = { (size_t) 160 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    srand(1);
    auto rnd = [] { return (float) rand() / RAND_MAX * 2.0f - 1.0f; };

    // ===================================================== STFT
    {
        const int N = 512, hop = 160, F = 1101, B = N / 2 + 1, L = (F - 1) * hop + N;
        std::vector<double> win(N, 0.0);
        for (int i = 0; i < 400; ++i) win[56 + i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / 400);   // torch.hann_window(400), periodic=False? close enough
        std::vector<float> x(L);
        for (auto & v : x) v = rnd();

        // f64 reference, [2B][F]
        std::vector<double> ref((size_t) 2 * B * F);
        for (int f = 0; f < F; ++f)
            for (int k = 0; k < B; ++k) {
                double sr = 0, si = 0;
                for (int n = 0; n < N; ++n) {
                    const double v = x[(size_t) f * hop + n] * win[n];
                    const long ph = ((long) k * n) % N;
                    sr += v * std::cos(2 * M_PI * ph / N);
                    si -= v * std::sin(2 * M_PI * ph / N);
                }
                ref[(size_t) k * F + f] = sr;
                ref[(size_t) (B + k) * F + f] = si;
            }

        // the windowed basis, [K=512, OC] per half
        std::vector<float> bre((size_t) N * B), bim((size_t) N * B);
        for (int k = 0; k < B; ++k)
            for (int n = 0; n < N; ++n) {
                const long ph = ((long) k * n) % N;
                bre[(size_t) k * N + n] = (float) (win[n] * std::cos(2 * M_PI * ph / N));
                bim[(size_t) k * N + n] = (float) (-win[n] * std::sin(2 * M_PI * ph / N));
            }
        std::vector<float> both(bre);
        both.insert(both.end(), bim.begin(), bim.end());

        ggml_tensor * xs = new_f32(ctx, {L}, x);
        ggml_tensor * x4 = ggml_reshape_4d(ctx, xs, L, 1, 1, 1);
        std::vector<float> out((size_t) 2 * B * F);
        const double gmac_direct = (double) F * N * 2 * B / 1e9;

        auto report = [&](const char * name, ggml_tensor * t, double ms, bool frames_fastest) {
            // bring to [2B][F]
            const float * d = (const float *) t->data;
            if (frames_fastest) memcpy(out.data(), d, out.size() * sizeof(float));
            else for (int f = 0; f < F; ++f) for (int c = 0; c < 2 * B; ++c) out[(size_t) c * F + f] = d[(size_t) f * 2 * B + c];
            printf("stft  %-6s %9.2f ms   err %.2e   (%.3f GMAC as a direct DFT)\n", name, ms, rel_err(out.data(), ref), gmac_direct);
            fflush(stdout);
        };

        if (want("conv")) {
            ggml_tensor * kr = ggml_reshape_4d(ctx, new_f32(ctx, {(int64_t) bre.size()}, bre), N, 1, 1, B);
            ggml_tensor * ki = ggml_reshape_4d(ctx, new_f32(ctx, {(int64_t) bim.size()}, bim), N, 1, 1, B);
            ggml_tensor * cr = ggml_conv_2d_direct(ctx, kr, x4, hop, 1, 0, 0, 1, 1);   // [F,1,B,1]
            ggml_tensor * ci = ggml_conv_2d_direct(ctx, ki, x4, hop, 1, 0, 0, 1, 1);
            ggml_tensor * o = ggml_concat(ctx, cr, ci, 2);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            report("conv", o, timed(gf, reps), true);

            ggml_tensor * kb = ggml_reshape_4d(ctx, new_f32(ctx, {(int64_t) both.size()}, both), N, 1, 1, 2 * B);
            ggml_tensor * c1 = ggml_conv_2d_direct(ctx, kb, x4, hop, 1, 0, 0, 1, 1);
            ggml_cgraph * g1 = ggml_new_graph(ctx);
            ggml_build_forward_expand(g1, c1);
            report("conv1", c1, timed(g1, reps), true);
        }
        if (want("mm")) {
            // frames: an overlapping view [N, F] with row stride hop
            // ggml rejects a view whose dense size exceeds its source, so the overlap is set after creation
            // (the exporter cannot do this: its framing would be an IM2COL, a copy the size of the CONT below)
            ggml_tensor * fr = ggml_view_1d(ctx, xs, N, 0);
            fr->ne[1] = F; fr->nb[1] = hop * sizeof(float); fr->nb[2] = fr->nb[3] = fr->nb[1] * F;
            ggml_tensor * W = new_f32(ctx, {N, 2 * B}, both);
            ggml_tensor * o = ggml_mul_mat(ctx, fr, W);          // [F, 2B]: frames fastest
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            report("mm", o, timed(gf, reps), true);
        }
        if (want("mm2")) {
            // n = N2*n1 + n2, k = k1 + N1*k2; X[k] = sum_n2 W_N^{n2 k} Y_n2[k1], Y_n2[k1] = sum_n1 w x W_N1^{n1 k1}
            const int N1 = 16, N2 = 32, K2 = N2 / 2 + 1;
            // frames as [n1, F, n2]: element (n1, f, n2) = x[f*hop + N2*n1 + n2]
            ggml_tensor * fv = ggml_view_1d(ctx, xs, N1, 0);
            fv->ne[1] = F; fv->ne[2] = N2;
            fv->nb[0] = N2 * sizeof(float); fv->nb[1] = hop * sizeof(float); fv->nb[2] = sizeof(float);
            fv->nb[3] = (size_t) L * sizeof(float);
            ggml_tensor * fc = ggml_cont(ctx, fv);                                       // [N1, F, N2]
            // stage 1, one matrix per n2: [N1 (n1), 2*N1 (ri, k1), N2]
            std::vector<float> s1((size_t) N1 * 2 * N1 * N2);
            for (int n2 = 0; n2 < N2; ++n2)
                for (int ri = 0; ri < 2; ++ri)
                    for (int k1 = 0; k1 < N1; ++k1)
                        for (int n1 = 0; n1 < N1; ++n1) {
                            const int n = N2 * n1 + n2;
                            const double a = 2 * M_PI * ((n1 * k1) % N1) / N1;
                            s1[(((size_t) n2 * 2 * N1) + ri * N1 + k1) * N1 + n1] =
                                (float) (win[n] * (ri == 0 ? std::cos(a) : -std::sin(a)));
                        }
            ggml_tensor * S1 = new_f32(ctx, {N1, 2 * N1, N2}, s1);
            ggml_tensor * y = ggml_mul_mat(ctx, S1, fc);                                 // [2N1, F, N2]
            // -> Z [N2 (n2), 2 (ri), F, N1 (k1)]
            ggml_tensor * y4 = ggml_reshape_4d(ctx, y, N1, 2, F, N2);                    // [k1, ri, F, n2]
            ggml_tensor * z = ggml_cont(ctx, ggml_permute(ctx, y4, 3, 1, 2, 0));          // [n2, ri, F, k1]
            z = ggml_reshape_3d(ctx, z, 2 * N2, F, N1);
            // stage 2, one matrix per k1: [2*N2 (ri_in, n2), 2*K2 (ri_out, k2), N1]
            std::vector<float> s2((size_t) 2 * N2 * 2 * K2 * N1);
            for (int k1 = 0; k1 < N1; ++k1)
                for (int ro = 0; ro < 2; ++ro)
                    for (int k2 = 0; k2 < K2; ++k2)
                        for (int ri = 0; ri < 2; ++ri)
                            for (int n2 = 0; n2 < N2; ++n2) {
                                const int k = k1 + N1 * k2;
                                const double a = 2 * M_PI * ((long) n2 * k % N) / N;
                                const double c = std::cos(a), s = std::sin(a);
                                // (Zr + i Zi)(c - i s) = Zr c + Zi s + i (Zi c - Zr s)
                                const double v = ro == 0 ? (ri == 0 ? c : s) : (ri == 0 ? -s : c);
                                s2[((((size_t) k1 * 2 * K2) + ro * K2 + k2) * 2 * N2) + ri * N2 + n2] = (float) v;
                            }
            ggml_tensor * S2 = new_f32(ctx, {2 * N2, 2 * K2, N1}, s2);
            ggml_tensor * X = ggml_mul_mat(ctx, S2, z);                                  // [2K2 (ro, k2), F, N1 (k1)]
            // -> [k1, k2, F, ro] then the first B bins of each (ro, F) row
            ggml_tensor * X4 = ggml_reshape_4d(ctx, X, K2, 2, F, N1);                    // [k2, ro, F, k1]
            ggml_tensor * Xn = ggml_cont(ctx, ggml_permute(ctx, X4, 1, 3, 2, 0));         // [k1, k2, F, ro]
            Xn = ggml_reshape_3d(ctx, Xn, N1 * K2, F, 2);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Xn);
            const double ms = timed(gf, reps);
            const float * d = (const float *) Xn->data;
            for (int ro = 0; ro < 2; ++ro)
                for (int f = 0; f < F; ++f)
                    for (int k = 0; k < B; ++k)
                        out[(size_t) (ro * B + k) * F + f] = d[((size_t) ro * F + f) * N1 * K2 + k];
            printf("stft  %-6s %9.2f ms   err %.2e   (%.3f GMAC)\n", "mm2", ms, rel_err(out.data(), ref),
                   (double) F * (N1 * 2 * N1 * N2 + 2 * N2 * 2 * K2 * N1) / 1e9);
            fflush(stdout);
        }
        if (want("fft")) {
            static Plan plan = make_plan(N, win);
            static StftArgs args{&plan, hop};
            ggml_tensor * args_[] = {xs};
            ggml_tensor * o = ggml_custom_4d(ctx, GGML_TYPE_F32, F, 2 * B, 1, 1, args_, 1, stft_op, 1, &args);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            report("fft", o, timed(gf, reps), true);
        }
    }

    // ===================================================== iSTFT
    if (want("ist")) {
        const int N = 1024, hop = 256, F = 299, B = N / 2 + 1, L = (F - 1) * hop + N;
        std::vector<double> win(N);
        for (int i = 0; i < N; ++i) win[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / N);
        std::vector<float> re((size_t) B * F), im((size_t) B * F);
        for (auto & v : re) v = rnd();
        for (auto & v : im) v = rnd();

        // f64 reference: irfft per frame (Hermitian, imag of DC/Nyquist ignored), window, OLA, / wsum
        std::vector<double> ref(L, 0.0), ws(L, 0.0);
        for (int f = 0; f < F; ++f)
            for (int n = 0; n < N; ++n) {
                double s = 0;
                for (int k = 0; k < B; ++k) {
                    const double wk = (k == 0 || k == N / 2) ? 1.0 / N : 2.0 / N;
                    const long ph = ((long) k * n) % N;
                    s += wk * (re[(size_t) k * F + f] * std::cos(2 * M_PI * ph / N) - im[(size_t) k * F + f] * std::sin(2 * M_PI * ph / N));
                }
                ref[(size_t) f * hop + n] += s * win[n];
                ws[(size_t) f * hop + n] += win[n] * win[n];
            }
        for (int i = 0; i < L; ++i) ref[i] /= ws[i];
        // the center crop leaves the edges (wsum ~ 0 there) out of both the check and the model
        std::vector<double> refc(ref.begin() + N / 2, ref.end() - N / 2);

        ggml_tensor * tre = new_f32(ctx, {F, B}, re);
        ggml_tensor * tim = new_f32(ctx, {F, B}, im);

        if (want("convT")) {
            // the ISTFT module's kernels: [K, OC=1, IC=B]
            std::vector<float> kc((size_t) N * B), ks((size_t) N * B), kw(N), ones(F, 1.0f);
            for (int k = 0; k < B; ++k)
                for (int n = 0; n < N; ++n) {
                    const double wk = (k == 0 || k == N / 2) ? 1.0 / N : 2.0 / N;
                    const long ph = ((long) k * n) % N;
                    kc[(size_t) k * N + n] = (float) (wk * win[n] * std::cos(2 * M_PI * ph / N));
                    ks[(size_t) k * N + n] = (float) (-wk * win[n] * std::sin(2 * M_PI * ph / N));
                }
            for (int n = 0; n < N; ++n) kw[n] = (float) (win[n] * win[n]);
            ggml_tensor * Kc = new_f32(ctx, {N, 1, B}, kc), * Ks = new_f32(ctx, {N, 1, B}, ks);
            ggml_tensor * Kw = new_f32(ctx, {N, 1, 1}, kw);
            ggml_tensor * on = new_f32(ctx, {F, 1}, ones);
            ggml_tensor * a = ggml_conv_transpose_1d(ctx, Kc, tre, hop, 0, 1);
            ggml_tensor * b = ggml_conv_transpose_1d(ctx, Ks, tim, hop, 0, 1);
            ggml_tensor * w = ggml_conv_transpose_1d(ctx, Kw, on, hop, 0, 1);
            ggml_tensor * o = ggml_div(ctx, ggml_add(ctx, a, b), w);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            const double ms = timed(gf, reps);
            printf("istft %-6s %9.2f ms   err %.2e   (%.3f GMAC as a direct DFT)\n", "convT", ms,
                   rel_err((const float *) o->data + N / 2, refc), (double) F * N * 2 * B / 1e9);
            fflush(stdout);
        }
        if (want("ifft")) {
            static Plan plan = make_plan(N, win);
            static IstftArgs args{&plan, hop};
            ggml_tensor * args_[] = {tre, tim};
            ggml_tensor * o = ggml_custom_4d(ctx, GGML_TYPE_F32, L, 1, 1, 1, args_, 2, istft_op, 1, &args);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            const double ms = timed(gf, reps);
            printf("istft %-6s %9.2f ms   err %.2e\n", "ifft", ms, rel_err((const float *) o->data + N / 2, refc));
            fflush(stdout);
        }
    }
    ggml_free(ctx);
    ggml_backend_free(g_backend);
    return 0;
}
