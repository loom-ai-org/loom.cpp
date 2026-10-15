// ADR-079 P4.32e: CITRINET-256'S DEPTHWISE CONVOLUTIONS ON THE VFP11'S SHORT VECTORS.
//
// bench43's jfk schedule (Citrinet-256-ls, 107 depthwise calls, K = 5..41), stride-1 calls only (the
// three stride-2 ones stay on ggml's path), one thread, RunFast. Arms:
//
//   r   y[t] = 0 + w[0] x[t] + w[1] x[t+1] + ..., in k order: ggml's sweep (ggml-0021) on a core without
//       SIMD, over prepared padded rows
//   c   bench43's f32 arm: 8 outputs a register block, the window renamed by an 8-tap unroll (= r's bits)
//   s   gen_armv6_dw_f32.py's kernel over the same prepared rows, PLD on the next row
//   n   s with the PLD pointed at a line already in cache
//   e   what ops.cpp runs: per channel, the source row copied into one padded scratch (the copy timed),
//       the kernel PLDs the next channel's source row
//   g   ggml_conv_2d_dw_direct through ggml (ggml-0035) as the engine runs it: tensors in a CPU backend
//       buffer, one thread, ggml_backend_graph_compute. Built only with -DBENCH52_GGML. ADR-079's gate 2
//       is g against e. (bench43's ggml arm is not that: ggml_graph_compute_with_ctx plans and builds
//       a threadpool per call, ~0.35 ms each, and its context aligns tensors to 4 bytes.)
//   f   ops.cpp's per-channel code copied verbatim (int64 index math, std::fill/std::copy, the scratch
//       allocated per call), on e's arrays: to tell a cost in that code from one in ggml around it
//   h   f with the channel index counted instead of taken modulo (an int64 % is a library call here)
//   G   g with its tensors placed on e's own arrays (ggml_backend_cpu_buffer_from_ptr): the same code as
//       g at e's placement, to tell the path's cost from the addresses'
//
// s, n, e and g are checked bit for bit against the kernel's contract (even taps + odd taps, each in k
// order), and the contract against r for the largest |diff| / max |y|.
//
//   python3 scripts/gen_armv6_dw_f32.py armv6-dw-f32.inc
//   g++ -O3 -std=c++17 -I. scripts/bench52.cpp -o bench52        (loom-armv6-build:bookworm)
//   g++ -O3 -std=c++17 -I. -I<ggml>/include -DBENCH52_GGML scripts/bench52.cpp <build>/src/libggml.a
//       <build>/src/libggml-cpu.a <build>/src/libggml-base.a -fopenmp -lpthread -latomic -o bench52g
//   ./bench52 [reps=5] [arms=rcsne(g)]        GGML_CPU_ARMV6_DW=0: g runs ggml's own sweep
#ifdef BENCH52_GGML
#define GGML_ARMV6_DW_DECLS_ONLY      // the kernels come from libggml-cpu
#endif
#include "armv6-dw-f32.inc"
#ifdef BENCH52_GGML
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Call { int C, T, K; };

static std::vector<Call> jfk_schedule() {             // bench43's, without the stride-2 calls
    std::vector<Call> v;
    v.push_back({80, 1101, 5});
    for (int r = 0; r < 4; ++r) v.push_back({256, 1101, 11});
    for (int k : {13, 15, 17, 19, 21}) for (int r = 0; r < 5; ++r) v.push_back({256, 551, k});
    for (int r = 0; r < 4; ++r) v.push_back({256, 551, 13});
    for (int k : {15, 17, 19, 21, 23, 25}) for (int r = 0; r < 5; ++r) v.push_back({256, 276, k});
    for (int r = 0; r < 4; ++r) v.push_back({256, 276, 25});
    for (int k : {27, 29, 31, 33, 35, 37, 39}) for (int r = 0; r < 5; ++r) v.push_back({256, 138, k});
    v.push_back({256, 138, 41});
    return v;
}

// 32-byte aligned, as ggml requires of a buffer made from a pointer (arm G places tensors on these).
template <class T> struct Align32 {
    using value_type = T;
    Align32() = default;
    template <class U> Align32(const Align32<U> &) {}
    T * allocate(size_t n) { return (T *) aligned_alloc(32, (n * sizeof(T) + 31) & ~(size_t) 31); }
    void deallocate(T * p, size_t) { free(p); }
    template <class U> bool operator==(const Align32<U> &) const { return true; }
    template <class U> bool operator!=(const Align32<U> &) const { return false; }
};
using fvec = std::vector<float, Align32<float>>;

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

static void row_ref(const float * xp, const float * w, int K, int To, float * y) {
    for (int t = 0; t < To; ++t) {
        float a = 0.0f;
        for (int k = 0; k < K; ++k) a += w[k] * xp[t + k];
        y[t] = a;
    }
}

// bench43's dw_f32_row, stride 1.
static void row_c(const float * xp, const float * w, int K, int To, float * y) {
    int t = 0;
    for (; t + 8 <= To; t += 8) {
        const float * p = xp + t;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        float s0 = p[0], s1 = p[1], s2 = p[2], s3 = p[3], s4 = p[4], s5 = p[5], s6 = p[6], s7;
        int k = 0;
#define TAP(J, N, X0, X1, X2, X3, X4, X5, X6, X7) { \
            N = p[k + (J) + 7]; const float wk = w[k + (J)]; \
            a0 += wk * X0; a1 += wk * X1; a2 += wk * X2; a3 += wk * X3; \
            a4 += wk * X4; a5 += wk * X5; a6 += wk * X6; a7 += wk * X7; }
        for (; k + 8 <= K; k += 8) {
            TAP(0, s7, s0, s1, s2, s3, s4, s5, s6, s7)
            TAP(1, s0, s1, s2, s3, s4, s5, s6, s7, s0)
            TAP(2, s1, s2, s3, s4, s5, s6, s7, s0, s1)
            TAP(3, s2, s3, s4, s5, s6, s7, s0, s1, s2)
            TAP(4, s3, s4, s5, s6, s7, s0, s1, s2, s3)
            TAP(5, s4, s5, s6, s7, s0, s1, s2, s3, s4)
            TAP(6, s5, s6, s7, s0, s1, s2, s3, s4, s5)
            TAP(7, s6, s7, s0, s1, s2, s3, s4, s5, s6)
        }
#undef TAP
        for (; k < K; ++k) {
            const float wk = w[k]; const float * q = p + k;
            a0 += wk * q[0]; a1 += wk * q[1]; a2 += wk * q[2]; a3 += wk * q[3];
            a4 += wk * q[4]; a5 += wk * q[5]; a6 += wk * q[6]; a7 += wk * q[7];
        }
        y[t] = a0; y[t + 1] = a1; y[t + 2] = a2; y[t + 3] = a3;
        y[t + 4] = a4; y[t + 5] = a5; y[t + 6] = a6; y[t + 7] = a7;
    }
    for (; t < To; ++t) {
        float a = 0;
        for (int k = 0; k < K; ++k) a += w[k] * xp[t + k];
        y[t] = a;
    }
}

// The kernel's contract: the even taps and the odd taps summed separately, each in k order, then added.
static void row_contract(const float * xp, const float * w, int K, int To, float * y) {
    for (int t = 0; t < To; ++t) {
        float e = w[0] * xp[t], o = w[1] * xp[t + 1];
        for (int k = 2; k < K; ++k) (k % 2 ? o : e) += w[k] * xp[t + k];
        y[t] = e + o;
    }
}

// The kernel: full blocks straight into y, a ragged tail through 8 floats of staging.
static void row_kernel(const float * xp, const float * wp, int K, int To, float * y, const void * pf) {
    const ggml_armv6_dw_fn fn = ggml_armv6_dw_f32[K >= 8][K % 8];
    ggml_armv6_dw_args a = { xp, wp, y, To / 8, K / 8, pf };
    if (a.nblk > 0) fn(&a);
    if (To % 8) {
        float tmp[8];
        a.x = xp + 8 * (To / 8); a.y = tmp; a.nblk = 1;
        fn(&a);
        memcpy(y + 8 * (To / 8), tmp, (size_t) (To % 8) * 4);
    }
}

// Arm f: ggml-0035's ops.cpp code for one call, copied (ggml_conv_2d_dw_params, the row function and the
// impl's channel loop as they reach it).
struct DwParams { int64_t channels, batch, src_w, src_h, dst_w, dst_h, knl_w, knl_h; int stride_x, stride_y, pad_x, pad_y, dilation_x, dilation_y; };

static void ops_like_row(const float * src, const float * w, float * dst, const DwParams & p, float * xp, float * wp,
                         const float * next) {
    const int64_t K = p.knl_w, OW = p.dst_w;
    const int64_t L = 8 * (OW / 8 + 1) + K + 16;
    const int64_t lo = std::min(std::max((int64_t) p.pad_x, (int64_t) 0), L);
    const int64_t hi = std::min(std::max((int64_t) p.pad_x + p.src_w, lo), L);
    std::fill(xp, xp + lo, 0.0f);
    std::copy(src + (lo - p.pad_x), src + (hi - p.pad_x), xp + lo);
    std::fill(xp + hi, xp + L, 0.0f);
    std::copy(w, w + K, wp);
    std::fill(wp + K, wp + K + 8, 0.0f);
    const ggml_armv6_dw_fn fn = ggml_armv6_dw_f32[K >= 8][K % 8];
    ggml_armv6_dw_args a = { xp, wp, dst, (int32_t) (OW / 8), (int32_t) (K / 8), next };
    fn(&a);
    if (OW % 8) {
        float tmp[8];
        a.x = xp + 8 * (OW / 8);
        a.y = tmp;
        a.nblk = 1;
        fn(&a);
        std::copy(tmp, tmp + OW % 8, dst + 8 * (OW / 8));
    }
}

static void ops_like(const float * src_base, const float * knl, float * dst_base, const DwParams & p, bool counter) {
    const int64_t n = p.channels * p.batch, start = 0, end = n;
    std::vector<float> armv6_buf(8 * (p.dst_w / 8 + 1) + 2 * p.knl_w + 24);
    int64_t ch = start % p.channels;
    for (int64_t i = start; i < end; ++i, ch = ch + 1 == p.channels ? 0 : ch + 1) {
        const int64_t knl_offset = (counter ? ch : i % p.channels) * p.knl_w * p.knl_h;
        const float * src_data = src_base + i * p.src_w * p.src_h;
        float * dst_data = dst_base + i * p.dst_w * p.dst_h;
        float * xp = armv6_buf.data();
        ops_like_row(src_data, knl + knl_offset, dst_data, p, xp, xp + 8 * (p.dst_w / 8 + 1) + p.knl_w + 16,
                     i + 1 < end ? src_data + p.src_w : src_data);
    }
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const char * arms = argc > 2 ? argv[2] : "rcsne";
    {   // RunFast: FZ | DN, as ggml-0031 sets it on the compute threads
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    const auto sched = jfk_schedule();
    double macs = 0;
    for (auto & c : sched) macs += (double) c.C * c.T * c.K;
    printf("jfk depthwise schedule, stride 1: %zu calls, %.1f MMAC\n", sched.size(), macs / 1e6);

    std::mt19937 rng(42);
    std::normal_distribution<float> nx(0, 1), nw(0, 0.15f);
    const char * names = "rcsnegGfh";
    double tt[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, worst = 0;
    long bad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, total = 0;
#ifdef BENCH52_GGML
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 1);
#endif
    float hot[8] = {0};

    printf("%4s %5s %3s |", "C", "T", "K");
    for (int a = 0; a < 9; ++a) printf("   %c ms", names[a]);
    printf("\n");
    for (const Call & c : sched) {
        const int p = c.K / 2, To = c.T;                 // 'same' padding
        const int L = 8 * (To / 8 + 1) + c.K + 16;       // the kernel's read bound, ragged tail included
        fvec x((size_t) c.C * c.T), w((size_t) c.C * c.K);
        for (auto & v : x) v = nx(rng);
        for (auto & v : w) v = nw(rng);
        // Padded rows and weights made once for r, c, s, n.
        std::vector<float> xp((size_t) c.C * L, 0.0f), wp((size_t) c.C * (c.K + 8), 0.0f);
        for (int ch = 0; ch < c.C; ++ch) {
            memcpy(&xp[(size_t) ch * L + p], &x[(size_t) ch * c.T], (size_t) c.T * 4);
            memcpy(&wp[(size_t) ch * (c.K + 8)], &w[(size_t) ch * c.K], (size_t) c.K * 4);
        }
        std::vector<float> scratch(L + c.K + 8);
        fvec y[9], yc((size_t) c.C * To);
        double best[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int a = 7; a < 9; ++a) {
            if (!strchr(arms, names[a])) continue;
            DwParams P = { c.C, 1, c.T, 1, To, 1, c.K, 1, 1, 1, p, 0, 1, 1 };
            y[a].assign((size_t) c.C * To, 0.0f);
            best[a] = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                ops_like(x.data(), w.data(), y[a].data(), P, a == 8);
                best[a] = std::min(best[a], ms_since(t0));
            }
            tt[a] += best[a];
        }
#ifdef BENCH52_GGML
        for (int a = 5; a < 7; ++a) {
            if (!strchr(arms, names[a])) continue;
            ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.K, 1, 1, c.C);
            ggml_tensor * tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.T, 1, c.C, 1);
            ggml_tensor * ty = ggml_conv_2d_dw_direct(ctx, tk, tx, 1, 1, p, 0, 1, 1);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, ty);
            y[a].assign((size_t) c.C * To, 0.0f);
            std::vector<ggml_backend_buffer_t> bufs;
            if (a == 5) {
                bufs.push_back(ggml_backend_alloc_ctx_tensors(ctx, be));
                ggml_backend_tensor_set(tk, w.data(), 0, ggml_nbytes(tk));
                ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
            } else {
                for (auto [t, ptr] : { std::pair<ggml_tensor *, float *>{tk, w.data()}, {tx, x.data()}, {ty, y[a].data()} }) {
                    bufs.push_back(ggml_backend_cpu_buffer_from_ptr(ptr, ggml_nbytes(t)));
                    ggml_backend_tensor_alloc(bufs.back(), t, ptr);
                }
            }
            best[a] = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                ggml_backend_graph_compute(be, gf);
                best[a] = std::min(best[a], ms_since(t0));
            }
            tt[a] += best[a];
            if (a == 5) ggml_backend_tensor_get(ty, y[a].data(), 0, ggml_nbytes(ty));
            for (auto b : bufs) ggml_backend_buffer_free(b);
            ggml_free(ctx);
        }
#endif
        for (int a = 0; a < 5; ++a) {
            if (!strchr(arms, names[a])) continue;
            y[a].assign((size_t) c.C * To, 0.0f);
            best[a] = 1e30;
            for (int r = 0; r < reps; ++r) {
                auto t0 = clk::now();
                for (int ch = 0; ch < c.C; ++ch) {
                    const float * xr = &xp[(size_t) ch * L];
                    const float * wr = &wp[(size_t) ch * (c.K + 8)];
                    float * yr = &y[a][(size_t) ch * To];
                    const int nx_ = ch + 1 < c.C ? ch + 1 : ch;
                    switch (names[a]) {
                        case 'r': row_ref(xr, wr, c.K, To, yr); break;
                        case 'c': row_c(xr, wr, c.K, To, yr); break;
                        case 's': row_kernel(xr, wr, c.K, To, yr, &xp[(size_t) nx_ * L]); break;
                        case 'n': row_kernel(xr, wr, c.K, To, yr, hot); break;
                        case 'e': {
                            float * xs = scratch.data(), * ws = xs + L;
                            std::fill(xs, xs + p, 0.0f);
                            memcpy(xs + p, &x[(size_t) ch * c.T], (size_t) c.T * 4);
                            std::fill(xs + p + c.T, xs + L, 0.0f);
                            memcpy(ws, &w[(size_t) ch * c.K], (size_t) c.K * 4);
                            std::fill(ws + c.K, ws + c.K + 8, 0.0f);
                            row_kernel(xs, ws, c.K, To, yr, &x[(size_t) nx_ * c.T]);
                            break;
                        }
                    }
                }
                best[a] = std::min(best[a], ms_since(t0));
            }
            tt[a] += best[a];
        }
        for (int ch = 0; ch < c.C; ++ch)
            row_contract(&xp[(size_t) ch * L], &wp[(size_t) ch * (c.K + 8)], c.K, To, &yc[(size_t) ch * To]);
        for (int a = 0; a < 9; ++a) {
            if (y[a].empty()) continue;
            const fvec & want = a < 2 ? (y[0].empty() ? y[a] : y[0]) : yc;
            for (size_t i = 0; i < want.size(); ++i) bad[a] += memcmp(&want[i], &y[a][i], 4) != 0;
        }
        if (!y[0].empty()) {
            double mx = 0, d = 0;
            for (size_t i = 0; i < yc.size(); ++i) { mx = std::max(mx, (double) fabsf(y[0][i])); d = std::max(d, (double) fabsf(y[0][i] - yc[i])); }
            worst = std::max(worst, d / mx);
        }
        total += (long) yc.size();
        printf("%4d %5d %3d |", c.C, c.T, c.K);
        for (int a = 0; a < 9; ++a) printf(" %7.2f", best[a]);
        printf("\n");
        fflush(stdout);
    }
#ifdef BENCH52_GGML
    if (tt[5] + tt[6] > 0) {   // ggml's fixed cost per graph compute: one 1-channel, 8-output depthwise node
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 1, 1);
        ggml_tensor * tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 1, 1, 1);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_conv_2d_dw_direct(ctx, tk, tx, 1, 1, 1, 0, 1, 1));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_clear(buf, 0);
        double bo = 1e30;
        for (int r = 0; r < 5; ++r) {
            auto t0 = clk::now();
            for (int i = 0; i < 200; ++i) ggml_backend_graph_compute(be, gf);
            bo = std::min(bo, ms_since(t0) / 200);
        }
        printf("g: %.3f ms per graph compute of a trivial node, %.1f ms over the %zu calls\n", bo,
               bo * sched.size(), sched.size());
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
#endif
    for (int a = 0; a < 9; ++a)
        if (tt[a] > 0) printf("%c %8.1f ms  %.3f GMAC/s  %ld bit mismatches (of %ld; r and c against r, the rest against the contract)\n",
                              names[a], tt[a], macs / tt[a] / 1e6, bad[a], total);
    if (tt[0] > 0) printf("contract vs r (ggml's order): max |diff| / max |y| = %.3g\n", worst);
    return 0;
}
