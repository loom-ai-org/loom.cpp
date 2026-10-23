// P4.38: an outer-product F32 GEMM tile for the Cortex-A72 (XNNPACK's 4x8) under an L2-blocked schedule,
// against tinyBLAS's 4x3 dot-product tile, at whisper-small's encoder linear shapes, in ONE process.
//
// NOT part of the build.
//
// WHY IT EXISTS. On the Pi 4, whisper's dense encoder GEMMs run at 1.20-1.46x MLAS's time at four threads.
// tinyBLAS reaches ~9 GFLOP/s on one core at K = 768 (63% of the 14.3 GF FMA roofline) but scales only
// 2.85x to four. XNNPACK's A72 kernel (`f32-gemm 4x8 aarch64-neonfma-cortex-a75-prfm`) reached 30-44
// GFLOP/s at four threads on shapes whose weights fit the 1 MB L2, and 7-10 at whisper's, where its
// 4-row x all-N tiling re-streams every weight from DRAM for each 4 rows. So the kernel is not the
// question by itself; the kernel under a schedule that keeps packed weight panels in L2 is.
//
//   ggml                           ggml_mul_mat through the patched ggml (tinyBLAS, LLAMAFILE ON)
//   asm:NB:MB:ORD:MODE             XNNPACK's asm 4x8 kernel (full K only)
//   port:NB:KC:MB:ORD:MODE         the same tile in NEON intrinsics, optionally K-chunked
//   porta..porte:...               the same with prefetch mode 0..4 (see tile4x8); plain port is 1
//
//     NB    output columns (weight rows) per packed block, a multiple of 8
//     KC    contraction chunk, a multiple of 4; 0 = all of K
//     MB    activation rows per inner block, a multiple of 4 (ORD p only)
//     ORD   r: per 4 activation rows, every panel of the block (A in L1, panels stream: XNNPACK's order)
//           p: per panel, every 4-row group of an MB block (the panel in L1, A streams)
//     MODE  s: one block shared by all threads, packed cooperatively, rows split between threads
//           t: blocks dealt to threads, each packs its own and walks every row
//
// The weights are packed INSIDE the timed region (a per-call pack, the cheap-to-integrate design); a
// packed-at-load design would save that, and `pack` prints what it costs.
//
// BUILD (on the board):
//
//   B=~/p437/loom.cpp/build; XB=~/xnnbench/xnn-build
//   g++ -O3 -std=c++17 -mcpu=cortex-a72 -I $B/_deps/ggml-src/include scripts/bench66.cpp -o bench66 \
//       $XB/libxnnpack-microkernels-prod.a -L $B/_deps/ggml-build/src -lggml -lggml-base -lggml-cpu \
//       -lpthread -lm -Wl,-rpath,$B/_deps/ggml-build/src
//
//   ./bench66 <threads> <rounds> <shape> arm [arm ...]
//       shape: qkv | fc1 | fc2 | KxNxM. The first arm is the baseline every ratio is against.
//
// ESTIMATOR as bench65: every round runs each arm once over all nodes, the order rotated round to round;
// ratios are per round and medianed. ACCURACY first: each arm's relative Frobenius error on node 0
// against a double-precision product.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

extern "C" void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm(
    size_t mr, size_t nc, size_t kc, const float * a, size_t a_stride, const void * w, float * c, size_t cm_stride,
    size_t cn_stride, const void * params);

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

// ------------------------------------------------------------------------------------------ thread pool
// Idle workers SLEEP on a condition variable: a pool whose idle threads spin steals the cores from
// ggml's own threadpool while the ggml arm runs (the first version of this bench did, and put ggml at
// 1.5-2x its in-model time). Inside a run, the barrier spins.
struct Pool {
    int nth;
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv;
    int gen = 0;
    bool quit = false;
    std::atomic<int> done{0};
    std::function<void(int)> fn;
    std::atomic<int> bar_count{0}, bar_gen{0};

    explicit Pool(int n) : nth(n) {
        for (int i = 1; i < n; ++i) {
            th.emplace_back([this, i] {
                int seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return quit || gen != seen; });
                        if (quit) return;
                        seen = gen;
                    }
                    fn(i);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        for (auto & t : th) t.join();
    }
    void run(std::function<void(int)> f) {
        fn = std::move(f);
        done.store(0);
        {
            std::lock_guard<std::mutex> lk(mu);
            ++gen;
        }
        cv.notify_all();
        fn(0);
        while (done.load(std::memory_order_acquire) != nth - 1) {}
    }
    void barrier() {
        if (nth == 1) return;
        const int g = bar_gen.load(std::memory_order_acquire);
        if (bar_count.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
            bar_count.store(0, std::memory_order_relaxed);
            bar_gen.fetch_add(1, std::memory_order_acq_rel);
        } else {
            while (bar_gen.load(std::memory_order_acquire) == g) {}
        }
    }
};

// ------------------------------------------------------------------------------------------ packing
// Panels of 8 weight rows, K-major: for each k, the 8 rows' k-th values. With `bias`, each panel starts
// with 8 zero floats (XNNPACK's layout). Rows past N are zero.
static void pack_panels(const float * W, int64_t K, int64_t N, int64_t n0, int64_t n1, int64_t k0, int64_t k1,
                        bool bias, float * out) {
    const int64_t kc = k1 - k0;
    for (int64_t n = n0; n < n1; n += 8) {
        if (bias) {
            memset(out, 0, 8 * sizeof(float));
            out += 8;
        }
        const float * r[8];
        for (int j = 0; j < 8; ++j) r[j] = n + j < N ? W + (n + j) * K + k0 : nullptr;
        int64_t k = 0;
        if (n + 8 <= N) {
            for (; k + 4 <= kc; k += 4) {
                float32x4_t v[8];
                for (int j = 0; j < 8; ++j) v[j] = vld1q_f32(r[j] + k);
                // two 4x4 transposes: rows 0-3 and rows 4-7
                for (int h = 0; h < 2; ++h) {
                    float32x4_t * q = v + 4 * h;
                    const float32x4x2_t t01 = vtrnq_f32(q[0], q[1]);
                    const float32x4x2_t t23 = vtrnq_f32(q[2], q[3]);
                    const float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
                    const float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
                    const float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
                    const float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
                    vst1q_f32(out + 0 * 8 + 4 * h, c0);
                    vst1q_f32(out + 1 * 8 + 4 * h, c1);
                    vst1q_f32(out + 2 * 8 + 4 * h, c2);
                    vst1q_f32(out + 3 * 8 + 4 * h, c3);
                }
                out += 32;
            }
        }
        for (; k < kc; ++k) {
            for (int j = 0; j < 8; ++j) out[j] = r[j] ? r[j][k] : 0.0f;
            out += 8;
        }
    }
}

// ------------------------------------------------------------------------------------------ the port
// C[mr x 8] (+)= A[mr x kc] * P[kc x 8]. A rows at stride lda, C rows at stride ldc, mr in 1..4; the
// missing rows alias the last real one, as XNNPACK's kernel does.
// PF: 0 none, 1 one line 512 B ahead (the first cut: half the panel's lines), 2/3/4 both lines of the
// step at 256 / 512 / 1024 B ahead.
template <int PF>
static inline void tile4x8(int64_t mr, const float * a, int64_t lda, const float * p, int64_t kc, float * c,
                           int64_t ldc, bool accumulate) {
    const float * a0 = a;
    const float * a1 = mr > 1 ? a0 + lda : a0;
    const float * a2 = mr > 2 ? a1 + lda : a1;
    const float * a3 = mr > 3 ? a2 + lda : a2;
    float * c0 = c;
    float * c1 = mr > 1 ? c0 + ldc : c0;
    float * c2 = mr > 2 ? c1 + ldc : c1;
    float * c3 = mr > 3 ? c2 + ldc : c2;
    float32x4_t x0, x1, y0, y1, z0, z1, w0, w1;
    if (accumulate) {
        x0 = vld1q_f32(c0); x1 = vld1q_f32(c0 + 4);
        y0 = vld1q_f32(c1); y1 = vld1q_f32(c1 + 4);
        z0 = vld1q_f32(c2); z1 = vld1q_f32(c2 + 4);
        w0 = vld1q_f32(c3); w1 = vld1q_f32(c3 + 4);
    } else {
        x0 = x1 = y0 = y1 = z0 = z1 = w0 = w1 = vdupq_n_f32(0.0f);
    }
    int64_t k = 0;
    for (; k + 4 <= kc; k += 4) {
        const float32x4_t va0 = vld1q_f32(a0 + k), va1 = vld1q_f32(a1 + k);
        const float32x4_t va2 = vld1q_f32(a2 + k), va3 = vld1q_f32(a3 + k);
        if constexpr (PF == 1) { __builtin_prefetch(p + 128); }
        if constexpr (PF == 2) { __builtin_prefetch(p + 64); __builtin_prefetch(p + 80); }
        if constexpr (PF == 3) { __builtin_prefetch(p + 128); __builtin_prefetch(p + 144); }
        if constexpr (PF == 4) { __builtin_prefetch(p + 256); __builtin_prefetch(p + 272); }
        float32x4_t b0 = vld1q_f32(p), b1 = vld1q_f32(p + 4);
        x0 = vfmaq_laneq_f32(x0, b0, va0, 0); x1 = vfmaq_laneq_f32(x1, b1, va0, 0);
        y0 = vfmaq_laneq_f32(y0, b0, va1, 0); y1 = vfmaq_laneq_f32(y1, b1, va1, 0);
        z0 = vfmaq_laneq_f32(z0, b0, va2, 0); z1 = vfmaq_laneq_f32(z1, b1, va2, 0);
        w0 = vfmaq_laneq_f32(w0, b0, va3, 0); w1 = vfmaq_laneq_f32(w1, b1, va3, 0);
        b0 = vld1q_f32(p + 8); b1 = vld1q_f32(p + 12);
        x0 = vfmaq_laneq_f32(x0, b0, va0, 1); x1 = vfmaq_laneq_f32(x1, b1, va0, 1);
        y0 = vfmaq_laneq_f32(y0, b0, va1, 1); y1 = vfmaq_laneq_f32(y1, b1, va1, 1);
        z0 = vfmaq_laneq_f32(z0, b0, va2, 1); z1 = vfmaq_laneq_f32(z1, b1, va2, 1);
        w0 = vfmaq_laneq_f32(w0, b0, va3, 1); w1 = vfmaq_laneq_f32(w1, b1, va3, 1);
        b0 = vld1q_f32(p + 16); b1 = vld1q_f32(p + 20);
        x0 = vfmaq_laneq_f32(x0, b0, va0, 2); x1 = vfmaq_laneq_f32(x1, b1, va0, 2);
        y0 = vfmaq_laneq_f32(y0, b0, va1, 2); y1 = vfmaq_laneq_f32(y1, b1, va1, 2);
        z0 = vfmaq_laneq_f32(z0, b0, va2, 2); z1 = vfmaq_laneq_f32(z1, b1, va2, 2);
        w0 = vfmaq_laneq_f32(w0, b0, va3, 2); w1 = vfmaq_laneq_f32(w1, b1, va3, 2);
        b0 = vld1q_f32(p + 24); b1 = vld1q_f32(p + 28);
        x0 = vfmaq_laneq_f32(x0, b0, va0, 3); x1 = vfmaq_laneq_f32(x1, b1, va0, 3);
        y0 = vfmaq_laneq_f32(y0, b0, va1, 3); y1 = vfmaq_laneq_f32(y1, b1, va1, 3);
        z0 = vfmaq_laneq_f32(z0, b0, va2, 3); z1 = vfmaq_laneq_f32(z1, b1, va2, 3);
        w0 = vfmaq_laneq_f32(w0, b0, va3, 3); w1 = vfmaq_laneq_f32(w1, b1, va3, 3);
        p += 32;
    }
    for (; k < kc; ++k) {
        const float32x4_t b0 = vld1q_f32(p), b1 = vld1q_f32(p + 4);
        x0 = vfmaq_n_f32(x0, b0, a0[k]); x1 = vfmaq_n_f32(x1, b1, a0[k]);
        y0 = vfmaq_n_f32(y0, b0, a1[k]); y1 = vfmaq_n_f32(y1, b1, a1[k]);
        z0 = vfmaq_n_f32(z0, b0, a2[k]); z1 = vfmaq_n_f32(z1, b1, a2[k]);
        w0 = vfmaq_n_f32(w0, b0, a3[k]); w1 = vfmaq_n_f32(w1, b1, a3[k]);
        p += 8;
    }
    // highest row first, so an aliased row ends with its own values
    vst1q_f32(c3, w0); vst1q_f32(c3 + 4, w1);
    vst1q_f32(c2, z0); vst1q_f32(c2 + 4, z1);
    vst1q_f32(c1, y0); vst1q_f32(c1 + 4, y1);
    vst1q_f32(c0, x0); vst1q_f32(c0 + 4, x1);
}

// ------------------------------------------------------------------------------------------ schedules
struct Sched {
    bool asm_kernel = false;
    int pf = 1;
    int64_t NB = 64, KC = 0, MB = 64;
    char ord = 'r', mode = 's';
};

struct Problem {
    int64_t K, N, M;
    const float * W;   // [N][K]
    const float * X;   // [M][K]
    float * Y;         // [M][N]
};

// One block: columns [n0, n1) of the output, contraction [k0, k1), activation rows [m0, m1), packed panels
// at `pk` (laid out for the chosen kernel).
static void compute_block(const Sched & s, const Problem & P, const float * pk, int64_t n0, int64_t n1, int64_t k0,
                          int64_t k1, int64_t m0, int64_t m1) {
    const int64_t kc = k1 - k0;
    const bool acc = k0 > 0;
    const int64_t npan = (n1 - n0 + 7) / 8;
    const int64_t pstride = kc * 8 + (s.asm_kernel ? 8 : 0);
    static const float minmax[2] = { -INFINITY, INFINITY };
    if (s.asm_kernel) {
        for (int64_t m = m0; m < m1; m += 4) {
            const int64_t mr = std::min<int64_t>(4, m1 - m);
            xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm(
                mr, n1 - n0, kc * sizeof(float), P.X + m * P.K, P.K * sizeof(float), pk, P.Y + m * P.N + n0,
                P.N * sizeof(float), 8 * sizeof(float), minmax);
        }
        return;
    }
    auto tile = s.pf == 0 ? tile4x8<0> : s.pf == 1 ? tile4x8<1> : s.pf == 2 ? tile4x8<2> : s.pf == 3 ? tile4x8<3> : tile4x8<4>;
    if (s.ord == 'r') {
        for (int64_t m = m0; m < m1; m += 4) {
            const int64_t mr = std::min<int64_t>(4, m1 - m);
            for (int64_t p = 0; p < npan; ++p) {
                tile(mr, P.X + m * P.K + k0, P.K, pk + p * pstride, kc, P.Y + m * P.N + n0 + p * 8, P.N, acc);
            }
        }
    } else {
        for (int64_t mb = m0; mb < m1; mb += s.MB) {
            const int64_t me = std::min(m1, mb + s.MB);
            for (int64_t p = 0; p < npan; ++p) {
                for (int64_t m = mb; m < me; m += 4) {
                    const int64_t mr = std::min<int64_t>(4, me - m);
                    tile(mr, P.X + m * P.K + k0, P.K, pk + p * pstride, kc, P.Y + m * P.N + n0 + p * 8, P.N, acc);
                }
            }
        }
    }
}

struct Packed {
    std::vector<float *> buf;   // per thread (mode t) or one shared (mode s)
};

static float * aligned_floats(size_t n) {
    void * p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) abort();
    return (float *) p;
}

static void run_sched(Pool & pool, const Sched & s, const Problem & P, std::vector<float *> & bufs) {
    const int nth = pool.nth;
    const int64_t KC = s.KC > 0 ? std::min(s.KC, P.K) : P.K;
    const int64_t nblk = (P.N + s.NB - 1) / s.NB;
    pool.run([&](int ith) {
        if (s.mode == 's') {
            float * pk = bufs[0];
            // rows split in multiples of 4
            const int64_t g = (P.M + 3) / 4;
            const int64_t m0 = std::min(P.M, (g * ith / nth) * 4), m1 = std::min(P.M, (g * (ith + 1) / nth) * 4);
            for (int64_t b = 0; b < nblk; ++b) {
                const int64_t n0 = b * s.NB, n1 = std::min(P.N, n0 + s.NB);
                const int64_t npan = (n1 - n0 + 7) / 8;
                for (int64_t k0 = 0; k0 < P.K; k0 += KC) {
                    const int64_t k1 = std::min(P.K, k0 + KC);
                    const int64_t pstride = (k1 - k0) * 8 + (s.asm_kernel ? 8 : 0);
                    for (int64_t p = ith; p < npan; p += nth) {
                        pack_panels(P.W, P.K, P.N, n0 + p * 8, std::min(n1, n0 + p * 8 + 8), k0, k1, s.asm_kernel,
                                    pk + p * pstride);
                    }
                    pool.barrier();
                    compute_block(s, P, pk, n0, n1, k0, k1, m0, m1);
                    pool.barrier();
                }
            }
        } else {
            float * pk = bufs[ith];
            for (int64_t b = ith; b < nblk; b += nth) {
                const int64_t n0 = b * s.NB, n1 = std::min(P.N, n0 + s.NB);
                for (int64_t k0 = 0; k0 < P.K; k0 += KC) {
                    const int64_t k1 = std::min(P.K, k0 + KC);
                    pack_panels(P.W, P.K, P.N, n0, n1, k0, k1, s.asm_kernel, pk);
                    compute_block(s, P, pk, n0, n1, k0, k1, 0, P.M);
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------------------ main
int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s threads rounds shape arm [arm ...]\n", argv[0]);
        return 2;
    }
    const int threads = atoi(argv[1]), rounds = atoi(argv[2]);
    const std::string shape = argv[3];
    int64_t K, N, M;
    if (shape == "qkv")      { K = 768;  N = 768;  M = 1500; }
    else if (shape == "fc1") { K = 768;  N = 3072; M = 1500; }
    else if (shape == "fc2") { K = 3072; N = 768;  M = 1500; }
    else if (sscanf(shape.c_str(), "%ldx%ldx%ld", &K, &N, &M) != 3) return 2;
    // Enough distinct weights to be past the L2, at least four nodes, and no more than ~100 MB of
    // activations and outputs per arm (VITS's 73472-row inputs would otherwise take the board's 3.8 GB).
    const int nodes = (int) std::max<int64_t>(
        4, std::min<int64_t>({ 48, (int64_t) (8e6 / (K * N)) + 1, (int64_t) (2.5e7 / (M * (K + N))) }));

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<std::vector<float>> W(nodes), X(nodes);
    for (int j = 0; j < nodes; ++j) {
        W[j].resize((size_t) K * N);
        X[j].resize((size_t) K * M);
        for (float & v : W[j]) v = gauss(rng) * 0.05f;
        for (float & v : X[j]) v = gauss(rng);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    Pool pool(threads);

    struct Arm {
        std::string name;
        bool is_ggml = false;
        Sched s;
        std::vector<std::vector<float>> Y;
        std::vector<float *> bufs;
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t wbuf = nullptr, xbuf = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_cgraph * gf = nullptr;
        std::vector<ggml_tensor *> outs;
    };
    std::vector<Arm> A;
    for (int i = 4; i < argc; ++i) {
        Arm a;
        a.name = argv[i];
        if (a.name == "ggml") {
            a.is_ggml = true;
            ggml_init_params p = { (size_t) (8 * nodes + 64) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            a.ctx = ggml_init(p);
            std::vector<ggml_tensor *> w(nodes), x(nodes);
            for (int j = 0; j < nodes; ++j) w[j] = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, K, N);
            a.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(a.ctx, ggml_backend_cpu_buffer_type());
            for (int j = 0; j < nodes; ++j) ggml_backend_tensor_set(w[j], W[j].data(), 0, ggml_nbytes(w[j]));
            ggml_init_params gp = { (size_t) (8 * nodes + 64) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * gctx = ggml_init(gp);
            for (int j = 0; j < nodes; ++j) {
                x[j] = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, M);
                ggml_set_input(x[j]);
            }
            a.xbuf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_cpu_buffer_type());
            for (int j = 0; j < nodes; ++j) ggml_backend_tensor_set(x[j], X[j].data(), 0, ggml_nbytes(x[j]));
            a.gf = ggml_new_graph(gctx);
            for (int j = 0; j < nodes; ++j) {
                ggml_tensor * y = ggml_mul_mat(gctx, w[j], x[j]);
                ggml_set_output(y);
                ggml_build_forward_expand(a.gf, y);
                a.outs.push_back(y);
            }
            a.alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            ggml_gallocr_alloc_graph(a.alloc, a.gf);
        } else {
            char kind[16] = {};
            long nb = 64, kc = 0, mb = 64;
            char ord = 'r', mode = 's';
            if (a.name.rfind("asm:", 0) == 0) {
                if (sscanf(a.name.c_str(), "asm:%ld:%ld:%c:%c", &nb, &mb, &ord, &mode) != 4) return 2;
                a.s.asm_kernel = true;
            } else if (sscanf(a.name.c_str(), "%15[a-z]:%ld:%ld:%ld:%c:%c", kind, &nb, &kc, &mb, &ord, &mode) != 6 ||
                       strncmp(kind, "port", 4) != 0 || strlen(kind) > 5) {
                fprintf(stderr, "bad arm %s\n", a.name.c_str());
                return 2;
            }
            if (kind[4]) a.s.pf = kind[4] - 'a';   // porta..porte = PF 0..4; plain port = PF 1
            a.s.NB = nb; a.s.KC = kc; a.s.MB = mb; a.s.ord = ord; a.s.mode = mode;
            if (nb % 8 || kc % 4 || mb % 4 || (a.s.asm_kernel && kc)) return 2;
            const int64_t kcap = kc > 0 ? std::min<int64_t>(kc, K) : K;
            const size_t floats = (size_t) ((nb + 7) / 8) * (kcap * 8 + 8);
            const int nbuf = mode == 's' ? 1 : threads;
            for (int t = 0; t < nbuf; ++t) a.bufs.push_back(aligned_floats(floats));
            a.Y.assign(nodes, std::vector<float>((size_t) M * N));
        }
        A.push_back(std::move(a));
    }

    auto run = [&](Arm & a) {
        const double t0 = now();
        if (a.is_ggml) {
            ggml_backend_graph_compute(backend, a.gf);
        } else {
            for (int j = 0; j < nodes; ++j) {
                Problem P = { K, N, M, W[j].data(), X[j].data(), a.Y[j].data() };
                run_sched(pool, a.s, P, a.bufs);
            }
        }
        return (now() - t0) / nodes;
    };

    std::vector<double> ref((size_t) N * M);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) s += (double) W[0][n * K + k] * X[0][m * K + k];
            ref[m * N + n] = s;
        }
    printf("shape %s  K=%ld N=%ld M=%ld  nodes=%d  threads=%d  rounds=%d\n", shape.c_str(), K, N, M, nodes, threads, rounds);
    for (Arm & a : A) {
        run(a);
        std::vector<float> y((size_t) N * M);
        if (a.is_ggml) ggml_backend_tensor_get(a.outs[0], y.data(), 0, ggml_nbytes(a.outs[0]));
        else y = a.Y[0];
        double e = 0, r = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            e += (y[i] - ref[i]) * (y[i] - ref[i]);
            r += ref[i] * ref[i];
        }
        printf("  %-24s rel err %.3e\n", a.name.c_str(), std::sqrt(e / r));
    }

    // what the per-call pack alone costs, all threads, one node
    {
        std::vector<float> pk((size_t) N * K + N);
        const double t0 = now();
        for (int r = 0; r < 5; ++r) {
            pool.run([&](int ith) {
                const int64_t npan = (N + 7) / 8;
                for (int64_t p = ith; p < npan; p += threads)
                    pack_panels(W[r % nodes].data(), K, N, p * 8, std::min(N, p * 8 + 8), 0, K, false, pk.data() + p * 8 * K);
            });
        }
        printf("  pack (whole weight, %d threads): %.3f ms\n", threads, (now() - t0) / 5 * 1e3);
    }

    std::vector<std::vector<double>> t(A.size()), ratio(A.size());
    for (int r = 0; r < rounds; ++r) {
        std::vector<double> tr(A.size());
        for (size_t s = 0; s < A.size(); ++s) {
            const size_t i = (s + r) % A.size();
            tr[i] = run(A[i]);
        }
        for (size_t i = 0; i < A.size(); ++i) {
            t[i].push_back(tr[i]);
            ratio[i].push_back(tr[0] / tr[i]);
        }
    }
    const double op = 2.0 * K * N * M;
    printf("%-26s %10s %9s %8s %15s\n", "arm", "ms/node", "GFLOP/s", "x base", "range");
    for (size_t i = 0; i < A.size(); ++i) {
        std::vector<double> s = ratio[i];
        std::sort(s.begin(), s.end());
        const double mt = median(t[i]);
        printf("%-26s %10.3f %9.2f %8.3f %7.3f-%.3f\n", A[i].name.c_str(), mt * 1e3, op / mt / 1e9, median(ratio[i]),
               s.front(), s.back());
    }
    ggml_backend_free(backend);
    return 0;
}
