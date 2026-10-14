// Is XNNPACK's F32 CPU kernel set worth linking under ggml? Step 3 of the LiteRT thread (hub, Engine --
// performance): time XNNPACK's operator API against THIS tree's patched ggml on the shapes loom
// actually runs, before anyone writes an integration. NOT part of the build; a standalone measurement
// kept so the answer stays re-runnable on the next machine (the Pi 4 above all).
//
// Build (x86; on aarch64 drop nothing, XNNPACK picks its NEON kernels itself):
//   XNN=<XNNPACK checkout>; XB=<its build dir>; B=build   # a loom.cpp build with the patched ggml
//   g++ -O3 -std=c++17 -I $B/_deps/ggml-src/include -I $XNN/include -I $XB/pthreadpool-source/include \
//       scripts/bench_xnnpack.cpp -o bench_xnnpack \
//       $XB/libXNNPACK.a $XB/libxnnpack-microkernels-prod.a $XB/pthreadpool/libpthreadpool.a \
//       $XB/cpuinfo/libcpuinfo.a -L $B/_deps/ggml-build/src -lggml -lggml-base -lpthread -lm
//   On arm64 also link $XB/kleidiai/libkleidiai.a (XNNPACK enables KleidiAI there), add -lggml-cpu,
//   and on macOS a -Wl,-rpath to the ggml libs. ARMv6 (Pi Zero): XNNPACK cross-compiles with clang
//   (--target=armv6-linux-gnueabihf -march=armv6kz -marm -mfpu=vfp -mfloat-abi=hard, a Raspbian
//   sysroot); link with the board's gcc. Its prod microkernels build; `microkernels-all` crashes clang 14.
//   ./bench_xnnpack <threads> [reps] [heads] [f32|f16|q8|q4|citrinet]
//   f16 / q8 / q4 run the low-precision mode (run_lowp, below); citrinet runs ADR-077's gate 2
//   (run_citrinet), which on ARMv6 also needs the generated asm tile on the link line.
//   heads (default 12) scales QK^T and softmax for a small-RAM board: softmax alone holds ~540 MB at 12.
//   The weights rescale by 12/heads, so the whisper total is still one encoder.
//
// WHAT IS COMPARED, AND WHAT IS NOT.
//   * GEMM: ggml `mul_mat(a=[K,M], b=[K,N])` -- loom's conv and linear form, both operands K-contiguous
//     -- against `xnn_fully_connected_nc_f32` (input [M,K], weights [N,K]): the same product. XNNPACK
//     packs the weights at create time, OUTSIDE the timing, which is what a load-time repack would do.
//   * QK^T: per head [1500,64] x [64,1500], 12 heads (whisper-small's encoder), contiguous on both
//     sides. In the model ggml reads K through a permute; this measures the KERNEL, not that layout.
//   * Softmax over 1500-wide rows, 18 000 of them. ggml's `soft_max` has no fused scale here, matching
//     XNNPACK's operator.
//   * Conv1D: VITS's 20 vocoder shapes with their dilations, weighted by how often each runs per
//     synthesis. ggml runs `ggml_conv_2d_direct`, the engine's own CONV path with this tree's patches
//     and heuristics. XNNPACK runs `convolution2d_nhwc_f32` at H=1 -- CHANNELS-LAST, which is not
//     ggml's layout, so a separate column prices the two transposes an integration would pay per conv.
//
// Every ggml input is ALSO marked an output: otherwise gallocr may run an op in place over its input
// (soft_max does), and every repetition after the first computes on its own previous result -- the
// first run of this file compared softmax-of-softmax against XNNPACK and reported a 6.9 relative error.
//
// THE PROTOCOL. Same thread count for both. Every input is explicitly filled (an untouched page reads
// as the shared zero page and inflates a benchmark 2x -- see P4.14). Arms alternate repetition by
// repetition, the per-shape number is the median, and the two outputs are compared so a "win" cannot
// come from computing something else.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <pthreadpool.h>
#include <xnnpack.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static void fill(std::vector<float>& v, int a, float s) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = s * (float)((int)(i % (size_t)a) - a / 2) / (float)a;
}
// Gaussian weights for the q4 mode. On fill()'s ramp every 32-block spans nearly the same range as its
// whole row, and both of its ends reach the 4-bit clamp, which ranked the 4-bit quantizers backwards:
// per-channel qc4w came out 2-3x MORE accurate than block-32 Q4_0. On Gaussian weights the order is the
// expected one (weight RMS: Q4_0 1.66e-2, qb4w's positive scale 1.77e-2 of the max).
static void fill_normal(std::vector<float>& v, unsigned seed, float sigma) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (float& x : v) x = nd(rng);
}
static void check(xnn_status st, const char* what) {
    if (st != xnn_status_success) { std::fprintf(stderr, "%s failed: %d\n", what, (int)st); std::exit(1); }
}
// XNNPACK's operator workspace is written with ALIGNED vector stores, so it must be 64-byte aligned.
// A std::vector<char> only happens to be: the first run of this file segfaulted at one thread inside
// xnn_x32_packw_gemm_goi_ukernel (batch matmul packing B into its workspace) and not at two.
struct Workspace {
    void* p = nullptr;
    explicit Workspace(size_t n) : p(std::aligned_alloc(64, ((n + 63) / 64 + 1) * 64)) {}
    ~Workspace() { std::free(p); }
};
static double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double scale = 1e-12, worst = 0;
    for (float x : b) scale = std::max(scale, (double)std::fabs(x));
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs((double)a[i] - (double)b[i]));
    return worst / scale;
}

// One ggml graph, allocated once, re-run per repetition.
struct GgmlRun {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_tensor* out = nullptr;
    ~GgmlRun() { if (ga) ggml_gallocr_free(ga); if (ctx) ggml_free(ctx); }
};

struct Row { std::string name; double flop, weight, g_ms, x_ms, layout_ms, rel; };

// ------------------------------------------------------------------------------------------------
// LOW-PRECISION MODE (`f16` or `q8` as the fourth argument): the same GEMM and conv shapes, each
// engine in the precision a loom export of that dtype would actually run. `QK^T` and softmax are
// left out -- they read activations, which stay F32 in loom whatever the weights are.
//
//   f16:  ggml, F16 weights x F32 activations (mul_mat / ggml_conv_2d_direct with an F16 kernel --
//         what an F16 GGUF runs) against XNNPACK's NATIVE F16 operators (F16 in, F16 out; inputs
//         converted outside the timing, as an all-F16 graph would hold them). XNNPACK declines
//         those without FP16 hardware (the A72, ARMv6), and the row says n/a. Its `f32_f16`
//         variant is not timed: it widens the weights to F32 at create time, so it IS the F32 arm.
//   q8:   ggml Q8_0 (block-32 int8 weights; mul_mat quantizes the activation rows to Q8_0 itself
//         and runs int8 dot products, while a CONV dequantizes its kernel to F32 every call and
//         runs the F32 sweep -- ggml-0013, which is what a Q8_0 loom export does) against two
//         XNNPACK arms (plus `ggml rpk`, ggml's own CPU_REPACK int8 GEMM -- see cpu_repack_buft):
//           qd8   dynamic: per-channel int8 weights, activations quantized on the fly (per row for
//                 a GEMM, per image for a conv), F32 out. The convert op is INSIDE the timing --
//                 the drop-in counterpart of a Q8_0 GGUF.
//           qs8   static: int8 in, int8 out, scales fixed ahead of time (here from the reference
//                 output's range). What a LiteRT full-int8 model runs; it needs a calibrated
//                 model, not just quantized weights, so it is a ceiling, not a drop-in.
//
//   q4:   ggml Q4_0 (block-32 4-bit weights, one F16 scale per block; mul_mat runs Q4_0 x Q8_0, a
//         conv dequantizes its folded kernel to F32 as for Q8_0) and `ggml rpk`, the same Q4_0 weight
//         through CPU_REPACK -- which, unlike Q8_0, ggml repacks for AVX2 (8x8) as well as for NEON
//         dotprod (4x4) / i8mm (4x8). Against four XNNPACK arms, every one dynamic or static as in q8:
//           qb4w   qd8 x blockwise 4-bit, block 32: Q4_0's arithmetic, but NOT Q4_0's blocks --
//                  qb4w requires a positive scale and Q4_0's is signed (quant_block32_4bit_bf16
//                  says why that cannot be re-laid losslessly). So the weights differ slightly, and
//                  its error column should sit NEAR ggml's; far from it is a layout bug.
//           qc4w   qd8 x per-channel 4-bit (one scale per output row): coarser than block-32.
//           qs8c4  static int8 activations x per-channel 4-bit (`qs8_qc4w`), the ceiling as qs8 is.
//           qd8c8  the q8 mode's qd8 arm again (per-channel int8), so the 4-bit arms have their
//                  8-bit reference in the same process.
//         XNNPACK has no 4-bit CONVOLUTION operator at all, so on the conv rows only qd8c8 runs.
//
// Every arm's error is reported against XNNPACK's F32 operator on the same inputs (the reference the
// F32 mode showed agreeing with ggml to ~2e-6), as max |y - ref| / max |ref|.
static void quant_per_channel(const std::vector<float>& w, int64_t rows, int64_t cols,
                              std::vector<int8_t>& q, std::vector<float>& scale) {
    q.resize(rows * cols); scale.resize(rows);
    for (int64_t r = 0; r < rows; ++r) {
        float amax = 0;
        for (int64_t c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(w[r * cols + c]));
        const float s = amax > 0 ? amax / 127.0f : 1.0f;
        scale[r] = s;
        for (int64_t c = 0; c < cols; ++c)
            q[r * cols + c] = (int8_t)std::lround(std::clamp(w[r * cols + c] / s, -127.0f, 127.0f));
    }
}
static float amax_of(const std::vector<float>& v) {
    float a = 0; for (float x : v) a = std::max(a, std::fabs(x)); return a > 0 ? a : 1.0f;
}
static void quant_tensor(const std::vector<float>& x, float s, std::vector<int8_t>& q) {
    q.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) q[i] = (int8_t)std::lround(std::clamp(x[i] / s, -128.0f, 127.0f));
}
static std::vector<float> f16_to_f32(const std::vector<uint16_t>& h) {
    std::vector<float> f(h.size());
    ggml_fp16_to_fp32_row((const ggml_fp16_t*)h.data(), f.data(), (int64_t)h.size());
    return f;
}
static std::vector<uint16_t> f32_to_f16(const std::vector<float>& f) {
    std::vector<uint16_t> h(f.size());
    ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t*)h.data(), (int64_t)f.size());
    return h;
}
// 4-bit weights the way XNNPACK reads them: row-major [rows, cols], element k in byte k/2, EVEN k in
// the LOW nibble, stored unsigned with zero point 8.
static void quant_per_channel_4bit(const std::vector<float>& w, int64_t rows, int64_t cols,
                                   std::vector<uint8_t>& q, std::vector<float>& scale) {
    q.assign(rows * cols / 2, 0); scale.resize(rows);
    for (int64_t r = 0; r < rows; ++r) {
        float amax = 0;
        for (int64_t c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(w[r * cols + c]));
        const float s = amax > 0 ? amax / 7.0f : 1.0f;
        scale[r] = s;
        for (int64_t c = 0; c < cols; ++c) {
            const int v = 8 + (int)std::lround(std::clamp(w[r * cols + c] / s, -7.0f, 7.0f));
            q[(r * cols + c) / 2] |= (uint8_t)(c % 2 ? v << 4 : v);
        }
    }
}
// XNNPACK's qb4w cannot take ggml's Q4_0 blocks as they are. Q4_0's scale is SIGNED -- d = max/-8,
// so the block's largest-magnitude element always lands on nibble 0 -- and qb4w refuses any scale that
// is not positive (create returns invalid_parameter; fully-connected-nc.c's check_block_size). With
// d < 0 the flip q -> 16 - q has no image for nibble 0, so about half the blocks would lose their
// largest element. This quantizes the same F32 weights block-32 with a POSITIVE scale instead: the
// largest element lands on -8 when it is negative (exactly Q4_0's grid) and on +7 when it is positive
// (one level coarser than Q4_0 in that block). The scale is rounded to BF16 -- qb4w's scale type --
// BEFORE the nibbles are chosen, so they are consistent with the scale XNNPACK will actually use.
static void quant_block32_4bit_bf16(const std::vector<float>& w, int64_t rows, int64_t cols,
                                    std::vector<uint8_t>& q, std::vector<uint16_t>& scale_bf16) {
    const int64_t nb = cols / 32;
    q.assign(rows * cols / 2, 0); scale_bf16.resize(rows * nb);
    for (int64_t r = 0; r < rows; ++r) for (int64_t b = 0; b < nb; ++b) {
        const float* x = w.data() + r * cols + b * 32;
        float big = 0;
        for (int j = 0; j < 32; ++j) if (std::fabs(x[j]) > std::fabs(big)) big = x[j];
        float s = big < 0 ? -big / 8.0f : big > 0 ? big / 7.0f : 1.0f;
        uint32_t u; std::memcpy(&u, &s, 4);
        const uint16_t h = (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
        scale_bf16[r * nb + b] = h;
        u = (uint32_t)h << 16; std::memcpy(&s, &u, 4);
        for (int j = 0; j < 32; ++j) {
            const int v = 8 + (int)std::lround(std::clamp(x[j] / s, -8.0f, 7.0f));
            const int64_t k = r * cols + b * 32 + j;
            q[k / 2] |= (uint8_t)(k % 2 ? v << 4 : v);
        }
    }
}

// ggml's own int8 GEMM for Q8_0 / Q4_0: the CPU backend's "CPU_REPACK" extra buffer type re-lays the
// weight as interleaved blocks at upload (repack.cpp's ggml_repack_get_optimal_repack_type): Q8_0 for
// NEON with dotprod or i8mm only, Q4_0 for those AND for AVX2. loom's engine never places weights
// there, so this arm is what ggml ALREADY has that loom does not use. Where it declines, it reads n/a.
static ggml_backend_buffer_type_t cpu_repack_buft(ggml_backend_t be) {
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!fn) return nullptr;
    for (ggml_backend_buffer_type_t* p = fn(dev); p && *p; ++p)
        if (std::string(ggml_backend_buft_name(*p)) == "CPU_REPACK") return *p;
    return nullptr;
}

struct Arm {
    std::string name;
    std::function<void()> run;              // empty: this hardware declines the operator
    std::function<std::vector<float>()> out;  // result in the REFERENCE layout
};
struct LowRow { std::string name; double flop, weight; std::vector<double> ms, err; double layout_ms; };

static int run_lowp(const std::string& mode, ggml_backend_t be, pthreadpool_t pool, int reps) {
    const bool q8 = mode == "q8", q4 = mode == "q4", quant = q8 || q4;
    const ggml_type wt = q4 ? GGML_TYPE_Q4_0 : q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    std::vector<std::string> names =
        q4 ? std::vector<std::string>{"ggml q4_0", "ggml rpk", "xnn qb4w", "xnn qc4w", "xnn qs8c4", "xnn qd8c8"}
      : q8 ? std::vector<std::string>{"ggml q8_0", "ggml rpk", "xnn qd8", "xnn qs8"}
           : std::vector<std::string>{"ggml f16", "xnn f16"};
    std::vector<LowRow> rows;
    ggml_backend_buffer_type_t repack = quant ? cpu_repack_buft(be) : nullptr;

    auto race = [&](std::vector<Arm>& arms) {
        for (int w = 0; w < 2; ++w) for (Arm& a : arms) if (a.run) a.run();
        std::vector<std::vector<double>> t(arms.size());
        for (int r = 0; r < reps; ++r)
            for (size_t i = 0; i < arms.size(); ++i) {
                if (!arms[i].run) continue;
                const double t0 = now(); arms[i].run(); t[i].push_back(now() - t0);
            }
        static const bool dump = std::getenv("BENCH_DUMP_REPS") != nullptr;   // every repetition, to stderr
        std::vector<double> ms;
        for (size_t i = 0; i < t.size(); ++i) {
            ms.push_back(t[i].empty() ? NAN : median(t[i]));
            if (dump && !t[i].empty()) {
                std::fprintf(stderr, "reps %s:", arms[i].name.c_str());
                for (double x : t[i]) std::fprintf(stderr, " %.1f", x * 1e3);
                std::fprintf(stderr, "\n");
            }
        }
        return ms;
    };
    auto weights_for = [&](ggml_tensor* t, const std::vector<float>& w) {   // w is t's F32 content
        if (wt == GGML_TYPE_F16) { auto h = f32_to_f16(w); ggml_backend_tensor_set(t, h.data(), 0, h.size() * 2); return; }
        const int64_t rowlen = t->ne[0], nrows = (int64_t)w.size() / rowlen;
        std::vector<char> q(ggml_row_size(wt, rowlen) * nrows);
        ggml_quantize_chunk(wt, w.data(), q.data(), 0, nrows, rowlen, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    };

    auto alloc = [&](GgmlRun& g, auto build) {
        ggml_init_params ip = {256u * 1024 * 1024, nullptr, true};
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, 64, false);
        g.out = build(g.ctx);
        ggml_build_forward_expand(g.gf, g.out);
        g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { std::fprintf(stderr, "ggml alloc failed\n"); std::exit(1); }
    };

    // ---------------------------------------------------------------- GEMMs (same list as F32 mode)
    struct G { const char* tag; int64_t K, M, N; double w; };
    const G gemms[] = {
        {"vits", 224, 73472, 32, 1}, {"vits", 160, 73472, 32, 1}, {"vits", 96, 73472, 32, 1},
        {"vits", 448, 18368, 64, 1}, {"vits", 320, 18368, 64, 1}, {"vits", 192, 18368, 64, 1},
        {"vits", 896, 2296, 128, 1}, {"vits", 640, 2296, 128, 1}, {"vits", 384, 2296, 128, 1},
        {"vits", 960, 288, 384, 1}, {"vits", 1344, 288, 256, 1},
        {"whisper qkv", 768, 1500, 768, 36}, {"whisper fc1", 768, 1500, 3072, 12},
        {"whisper fc2", 3072, 1500, 768, 12},
    };
    for (const G& s : gemms) {
        std::vector<float> A(s.M * s.K), W(s.N * s.K), R(s.M * s.N);
        fill(A, 97, 0.5f);
        if (q4) fill_normal(W, 53, 0.1f); else fill(W, 53, 0.3f);
        {   // F32 reference
            xnn_operator_t op = nullptr;
            check(xnn_create_fully_connected_nc_f32(s.K, s.N, s.K, s.N, W.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &op), "create fc ref");
            check(xnn_reshape_fully_connected_nc_f32(op, s.M, pool), "reshape fc ref");
            check(xnn_setup_fully_connected_nc_f32(op, A.data(), R.data()), "setup fc ref");
            check(xnn_run_operator(op, pool), "run fc ref");
            xnn_delete_operator(op);
        }
        GgmlRun g;
        ggml_tensor *ta = nullptr, *tb = nullptr;
        alloc(g, [&](ggml_context* c) {
            ta = ggml_new_tensor_2d(c, wt, s.K, s.N); ggml_set_input(ta); ggml_set_output(ta);
            tb = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.K, s.M); ggml_set_input(tb); ggml_set_output(tb);
            return ggml_mul_mat(c, ta, tb);
        });
        weights_for(ta, W);
        ggml_backend_tensor_set(tb, A.data(), 0, A.size() * 4);
        std::vector<Arm> arms;
        arms.push_back({names[0], [&] { ggml_backend_graph_compute(be, g.gf); },
                        [&] { std::vector<float> c(s.M * s.N); ggml_backend_tensor_get(g.out, c.data(), 0, c.size() * 4); return c; }});
        // The same quantized weight, uploaded through CPU_REPACK (which repacks it in set_tensor).
        GgmlRun g2;
        ggml_context* wctx = nullptr;
        ggml_backend_buffer_t wbuf = nullptr;
        if (quant) {
            bool repacked = false;
            if (repack) {
                ggml_init_params wp = {ggml_tensor_overhead() * 4, nullptr, true};
                wctx = ggml_init(wp);
                ggml_tensor* tw = ggml_new_tensor_2d(wctx, wt, s.K, s.N);
                wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, repack);
                // init_tensor stores the chosen repack in `extra`, and set_tensor dereferences it
                // unchecked -- so a tensor no repack applies to must not be uploaded at all.
                repacked = wbuf && tw->extra != nullptr;
                if (repacked) weights_for(tw, W);
                if (repacked) {
                    ggml_tensor* tb2 = nullptr;
                    alloc(g2, [&](ggml_context* c) {
                        tb2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.K, s.M); ggml_set_input(tb2); ggml_set_output(tb2);
                        return ggml_mul_mat(c, tw, tb2);
                    });
                    ggml_backend_tensor_set(tb2, A.data(), 0, A.size() * 4);
                }
            }
            if (repacked)
                arms.push_back({names[1], [&] { ggml_backend_graph_compute(be, g2.gf); },
                                [&] { std::vector<float> c(s.M * s.N); ggml_backend_tensor_get(g2.out, c.data(), 0, c.size() * 4); return c; }});
            else
                arms.push_back({names[1], {}, {}});
        }
        std::vector<xnn_operator_t> ops;
        // Buffers shared by the XNNPACK arms (kept alive for the lambdas).
        std::vector<uint16_t> A16, W16, C16(s.M * s.N);
        std::vector<int8_t> Wq, Aq(s.M * s.K + 64), Aq_s, Cq(s.M * s.N);
        std::vector<uint8_t> Wb4, Wc4;
        std::vector<uint16_t> Wb4s;
        std::vector<float> Ws, Wc4s, C(s.M * s.N), rowsum(s.M + XNN_EXTRA_QUANTIZATION_PARAMS);
        std::deque<std::vector<float>> Cs;   // one output per q4 arm; a deque keeps their addresses
        std::vector<std::unique_ptr<Workspace>> works;
        std::vector<int32_t> bias32(s.N, 0);
        std::vector<xnn_quantization_params> qp(s.M + XNN_EXTRA_QUANTIZATION_PARAMS);
        float sx = 1, sy = 1;
        std::unique_ptr<Workspace> work;
        if (!quant) {
            A16 = f32_to_f16(A); W16 = f32_to_f16(W);
            xnn_operator_t op = nullptr;
            const xnn_status st = xnn_create_fully_connected_nc_f16(s.K, s.N, s.K, s.N, W16.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &op);
            if (st == xnn_status_success) {
                check(xnn_reshape_fully_connected_nc_f16(op, s.M, pool), "reshape fc f16");
                check(xnn_setup_fully_connected_nc_f16(op, A16.data(), C16.data()), "setup fc f16");
                ops.push_back(op);
                arms.push_back({names[1], [&, op] { check(xnn_run_operator(op, pool), "run fc f16"); }, [&] { return f16_to_f32(C16); }});
            } else {
                arms.push_back({names[1], {}, {}});
            }
        } else if (q4) {
            // One activation convert, re-run INSIDE every qd8 arm's timing, as in q8.
            xnn_operator_t cv = nullptr;
            check(xnn_create_convert_nc_f32_qd8(0, &cv), "create convert");
            check(xnn_reshape_convert_nc_f32_qd8(cv, s.M, s.K, s.K, s.K, pool), "reshape convert");
            check(xnn_setup_convert_nc_f32_qd8(cv, A.data(), Aq.data(), rowsum.data(), qp.data()), "setup convert");
            ops.push_back(cv);
            auto dyn_arm = [&, cv](const std::string& nm, xnn_operator_t fc, auto reshape, auto setup) {
                size_t ws = 0;
                check(reshape(fc, s.M, &ws, pool), ("reshape " + nm).c_str());
                works.push_back(std::make_unique<Workspace>(ws));
                std::vector<float>* out = &Cs.emplace_back(s.M * s.N);
                check(setup(fc, Aq.data(), out->data(), works.back()->p, qp.data()), ("setup " + nm).c_str());
                ops.push_back(fc);
                arms.push_back({nm, [&, cv, fc] { check(xnn_run_operator(cv, pool), "run convert"); check(xnn_run_operator(fc, pool), "run fc"); },
                                [out] { return *out; }});
            };
            {   // qb4w: block-32 like Q4_0, but with the positive scale it requires (see its quantizer)
                quant_block32_4bit_bf16(W, s.N, s.K, Wb4, Wb4s);
                xnn_operator_t fc = nullptr;
                check(xnn_create_fully_connected_nc_qd8_f32_qb4w(s.K, s.N, s.K, s.N, 32, 8, Wb4s.data(), Wb4.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &fc), "create fc qb4w");
                dyn_arm(names[2], fc, xnn_reshape_fully_connected_nc_qd8_f32_qb4w, xnn_setup_fully_connected_nc_qd8_f32_qb4w);
            }
            quant_per_channel_4bit(W, s.N, s.K, Wc4, Wc4s);
            {
                xnn_operator_t fc = nullptr;
                check(xnn_create_fully_connected_nc_qd8_f32_qc4w(s.K, s.N, s.K, s.N, 8, Wc4s.data(), Wc4.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &fc), "create fc qc4w");
                dyn_arm(names[3], fc, xnn_reshape_fully_connected_nc_qd8_f32_qc4w, xnn_setup_fully_connected_nc_qd8_f32_qc4w);
            }
            sx = amax_of(A) / 127.0f; sy = amax_of(R) / 127.0f;
            quant_tensor(A, sx, Aq_s);
            xnn_operator_t fs = nullptr;
            check(xnn_create_fully_connected_nc_qs8_qc4w(s.K, s.N, s.K, s.N, 0, sx, 8, Wc4s.data(), Wc4.data(), bias32.data(), 0, sy, -128, 127, 0, nullptr, &fs), "create fc qs8c4");
            check(xnn_reshape_fully_connected_nc_qs8_qc4w(fs, s.M, pool), "reshape fc qs8c4");
            check(xnn_setup_fully_connected_nc_qs8_qc4w(fs, Aq_s.data(), Cq.data()), "setup fc qs8c4");
            ops.push_back(fs);
            arms.push_back({names[4], [&, fs] { check(xnn_run_operator(fs, pool), "run fc qs8c4"); },
                            [&] { std::vector<float> c(Cq.size()); for (size_t i = 0; i < c.size(); ++i) c[i] = sy * Cq[i]; return c; }});
            quant_per_channel(W, s.N, s.K, Wq, Ws);
            {
                xnn_operator_t fc = nullptr;
                check(xnn_create_fully_connected_nc_qd8_f32_qc8w(s.K, s.N, s.K, s.N, Ws.data(), Wq.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &fc), "create fc qd8c8");
                dyn_arm(names[5], fc, xnn_reshape_fully_connected_nc_qd8_f32_qc8w, xnn_setup_fully_connected_nc_qd8_f32_qc8w);
            }
        } else {
            quant_per_channel(W, s.N, s.K, Wq, Ws);
            xnn_operator_t cv = nullptr, fc = nullptr;
            check(xnn_create_convert_nc_f32_qd8(0, &cv), "create convert");
            check(xnn_reshape_convert_nc_f32_qd8(cv, s.M, s.K, s.K, s.K, pool), "reshape convert");
            check(xnn_setup_convert_nc_f32_qd8(cv, A.data(), Aq.data(), rowsum.data(), qp.data()), "setup convert");
            check(xnn_create_fully_connected_nc_qd8_f32_qc8w(s.K, s.N, s.K, s.N, Ws.data(), Wq.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &fc), "create fc qd8");
            size_t ws = 0;
            check(xnn_reshape_fully_connected_nc_qd8_f32_qc8w(fc, s.M, &ws, pool), "reshape fc qd8");
            work = std::make_unique<Workspace>(ws);
            check(xnn_setup_fully_connected_nc_qd8_f32_qc8w(fc, Aq.data(), C.data(), work->p, qp.data()), "setup fc qd8");
            ops.push_back(cv); ops.push_back(fc);
            arms.push_back({names[2], [&, cv, fc] { check(xnn_run_operator(cv, pool), "run convert"); check(xnn_run_operator(fc, pool), "run fc qd8"); },
                            [&] { return C; }});
            sx = amax_of(A) / 127.0f; sy = amax_of(R) / 127.0f;
            quant_tensor(A, sx, Aq_s);
            xnn_operator_t fs = nullptr;
            check(xnn_create_fully_connected_nc_qs8_qc8w(s.K, s.N, s.K, s.N, 0, sx, Ws.data(), Wq.data(), bias32.data(), 0, sy, -128, 127, 0, nullptr, &fs), "create fc qs8");
            check(xnn_reshape_fully_connected_nc_qs8_qc8w(fs, s.M, pool), "reshape fc qs8");
            check(xnn_setup_fully_connected_nc_qs8_qc8w(fs, Aq_s.data(), Cq.data()), "setup fc qs8");
            ops.push_back(fs);
            arms.push_back({names[3], [&, fs] { check(xnn_run_operator(fs, pool), "run fc qs8"); },
                            [&] { std::vector<float> c(Cq.size()); for (size_t i = 0; i < c.size(); ++i) c[i] = sy * Cq[i]; return c; }});
        }
        std::vector<double> ms = race(arms), err;
        for (Arm& a : arms) err.push_back(a.run ? max_rel(a.out(), R) : NAN);
        char nm[96]; std::snprintf(nm, sizeof nm, "gemm %s %lldx%lldx%lld", s.tag, (long long)s.K, (long long)s.M, (long long)s.N);
        rows.push_back({nm, 2.0 * s.K * s.M * s.N, s.w, ms, err, 0});
        for (xnn_operator_t op : ops) xnn_delete_operator(op);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
    }

    // ---------------------------------------------------------------- VITS conv1d (same list)
    struct Cv { int64_t IC, OC, kw, L; int calls; int64_t dil; };
    const Cv convs[] = {
        { 32,  32, 7, 73472, 1, 3}, { 32,  32, 7, 73472, 1,12}, { 32,  32, 5, 73472, 1, 2},
        { 32,  32, 5, 73472, 1, 6}, { 32,  32, 3, 73472, 1, 1}, { 32,  32, 3, 73472, 1, 2},
        { 64,  64, 7, 18368, 1, 3}, { 64,  64, 7, 18368, 1,12}, { 64,  64, 5, 18368, 1, 2},
        { 64,  64, 5, 18368, 1, 6}, { 64,  64, 3, 18368, 1, 1}, { 64,  64, 3, 18368, 1, 2},
        {128, 128, 7,  2296, 1, 3}, {128, 128, 7,  2296, 1,12}, {128, 128, 5,  2296, 1, 2},
        {128, 128, 5,  2296, 1, 6}, {128, 128, 3,  2296, 1, 1}, {128, 128, 3,  2296, 1, 2},
        {192, 384, 5,   287,16, 1}, {768, 768, 3,   100,12, 1},
    };
    for (const Cv& c : convs) {
        const int64_t pad = c.dil * (c.kw - 1) / 2;
        std::vector<float> xg(c.L * c.IC), kg(c.kw * c.IC * c.OC);
        fill(xg, 101, 0.6f);
        if (q4) fill_normal(kg, 59, 0.07f); else fill(kg, 59, 0.2f);
        std::vector<float> xx(c.L * c.IC), kx(c.OC * c.kw * c.IC), R(c.L * c.OC);
        for (int64_t i = 0; i < c.IC; ++i) for (int64_t t = 0; t < c.L; ++t) xx[t * c.IC + i] = xg[i * c.L + t];
        for (int64_t o = 0; o < c.OC; ++o) for (int64_t i = 0; i < c.IC; ++i) for (int64_t k = 0; k < c.kw; ++k)
            kx[(o * c.kw + k) * c.IC + i] = kg[(o * c.IC + i) * c.kw + k];
        auto make_conv = [&](auto create) {   // shared geometry for every XNNPACK conv arm
            xnn_operator_t op = nullptr;
            check(create((uint32_t)pad, (uint32_t)c.kw, (uint32_t)c.dil, &op), "create conv");
            return op;
        };
        {   // F32 reference
            xnn_operator_t op = make_conv([&](uint32_t p, uint32_t kw, uint32_t d, xnn_operator_t* o) {
                return xnn_create_convolution2d_nhwc_f32(0, p, 0, p, 1, kw, 1, 1, 1, d, 1, c.IC, c.OC, c.IC, c.OC, kx.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, o); });
            size_t ws = 0, oh = 0, ow = 0;
            check(xnn_reshape_convolution2d_nhwc_f32(op, 1, 1, c.L, &ws, &oh, &ow, pool), "reshape conv ref");
            Workspace w(ws);
            check(xnn_setup_convolution2d_nhwc_f32(op, w.p, xx.data(), R.data()), "setup conv ref");
            check(xnn_run_operator(op, pool), "run conv ref");
            xnn_delete_operator(op);
        }
        GgmlRun g;
        ggml_tensor *tk = nullptr, *tx = nullptr;
        alloc(g, [&](ggml_context* ctx) {
            tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.L, 1, c.IC, 1); ggml_set_input(tx); ggml_set_output(tx);
            if (!quant) {
                tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kw, 1, c.IC, c.OC); ggml_set_input(tk); ggml_set_output(tk);
                return ggml_conv_2d_direct(ctx, tk, tx, 1, 1, (int)pad, 0, (int)c.dil, 1);
            }
            // The folded [IC*KW, OC] kernel a Q8_0 / Q4_0 export carries: a pure reinterpretation of
            // the [KW, 1, IC, OC] order (ggml-0013), so quantizing kg in rows of IC*KW IS that kernel.
            tk = ggml_new_tensor_2d(ctx, wt, c.IC * c.kw, c.OC); ggml_set_input(tk); ggml_set_output(tk);
            return ggml_conv_2d_direct_packed(ctx, tk, tx, 1, 1, (int)pad, 0, (int)c.dil, 1, (int)c.kw, 1, (int)c.IC);
        });
        weights_for(tk, kg);
        ggml_backend_tensor_set(tx, xg.data(), 0, xg.size() * 4);
        std::vector<Arm> arms;
        arms.push_back({names[0], [&] { ggml_backend_graph_compute(be, g.gf); }, [&] {
            std::vector<float> yg(c.L * c.OC), y(c.L * c.OC);
            ggml_backend_tensor_get(g.out, yg.data(), 0, yg.size() * 4);
            for (int64_t o = 0; o < c.OC; ++o) for (int64_t t = 0; t < c.L; ++t) y[t * c.OC + o] = yg[o * c.L + t];
            return y; }});
        if (quant) arms.push_back({names[1], {}, {}});   // a repacked kernel cannot feed ggml-0013's dequantize
        if (q4) for (int i = 2; i <= 4; ++i) arms.push_back({names[i], {}, {}});   // XNNPACK has no 4-bit conv
        std::vector<xnn_operator_t> ops;
        std::vector<std::unique_ptr<Workspace>> works;
        std::vector<uint16_t> x16, k16, y16(c.L * c.OC);
        std::vector<int8_t> kq, xq(c.L * c.IC + 64), xq_s, yq(c.L * c.OC);
        std::vector<float> ks, y(c.L * c.OC), rowsum(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
        std::vector<int32_t> bias32(c.OC, 0);
        std::vector<xnn_quantization_params> qp(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
        float sx = 1, sy = 1;
        size_t ws = 0, oh = 0, ow = 0;
        if (!quant) {
            x16 = f32_to_f16(xx); k16 = f32_to_f16(kx);
            xnn_operator_t op = nullptr;
            const xnn_status st = xnn_create_convolution2d_nhwc_f16(0, (uint32_t)pad, 0, (uint32_t)pad, 1, (uint32_t)c.kw, 1, 1, 1, (uint32_t)c.dil, 1,
                                                                    c.IC, c.OC, c.IC, c.OC, k16.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, &op);
            if (st == xnn_status_success) {
                check(xnn_reshape_convolution2d_nhwc_f16(op, 1, 1, c.L, &ws, &oh, &ow, pool), "reshape conv f16");
                works.push_back(std::make_unique<Workspace>(ws));
                check(xnn_setup_convolution2d_nhwc_f16(op, works.back()->p, x16.data(), y16.data()), "setup conv f16");
                ops.push_back(op);
                arms.push_back({names[1], [&, op] { check(xnn_run_operator(op, pool), "run conv f16"); }, [&] { return f16_to_f32(y16); }});
            } else {
                arms.push_back({names[1], {}, {}});
            }
        } else {
            quant_per_channel(kx, c.OC, c.kw * c.IC, kq, ks);
            xnn_operator_t cv = nullptr;
            check(xnn_create_convert_nc_f32_qd8(0, &cv), "create convert");
            // One quantization scale per IMAGE: XNNPACK's dynamic-quant igemm indexes its params by batch.
            check(xnn_reshape_convert_nc_f32_qd8(cv, 1, c.L * c.IC, c.L * c.IC, c.L * c.IC, pool), "reshape convert");
            check(xnn_setup_convert_nc_f32_qd8(cv, xx.data(), xq.data(), rowsum.data(), qp.data()), "setup convert");
            xnn_operator_t op = make_conv([&](uint32_t p, uint32_t kw, uint32_t d, xnn_operator_t* o) {
                return xnn_create_convolution2d_nhwc_qd8_f32_qc8w(0, p, 0, p, 1, kw, 1, 1, 1, d, 1, c.IC, c.OC, c.IC, c.OC, ks.data(), kq.data(), nullptr, -INFINITY, INFINITY, 0, nullptr, o); });
            check(xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(op, 1, 1, c.L, &ws, &oh, &ow, pool), "reshape conv qd8");
            works.push_back(std::make_unique<Workspace>(ws));
            check(xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(op, works.back()->p, xq.data(), y.data(), qp.data()), "setup conv qd8");
            ops.push_back(cv); ops.push_back(op);
            arms.push_back({names[q4 ? 5 : 2], [&, cv, op] { check(xnn_run_operator(cv, pool), "run convert"); check(xnn_run_operator(op, pool), "run conv qd8"); },
                            [&] { return y; }});
        }
        if (q8) {   // the static int8 conv; q4's static arm is a 4-bit GEMM, which has no conv form
            sx = amax_of(xx) / 127.0f; sy = amax_of(R) / 127.0f;
            quant_tensor(xx, sx, xq_s);
            xnn_operator_t os = make_conv([&](uint32_t p, uint32_t kw, uint32_t d, xnn_operator_t* o) {
                return xnn_create_convolution2d_nhwc_qs8_qc8w(0, p, 0, p, 1, kw, 1, 1, 1, d, 1, c.IC, c.OC, c.IC, c.OC, 0, sx, ks.data(), kq.data(), bias32.data(), 0, sy, -128, 127, 0, nullptr, o); });
            check(xnn_reshape_convolution2d_nhwc_qs8_qc8w(os, 1, 1, c.L, &ws, &oh, &ow, pool), "reshape conv qs8");
            works.push_back(std::make_unique<Workspace>(ws));
            check(xnn_setup_convolution2d_nhwc_qs8_qc8w(os, works.back()->p, xq_s.data(), yq.data()), "setup conv qs8");
            ops.push_back(os);
            arms.push_back({names[3], [&, os] { check(xnn_run_operator(os, pool), "run conv qs8"); },
                            [&] { std::vector<float> o(yq.size()); for (size_t i = 0; i < o.size(); ++i) o[i] = sy * yq[i]; return o; }});
        }
        std::vector<double> ms = race(arms), err;
        for (Arm& a : arms) err.push_back(a.run ? max_rel(a.out(), R) : NAN);
        // Same naive transpose tax as the F32 mode: an upper bound, F32-sized even for the int8 arms.
        std::vector<double> tl;
        std::vector<float> ygt(c.L * c.OC);
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int64_t i = 0; i < c.IC; ++i) for (int64_t t = 0; t < c.L; ++t) xx[t * c.IC + i] = xg[i * c.L + t];
            for (int64_t t = 0; t < c.L; ++t) for (int64_t o = 0; o < c.OC; ++o) ygt[o * c.L + t] = R[t * c.OC + o];
            tl.push_back(now() - t0);
        }
        char nm[96]; std::snprintf(nm, sizeof nm, "conv %lldx%lld k%lld d%lld L%lld", (long long)c.IC,
                                   (long long)c.OC, (long long)c.kw, (long long)c.dil, (long long)c.L);
        rows.push_back({nm, 2.0 * c.IC * c.OC * c.kw * c.L, (double)c.calls, ms, err, median(tl)});
        for (xnn_operator_t op : ops) xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- report
    const size_t na = names.size();
    std::printf("\n%-34s", "shape");
    for (auto& n : names) std::printf(" %10s", (n + " ms").c_str());
    for (size_t i = 1; i < na; ++i) std::printf(" %9s", ("/" + names[i].substr(names[i].find(' ') + 1)).c_str());
    for (size_t i = 0; i < na; ++i) std::printf(" %9s", ("err " + (i ? names[i].substr(names[i].find(' ') + 1) : std::string("ggml"))).c_str());
    std::printf("\n");
    std::vector<std::vector<double>> sum(3, std::vector<double>(na, 0));
    double sum_l = 0;
    for (const LowRow& r : rows) {
        std::printf("%-34s", r.name.c_str());
        for (double m : r.ms) if (std::isnan(m)) std::printf(" %10s", "n/a"); else std::printf(" %10.3f", m * 1e3);
        for (size_t i = 1; i < na; ++i) if (std::isnan(r.ms[i])) std::printf(" %9s", "n/a"); else std::printf(" %9.3f", r.ms[0] / r.ms[i]);
        for (double e : r.err) if (std::isnan(e)) std::printf(" %9s", "n/a"); else std::printf(" %9.1e", e);
        std::printf("\n");
        const int grp = r.name.rfind("conv", 0) == 0 ? 2 : r.name.find("whisper") != std::string::npos ? 1 : 0;
        for (size_t i = 0; i < na; ++i) sum[grp][i] += r.weight * r.ms[i];
        if (grp == 2) sum_l += r.weight * r.layout_ms;
    }
    const char* grp_name[3] = {"VITS GEMMs (one synthesis)", "whisper linears (weighted)", "VITS convs (weighted)"};
    std::printf("\n");
    for (int g = 0; g < 3; ++g) {
        std::printf("%-28s %s %8.1f ms", grp_name[g], names[0].c_str(), sum[g][0] * 1e3);
        for (size_t i = 1; i < na; ++i)
            if (std::isnan(sum[g][i])) std::printf("  %s n/a", names[i].c_str());
            else std::printf("  %s %8.1f ms (ggml/xnn %.3f)", names[i].c_str(), sum[g][i] * 1e3, sum[g][0] / sum[g][i]);
        std::printf("\n");
    }
    std::printf("VITS convs incl. layout tax  ");
    for (size_t i = 1; i < na; ++i)
        if (!std::isnan(sum[2][i])) std::printf("  %s ggml/(xnn+layout) %.3f", names[i].c_str(), sum[2][0] / (sum[2][i] + sum_l));
    std::printf("\n");
    return 0;
}

// ------------------------------------------------------------------------------------------------
// CITRINET MODE (`citrinet` as the fourth argument): ADR-077's gate 2. Citrinet-256-ls's 1x1 convs,
// Y[N][T] = W[N][K] . X[K][T] + b with K = N = 256, at the four sequence lengths jfk runs them at,
// weighted by how often it does (7 calls at T = 1101, 36 at 551, 42 at 276, 46 at 138). Bias included
// in every arm, because the model pays it either as a fused epilogue or as an ADD node.
//
//   ggml q8_0   ggml_conv_2d_direct_packed with a Q8_0 [256, 256] kernel + ADD: what the Q8_0 card
//               runs today (ggml-0013 dequantizes the kernel to F32 every call).
//   ggml f32    ggml_conv_2d_direct with an F32 kernel + ADD: the F32 card's path.
//   ggml f32rf  the same graph with the VFP in RunFast (FPSCR FZ | DN, ADR-078). ARMv6 only, and only
//               at one thread: FPSCR is per thread and only the calling thread's is set.
//   ggml q8mm   mul_mat(Q8_0 W, X) + ADD on a CHANNELS-LAST X: what a channels-last export would run.
//   xnn qd8     XNNPACK dynamic int8 (convert inside the timing), channels-last.
//   xnn qs8     XNNPACK static int8, channels-last; scales from the reference's range, so a ceiling.
//   loom i8cf   ADR-077's ARMv6 kernel on today's CHANNELS-FIRST X: per-time-step symmetric quantize
//               to int16 [T][K] (a strided transpose), per-channel int8 W held as int16, a 2x4 SMLAD
//               tile in asm (scripts/gen_armv6_i8_tile.py, K = 256 unrolled), F32 epilogue.
//   loom i8cl   the same kernel on a CHANNELS-LAST X and Y: the quantize becomes one sequential pass.
//   loom qcf / loom qcl   the two quantize passes alone, so GEMM-only = arm - quantize.
// The loom arms exist only where the asm does (ARMv6: SIMD32 without NEON); elsewhere they read n/a.
//
// Build on ARMv6 adds the generated tile:  python3 scripts/gen_armv6_i8_tile.py tile.S  and  tile.S
// on the g++ line above.
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON)
#include <arm_acle.h>
#define BENCH_ARMV6 1
extern "C" void tile2x4_k256(const uint32_t* x, const uint32_t* w, int32_t* out);
static inline unsigned fpscr_get() { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static inline void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f)); }
// vcvtr rounds by FPSCR (nearest-even), as lrintf does -- but lrintf is an out-of-line libm call on
// VFPv2 (math-errno), which made the first quantize 2.2x slower.
static inline int32_t vcvtr_s32(float x) {
    float r;
    __asm__("vcvtr.s32.f32 %0, %1" : "=t"(r) : "t"(x));
    int32_t i; std::memcpy(&i, &r, 4);
    return i;
}
namespace cit {
constexpr int K = 256, N = 256, KW = K / 2, MC = 24;
// Channels-first X [K][T] -> int16 pairs [T][KW]. The per-t max is taken on the bit patterns (sign
// cleared; unsigned order is IEEE order for non-NaN), 8 time steps at a time so each k reads one 32 B
// line. The convert goes 4 time steps x 2 k at a time in asm, each lane with its own scale (s24-s27),
// for the reason given at quantize_cl; the last T % 4 steps take the C path.
static void quantize_cf(const float* X, int T, uint32_t* Xq, float* sx) {
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int tb = std::min(8, T - t0);
        uint32_t mx[8] = {0};
        for (int k = 0; k < K; ++k) {
            const uint32_t* x = (const uint32_t*)(X + (size_t)k * T + t0);
            for (int j = 0; j < tb; ++j) { const uint32_t v = x[j] & 0x7fffffffu; mx[j] = v > mx[j] ? v : mx[j]; }
        }
        float inv[8];
        for (int j = 0; j < tb; ++j) {
            float m; std::memcpy(&m, &mx[j], 4);
            sx[t0 + j] = m > 0 ? m / 127.0f : 1.0f; inv[j] = 1.0f / sx[t0 + j];
        }
        int g = 0;
        for (; g + 4 <= tb; g += 4) {
            const float* x0 = X + t0 + g;
            uint32_t* q = Xq + (size_t)(t0 + g) * KW;
            __asm__ volatile("vldmia %0, {s24-s27}" : : "r"(inv + g) : "s24", "s25", "s26", "s27");
            for (int k = 0; k < K; k += 2, x0 += 2 * (size_t)T, ++q)
                __asm__ volatile(
                    "vldmia %[a], {s16-s19}\n vldmia %[b], {s20-s23}\n"
                    "vmul.f32 s16, s16, s24\n vmul.f32 s17, s17, s25\n vmul.f32 s18, s18, s26\n vmul.f32 s19, s19, s27\n"
                    "vmul.f32 s20, s20, s24\n vmul.f32 s21, s21, s25\n vmul.f32 s22, s22, s26\n vmul.f32 s23, s23, s27\n"
                    "vcvtr.s32.f32 s16, s16\n vcvtr.s32.f32 s17, s17\n vcvtr.s32.f32 s18, s18\n vcvtr.s32.f32 s19, s19\n"
                    "vcvtr.s32.f32 s20, s20\n vcvtr.s32.f32 s21, s21\n vcvtr.s32.f32 s22, s22\n vcvtr.s32.f32 s23, s23\n"
                    "vmov r0, r1, s16, s17\n vmov r2, r3, s20, s21\n"   // k for t, t+1; then k+1
                    "pkhbt r0, r0, r2, lsl #16\n pkhbt r1, r1, r3, lsl #16\n"
                    "str r0, [%[q]]\n str r1, [%[q], #512]\n"
                    "vmov r0, r1, s18, s19\n vmov r2, r3, s22, s23\n"
                    "pkhbt r0, r0, r2, lsl #16\n pkhbt r1, r1, r3, lsl #16\n"
                    "str r0, [%[q], #1024]\n str r1, [%[q], #1536]\n"
                    : : [a] "r"(x0), [b] "r"(x0 + T), [q] "r"(q)
                    : "r0", "r1", "r2", "r3", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "memory");
        }
        for (; g < tb; ++g)
            for (int k = 0; k < K; k += 2) {
                const float* x0 = X + (size_t)k * T + t0 + g;
                Xq[(size_t)(t0 + g) * KW + k / 2] = (uint16_t)vcvtr_s32(x0[0] * inv[g]) | ((uint32_t)vcvtr_s32(x0[T] * inv[g]) << 16);
            }
    }
}
// Channels-last X [T][K]: the same arithmetic, read sequentially.
static void quantize_cl(const float* X, int T, uint32_t* Xq, float* sx) {
    for (int t = 0; t < T; ++t) {
        const float* x = X + (size_t)t * K;
        const uint32_t* xb = (const uint32_t*)x;
        uint32_t m0 = 0, m1 = 0;
        for (int k = 0; k < K; k += 2) {
            const uint32_t a = xb[k] & 0x7fffffffu, b = xb[k + 1] & 0x7fffffffu;
            m0 = a > m0 ? a : m0; m1 = b > m1 ? b : m1;
        }
        const uint32_t mb = m0 > m1 ? m0 : m1;
        float m; std::memcpy(&m, &mb, 4);
        sx[t] = m > 0 ? m / 127.0f : 1.0f;
        const float inv = 1.0f / sx[t];
        uint32_t* q = Xq + (size_t)t * KW;
        // Eight elements at a time, in asm: gcc 12 emits each element's vmul -> vcvtr -> vmov as one
        // dependent chain even when unrolled, so the VFP's latency is paid per element.
        const float* xp = x;
        for (int k = 0; k < K; k += 8, xp += 8, q += 4)
            __asm__ volatile(
                "vldmia %[x], {s16-s23}\n"
                "vmul.f32 s16, s16, %[i]\n vmul.f32 s17, s17, %[i]\n vmul.f32 s18, s18, %[i]\n vmul.f32 s19, s19, %[i]\n"
                "vmul.f32 s20, s20, %[i]\n vmul.f32 s21, s21, %[i]\n vmul.f32 s22, s22, %[i]\n vmul.f32 s23, s23, %[i]\n"
                "vcvtr.s32.f32 s16, s16\n vcvtr.s32.f32 s17, s17\n vcvtr.s32.f32 s18, s18\n vcvtr.s32.f32 s19, s19\n"
                "vcvtr.s32.f32 s20, s20\n vcvtr.s32.f32 s21, s21\n vcvtr.s32.f32 s22, s22\n vcvtr.s32.f32 s23, s23\n"
                "vmov r0, r1, s16, s17\n vmov r2, r3, s18, s19\n"
                "pkhbt r0, r0, r1, lsl #16\n pkhbt r2, r2, r3, lsl #16\n"
                "vmov r1, r3, s20, s21\n stmia %[q], {r0, r2}\n"
                "vmov r0, r2, s22, s23\n"
                "pkhbt r1, r1, r3, lsl #16\n pkhbt r0, r0, r2, lsl #16\n"
                "str r1, [%[q], #8]\n str r0, [%[q], #12]\n"
                : : [x] "r"(xp), [q] "r"(q), [i] "t"(inv)
                : "r0", "r1", "r2", "r3", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "memory");
    }
}
// Y = epilogue(Xq . Wq^T). CL picks the output layout: [N][T] (channels-first) or [T][N].
template <bool CL>
static void gemm(const uint32_t* Xq, int T, const uint32_t* Wq, const float* sx, const float* sw, const float* b, float* Y) {
    int32_t c[8];
    auto put = [&](int t, int n, float v) { Y[CL ? (size_t)t * N + n : (size_t)n * T + t] = v; };
    for (int t0 = 0; t0 < T; t0 += MC) {
        const int tend = std::min(T, t0 + MC);
        for (int n = 0; n < N; n += 4) {
            const uint32_t* wp = Wq + (size_t)n * KW;
            int t = t0;
            for (; t + 2 <= tend; t += 2) {
                tile2x4_k256(Xq + (size_t)t * KW, wp, c);
                const float s0 = sx[t], s1 = sx[t + 1];
                for (int j = 0; j < 4; ++j) {
                    put(t, n + j, (float)c[j] * s0 * sw[n + j] + b[n + j]);
                    put(t + 1, n + j, (float)c[4 + j] * s1 * sw[n + j] + b[n + j]);
                }
            }
            for (; t < tend; ++t)
                for (int j = 0; j < 4; ++j) {
                    int32_t a = 0;
                    for (int k = 0; k < KW; ++k) a = __smlad(Xq[(size_t)t * KW + k], wp[(size_t)j * KW + k], a);
                    put(t, n + j, (float)a * sx[t] * sw[n + j] + b[n + j]);
                }
        }
    }
}
}  // namespace cit
#endif

static int run_citrinet(ggml_backend_t be, pthreadpool_t pool, int reps, int nth) {
    const int64_t K = 256, N = 256;
    struct S { int64_t T; int calls; };
    const S shapes[] = {{1101, 7}, {551, 36}, {276, 42}, {138, 46}};   // jfk's CONV_2D calls by T
    const std::vector<std::string> names = {"ggml q8_0", "ggml f32", "ggml f32rf", "ggml q8mm", "xnn qd8",
                                            "xnn qs8", "loom i8cf", "loom i8cl", "loom qcf", "loom qcl"};
    const size_t na = names.size();
    std::mt19937 rng(7);
    std::normal_distribution<float> nx(0, 1), nw(0, 0.06f);
    std::vector<float> W(N * K), B(N);   // W row-major [N][K], which is also ggml's [K, N]
    for (float& v : W) v = nw(rng);
    for (float& v : B) v = nw(rng);
    std::vector<int8_t> Wq8; std::vector<float> Ws;
    quant_per_channel(W, N, K, Wq8, Ws);
#ifdef BENCH_ARMV6
    std::vector<uint32_t> Wq16(N * K / 2);   // int8 values held as int16 pairs, [N][K/2]
    for (int64_t n = 0; n < N; ++n) for (int64_t k = 0; k < K; k += 2)
        Wq16[(n * K + k) / 2] = (uint16_t)(int16_t)Wq8[n * K + k] | ((uint32_t)(uint16_t)(int16_t)Wq8[n * K + k + 1] << 16);
#endif
    if (nth != 1) std::printf("note: ggml f32rf sets FPSCR on the calling thread only; read it at threads=1\n");

    std::vector<std::vector<double>> sum_ms(1, std::vector<double>(na, 0));
    double flop_w = 0;
    std::printf("\n%-14s", "T x calls");
    for (auto& n : names) std::printf(" %11s", (n + " ms").c_str());
    std::printf("\n");
    std::vector<std::vector<double>> errs;
    for (const S& s : shapes) {
        const int64_t T = s.T;
        std::vector<float> Xcf(K * T), Xcl(T * K), R(T * N);   // R: reference, [T][N]
        for (float& v : Xcf) v = nx(rng);
        for (int64_t k = 0; k < K; ++k) for (int64_t t = 0; t < T; ++t) Xcl[t * K + k] = Xcf[k * T + t];
        {
            xnn_operator_t op = nullptr;
            check(xnn_create_fully_connected_nc_f32(K, N, K, N, W.data(), B.data(), -INFINITY, INFINITY, 0, nullptr, &op), "create fc ref");
            check(xnn_reshape_fully_connected_nc_f32(op, T, pool), "reshape fc ref");
            check(xnn_setup_fully_connected_nc_f32(op, Xcl.data(), R.data()), "setup fc ref");
            check(xnn_run_operator(op, pool), "run fc ref");
            xnn_delete_operator(op);
        }
        auto build = [&](GgmlRun& g, auto body) {
            ggml_init_params ip = {64u * 1024 * 1024, nullptr, true};
            g.ctx = ggml_init(ip);
            g.gf = ggml_new_graph_custom(g.ctx, 64, false);
            g.out = body(g.ctx);
            ggml_build_forward_expand(g.gf, g.out);
            g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { std::fprintf(stderr, "ggml alloc failed\n"); std::exit(1); }
        };
        auto set_q8 = [&](ggml_tensor* t) {
            std::vector<char> q(ggml_row_size(GGML_TYPE_Q8_0, K) * N);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, W.data(), q.data(), 0, N, K, nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
        };
        // Channels-first graphs: X [T, 1, K, 1], Y [T, 1, N, 1], bias broadcast over T.
        GgmlRun gq, gf, gm;
        ggml_tensor *qk, *qx, *qb, *fk, *fx, *fb, *mk, *mx, *mb;
        build(gq, [&](ggml_context* c) {
            qx = ggml_new_tensor_4d(c, GGML_TYPE_F32, T, 1, K, 1); ggml_set_input(qx); ggml_set_output(qx);
            qk = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, K, N); ggml_set_input(qk); ggml_set_output(qk);
            qb = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, N, 1); ggml_set_input(qb); ggml_set_output(qb);
            return ggml_add(c, ggml_conv_2d_direct_packed(c, qk, qx, 1, 1, 0, 0, 1, 1, 1, 1, (int)K), qb);
        });
        build(gf, [&](ggml_context* c) {
            fx = ggml_new_tensor_4d(c, GGML_TYPE_F32, T, 1, K, 1); ggml_set_input(fx); ggml_set_output(fx);
            fk = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, K, N); ggml_set_input(fk); ggml_set_output(fk);
            fb = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, N, 1); ggml_set_input(fb); ggml_set_output(fb);
            return ggml_add(c, ggml_conv_2d_direct(c, fk, fx, 1, 1, 0, 0, 1, 1), fb);
        });
        build(gm, [&](ggml_context* c) {   // channels-last: X [K, T], Y [N, T]
            mx = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, T); ggml_set_input(mx); ggml_set_output(mx);
            mk = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, K, N); ggml_set_input(mk); ggml_set_output(mk);
            mb = ggml_new_tensor_1d(c, GGML_TYPE_F32, N); ggml_set_input(mb); ggml_set_output(mb);
            return ggml_add(c, ggml_mul_mat(c, mk, mx), mb);
        });
        set_q8(qk); set_q8(mk);
        ggml_backend_tensor_set(fk, W.data(), 0, W.size() * 4);   // [1,1,K,N] is W's [N][K] order
        ggml_backend_tensor_set(qx, Xcf.data(), 0, Xcf.size() * 4);
        ggml_backend_tensor_set(fx, Xcf.data(), 0, Xcf.size() * 4);
        ggml_backend_tensor_set(mx, Xcl.data(), 0, Xcl.size() * 4);
        for (ggml_tensor* t : {qb, fb, mb}) ggml_backend_tensor_set(t, B.data(), 0, B.size() * 4);
        auto get_cf = [&](GgmlRun& g) {
            std::vector<float> yg(T * N), y(T * N);
            ggml_backend_tensor_get(g.out, yg.data(), 0, yg.size() * 4);
            for (int64_t n = 0; n < N; ++n) for (int64_t t = 0; t < T; ++t) y[t * N + n] = yg[n * T + t];
            return y;
        };
        std::vector<Arm> arms;
        arms.push_back({names[0], [&] { ggml_backend_graph_compute(be, gq.gf); }, [&] { return get_cf(gq); }});
        arms.push_back({names[1], [&] { ggml_backend_graph_compute(be, gf.gf); }, [&] { return get_cf(gf); }});
#ifdef BENCH_ARMV6
        arms.push_back({names[2], [&] {
            const unsigned f = fpscr_get();
            fpscr_set(f | (1u << 24) | (1u << 25));
            ggml_backend_graph_compute(be, gf.gf);
            fpscr_set(f); }, [&] { return get_cf(gf); }});
#else
        arms.push_back({names[2], {}, {}});
#endif
        arms.push_back({names[3], [&] { ggml_backend_graph_compute(be, gm.gf); }, [&] {
            std::vector<float> y(T * N); ggml_backend_tensor_get(gm.out, y.data(), 0, y.size() * 4); return y; }});
        // XNNPACK, channels-last.
        std::vector<xnn_operator_t> ops;
        std::vector<int8_t> Aq(T * K + 64), Aq_s, Cq(T * N);
        std::vector<float> C(T * N), rowsum(T + XNN_EXTRA_QUANTIZATION_PARAMS);
        std::vector<xnn_quantization_params> qp(T + XNN_EXTRA_QUANTIZATION_PARAMS);
        std::unique_ptr<Workspace> work;
        {
            xnn_operator_t cv = nullptr, fc = nullptr;
            check(xnn_create_convert_nc_f32_qd8(0, &cv), "create convert");
            check(xnn_reshape_convert_nc_f32_qd8(cv, T, K, K, K, pool), "reshape convert");
            check(xnn_setup_convert_nc_f32_qd8(cv, Xcl.data(), Aq.data(), rowsum.data(), qp.data()), "setup convert");
            check(xnn_create_fully_connected_nc_qd8_f32_qc8w(K, N, K, N, Ws.data(), Wq8.data(), B.data(), -INFINITY, INFINITY, 0, nullptr, &fc), "create fc qd8");
            size_t ws = 0;
            check(xnn_reshape_fully_connected_nc_qd8_f32_qc8w(fc, T, &ws, pool), "reshape fc qd8");
            work = std::make_unique<Workspace>(ws);
            check(xnn_setup_fully_connected_nc_qd8_f32_qc8w(fc, Aq.data(), C.data(), work->p, qp.data()), "setup fc qd8");
            ops.push_back(cv); ops.push_back(fc);
            arms.push_back({names[4], [&, cv, fc] { check(xnn_run_operator(cv, pool), "run convert"); check(xnn_run_operator(fc, pool), "run fc qd8"); },
                            [&] { return C; }});
        }
        const float sx = amax_of(Xcl) / 127.0f, sy = amax_of(R) / 127.0f;
        quant_tensor(Xcl, sx, Aq_s);
        std::vector<int32_t> bias32(N);
        for (int64_t n = 0; n < N; ++n) bias32[n] = (int32_t)std::lround(B[n] / (sx * Ws[n]));
        {
            xnn_operator_t fs = nullptr;
            check(xnn_create_fully_connected_nc_qs8_qc8w(K, N, K, N, 0, sx, Ws.data(), Wq8.data(), bias32.data(), 0, sy, -128, 127, 0, nullptr, &fs), "create fc qs8");
            check(xnn_reshape_fully_connected_nc_qs8_qc8w(fs, T, pool), "reshape fc qs8");
            check(xnn_setup_fully_connected_nc_qs8_qc8w(fs, Aq_s.data(), Cq.data()), "setup fc qs8");
            ops.push_back(fs);
            arms.push_back({names[5], [&, fs] { check(xnn_run_operator(fs, pool), "run fc qs8"); },
                            [&] { std::vector<float> y(Cq.size()); for (size_t i = 0; i < y.size(); ++i) y[i] = sy * Cq[i]; return y; }});
        }
#ifdef BENCH_ARMV6
        std::vector<uint32_t> Xq(T * K / 2);
        std::vector<float> sxv(T), Ycf(N * T), Ycl(T * N);
        arms.push_back({names[6], [&] { cit::quantize_cf(Xcf.data(), (int)T, Xq.data(), sxv.data());
                                        cit::gemm<false>(Xq.data(), (int)T, Wq16.data(), sxv.data(), Ws.data(), B.data(), Ycf.data()); },
                        [&] { std::vector<float> y(T * N);
                              for (int64_t n = 0; n < N; ++n) for (int64_t t = 0; t < T; ++t) y[t * N + n] = Ycf[n * T + t];
                              return y; }});
        arms.push_back({names[7], [&] { cit::quantize_cl(Xcl.data(), (int)T, Xq.data(), sxv.data());
                                        cit::gemm<true>(Xq.data(), (int)T, Wq16.data(), sxv.data(), Ws.data(), B.data(), Ycl.data()); },
                        [&] { return Ycl; }});
        {   // both asm quantizes against a plain scalar one, word for word, before anything is timed
            std::vector<uint32_t> ref(T * K / 2), got(T * K / 2);
            std::vector<float> sr(T), sg(T);
            for (int64_t t = 0; t < T; ++t) {
                float m = 0;
                for (int64_t k = 0; k < K; ++k) m = std::max(m, std::fabs(Xcl[t * K + k]));
                sr[t] = m > 0 ? m / 127.0f : 1.0f;
                const float inv = 1.0f / sr[t];
                for (int64_t k = 0; k < K; k += 2)
                    ref[(t * K + k) / 2] = (uint16_t)vcvtr_s32(Xcl[t * K + k] * inv) | ((uint32_t)vcvtr_s32(Xcl[t * K + k + 1] * inv) << 16);
            }
            cit::quantize_cf(Xcf.data(), (int)T, got.data(), sg.data());
            const long dcf = (long)(std::mismatch(ref.begin(), ref.end(), got.begin()).first != ref.end()) + (sg != sr);
            cit::quantize_cl(Xcl.data(), (int)T, got.data(), sg.data());
            const long dcl = (long)(std::mismatch(ref.begin(), ref.end(), got.begin()).first != ref.end()) + (sg != sr);
            if (dcf || dcl) { std::fprintf(stderr, "T=%lld: quantize differs from the scalar one (cf %ld, cl %ld)\n", (long long)T, dcf, dcl); std::exit(1); }
        }
        arms.push_back({names[8], [&] { cit::quantize_cf(Xcf.data(), (int)T, Xq.data(), sxv.data()); }, {}});
        arms.push_back({names[9], [&] { cit::quantize_cl(Xcl.data(), (int)T, Xq.data(), sxv.data()); }, {}});
#else
        for (size_t i = 6; i < na; ++i) arms.push_back({names[i], {}, {}});
#endif
        // Alternate the arms repetition by repetition and take the median, as run_lowp does.
        for (int w = 0; w < 2; ++w) for (Arm& a : arms) if (a.run) a.run();
        std::vector<std::vector<double>> t(na);
        for (int r = 0; r < reps; ++r)
            for (size_t i = 0; i < na; ++i) {
                if (!arms[i].run) continue;
                const double t0 = now(); arms[i].run(); t[i].push_back(now() - t0);
            }
#ifdef BENCH_ARMV6
        {   // the two layouts run the same arithmetic, so their outputs must agree bit for bit
            const std::vector<float> a = arms[6].out(), b = arms[7].out();
            if (a != b) { std::fprintf(stderr, "T=%lld: loom i8cf and i8cl differ\n", (long long)T); std::exit(1); }
        }
#endif
        std::vector<double> e(na, NAN);
        std::printf("%5lld x %-6d", (long long)T, s.calls);
        for (size_t i = 0; i < na; ++i) {
            const double m = t[i].empty() ? NAN : median(t[i]);
            sum_ms[0][i] += s.calls * m;
            if (std::isnan(m)) std::printf(" %11s", "n/a"); else std::printf(" %11.3f", m * 1e3);
            if (arms[i].out) e[i] = max_rel(arms[i].out(), R);
        }
        std::printf("\n");
        errs.push_back(e);
        flop_w += s.calls * 2.0 * K * N * T;
        for (xnn_operator_t op : ops) xnn_delete_operator(op);
    }
    std::printf("\n%-14s %10s %8s  %s\n", "arm", "jfk s", "GOP/s", "max |y - ref| / max |ref| per T");
    for (size_t i = 0; i < na; ++i) {
        const double sec = sum_ms[0][i];
        std::printf("%-14s", names[i].c_str());
        if (std::isnan(sec)) std::printf(" %10s %8s ", "n/a", "n/a");
        else std::printf(" %10.3f %8.3f ", sec, flop_w / sec / 1e9);
        for (auto& e : errs) if (std::isnan(e[i])) std::printf(" %8s", "-"); else std::printf(" %8.1e", e[i]);
        std::printf("\n");
    }
    for (int cl = 0; cl < 2; ++cl) {   // the gate's own number: the GEMM without its quantize
        const double g = sum_ms[0][6 + cl] - sum_ms[0][8 + cl];
        if (!std::isnan(g)) std::printf("loom i8%s GEMM only (arm - quantize): %.3f s, %.3f GOP/s\n", cl ? "cl" : "cf", g, flop_w / g / 1e9);
    }
    return 0;
}

int main(int argc, char** argv) {
    const int nth = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 7;
    const int64_t heads = argc > 3 ? std::atoll(argv[3]) : 12;
    const std::string mode = argc > 4 ? argv[4] : "f32";
    if (mode != "f32" && mode != "f16" && mode != "q8" && mode != "q4" && mode != "citrinet") {
        std::fprintf(stderr, "mode must be f32, f16, q8, q4 or citrinet\n"); return 1;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    check(xnn_initialize(nullptr), "xnn_initialize");
    pthreadpool_t pool = pthreadpool_create(nth);
    std::printf("threads=%d reps=%d heads=%lld mode=%s  ggml llamafile=%d\n", nth, reps, (long long)heads,
                mode.c_str(), ggml_cpu_has_llamafile());
    if (mode != "f32") {
        const int rc = mode == "citrinet" ? run_citrinet(be, pool, reps, nth) : run_lowp(mode, be, pool, reps);
        pthreadpool_destroy(pool);
        ggml_backend_free(be);
        return rc;
    }

    auto alloc_graph = [&](GgmlRun& g, auto build) {
        ggml_init_params ip = {256u * 1024 * 1024, nullptr, true};
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, 64, false);
        g.out = build(g.ctx);
        ggml_build_forward_expand(g.gf, g.out);
        g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { std::fprintf(stderr, "ggml alloc failed\n"); std::exit(1); }
    };
    // Alternating arms; returns {ggml median s, xnn median s}.
    auto race = [&](const std::function<void()>& g, const std::function<void()>& x) {
        g(); x(); g(); x();
        std::vector<double> tg, tx;
        for (int r = 0; r < reps; ++r) {
            double t0 = now(); g(); tg.push_back(now() - t0);
            t0 = now(); x(); tx.push_back(now() - t0);
        }
        return std::make_pair(median(tg), median(tx));
    };

    std::vector<Row> rows;

    // ---------------------------------------------------------------- GEMM: VITS conv GEMMs + whisper linears
    struct G { const char* tag; int64_t K, M, N; double w; };
    const G gemms[] = {
        {"vits", 224, 73472, 32, 1}, {"vits", 160, 73472, 32, 1}, {"vits", 96, 73472, 32, 1},
        {"vits", 448, 18368, 64, 1}, {"vits", 320, 18368, 64, 1}, {"vits", 192, 18368, 64, 1},
        {"vits", 896, 2296, 128, 1}, {"vits", 640, 2296, 128, 1}, {"vits", 384, 2296, 128, 1},
        {"vits", 960, 288, 384, 1}, {"vits", 1344, 288, 256, 1},
        {"whisper qkv", 768, 1500, 768, 36}, {"whisper fc1", 768, 1500, 3072, 12},
        {"whisper fc2", 3072, 1500, 768, 12},
    };
    for (const G& s : gemms) {
        std::vector<float> A(s.M * s.K), W(s.N * s.K), Cx(s.M * s.N), Cg(s.M * s.N);
        fill(A, 97, 0.5f); fill(W, 53, 0.3f);
        GgmlRun g;
        ggml_tensor *ta = nullptr, *tb = nullptr;
        alloc_graph(g, [&](ggml_context* c) {
            ta = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.K, s.N); ggml_set_input(ta); ggml_set_output(ta);   // weights [K,N]
            tb = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.K, s.M); ggml_set_input(tb); ggml_set_output(tb);   // input   [K,M]
            return ggml_mul_mat(c, ta, tb);                                             // [N,M]
        });
        ggml_backend_tensor_set(ta, W.data(), 0, W.size() * 4);
        ggml_backend_tensor_set(tb, A.data(), 0, A.size() * 4);
        xnn_operator_t op = nullptr;
        check(xnn_create_fully_connected_nc_f32(s.K, s.N, s.K, s.N, W.data(), nullptr, -INFINITY, INFINITY,
                                                0, nullptr, &op), "create fc");
        check(xnn_reshape_fully_connected_nc_f32(op, s.M, pool), "reshape fc");
        check(xnn_setup_fully_connected_nc_f32(op, A.data(), Cx.data()), "setup fc");
        auto [tg, tx] = race([&] { ggml_backend_graph_compute(be, g.gf); },
                             [&] { check(xnn_run_operator(op, pool), "run fc"); });
        ggml_backend_tensor_get(g.out, Cg.data(), 0, Cg.size() * 4);
        char nm[96]; std::snprintf(nm, sizeof nm, "gemm %s %lldx%lldx%lld", s.tag, (long long)s.K,
                                   (long long)s.M, (long long)s.N);
        rows.push_back({nm, 2.0 * s.K * s.M * s.N, s.w, tg, tx, 0, max_rel(Cx, Cg)});
        xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- whisper QK^T (12 heads)
    {
        const int64_t H = heads, T = 1500, D = 64;
        std::vector<float> Q(H * T * D), Kt(H * T * D), Sx(H * T * T), Sg(H * T * T);
        fill(Q, 89, 0.4f); fill(Kt, 61, 0.4f);
        GgmlRun g;
        ggml_tensor *tk = nullptr, *tq = nullptr;
        alloc_graph(g, [&](ggml_context* c) {
            tk = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, T, H); ggml_set_input(tk); ggml_set_output(tk);
            tq = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, T, H); ggml_set_input(tq); ggml_set_output(tq);
            return ggml_mul_mat(c, tk, tq);                                             // [T(k), T(q), H]
        });
        ggml_backend_tensor_set(tk, Kt.data(), 0, Kt.size() * 4);
        ggml_backend_tensor_set(tq, Q.data(), 0, Q.size() * 4);
        xnn_operator_t op = nullptr;
        check(xnn_create_batch_matrix_multiply_nc_f32(XNN_FLAG_TRANSPOSE_B, &op), "create bmm");
        size_t ws = 0; const size_t bd[1] = {(size_t)H};
        check(xnn_reshape_batch_matrix_multiply_nc_f32(op, 1, bd, bd, T, D, T, &ws, pool), "reshape bmm");
        Workspace work(ws);
        check(xnn_setup_batch_matrix_multiply_nc_f32(op, work.p, Q.data(), Kt.data(), Sx.data()), "setup bmm");
        auto [tg, tx] = race([&] { ggml_backend_graph_compute(be, g.gf); },
                             [&] { check(xnn_run_operator(op, pool), "run bmm"); });
        ggml_backend_tensor_get(g.out, Sg.data(), 0, Sg.size() * 4);
        char nm[96]; std::snprintf(nm, sizeof nm, "whisper QK^T %lldx1500x64x1500", (long long)H);
        rows.push_back({nm, 2.0 * H * T * T * D, 12.0 * 12 / H, tg, tx, 0, max_rel(Sx, Sg)});
        xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- whisper softmax
    {
        const int64_t C = 1500, R = 1500 * heads;
        std::vector<float> X(C * R), Yx(C * R), Yg(C * R);
        fill(X, 113, 8.0f);
        GgmlRun g;
        ggml_tensor* tx_ = nullptr;
        alloc_graph(g, [&](ggml_context* c) {
            tx_ = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, R); ggml_set_input(tx_); ggml_set_output(tx_);
            return ggml_soft_max(c, tx_);
        });
        ggml_backend_tensor_set(tx_, X.data(), 0, X.size() * 4);
        xnn_operator_t op = nullptr;
        check(xnn_create_softmax_nc_f32(0, &op), "create softmax");
        check(xnn_reshape_softmax_nc_f32(op, C, C, C, R, pool), "reshape softmax");
        check(xnn_setup_softmax_nc_f32(op, X.data(), Yx.data()), "setup softmax");
        auto [tg, tx] = race([&] { ggml_backend_graph_compute(be, g.gf); },
                             [&] { check(xnn_run_operator(op, pool), "run softmax"); });
        ggml_backend_tensor_get(g.out, Yg.data(), 0, Yg.size() * 4);
        char nm[96]; std::snprintf(nm, sizeof nm, "whisper softmax 1500 x %lld", (long long)R);
        rows.push_back({nm, 5.0 * C * R, 12.0 * 12 / heads, tg, tx, 0, max_rel(Yx, Yg)});
        xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- VITS conv1d (dilated resblocks)
    struct Cv { int64_t IC, OC, kw, L; int calls; int64_t dil; };
    const Cv convs[] = {
        { 32,  32, 7, 73472, 1, 3}, { 32,  32, 7, 73472, 1,12}, { 32,  32, 5, 73472, 1, 2},
        { 32,  32, 5, 73472, 1, 6}, { 32,  32, 3, 73472, 1, 1}, { 32,  32, 3, 73472, 1, 2},
        { 64,  64, 7, 18368, 1, 3}, { 64,  64, 7, 18368, 1,12}, { 64,  64, 5, 18368, 1, 2},
        { 64,  64, 5, 18368, 1, 6}, { 64,  64, 3, 18368, 1, 1}, { 64,  64, 3, 18368, 1, 2},
        {128, 128, 7,  2296, 1, 3}, {128, 128, 7,  2296, 1,12}, {128, 128, 5,  2296, 1, 2},
        {128, 128, 5,  2296, 1, 6}, {128, 128, 3,  2296, 1, 1}, {128, 128, 3,  2296, 1, 2},
        {192, 384, 5,   287,16, 1}, {768, 768, 3,   100,12, 1},
    };
    for (const Cv& c : convs) {
        const int64_t pad = c.dil * (c.kw - 1) / 2;
        // ggml layout: x [L, IC] (L fastest), kernel [kw, IC, OC]. XNNPACK: x [L, IC] channels-last,
        // kernel [OC, kw, IC]. Same VALUES, laid out for each engine.
        std::vector<float> xg(c.L * c.IC), kg(c.kw * c.IC * c.OC);
        fill(xg, 101, 0.6f); fill(kg, 59, 0.2f);
        std::vector<float> xx(c.L * c.IC), kx(c.OC * c.kw * c.IC);
        for (int64_t i = 0; i < c.IC; ++i) for (int64_t t = 0; t < c.L; ++t) xx[t * c.IC + i] = xg[i * c.L + t];
        for (int64_t o = 0; o < c.OC; ++o) for (int64_t i = 0; i < c.IC; ++i) for (int64_t k = 0; k < c.kw; ++k)
            kx[(o * c.kw + k) * c.IC + i] = kg[(o * c.IC + i) * c.kw + k];
        std::vector<float> yx(c.L * c.OC), yg(c.L * c.OC), ygt(c.L * c.OC);

        GgmlRun g;
        ggml_tensor *tk = nullptr, *tx = nullptr;
        alloc_graph(g, [&](ggml_context* ctx) {
            tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kw, 1, c.IC, c.OC); ggml_set_input(tk); ggml_set_output(tk);
            tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.L, 1, c.IC, 1);     ggml_set_input(tx); ggml_set_output(tx);
            return ggml_conv_2d_direct(ctx, tk, tx, 1, 1, (int)pad, 0, (int)c.dil, 1);
        });
        ggml_backend_tensor_set(tk, kg.data(), 0, kg.size() * 4);
        ggml_backend_tensor_set(tx, xg.data(), 0, xg.size() * 4);

        xnn_operator_t op = nullptr;
        check(xnn_create_convolution2d_nhwc_f32(0, (uint32_t)pad, 0, (uint32_t)pad, 1, (uint32_t)c.kw, 1, 1,
                                                1, (uint32_t)c.dil, 1, c.IC, c.OC, c.IC, c.OC, kx.data(),
                                                nullptr, -INFINITY, INFINITY, 0, nullptr, &op), "create conv");
        size_t ws = 0, oh = 0, ow = 0;
        check(xnn_reshape_convolution2d_nhwc_f32(op, 1, 1, c.L, &ws, &oh, &ow, pool), "reshape conv");
        Workspace work(ws);
        check(xnn_setup_convolution2d_nhwc_f32(op, work.p, xx.data(), yx.data()), "setup conv");

        auto [tg, txn] = race([&] { ggml_backend_graph_compute(be, g.gf); },
                              [&] { check(xnn_run_operator(op, pool), "run conv"); });
        // What an integration pays on top: channel-major -> channels-last in, and back out, per conv.
        // Timed as a NAIVE single-threaded transpose, so it is an UPPER bound on that tax: a tuned,
        // threaded transpose is cheaper, and a graph kept channels-last end to end pays none of it.
        std::vector<double> tl;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int64_t i = 0; i < c.IC; ++i) for (int64_t t = 0; t < c.L; ++t) xx[t * c.IC + i] = xg[i * c.L + t];
            for (int64_t t = 0; t < c.L; ++t) for (int64_t o = 0; o < c.OC; ++o) ygt[o * c.L + t] = yx[t * c.OC + o];
            tl.push_back(now() - t0);
        }
        ggml_backend_tensor_get(g.out, yg.data(), 0, yg.size() * 4);
        for (int64_t t = 0; t < c.L; ++t) for (int64_t o = 0; o < c.OC; ++o) ygt[o * c.L + t] = yx[t * c.OC + o];
        char nm[96]; std::snprintf(nm, sizeof nm, "conv %lldx%lld k%lld d%lld L%lld", (long long)c.IC,
                                   (long long)c.OC, (long long)c.kw, (long long)c.dil, (long long)c.L);
        rows.push_back({nm, 2.0 * c.IC * c.OC * c.kw * c.L, (double)c.calls, tg, txn, median(tl), max_rel(ygt, yg)});
        xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- report
    std::printf("\n%-34s %9s %9s %8s %8s %8s %9s %9s\n", "shape", "ggml ms", "xnn ms", "ggml GF", "xnn GF",
                "ggml/xnn", "layout ms", "max rel");
    double sum_g[3] = {0, 0, 0}, sum_x[3] = {0, 0, 0}, sum_l = 0;
    for (const Row& r : rows) {
        std::printf("%-34s %9.3f %9.3f %8.1f %8.1f %8.3f %9.3f %9.1e\n", r.name.c_str(), r.g_ms * 1e3,
                    r.x_ms * 1e3, r.flop / r.g_ms / 1e9, r.flop / r.x_ms / 1e9, r.g_ms / r.x_ms,
                    r.layout_ms * 1e3, r.rel);
        const int grp = r.name.rfind("conv", 0) == 0 ? 2 : r.name.find("whisper") != std::string::npos ? 1 : 0;
        sum_g[grp] += r.weight * r.g_ms; sum_x[grp] += r.weight * r.x_ms;
        if (grp == 2) sum_l += r.weight * r.layout_ms;
    }
    const char* grp_name[3] = {"VITS GEMMs (one synthesis)", "whisper encoder (weighted)", "VITS convs (weighted)"};
    std::printf("\n");
    for (int i = 0; i < 3; ++i)
        std::printf("%-30s ggml %8.1f ms  xnn %8.1f ms  ggml/xnn %.3f\n", grp_name[i], sum_g[i] * 1e3,
                    sum_x[i] * 1e3, sum_g[i] / sum_x[i]);
    std::printf("VITS convs incl. layout tax      xnn+layout %8.1f ms  ggml/(xnn+layout) %.3f\n",
                (sum_x[2] + sum_l) * 1e3, sum_g[2] / (sum_x[2] + sum_l));
    pthreadpool_destroy(pool);
    ggml_backend_free(be);
    return 0;
}
