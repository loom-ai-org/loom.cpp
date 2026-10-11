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
//   ./bench_xnnpack <threads> [reps]
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
#include <functional>
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

int main(int argc, char** argv) {
    const int nth = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 7;

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    check(xnn_initialize(nullptr), "xnn_initialize");
    pthreadpool_t pool = pthreadpool_create(nth);
    std::printf("threads=%d reps=%d  ggml llamafile=%d\n", nth, reps, ggml_cpu_has_llamafile());

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
        const int64_t H = 12, T = 1500, D = 64;
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
        rows.push_back({"whisper QK^T 12x1500x64x1500", 2.0 * H * T * T * D, 12, tg, tx, 0, max_rel(Sx, Sg)});
        xnn_delete_operator(op);
    }

    // ---------------------------------------------------------------- whisper softmax
    {
        const int64_t C = 1500, R = 1500 * 12;
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
        rows.push_back({"whisper softmax 1500 x 18000", 5.0 * C * R, 12, tg, tx, 0, max_rel(Yx, Yg)});
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
