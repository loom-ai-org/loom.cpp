// ADR-079 gate 2 for P4.32b: DOES THE SHORT-VECTOR CONV TILE KEEP ITS BENCH RATE INSIDE GGML? bench49's
// seven vocoder shapes (amy, de051, VITS: k = 3-7, dilation 1-5, 12-64 channels), each run as the graph
// loom's lowering emits for a 1-D convolution: CONV_2D (no padding, as bench49) -> reshape -> per-channel
// bias ADD, which ggml-0005 fuses into one node. ggml computes it on one thread, in RunFast (ggml-0031),
// through ggml-0032's path, which packs the weights per call -- bench49 packs them once and does not time
// it, so this arm pays slightly more for the same kernel.
//
//   ./bench50 [reps] [T divisor] [set] prints ms and GOP/s per shape; compare with `BENCH_TB=64 ./bench49`
//   set `model`: the shapes bench49 lacks and the models' board profiles found slow -- the dilated half of
//   each resblock (d = 3, 5) and the wide stages (96-384 channels, a few hundred to a few thousand steps),
//   VITS's OC = 1 output conv; amy, VITS and sanoTTS heart
//   set `citrinet`: Citrinet-256-ls's 1x1 convolutions (256 -> 256 at its four time resolutions), the
//   GEMM class of ADR-079 P4.32d
//   GGML_CPU_ARMV6_CONV=0 ./bench50   the same graphs through ggml's own paths (the "ggml today" column)
// Every shape is checked against a double reference on every 7th time step before it is timed.
//
// Build (loom-armv6-build:bookworm), against a static ggml built from the patched tree:
//   g++ -O2 -std=c++17 -I<ggml>/include scripts/bench50.cpp <build>/src/libggml.a <build>/src/libggml-cpu.a
//       <build>/src/libggml-base.a -lpthread -o bench50
#include <ggml.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Shape { int ic, oc, k, d, T; const char* who; };

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int div = argc > 2 ? std::atoi(argv[2]) : 1;
    const bool model = argc > 3 && !std::strcmp(argv[3], "model");
    const bool citrinet = argc > 3 && !std::strcmp(argv[3], "citrinet");
    const Shape citrinet_shapes[] = {
        {256, 256, 1, 1, 1101, "ct 1x1 T1101"}, {256, 256, 1, 1, 551, "ct 1x1 T551"},
        {256, 256, 1, 1, 276, "ct 1x1 T276"}, {256, 256, 1, 1, 138, "ct 1x1 T138"},
    };
    const Shape model_shapes[] = {
        {48, 48, 7, 3, 32512, "amy 48ch k7 d3"}, {48, 48, 7, 5, 32512, "amy 48ch k7 d5"},
        {96, 96, 7, 1, 4064, "amy 96ch k7"}, {96, 96, 7, 5, 4064, "amy 96ch k7 d5"},
        {32, 32, 7, 5, 70400, "VITS 32ch k7 d5"}, {64, 64, 7, 5, 17600, "VITS 64ch k7 d5"},
        {128, 128, 7, 1, 2200, "VITS 128ch k7"}, {128, 128, 7, 5, 2200, "VITS 128ch k7 d5"},
        {192, 384, 5, 1, 288, "VITS 192>384 k5"}, {32, 1, 7, 1, 70400, "VITS 32>1 k7"},
        {96, 96, 5, 1, 288, "heart 96ch k5"}, {100, 192, 7, 1, 288, "heart 100>192 k7"},
    };
    // bench49's, in its order.
    const Shape bench_shapes[] = {
        {24, 24, 7, 1, 130048, "amy 24ch k7"}, {24, 24, 7, 3, 130048, "amy 24ch k7 d3"},
        {24, 24, 7, 5, 130048, "amy 24ch k7 d5"}, {48, 48, 3, 1, 32512, "amy 48ch k3"},
        {48, 48, 7, 1, 32512, "amy 48ch k7"}, {12, 12, 7, 1, 97024, "de051 12ch k7"},
        {64, 64, 7, 1, 17600, "VITS 64ch k7"},
    };
    std::printf("reps=%d T/%d\n%-16s %8s %9s %8s %9s\n", reps, div, "shape", "GFLOP", "ms", "GOP/s", "err");
    double tot = 0, totf = 0;
    const Shape* shapes = citrinet ? citrinet_shapes : model ? model_shapes : bench_shapes;
    const int n_shapes = citrinet ? 4 : model ? (int) (sizeof(model_shapes) / sizeof(Shape))
                                              : (int) (sizeof(bench_shapes) / sizeof(Shape));
    for (int si = 0; si < n_shapes; ++si) {
        const Shape& s0 = shapes[si];
        Shape s = s0;
        if (!citrinet) s.T = (s.T / div) & ~31;
        const int span = (s.k - 1) * s.d, Tin = s.T + span;
        std::mt19937 rng(11);
        std::normal_distribution<float> nx(0, 1), nw(0, 0.08f);
        ggml_init_params ip = { (size_t) s.ic * Tin * 4 + (size_t) s.oc * s.T * 8 + (size_t) s.ic * s.oc * s.k * 4 +
                                    (size_t) 8 * 1024 * 1024, nullptr, false };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, s.ic, s.oc);
        ggml_tensor* tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Tin, 1, s.ic, 1);
        ggml_tensor* tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, s.oc);
        float* X = (float*) tx->data; float* W = (float*) tk->data; float* B = (float*) tb->data;
        // bench49's draw order (X, W, B) and layouts (X [IC][Tin], W [OC][IC][k]).
        for (int i = 0; i < s.ic * Tin; ++i) X[i] = nx(rng);
        for (int i = 0; i < s.oc * s.ic * s.k; ++i) W[i] = nw(rng);
        for (int i = 0; i < s.oc; ++i) B[i] = nw(rng);
        ggml_tensor* conv = ggml_conv_2d_direct(ctx, tk, tx, 1, 1, 0, 0, s.d, 1);
        ggml_tensor* out = ggml_add(ctx, ggml_reshape_2d(ctx, conv, s.T, s.oc), tb);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        // Planned once: ggml_graph_compute_with_ctx allocates a new work buffer from the context per call.
        ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        auto run = [&] { ggml_graph_compute(gf, &plan); };
        run();
        const float* Y = (const float*) out->data;
        double ef = 0, mr = 0;
        for (int o = 0; o < s.oc; ++o)
            for (int t = 0; t < s.T; t += 7) {
                double r = B[o];
                for (int c = 0; c < s.ic; ++c) for (int j = 0; j < s.k; ++j)
                    r += (double) W[((size_t) o * s.ic + c) * s.k + j] * X[(size_t) c * Tin + t + j * s.d];
                mr = std::max(mr, std::fabs(r));
                ef = std::max(ef, std::fabs(Y[(size_t) o * s.T + t] - r));
            }
        ef /= mr;
        if (ef > 1e-5) { std::fprintf(stderr, "%s: error %.2e\n", s.who, ef); return 1; }
        std::vector<double> tt;
        for (int r = 0; r < reps; ++r) { const double t0 = now(); run(); tt.push_back(now() - t0); }
        std::sort(tt.begin(), tt.end());
        const double m = tt[tt.size() / 2], flop = 2.0 * s.oc * s.ic * s.k * (double) s.T;
        std::printf("%-16s %8.3f %9.2f %8.3f %9.1e\n", s.who, flop / 1e9, m * 1e3, flop / m / 1e9, ef);
        tot += m; totf += flop;
        ggml_free(ctx);
    }
    std::printf("%-16s %8.3f %9.2f %8.3f\n", "all", totf / 1e9, tot * 1e3, totf / tot / 1e9);
    return 0;
}
