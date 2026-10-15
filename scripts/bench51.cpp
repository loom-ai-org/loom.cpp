// ADR-079 gate 2 for P4.32c: THE ONE-COLUMN F32 MUL_MAT, AS GGML RUNS IT, AGAINST bench48. Moonshine
// tiny's decode GEMVs (its 288 x 32768 vocabulary projection, a 288 x 2304 per-layer step) and an
// LSTM-sized one, each a ggml MUL_MAT with one column on one thread in RunFast (ggml-0031). bench48's
// F32 GEMV streamed 316 MB/s on the first shape, the board's PLD read ceiling 338; ggml without 0033, 91.
//
//   ./bench51 [reps]                      ms, GOP/s and the weight stream in MB/s per shape
//   ./bench51 [reps] gemm                 P4.32d's shapes instead (ggml-0034, n >= 8): Moonshine tiny's
//                                         prefill GEMMs, conformer-ctc-small's feed-forward, and n = 16-64
//                                         to place the n threshold;
//                                         GGML_CPU_ARMV6_GEMM=0 for ggml's own (tinyBLAS_F32_ARMV6)
//   GGML_CPU_ARMV6_GEMV=0 ./bench51       ggml's own path (vec_dot_f32: a double sum). The checksums
//                                         differ: ggml-0033 sums in float (tests/ci/test_mul_mat_f32_sweep
//                                         checks its own bits)
//
// The tensors live in a CPU backend buffer, as the engine's weights and activations do. NOT in a
// no_alloc=false context: on a 32-bit build that aligns tensor data to 4 bytes, and the same kernel on
// the same bytes measured 216 MB/s there (page offset 340) against 274 from a heap copy (page offset 8).
//
// Build (loom-armv6-build:bookworm), against a static ggml built from the patched tree:
//   g++ -O2 -std=c++17 -I<ggml>/include scripts/bench51.cpp <build>/src/libggml.a <build>/src/libggml-cpu.a
//       <build>/src/libggml-base.a -fopenmp -lpthread -latomic -o bench51
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const bool gemm = argc > 2 && !std::strcmp(argv[2], "gemm");
    struct Shape3 { int k, n, cols; const char* who; };
    const Shape3 vec_shapes[] = { {288, 32768, 1, "moonshine vocab"}, {288, 2304, 1, "moonshine step"},
                                  {512, 2048, 1, "lstm-sized"}, {256, 4096, 1, "row 1 KB"},
                                  {768, 2048, 1, "row 3 KB"}, {1024, 1024, 1, "row 4 KB"} };
    const Shape3 gemm_shapes[] = { {288, 1152, 457, "mt ffn up"}, {1152, 288, 457, "mt ffn down"},
                                   {288, 288, 457, "mt attn proj"}, {176, 704, 75, "conformer ffn"},
                                   {288, 1152, 16, "mt n=16"}, {288, 1152, 24, "mt n=24"}, {288, 1152, 32, "mt n=32"},
                                   {288, 1152, 64, "mt n=64"} };
    const Shape3* shapes = gemm ? gemm_shapes : vec_shapes;
    const int n_shapes = gemm ? 8 : 6;
    std::printf("reps=%d\n%-16s %9s %8s %8s  %s\n", reps, "shape", "ms", "GOP/s", "MB/s", "checksum");
    for (int si = 0; si < n_shapes; ++si) {
        const Shape3& s = shapes[si];
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.cols);
        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0, 1);
        for (int64_t i = 0; i < ggml_nelements(w); ++i) ((float*) w->data)[i] = nd(rng) * 0.05f;
        for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float*) x->data)[i] = nd(rng);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
        std::vector<uint8_t> work(plan.work_size + 1);
        plan.work_data = work.data();
        ggml_graph_compute(gf, &plan);
        uint64_t h = 1469598103934665603ull;               // FNV-1a over the output's bits
        for (int64_t i = 0; i < ggml_nbytes(y); ++i) h = (h ^ ((const uint8_t*) y->data)[i]) * 1099511628211ull;
        std::vector<double> tt;
        for (int r = 0; r < reps; ++r) { const double t0 = now(); ggml_graph_compute(gf, &plan); tt.push_back(now() - t0); }
        std::sort(tt.begin(), tt.end());
        const double m = tt[tt.size() / 2];
        std::printf("%-16s %9.2f %8.3f %8.1f  %016llx\n", s.who, m * 1e3, 2.0 * s.k * s.n * s.cols / m / 1e9,
                    (double) s.k * s.n * 4 / m / 1e6, (unsigned long long) h);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return 0;
}
