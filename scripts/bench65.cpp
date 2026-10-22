// P4.37: an int8 mul_mat for the Cortex-A72 (CPU_NEON_I8, ggml-0047) against what a Q8_0_PC weight runs
// on that core without it, and against F32, at whisper-small's linear shapes, with every arm in ONE
// process.
//
// NOT part of the build.
//
// WHY IT EXISTS. The A72 has no DOTPROD and tinyBLAS's Q8_0 GEMM needs it, so a Q8_0 weight runs ggml's
// one-row vec_dot_q8_0_q8_0: on a Pi 4 whisper's encoder linears took as long at Q8_0_PC as at F32. This
// races the three ways the same weight can run, on the same inputs:
//
//   f32   the F32 weight, tinyBLAS's F32 GEMM (what an F32 card runs)
//   q8    its row-uniform Q8_0 copy in the plain CPU buffer (what a Q8_0_PC card runs today)
//   i8    the same Q8_0 bytes in CPU_NEON_I8
//
// Each node has its own weight and input (whisper's 12 layers each read their own; one weight re-read
// from L2 would flatter every arm). The dynamic quantization of the activations is inside the timed
// region for both int8 arms.
//
// BUILD (on the board, against a ggml built from the patched tree):
//
//   g++ -O3 -std=c++17 -mcpu=native -I <ggml-src>/include scripts/bench65.cpp -o bench65 \
//       -L <ggml-build>/src -lggml -lggml-base -lggml-cpu -lpthread -lm -Wl,-rpath,<ggml-build>/src
//
//   ./bench65 <threads> <rounds> <shape> [arm ...]
//       shape: qkv | fc1 | fc2 | dqkv | dfc1 | dfc2 (the d* are the decoder's one-row steps) | KxNxM
//       arm:   f32 | q8 | i8, default all three; the first is the baseline every ratio is against
//
// ESTIMATOR. Each round runs every arm once, the arm order rotated round to round; the ratio to the
// baseline is taken per round and medianed. ACCURACY, printed before timing: each arm's relative
// Frobenius error on node 0 against a double-precision product of the F32 weight and input.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

static ggml_backend_buffer_type_t extra_buft(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    auto get = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t * p = get ? get(dev) : nullptr; p && *p; ++p) {
        if (strcmp(ggml_backend_buft_name(*p), name) == 0) return *p;
    }
    return nullptr;
}

// Q8_0_PC as the exporter writes it: the row's absmax / 127 as every block's F16 scale.
static std::vector<uint8_t> quantize_per_channel(const float * w, int64_t K, int64_t N) {
    const int64_t nb = K / 32;
    std::vector<uint8_t> out((size_t) N * nb * 34);
    for (int64_t n = 0; n < N; ++n) {
        float amax = 0;
        for (int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w[n * K + k]));
        const ggml_fp16_t d16 = ggml_fp32_to_fp16(amax / 127.0f);
        const float d = ggml_fp16_to_fp32(d16), id = d != 0 ? 1.0f / d : 0.0f;
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t * blk = out.data() + (size_t) (n * nb + b) * 34;
            memcpy(blk, &d16, 2);
            for (int i = 0; i < 32; ++i) {
                const float q = std::nearbyint(w[n * K + b * 32 + i] * id);
                blk[2 + i] = (uint8_t) (int8_t) std::max(-127.0f, std::min(127.0f, q));
            }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s threads rounds shape [f32|q8|i8 ...]\n", argv[0]);
        return 2;
    }
    const int threads = atoi(argv[1]), rounds = atoi(argv[2]);
    const std::string shape = argv[3];
    std::vector<std::string> arms;
    for (int i = 4; i < argc; ++i) arms.push_back(argv[i]);
    if (arms.empty()) arms = { "q8", "f32", "i8" };

    int64_t K, N, M;   // weight [K, N], input [K, M], output [N, M]
    if (shape == "qkv")       { K = 768;  N = 768;  M = 1500; }
    else if (shape == "fc1")  { K = 768;  N = 3072; M = 1500; }
    else if (shape == "fc2")  { K = 3072; N = 768;  M = 1500; }
    else if (shape == "dqkv") { K = 768;  N = 768;  M = 1; }
    else if (shape == "dfc1") { K = 768;  N = 3072; M = 1; }
    else if (shape == "dfc2") { K = 3072; N = 768;  M = 1; }
    else if (sscanf(shape.c_str(), "%ldx%ldx%ld", &K, &N, &M) != 3) return 2;
    // Enough distinct weights that the set is past the 1 MB L2 even at int8, and at least four nodes.
    const int nodes = (int) std::max<int64_t>(4, std::min<int64_t>(48, (int64_t) (8e6 / (K * N)) + 1));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_backend_buffer_type_t i8_buft = extra_buft(backend, "CPU_NEON_I8");
    if (i8_buft == nullptr && std::find(arms.begin(), arms.end(), "i8") != arms.end()) {
        fprintf(stderr, "this ggml registers no CPU_NEON_I8 (not AArch64 without DOTPROD, or GGML_CPU_NEON_I8=0)\n");
        return 1;
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<std::vector<float>> W(nodes), X(nodes);
    std::vector<std::vector<uint8_t>> Q(nodes);
    for (int j = 0; j < nodes; ++j) {
        W[j].resize((size_t) K * N);
        X[j].resize((size_t) K * M);
        for (float & v : W[j]) v = gauss(rng) * 0.05f;
        for (float & v : X[j]) v = gauss(rng);
        Q[j] = quantize_per_channel(W[j].data(), K, N);
    }

    struct Arm {
        std::string name;
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t wbuf = nullptr, xbuf = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_cgraph * gf = nullptr;
        std::vector<ggml_tensor *> outs;
    };
    std::vector<Arm> A(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        Arm & a = A[i];
        a.name = arms[i];
        ggml_init_params p = { (size_t) (8 * nodes + 64) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        a.ctx = ggml_init(p);
        const ggml_type wt = a.name == "f32" ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
        std::vector<ggml_tensor *> w(nodes), x(nodes);
        for (int j = 0; j < nodes; ++j) w[j] = ggml_new_tensor_2d(a.ctx, wt, K, N);
        a.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(a.ctx, a.name == "i8" ? i8_buft : ggml_backend_cpu_buffer_type());
        for (int j = 0; j < nodes; ++j) {
            ggml_backend_tensor_set(w[j], a.name == "f32" ? (const void *) W[j].data() : (const void *) Q[j].data(), 0,
                                    ggml_nbytes(w[j]));
        }
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
    }

    auto run = [&](Arm & a) {
        const double t0 = now();
        ggml_backend_graph_compute(backend, a.gf);
        return (now() - t0) / nodes;
    };

    // Accuracy on node 0, against a double product of the F32 weight and input.
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
        ggml_backend_tensor_get(a.outs[0], y.data(), 0, ggml_nbytes(a.outs[0]));
        double e = 0, r = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            e += (y[i] - ref[i]) * (y[i] - ref[i]);
            r += ref[i] * ref[i];
        }
        printf("  %-4s rel err %.3e\n", a.name.c_str(), std::sqrt(e / r));
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
    printf("%-6s %10s %9s %8s %15s\n", "arm", "ms/node", "GOP/s", "x base", "range");
    for (size_t i = 0; i < A.size(); ++i) {
        std::vector<double> s = ratio[i];
        std::sort(s.begin(), s.end());
        const double mt = median(t[i]);
        printf("%-6s %10.3f %9.2f %8.3f %7.3f-%.3f\n", A[i].name.c_str(), mt * 1e3, op / mt / 1e9, median(ratio[i]),
               s.front(), s.back());
    }
    ggml_backend_free(backend);
    return 0;
}
