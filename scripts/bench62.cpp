// ADR-079 P4.33 step 3: WHY CITRINET'S POINTWISE CONVOLUTIONS DO NOT GAIN. On the board Citrinet-256-ls's
// Q8_0_PC card ran its 1x1 convolutions at 0.60-0.64 GOP/s through CPU_ARMV6_I8 (ggml-0043), against
// 0.665 for the F32 card's ggml-0034 path, where bench60 had the same int8 tile at 0.93 on contiguous
// columns. This times one convolution through ggml, as the engine builds it, three ways on the same
// shapes (K = IC = 256, N = OC = 256, T = Citrinet's four lengths):
//
//   f32      ggml_conv_2d_direct on an F32 [1, 1, IC, OC] kernel (ggml-0034)
//   i8conv   ggml_conv_2d_direct_packed on the Q8_0_PC kernel in CPU_ARMV6_I8 (ggml-0043), channels-first
//   i8mm     ggml_mul_mat on the same packed kernel, the activations channels-last (ggml-0042, as bench60)
//
//   ./bench62 [reps]
//   GGML_CPU_ARMV6_I8_{TWB,TKC,TCPB} change the GEMM's blocking (read once)
//
// Build (loom-armv6-build:bookworm), against a static ggml built from the patched tree:
//   g++ -O2 -std=c++17 -Wno-psabi -I<ggml>/include scripts/bench62.cpp <build>/src/libggml.a
//       <build>/src/libggml-cpu.a <build>/src/libggml-base.a -fopenmp -lpthread -latomic -o bench62
#include <ggml.h>
#include <ggml-alloc.h>
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

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static ggml_backend_buffer_type_t i8_buft(ggml_backend_t be) {
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t* p = get_extra(dev); p && *p; ++p)
        if (!std::strcmp(ggml_backend_buft_name(*p), "CPU_ARMV6_I8")) return *p;
    return nullptr;
}

struct Block { uint16_t d; int8_t qs[32]; };

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 1);
    ggml_backend_buffer_type_t buft = i8_buft(be);
    if (!buft) { std::printf("no CPU_ARMV6_I8\n"); return 1; }
    const int64_t K = 256, N = 256;
    // Pseudo-random, so the outputs do not cancel: a smooth pattern summed to near zero and left the
    // quantization noise at 10% of a tiny peak.
    uint32_t seed = 12345;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return (float) (seed >> 8) / 16777216.0f - 0.5f; };
    std::vector<float> W((size_t) N * K);
    for (float& w : W) w = 0.1f * rnd();
    std::vector<Block> Wq((size_t) N * K / 32);
    for (int64_t n = 0; n < N; ++n) {   // per channel, as the exporter writes Q8_0_PC
        float m = 0;
        for (int64_t k = 0; k < K; ++k) m = std::fmax(m, std::fabs(W[n * K + k]));
        const float d = m / 127;
        for (int64_t b = 0; b < K / 32; ++b) {
            Block& blk = Wq[n * (K / 32) + b];
            blk.d = ggml_fp32_to_fp16(d);
            for (int i = 0; i < 32; ++i) blk.qs[i] = (int8_t) std::lrint(W[n * K + b * 32 + i] / d);
        }
    }
    std::printf("%6s | %9s %9s %9s | %7s %7s %7s\n", "T", "f32 ms", "i8conv", "i8mm", "f32 GOP", "i8conv", "i8mm");
    for (int64_t T : {1101, 551, 276, 138}) {
        ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true });
        ggml_context* cq = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true });
        ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 16 + 3 * ggml_graph_overhead(), nullptr, true });
        ggml_tensor* wf = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, K, N);
        ggml_tensor* wq = ggml_new_tensor_2d(cq, GGML_TYPE_Q8_0, K, N);
        ggml_tensor* xc = ggml_new_tensor_4d(cx, GGML_TYPE_F32, T, 1, K, 1);   // channels-first [T, 1, IC, 1]
        ggml_tensor* xl = ggml_new_tensor_2d(cx, GGML_TYPE_F32, K, T);         // channels-last [IC, T]
        ggml_tensor* yf = ggml_conv_2d_direct(cx, wf, xc, 1, 1, 0, 0, 1, 1);
        ggml_tensor* yc = ggml_conv_2d_direct_packed(cx, wq, xc, 1, 1, 0, 0, 1, 1, 1, 1, (int) K);
        ggml_tensor* ym = ggml_mul_mat(cx, wq, xl);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bq = ggml_backend_alloc_ctx_tensors_from_buft(cq, buft);
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
        // the F32 kernel [1, 1, IC, OC]: element (ic, oc) at oc * K + ic, as W is laid out
        std::memcpy(wf->data, W.data(), W.size() * 4);
        ggml_backend_tensor_set(wq, Wq.data(), 0, ggml_nbytes(wq));
        for (int64_t i = 0; i < K * T; ++i) ((float*) xc->data)[i] = rnd();
        for (int64_t t = 0; t < T; ++t)
            for (int64_t k = 0; k < K; ++k) ((float*) xl->data)[t * K + k] = ((float*) xc->data)[k * T + t];
        ggml_cgraph* gf = ggml_new_graph(cx); ggml_build_forward_expand(gf, yf);
        ggml_cgraph* gc = ggml_new_graph(cx); ggml_build_forward_expand(gc, yc);
        ggml_cgraph* gm = ggml_new_graph(cx); ggml_build_forward_expand(gm, ym);
        ggml_backend_graph_compute(be, gf); ggml_backend_graph_compute(be, gc); ggml_backend_graph_compute(be, gm);
        // the two int8 forms agree to the bit (same columns, same contract); each arm near a double reference
        double e = 0, ef = 0, ec = 0, ref = 0;
        for (int64_t o = 0; o < N; ++o)
            for (int64_t t = 0; t < T; ++t) {
                double r = 0;
                for (int64_t k = 0; k < K; ++k) r += (double) W[o * K + k] * ((const float*) xc->data)[k * T + t];
                const float c = ((float*) yc->data)[o * T + t], m = ((float*) ym->data)[t * N + o], f = ((float*) yf->data)[o * T + t];
                e = std::fmax(e, std::fabs(c - m)); ef = std::fmax(ef, std::fabs(f - r)); ec = std::fmax(ec, std::fabs(c - r));
                ref = std::fmax(ref, std::fabs(r));
            }
        if (e != 0 || !(ef / ref < 1e-4) || !(ec / ref < 3e-2)) {
            std::printf("T=%lld: i8conv vs i8mm %.3g; of peak: f32 %.3g, i8 %.3g\n", (long long) T, e, ef / ref, ec / ref);
            return 1;
        }
        std::vector<double> tf, tc, tm;
        for (int r = 0; r < reps; ++r) {
            double t0 = now(); ggml_backend_graph_compute(be, gf); tf.push_back(now() - t0);
            t0 = now(); ggml_backend_graph_compute(be, gc); tc.push_back(now() - t0);
            t0 = now(); ggml_backend_graph_compute(be, gm); tm.push_back(now() - t0);
        }
        std::sort(tf.begin(), tf.end()); std::sort(tc.begin(), tc.end()); std::sort(tm.begin(), tm.end());
        const double f = tf[reps / 2], c = tc[reps / 2], m = tm[reps / 2], op = 2.0 * K * N * T / 1e9;
        std::printf("%6lld | %9.2f %9.2f %9.2f | %7.3f %7.3f %7.3f\n", (long long) T, f * 1e3, c * 1e3, m * 1e3, op / f, op / c, op / m);
        ggml_backend_buffer_free(bw); ggml_backend_buffer_free(bq); ggml_backend_buffer_free(bx);
        ggml_free(cw); ggml_free(cq); ggml_free(cx);
    }
    ggml_backend_free(be);
    return 0;
}
