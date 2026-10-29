// ggml-vc4 through ggml's scheduler, the way the engine uses it (P4.44, ADR-088): backends
// [VC4, CPU], weights in the VC4 buffer type, activations in CPU buffers. For each graph: that every
// MUL_MAT and CONV_2D landed where it must (a VC4 that declined everything would otherwise pass, since
// the reference is the CPU), the result against the same graph on [CPU] alone, and the time of both.
// P4.46: Q4_0 / Q8_0 MUL_MATs, convolutions with the elementwise nodes the backend folds into them
// (and one whose chain it must not fold), and a 1x1 convolution over a weight in VC4_Q8. Run it again
// with GGML_VC4_FUSE=0 for the unfused times; GGML_VC4_STATS=1 says how many nodes were folded.
//
// VC4 is found the way the engine finds it -- loom's backend sweep, which loads libggml-vc4.so or
// registers the linked one -- so this also gates that loading. Exits 77 without a VC4 device: any
// machine but a Pi Zero / Pi 1-3 booted without vc4-kms-v3d (read the KMS trap in ADR-088 first).
// `test_vc4_backend cpu-only` runs both arms on the CPU, which must pass with placements ignored --
// the harness and the loading checked anywhere, which is how CI runs it.
#include "cpu_backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <random>
#include <string>
#include <vector>

static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

struct Case {
    std::string name;
    int on_vc4;  // how many of the graph's MUL_MATs and CONV_2Ds the scheduler must give the QPUs
    // builds the graph from (ctx for weights, ctx for inputs); returns the output tensor
    std::function<ggml_tensor *(ggml_context *, ggml_context *, ggml_context *)> build;
    // P4.46: weights built in a fourth context, in VC4_Q8 on the VC4 arm (per-channel Q8_0)
    std::function<ggml_tensor *(ggml_context *, ggml_context *, ggml_context *, ggml_context *)> build_q8;
};

// Random values a tensor of any of the types here can hold: quantized ones through ggml's quantizer
// (Q8_0 per channel -- one scale for every block of a row -- when its name says "pc"), so no block
// carries a scale of random bits.
static void fill(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = d(rng);
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); return; }
    const int64_t K = t->ne[0], rows = ggml_nelements(t) / K;
    std::vector<uint8_t> q(ggml_nbytes(t));
    if (strstr(ggml_get_name(t), "pc") != nullptr) {
        struct Block { ggml_fp16_t d; int8_t qs[32]; };
        Block * b = reinterpret_cast<Block *>(q.data());
        for (int64_t r = 0; r < rows; r++) {
            float amax = 0;
            for (int64_t k = 0; k < K; k++) amax = std::max(amax, std::fabs(v[r * K + k]));
            const ggml_fp16_t dh = ggml_fp32_to_fp16(amax / 127.f);
            const float dq = ggml_fp16_to_fp32(dh);
            for (int64_t j = 0; j < K / 32; j++) {
                b[r * (K / 32) + j].d = dh;
                for (int i = 0; i < 32; i++) b[r * (K / 32) + j].qs[i] = int8_t(std::lround(v[r * K + j * 32 + i] / dq));
            }
        }
    } else {
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, rows, K, nullptr);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static ggml_backend_t g_vc4, g_cpu;
static bool g_cpu_only = false;

static int run(const Case & c) {
    // Each arm builds its own contexts, buffers and graph from the same seed: a graph whose tensors a
    // freed scheduler had allocated cannot be handed to the next one (ASan: use-after-free).
    std::vector<float> res[2], ref;   // ref: f64 over dequantized weights, for a quantized MUL_MAT
    double best[2] = {1e9, 1e9};
    std::string placement;
    int landed = 0;
    for (int arm = 0; arm < 2; arm++) {
        ggml_init_params wp = {64 * ggml_tensor_overhead(), nullptr, true};
        ggml_context * cw = ggml_init(wp), * cx = ggml_init(wp), * cq = ggml_init(wp);
        ggml_init_params gp = {512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ggml_context * cg = ggml_init(gp);
        ggml_tensor * out = c.build_q8 ? c.build_q8(cw, cx, cg, cq) : c.build(cw, cx, cg);
        ggml_backend_buffer_t bw = ggml_get_first_tensor(cw) ? ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_get_default_buffer_type(g_vc4)) : nullptr;
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_t bq = nullptr;
        if (ggml_get_first_tensor(cq)) {
            ggml_backend_buffer_type_t qb = ggml_backend_get_default_buffer_type(g_vc4);
            if (arm == 0 && !g_cpu_only) {
                auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(ggml_backend_get_device(g_vc4)), "ggml_backend_dev_get_extra_bufts"));
                qb = get_extra(ggml_backend_get_device(g_vc4))[0];
            }
            bq = ggml_backend_alloc_ctx_tensors_from_buft(cq, qb);
        }
        std::mt19937 rng(42);
        for (ggml_context * ctx : {cw, cx, cq})
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) fill(t, rng);
        // ggml's CPU quantizes a MUL_MAT's activations to Q8_0 when the weight is quantized, so for
        // those the CPU is not the reference: an f64 product of the dequantized weight is, and the QPUs
        // (F32 activations) must be at least as close to it as the CPU (P4.46).
        if (arm == 0 && !g_cpu_only && out->op == GGML_OP_MUL_MAT && out->src[0]->type != GGML_TYPE_F32) {
            const ggml_tensor * w = out->src[0], * x = out->src[1];
            const int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1];
            std::vector<uint8_t> wq(ggml_nbytes(w));
            std::vector<float> wf(size_t(M) * K), xf(size_t(N) * K);
            ggml_backend_tensor_get(w, wq.data(), 0, wq.size());
            ggml_backend_tensor_get(x, xf.data(), 0, xf.size() * 4);
            ggml_get_type_traits(w->type)->to_float(wq.data(), wf.data(), M * K);
            ref.assign(size_t(M) * N, 0.f);
            for (int64_t n = 0; n < N; n++)
                for (int64_t m = 0; m < M; m++) {
                    double acc = 0;
                    for (int64_t k = 0; k < K; k++) acc += double(wf[m * K + k]) * xf[n * K + k];
                    ref[n * M + m] = float(acc);
                }
        }
        ggml_cgraph * gf = ggml_new_graph(cg);
        ggml_build_forward_expand(gf, out);
        ggml_backend_t list[2] = {g_vc4, g_cpu};
        ggml_backend_t * bl = arm == 0 ? list : list + 1;
        const int nb = arm == 0 ? 2 : 1;
        ggml_backend_sched_t sched = ggml_backend_sched_new(bl, nullptr, nb, GGML_DEFAULT_GRAPH_SIZE, false, true);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { printf("%s: alloc failed\n", c.name.c_str()); return 1; }
        if (arm == 0)
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_CONV_2D) continue;
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, n);
                placement += std::string(placement.empty() ? "" : ",") + (b ? ggml_backend_name(b) : "?");
                landed += b == g_vc4;
            }
        for (int rep = 0; rep < 4; rep++) {
            const double t0 = now();
            if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
            best[arm] = std::min(best[arm], now() - t0);
        }
        res[arm].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res[arm].data(), 0, ggml_nbytes(out));
        ggml_backend_sched_free(sched);
        if (bw) ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bx);
        if (bq) ggml_backend_buffer_free(bq);
        ggml_free(cw); ggml_free(cx); ggml_free(cq); ggml_free(cg);
    }
    double maxd = 0, maxv = 0, cpu_d = 0;
    const std::vector<float> & want = ref.empty() ? res[1] : ref;
    for (size_t i = 0; i < res[0].size(); i++) {
        maxd = std::max(maxd, double(std::fabs(res[0][i] - want[i])));
        maxv = std::max(maxv, double(std::fabs(want[i])));
        if (!ref.empty()) cpu_d = std::max(cpu_d, double(std::fabs(res[1][i] - want[i])));
    }
    // QPU float rounding is not IEEE (no denormals, its own fmul rounding), so a tolerance, not bits.
    const bool placed = g_cpu_only || landed == c.on_vc4;
    const double tol = ref.empty() ? 1e-4 : 1e-3;   // the int8 kernels' offset sums (qpu_gemm_q8.h)
    const bool ok = maxd <= tol * std::max(1.0, maxv) && std::isfinite(maxd) && placed && (ref.empty() || maxd <= cpu_d);
    printf("%-36s on [%s]  max|d| %.3g (max|y| %.3g) %s%s | VC4+CPU %.2f ms, CPU %.2f ms (%.2fx)\n", c.name.c_str(),
           placement.c_str(), maxd, maxv, ok ? "OK" : "WRONG", placed ? "" : " (placement)", best[0] * 1e3, best[1] * 1e3, best[1] / best[0]);
    if (!ref.empty()) printf("  (against f64 over the dequantized weight; the CPU, which quantizes the activations: max|d| %.3g)\n", cpu_d);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    g_cpu_only = argc > 1 && strcmp(argv[1], "cpu-only") == 0;  // harness check: both arms on the CPU
    g_cpu = loom_test::cpu_backend();
    if (g_cpu == nullptr) return 1;
    // loom's sweep must have REGISTERED ggml-vc4 on any machine -- with no device off a board -- since
    // ggml's own never does. Without this a loader that lost the module would read as "no board".
    if (ggml_backend_reg_by_name("VC4") == nullptr) {
        printf("FAIL: ggml-vc4 is built but loom's backend sweep did not register it\n");
        return 1;
    }
    if (g_cpu_only) {
        g_vc4 = loom_test::cpu_backend();
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("VC4");
        if (dev == nullptr) {
            printf("SKIP: no VC4 device registered (not a Pi Zero / Pi 1-3 without KMS, or LOOM_GGML_VC4=OFF)\n");
            return 77;
        }
        printf("device: %s (%s), type %d\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               int(ggml_backend_dev_type(dev)));
        g_vc4 = ggml_backend_dev_init(dev, nullptr);
    }
    auto mm = [](int M, int N, int K) {
        return [=](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
            ggml_tensor * w = ggml_new_tensor_2d(cw, GGML_TYPE_F32, K, M);
            ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, K, N);
            return ggml_mul_mat(cg, w, x);
        };
    };
    std::vector<Case> cases = {
        {"below threshold 16x16x4 (CPU)", 0, mm(16, 16, 4)},
        {"29x37x193", 1, mm(29, 37, 193)},
        {"decode 192x1x192 (CPU: 74 kflop)", 0, mm(192, 1, 192)},
        {"VITS enc 192x101x576", 1, mm(192, 101, 576)},
        {"VITS dec 384x457x192", 1, mm(384, 457, 192)},
        {"chain mm->add->relu->mm", 2, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * w1 = ggml_new_tensor_2d(cw, GGML_TYPE_F32, 192, 384);
             ggml_tensor * b1 = ggml_new_tensor_1d(cw, GGML_TYPE_F32, 384);
             ggml_tensor * w2 = ggml_new_tensor_2d(cw, GGML_TYPE_F32, 384, 192);
             ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, 192, 101);
             ggml_tensor * h = ggml_relu(cg, ggml_add(cg, ggml_mul_mat(cg, w1, x), b1));
             return ggml_mul_mat(cg, w2, h);
         }},
        {"batched broadcast [K,M,2] x [K,N,4]", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * w = ggml_new_tensor_3d(cw, GGML_TYPE_F32, 64, 48, 2);
             ggml_tensor * x = ggml_new_tensor_3d(cx, GGML_TYPE_F32, 64, 40, 4);
             return ggml_mul_mat(cg, w, x);
         }},
        // P4.46
        {"Q4_0 176x76x704 (conformer)", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * w = ggml_new_tensor_2d(cw, GGML_TYPE_Q4_0, 704, 176);
             ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, 704, 76);
             return ggml_mul_mat(cg, w, x);
         }},
        {"Q8_0 288x64x288, a scale a block", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * w = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, 288, 288);
             ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, 288, 64);
             return ggml_mul_mat(cg, w, x);
         }},
        {"HiFi-GAN lrelu+conv7+bias+residual", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * k = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 7, 1, 32, 32);
             ggml_tensor * b = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, 32, 1);
             ggml_tensor * x = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 3000, 1, 32, 1);
             ggml_tensor * h = ggml_leaky_relu(cg, x, 0.1f, false);
             ggml_tensor * c = ggml_conv_2d_direct(cg, k, h, 1, 1, 9, 0, 3, 1);
             return ggml_add(cg, ggml_add(cg, c, b), x);
         }},
        {"Citrinet conv1+bias+relu+mask", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * k = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, 64, 80);
             ggml_tensor * b = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, 80, 1);
             ggml_tensor * x = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 551, 1, 64, 1);
             ggml_tensor * m = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 551, 1, 1, 1);
             ggml_tensor * c = ggml_conv_2d_direct(cg, k, x, 1, 1, 0, 0, 1, 1);
             return ggml_mul(cg, ggml_relu(cg, ggml_add(cg, c, b)), m);
         }},
        {"conv chain read twice (not folded)", 1, [](ggml_context * cw, ggml_context * cx, ggml_context * cg) {
             ggml_tensor * k = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 5, 1, 24, 24);
             ggml_tensor * b = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, 24, 1);
             ggml_tensor * x = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 700, 1, 24, 1);
             ggml_tensor * y = ggml_add(cg, ggml_conv_2d_direct(cg, k, x, 1, 1, 2, 0, 1, 1), b);
             return ggml_add(cg, ggml_relu(cg, ggml_add(cg, y, x)), y);
         }},
        {"1x1 Q8_0 VC4_Q8 conv+bias+relu+mask", 1, nullptr, [](ggml_context * cw, ggml_context * cx, ggml_context * cg, ggml_context * cq) {
             ggml_tensor * w = ggml_new_tensor_2d(cq, GGML_TYPE_Q8_0, 256, 256);
             ggml_set_name(w, "pointwise.pc");
             ggml_tensor * b = ggml_new_tensor_4d(cw, GGML_TYPE_F32, 1, 1, 256, 1);
             ggml_tensor * x = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 551, 1, 256, 1);
             ggml_tensor * m = ggml_new_tensor_4d(cx, GGML_TYPE_F32, 551, 1, 1, 1);
             ggml_tensor * c = ggml_conv_2d_direct_packed(cg, w, x, 1, 1, 0, 0, 1, 1, 1, 1, 256);
             return ggml_mul(cg, ggml_relu(cg, ggml_add(cg, c, b)), m);
         }},
    };
    int fails = 0;
    for (auto & c : cases) fails += run(c);
    printf("%s\n", fails ? "FAILURES" : "ALL OK");
    ggml_backend_free(g_vc4);
    ggml_backend_free(g_cpu);
    return fails ? 1 : 0;
}
