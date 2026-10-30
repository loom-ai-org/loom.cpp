// ggml-vc4 through ggml's scheduler, the way the engine uses it (P4.44, ADR-090): backends
// [VC4, CPU], weights in the VC4 buffer type, activations in CPU buffers. For each graph: that every
// MUL_MAT landed where it must (a VC4 that declined everything would otherwise pass, since the
// reference is the CPU), the result against the same graph on [CPU] alone, and the time of both.
//
// VC4 is found the way the engine finds it -- loom's backend sweep, which loads libggml-vc4.so or
// registers the linked one -- so this also gates that loading. Exits 77 without a VC4 device: any
// machine but a Pi Zero / Pi 1-3 booted without vc4-kms-v3d (read the KMS trap in ADR-090 first).
// `test_vc4_backend cpu-only` runs both arms on the CPU, which must pass with placements ignored --
// the harness and the loading checked anywhere, which is how CI runs it.
#include "cpu_backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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
    int on_vc4;  // how many of the graph's MUL_MATs the scheduler must give the QPUs
    // builds the graph from (ctx for weights, ctx for inputs); returns the output tensor
    std::function<ggml_tensor *(ggml_context *, ggml_context *, ggml_context *)> build;
};

static ggml_backend_t g_vc4, g_cpu;
static bool g_cpu_only = false;

static int run(const Case & c) {
    // Each arm builds its own contexts, buffers and graph from the same seed: a graph whose tensors a
    // freed scheduler had allocated cannot be handed to the next one (ASan: use-after-free).
    std::vector<float> res[2];
    double best[2] = {1e9, 1e9};
    std::string placement;
    int landed = 0;
    for (int arm = 0; arm < 2; arm++) {
        ggml_init_params wp = {64 * ggml_tensor_overhead(), nullptr, true};
        ggml_context * cw = ggml_init(wp), * cx = ggml_init(wp);
        ggml_init_params gp = {512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ggml_context * cg = ggml_init(gp);
        ggml_tensor * out = c.build(cw, cx, cg);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_get_default_buffer_type(g_vc4));
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (ggml_context * ctx : {cw, cx})
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                std::vector<float> v(ggml_nelements(t));
                for (auto & x : v) x = d(rng);
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
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
                if (n->op != GGML_OP_MUL_MAT) continue;
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
        ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bx);
        ggml_free(cw); ggml_free(cx); ggml_free(cg);
    }
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < res[0].size(); i++) {
        maxd = std::max(maxd, double(std::fabs(res[0][i] - res[1][i])));
        maxv = std::max(maxv, double(std::fabs(res[1][i])));
    }
    // QPU float rounding is not IEEE (no denormals, its own fmul rounding), so a tolerance, not bits.
    const bool placed = g_cpu_only || landed == c.on_vc4;
    const bool ok = maxd <= 1e-4 * std::max(1.0, maxv) && std::isfinite(maxd) && placed;
    printf("%-36s MUL_MATs on [%s]  max|d| %.3g (max|y| %.3g) %s%s | VC4+CPU %.2f ms, CPU %.2f ms (%.2fx)\n", c.name.c_str(),
           placement.c_str(), maxd, maxv, ok ? "OK" : "WRONG", placed ? "" : " (placement)", best[0] * 1e3, best[1] * 1e3, best[1] / best[0]);
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
    };
    int fails = 0;
    for (auto & c : cases) fails += run(c);
    printf("%s\n", fails ? "FAILURES" : "ALL OK");
    ggml_backend_free(g_vc4);
    ggml_backend_free(g_cpu);
    return fails ? 1 : 0;
}
