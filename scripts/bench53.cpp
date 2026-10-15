// ADR-079 P4.32f (ggml-0036): what the convolution epilogue costs and saves, per shape, on the board.
//
// Citrinet-256-ls's sub-block tail, one graph per arm:
//   conv   CONV_2D (1x1 or k) + bias ADD                       -- the floor: what ggml-0005/0007 already fuse
//   full   + RELU + MUL(time mask)                             -- the epilogue's job
//   dw     CONV_2D_DW + MUL(time mask)
// GGML_CPU_CONV_EPILOGUE is read once per process, so run the binary twice (=0, =1) and compare: arm
// `full` at 0 pays two unfused passes, at 1 the fused epilogue; `full - conv` is the epilogue's price.
//
// Build (ARMv6, in the loom-armv6-build image, against a ggml-only static build):
//   c++ -O2 -std=c++17 -I<ggml>/include bench53.cpp <build>/src/libggml{,-cpu,-base}.a -fopenmp -latomic -o bench53
// Run on the board, one thread: GGML_CPU_CONV_EPILOGUE=0 ./bench53 [reps]; GGML_CPU_CONV_EPILOGUE=1 ./bench53 [reps]
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_backend_t g_backend;

// median ms of `reps` computes of the graph ending in `out`
static double timed(ggml_context * ctx, ggml_tensor * out, int reps) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(g_backend, gf);   // warm
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now();
        ggml_backend_graph_compute(g_backend, gf);
        t.push_back((now() - t0) * 1e3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void fill(ggml_tensor * t, float a, float b) {
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = a + b * (float) ((i * 7919) % 113 - 56);
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    g_backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g_backend, 1);
    const char * e = getenv("GGML_CPU_CONV_EPILOGUE");
    printf("GGML_CPU_CONV_EPILOGUE=%s, reps %d, median ms\n", e ? e : "(default)", reps);

    struct Shape { int64_t k, c, t; };
    // Citrinet's 1x1 widths at its three resolutions, and a depthwise kernel at each
    const Shape shapes[] = { { 1, 256, 551 }, { 1, 256, 276 }, { 1, 256, 138 } };
    const int64_t dwk[] = { 11, 13, 15 };
    printf("%-14s %9s %9s %9s %9s\n", "shape", "conv", "full", "dw", "dw+mask");
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s) {
        const Shape & sh = shapes[s];
        ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.k, 1, sh.c, sh.c);
        ggml_tensor * x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.t, 1, sh.c, 1);
        ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, sh.c, 1);
        ggml_tensor * m  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.t, 1, 1);
        ggml_tensor * wd = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dwk[s], 1, 1, sh.c);
        fill(w, 0.001f, 0.0003f); fill(x, 0.01f, 0.002f); fill(b, 0.0f, 0.01f); fill(wd, 0.01f, 0.001f);
        for (int64_t t = 0; t < sh.t; ++t) ((float *) m->data)[t] = t < sh.t - 3 ? 1.0f : 0.0f;

        ggml_tensor * conv = ggml_conv_2d_direct(ctx, w, x, 1, 1, (int) (sh.k / 2), 0, 1, 1);
        ggml_tensor * yb   = ggml_add(ctx, ggml_reshape_3d(ctx, conv, sh.t, sh.c, 1), b);
        const double t_conv = timed(ctx, yb, reps);

        ggml_tensor * conv2 = ggml_conv_2d_direct(ctx, w, x, 1, 1, (int) (sh.k / 2), 0, 1, 1);
        ggml_tensor * full  = ggml_mul(ctx, ggml_relu(ctx, ggml_add(ctx, ggml_reshape_3d(ctx, conv2, sh.t, sh.c, 1), b)), m);
        const double t_full = timed(ctx, full, reps);

        ggml_tensor * dw  = ggml_conv_2d_dw_direct(ctx, wd, x, 1, 1, (int) (dwk[s] / 2), 0, 1, 1);
        const double t_dw = timed(ctx, dw, reps);
        ggml_tensor * dw2 = ggml_conv_2d_dw_direct(ctx, wd, x, 1, 1, (int) (dwk[s] / 2), 0, 1, 1);
        ggml_tensor * dwm = ggml_mul(ctx, ggml_reshape_3d(ctx, dw2, sh.t, sh.c, 1), m);
        const double t_dwm = timed(ctx, dwm, reps);

        char name[32];
        snprintf(name, sizeof name, "k%lld c%lld t%lld", (long long) sh.k, (long long) sh.c, (long long) sh.t);
        printf("%-14s %9.3f %9.3f %9.3f %9.3f\n", name, t_conv, t_full, t_dw, t_dwm);
        ggml_free(ctx);
    }
    ggml_backend_free(g_backend);
    return 0;
}
