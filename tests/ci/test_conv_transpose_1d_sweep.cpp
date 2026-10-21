// CONV_TRANSPOSE_1D over the shapes the ARMv6 polyphase driver (ADR-079 P4.32f f2, ggml-0038) has to
// get right, against a double-precision reference, every output.
//
// On ARMv6 a stride-s transposed convolution with K = J*s taps runs as s phases of a J-tap direct
// convolution on the ggml-0032 tiles, interleaved into dst; elsewhere, and with K % s != 0, it is
// ggml-0009's GEMM + overlap-add. The driver blocks the phase positions by TB and the input channels by
// what fits L1, pads the input by J - 1 zeros either side, pads a partial quad of output channels, and
// splits blocks across threads, so the sweep covers: J = 1..4 (s = K is J = 1), the vocoders' J = 2
// stages at three strides, an output-channel count that is not a multiple of four, inputs of 1-3
// positions (shorter than a tile), an input long enough for several blocks with a ragged last one, an
// input-channel count that splits into chunks, a K that does not divide by s (the fallback), and one
// and three threads. Registered with GGML_CPU_ARMV6_CONVT=0 too, so the fallback is pinned the same way.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool compute(ggml_cgraph* gf, int n_threads) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

void fill(ggml_tensor* t, float a, float b, int seed) {
    float* d = (float*) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = a + b * (float) (((i + seed) * 7919) % 113 - 56);
}

struct Shape {
    const char* name;
    int64_t cin, cout, l, s, k;
};

// max |got - ref| / max |ref| over every output
double run(const Shape& sh, int n_threads) {
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.k, sh.cout, sh.cin);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.l, sh.cin);
    fill(w, 0.0f, 0.002f, 1);
    fill(x, 0.01f, 0.01f, 2);
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, x, (int) sh.s, 0, 1);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    LOOM_CHECK(compute(gf, n_threads));

    const int64_t OL = (sh.l - 1) * sh.s + sh.k;
    LOOM_CHECK(y->ne[0] == OL && y->ne[1] == sh.cout);
    const float* W = (const float*) w->data;
    const float* X = (const float*) x->data;
    const float* Y = (const float*) y->data;
    std::vector<double> ref((size_t) sh.cout * OL, 0.0);
    for (int64_t ic = 0; ic < sh.cin; ++ic)
        for (int64_t oc = 0; oc < sh.cout; ++oc)
            for (int64_t i = 0; i < sh.l; ++i)
                for (int64_t k = 0; k < sh.k; ++k)
                    ref[oc * OL + i * sh.s + k] += (double) W[(ic * sh.cout + oc) * sh.k + k] * X[ic * sh.l + i];
    double mx = 0, err = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mx  = std::fmax(mx, std::fabs(ref[i]));
        err = std::fmax(err, std::fabs(Y[i] - ref[i]));
    }
    ggml_free(ctx);
    return err / mx;
}

} // namespace

int main() {
    const Shape shapes[] = {
        { "J=2 s8 k16, piper up0",       192, 96, 61, 8, 16 },
        { "J=2 s4 k8, OC % 4 = 2",        48, 22, 301, 4, 8 },
        { "J=2 s2 k4, several blocks",    32, 16, 517, 2, 4 },
        { "J=1 s4 k4",                    24, 12, 77, 4, 4 },
        { "J=3 s2 k6",                    20, 9, 131, 2, 6 },
        { "J=4 s3 k12",                   16, 8, 47, 3, 12 },
        { "L=1",                          16, 8, 1, 8, 16 },
        { "L=3",                          16, 5, 3, 4, 8 },
        { "Cin 300, chunked",            300, 8, 70, 2, 4 },
        { "k5 s2 (K % s != 0, fallback)", 16, 8, 50, 2, 5 },
    };
    for (const Shape& sh : shapes) {
        for (int nt : { 1, 3 }) {
            const double e = run(sh, nt);
            std::printf("  %-32s threads %d  rel err %.2e\n", sh.name, nt, e);
            LOOM_CHECK(e < 1e-5);
        }
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
