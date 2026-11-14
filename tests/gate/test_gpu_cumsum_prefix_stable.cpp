// A GPU's CUMSUM is PREFIX-STABLE: a row's first n sums are bit-identical whether the row holds n elements
// or more, and the same from one run to the next (cmake/patches/ggml-0064 for CUDA).
//
// Why it matters: a model that re-runs a cumsum over a growing input relies on the frames it already had
// coming back unchanged. CosyVoice3's live stream (ADR-112) re-runs CausalHiFT over every mel frame so far,
// and the vocoder's sine phase is a cumsum reaching ~1e6 rad, where an f32 ulp is 0.06 rad. Stock CUDA
// sized its block by the row length, built each offset from a total that includes the elements AFTER i,
// and handed long rows to `cub::DeviceScan`. On an RTX 5090, 223 of these 250 rows came back different,
// and the stream's join was 5.9e-03 from one vocoder call over the same mel; with the patch it is 0.
//
// Rows of 1 and 9 (the vocoder's harmonics) at lengths across every point where the stock launcher switched
// block size or kernel: 32, 256, 1024 and 4096 per row. The sums are also checked against the CPU's
// sequential ones, so a scan that is stable but wrong still fails. Skips (77) without a GPU.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

std::vector<float> cumsum_on(ggml_backend_t backend, const std::vector<float>& x, int n, int rows, int stride) {
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, /*no_alloc=*/true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
    ggml_tensor* y = ggml_cumsum(ctx, a);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> in(static_cast<size_t>(n) * rows);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(&in[static_cast<size_t>(r) * n], &x[static_cast<size_t>(r) * stride], sizeof(float) * n);
    }
    ggml_backend_tensor_set(a, in.data(), 0, sizeof(float) * in.size());
    LOOM_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out(in.size());
    ggml_backend_tensor_get(y, out.data(), 0, sizeof(float) * out.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

} // namespace

int main() {
    bool any_device = false;
    for (const loom::DeviceInfo& d : loom::available_devices()) any_device = any_device || !d.is_cpu;
    if (!any_device) {
        std::fprintf(stderr, "skipping: no GPU/accelerator device in this build\n");
        return kSkipReturnCode;
    }
    loom::Device gpu = loom::Device::open("gpu");
    loom::Device cpu = loom::Device::open("cpu");
    std::fprintf(stderr, "device: %s\n", gpu.name().c_str());
    ggml_backend_t gb = gpu.backends().primary;
    ggml_backend_t cb = cpu.backends().primary;

    constexpr int kMax = 40000;
    const std::vector<int> lengths = {1, 31, 32, 33, 76, 96, 127, 128, 129, 176, 255, 256, 257, 376, 446,
                                      1023, 1024, 1025, 2000, 4096, 4097, 9000, 36864, 36865, kMax};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);   // like the phase increments: in [0, 1)
    int unstable = 0, rows_checked = 0;
    double worst = 0.0;
    for (const int rows : {1, 9}) {
        std::vector<float> x(static_cast<size_t>(rows) * kMax);
        for (float& v : x) v = uniform(rng);
        const std::vector<float> full = cumsum_on(gb, x, kMax, rows, kMax);
        unstable += cumsum_on(gb, x, kMax, rows, kMax) != full;   // run to run
        for (const int n : lengths) {
            const std::vector<float> part = cumsum_on(gb, x, n, rows, kMax);
            const std::vector<float> ref = cumsum_on(cb, x, n, rows, kMax);
            for (int r = 0; r < rows; ++r) {
                const float* p = &part[static_cast<size_t>(r) * n];
                unstable += std::memcmp(p, &full[static_cast<size_t>(r) * kMax], sizeof(float) * n) != 0;
                for (int i = 0; i < n; ++i) {
                    const float want = ref[static_cast<size_t>(r) * n + i];
                    worst = std::max(worst, std::fabs(static_cast<double>(p[i]) - want) / std::max(1.0f, std::fabs(want)));
                }
                ++rows_checked;
            }
        }
    }
    std::fprintf(stderr, "prefix or run-to-run differences: %d of %d rows; worst relative distance from the CPU %.3g\n",
                 unstable, rows_checked, worst);
    LOOM_CHECK(unstable == 0);
    // f32 sums of up to 40000 values in [0, 1): the two summation orders sit ~1e-6 apart (6.7e-06 measured).
    LOOM_CHECK(worst < 1e-5);
    LOOM_TEST_REPORT_AND_RETURN();
}
