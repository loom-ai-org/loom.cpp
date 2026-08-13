// POOL_1D falls back to a one-tall `ggml_pool_2d` when the backend has no `ggml_pool_1d` -- CUDA and
// Vulkan do not, Metal and SYCL do (BACKLOG.md P4.7d/P4.7e). This is the test that says "the fallback is
// the same operation" is a checked claim and not a hopeful one.
//
// The CHOICE is not tested here and cannot be: `backend_can_run` answers yes for everything on a CPU
// backend, so a hermetic test always takes the native branch. What is testable, and what matters, is
// that the thing on the other branch is indistinguishable from the thing on this one.
//
// **The claim is bit-identity against `ggml_pool_1d` itself**, across a matrix of (op, kernel, stride,
// padding), computed on the same input in the same graph. A tolerance would be the wrong instrument:
// pooling selects or averages, and the two spellings either visit the same elements and divide by the
// same number or they do not.
//
// **One row of that matrix is why the guard exists.** An AVERAGE pool with padding is NOT the same
// operation in the two spellings, and the difference is invisible in every interior window:
//
//   * `ggml_pool_1d` divides by `count` -- how many in-bounds elements the window actually covered;
//   * `ggml_pool_2d` divides by `ka = k0*k1` -- the whole kernel, counting the padded cells as zeros.
//
// So they agree everywhere except the windows that overhang an edge, which is a handful of values at
// each end of each row. That is a difference no shape check catches, no gate test with a tolerance is
// guaranteed to catch, and no reviewer would suspect from the ggml API. It was found by running exactly
// the comparison below before shipping the lowering, which is the only reason the guard exists at all.
//
// The test therefore asserts BOTH directions: the combinations the engine promises to lower are
// bit-identical, and the combination it refuses to lower really would have differed.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

// A deterministic input with both signs and no ties, so a max pool's answer is a specific element
// rather than a coincidence.
std::vector<float> ramp(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = std::sin(0.37f * static_cast<float>(i)) * 3.0f - 0.5f;
    return v;
}

struct PoolCase {
    const char* label;
    int64_t ne0;
    int64_t ne1;
    int k0;
    int s0;
    int p0;
    ggml_op_pool op;
};

// Runs both spellings over one input in one graph and returns how many outputs differ in any bit.
size_t bit_differences(const PoolCase& c, ggml_backend_t backend) {
    ggml_init_params params{static_cast<size_t>(64) * 1024 * 1024, nullptr, /*no_alloc=*/true};
    ggml_context_ptr ctx(ggml_init(params));
    LOOM_CHECK(ctx != nullptr);

    ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.ne0, c.ne1);
    ggml_set_input(x);
    ggml_tensor* one_d = ggml_pool_1d(ctx.get(), x, c.op, c.k0, c.s0, c.p0);
    ggml_tensor* two_d = ggml_pool_2d(ctx.get(), x, c.op, c.k0, /*k1=*/1, c.s0, /*s1=*/1,
                                       static_cast<float>(c.p0), /*p1=*/0.0f);
    ggml_set_output(one_d);
    ggml_set_output(two_d);

    // The shapes must agree unconditionally -- calc_pool_output_size(ne1, 1, 1, 0) == ne1 -- and they do
    // so even in the average-with-padding case, which is precisely what makes that case dangerous.
    LOOM_CHECK(one_d->ne[0] == two_d->ne[0]);
    LOOM_CHECK(one_d->ne[1] == two_d->ne[1]);
    LOOM_CHECK(one_d->ne[2] == two_d->ne[2]);
    LOOM_CHECK(one_d->ne[3] == two_d->ne[3]);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, one_d);
    ggml_build_forward_expand(gf, two_d);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    LOOM_CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));

    const std::vector<float> input = ramp(static_cast<size_t>(c.ne0) * static_cast<size_t>(c.ne1));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    LOOM_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> a(static_cast<size_t>(ggml_nelements(one_d)));
    std::vector<float> b(static_cast<size_t>(ggml_nelements(two_d)));
    ggml_backend_tensor_get(one_d, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(two_d, b.data(), 0, b.size() * sizeof(float));

    size_t differences = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++differences;
    }
    return differences;
}

} // namespace

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- Everything op_pool_1d promises to lower is bit-identical ------------------------------------
    const PoolCase equivalent[] = {
        // Whisper's own case: a global maximum spelled as a pool whose kernel is the whole axis. This is
        // the one that motivated the lowering (BACKLOG.md P4.7d).
        {"global max, k = s = the whole axis", 4096, 1, 4096, 4096, 0, GGML_OP_POOL_MAX},
        {"global max over many rows", 1000, 80, 1000, 1000, 0, GGML_OP_POOL_MAX},
        {"max, k=2 s=2, unpadded", 64, 8, 2, 2, 0, GGML_OP_POOL_MAX},
        {"max, k=3 s=2, PADDED", 64, 8, 3, 2, 1, GGML_OP_POOL_MAX},
        {"max, ragged tail (101 with k=4 s=3)", 101, 5, 4, 3, 0, GGML_OP_POOL_MAX},
        {"avg, k=2 s=2, unpadded", 64, 8, 2, 2, 0, GGML_OP_POOL_AVG},
        {"avg, ragged tail (101 with k=4 s=3)", 101, 5, 4, 3, 0, GGML_OP_POOL_AVG},
    };
    for (const PoolCase& c : equivalent) {
        LOOM_CHECK(loom::pool_2d_fallback_is_equivalent(c.op, c.p0));
        const size_t differences = bit_differences(c, backend.get());
        if (differences != 0) {
            std::fprintf(stderr, "  %s: %zu output(s) differ between the two spellings\n", c.label,
                          differences);
        }
        LOOM_CHECK(differences == 0);
    }

    // --- ...and the one it refuses really would have differed ----------------------------------------
    // A test that only checked the equivalent cases would still pass if the guard were deleted, because
    // it would never construct the case the guard exists for. This is that case.
    {
        const PoolCase padded_average{"avg, k=3 s=2, PADDED", 64, 8, 3, 2, 1, GGML_OP_POOL_AVG};
        LOOM_CHECK(!loom::pool_2d_fallback_is_equivalent(padded_average.op, padded_average.p0));
        const size_t differences = bit_differences(padded_average, backend.get());
        std::fprintf(stderr, "padded average: %zu of %d outputs differ between the spellings "
                              "(pool_1d divides by the in-bounds count, pool_2d by the full kernel)\n",
                     differences, 32 * 8);
        LOOM_CHECK(differences > 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
