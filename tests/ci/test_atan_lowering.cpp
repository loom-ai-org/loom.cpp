// ATAN's polynomial fallback, measured against libm (BACKLOG.md P4.7f).
//
// **This is the one lowering in the engine that is not exact**, so unlike the reflect pad and the 1-D
// pool it cannot be checked for bit-identity — it has to be checked for ACCURACY, against a stated
// bound, over a range that includes both sides of the range reduction and the points where a naive
// implementation goes wrong.
//
// The bound is stated in ULPs of the correctly-rounded fp32 result, computed against `atan` in double
// precision. **2.5 ULP**, where the measured maximum is 1.84 — enough headroom that a different
// compiler's FMA contraction or a slightly different rounding on the divide will not turn this red,
// tight enough that a wrong coefficient or a broken reduction cannot pass. For reference, a typical
// GPU vendor's own `atanf` is specified at 2-4 ULP; glibc's is under 1, and glibc is what a CPU build
// still gets, because `op_atan` only reaches for this where the backend cannot run the callback.
//
// The special values matter more than the bulk here, and they are asserted EXACTLY:
//
//   * `atan(0) == 0` — the composition multiplies by `sgn(x)`, and `sgn(0)` is 0, so this works by
//     construction rather than by luck; if that ever stops being true the sign handling needs a rethink.
//   * `atan(±inf) == ±pi/2` — the reduction sends an infinity to `t = 0`, which the reconstruction turns
//     into exactly `pi/2`. An implementation that computed `1/x` naively would produce a NaN here.
//   * `atan(±1) == ±pi/4` — the boundary between the two reduction branches, where `step(a-1)` is 0 and
//     the unreduced branch has to give the same answer the reduced one converges to.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

// ULPs of the fp32 result: the gap between neighbouring floats at that magnitude.
double ulp_error(float got, double want) {
    if (want == 0.0) return got == 0.0f ? 0.0 : std::numeric_limits<double>::infinity();
    const float w = static_cast<float>(want);
    const float spacing = std::nextafter(std::fabs(w), std::numeric_limits<float>::infinity()) - std::fabs(w);
    return std::fabs(static_cast<double>(got) - want) / static_cast<double>(spacing);
}

std::vector<float> run_compose_atan(const std::vector<float>& xs, ggml_backend_t backend) {
    ggml_init_params params{static_cast<size_t>(256) * 1024 * 1024, nullptr, /*no_alloc=*/true};
    ggml_context_ptr ctx(ggml_init(params));
    LOOM_CHECK(ctx != nullptr);

    ggml_tensor* x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, static_cast<int64_t>(xs.size()));
    ggml_set_input(x);
    ggml_tensor* out = loom::compose_atan(ctx.get(), x);
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), 256, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    LOOM_CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));

    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
    LOOM_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> got(xs.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    return got;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    constexpr double kMaxUlp = 2.5;
    constexpr float kPi = 3.14159265358979323846f;

    // --- Special values, exactly ---------------------------------------------------------------------
    {
        const float inf = std::numeric_limits<float>::infinity();
        const std::vector<float> xs{0.0f, -0.0f, 1.0f, -1.0f, inf, -inf};
        const std::vector<float> got = run_compose_atan(xs, backend.get());
        LOOM_CHECK(got[0] == 0.0f);
        LOOM_CHECK(got[1] == 0.0f);
        LOOM_CHECK(got[2] == kPi / 4.0f);
        LOOM_CHECK(got[3] == -kPi / 4.0f);
        LOOM_CHECK(got[4] == kPi / 2.0f);
        LOOM_CHECK(got[5] == -kPi / 2.0f);
    }

    // --- Accuracy across both reduction branches -----------------------------------------------------
    // Three sweeps on purpose: the reduced branch, the folded branch, and the decades around zero where
    // atan(x) ~ x and a relative-error bound is hardest to hold.
    struct Sweep {
        const char* label;
        std::vector<float> xs;
    };
    std::vector<Sweep> sweeps;
    {
        std::vector<float> unit;
        for (int i = -20000; i <= 20000; ++i) unit.push_back(static_cast<float>(i) / 20000.0f);
        sweeps.push_back({"|x| <= 1 (the unreduced branch)", std::move(unit)});

        std::vector<float> folded;
        for (int i = 1; i <= 40000; ++i) {
            const double u = 1.5707 * (static_cast<double>(i) / 40000.0);
            folded.push_back(static_cast<float>(std::tan(u)));
            folded.push_back(static_cast<float>(-std::tan(u)));
        }
        sweeps.push_back({"|x| > 1 (the folded branch)", std::move(folded)});

        std::vector<float> tiny;
        for (int e = -30; e <= 30; ++e) {
            for (int m = 1; m < 10; ++m) {
                tiny.push_back(static_cast<float>(m * std::pow(10.0, e)));
                tiny.push_back(static_cast<float>(-m * std::pow(10.0, e)));
            }
        }
        sweeps.push_back({"decades from 1e-30 to 1e30", std::move(tiny)});
    }

    double worst_overall = 0.0;
    for (const Sweep& sweep : sweeps) {
        const std::vector<float> got = run_compose_atan(sweep.xs, backend.get());
        double worst = 0.0;
        float worst_x = 0.0f;
        for (size_t i = 0; i < sweep.xs.size(); ++i) {
            if (!std::isfinite(sweep.xs[i])) continue;
            LOOM_CHECK(std::isfinite(got[i]));
            const double e = ulp_error(got[i], std::atan(static_cast<double>(sweep.xs[i])));
            if (e > worst) {
                worst = e;
                worst_x = sweep.xs[i];
            }
        }
        std::fprintf(stderr, "  %-34s max %.2f ULP (at x = %g)\n", sweep.label, worst,
                     static_cast<double>(worst_x));
        worst_overall = std::max(worst_overall, worst);
        LOOM_CHECK(worst <= kMaxUlp);
    }
    std::fprintf(stderr, "worst overall: %.2f ULP (bound %.1f)\n", worst_overall, kMaxUlp);

    // --- Monotonicity, which the bound alone does not give ---------------------------------------------
    // atan is strictly increasing, and a polynomial that wobbles inside the ULP bound could still invert
    // two neighbouring inputs. That would be invisible to an error bound and visible to anything that
    // compares phases.
    {
        std::vector<float> xs;
        for (int i = -4000; i <= 4000; ++i) xs.push_back(static_cast<float>(i) / 1000.0f);
        const std::vector<float> got = run_compose_atan(xs, backend.get());
        size_t inversions = 0;
        for (size_t i = 1; i < got.size(); ++i) {
            if (got[i] < got[i - 1]) ++inversions;
        }
        LOOM_CHECK(inversions == 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
