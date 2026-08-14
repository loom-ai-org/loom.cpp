// PAD_1D_REFLECT composes itself from views and concatenations when the backend has no
// `ggml_pad_reflect_1d`, and this is the test that the composition is the SAME operation
// (BACKLOG.md P4.7e).
//
// **Bit-identity, not a tolerance.** A slice does no arithmetic, so there is nothing to round: the two
// spellings either select the same elements or they do not. A tolerance here would hide exactly the bug
// worth catching -- an off-by-one in the reflection, which produces a plausible waveform that is wrong
// at both edges of every frame.
//
// **The composition is called directly, and it has to be.** The decision that selects it
// (`backend_can_run`) cannot be reached hermetically: a CPU backend implements every op, so the
// primitive always takes the native branch on any machine CI runs on. So rather than trying to provoke
// the branch, this holds the two spellings against each other over one input -- and whichever branch a
// device later takes, it takes one of two things already known to be identical.
//
// Reflect padding excludes the edge element (torch's "reflect", not "symmetric"): `[a,b,c,d]` with
// (1,1) becomes `[b,a,b,c,d,c]`. The hand-written expectation below states that for one small case,
// so the test does not merely check the two implementations against each other -- both could be wrong
// the same way, and ggml's own convention is what neither of them gets to define.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct PadCase {
    const char* label;
    int64_t ne0;
    int64_t ne1;
    int lp0;
    int rp0;
};

// Builds `PAD_1D_REFLECT` through the registry (whatever it decides to emit) and, separately, the raw
// `ggml_pad_reflect_1d`, over one input in one graph. Returns how many outputs differ in any bit.
size_t bit_differences(const PadCase& c, ggml_backend_t backend, std::vector<float>* composed_out) {
    ggml_init_params params{static_cast<size_t>(64) * 1024 * 1024, nullptr, /*no_alloc=*/true};
    ggml_context_ptr ctx(ggml_init(params));
    LOOM_CHECK(ctx != nullptr);

    ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.ne0, c.ne1);
    ggml_set_input(x);

    ggml_tensor* native = ggml_pad_reflect_1d(ctx.get(), x, c.lp0, c.rp0);

    // The helper the primitive delegates to when the backend lacks the op -- called directly, because
    // that branch is unreachable on a CPU backend (which implements everything) and would otherwise
    // never be exercised by a hermetic test at all.
    ggml_tensor* composed = loom::compose_pad_reflect_1d(ctx.get(), x, c.lp0, c.rp0);

    ggml_set_output(native);
    ggml_set_output(composed);
    LOOM_CHECK(native->ne[0] == composed->ne[0]);
    LOOM_CHECK(native->ne[1] == composed->ne[1]);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), 4096, false);
    ggml_build_forward_expand(gf, native);
    ggml_build_forward_expand(gf, composed);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    LOOM_CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));

    std::vector<float> input(static_cast<size_t>(c.ne0) * static_cast<size_t>(c.ne1));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(0.41f * static_cast<float>(i)) * 2.0f - 0.25f;
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    LOOM_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> a(static_cast<size_t>(ggml_nelements(native)));
    std::vector<float> b(static_cast<size_t>(ggml_nelements(composed)));
    ggml_backend_tensor_get(native, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(composed, b.data(), 0, b.size() * sizeof(float));
    if (composed_out != nullptr) *composed_out = b;

    size_t differences = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++differences;
    }
    return differences;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // --- The composition IS reflect padding, stated independently of both implementations -------------
    // [a,b,c,d] with (2,1) -> [c,b,a,b,c,d,c]. If ggml's convention and this project's understanding of
    // it ever diverge, this is the assertion that notices; the comparisons below could not, since they
    // only hold the two spellings against each other.
    {
        ggml_init_params params{static_cast<size_t>(1) * 1024 * 1024, nullptr, /*no_alloc=*/true};
        ggml_context_ptr ctx(ggml_init(params));
        ggml_tensor* x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        ggml_set_input(x);
        ggml_tensor* out = loom::compose_pad_reflect_1d(ctx.get(), x, 2, 1);
        ggml_set_output(out);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
        LOOM_CHECK(ggml_gallocr_alloc_graph(galloc.get(), gf));
        const std::vector<float> in{10.0f, 20.0f, 30.0f, 40.0f};
        ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));
        LOOM_CHECK(ggml_backend_graph_compute(backend.get(), gf) == GGML_STATUS_SUCCESS);
        std::vector<float> got(static_cast<size_t>(ggml_nelements(out)));
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
        const std::vector<float> want{30.0f, 20.0f, 10.0f, 20.0f, 30.0f, 40.0f, 30.0f};
        LOOM_CHECK(got.size() == want.size());
        for (size_t i = 0; i < want.size(); ++i) LOOM_CHECK(got[i] == want[i]);
    }

    // --- ...and it matches ggml's own primitive, bit for bit, across shapes and widths ----------------
    const PadCase cases[] = {
        {"(1,0), the narrowest real case (kokoro)", 64, 3, 1, 0},
        {"(0,1), the other side alone", 64, 3, 0, 1},
        {"(1,1), symmetric and minimal", 64, 3, 1, 1},
        {"(10,10), kokoro/styletts2's STFT centre-framing", 4096, 1, 10, 10},
        {"(10,10) over several rows -- strides matter", 512, 7, 10, 10},
        {"(3,7), asymmetric", 128, 5, 3, 7},
        {"(31,1), just inside the compose limit", 256, 2, 31, 1},
    };
    for (const PadCase& c : cases) {
        const size_t differences = bit_differences(c, backend.get(), nullptr);
        if (differences != 0) {
            std::fprintf(stderr, "  %s: %zu output(s) differ from ggml_pad_reflect_1d\n", c.label,
                          differences);
        }
        LOOM_CHECK(differences == 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
