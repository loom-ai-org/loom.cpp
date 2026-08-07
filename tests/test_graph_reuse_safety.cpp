// Regression test for the ggml graph-reuse finding documented in BACKLOG.md. It was found through
// `OdeStepper`, which built its graph once and reused the same ggml_cgraph across every integration
// step, and was only safe because EVERY declared input tensor was rewritten before EVERY
// ggml_backend_graph_compute call -- including ones that never logically change (its "conditioning").
// That class is retired (P4.0.8's follow-up; its loop is a `FlowMatchingSampler` component in Lua now),
// but the ggml property is not, and this file was always the thing that pinned it: it isolates the
// behaviour entirely outside GraphBuilder (plain ggml calls only) and shows it is a real, reproducible
// property of ggml_gallocr rather than an artifact of our own primitives.
//
// It still matters to a caller that never sees a driver. GraphBuilder no longer needs the
// rewrite-everything discipline -- its declared inputs live outside the gallocr pool entirely, so
// nothing gallocr placed can alias them (graph_builder.h) -- and this test is what would catch a ggml
// upgrade that made that reasoning wrong.
//
// Root cause: ggml_gallocr may alias a computed tensor's buffer with one of the graph's OWN declared
// INPUT tensors (confirmed via pointer comparison below) -- despite ggml_set_input()'s documented
// contract that inputs get "non-overlapping addresses." This aliasing is invisible and harmless within a
// single compute() pass, but if the graph is REUSED for a second compute() without rewriting an input
// whose buffer got aliased as a previous node's output, that input now silently holds stale/corrupted
// data instead of its last-set value.
//
// This file is still about raw ggml, and everything below still holds there. GraphBuilder itself no
// longer exposes the hazard: since BACKLOG.md P4.0.13 it allocates a topology's declared inputs in
// their own persistent context and backend buffer, outside the gallocr pool, and gallocr never places
// anything on top of a tensor whose data is already set. tests/test_graph_builder_reuse.cpp is that
// half -- it asserts the absence directly, and is what a retained graph's correctness rests on.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include <ggml-cpu.h>

#include <cmath>
#include <vector>

using loom_test::get_f32;
using loom_test::set_f32;

namespace {

constexpr int64_t kIC = 8, kOC = 16, kK = 3, kIL = 8, kP0 = 1; // "same" padding for K=3

// Mirrors loom::op_conv_1d exactly (im2col F32 + mul_mat + reshape).
ggml_tensor* build_conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* data) {
    ggml_tensor* im2col = ggml_im2col(ctx, kernel, data, 1, 0, kP0, 0, 1, 0, /*is_2D=*/false, GGML_TYPE_F32);
    ggml_tensor* result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    return result;
}

std::vector<float> fresh_build_and_compute(ggml_backend_t backend, const std::vector<float>& kernel_data,
                                            const std::vector<float>& input_data) {
    loom_test::GgmlScratch s(backend);
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kK, kIC, kOC);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kIL, kIC);
    ggml_set_input(data);
    ggml_tensor* out = build_conv1d(s.ctx.get(), kernel, data);
    ggml_set_output(out);

    ggml_cgraph* gf = s.expand(out);
    set_f32(kernel, kernel_data);
    set_f32(data, input_data);
    s.compute(gf);
    return get_f32(out);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// CONV_1D reused across two computes, refreshing its one true "input" (the CONV_1D data operand) every
// time -- the same rewrite-every-input discipline the finding above forced. Must match an
// independent fresh rebuild.
void test_conv1d_reuse_with_full_refresh_matches_fresh_rebuild(ggml_backend_t backend) {
    std::vector<float> kernel_data(kK * kIC * kOC);
    for (size_t i = 0; i < kernel_data.size(); ++i) kernel_data[i] = 0.01f * static_cast<float>((i % 13) - 6);
    std::vector<float> input_a(kIL * kIC), input_b(kIL * kIC);
    for (size_t i = 0; i < input_a.size(); ++i) input_a[i] = 0.1f * static_cast<float>((i % 7) - 3);
    for (size_t i = 0; i < input_b.size(); ++i) input_b[i] = 0.1f * static_cast<float>((i % 11) - 5);

    std::vector<float> expected_a = fresh_build_and_compute(backend, kernel_data, input_a);
    std::vector<float> expected_b = fresh_build_and_compute(backend, kernel_data, input_b);

    loom_test::GgmlScratch s(backend);
    ggml_tensor* kernel = ggml_new_tensor_3d(s.ctx.get(), GGML_TYPE_F32, kK, kIC, kOC);
    ggml_set_input(kernel);
    ggml_tensor* data = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, kIL, kIC);
    ggml_set_input(data);
    ggml_tensor* out = build_conv1d(s.ctx.get(), kernel, data);
    ggml_set_output(out);
    ggml_cgraph* gf = s.expand(out);

    set_f32(kernel, kernel_data);
    set_f32(data, input_a);
    s.compute(gf);
    LOOM_CHECK(max_abs_diff(get_f32(out), expected_a) < 1e-4f);

    set_f32(data, input_b); // the only input that changes -- but it's the only one that needs to here
    s.compute(gf);
    LOOM_CHECK(max_abs_diff(get_f32(out), expected_b) < 1e-4f);
}

// Documents the underlying mechanism: ggml_gallocr may assign a computed tensor's buffer to the SAME
// address as one of the graph's own declared inputs (here, the ADD's first operand `a`), even though `a`
// is marked ggml_set_input(). Reusing the graph for a second compute WITHOUT rewriting `a` (only `b`
// changes) then silently uses `a`'s previous-pass *output* value instead of its original one. This test
// exists to catch it if a future ggml upgrade changes this aliasing behavior -- if this assertion starts
// failing, the "rewrite every declared input every step" discipline may no longer be necessary (or a
// different one may be), and BACKLOG.md's finding should be revisited.
void test_unrefreshed_input_gets_silently_aliased(ggml_backend_t backend) {
    loom_test::GgmlScratch s(backend);
    ggml_tensor* a = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 16);
    ggml_set_input(a);
    ggml_tensor* b = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_F32, 16);
    ggml_set_input(b);
    ggml_tensor* out = ggml_add(s.ctx.get(), a, b);
    ggml_set_output(out);
    ggml_cgraph* gf = s.expand(out);

    // Confirms the mechanism directly: out's buffer is the SAME address as one of its declared inputs.
    LOOM_CHECK(out->data == a->data || out->data == b->data);

    std::vector<float> a_data(16), b1(16), b2(16);
    for (int i = 0; i < 16; ++i) { a_data[i] = static_cast<float>(i); b1[i] = 1.0f; b2[i] = 2.0f; }
    set_f32(a, a_data);
    set_f32(b, b1);
    s.compute(gf);
    auto r1 = get_f32(out);
    bool first_call_correct = true;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(r1[i] - (a_data[i] + 1.0f)) > 1e-5f) first_call_correct = false;
    }
    LOOM_CHECK(first_call_correct);

    set_f32(b, b2); // deliberately NOT rewriting `a` -- the unsafe pattern this test documents
    s.compute(gf);
    auto r2 = get_f32(out);
    bool second_call_matches_naive_expectation = true;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(r2[i] - (a_data[i] + 2.0f)) > 1e-5f) second_call_matches_naive_expectation = false;
    }
    LOOM_CHECK(!second_call_matches_naive_expectation); // documents that this naive pattern is unsafe
}

// The actual pattern OdeStepper used: a full vector-field-shaped topology (ADD, ADD, CONV_1D, GELU,
// CONV_1D) built once, with ALL THREE declared inputs (latent/timestep/conditioning) rewritten before
// every compute -- including "conditioning", which never logically changes between steps. Must match an
// independent fresh rebuild given the same inputs.
//
// kernel1/kernel2 (standing in for real model weights) deliberately live in their OWN persistent
// ggml_context + ggml_backend_alloc_ctx_tensors buffer, set once and never touched again -- mirroring how
// GgufModel::load() (gguf_model.cpp) and KvCache's constructor (kv_cache.cpp) actually allocate weights
// and KV storage, entirely outside the ephemeral no_alloc scratch context GraphBuilder/gallocr manages
// per build() call. An earlier version of this test instead created kernel1/kernel2 in the SAME
// GgmlScratch context as latent/timestep/conditioning and marked them ggml_set_input() like a real
// per-step input, but never rewrote them before the second compute() -- exactly the unsafe pattern
// test_unrefreshed_input_gets_silently_aliased() documents above. gallocr aliased kernel1's buffer with
// an intermediate tensor's output during the first compute, silently corrupting it (confirmed via direct
// diagnostic: kernel1's readback differed from its original data, kernel2's didn't -- consistent with
// gallocr's aliasing being real but not guaranteed for every input) before the second compute ran,
// producing NaNs that tripped ggml_compute_forward_gelu_erf_f32's assertion. That was a bug in the TEST's
// own setup, not a production regression: real weights are never gallocr-managed or ggml_set_input()'d in
// the first place, so this exact failure mode cannot happen to them.
void test_full_topology_reuse_with_full_refresh_matches_fresh_rebuild(ggml_backend_t backend) {
    constexpr int64_t T = 8, C = 8, HID = 16, K = 3, P0 = 1;

    auto build_vector_field = [](ggml_context* ctx, ggml_tensor* latent, ggml_tensor* timestep,
                                  ggml_tensor* conditioning, ggml_tensor* kernel1, ggml_tensor* kernel2) {
        ggml_tensor* cur = ggml_add(ctx, latent, timestep);
        cur = ggml_add(ctx, cur, conditioning);
        ggml_tensor* im2col1 = ggml_im2col(ctx, kernel1, cur, 1, 0, P0, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor* hidden = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im2col1, im2col1->ne[0], im2col1->ne[2] * im2col1->ne[1]),
            ggml_reshape_2d(ctx, kernel1, kernel1->ne[0] * kernel1->ne[1], kernel1->ne[2]));
        hidden = ggml_reshape_3d(ctx, hidden, im2col1->ne[1], kernel1->ne[2], im2col1->ne[2]);
        hidden = ggml_gelu_erf(ctx, hidden);
        ggml_tensor* im2col2 = ggml_im2col(ctx, kernel2, hidden, 1, 0, P0, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor* velocity = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im2col2, im2col2->ne[0], im2col2->ne[2] * im2col2->ne[1]),
            ggml_reshape_2d(ctx, kernel2, kernel2->ne[0] * kernel2->ne[1], kernel2->ne[2]));
        velocity = ggml_reshape_3d(ctx, velocity, im2col2->ne[1], kernel2->ne[2], im2col2->ne[2]);
        return velocity;
    };

    std::vector<float> latent_data(T * C), conditioning_data(C, 0.1f);
    std::vector<float> k1_data(K * C * HID), k2_data(K * HID * C);
    for (size_t i = 0; i < latent_data.size(); ++i) latent_data[i] = 0.1f * static_cast<float>((i % 9) - 4);
    for (size_t i = 0; i < k1_data.size(); ++i) k1_data[i] = 0.01f * static_cast<float>((i % 13) - 6);
    for (size_t i = 0; i < k2_data.size(); ++i) k2_data[i] = 0.01f * static_cast<float>((i % 11) - 5);
    const std::vector<float> t0(C, 0.0f), t1(C, 0.125f);

    // Persistent "weight" buffer for kernel1/kernel2 -- separate ggml_context, separate backend buffer,
    // never marked ggml_set_input(), set once and never rewritten. See the comment above for why.
    ggml_context_ptr weights_ctx(ggml_init(ggml_init_params{2 * ggml_tensor_overhead() + 4096, nullptr, /*no_alloc=*/true}));
    ggml_tensor* kernel1 = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, K, C, HID);
    ggml_tensor* kernel2 = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, K, HID, C);
    ggml_backend_buffer_ptr weights_buf(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), backend));
    LOOM_CHECK(weights_buf != nullptr);
    set_f32(kernel1, k1_data);
    set_f32(kernel2, k2_data);

    // The reused graph: two compute() calls, step 1 at t=0 then step 2 at t=0.125, refreshing latent,
    // timestep, AND conditioning both times (conditioning's value never actually changes -- that's the
    // point being tested). kernel1/kernel2 are NOT part of this refresh -- they're weights, not per-step
    // inputs, and live outside gallocr's allocation pool entirely (see above).
    loom_test::GgmlScratch s(backend);
    ggml_tensor* latent = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, T, C);
    ggml_set_input(latent);
    ggml_tensor* timestep = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 1, C);
    ggml_set_input(timestep);
    ggml_tensor* conditioning = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 1, C);
    ggml_set_input(conditioning);
    ggml_tensor* velocity = build_vector_field(s.ctx.get(), latent, timestep, conditioning, kernel1, kernel2);
    ggml_set_output(velocity);
    ggml_cgraph* gf = s.expand(velocity);

    set_f32(latent, latent_data);
    set_f32(timestep, t0);
    set_f32(conditioning, conditioning_data);
    s.compute(gf);

    set_f32(latent, latent_data);
    set_f32(timestep, t1);
    set_f32(conditioning, conditioning_data);
    s.compute(gf);
    std::vector<float> step2_reused = get_f32(velocity);

    // Ground truth: independent fresh build+compute for step 2's inputs.
    loom_test::GgmlScratch s2(backend);
    ggml_tensor* latent2 = ggml_new_tensor_2d(s2.ctx.get(), GGML_TYPE_F32, T, C);
    ggml_set_input(latent2);
    ggml_tensor* timestep2 = ggml_new_tensor_2d(s2.ctx.get(), GGML_TYPE_F32, 1, C);
    ggml_set_input(timestep2);
    ggml_tensor* conditioning2 = ggml_new_tensor_2d(s2.ctx.get(), GGML_TYPE_F32, 1, C);
    ggml_set_input(conditioning2);
    ggml_tensor* k1b = ggml_new_tensor_3d(s2.ctx.get(), GGML_TYPE_F32, K, C, HID);
    ggml_set_input(k1b);
    ggml_tensor* k2b = ggml_new_tensor_3d(s2.ctx.get(), GGML_TYPE_F32, K, HID, C);
    ggml_set_input(k2b);
    ggml_tensor* velocity_b = build_vector_field(s2.ctx.get(), latent2, timestep2, conditioning2, k1b, k2b);
    ggml_set_output(velocity_b);
    ggml_cgraph* gf2 = s2.expand(velocity_b);
    set_f32(k1b, k1_data);
    set_f32(k2b, k2_data);
    set_f32(latent2, latent_data);
    set_f32(timestep2, t1);
    set_f32(conditioning2, conditioning_data);
    s2.compute(gf2);
    std::vector<float> step2_fresh = get_f32(velocity_b);

    LOOM_CHECK(max_abs_diff(step2_reused, step2_fresh) < 1e-4f);
}

} // namespace

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    test_conv1d_reuse_with_full_refresh_matches_fresh_rebuild(backend.get());
    test_unrefreshed_input_gets_silently_aliased(backend.get());
    test_full_topology_reuse_with_full_refresh_matches_fresh_rebuild(backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
