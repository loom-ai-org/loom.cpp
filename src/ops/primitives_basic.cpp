#include "loom/ops/primitive_registry.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

namespace loom {
namespace {

using Json = nlohmann::json;
using Inputs = std::vector<ggml_tensor*>;
using Outputs = std::vector<ggml_tensor*>;

void expect_n_inputs(const char* op, const Inputs& in, size_t n) {
    if (in.size() != n) {
        throw SchemaError(std::string(op) + " expects " + std::to_string(n) + " input(s), got " + std::to_string(in.size()));
    }
}

// ggml's own elementwise-arithmetic kernels (ADD/SUB/MUL/DIV, ggml-cpu/binary-ops.cpp and ops.cpp) only
// ever support same-family FLOAT type combos (F32/F16/BF16) -- there is no integer-arithmetic path at
// all, not even I32-with-I32 (confirmed by reading ggml_compute_forward_add's own type switch: I32 isn't
// one of the listed cases, so it falls to the generic "fatal error" abort). MIL, in contrast, freely does
// real int32 arithmetic wherever a model computes on the "length" input directly (e.g. Conformer-CTC's
// `current_lengths = (length - kernel) // stride + 1`-style subsampled-length formula). Casting any I32
// operand up to F32 before it reaches one of these ops keeps the VALUE correct (this exporter's target
// models only ever do exact, small-integer arithmetic here, well within F32's exact-integer range) while
// staying inside what ggml's kernels can actually execute.
ggml_tensor* promote_i32_to_f32(ggml_context* ctx, ggml_tensor* t) {
    return t->type == GGML_TYPE_I32 ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
}

// `ggml_is_contiguous()` is NOT a sufficient guard before feeding a tensor into ggml-cpu's own
// elementwise binary kernels (ggml-cpu/binary-ops.cpp, backing ADD/SUB/MUL/DIV): reading
// `ggml_is_contiguous_n`'s own real implementation directly (ggml.c) shows it treats `ne[0]==1` as
// VACUOUSLY satisfying the "nb[0] == element size" check regardless of the tensor's actual declared
// stride there (`tensor->ne[0] != ggml_blck_size(tensor->type)` is false whenever `ne[0]==1` for any
// unquantized type, short-circuiting the whole AND before `nb[0]` is even looked at) -- correct for most
// of ggml's own ne[0]-agnostic consumers, but binary-ops.cpp's own vectorized compute loop asserts
// `nb00 == sizeof(src0_t)` unconditionally, with no such carve-out. Confirmed the hard way on Kokoro's
// SineGen: `f0 * float(k)` (this project's own trace-friendly rewrite of a real broadcast multiply,
// see BACKLOG.md) operates on `f0` fresh off a real `.transpose(1,2)` call, producing a genuinely
// PERMUTED tensor with `ne[0]=1` (the trailing torch axis) and a non-unit `nb[0]` -- `ggml_is_
// contiguous()` reports this tensor as contiguous, `op_mul`'s own existing guard (built on exactly that
// call) let it straight through, and the compute-time `GGML_ASSERT(nb00 == sizeof(src0_t))` aborted
// with no informative message at all (a raw crash, not a catchable SchemaError -- this bug predates
// every other informative-error convention this project already established for shape mismatches).
// This helper is the one, general fix: explicitly check `nb[0]` too, not just `ggml_is_contiguous()`.
ggml_tensor* ensure_packed(ggml_context* ctx, ggml_tensor* t) {
    if (!ggml_is_contiguous(t) || t->nb[0] != ggml_type_size(t->type)) {
        return ggml_cont(ctx, t);
    }
    return t;
}

Outputs op_get_rows(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("GET_ROWS", in, 2);
    return {ggml_get_rows(pc.ctx, in[0], in[1])};
}

Outputs op_mul_mat(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("MUL_MAT", in, 2);
    ggml_tensor* a = in[0];
    ggml_tensor* b = in[1];
    
    // Dynamically heal transposed/permuted layouts in matrix multiplication (heads/seq swapped)
    if (a->ne[0] == b->ne[0] && a->ne[1] == b->ne[2] && a->ne[2] == b->ne[1]) {
        a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
    } else if (b->ne[0] == a->ne[0] && b->ne[1] == a->ne[2] && b->ne[2] == a->ne[1]) {
        b = ggml_permute(pc.ctx, b, 0, 2, 1, 3);
    }
    
    // Dynamically heal transposed/permuted input tensors by transposing on the fly
    if (a->ne[0] != b->ne[0] && a->ne[0] == b->ne[1]) {
        b = ggml_transpose(pc.ctx, b);
    }
    
    // Ensure the input tensor b is contiguous in memory for ggml_mul_mat execution
    if (!ggml_is_contiguous(b)) {
        b = ggml_cont(pc.ctx, b);
    }

    // Mirrors ggml.c's own (internal, not exported) ggml_can_mul_mat check.
    const bool can_mul_mat = a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0;
    if (!can_mul_mat) {
        throw SchemaError("MUL_MAT: incompatible shapes a=[" + std::to_string(a->ne[0]) + "," +
                           std::to_string(a->ne[1]) + "," + std::to_string(a->ne[2]) + "," +
                           std::to_string(a->ne[3]) + "] b=[" + std::to_string(b->ne[0]) + "," +
                           std::to_string(b->ne[1]) + "," + std::to_string(b->ne[2]) + "," +
                           std::to_string(b->ne[3]) + "]");
    }
    return {ggml_mul_mat(pc.ctx, a, b)};
}

Outputs op_add(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("ADD", in, 2);
    ggml_tensor* a = promote_i32_to_f32(pc.ctx, in[0]);
    ggml_tensor* b = promote_i32_to_f32(pc.ctx, in[1]);

    // Commutative swapping: ensure the first tensor (a) is always the larger/broadcasting tensor
    if (ggml_nelements(b) > ggml_nelements(a)) {
        std::swap(a, b);
    }
    
    // Dynamically heal heads/seq layout transpositions (axis 1 and 2 swapped)
    if (a->ne[0] == b->ne[0] && a->ne[1] == b->ne[2] && a->ne[2] == b->ne[1]) {
        b = ggml_permute(pc.ctx, b, 0, 2, 1, 3);
    } else if (b->ne[0] == a->ne[0] && b->ne[1] == a->ne[2] && b->ne[2] == a->ne[1]) {
        a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
    }
    
    // Dynamically heal heads/seq layout transpositions with broadcasting (axis 1 and 2 swapped)
    if (a->ne[0] == b->ne[0] && a->ne[2] == b->ne[1] && b->ne[2] == 1) {
        b = ggml_permute(pc.ctx, b, 0, 2, 1, 3);
    } else if (b->ne[0] == a->ne[0] && b->ne[2] == a->ne[1] && a->ne[2] == 1) {
        a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
    }
    
    // NOTE: an earlier revision of this function also had an "axis 0 and 1 swapped" healing branch here
    // (permute+cont whichever operand looked transposed relative to the other, by ne[0]==other.ne[1] &&
    // ne[1]==other.ne[0]). It was a band-aid for the attention-score MUL_MAT operand-order bug (see
    // EXPORT-BACKLOG.md item 1): since ADD's operands here are the same fixed size in the common
    // square-attention-mask case (n_tokens == n_tokens), that shape check can't actually distinguish
    // "needs a swap" from "already correct", so once the exporter started emitting the correct MUL_MAT
    // operand order/transposes directly (tools/loom_mil_compiler/exporter.py's "matmul" op_type
    // handling), this heuristic started re-corrupting already-correct tensors instead of fixing broken
    // ones. Removed rather than made smarter -- the real fix belongs at the exporter, which now knows
    // the true layout instead of guessing from ambiguous shapes.
    a = ensure_packed(pc.ctx, a);
    b = ensure_packed(pc.ctx, b);

    if (!ggml_can_repeat(b, a)) {
        auto ne_str = [](ggml_tensor* t) {
            return "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                   std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]";
        };
        throw SchemaError("ADD: incompatible shapes a=" + ne_str(a) + " b=" + ne_str(b));
    }
    return {ggml_add(pc.ctx, a, b)};
}

Outputs op_sub(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SUB", in, 2);
    ggml_tensor* a = promote_i32_to_f32(pc.ctx, in[0]);
    ggml_tensor* b = promote_i32_to_f32(pc.ctx, in[1]);

    // Dynamically optimize scalar subtraction (e.g. 0.0 - b) to ggml_neg(b) to prevent broadcast failures
    if (ggml_nelements(a) == 1 && ggml_nelements(b) > 1) {
        b = ensure_packed(pc.ctx, b);
        return {ggml_neg(pc.ctx, b)};
    }

    a = ensure_packed(pc.ctx, a);
    b = ensure_packed(pc.ctx, b);
    return {ggml_sub(pc.ctx, a, b)};
}

Outputs op_mul(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("MUL", in, 2);
    ggml_tensor* a = promote_i32_to_f32(pc.ctx, in[0]);
    ggml_tensor* b = promote_i32_to_f32(pc.ctx, in[1]);

    // Commutative swapping: ensure the first tensor (a) is always the larger/broadcasting tensor
    if (ggml_nelements(b) > ggml_nelements(a)) {
        std::swap(a, b);
    }
    
    // Dynamically heal heads/seq layout transpositions (axis 1 and 2 swapped)
    if (a->ne[0] == b->ne[0] && a->ne[1] == b->ne[2] && a->ne[2] == b->ne[1]) {
        b = ggml_permute(pc.ctx, b, 0, 2, 1, 3);
    } else if (b->ne[0] == a->ne[0] && b->ne[1] == a->ne[2] && b->ne[2] == a->ne[1]) {
        a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
    }
    
    // Dynamically heal heads/seq layout transpositions with broadcasting (axis 1 and 2 swapped)
    if (a->ne[0] == b->ne[0] && a->ne[2] == b->ne[1] && b->ne[2] == 1) {
        b = ggml_permute(pc.ctx, b, 0, 2, 1, 3);
    } else if (b->ne[0] == a->ne[0] && b->ne[2] == a->ne[1] && a->ne[2] == 1) {
        a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
    }
    
    // Dynamically heal transposed layouts where axis 0 and 1 are swapped
    if (a->ne[0] == b->ne[1] && a->ne[1] == b->ne[0]) {
        a = ggml_permute(pc.ctx, a, 1, 0, 2, 3);
        a = ggml_cont(pc.ctx, a);
    } else if (b->ne[0] == a->ne[1] && b->ne[1] == a->ne[0]) {
        b = ggml_permute(pc.ctx, b, 1, 0, 2, 3);
        b = ggml_cont(pc.ctx, b);
    }
    
    a = ensure_packed(pc.ctx, a);
    b = ensure_packed(pc.ctx, b);

    if (!ggml_can_repeat(b, a)) {
        auto ne_str = [](ggml_tensor* t) {
            return "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                   std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]";
        };
        throw SchemaError("MUL: incompatible shapes a=" + ne_str(a) + " b=" + ne_str(b));
    }
    return {ggml_mul(pc.ctx, a, b)};
}

Outputs op_div(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("DIV", in, 2);
    ggml_tensor* a = promote_i32_to_f32(pc.ctx, in[0]);
    ggml_tensor* b = promote_i32_to_f32(pc.ctx, in[1]);
    a = ensure_packed(pc.ctx, a);
    b = ensure_packed(pc.ctx, b);
    return {ggml_div(pc.ctx, a, b)};
}

// MIL's `floor_div` (PyTorch `//`) is genuinely NOT the same op as `real_div` -- it floors the quotient,
// same as Python's own integer-division semantics. Needed as its own primitive (not just DIV) because
// this project's own MIL exporter previously mapped both "real_div" and "floor_div" MIL op types to the
// SAME plain "DIV" JSON op, silently dropping the floor -- confirmed as a real, load-bearing bug on
// Conformer-CTC's NeMo-style `calc_length()` subsampled-length formula (`torch.div(lengths+pad-kernel,
// stride) + 1` then floored): coremltools' own tracing had ALSO already eliminated the standalone
// `torch.floor()` MIL op entirely (constant-folded away as a no-op for the specific dummy trace length
// used, an unrelated coremltools quirk), so the `floor_div` op itself is the ONLY place this exporter
// can still recover the real floor semantics for a length computed from user-supplied "length" input
// values the trace never saw. Composed as DIV+FLOOR rather than a native ggml integer-divide (ggml has
// none).
Outputs op_floor_div(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("FLOOR_DIV", in, 2);
    ggml_tensor* a = promote_i32_to_f32(pc.ctx, in[0]);
    ggml_tensor* b = promote_i32_to_f32(pc.ctx, in[1]);
    a = ensure_packed(pc.ctx, a);
    b = ensure_packed(pc.ctx, b);
    return {ggml_floor(pc.ctx, ggml_div(pc.ctx, a, b))};
}

Outputs op_scale(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("SCALE", in, 1);
    const double s = resolve_attr_number(attrs, "s", pc.symbols);
    return {ggml_scale(pc.ctx, in[0], static_cast<float>(s))};
}

Outputs op_sqr(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SQR", in, 1);
    return {ggml_sqr(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_sqrt(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SQRT", in, 1);
    return {ggml_sqrt(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

void rsqrt_custom_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
    const int64_t ne = ggml_nelements(dst);
    const auto* pa = static_cast<const float*>(a->data);
    auto* pd = static_cast<float*>(dst->data);
    const int64_t per_thread = (ne + nth - 1) / nth;
    const int64_t start = std::min(static_cast<int64_t>(ith) * per_thread, ne);
    const int64_t end = std::min(start + per_thread, ne);
    for (int64_t i = start; i < end; ++i) {
        pd[i] = 1.0f / std::sqrt(pa[i]);
    }
}

Outputs op_rsqrt(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("RSQRT", in, 1);
    return {ggml_map_custom1(pc.ctx, ensure_packed(pc.ctx, in[0]), rsqrt_custom_op, GGML_N_TASKS_MAX, nullptr)};
}

Outputs op_log(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("LOG", in, 1);
    return {ggml_log(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

// ggml has no native atan2 (nor even single-arg atan) -- needed for Kokoro's Generator, which computes
// the real phase of its harmonic-source STFT (torch.angle == atan2(imag,real)) as an auxiliary input to
// noise_convs, a genuine trained-weight dependency with no reformulation that avoids the transcendental
// function entirely (unlike RQ_SPLINE_INVERSE's gather-avoidance trick elsewhere in this file). Added via
// ggml_map_custom2, the same "no native op, no viable composition" escape hatch ggml itself provides for
// exactly this case -- CPU-only, but this whole engine is CPU-only today (see BACKLOG.md).
void atan2_custom_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b, int ith, int nth, void*) {
    const int64_t ne = ggml_nelements(dst);
    const auto* pa = static_cast<const float*>(a->data);
    const auto* pb = static_cast<const float*>(b->data);
    auto* pd = static_cast<float*>(dst->data);
    const int64_t per_thread = (ne + nth - 1) / nth;
    const int64_t start = std::min(static_cast<int64_t>(ith) * per_thread, ne);
    const int64_t end = std::min(start + per_thread, ne);
    for (int64_t i = start; i < end; ++i) pd[i] = std::atan2(pa[i], pb[i]);
}

// A single-argument ggml_map_custom1 companion to atan2_custom_op above -- needed because MIL's own
// default pipeline doesn't always keep a traced `torch.atan2` call as one opaque `atan2` op: for Kokoro's
// own STFT phase computation it gets lowered to a plain `atan` op plus separate quadrant-correction
// ops (select/sign/etc., all otherwise-already-supported) instead, confirmed by reading the exported
// topology directly -- not a case this exporter's own "atan2" translation ever gets to run.
void atan_custom_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
    const int64_t ne = ggml_nelements(dst);
    const auto* pa = static_cast<const float*>(a->data);
    auto* pd = static_cast<float*>(dst->data);
    const int64_t per_thread = (ne + nth - 1) / nth;
    const int64_t start = std::min(static_cast<int64_t>(ith) * per_thread, ne);
    const int64_t end = std::min(start + per_thread, ne);
    for (int64_t i = start; i < end; ++i) pd[i] = std::atan(pa[i]);
}

Outputs op_atan(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("ATAN", in, 1);
    return {ggml_map_custom1(pc.ctx, ensure_packed(pc.ctx, in[0]), atan_custom_op, GGML_N_TASKS_MAX, nullptr)};
}

Outputs op_atan2(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // atan2(y=in[0], x=in[1]) -- same argument order as std::atan2/torch.atan2 (y first).
    expect_n_inputs("ATAN2", in, 2);
    return {ggml_map_custom2(pc.ctx, ensure_packed(pc.ctx, in[0]), ensure_packed(pc.ctx, in[1]), atan2_custom_op, GGML_N_TASKS_MAX, nullptr)};
}

void pow_custom_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b, int ith, int nth, void*) {
    const int64_t ne = ggml_nelements(dst);
    const auto* pa = static_cast<const float*>(a->data);
    const auto* pb = static_cast<const float*>(b->data);
    auto* pd = static_cast<float*>(dst->data);
    const int64_t per_thread = (ne + nth - 1) / nth;
    const int64_t start = std::min(static_cast<int64_t>(ith) * per_thread, ne);
    const int64_t end = std::min(start + per_thread, ne);
    const int64_t ne_a = ggml_nelements(a);
    const int64_t ne_b = ggml_nelements(b);
    for (int64_t i = start; i < end; ++i) {
        float val_a = pa[ne_a == 1 ? 0 : i % ne_a];
        float val_b = pb[ne_b == 1 ? 0 : i % ne_b];
        pd[i] = std::pow(val_a, val_b);
    }
}

Outputs op_pow(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // pow(base=in[0], exponent=in[1])
    expect_n_inputs("POW", in, 2);
    return {ggml_map_custom2(pc.ctx, ensure_packed(pc.ctx, in[0]), ensure_packed(pc.ctx, in[1]), pow_custom_op, GGML_N_TASKS_MAX, nullptr)};
}

Outputs op_sum_rows(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SUM_ROWS", in, 1);
    // Sums along ne[0] ("rows" in ggml's terms): input [a,b,c,d] -> output [1,b,c,d]. Callers that need a
    // reduction along a different axis (e.g. mel-spectrogram's time axis) must PERMUTE+CONT first, same
    // pattern as every other axis-sensitive primitive in this engine.
    return {ggml_sum_rows(pc.ctx, in[0])};
}

Outputs op_mean_rows(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("MEAN", in, 1);
    // Computes the mean along ne[0] (the fastest-varying dimension, "rows" in ggml).
    return {ggml_mean(pc.ctx, in[0])};
}

void shape_custom_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
    if (ith != 0) return;
    int32_t* data = static_cast<int32_t*>(dst->data);
    data[0] = static_cast<int32_t>(a->ne[0]);
    data[1] = static_cast<int32_t>(a->ne[1]);
    data[2] = static_cast<int32_t>(a->ne[2]);
    data[3] = static_cast<int32_t>(a->ne[3]);
}

Outputs op_shape(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SHAPE", in, 1);
    // ggml_map_custom1's dst is always dup-shaped from `a` (ggml_dup_tensor internally -- there's no
    // variant that lets a custom op request a different output shape), but shape_custom_op only ever
    // writes the first 4 int32 slots (a's ne[0..3]) regardless of dst's declared size. Downstream
    // consumers (e.g. GET_ROWS extracting one dim via a gather) need a genuine small 4-element
    // shape-vector, not something dup-shaped from the original input -- confirmed by a real crash
    // otherwise (ggml_get_rows's own `a->ne[2] == b->ne[1]` assertion, since the un-viewed tensor's
    // ne[2] equalled the ORIGINAL input's ne[2] rather than 1, whenever that happened to differ from the
    // index tensor's own ne[1]). A zero-copy 1D view of just those first 4 elements is both correct and
    // gives it the shape those consumers actually expect.
    ggml_tensor* f32_shape = ggml_map_custom1(pc.ctx, in[0], shape_custom_op, 1, nullptr);
    ggml_tensor* f32_shape_vec = ggml_view_1d(pc.ctx, f32_shape, 4, 0);
    return {ggml_cast(pc.ctx, f32_shape_vec, GGML_TYPE_I32)};
}

Outputs op_fill(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("FILL", in, 2);
    // in[0] contains the shape (a 1D integer tensor containing the target dimensions)
    // in[1] contains the fill value (a scalar tensor)
    const int32_t* dims = static_cast<const int32_t*>(in[0]->data);
    int64_t ne[4] = {1, 1, 1, 1};
    int64_t rank = ggml_nelements(in[0]);
    for (int64_t i = 0; i < rank && i < 4; ++i) {
        ne[i] = dims[i];
    }
    ggml_tensor* dst = ggml_new_tensor_4d(pc.ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    const float fill_val = *static_cast<const float*>(in[1]->data);
    
    // Fill the newly allocated float tensor elements directly
    float* dst_data = static_cast<float*>(dst->data);
    int64_t num_elements = ggml_nelements(dst);
    for (int64_t i = 0; i < num_elements; ++i) {
        dst_data[i] = fill_val;
    }
    
    return {dst};
}

Outputs op_diag_mask_inf(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("DIAG_MASK_INF", in, 1);
    const int n_past = attrs.contains("n_past") ? static_cast<int>(resolve_attr_int(attrs, "n_past", pc.symbols)) : 0;
    return {ggml_diag_mask_inf(pc.ctx, in[0], n_past)};
}

Outputs op_diag_mask_zero(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("DIAG_MASK_ZERO", in, 1);
    const int n_past = attrs.contains("n_past") ? static_cast<int>(resolve_attr_int(attrs, "n_past", pc.symbols)) : 0;
    return {ggml_diag_mask_zero(pc.ctx, in[0], n_past)};
}

Outputs op_pad_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("PAD_1D", in, 1);
    // Zero-pads ne[0] only (lp0 left, rp0 right) via ggml_pad_ext with every other dimension's pad at 0.
    const int lp0 = static_cast<int>(resolve_attr_int(attrs, "lp0", pc.symbols));
    const int rp0 = static_cast<int>(resolve_attr_int(attrs, "rp0", pc.symbols));
    ggml_tensor* a = ensure_packed(pc.ctx, in[0]);
    return {ggml_pad_ext(pc.ctx, a, lp0, rp0, 0, 0, 0, 0, 0, 0)};
}

Outputs op_pad_1d_reflect(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    // Reflect-pads ne[0] only (lp0 left, rp0 right) via ggml_pad_reflect_1d: [a,b,c,d] -> [b,a,b,c,d,c]
    // (excludes the edge element itself, matching numpy/torch's "reflect" -- not "symmetric" -- padding
    // convention). Needed for MIL's "pad" op with mode="reflect", the shape STFT's center-framing
    // (torch.stft(..., center=True)) traces as once decomposed via coremltools' own
    // common::lower_complex_dialect_ops pass (EXPORT-IMPROVEMENT-BACKLOG.md item 4) -- unlike PAD_1D's
    // zero-fill, this is numerically required for a correct STFT export, not just a shape formality.
    expect_n_inputs("PAD_1D_REFLECT", in, 1);
    const int lp0 = static_cast<int>(resolve_attr_int(attrs, "lp0", pc.symbols));
    const int rp0 = static_cast<int>(resolve_attr_int(attrs, "rp0", pc.symbols));
    ggml_tensor* a = ensure_packed(pc.ctx, in[0]);
    return {ggml_pad_reflect_1d(pc.ctx, a, lp0, rp0)};
}

Outputs op_silu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SILU", in, 1);
    return {ggml_silu(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_relu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("RELU", in, 1);
    return {ggml_relu(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_leaky_relu(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("LEAKY_RELU", in, 1);
    const double slope = resolve_attr_number(attrs, "slope", pc.symbols);
    return {ggml_leaky_relu(pc.ctx, ensure_packed(pc.ctx, in[0]), static_cast<float>(slope), /*inplace=*/false)};
}

Outputs op_concat(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    // Real in-graph channel concatenation (ggml_concat is 2-input; chain multiple CONCAT nodes for 3+
    // tensors, e.g. Kokoro's Decoder torch.cat([asr, F0, N], axis=1)). "dim" is a plain ggml `ne[]` axis
    // index (0 = fastest), NOT a PyTorch axis -- e.g. this project's [T,C] convention (T=ne[0], C=ne[1])
    // wants "dim": 1 to concatenate along channels, matching torch.cat(..., axis=1) on a (C,T)-native
    // tensor once the ggml<->numpy axis-reversal convention is accounted for.
    expect_n_inputs("CONCAT", in, 2);
    const int dim = static_cast<int>(resolve_attr_int(attrs, "dim", pc.symbols));
    return {ggml_concat(pc.ctx, in[0], in[1], dim)};
}

Outputs op_step(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // Heaviside step: 1.0 where input > 0, else 0.0 -- composed with a preceding SUB to build a general
    // "a >= b" comparison mask (STEP(SUB(a,b))), needed by VITS's generate_path/rational-quadratic-spline
    // bucketize, neither of which has any other loom precedent to reuse.
    expect_n_inputs("STEP", in, 1);
    return {ggml_step(pc.ctx, in[0])};
}

Outputs op_cumsum(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("CUMSUM", in, 1);
    return {ggml_cumsum(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_softmax(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SOFTMAX", in, 1);
    return {ggml_soft_max(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_softplus(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SOFTPLUS", in, 1);
    return {ggml_softplus(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_clamp(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CLAMP", in, 1);
    const double lo = resolve_attr_number(attrs, "min", pc.symbols);
    const double hi = resolve_attr_number(attrs, "max", pc.symbols);
    // ggml_clamp's result is a VIEW aliasing its source's buffer (ggml.c calls ggml_view_tensor, not
    // ggml_dup_tensor), i.e. it clamps in place -- if `in[0]` is also read elsewhere in the topology
    // (e.g. by another node besides this CLAMP), that other read would silently observe the clamped
    // value instead of the original once this op runs, regardless of the two nodes' relative order in
    // the JSON topology. Clamping a ggml_cont'd copy instead (a genuine, separately-allocated tensor)
    // keeps `in[0]`'s own buffer untouched for any other consumer. Found via a real instance of this
    // exact hazard in RQ_SPLINE_INVERSE (see primitives_spline.cpp) -- fixed here too since CLAMP is a
    // generic, independently reusable primitive with no way to know its input won't be shared.
    return {ggml_clamp(pc.ctx, ggml_cont(pc.ctx, in[0]), static_cast<float>(lo), static_cast<float>(hi))};
}

Outputs op_gelu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("GELU", in, 1);
    // ggml_gelu_erf computes the exact 0.5*x*(1+erf(x/sqrt(2))) formula (no lookup table), unlike
    // ggml_gelu/ggml_gelu_quick which are tanh/sigmoid approximations -- chosen for exact reproducibility
    // against a numpy reference, same rationale as ATTENTION's composite (non-flash) path.
    // ggml_unary (which ggml_gelu_erf composes to) asserts contiguous rows -- needed once VITS's DDSConv
    // started feeding a real strided VIEW/CONT-less intermediate straight into GELU, same "cont before
    // unary/softmax/conv ops that assert dense strides internally" fix already applied to op_softmax and
    // every conv primitive.
    ggml_tensor* x = ensure_packed(pc.ctx, in[0]);
    return {ggml_gelu_erf(pc.ctx, x)};
}

Outputs op_swiglu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SWIGLU", in, 2);
    return {ggml_swiglu_split(pc.ctx, ensure_packed(pc.ctx, in[0]), ensure_packed(pc.ctx, in[1]))};
}

Outputs op_rms_norm(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("RMS_NORM", in, 1);
    const double eps = resolve_attr_number(attrs, "eps", pc.symbols);
    return {ggml_rms_norm(pc.ctx, ensure_packed(pc.ctx, in[0]), static_cast<float>(eps))};
}

Outputs op_layer_norm(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("LAYER_NORM", in, 1);
    // ggml_norm is the full mean+variance normalization (unlike RMS_NORM, which skips mean-centering);
    // like RMS_NORM, it leaves the learned affine (weight/bias) to separate MUL/ADD nodes in the
    // topology rather than folding them in here.
    ggml_tensor* x0 = ensure_packed(pc.ctx, in[0]);
    const double eps = resolve_attr_number(attrs, "eps", pc.symbols);
    return {ggml_norm(pc.ctx, x0, static_cast<float>(eps))};
}

Outputs op_group_norm(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("GROUP_NORM", in, 1);
    // ggml_group_norm groups over ne[2] ("channels") and reduces jointly over ne[0]*ne[1] ("spatial")
    // within each channel-group (confirmed directly against ggml-cpu/ops.cpp's
    // ggml_compute_forward_group_norm_f32) -- this project's own Layout A [T,C] convention (T=ne[0],
    // C=ne[1]) must be RESHAPE'd to [T,1,C,1] before calling this and RESHAPE'd back afterward, same
    // "reshape into the convention a native op expects" precedent as CONV_1D_DW's internal 4D reshape.
    // Like RMS_NORM/LAYER_NORM, the learned per-channel affine (weight/bias) is left to separate
    // MUL/ADD nodes rather than folded in here.
    const int n_groups = static_cast<int>(resolve_attr_int(attrs, "n_groups", pc.symbols));
    const double eps = resolve_attr_number(attrs, "eps", pc.symbols);
    return {ggml_group_norm(pc.ctx, in[0], n_groups, static_cast<float>(eps))};
}

Outputs op_sigmoid(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SIGMOID", in, 1);
    return {ggml_sigmoid(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_tanh(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("TANH", in, 1);
    return {ggml_tanh(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

// 1D resize along ne[0] only (ne[1..3] held fixed at the input's own size) -- added for Kokoro TTS's
// several 1D up/downsample spots (SineGen's phase pre/post-filtering, UpSample1d's 2x nearest upsample,
// the F0 curve's huge nearest upsample to full waveform rate). Wraps ggml's own native
// ggml_interpolate(...) directly rather than hand-composing (same "use ggml's op when one exists"
// principle as ggml_clamp/ggml_leaky_relu/POOL_1D elsewhere in this project).
//
// "linear" maps to GGML_SCALE_MODE_BILINEAR, NOT a dedicated 1D-linear ggml mode (none exists) -- but
// with ne[1] held at the input's own value, ggml_compute_forward_interpolate's bilinear branch
// (ops.cpp) degenerates to an EXACT 1D linear interpolation: its own `dy` blend factor is always
// exactly 0 whenever sf1==1.0 (ne1==ne01), so the ne[1]-axis contributes nothing and only the ne[0]-axis
// formula (half-pixel-center, matching PyTorch's own F.interpolate(mode='linear', align_corners=False)
// convention) applies -- confirmed by reading ops.cpp's actual computation directly, then verified
// numerically against real torch.nn.functional.interpolate (see test_interpolate_1d).
Outputs op_interpolate_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("INTERPOLATE_1D", in, 1);
    const int64_t ne0 = resolve_attr_int(attrs, "ne0", pc.symbols);
    const std::string mode = attrs.value("mode", "nearest");
    ggml_scale_mode scale_mode;
    if (mode == "nearest") scale_mode = GGML_SCALE_MODE_NEAREST;
    else if (mode == "linear") scale_mode = GGML_SCALE_MODE_BILINEAR;
    else throw SchemaError("INTERPOLATE_1D: unsupported 'mode' \"" + mode + "\" (expected \"nearest\" or \"linear\")");
    ggml_tensor* a = ensure_packed(pc.ctx, in[0]);
    return {ggml_interpolate(pc.ctx, a, ne0, a->ne[1], a->ne[2], a->ne[3], static_cast<uint32_t>(scale_mode))};
}

Outputs op_exp(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // Needed for Kokoro Generator's final spec/phase split (`spec = exp(x[:n_freq])`).
    expect_n_inputs("EXP", in, 1);
    return {ggml_exp(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_sin(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SIN", in, 1);
    return {ggml_sin(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_cos(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("COS", in, 1);
    return {ggml_cos(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_floor(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // Needed for Kokoro's SineGen (`(f0/sampling_rate) % 1`, expressed as `x - floor(x)` since ggml has
    // no native modulo op and every operand here is non-negative, so this is an exact match to `%`).
    expect_n_inputs("FLOOR", in, 1);
    return {ggml_floor(pc.ctx, ensure_packed(pc.ctx, in[0]))};
}

Outputs op_glu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("GLU", in, 1);
    // Sigmoid-gated GLU (Dauphin et al.): out = a * sigmoid(b), where [a, b] = split(x, dim=channels).
    // This is NOT the same as ggml's own GGML_GLU_OP_* family (SWIGLU/GEGLU/REGLU all gate with
    // SiLU/GELU/ReLU instead of a plain sigmoid), so there's no single ggml call for it -- composed here
    // from two VIEWs + SIGMOID + MUL. Channels are ne[1] in this engine's CONV_1D output convention
    // (matching PyTorch's nn.functional.glu(x, dim=1) on a channels-first conv activation), so the split
    // is along ne[1].
    ggml_tensor* x = in[0];
    if (x->ne[1] % 2 != 0) {
        throw SchemaError("GLU: input's ne[1] (channel dim) must be even, got " + std::to_string(x->ne[1]));
    }
    const int64_t half = x->ne[1] / 2;
    ggml_tensor* a = ggml_view_3d(pc.ctx, x, x->ne[0], half, x->ne[2], x->nb[1], x->nb[2], 0);
    ggml_tensor* b = ggml_view_3d(pc.ctx, x, x->ne[0], half, x->ne[2], x->nb[1], x->nb[2], half * x->nb[1]);
    return {ggml_mul(pc.ctx, ggml_cont(pc.ctx, a), ggml_sigmoid(pc.ctx, ggml_cont(pc.ctx, b)))};
}

Outputs op_reshape(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("RESHAPE", in, 1);
    if (!attrs.contains("shape") || !attrs.at("shape").is_array()) {
        throw SchemaError("RESHAPE: 'shape' attribute must be an array");
    }
    // ggml_reshape_*d requires a contiguous source (it reinterprets the flat buffer in place) -- a
    // real (non-identity) PERMUTE immediately before a RESHAPE, as attention head-splitting produces,
    // yields a non-contiguous view and would otherwise hit ggml_reshape's own assertion.
    ggml_tensor* src = in[0];
    if (!ggml_is_contiguous(src)) {
        src = ggml_cont(pc.ctx, src);
    }
    // Same convention as numpy/PyTorch reshape: at most one entry may be the literal -1, meaning "infer
    // this dimension from the input's total element count" -- needed when a dimension (e.g. a
    // subsampled frame count) isn't known as a named symbol at topology-authoring time, only as
    // whatever the actual input tensor's size happens to be once the graph is built.
    std::vector<int64_t> shape;
    int infer_idx = -1;
    int64_t known_product = 1;
    const Json& shape_json = attrs.at("shape");
    for (size_t i = 0; i < shape_json.size(); ++i) {
        const Json& v = shape_json[i];
        if (v.is_number_integer() && v.get<int64_t>() == -1) {
            if (infer_idx != -1) {
                throw SchemaError("RESHAPE: at most one -1 entry is allowed in 'shape'");
            }
            infer_idx = static_cast<int>(i);
            shape.push_back(-1);
        } else {
            const int64_t d = v.is_string() ? static_cast<int64_t>(std::llround(pc.symbols.eval(v.get<std::string>())))
                                             : static_cast<int64_t>(std::llround(v.get<double>()));
            shape.push_back(d);
            known_product *= d;
        }
    }
    if (infer_idx != -1) {
        if (known_product == 0 || ggml_nelements(src) % known_product != 0) {
            throw SchemaError("RESHAPE: input element count is not evenly divisible by the known 'shape' dimensions");
        }
        shape[infer_idx] = ggml_nelements(src) / known_product;
    } else if (known_product != ggml_nelements(src)) {
        std::string shape_str;
        for (auto d : shape) shape_str += std::to_string(d) + ",";
        throw SchemaError("RESHAPE: target shape [" + shape_str + "] has " + std::to_string(known_product) +
                           " elements but input has " + std::to_string(ggml_nelements(src)) +
                           " (src ne=[" + std::to_string(src->ne[0]) + "," + std::to_string(src->ne[1]) + "," +
                           std::to_string(src->ne[2]) + "," + std::to_string(src->ne[3]) + "])");
    }
    switch (shape.size()) {
        case 1: return {ggml_reshape_1d(pc.ctx, src, shape[0])};
        case 2: return {ggml_reshape_2d(pc.ctx, src, shape[0], shape[1])};
        case 3: return {ggml_reshape_3d(pc.ctx, src, shape[0], shape[1], shape[2])};
        case 4: return {ggml_reshape_4d(pc.ctx, src, shape[0], shape[1], shape[2], shape[3])};
        default: throw SchemaError("RESHAPE 'shape' attribute must have 1-4 entries, got " + std::to_string(shape.size()));
    }
}

Outputs op_repeat(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    // Wraps ggml_repeat_4d (broadcasts `a` up to an explicit target shape, no template tensor needed --
    // unlike ggml's own 2-tensor ggml_repeat, which would force fabricating an otherwise-unused tensor
    // just to carry a shape). Needed for StyleTTS2's diffusion-net Transformer1d: a single per-batch
    // style vector [channels] must be broadcast to [channels, T] (T dynamic, a "$n_tokens"-style symbol)
    // before CONCAT-ing with the per-position BERT context embedding -- CONCAT itself requires matching
    // shape on every non-concat axis, so the broadcast has to happen as its own explicit step first.
    // `shape`'s entries follow RESHAPE's own convention (string symbols evaluated via SymbolEnv, or
    // plain numbers) but with NO -1 inference (repeat's target must always be fully explicit -- ggml
    // itself requires each target dim to be an exact multiple of `a`'s own, checked by ggml_repeat_4d's
    // own assertion).
    expect_n_inputs("REPEAT", in, 1);
    if (!attrs.contains("shape") || !attrs.at("shape").is_array()) {
        throw SchemaError("REPEAT: 'shape' attribute must be an array");
    }
    const Json& shape_json = attrs.at("shape");
    size_t shape_size = std::min(shape_json.size(), (size_t)4);
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < shape_size; ++i) {
        const Json& v = shape_json[i];
        ne[i] = v.is_string() ? static_cast<int64_t>(std::llround(pc.symbols.eval(v.get<std::string>())))
                               : static_cast<int64_t>(std::llround(v.get<double>()));
    }
    ggml_tensor* a = in[0];
    
    // Dynamically heal transposed/permuted input layouts (axis 1 and 2 swapped) relative to repeat target
    if (a->ne[0] == ne[0] && a->ne[1] != ne[1] && a->ne[2] != ne[2]) {
        if (a->ne[1] == ne[2] / (ne[2] / a->ne[1]) && a->ne[2] == ne[1]) {
            a = ggml_permute(pc.ctx, a, 0, 2, 1, 3);
        }
    }
    
    // Dynamically heal transposed layouts where axis 0 and 1 are swapped
    if (a->ne[0] == ne[1] && a->ne[1] == ne[0]) {
        a = ggml_permute(pc.ctx, a, 1, 0, 2, 3);
        a = ggml_cont(pc.ctx, a);
    }
    
    // Guarantee memory contiguity for ggml_repeat
    if (!ggml_is_contiguous(a)) {
        a = ggml_cont(pc.ctx, a);
    }
    
    return {ggml_repeat_4d(pc.ctx, a, ne[0], ne[1], ne[2], ne[3])};
}

Outputs op_view(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("VIEW", in, 1);
    ggml_tensor* parent = in[0];
    if (!ggml_is_contiguous(parent)) {
        parent = ggml_cont(pc.ctx, parent);
    }
    const std::vector<int64_t> shape = resolve_attr_int_array(attrs, "shape", pc.symbols);
    const int64_t offset = attrs.contains("offset") ? resolve_attr_int(attrs, "offset", pc.symbols) : 0;
    for (int64_t d : shape) {
        if (d <= 0) {
            std::string shape_str;
            for (auto s : shape) shape_str += std::to_string(s) + ",";
            throw SchemaError("VIEW: non-positive dimension in resolved shape [" + shape_str +
                               "], offset=" + std::to_string(offset) + ", parent ne=[" +
                               std::to_string(parent->ne[0]) + "," + std::to_string(parent->ne[1]) + "," +
                               std::to_string(parent->ne[2]) + "," + std::to_string(parent->ne[3]) + "]");
        }
    }
    // Default nb1/nb2/nb3 MUST come from the PARENT's own existing strides, not be recomputed from the
    // new (target) shape -- recomputing from the target shape is only correct when the sliced axis's
    // size is unchanged from the parent. When the view shrinks ne0 itself (e.g. truncating a causal
    // conv's output from 130 back down to 128 real timesteps, keeping ne1=channels), the parent's true
    // row stride is still based on its ORIGINAL (larger) ne0, so `shape[0]*elem_size` silently
    // underestimates every row's stride -- row 0 still starts at the right place (offset 0 either way),
    // but every subsequent row/channel reads from the wrong location. Confirmed on LFM2's ShortConv
    // layers: channel 0 of the truncated conv output matched the reference model exactly, every other
    // channel didn't -- exactly this failure signature.
    const size_t nb1 = attrs.contains("nb1") ? static_cast<size_t>(resolve_attr_int(attrs, "nb1", pc.symbols)) : parent->nb[1];
    const size_t nb2 = attrs.contains("nb2") ? static_cast<size_t>(resolve_attr_int(attrs, "nb2", pc.symbols)) : parent->nb[2];
    const size_t nb3 = attrs.contains("nb3") ? static_cast<size_t>(resolve_attr_int(attrs, "nb3", pc.symbols)) : parent->nb[3];
    const size_t nb[4] = {parent->nb[0], nb1, nb2, nb3};

    // A strided view's highest byte it can ever touch is `offset + sum((shape[i]-1) * nb[i]) + elemsize`
    // -- checked explicitly here (rather than leaving it to ggml_view_Nd's own internal GGML_ASSERT,
    // which aborts the process outright with no context) so an out-of-bounds VIEW -- e.g. a mis-derived
    // dynamic shape/offset expression evaluating to something the parent tensor was never sized for --
    // surfaces as a normal SchemaError with the offending node's name (via GraphBuilder's own wrapping),
    // the same diagnostic upgrade every other primitives_basic.cpp bounds check already got.
    size_t last_byte = static_cast<size_t>(offset);
    for (size_t i = 0; i < shape.size(); ++i) {
        last_byte += static_cast<size_t>(shape[i] - 1) * nb[i];
    }
    last_byte += parent->nb[0];
    if (last_byte > ggml_nbytes(parent)) {
        std::string shape_str;
        for (auto s : shape) shape_str += std::to_string(s) + ",";
        throw SchemaError("VIEW: resolved shape [" + shape_str + "] at offset " + std::to_string(offset) +
                           " (nb=[" + std::to_string(nb[0]) + "," + std::to_string(nb[1]) + "," +
                           std::to_string(nb[2]) + "," + std::to_string(nb[3]) + "]) needs " +
                           std::to_string(last_byte) + " bytes but parent has " +
                           std::to_string(ggml_nbytes(parent)) + " (parent ne=[" +
                           std::to_string(parent->ne[0]) + "," + std::to_string(parent->ne[1]) + "," +
                           std::to_string(parent->ne[2]) + "," + std::to_string(parent->ne[3]) + "])");
    }

    switch (shape.size()) {
        case 1:
            return {ggml_view_1d(pc.ctx, parent, shape[0], offset)};
        case 2:
            return {ggml_view_2d(pc.ctx, parent, shape[0], shape[1], nb1, static_cast<size_t>(offset))};
        case 3:
            return {ggml_view_3d(pc.ctx, parent, shape[0], shape[1], shape[2], nb1, nb2, static_cast<size_t>(offset))};
        case 4:
            return {ggml_view_4d(pc.ctx, parent, shape[0], shape[1], shape[2], shape[3], nb1, nb2, nb3, static_cast<size_t>(offset))};
        default: throw SchemaError("VIEW 'shape' attribute must have 1-4 entries, got " + std::to_string(shape.size()));
    }
}

Outputs op_permute(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("PERMUTE", in, 1);
    const std::vector<int64_t> axes = resolve_attr_int_array(attrs, "axes", pc.symbols);
    if (axes.size() != 4) {
        throw SchemaError("PERMUTE 'axes' attribute must have exactly 4 entries, got " + std::to_string(axes.size()));
    }
    return {ggml_permute(pc.ctx, in[0], static_cast<int>(axes[0]), static_cast<int>(axes[1]),
                          static_cast<int>(axes[2]), static_cast<int>(axes[3]))};
}

Outputs op_cont(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("CONT", in, 1);
    return {ggml_cont(pc.ctx, in[0])};
}

// Genuine dtype conversions (e.g. MIL's `cast(x, dtype="fp32")` on an integer tensor, as HF's rotary
// embedding code does to `position_ids` before the inv-freq matmul) -- NOT to be confused with the
// exporter's own fp16<->fp32 "cast" ops, which it aliases away as no-ops since this engine always
// computes in f32 internally regardless of a GGUF weight's storage dtype. A real int<->float
// reinterpretation must go through ggml_cast, or a downstream op (e.g. MUL_MAT) that assumes float data
// will read garbage / crash calling a null vec_dot for an integer src0 type.
Outputs op_cast(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CAST", in, 1);
    if (!attrs.contains("dtype") || !attrs.at("dtype").is_string()) {
        throw SchemaError("CAST requires a string 'dtype' attribute");
    }
    const std::string dtype = attrs.at("dtype").get<std::string>();
    ggml_type type;
    if (dtype == "f32") type = GGML_TYPE_F32;
    else if (dtype == "f16") type = GGML_TYPE_F16;
    else if (dtype == "i32") type = GGML_TYPE_I32;
    else throw SchemaError("CAST: unsupported target dtype '" + dtype + "'");
    return {ggml_cast(pc.ctx, in[0], type)};
}

} // namespace

LOOM_REGISTER_OP(GET_ROWS, op_get_rows)
LOOM_REGISTER_OP(CAST, op_cast)
LOOM_REGISTER_OP(MUL_MAT, op_mul_mat)
LOOM_REGISTER_OP(ADD, op_add)
LOOM_REGISTER_OP(SUB, op_sub)
LOOM_REGISTER_OP(MUL, op_mul)
LOOM_REGISTER_OP(DIV, op_div)
LOOM_REGISTER_OP(FLOOR_DIV, op_floor_div)
LOOM_REGISTER_OP(SCALE, op_scale)
LOOM_REGISTER_OP(SQR, op_sqr)
LOOM_REGISTER_OP(SQRT, op_sqrt)
LOOM_REGISTER_OP(RSQRT, op_rsqrt)
LOOM_REGISTER_OP(LOG, op_log)
LOOM_REGISTER_OP(ATAN, op_atan)
LOOM_REGISTER_OP(ATAN2, op_atan2)
LOOM_REGISTER_OP(POW, op_pow)
LOOM_REGISTER_OP(SUM_ROWS, op_sum_rows)
LOOM_REGISTER_OP(MEAN, op_mean_rows)
LOOM_REGISTER_OP(SHAPE, op_shape)
LOOM_REGISTER_OP(FILL, op_fill)
LOOM_REGISTER_OP(DIAG_MASK_INF, op_diag_mask_inf)
LOOM_REGISTER_OP(DIAG_MASK_ZERO, op_diag_mask_zero)
LOOM_REGISTER_OP(PAD_1D, op_pad_1d)
LOOM_REGISTER_OP(PAD_1D_REFLECT, op_pad_1d_reflect)
LOOM_REGISTER_OP(SILU, op_silu)
LOOM_REGISTER_OP(RELU, op_relu)
LOOM_REGISTER_OP(LEAKY_RELU, op_leaky_relu)
LOOM_REGISTER_OP(STEP, op_step)
LOOM_REGISTER_OP(CONCAT, op_concat)
LOOM_REGISTER_OP(CUMSUM, op_cumsum)
LOOM_REGISTER_OP(SOFTMAX, op_softmax)
LOOM_REGISTER_OP(SOFTPLUS, op_softplus)
LOOM_REGISTER_OP(CLAMP, op_clamp)
LOOM_REGISTER_OP(SWIGLU, op_swiglu)
LOOM_REGISTER_OP(RMS_NORM, op_rms_norm)
LOOM_REGISTER_OP(LAYER_NORM, op_layer_norm)
LOOM_REGISTER_OP(GROUP_NORM, op_group_norm)
LOOM_REGISTER_OP(SIGMOID, op_sigmoid)
LOOM_REGISTER_OP(TANH, op_tanh)
LOOM_REGISTER_OP(EXP, op_exp)
LOOM_REGISTER_OP(SIN, op_sin)
LOOM_REGISTER_OP(COS, op_cos)
LOOM_REGISTER_OP(INTERPOLATE_1D, op_interpolate_1d)
LOOM_REGISTER_OP(FLOOR, op_floor)
LOOM_REGISTER_OP(GLU, op_glu)
LOOM_REGISTER_OP(RESHAPE, op_reshape)
LOOM_REGISTER_OP(REPEAT, op_repeat)
LOOM_REGISTER_OP(VIEW, op_view)
LOOM_REGISTER_OP(PERMUTE, op_permute)
LOOM_REGISTER_OP(CONT, op_cont)
LOOM_REGISTER_OP(GELU, op_gelu)

} // namespace loom
