// Implementations of CoreML Model Intermediate Language (MIL) dialect primitives using GGML.
// Dynamically maps all standard MIL operators and registers lowercase aliases to Loom primitives.

#include "loom/ops/primitive_registry.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

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

// ggml's own elementwise-arithmetic kernels (ADD/SUB/MUL/DIV) only support same-family FLOAT type combos
// (F32/F16/BF16), never I32 -- see primitives_basic.cpp's own identical helper for the full explanation
// (duplicated here rather than shared via a header, matching this file's existing per-TU-helper
// convention, e.g. `expect_n_inputs` just above).
ggml_tensor* promote_i32_to_f32(ggml_context* ctx, ggml_tensor* t) {
    return t->type == GGML_TYPE_I32 ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
}

// 1. Implementation of brand new CoreML MIL primitives using GGML:

Outputs op_abs(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("abs", in, 1);
    return {ggml_abs(pc.ctx, in[0])};
}

Outputs op_neg(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("neg", in, 1);
    return {ggml_neg(pc.ctx, in[0])};
}

Outputs op_sign(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("sign", in, 1);
    return {ggml_sgn(pc.ctx, in[0])};
}

Outputs op_minimum(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("minimum", in, 2);
    // Algebraic reduction: min(a, b) = 0.5 * (a + b - abs(a - b))
    ggml_tensor* sum = ggml_add(pc.ctx, in[0], in[1]);
    ggml_tensor* diff = ggml_sub(pc.ctx, in[0], in[1]);
    ggml_tensor* abs_diff = ggml_abs(pc.ctx, diff);
    ggml_tensor* sub_diff = ggml_sub(pc.ctx, sum, abs_diff);
    return {ggml_scale(pc.ctx, sub_diff, 0.5f)};
}

Outputs op_maximum(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("maximum", in, 2);
    // Algebraic reduction: max(a, b) = 0.5 * (a + b + abs(a - b))
    ggml_tensor* sum = ggml_add(pc.ctx, in[0], in[1]);
    ggml_tensor* diff = ggml_sub(pc.ctx, in[0], in[1]);
    ggml_tensor* abs_diff = ggml_abs(pc.ctx, diff);
    ggml_tensor* add_diff = ggml_add(pc.ctx, sum, abs_diff);
    return {ggml_scale(pc.ctx, add_diff, 0.5f)};
}

Outputs op_reduce_sum(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    // Real per-axis reduction, driven by an explicit "axis" (ne-order) + "keep_dims" attr the exporter
    // now writes (tools/loom_mil_compiler/exporter.py's dedicated "reduce_sum" translation) -- MIL's
    // `reduce_sum` almost always reduces over ONE specific axis (e.g. Conformer-CTC's mel-frontend
    // magnitude computation, `sum(x**2, axis=-1)` over a stacked real/imag pair; CMVN's per-channel
    // mean/variance, summed over time), never every element -- the old always-`ggml_sum`-to-a-scalar
    // behavior silently produced a degenerate 1-element result for every one of these, discovered via a
    // downstream MUL_MAT shape-mismatch crash while exporting Conformer-CTC.
    expect_n_inputs("reduce_sum", in, 1);
    ggml_tensor* x = in[0];
    if (!attrs.contains("axis")) {
        // No axis info reached this primitive (e.g. an older/hand-written topology never updated to
        // pass one) -- fall back to the previous full-reduction behavior rather than failing outright.
        return {ggml_sum(pc.ctx, x)};
    }
    const int axis = static_cast<int>(resolve_attr_number(attrs, "axis", pc.symbols));
    const bool keep_dims = attrs.value("keep_dims", false);
    if (axis < 0 || axis > 3) {
        throw SchemaError("REDUCE_SUM: 'axis' must resolve to 0-3 (ne-order), got " + std::to_string(axis));
    }

    // ggml_sum_rows only ever sums ne[0] -- permute the target axis into position 0, sum, then permute
    // back. A single-swap permutation is its own inverse, so the same `perm` array undoes it.
    int perm[4] = {0, 1, 2, 3};
    perm[0] = axis;
    perm[axis] = 0;
    ggml_tensor* permuted = ggml_cont(pc.ctx, ggml_permute(pc.ctx, x, perm[0], perm[1], perm[2], perm[3]));
    ggml_tensor* summed = ggml_sum_rows(pc.ctx, permuted);
    ggml_tensor* result = ggml_cont(pc.ctx, ggml_permute(pc.ctx, summed, perm[0], perm[1], perm[2], perm[3]));

    if (keep_dims) {
        return {result};
    }
    // Genuinely drop the now-size-1 axis (not just leave it in place) -- downstream ops expecting the
    // MIL-declared reduced rank need the real element layout a RESHAPE gives, not merely a size-1 ne
    // entry sitting between faster-varying axes.
    int64_t new_ne[3];
    int j = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == axis) continue;
        new_ne[j++] = result->ne[i];
    }
    switch (j) {
        case 1: return {ggml_reshape_1d(pc.ctx, result, new_ne[0])};
        case 2: return {ggml_reshape_2d(pc.ctx, result, new_ne[0], new_ne[1])};
        case 3: return {ggml_reshape_3d(pc.ctx, result, new_ne[0], new_ne[1], new_ne[2])};
        default: return {result};
    }
}

Outputs op_identity(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("identity", in, 1);
    return {ggml_dup(pc.ctx, in[0])};
}

float get_tensor_scalar_value(ggml_tensor* t) {
    if (!t || !t->data) return 0.0f;
    if (t->type == GGML_TYPE_F32) {
        return *(float*)(t->data);
    } else if (t->type == GGML_TYPE_I32) {
        return static_cast<float>(*(int32_t*)(t->data));
    } else if (t->type == GGML_TYPE_I16) {
        return static_cast<float>(*(int16_t*)(t->data));
    } else if (t->type == GGML_TYPE_I8) {
        return static_cast<float>(*(int8_t*)(t->data));
    }
    return 0.0f;
}

Outputs op_range_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    float start = 0.0f;
    float end = 0.0f;
    float step = 1.0f;

    if (attrs.contains("start")) {
        start = static_cast<float>(resolve_attr_number(attrs, "start", pc.symbols));
    } else if (!in.empty()) {
        if (in[0]->data) {
            start = get_tensor_scalar_value(in[0]);
        }
    }
    
    if (attrs.contains("end")) {
        end = static_cast<float>(resolve_attr_number(attrs, "end", pc.symbols));
    } else if (in.size() > 1) {
        if (in[1]->data) {
            end = get_tensor_scalar_value(in[1]);
        } else {
            // Dynamic sequence length fallback to prevent GGML stop > start failures
            double n_tokens = 1.0;
            try {
                n_tokens = pc.symbols.eval("n_tokens");
            } catch (...) {}
            end = start + static_cast<float>(n_tokens) * step;
        }
    }

    if (attrs.contains("step")) {
        step = static_cast<float>(resolve_attr_number(attrs, "step", pc.symbols));
    } else if (in.size() > 2) {
        if (in[2]->data) {
            step = get_tensor_scalar_value(in[2]);
        }
    }

    // Force strict stop > start bound to guarantee no GGML assertion crashes
    if (end <= start) {
        end = start + 1.0f;
    }

    return {ggml_arange(pc.ctx, start, end, step)};
}

// ggml_sub(x, y) requires x's shape to be the OUTPUT/target shape, with y merely repeated into it (see
// ggml.c's own ggml_sub_impl: `GGML_ASSERT(ggml_can_repeat(b, a))`, i.e. b must be repeatable INTO a, never
// the reverse). MIL's comparison ops place no such constraint on which operand is x vs y, so computing the
// difference in the "wrong" direction (smaller operand first) crashes outright the moment the two operands
// genuinely differ in size -- confirmed as a real, reproducible crash (EXPORT-BACKLOG.md item 3: LFM2's
// dynamic-shape export hits an `EQUAL` comparing a small per-axis concat against a scalar constant).
// Returns `x - y`, correctly broadcast regardless of which operand is larger, mirroring the same
// "commutative swap to the larger operand" convention op_add/op_mul already use in primitives_basic.cpp.
ggml_tensor* sub_broadcast(ggml_context* ctx, ggml_tensor* x, ggml_tensor* y) {
    // MIL freely mixes an int32 "length" input straight into arithmetic/comparisons against an
    // f32-typed tensor (e.g. `torch.arange(T) < length`, RANGE_1D's own output is always F32), matching
    // PyTorch's own implicit int->float promotion -- but ggml's SUB kernel has no I32 support at all
    // (see `promote_i32_to_f32`'s own docstring). Confirmed as the real, previously-undiagnosed root
    // cause of a "binary_op: unsupported types" COMPUTE-time abort on Conformer-CTC's length-validity
    // mask (`valid_mask`/`pad_mask`/`time_mask`, all comparisons against the real "length" graph input),
    // reached only once every upstream shape/attention bug was fixed -- earlier attempts crashed before
    // ever exercising this code path.
    x = promote_i32_to_f32(ctx, x);
    y = promote_i32_to_f32(ctx, y);
    if (ggml_nelements(y) > ggml_nelements(x)) {
        return ggml_neg(ctx, ggml_sub(ctx, y, x));
    }
    return ggml_sub(ctx, x, y);
}

// ggml's own CPU kernel for STEP is a STRICT inequality (ggml-cpu/unary-ops.cpp: `op_step(x) = (x > 0.f)
// ? 1.f : 0.f`, i.e. step(0) = 0), not `x >= 0`. So `step(b-a)` alone computes strict "a < b" (correct for
// LESS, but wrong for LESS_EQUAL: at a==b it must be true, but step(0)=0 gives false). less_equal/
// greater_equal need an explicit complement of the STRICT comparison, not the same formula less/greater
// already use correctly.
Outputs op_less_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("less_equal", in, 2);
    // a <= b <=> NOT(a > b) <=> 1 - step(a - b)
    ggml_tensor* gt = ggml_step(pc.ctx, sub_broadcast(pc.ctx, in[0], in[1]));
    return {ggml_scale_bias(pc.ctx, gt, -1.0f, 1.0f)};
}

Outputs op_greater_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("greater_equal", in, 2);
    // a >= b <=> NOT(b > a) <=> 1 - step(b - a)
    ggml_tensor* lt = ggml_step(pc.ctx, sub_broadcast(pc.ctx, in[1], in[0]));
    return {ggml_scale_bias(pc.ctx, lt, -1.0f, 1.0f)};
}

Outputs op_less(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("less", in, 2);
    // a < b <=> b - a > 0 <=> step(b - a)
    return {ggml_step(pc.ctx, sub_broadcast(pc.ctx, in[1], in[0]))};
}

Outputs op_greater(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("greater", in, 2);
    // a > b <=> a - b > 0 <=> step(a - b)
    return {ggml_step(pc.ctx, sub_broadcast(pc.ctx, in[0], in[1]))};
}

Outputs op_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("equal", in, 2);
    // a == b <=> NOT(a > b) AND NOT(a < b) <=> (1 - step(d)) * (1 - step(-d)), where d = a - b.
    // NOTE: the previous formula (step(a-b) * step(b-a)) was simply wrong, independent of any
    // broadcasting concern: step(x) * step(-x) requires x>0 AND -x>0 simultaneously, which is never
    // true, so it evaluated to 0 (false) for EVERY input, including a==b.
    ggml_tensor* d = sub_broadcast(pc.ctx, in[0], in[1]);
    ggml_tensor* not_gt = ggml_scale_bias(pc.ctx, ggml_step(pc.ctx, d), -1.0f, 1.0f);
    ggml_tensor* not_lt = ggml_scale_bias(pc.ctx, ggml_step(pc.ctx, ggml_neg(pc.ctx, d)), -1.0f, 1.0f);
    return {ggml_mul(pc.ctx, not_gt, not_lt)};
}

Outputs op_not_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("not_equal", in, 2);
    // a != b <=> step(abs(a - b))
    return {ggml_step(pc.ctx, ggml_abs(pc.ctx, sub_broadcast(pc.ctx, in[0], in[1])))};
}

Outputs op_not(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("logical_not", in, 1);
    // NOT(x) <=> 1 - x, for the boolean-as-0/1-float convention every comparison op above already uses.
    return {ggml_scale_bias(pc.ctx, in[0], -1.0f, 1.0f)};
}

ggml_tensor* mul_broadcast(ggml_context* ctx, ggml_tensor* x, ggml_tensor* y) {
    // ggml_mul(a, b) requires `a` to be the OUTPUT/target shape with `b` merely repeated into it (same
    // constraint ggml_sub has -- see sub_broadcast above), but MIL's `select`/elementwise ops place no
    // such constraint on operand order. Orient correctly regardless of which operand is larger, same
    // "commutative swap" convention op_add/op_mul (primitives_basic.cpp) already use.
    x = promote_i32_to_f32(ctx, x);
    y = promote_i32_to_f32(ctx, y);
    ggml_tensor* a = x;
    ggml_tensor* b = y;
    if (ggml_nelements(y) > ggml_nelements(x)) {
        a = y;
        b = x;
    }
    if (!ggml_can_repeat(b, a)) {
        auto ne_str = [](ggml_tensor* t) {
            return "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                   std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]";
        };
        throw SchemaError("MUL: incompatible shapes a=" + ne_str(a) + " b=" + ne_str(b));
    }
    return ggml_mul(ctx, a, b);
}

ggml_tensor* add_broadcast(ggml_context* ctx, ggml_tensor* x, ggml_tensor* y) {
    // Same "commutative swap to the larger operand" convention as sub_broadcast/mul_broadcast above --
    // needed because op_select's two mul_broadcast products (cond*x and (1-cond)*y) are each
    // independently oriented to THEIR OWN larger operand, so the smaller product can end up first
    // (e.g. cond broadcasts a scalar `x` while `y` is the full-size operand: cond_x stays cond-sized,
    // term_y grows to y's full size) -- a plain ggml_add(cond_x, term_y) then has its target/repeat
    // operands backwards.
    x = promote_i32_to_f32(ctx, x);
    y = promote_i32_to_f32(ctx, y);
    ggml_tensor* a = x;
    ggml_tensor* b = y;
    if (ggml_nelements(y) > ggml_nelements(x)) {
        a = y;
        b = x;
    }
    if (!ggml_can_repeat(b, a)) {
        auto ne_str = [](ggml_tensor* t) {
            return "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                   std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]";
        };
        throw SchemaError("ADD: incompatible shapes a=" + ne_str(a) + " b=" + ne_str(b));
    }
    return ggml_add(ctx, a, b);
}

Outputs op_select(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("select", in, 3);
    // Algebraic select: cond * x + (1.0f - cond) * y
    ggml_tensor* cond = in[0];
    ggml_tensor* x = in[1];
    ggml_tensor* y = in[2];

    ggml_tensor* cond_x = mul_broadcast(pc.ctx, cond, x);
    ggml_tensor* one_minus_cond = ggml_scale_bias(pc.ctx, cond, -1.0f, 1.0f);
    ggml_tensor* term_y = mul_broadcast(pc.ctx, one_minus_cond, y);
    return {add_broadcast(pc.ctx, cond_x, term_y)};
}


// 2. static registrar mapping CoreML MIL names to Loom definitions:
//
// Only uppercase names are registered here. tools/loom_mil_compiler/exporter.py's OP_MAP always
// normalizes every MIL op to its uppercase Loom name before ever writing topology JSON, so the C++
// primitive registry never sees a lowercase op string from that exporter's output -- confirmed by
// reading generate_graph_topology, whose emitted `node["op"]` is always OP_MAP's uppercase value. A
// prior revision of this registrar also registered every op (and aliased ADD/SUB/MUL/... to their MIL
// lowercase names via a `try_alias` helper) under its lowercase MIL name too; that half was dead code by
// the same argument and has been removed (EXPORT-BACKLOG.md item 5).
struct MilDialectRegistrar {
    MilDialectRegistrar() {
        PrimitiveRegistry& reg = PrimitiveRegistry::instance();

        // Register new C++ custom CoreML MIL operators
        reg.register_op("ABS", op_abs);
        reg.register_op("NEG", op_neg);
        reg.register_op("SIGN", op_sign);
        reg.register_op("MINIMUM", op_minimum);
        reg.register_op("MAXIMUM", op_maximum);
        reg.register_op("REDUCE_SUM", op_reduce_sum);
        reg.register_op("IDENTITY", op_identity);
        reg.register_op("RANGE_1D", op_range_1d);
        reg.register_op("LESS_EQUAL", op_less_equal);
        reg.register_op("GREATER_EQUAL", op_greater_equal);
        reg.register_op("LESS", op_less);
        reg.register_op("GREATER", op_greater);
        reg.register_op("EQUAL", op_equal);
        reg.register_op("NOT_EQUAL", op_not_equal);
        reg.register_op("NOT", op_not);
        reg.register_op("SELECT", op_select);
    }
};

static const MilDialectRegistrar mil_dialect_registrar_instance;

} // namespace
} // namespace loom
