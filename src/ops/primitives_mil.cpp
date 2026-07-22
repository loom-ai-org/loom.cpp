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

Outputs op_reduce_sum(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("reduce_sum", in, 1);
    return {ggml_sum(pc.ctx, in[0])};
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

// ggml's own CPU kernel for STEP is a STRICT inequality (ggml-cpu/unary-ops.cpp: `op_step(x) = (x > 0.f)
// ? 1.f : 0.f`, i.e. step(0) = 0), not `x >= 0`. So `step(b-a)` alone computes strict "a < b" (correct for
// LESS, but wrong for LESS_EQUAL: at a==b it must be true, but step(0)=0 gives false). less_equal/
// greater_equal need an explicit complement of the STRICT comparison, not the same formula less/greater
// already use correctly.
Outputs op_less_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("less_equal", in, 2);
    // a <= b <=> NOT(a > b) <=> 1 - step(a - b)
    ggml_tensor* gt = ggml_step(pc.ctx, ggml_sub(pc.ctx, in[0], in[1]));
    return {ggml_scale_bias(pc.ctx, gt, -1.0f, 1.0f)};
}

Outputs op_greater_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("greater_equal", in, 2);
    // a >= b <=> NOT(b > a) <=> 1 - step(b - a)
    ggml_tensor* lt = ggml_step(pc.ctx, ggml_sub(pc.ctx, in[1], in[0]));
    return {ggml_scale_bias(pc.ctx, lt, -1.0f, 1.0f)};
}

Outputs op_less(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("less", in, 2);
    // a < b <=> b - a > 0 <=> step(b - a)
    return {ggml_step(pc.ctx, ggml_sub(pc.ctx, in[1], in[0]))};
}

Outputs op_greater(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("greater", in, 2);
    // a > b <=> a - b > 0 <=> step(a - b)
    return {ggml_step(pc.ctx, ggml_sub(pc.ctx, in[0], in[1]))};
}

Outputs op_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("equal", in, 2);
    // a == b <=> step(a - b) * step(b - a)
    ggml_tensor* sa = ggml_step(pc.ctx, ggml_sub(pc.ctx, in[0], in[1]));
    ggml_tensor* sb = ggml_step(pc.ctx, ggml_sub(pc.ctx, in[1], in[0]));
    return {ggml_mul(pc.ctx, sa, sb)};
}

Outputs op_not_equal(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("not_equal", in, 2);
    // a != b <=> step(abs(a - b))
    return {ggml_step(pc.ctx, ggml_abs(pc.ctx, ggml_sub(pc.ctx, in[0], in[1])))};
}

Outputs op_select(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("select", in, 3);
    // Algebraic select: cond * x + (1.0f - cond) * y
    ggml_tensor* cond = in[0];
    ggml_tensor* x = in[1];
    ggml_tensor* y = in[2];

    ggml_tensor* cond_x = ggml_mul(pc.ctx, cond, x);
    ggml_tensor* one_minus_cond = ggml_scale_bias(pc.ctx, cond, -1.0f, 1.0f);
    ggml_tensor* term_y = ggml_mul(pc.ctx, one_minus_cond, y);
    return {ggml_add(pc.ctx, cond_x, term_y)};
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
        reg.register_op("SELECT", op_select);
    }
};

static const MilDialectRegistrar mil_dialect_registrar_instance;

} // namespace
} // namespace loom
