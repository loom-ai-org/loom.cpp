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


// 2. static registrar mapping CoreML MIL names to Loom definitions dynamically:
struct MilDialectRegistrar {
    MilDialectRegistrar() {
        PrimitiveRegistry& reg = PrimitiveRegistry::instance();
        
        // Register new C++ custom CoreML MIL operators
        reg.register_op("abs", op_abs);
        reg.register_op("ABS", op_abs);
        reg.register_op("neg", op_neg);
        reg.register_op("NEG", op_neg);
        reg.register_op("sign", op_sign);
        reg.register_op("SIGN", op_sign);
        reg.register_op("minimum", op_minimum);
        reg.register_op("MINIMUM", op_minimum);
        reg.register_op("maximum", op_maximum);
        reg.register_op("MAXIMUM", op_maximum);
        reg.register_op("reduce_sum", op_reduce_sum);
        reg.register_op("REDUCE_SUM", op_reduce_sum);
        reg.register_op("identity", op_identity);
        reg.register_op("IDENTITY", op_identity);
        
        // Dynamically clone and register aliases of standard Loom operators to their MIL lowercase keys
        auto try_alias = [&](const std::string& mil_name, const std::string& loom_name) {
            if (reg.has(loom_name)) {
                reg.register_op(mil_name, reg.get(loom_name));
            }
        };
        
        try_alias("add", "ADD");
        try_alias("sub", "SUB");
        try_alias("mul", "MUL");
        try_alias("div", "DIV");
        try_alias("real_div", "DIV");
        try_alias("floor_div", "DIV");
        try_alias("pow", "POW");
        try_alias("reduce_mean", "MEAN");
        try_alias("rsqrt", "RSQRT");
        try_alias("sqrt", "SQRT");
        try_alias("square", "SQUARE");
        try_alias("relu", "RELU");
        try_alias("relu6", "RELU6");
        try_alias("leaky_relu", "LEAKY_RELU");
        try_alias("gelu", "GELU");
        try_alias("sigmoid", "SIGMOID");
        try_alias("tanh", "TANH");
        try_alias("silu", "SILU");
        try_alias("softmax", "SOFTMAX");
        try_alias("softplus", "SOFTPLUS");
        try_alias("matmul", "MUL_MAT");
        try_alias("gather", "GET_ROWS");
        try_alias("transpose", "PERMUTE");
        try_alias("reshape", "RESHAPE");
        try_alias("concat", "CONCAT");
        try_alias("split", "VIEW");
        try_alias("slice_by_index", "VIEW");
        try_alias("tile", "REPEAT");
        try_alias("band_part", "DIAG_MASK_ZERO");
        try_alias("conv", "CONV_1D"); // Automatically heals/routes standard and depthwise shapes
        try_alias("layer_norm", "LAYER_NORM");
        try_alias("rms_norm", "RMS_NORM");
        try_alias("clipping", "CLAMP");
        try_alias("clip", "CLAMP");
        try_alias("fill", "FILL");
        try_alias("shape", "SHAPE");
    }
};

static const MilDialectRegistrar mil_dialect_registrar_instance;

} // namespace
} // namespace loom
