#include "loom/ops/primitive_registry.h"
#include "loom/loom_errors.h"

#include <nlohmann/json.hpp>

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

Outputs op_get_rows(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("GET_ROWS", in, 2);
    return {ggml_get_rows(pc.ctx, in[0], in[1])};
}

Outputs op_mul_mat(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("MUL_MAT", in, 2);
    return {ggml_mul_mat(pc.ctx, in[0], in[1])};
}

Outputs op_add(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("ADD", in, 2);
    return {ggml_add(pc.ctx, in[0], in[1])};
}

Outputs op_sub(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SUB", in, 2);
    return {ggml_sub(pc.ctx, in[0], in[1])};
}

Outputs op_mul(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("MUL", in, 2);
    return {ggml_mul(pc.ctx, in[0], in[1])};
}

Outputs op_div(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("DIV", in, 2);
    return {ggml_div(pc.ctx, in[0], in[1])};
}

Outputs op_scale(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("SCALE", in, 1);
    const double s = resolve_attr_number(attrs, "s", pc.symbols);
    return {ggml_scale(pc.ctx, in[0], static_cast<float>(s))};
}

Outputs op_sqr(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SQR", in, 1);
    return {ggml_sqr(pc.ctx, in[0])};
}

Outputs op_sqrt(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SQRT", in, 1);
    return {ggml_sqrt(pc.ctx, in[0])};
}

Outputs op_log(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("LOG", in, 1);
    return {ggml_log(pc.ctx, in[0])};
}

Outputs op_sum_rows(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SUM_ROWS", in, 1);
    // Sums along ne[0] ("rows" in ggml's terms): input [a,b,c,d] -> output [1,b,c,d]. Callers that need a
    // reduction along a different axis (e.g. mel-spectrogram's time axis) must PERMUTE+CONT first, same
    // pattern as every other axis-sensitive primitive in this engine.
    return {ggml_sum_rows(pc.ctx, in[0])};
}

Outputs op_pad_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("PAD_1D", in, 1);
    // Zero-pads ne[0] only (lp0 left, rp0 right) via ggml_pad_ext with every other dimension's pad at 0.
    const int lp0 = static_cast<int>(resolve_attr_int(attrs, "lp0", pc.symbols));
    const int rp0 = static_cast<int>(resolve_attr_int(attrs, "rp0", pc.symbols));
    return {ggml_pad_ext(pc.ctx, in[0], lp0, rp0, 0, 0, 0, 0, 0, 0)};
}

Outputs op_silu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SILU", in, 1);
    return {ggml_silu(pc.ctx, in[0])};
}

Outputs op_relu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("RELU", in, 1);
    return {ggml_relu(pc.ctx, in[0])};
}

Outputs op_leaky_relu(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("LEAKY_RELU", in, 1);
    const double slope = resolve_attr_number(attrs, "slope", pc.symbols);
    return {ggml_leaky_relu(pc.ctx, in[0], static_cast<float>(slope), /*inplace=*/false)};
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
    return {ggml_cumsum(pc.ctx, in[0])};
}

Outputs op_softmax(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SOFTMAX", in, 1);
    return {ggml_soft_max(pc.ctx, in[0])};
}

Outputs op_softplus(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SOFTPLUS", in, 1);
    return {ggml_softplus(pc.ctx, in[0])};
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
    return {ggml_gelu_erf(pc.ctx, in[0])};
}

Outputs op_swiglu(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SWIGLU", in, 2);
    return {ggml_swiglu_split(pc.ctx, in[0], in[1])};
}

Outputs op_rms_norm(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("RMS_NORM", in, 1);
    const double eps = resolve_attr_number(attrs, "eps", pc.symbols);
    return {ggml_rms_norm(pc.ctx, in[0], static_cast<float>(eps))};
}

Outputs op_layer_norm(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("LAYER_NORM", in, 1);
    // ggml_norm is the full mean+variance normalization (unlike RMS_NORM, which skips mean-centering);
    // like RMS_NORM, it leaves the learned affine (weight/bias) to separate MUL/ADD nodes in the
    // topology rather than folding them in here.
    const double eps = resolve_attr_number(attrs, "eps", pc.symbols);
    return {ggml_norm(pc.ctx, in[0], static_cast<float>(eps))};
}

Outputs op_sigmoid(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SIGMOID", in, 1);
    return {ggml_sigmoid(pc.ctx, in[0])};
}

Outputs op_tanh(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("TANH", in, 1);
    return {ggml_tanh(pc.ctx, in[0])};
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
        if (known_product == 0 || ggml_nelements(in[0]) % known_product != 0) {
            throw SchemaError("RESHAPE: input element count is not evenly divisible by the known 'shape' dimensions");
        }
        shape[infer_idx] = ggml_nelements(in[0]) / known_product;
    }
    switch (shape.size()) {
        case 1: return {ggml_reshape_1d(pc.ctx, in[0], shape[0])};
        case 2: return {ggml_reshape_2d(pc.ctx, in[0], shape[0], shape[1])};
        case 3: return {ggml_reshape_3d(pc.ctx, in[0], shape[0], shape[1], shape[2])};
        case 4: return {ggml_reshape_4d(pc.ctx, in[0], shape[0], shape[1], shape[2], shape[3])};
        default: throw SchemaError("RESHAPE 'shape' attribute must have 1-4 entries, got " + std::to_string(shape.size()));
    }
}

Outputs op_view(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("VIEW", in, 1);
    const std::vector<int64_t> shape = resolve_attr_int_array(attrs, "shape", pc.symbols);
    const int64_t offset = attrs.contains("offset") ? resolve_attr_int(attrs, "offset", pc.symbols) : 0;
    switch (shape.size()) {
        case 1:
            return {ggml_view_1d(pc.ctx, in[0], shape[0], offset)};
        case 2: {
            const size_t nb1 = attrs.contains("nb1") ? static_cast<size_t>(resolve_attr_int(attrs, "nb1", pc.symbols))
                                                      : in[0]->nb[1];
            return {ggml_view_2d(pc.ctx, in[0], shape[0], shape[1], nb1, static_cast<size_t>(offset))};
        }
        case 3: {
            const size_t nb1 = attrs.contains("nb1") ? static_cast<size_t>(resolve_attr_int(attrs, "nb1", pc.symbols))
                                                      : in[0]->nb[1];
            const size_t nb2 = attrs.contains("nb2") ? static_cast<size_t>(resolve_attr_int(attrs, "nb2", pc.symbols))
                                                      : in[0]->nb[2];
            return {ggml_view_3d(pc.ctx, in[0], shape[0], shape[1], shape[2], nb1, nb2, static_cast<size_t>(offset))};
        }
        default: throw SchemaError("VIEW 'shape' attribute must have 1-3 entries, got " + std::to_string(shape.size()));
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

} // namespace

LOOM_REGISTER_OP(GET_ROWS, op_get_rows)
LOOM_REGISTER_OP(MUL_MAT, op_mul_mat)
LOOM_REGISTER_OP(ADD, op_add)
LOOM_REGISTER_OP(SUB, op_sub)
LOOM_REGISTER_OP(MUL, op_mul)
LOOM_REGISTER_OP(DIV, op_div)
LOOM_REGISTER_OP(SCALE, op_scale)
LOOM_REGISTER_OP(SQR, op_sqr)
LOOM_REGISTER_OP(SQRT, op_sqrt)
LOOM_REGISTER_OP(LOG, op_log)
LOOM_REGISTER_OP(SUM_ROWS, op_sum_rows)
LOOM_REGISTER_OP(PAD_1D, op_pad_1d)
LOOM_REGISTER_OP(SILU, op_silu)
LOOM_REGISTER_OP(RELU, op_relu)
LOOM_REGISTER_OP(LEAKY_RELU, op_leaky_relu)
LOOM_REGISTER_OP(STEP, op_step)
LOOM_REGISTER_OP(CUMSUM, op_cumsum)
LOOM_REGISTER_OP(SOFTMAX, op_softmax)
LOOM_REGISTER_OP(SOFTPLUS, op_softplus)
LOOM_REGISTER_OP(CLAMP, op_clamp)
LOOM_REGISTER_OP(SWIGLU, op_swiglu)
LOOM_REGISTER_OP(RMS_NORM, op_rms_norm)
LOOM_REGISTER_OP(LAYER_NORM, op_layer_norm)
LOOM_REGISTER_OP(SIGMOID, op_sigmoid)
LOOM_REGISTER_OP(TANH, op_tanh)
LOOM_REGISTER_OP(GLU, op_glu)
LOOM_REGISTER_OP(RESHAPE, op_reshape)
LOOM_REGISTER_OP(VIEW, op_view)
LOOM_REGISTER_OP(PERMUTE, op_permute)
LOOM_REGISTER_OP(CONT, op_cont)
LOOM_REGISTER_OP(GELU, op_gelu)

} // namespace loom
