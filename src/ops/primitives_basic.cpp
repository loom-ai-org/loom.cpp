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

Outputs op_atan2(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // atan2(y=in[0], x=in[1]) -- same argument order as std::atan2/torch.atan2 (y first).
    expect_n_inputs("ATAN2", in, 2);
    return {ggml_map_custom2(pc.ctx, in[0], in[1], atan2_custom_op, GGML_N_TASKS_MAX, nullptr)};
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
    ggml_tensor* a = in[0];
    return {ggml_interpolate(pc.ctx, a, ne0, a->ne[1], a->ne[2], a->ne[3], static_cast<uint32_t>(scale_mode))};
}

Outputs op_exp(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // Needed for Kokoro Generator's final spec/phase split (`spec = exp(x[:n_freq])`).
    expect_n_inputs("EXP", in, 1);
    return {ggml_exp(pc.ctx, in[0])};
}

Outputs op_sin(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("SIN", in, 1);
    return {ggml_sin(pc.ctx, in[0])};
}

Outputs op_cos(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("COS", in, 1);
    return {ggml_cos(pc.ctx, in[0])};
}

Outputs op_floor(PrimitiveContext& pc, const Inputs& in, const Json&) {
    // Needed for Kokoro's SineGen (`(f0/sampling_rate) % 1`, expressed as `x - floor(x)` since ggml has
    // no native modulo op and every operand here is non-negative, so this is an exact match to `%`).
    expect_n_inputs("FLOOR", in, 1);
    return {ggml_floor(pc.ctx, in[0])};
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
LOOM_REGISTER_OP(ATAN2, op_atan2)
LOOM_REGISTER_OP(SUM_ROWS, op_sum_rows)
LOOM_REGISTER_OP(PAD_1D, op_pad_1d)
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
LOOM_REGISTER_OP(SIGMOID, op_sigmoid)
LOOM_REGISTER_OP(TANH, op_tanh)
LOOM_REGISTER_OP(EXP, op_exp)
LOOM_REGISTER_OP(SIN, op_sin)
LOOM_REGISTER_OP(COS, op_cos)
LOOM_REGISTER_OP(INTERPOLATE_1D, op_interpolate_1d)
LOOM_REGISTER_OP(FLOOR, op_floor)
LOOM_REGISTER_OP(GLU, op_glu)
LOOM_REGISTER_OP(RESHAPE, op_reshape)
LOOM_REGISTER_OP(VIEW, op_view)
LOOM_REGISTER_OP(PERMUTE, op_permute)
LOOM_REGISTER_OP(CONT, op_cont)
LOOM_REGISTER_OP(GELU, op_gelu)

} // namespace loom
