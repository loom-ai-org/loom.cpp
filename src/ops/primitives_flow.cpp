#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <nlohmann/json.hpp>

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

// Same recipe as CONV_1D (src/ops/primitives_conv.cpp) -- duplicated locally (matching this codebase's
// existing convention of each primitives_*.cpp keeping its own small helpers, e.g. expect_n_inputs
// above) rather than reaching across translation units, since WN/ResidualCouplingLayer below need to
// call this several times per node while looping in C++ over a statically-known layer count.
ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* data, int s0, int p0, int d0) {
    ggml_tensor* im2col = ggml_im2col(ctx, kernel, data, s0, 0, p0, 0, d0, 0, /*is_2D=*/false, GGML_TYPE_F32);
    ggml_tensor* result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    return result;
}

ggml_tensor* add_bias_3d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) {
    // x: [OL, OC, N], bias: [OC] -- broadcast add via a [1, OC, 1] reshape.
    return ggml_add(ctx, x, ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1));
}

ggml_tensor* channel_slice(ggml_context* ctx, ggml_tensor* x, int64_t c0, int64_t count) {
    // x: [T, C, N] -> [T, count, N], channels [c0, c0+count).
    ggml_tensor* view = ggml_view_3d(ctx, x, x->ne[0], count, x->ne[2], x->nb[1], x->nb[2],
                                      c0 * static_cast<int64_t>(x->nb[1]));
    return ggml_cont(ctx, view);
}

ggml_tensor* row_slice(ggml_context* ctx, ggml_tensor* x, int64_t r0, int64_t count) {
    // x: [R, T] -> [count, T], rows [r0, r0+count) along ne[0] (same "offset view" trick as
    // RQ_SPLINE_INVERSE's cw_lo/cw_hi bin-edge views in primitives_spline.cpp).
    ggml_tensor* view = ggml_view_2d(ctx, x, count, x->ne[1], x->nb[1], r0 * static_cast<int64_t>(x->nb[0]));
    return ggml_cont(ctx, view);
}

// VITS's `modules.LayerNorm`: normalizes over the CHANNEL axis (real code: transpose(1,-1) -> F.layer_norm
// over the new last dim -> transpose back), which in this engine's [T, C] flow-tensor convention (channel
// = ne[1], NOT ne[0]) means transposing so channels become ne[0] (ggml_norm's own normalization axis),
// applying the learned affine, then transposing back -- a direct ggml-order translation of the real
// module's own transpose-normalize-transpose trick, not an independent reformulation.
ggml_tensor* layer_norm_channels(ggml_context* ctx, ggml_tensor* x2d, ggml_tensor* gamma, ggml_tensor* beta, float eps) {
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x2d)); // [C, T]
    ggml_tensor* normed = ggml_norm(ctx, xt, eps);
    normed = ggml_add(ctx, ggml_mul(ctx, normed, gamma), beta); // gamma/beta [C] broadcast against [C, T]
    return ggml_cont(ctx, ggml_transpose(ctx, normed)); // [T, C]
}

// VITS's `WN` (modules.py): a WaveNet-style stack of dilated gated conv1d layers, real checkpoint
// confirmed single-speaker (gin_channels=0, so no conditioning input `g` -- fused_add_tanh_sigmoid_
// multiply's `input_b` term is always the zero tensor here, which collapses it to a plain
// tanh(x_in_first_half)*sigmoid(x_in_second_half) gate) and confirmed no padding mask needed (this
// engine always runs a single, unpadded utterance -- `x_mask` in the real code is all-ones here, so
// every `* x_mask` in the real forward is a no-op and is omitted).
//
// Inputs: x [T, hidden_channels], then per layer i in [0, n_layers): in_weight_i [K, hidden_channels,
// 2*hidden_channels], in_bias_i [2*hidden_channels], res_skip_weight_i [1, hidden_channels,
// res_skip_channels_i], res_skip_bias_i [res_skip_channels_i] (res_skip_channels_i is 2*hidden_channels
// for i<n_layers-1, hidden_channels for the last layer, matching the real module's own "last one is not
// necessary" skip-only optimization). Attrs: kernel_size, dilation_rate, n_layers. Output: [T,
// hidden_channels].
Outputs op_wn(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    const int64_t n_layers = resolve_attr_int(attrs, "n_layers", pc.symbols);
    expect_n_inputs("WN", in, static_cast<size_t>(1 + 4 * n_layers));
    const int kernel_size = static_cast<int>(resolve_attr_int(attrs, "kernel_size", pc.symbols));
    const int dilation_rate = static_cast<int>(resolve_attr_int(attrs, "dilation_rate", pc.symbols));

    ggml_context* ctx = pc.ctx;
    ggml_tensor* x = in[0]; // [T, hidden_channels]
    const int64_t T = x->ne[0];
    const int64_t hidden_channels = x->ne[1];

    ggml_tensor* x_cur = ggml_reshape_3d(ctx, x, T, hidden_channels, 1);
    ggml_tensor* output = nullptr;

    for (int64_t i = 0; i < n_layers; ++i) {
        ggml_tensor* in_w = in[1 + 4 * i];
        ggml_tensor* in_b = in[2 + 4 * i];
        ggml_tensor* rs_w = in[3 + 4 * i];
        ggml_tensor* rs_b = in[4 + 4 * i];

        int dilation = 1;
        for (int64_t k = 0; k < i; ++k) dilation *= dilation_rate;
        const int padding = (kernel_size * dilation - dilation) / 2;

        ggml_tensor* x_in = conv1d(ctx, in_w, x_cur, /*s0=*/1, padding, dilation); // [T, 2*hidden, 1]
        x_in = add_bias_3d(ctx, x_in, in_b);

        ggml_tensor* t_in = channel_slice(ctx, x_in, 0, hidden_channels);
        ggml_tensor* s_in = channel_slice(ctx, x_in, hidden_channels, hidden_channels);
        ggml_tensor* acts = ggml_mul(ctx, ggml_tanh(ctx, t_in), ggml_sigmoid(ctx, s_in)); // [T, hidden, 1]

        ggml_tensor* res_skip = conv1d(ctx, rs_w, acts, /*s0=*/1, /*p0=*/0, /*d0=*/1);
        res_skip = add_bias_3d(ctx, res_skip, rs_b);

        if (i < n_layers - 1) {
            ggml_tensor* res = channel_slice(ctx, res_skip, 0, hidden_channels);
            ggml_tensor* skip = channel_slice(ctx, res_skip, hidden_channels, hidden_channels);
            x_cur = ggml_add(ctx, x_cur, res);
            output = output ? ggml_add(ctx, output, skip) : skip;
        } else {
            output = output ? ggml_add(ctx, output, res_skip) : res_skip;
        }
    }

    return {ggml_reshape_2d(ctx, output, T, hidden_channels)};
}

// VITS's `ResidualCouplingLayer` (modules.py), REVERSE mode only, specialized to `mean_only=True` --
// confirmed via the real checkpoint's state dict that every coupling layer in this model's flow is
// built with mean_only=True (SynthesizerTrn's `flow = ResidualCouplingBlock(...)` -> each
// ResidualCouplingLayer defaults mean_only per ResidualCouplingBlock.__init__, which always passes
// mean_only=True), so `logs` is always the zero tensor and `exp(-logs)=1`: the affine reverse transform
// `x1' = (x1-m)*exp(-logs)` collapses to a plain subtraction `x1' = x1-m`. `x0` passes through
// unmodified. No padding mask needed, same reasoning as WN above.
//
// Inputs: x [T, channels] (channels=2*half_channels), pre_weight [1, half_channels, hidden_channels],
// pre_bias [hidden_channels], then WN's own per-layer inputs (see op_wn), then post_weight [1,
// hidden_channels, half_channels], post_bias [half_channels]. Attrs: kernel_size, dilation_rate,
// n_layers (WN's internal layer count). Output: [T, channels].
Outputs op_residual_coupling_layer_reverse(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    const int64_t n_layers = resolve_attr_int(attrs, "n_layers", pc.symbols);
    expect_n_inputs("RESIDUAL_COUPLING_LAYER_REVERSE", in, static_cast<size_t>(5 + 4 * n_layers));
    const int kernel_size = static_cast<int>(resolve_attr_int(attrs, "kernel_size", pc.symbols));
    const int dilation_rate = static_cast<int>(resolve_attr_int(attrs, "dilation_rate", pc.symbols));

    ggml_context* ctx = pc.ctx;
    ggml_tensor* x = in[0]; // [T, channels]
    ggml_tensor* pre_w = in[1];
    ggml_tensor* pre_b = in[2];
    ggml_tensor* post_w = in[3 + 4 * n_layers];
    ggml_tensor* post_b = in[4 + 4 * n_layers];

    const int64_t T = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t half_channels = channels / 2;
    const int64_t hidden_channels = pre_w->ne[2];

    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, T, channels, 1);
    ggml_tensor* x0 = channel_slice(ctx, x3, 0, half_channels);
    ggml_tensor* x1 = channel_slice(ctx, x3, half_channels, half_channels);

    ggml_tensor* h = conv1d(ctx, pre_w, x0, /*s0=*/1, /*p0=*/0, /*d0=*/1);
    h = add_bias_3d(ctx, h, pre_b); // [T, hidden_channels, 1]
    h = ggml_reshape_2d(ctx, h, T, hidden_channels);

    Inputs wn_inputs;
    wn_inputs.reserve(1 + 4 * n_layers);
    wn_inputs.push_back(h);
    for (int64_t i = 0; i < 4 * n_layers; ++i) wn_inputs.push_back(in[3 + i]);
    Json wn_attrs = {{"kernel_size", kernel_size}, {"dilation_rate", dilation_rate}, {"n_layers", n_layers}};
    ggml_tensor* wn_out = op_wn(pc, wn_inputs, wn_attrs)[0]; // [T, hidden_channels]

    ggml_tensor* m = conv1d(ctx, post_w, ggml_reshape_3d(ctx, wn_out, T, hidden_channels, 1), /*s0=*/1, /*p0=*/0, /*d0=*/1);
    m = add_bias_3d(ctx, m, post_b); // [T, half_channels, 1]
    m = ggml_reshape_2d(ctx, m, T, half_channels);

    ggml_tensor* x1_out = ggml_sub(ctx, ggml_reshape_2d(ctx, x1, T, half_channels), m);
    ggml_tensor* x0_out = ggml_reshape_2d(ctx, x0, T, half_channels);
    return {ggml_concat(ctx, x0_out, x1_out, 1)};
}

// VITS's `ElementwiseAffine` (modules.py), REVERSE mode only: `x' = (x-m)*exp(-logs)` per channel, `m`/
// `logs` learned [channels]-shaped parameters (no mask needed, same reasoning as WN/DDSConv above). Used
// as the first entry of `StochasticDurationPredictor`'s flow list (`channels=2`, matching `z`'s shape).
// Inputs: x [T, channels], m [channels], logs [channels]. Output: [T, channels].
Outputs op_elementwise_affine_reverse(PrimitiveContext& pc, const Inputs& in, const Json&) {
    expect_n_inputs("ELEMENTWISE_AFFINE_REVERSE", in, 3);
    ggml_context* ctx = pc.ctx;
    ggml_tensor* x = in[0];    // [T, C]
    ggml_tensor* m = in[1];    // [C]
    ggml_tensor* logs = in[2]; // [C]
    ggml_tensor* m2d = ggml_reshape_2d(ctx, m, 1, m->ne[0]);       // [1, C] -- broadcasts against [T, C]
    ggml_tensor* logs2d = ggml_reshape_2d(ctx, logs, 1, logs->ne[0]);
    ggml_tensor* diff = ggml_sub(ctx, x, m2d);
    ggml_tensor* inv_scale = ggml_exp(ctx, ggml_scale(ctx, logs2d, -1.0f));
    return {ggml_mul(ctx, diff, inv_scale)};
}

// VITS's `DDSConv` (Dilated and Depth-Separable Convolution, modules.py) -- a residual stack of
// depthwise-separable, GELU-activated conv blocks: each layer is a depthwise conv (dilation=
// kernel_size**i, "same" padding) -> channel LayerNorm -> GELU -> pointwise (1x1) conv -> channel
// LayerNorm -> GELU -> residual add. No mask/conditioning needed, same reasoning
// as WN above (`x_mask` is all-ones for a single unpadded utterance; DDSConv's own `g` conditioning
// input is only used by the TextEncoder's DDSConv call, never by ConvFlow's, which always passes
// g=None -- confirmed via `StochasticDurationPredictor`'s real `reverse=True` code path below).
//
// Inputs: x [T, channels], then per layer i in [0, n_layers): sep_weight_i [K, 1, channels], sep_bias_i
// [channels] (depthwise conv), ln1_gamma_i/ln1_beta_i [channels], oneone_weight_i [1, channels,
// channels], oneone_bias_i [channels] (pointwise conv), ln2_gamma_i/ln2_beta_i [channels]. Attrs:
// kernel_size, n_layers, eps (LayerNorm epsilon). Output: [T, channels].
Outputs op_dds_conv(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    const int64_t n_layers = resolve_attr_int(attrs, "n_layers", pc.symbols);
    expect_n_inputs("DDS_CONV", in, static_cast<size_t>(1 + 8 * n_layers));
    const int kernel_size = static_cast<int>(resolve_attr_int(attrs, "kernel_size", pc.symbols));
    const float eps = static_cast<float>(resolve_attr_number(attrs, "eps", pc.symbols));

    ggml_context* ctx = pc.ctx;
    ggml_tensor* x = in[0]; // [T, channels]
    const int64_t T = x->ne[0];
    const int64_t channels = x->ne[1];
    const PrimitiveFn& conv_1d_dw = PrimitiveRegistry::instance().get("CONV_1D_DW");

    for (int64_t i = 0; i < n_layers; ++i) {
        ggml_tensor* sep_w = in[1 + 8 * i];
        ggml_tensor* sep_b = in[2 + 8 * i];
        ggml_tensor* ln1_g = in[3 + 8 * i];
        ggml_tensor* ln1_b = in[4 + 8 * i];
        ggml_tensor* oo_w = in[5 + 8 * i];
        ggml_tensor* oo_b = in[6 + 8 * i];
        ggml_tensor* ln2_g = in[7 + 8 * i];
        ggml_tensor* ln2_b = in[8 + 8 * i];

        int dilation = 1;
        for (int64_t k = 0; k < i; ++k) dilation *= kernel_size;
        const int padding = (kernel_size * dilation - dilation) / 2;

        ggml_tensor* x3 = ggml_reshape_3d(ctx, x, T, channels, 1);
        Json dw_attrs = {{"s0", 1}, {"p0", padding}, {"d0", dilation}};
        ggml_tensor* y = conv_1d_dw(pc, {sep_w, x3}, dw_attrs)[0]; // [T, channels, 1]
        y = add_bias_3d(ctx, y, sep_b);
        ggml_tensor* y2 = layer_norm_channels(ctx, ggml_reshape_2d(ctx, y, T, channels), ln1_g, ln1_b, eps);
        y2 = ggml_gelu_erf(ctx, y2);

        ggml_tensor* oneone = conv1d(ctx, oo_w, ggml_reshape_3d(ctx, y2, T, channels, 1), 1, 0, 1);
        oneone = add_bias_3d(ctx, oneone, oo_b);
        ggml_tensor* y4 = layer_norm_channels(ctx, ggml_reshape_2d(ctx, oneone, T, channels), ln2_g, ln2_b, eps);
        y4 = ggml_gelu_erf(ctx, y4);

        x = ggml_add(ctx, x, y4);
    }
    return {x};
}

// VITS's `ConvFlow` (modules.py), REVERSE mode only, specialized to `half_channels=1` -- the ONLY
// configuration this checkpoint ever instantiates (`StochasticDurationPredictor` is the sole caller of
// `ConvFlow`, always with `in_channels=2`, confirmed in `models.py`), which collapses the real code's
// `h.reshape(b, half_channels, -1, t).permute(0,1,3,2)` bin-parameter reshape down to a plain transpose
// (no actual per-channel splitting needed when half_channels=1). Delegates the actual spline math to the
// already-verified `RQ_SPLINE_INVERSE` primitive via the registry (kept in its own file/primitive rather
// than duplicated here, same separation-of-concerns reasoning as reusing `CONV_1D_DW` above).
//
// `g` (last input): `StochasticDurationPredictor.forward`'s reverse branch always calls its flows with
// `g=x` (the SDP's own pre/convs/proj-processed conditioning features -- confirmed in `models.py`, the
// ONLY real caller of `ConvFlow`), which `ConvFlow.forward` threads straight into its *internal* DDSConv
// call (`self.convs(h, x_mask, g=g)`), which in turn adds it once to its own hidden state before its
// layer loop (`if g is not None: x = x+g`, see `DDSConv.forward`) -- done here as a plain add right after
// `pre`, before calling `op_dds_conv` (unchanged/unaware of conditioning), rather than threading a
// conditional `g` parameter through `op_dds_conv` itself.
//
// Inputs: x [T, 2], pre_weight [1, 1, filter_channels], pre_bias [filter_channels], then DDSConv's own
// per-layer inputs (see op_dds_conv, with channels=filter_channels), then proj_weight [1,
// filter_channels, 3*num_bins-1], proj_bias [3*num_bins-1], boundary_deriv_const [num_bins+1], eps_bump
// [num_bins] (RQ_SPLINE_INVERSE's own baked constants -- see primitives_spline.cpp), g [T,
// filter_channels] (conditioning features, added once before DDSConv). Attrs: kernel_size, n_layers
// (DDSConv's), num_bins, tail_bound, ln_eps (DDSConv's internal LayerNorm epsilon). Output: [T, 2].
Outputs op_conv_flow_reverse(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    const int64_t n_layers = resolve_attr_int(attrs, "n_layers", pc.symbols);
    const int64_t num_bins = resolve_attr_int(attrs, "num_bins", pc.symbols);
    expect_n_inputs("CONV_FLOW_REVERSE", in, static_cast<size_t>(3 + 8 * n_layers + 5));
    const int kernel_size = static_cast<int>(resolve_attr_int(attrs, "kernel_size", pc.symbols));
    const float tail_bound = static_cast<float>(resolve_attr_number(attrs, "tail_bound", pc.symbols));
    const float ln_eps = static_cast<float>(resolve_attr_number(attrs, "ln_eps", pc.symbols));

    ggml_context* ctx = pc.ctx;
    ggml_tensor* x = in[0]; // [T, 2]
    ggml_tensor* pre_w = in[1];
    ggml_tensor* pre_b = in[2];
    ggml_tensor* proj_w = in[3 + 8 * n_layers];
    ggml_tensor* proj_b = in[4 + 8 * n_layers];
    ggml_tensor* boundary_deriv_const = in[5 + 8 * n_layers];
    ggml_tensor* eps_bump = in[6 + 8 * n_layers];
    ggml_tensor* g = in[7 + 8 * n_layers];

    const int64_t T = x->ne[0];
    const int64_t filter_channels = pre_w->ne[2];

    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, T, 2, 1);
    ggml_tensor* x0 = channel_slice(ctx, x3, 0, 1); // [T, 1, 1]
    ggml_tensor* x1 = channel_slice(ctx, x3, 1, 1); // [T, 1, 1]

    ggml_tensor* h = conv1d(ctx, pre_w, x0, /*s0=*/1, /*p0=*/0, /*d0=*/1);
    h = add_bias_3d(ctx, h, pre_b);
    h = ggml_reshape_2d(ctx, h, T, filter_channels);
    h = ggml_add(ctx, h, g);

    Inputs dds_inputs;
    dds_inputs.reserve(1 + 8 * n_layers);
    dds_inputs.push_back(h);
    for (int64_t i = 0; i < 8 * n_layers; ++i) dds_inputs.push_back(in[3 + i]);
    Json dds_attrs = {{"kernel_size", kernel_size}, {"n_layers", n_layers}, {"eps", ln_eps}};
    ggml_tensor* dds_out = op_dds_conv(pc, dds_inputs, dds_attrs)[0]; // [T, filter_channels]

    ggml_tensor* proj_out = conv1d(ctx, proj_w, ggml_reshape_3d(ctx, dds_out, T, filter_channels, 1), 1, 0, 1);
    proj_out = add_bias_3d(ctx, proj_out, proj_b); // [T, 3*num_bins-1, 1]
    ggml_tensor* proj_t = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, proj_out, T, 3 * num_bins - 1))); // [3*num_bins-1, T]

    const float scale = 1.0f / std::sqrt(static_cast<float>(filter_channels));
    ggml_tensor* uw = ggml_scale(ctx, row_slice(ctx, proj_t, 0, num_bins), scale);
    ggml_tensor* uh = ggml_scale(ctx, row_slice(ctx, proj_t, num_bins, num_bins), scale);
    ggml_tensor* ud = row_slice(ctx, proj_t, 2 * num_bins, num_bins - 1);

    Json rq_attrs = {{"tail_bound", tail_bound}, {"min_bin_width", 1e-3}, {"min_bin_height", 1e-3}, {"min_derivative", 1e-3}};
    ggml_tensor* x1_out = PrimitiveRegistry::instance().get("RQ_SPLINE_INVERSE")(
        pc, {ggml_reshape_1d(ctx, x1, T), uw, uh, ud, boundary_deriv_const, eps_bump}, rq_attrs)[0];

    ggml_tensor* x0_out = ggml_reshape_2d(ctx, x0, T, 1);
    return {ggml_concat(ctx, x0_out, ggml_reshape_2d(ctx, x1_out, T, 1), 1)};
}

} // namespace

LOOM_REGISTER_OP(WN, op_wn)
LOOM_REGISTER_OP(RESIDUAL_COUPLING_LAYER_REVERSE, op_residual_coupling_layer_reverse)
LOOM_REGISTER_OP(DDS_CONV, op_dds_conv)
LOOM_REGISTER_OP(CONV_FLOW_REVERSE, op_conv_flow_reverse)
LOOM_REGISTER_OP(ELEMENTWISE_AFFINE_REVERSE, op_elementwise_affine_reverse)

} // namespace loom
