#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <nlohmann/json.hpp>

#include <limits>

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

// VITS's `ConvFlow` invertible transform (piper's transforms.py: piecewise_rational_quadratic_transform,
// inverse=True, tails="linear" -- confirmed this is the branch actually exercised at TTS inference: the
// StochasticDurationPredictor's reverse-mode flows run noise->data, i.e. the spline's INVERSE direction,
// not the simpler forward polynomial). Real algorithm (Neural Spline Flows, Durkan et al. 2019) verified
// numerically against the real `transforms.py` source directly (a standalone numpy reimplementation of
// this exact translation matched the real PyTorch function to ~1e-6 across both in-domain and
// outside-tail-bound inputs, including exact bin-boundary edge cases) before writing this.
//
// The real algorithm uses boolean-mask indexing (`torch.gather`, `inside_interval_mask` assignment) --
// neither of which has a direct ggml equivalent. Both are avoided here via one general trick: replace
// every "pick element i" operation with a one-hot SELECTION MASK (built from STEP-based comparisons) and
// a masked-sum reduction (MUL + SUM_ROWS) instead -- mathematically identical to a gather/boolean-index
// for a genuinely one-hot mask, and avoids needing any data-dependent control flow or a dynamic
// gather-by-computed-index primitive at all.
//
// Inputs: `inputs` [n_tokens] (values to transform, e.g. ConvFlow's x1 channel), `unnormalized_widths`/
// `unnormalized_heights` [num_bins, n_tokens] (per-position spline-bin logits, from the ConvFlow's own
// conv output), `unnormalized_derivatives` [num_bins-1, n_tokens], `boundary_deriv_const` [num_bins+1]
// (a conversion-time-baked constant: `log(exp(1-min_derivative)-1)` at the two boundary positions, 0
// elsewhere -- the "tails=linear" derivative pinning, added after a zero-pad since ggml's own PAD only
// pads with zero), `eps_bump` [num_bins] (a conversion-time-baked constant: `1e-6` at the last position,
// 0 elsewhere -- mirrors the real `searchsorted`'s own `bin_locations[...,-1] += eps` nudge). Output:
// [n_tokens] (the transformed values; logabsdet is not computed -- not needed at inference, only for
// training-time likelihood).
Outputs op_rq_spline_inverse(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("RQ_SPLINE_INVERSE", in, 6);
    ggml_tensor* inputs = in[0];
    ggml_tensor* uw = in[1];
    ggml_tensor* uh = in[2];
    ggml_tensor* ud = in[3];
    ggml_tensor* boundary_deriv_const = in[4];
    ggml_tensor* eps_bump = in[5];

    const int64_t num_bins = uw->ne[0];
    const int64_t n_tokens = uw->ne[1];
    const float tail_bound = static_cast<float>(resolve_attr_number(attrs, "tail_bound", pc.symbols));
    const float min_bin_width = static_cast<float>(resolve_attr_number(attrs, "min_bin_width", pc.symbols));
    const float min_bin_height = static_cast<float>(resolve_attr_number(attrs, "min_bin_height", pc.symbols));
    const float min_derivative = static_cast<float>(resolve_attr_number(attrs, "min_derivative", pc.symbols));
    const float left = -tail_bound, right = tail_bound, bottom = -tail_bound, top = tail_bound;

    ggml_context* ctx = pc.ctx;

    // widths = min_bin_width + (1 - min_bin_width*num_bins) * softmax(uw)
    ggml_tensor* widths = ggml_scale_bias(ctx, ggml_soft_max(ctx, uw),
                                          1.0f - min_bin_width * static_cast<float>(num_bins), min_bin_width);
    // cumwidths = (right-left)*pad(cumsum(widths), (1,0)) + left  -- the zero-pad's first entry becomes
    // exactly `left` after scale_bias (0*(right-left)+left), matching the real code's explicit
    // cumwidths[...,0]=left assignment automatically; the real code's cumwidths[...,-1]=right safety
    // overwrite is skipped (a sub-epsilon float32 approximation -- searchsorted's own eps-nudge below
    // already handles the boundary case this exists to protect).
    ggml_tensor* cumwidths = ggml_pad_ext(ctx, ggml_cumsum(ctx, widths), 1, 0, 0, 0, 0, 0, 0, 0); // [num_bins+1, T]
    cumwidths = ggml_scale_bias(ctx, cumwidths, right - left, left);
    // widths = cumwidths[1:] - cumwidths[:-1] (recomputed from the safety-adjusted cumwidths, matching
    // the real code) -- via two offset views of the same underlying tensor.
    ggml_tensor* cw_lo_c = ggml_cont(ctx, ggml_view_2d(ctx, cumwidths, num_bins, n_tokens, cumwidths->nb[1], 0));
    ggml_tensor* cw_hi_c = ggml_cont(ctx, ggml_view_2d(ctx, cumwidths, num_bins, n_tokens, cumwidths->nb[1], sizeof(float)));
    widths = ggml_sub(ctx, cw_hi_c, cw_lo_c);

    // derivatives = min_derivative + softplus(pad(ud,(1,1)) + boundary_deriv_const)
    ggml_tensor* ud_padded = ggml_pad_ext(ctx, ud, 1, 1, 0, 0, 0, 0, 0, 0); // [num_bins+1, T]
    ud_padded = ggml_add(ctx, ud_padded, boundary_deriv_const);
    ggml_tensor* derivatives = ggml_scale_bias(ctx, ggml_softplus(ctx, ud_padded), 1.0f, min_derivative);
    ggml_tensor* deriv_lo = ggml_cont(ctx, ggml_view_2d(ctx, derivatives, num_bins, n_tokens, derivatives->nb[1], 0));
    ggml_tensor* deriv_hi = ggml_cont(ctx, ggml_view_2d(ctx, derivatives, num_bins, n_tokens, derivatives->nb[1], sizeof(float)));

    // heights = min_bin_height + (1 - min_bin_height*num_bins) * softmax(uh); cumheights analogous to cumwidths.
    ggml_tensor* heights = ggml_scale_bias(ctx, ggml_soft_max(ctx, uh),
                                           1.0f - min_bin_height * static_cast<float>(num_bins), min_bin_height);
    ggml_tensor* cumheights = ggml_pad_ext(ctx, ggml_cumsum(ctx, heights), 1, 0, 0, 0, 0, 0, 0, 0);
    cumheights = ggml_scale_bias(ctx, cumheights, top - bottom, bottom);
    ggml_tensor* ch_lo = ggml_view_2d(ctx, cumheights, num_bins, n_tokens, cumheights->nb[1], 0);
    ggml_tensor* ch_hi_view = ggml_view_2d(ctx, cumheights, num_bins, n_tokens, cumheights->nb[1], sizeof(float));
    ggml_tensor* ch_lo_c = ggml_cont(ctx, ch_lo);
    ggml_tensor* ch_hi_c = ggml_cont(ctx, ch_hi_view);
    heights = ggml_sub(ctx, ch_hi_c, ch_lo_c);

    ggml_tensor* delta = ggml_div(ctx, heights, widths);

    // Bin-selection mask: sel[i,t] = 1 iff x_clamped[t] is in bin i's HEIGHT range [ch_lo[i,t], ch_hi[i,t])
    // (inverse=True searches cumheights, not cumwidths) -- half-open interval via STEP(hi-x) - STEP(lo-x)
    // (non-strict >= at lo, strict < at hi; STEP itself is strict x>0, confirmed against ggml's own
    // source -- see BACKLOG.md), matching the real searchsorted's `>=`-then-sum-minus-one exactly (proven
    // via the standalone numpy cross-check, not derived on paper alone). `eps_bump` mirrors
    // searchsorted's own `bin_locations[...,-1] += eps` nudge on the last bin's upper edge.
    // ggml_clamp's result is a VIEW aliasing its source's buffer (confirmed in ggml.c: it calls
    // ggml_view_tensor(ctx, a), not ggml_dup_tensor) -- i.e. it clamps in place. Clamping `inputs`
    // directly would corrupt every later read of the ORIGINAL (unclamped) `inputs` tensor in this same
    // graph -- including the outside-tail-bound blend below, which needs the true unclamped value to
    // both classify inside/outside correctly and to pass through unchanged when outside. Clamping a
    // ggml_cont'd copy (a genuine, separately-allocated tensor, per ggml_cont's own ggml_dup_tensor)
    // instead leaves `inputs` itself untouched. Caught by test_rq_spline_inverse_outside_tail_bound
    // (the first test to actually exercise an out-of-domain input -- without this fix, an outside input
    // silently comes back as its clamped boundary value instead of passing through unchanged).
    ggml_tensor* x_clamped = ggml_clamp(ctx, ggml_cont(ctx, inputs), bottom, top);
    ggml_tensor* x_row = ggml_reshape_2d(ctx, x_clamped, 1, n_tokens); // broadcasts against [num_bins,T]
    ggml_tensor* hi_bumped = ggml_add(ctx, ch_hi_c, eps_bump);
    ggml_tensor* sel = ggml_sub(ctx, ggml_step(ctx, ggml_sub(ctx, hi_bumped, x_row)),
                                ggml_step(ctx, ggml_sub(ctx, ch_lo_c, x_row)));

    auto gather = [&](ggml_tensor* table) {
        return ggml_sum_rows(ctx, ggml_mul(ctx, sel, table)); // [1, n_tokens]
    };
    ggml_tensor* input_cumwidths = ggml_reshape_1d(ctx, gather(cw_lo_c), n_tokens);
    ggml_tensor* input_bin_widths = ggml_reshape_1d(ctx, gather(widths), n_tokens);
    ggml_tensor* input_cumheights = ggml_reshape_1d(ctx, gather(ch_lo_c), n_tokens);
    ggml_tensor* input_delta = ggml_reshape_1d(ctx, gather(delta), n_tokens);
    ggml_tensor* input_derivatives = ggml_reshape_1d(ctx, gather(deriv_lo), n_tokens);
    ggml_tensor* input_derivatives_plus_one = ggml_reshape_1d(ctx, gather(deriv_hi), n_tokens);
    ggml_tensor* input_heights = ggml_reshape_1d(ctx, gather(heights), n_tokens);

    // Inverse rational-quadratic formula (Durkan et al. 2019 eq. 4-6, solved for the input given the
    // output): a*root^2 + b*root + c = 0, root = 2c / (-b - sqrt(b^2-4ac)).
    ggml_tensor* y_minus_ch = ggml_sub(ctx, x_clamped, input_cumheights); // "x_clamped" plays y's role (inverse)
    ggml_tensor* deriv_sum_term = ggml_sub(ctx, ggml_add(ctx, input_derivatives, input_derivatives_plus_one),
                                           ggml_scale(ctx, input_delta, 2.0f));
    ggml_tensor* a = ggml_add(ctx, ggml_mul(ctx, y_minus_ch, deriv_sum_term),
                              ggml_mul(ctx, input_heights, ggml_sub(ctx, input_delta, input_derivatives)));
    ggml_tensor* b = ggml_sub(ctx, ggml_mul(ctx, input_heights, input_derivatives),
                              ggml_mul(ctx, y_minus_ch, deriv_sum_term));
    ggml_tensor* c = ggml_scale(ctx, ggml_mul(ctx, input_delta, y_minus_ch), -1.0f);

    ggml_tensor* discriminant = ggml_sub(ctx, ggml_sqr(ctx, b), ggml_scale(ctx, ggml_mul(ctx, a, c), 4.0f));
    discriminant = ggml_clamp(ctx, discriminant, 0.0f, std::numeric_limits<float>::max());
    ggml_tensor* root = ggml_div(ctx, ggml_scale(ctx, c, 2.0f),
                                 ggml_sub(ctx, ggml_scale(ctx, b, -1.0f), ggml_sqrt(ctx, discriminant)));
    ggml_tensor* spline_out = ggml_add(ctx, ggml_mul(ctx, root, input_bin_widths), input_cumwidths);

    // Blend with identity outside [left,right] (linear tails): inside_mask*spline_out +
    // (1-inside_mask)*inputs, from the ORIGINAL (unclamped) inputs -- same STEP-based ">="/"<=" pattern
    // as the bin-selection mask above (1-STEP(a-b) gives "b>=a"), via scale_bias for the "1-x" step.
    ggml_tensor* ge_left = ggml_scale_bias(ctx, ggml_step(ctx, ggml_scale_bias(ctx, inputs, -1.0f, left)), -1.0f, 1.0f);
    ggml_tensor* le_right = ggml_scale_bias(ctx, ggml_step(ctx, ggml_scale_bias(ctx, inputs, 1.0f, -right)), -1.0f, 1.0f);
    ggml_tensor* inside_mask = ggml_mul(ctx, ge_left, le_right);
    ggml_tensor* outside_mask = ggml_scale_bias(ctx, inside_mask, -1.0f, 1.0f);
    return {ggml_add(ctx, ggml_mul(ctx, inside_mask, spline_out), ggml_mul(ctx, outside_mask, inputs))};
}

LOOM_REGISTER_OP(RQ_SPLINE_INVERSE, op_rq_spline_inverse)

} // namespace
} // namespace loom
