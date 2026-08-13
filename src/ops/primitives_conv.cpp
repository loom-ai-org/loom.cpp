#include "loom/core/conv_state_cache.h"
#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <nlohmann/json.hpp>

#include <cmath>

namespace loom {

// See op_pool_1d below for what this decides and the measurement behind it. Out of the anonymous
// namespace and declared in primitive_registry.h on purpose: tests/ci/test_pool_1d_lowering.cpp asserts
// that this predicate agrees with the two spellings' actual bit-level behaviour, which it cannot do if
// the predicate is private -- and a test that could not see it would still pass with the guard deleted.
bool pool_1d_lowers_to_pool_2d(ggml_op_pool op, int p0) {
    return op == GGML_OP_POOL_MAX || p0 == 0;
}

namespace {

using Json = nlohmann::json;
using Inputs = std::vector<ggml_tensor*>;
using Outputs = std::vector<ggml_tensor*>;

void expect_n_inputs(const char* op, const Inputs& in, size_t n) {
    if (in.size() != n) {
        throw SchemaError(std::string(op) + " expects " + std::to_string(n) + " input(s), got " + std::to_string(in.size()));
    }
}

// ggml_conv_1d/ggml_conv_2d force their internal im2col through F16 (unless the kernel is BF16),
// even for plain F32 inputs -- fine for inference speed, but it fights this engine's Milestone-1
// precedent of exact fp32 verification against a numpy reference (the same reason ATTENTION uses the
// composite path over ggml_flash_attn_ext). ggml_compute_forward_im2col fully supports GGML_TYPE_F32 on
// CPU, so these primitives replicate ggml_conv_1d/2d's own im2col->reshape->mul_mat(->permute->cont)
// recipe with an F32 im2col instead of calling the convenience wrappers directly.

Outputs op_conv_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_1D", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    const int d0 = static_cast<int>(resolve_attr_int(attrs, "d0", pc.symbols));

    // ggml_compute_forward_im2col asserts its `data` operand's fastest-varying axis is densely packed
    // (nb[0] == sizeof(float)) -- true for most producers, but not for a channel-split VIEW feeding
    // straight into a pointwise/depthwise conv (confirmed on Conformer-CTC's GLU-split conv module: the
    // first half of a channel-split tensor is a genuinely strided view, not a fresh contiguous buffer).
    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);
    ggml_tensor* im2col = ggml_im2col(pc.ctx, kernel, data, s0, 0, p0, 0, d0, 0, /*is_2D=*/false, GGML_TYPE_F32); // [IC*K, OL, N]
    ggml_tensor* result = ggml_mul_mat(pc.ctx,
        ggml_reshape_2d(pc.ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),       // [IC*K, N*OL]
        ggml_reshape_2d(pc.ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));       // [IC*K, OC]
    result = ggml_reshape_3d(pc.ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);    // [OL, OC, N]
    return {result};
}

Outputs op_conv_2d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_2D", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int s1 = static_cast<int>(resolve_attr_int(attrs, "s1", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    const int p1 = static_cast<int>(resolve_attr_int(attrs, "p1", pc.symbols));
    const int d0 = static_cast<int>(resolve_attr_int(attrs, "d0", pc.symbols));
    const int d1 = static_cast<int>(resolve_attr_int(attrs, "d1", pc.symbols));

    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);
    ggml_tensor* im2col = ggml_im2col(pc.ctx, kernel, data, s0, s1, p0, p1, d0, d1, /*is_2D=*/true, GGML_TYPE_F32); // [IC*KH*KW, OW, OH, N]
    ggml_tensor* result = ggml_mul_mat(pc.ctx,
        ggml_reshape_2d(pc.ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),          // [IC*KH*KW, N*OH*OW]
        ggml_reshape_2d(pc.ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]));         // [IC*KH*KW, OC]
    result = ggml_reshape_4d(pc.ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], kernel->ne[3]);       // [OW, OH, N, OC]
    result = ggml_cont(pc.ctx, ggml_permute(pc.ctx, result, 0, 1, 3, 2));                                        // [OW, OH, OC, N]
    return {result};
}

// Depthwise conv2d (one filter per channel, groups == channels) -- needed for FastConformer's real
// dw_striding subsampling (subsampling stages 2+ use a grouped Conv2d, unlike Conformer-CTC-small's
// plain/ungrouped 2-stage subsampling). ggml's own convenience wrapper (ggml_conv_2d_dw, ggml.c) hardcodes
// GGML_TYPE_F16 for its internal im2col regardless of the kernel's actual dtype (confirmed by reading its
// source directly -- unlike ggml_conv_2d, which already respects the kernel's own type) -- same precision
// concern op_conv_1d/op_conv_2d above were already written to avoid, so this is a faithful transcription of
// ggml_conv_2d_dw's exact recipe with F32 substituted for that one hardcoded F16, not an independent
// re-derivation.
Outputs op_conv_2d_dw(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_2D_DW", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int s1 = static_cast<int>(resolve_attr_int(attrs, "s1", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    const int p1 = static_cast<int>(resolve_attr_int(attrs, "p1", pc.symbols));
    const int d0 = static_cast<int>(resolve_attr_int(attrs, "d0", pc.symbols));
    const int d1 = static_cast<int>(resolve_attr_int(attrs, "d1", pc.symbols));

    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);
    ggml_tensor* new_kernel = ggml_reshape_4d(pc.ctx, kernel, kernel->ne[0], kernel->ne[1], 1,
                                               kernel->ne[2] * kernel->ne[3]);
    ggml_tensor* im2col = ggml_im2col(pc.ctx, new_kernel,
        ggml_reshape_4d(pc.ctx, data, data->ne[0], data->ne[1], 1, data->ne[2] * data->ne[3]),
        s0, s1, p0, p1, d0, d1, /*is_2D=*/true, GGML_TYPE_F32);
    ggml_tensor* new_data = ggml_reshape_4d(pc.ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1],
                                             data->ne[2], data->ne[3]);

    new_kernel = ggml_reshape_4d(pc.ctx, new_kernel, new_kernel->ne[0] * new_kernel->ne[1], new_kernel->ne[2],
                                  new_kernel->ne[3], 1);
    ggml_tensor* result = ggml_mul_mat(pc.ctx, new_kernel, new_data);
    result = ggml_reshape_4d(pc.ctx, result, im2col->ne[1], im2col->ne[2], data->ne[2], data->ne[3]);
    return {result};
}

// Unlike ggml_conv_1d/2d, ggml_conv_transpose_1d/2d_p0 are native ops that dispatch purely on the
// kernel's own dtype (ggml_compute_forward_conv_transpose_1d/2d: F32 kernel -> full F32 compute, no
// forced F16 cast) -- so these call the ggml convenience functions directly, no precision workaround
// needed. Both force padding=0 (ggml asserts it); ggml_conv_transpose_1d additionally forces
// dilation=1 and requires a batch-less 2D `data` tensor (ne=[IL,IC]) -- neither is read as an attr here
// since ggml doesn't accept any other value.

Outputs op_conv_transpose_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_TRANSPOSE_1D", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    return {ggml_conv_transpose_1d(pc.ctx, kernel, data, s0, /*p0=*/0, /*d0=*/1)};
}

Outputs op_conv_transpose_2d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_TRANSPOSE_2D", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int stride = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    return {ggml_conv_transpose_2d_p0(pc.ctx, kernel, data, stride)};
}

// Depthwise conv1d (one filter per channel, groups == channels): kernel ne=[K,1,channels], data
// ne=[IL,channels,N]. Mirrors ggml_conv_1d_dw's own recipe (reshape data to insert a dummy dim so
// im2col treats each channel as an independent "batch" slice, then a batched mul_mat pairs each
// channel's im2col slice with that SAME channel's kernel slice via ggml_mul_mat's per-index batching
// over ne[2], not a cross product) -- but with an F32 im2col instead of ggml_conv_1d_dw's forced F16
// cast, same rationale as CONV_1D/CONV_2D above. ggml's own header flags ggml_conv_1d_dw as "very
// likely wrong for some cases, needs more testing"; this reimplementation is verified independently via
// a hand-computed case in tests/test_primitive_registry.cpp rather than trusted on the header comment
// (or lack thereof) alone.
Outputs op_conv_1d_dw(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_1D_DW", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    const int d0 = static_cast<int>(resolve_attr_int(attrs, "d0", pc.symbols));

    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);
    ggml_tensor* data_4d = ggml_reshape_4d(pc.ctx, data, data->ne[0], 1, data->ne[1], data->ne[2]);
    ggml_tensor* im2col = ggml_im2col(pc.ctx, kernel, data_4d, s0, 0, p0, 0, d0, 0, /*is_2D=*/false, GGML_TYPE_F32);
    ggml_tensor* result = ggml_mul_mat(pc.ctx, im2col, kernel);
    result = ggml_reshape_3d(pc.ctx, result, result->ne[0], result->ne[2], 1);
    return {result};
}

// Causal depthwise conv1d that owns its cross-step history -- the conv family's answer to ATTENTION,
// and what makes a hybrid architecture (LFM2's ShortConv blocks, and Mamba/RWKV after it) able to
// decode a token at a time (BACKLOG.md P4.0.10).
//
// The problem it solves: a causal conv over kernel K needs the K-1 input columns BEFORE this step's
// first token. A prefill has them (they are in the same call); a decode step at n_tokens = 1 does not,
// and the KV cache holds K/V and nothing else. So this op keeps them, in a ConvStateCache slot
// addressed by the `layer` attr, exactly as op_attention keeps K/V.
//
// One uniform path, no prefill special case: history is either the layer's slot or the permanently-zero
// slot when n_past == 0, and everything after the concat is identical. That "n_past == 0 means no
// history" rule is load-bearing beyond tidiness -- it is what keeps iterated `infer` (which always
// calls at n_past = 0) a valid oracle for `infer_with_past`, and what makes a prefill issued after a
// generation correct without the host resetting anything (KV-CACHE.md 3.4).
//
// Ordering is free here, unlike ATTENTION's. The state write copies a view of the CONCATENATED buffer,
// which reads the slot -- so there is a real data-dependency edge from read to write and ggml cannot
// schedule the clobber first. It still goes through side_effects, because nothing downstream of the
// declared output references it.
Outputs op_short_conv(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("SHORT_CONV", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];

    const int64_t kernel_size = kernel->ne[0];
    const int64_t n_state = kernel_size - 1;
    if (n_state <= 0) {
        throw SchemaError("SHORT_CONV: kernel width is " + std::to_string(kernel_size) +
                          ", so the convolution is position-wise and carries no history -- it should "
                          "have been emitted as CONV_1D_DW, not SHORT_CONV");
    }

    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);

    const bool use_state = attrs.value("conv_state", true);
    ggml_tensor* conv_in = data;
    if (use_state) {
        if (!pc.conv_state) {
            throw SchemaError("SHORT_CONV: no ConvStateCache was provided to GraphBuilder, but the "
                              "topology uses SHORT_CONV with conv_state=true");
        }
        const uint32_t layer = static_cast<uint32_t>(resolve_attr_int(attrs, "layer", pc.symbols));
        const auto n_past = static_cast<uint32_t>(std::llround(pc.symbols.get("n_past")));

        if (n_state != pc.conv_state->n_state()) {
            throw SchemaError("SHORT_CONV: kernel width implies a history of " + std::to_string(n_state) +
                              " but the ConvStateCache was allocated for " +
                              std::to_string(pc.conv_state->n_state()) +
                              " -- 'loom.n_conv_state' must equal kernel-1");
        }

        ggml_tensor* history = (n_past == 0) ? pc.conv_state->read_zeros(pc.ctx)
                                             : pc.conv_state->read(pc.ctx, layer);
        conv_in = ggml_concat(pc.ctx, history, data, /*dim=*/0); // [n_state + n_tokens, channels]

        // The next step's history is the last n_state columns of what this step convolved over -- which
        // is why the concat, not `data`, is the source: a prompt shorter than n_state still leaves a
        // full window, zero-padded on the left, with no length special case.
        ggml_tensor* tail = ggml_view_2d(pc.ctx, conv_in, n_state, conv_in->ne[1], conv_in->nb[1],
                                          static_cast<size_t>(conv_in->ne[0] - n_state) * conv_in->nb[0]);
        ggml_tensor* write = pc.conv_state->write(pc.ctx, ggml_cont(pc.ctx, tail), layer);
        if (pc.side_effects) {
            pc.side_effects->push_back(write);
        }
    }

    // Same im2col recipe as CONV_1D_DW above, but with the padding carried by the history instead of by
    // p0: over [n_state + n_tokens] columns an unpadded kernel of width K produces exactly n_tokens
    // outputs, which is the causal alignment the padded-then-sliced form was expressing.
    const int p0 = use_state ? 0 : static_cast<int>(n_state);
    ggml_tensor* data_4d = ggml_reshape_4d(pc.ctx, conv_in, conv_in->ne[0], 1, conv_in->ne[1], conv_in->ne[2]);
    ggml_tensor* im2col = ggml_im2col(pc.ctx, kernel, data_4d, /*s0=*/1, 0, p0, 0, /*d0=*/1, 0,
                                       /*is_2D=*/false, GGML_TYPE_F32);
    ggml_tensor* result = ggml_mul_mat(pc.ctx, im2col, kernel);
    result = ggml_reshape_3d(pc.ctx, result, result->ne[0], result->ne[2], 1);

    if (!use_state) {
        // The stateless form pads on BOTH sides (n_state + n_tokens + n_state - (K-1) = n_tokens +
        // n_state outputs) and keeps the causal prefix, matching what the exporter used to emit as
        // CONV_1D_DW + VIEW.
        result = ggml_view_2d(pc.ctx, result, result->ne[0] - n_state, result->ne[1], result->nb[1], 0);
        result = ggml_cont(pc.ctx, result);
    }
    return {result};
}

ggml_op_pool parse_pool_op(const Json& attrs) {
    const std::string op = attrs.at("op").get<std::string>();
    if (op == "max") return GGML_OP_POOL_MAX;
    if (op == "avg") return GGML_OP_POOL_AVG;
    throw SchemaError("POOL: unsupported pool 'op' \"" + op + "\" (expected \"max\" or \"avg\")");
}

// Whether a 1-D pool can be spelled as a 2-D pool with a one-tall window, which is the whole of
// op_pool_1d's lowering decision below (BACKLOG.md P4.7d).
//
// The SHAPES always agree: `ggml_pool_2d` sizes its output
// `[calc(ne0,k0,s0,p0), calc(ne1,k1,s1,p1), ne2, ne3]` against `ggml_pool_1d`'s
// `[calc(ne0,k0,s0,p0), ne1, ne2, ne3]`, and `calc(ne1, 1, 1, 0) == ne1`.
//
// The VALUES do not, for exactly one combination, and it is the kind of difference that would never
// have announced itself: **an AVERAGE pool with padding divides by different numbers.** `pool_1d`
// divides by `count`, the number of in-bounds elements it actually visited; `pool_2d` divides by
// `ka = k0*k1`, the full kernel, treating the padded cells as zeros it still counts. So the two agree
// on every interior window and differ on every window that overhangs an edge. (Measured, not reasoned:
// tests/ci/test_pool_1d_lowering.cpp compares the two spellings across a matrix of parameters, and the
// padded-average row is the one that separates them.)
//
// A MAX pool is unaffected whatever the padding, because a padded cell contributes nothing to a maximum
// either way; and with `p0 == 0` no window overhangs anything, so `count == k0 == ka` and an average
// agrees too.


// Lowered to a 2D pool with a one-tall window wherever that is provably the same operation, because
// **`ggml-vulkan` implements `GGML_OP_POOL_2D` and not `GGML_OP_POOL_1D`**. The 1D spelling is a node
// the scheduler has to hand back to the CPU, splitting the graph around it for a reason no caller could
// see or fix from the export side -- Whisper's encoder is one node of it and paid a split (P4.7d).
//
// This lives here rather than in the exporter because nothing per-MODEL is involved: it is one
// primitive's lowering detail, and a topology that says POOL_1D should keep meaning what it says. Where
// the two are not equivalent, the 1D op stays -- a correct CPU fallback beats a fast wrong answer.
Outputs op_pool_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("POOL_1D", in, 1);
    const ggml_op_pool op = parse_pool_op(attrs);
    const int k0 = static_cast<int>(resolve_attr_int(attrs, "k0", pc.symbols));
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    if (pool_1d_lowers_to_pool_2d(op, p0)) {
        return {ggml_pool_2d(pc.ctx, in[0], op, k0, /*k1=*/1, s0, /*s1=*/1,
                              static_cast<float>(p0), /*p1=*/0.0f)};
    }
    return {ggml_pool_1d(pc.ctx, in[0], op, k0, s0, p0)};
}

Outputs op_pool_2d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("POOL_2D", in, 1);
    const ggml_op_pool op = parse_pool_op(attrs);
    const int k0 = static_cast<int>(resolve_attr_int(attrs, "k0", pc.symbols));
    const int k1 = static_cast<int>(resolve_attr_int(attrs, "k1", pc.symbols));
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int s1 = static_cast<int>(resolve_attr_int(attrs, "s1", pc.symbols));
    const float p0 = static_cast<float>(resolve_attr_number(attrs, "p0", pc.symbols));
    const float p1 = static_cast<float>(resolve_attr_number(attrs, "p1", pc.symbols));
    return {ggml_pool_2d(pc.ctx, in[0], op, k0, k1, s0, s1, p0, p1)};
}

} // namespace

LOOM_REGISTER_OP(CONV_1D, op_conv_1d)
LOOM_REGISTER_OP(CONV_2D, op_conv_2d)
LOOM_REGISTER_OP(CONV_2D_DW, op_conv_2d_dw)
LOOM_REGISTER_OP(CONV_TRANSPOSE_1D, op_conv_transpose_1d)
LOOM_REGISTER_OP(CONV_TRANSPOSE_2D, op_conv_transpose_2d)
LOOM_REGISTER_OP(CONV_1D_DW, op_conv_1d_dw)
LOOM_REGISTER_OP(SHORT_CONV, op_short_conv)
LOOM_REGISTER_OP(POOL_1D, op_pool_1d)
LOOM_REGISTER_OP(POOL_2D, op_pool_2d)

} // namespace loom
