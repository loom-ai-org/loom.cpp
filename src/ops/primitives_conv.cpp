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
bool pool_2d_fallback_is_equivalent(ggml_op_pool op, int p0) {
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
//
// WHICH OPERAND HOLDS THE KERNEL, AND WHY IT DEPENDS ON ITS DTYPE.
//
// ggml_mul_mat(a, b) can only read a NON-F32 `a`: ggml_compute_forward_mul_mat converts `b` to
// type_traits_cpu[a->type].vec_dot_type and asserts `b->type == GGML_TYPE_F32` while doing it. The
// recipe above (and stock ggml_conv_1d's) puts the kernel in `b`, so a conv kernel had to be F32 --
// which is why quantizing a convolutional model used to change nothing at all: only a MUL_MAT's FIRST
// operand is eligible, and every conv weight sat in the second. Measured over the shipped files, that
// was 53-92% of the weight bytes in Kokoro/Matcha/StyleTTS2/Supertonic and 73% of VITS, whose Q8_0
// export came out byte-identical to its F32 one.
//
// So the kernel moves to `a` -- but ONLY when it is not F32, and that condition is the whole design.
// mul_mat(kernel, im2col) yields [OC, ...] where the F32 recipe yields [..., OC], so recovering the
// declared layout costs a transpose + cont that the F32 path does not pay. Branching on the dtype means
// an ordinary F32 export runs precisely the graph it ran before -- same ops, same order, bit-identical
// -- and only a file that asked for quantized weights pays for them. `conv_kernel_is_packed` below is
// that predicate, and it covers F16 as well as the quantized types: vec_dot_type[F16] is F16, so an F16
// kernel in `a` needs no conversion either.
//
// The depthwise and causal forms need no transpose at all: their mul_mat is batched per channel, so
// swapping the operands moves OC=1 into ne[0] and the result reshapes back as a pure view. CONV_2D_DW
// was already written kernel-first for unrelated reasons and is unchanged. CONV_TRANSPOSE_1D/2D are
// native ggml ops with no mul_mat, so they are untouched here and their kernels stay F32.
//
// Padding and striding are unaffected by any of this -- they are im2col parameters (p0/s0/d0), and
// reflect padding is a separate op applied upstream (PAD_1D_REFLECT, primitives_basic.cpp).

// Whether CONV_1D and CONV_2D lower to ggml's single convolution op instead of im2col + mul_mat. On
// everywhere now, for the measured reason spelled out in op_conv_1d below; override it to build (and
// test) the other path anywhere. `LOOM_CONV1D_DIRECT` is the name it shipped under while only the 1-D
// form had a direct lowering (P4.29); it is still honoured so that a build line, a bench script or a
// suite run that sets it keeps working, and setting either to 0 turns off both.
#ifndef LOOM_CONV1D_DIRECT
#  define LOOM_CONV1D_DIRECT 1
#endif
#ifndef LOOM_CONV_DIRECT
#  define LOOM_CONV_DIRECT LOOM_CONV1D_DIRECT
#endif

// Whether `kernel` is stored in a form that must occupy mul_mat's FIRST operand. F32 is the only dtype
// that can sit in the second, and it is the one every pre-quantization export uses.
bool conv_kernel_is_packed(const ggml_tensor* kernel) {
    return ggml_is_quantized(kernel->type);
}

// The dtype the im2col patch matrix is materialised in, which is what decides whether the kernel can
// stay in mul_mat's SECOND operand. ggml converts src1 to `vec_dot_type[src0->type]` and asserts src1
// is F32 while doing it -- so an F16 kernel is legal in src1 exactly when src0 is also F16, because
// vec_dot_type[F16] is F16 and no conversion is attempted. Matching the im2col to the kernel is what
// stock ggml_conv_1d does for the same reason, and it is strictly cheaper than the alternative: the
// patch matrix is written in its final type once, instead of being written F32 and then converted to
// F16 into scratch on every call -- and the im2col is the BIG operand here, sized by the activations.
ggml_type conv_im2col_type(const ggml_tensor* kernel) {
    return kernel->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

// mul_mat with the kernel in `a`, transposed back to the [.., OC] the F32 recipe produces.
// Split out because CONV_1D and CONV_2D need exactly the same two lines around otherwise identical
// reshapes, and because the transpose is the entire runtime cost of quantized conv weights -- one
// place to find it when benchmarking, and one place to change if a future ggml can write it directly.
ggml_tensor* mul_mat_kernel_first(ggml_context* ctx, ggml_tensor* kernel_2d, ggml_tensor* im2col_2d) {
    ggml_tensor* result = ggml_mul_mat(ctx, kernel_2d, im2col_2d);   // [OC, N*OL]
    return ggml_cont(ctx, ggml_transpose(ctx, result));              // [N*OL, OC]
}

// ---------------------------------------------------------------------------------------------------
// FOLDED CONV KERNELS (P4.13) -- how a convolutional model becomes block-quantizable at all
// ---------------------------------------------------------------------------------------------------
// ggml lays quantization blocks along ne[0]. For a conv kernel stored the way torch declares it,
// [K, IC, OC] (1-D) or [KW, KH, IC, OC] (2-D), ne[0] is the KERNEL WIDTH -- 1, 3, 5, 7 -- and no block
// size divides that. So a conv kernel is not block-quantizable AS STORED, and the operand swap above
// bought nothing on its own: every kernel cleared the op gate and then failed the shape gate. Measured
// on vits-piper-en-gb-miro, 0 of its 117 conv kernels aligned for block 32.
//
// The exporter therefore FOLDS an eligible kernel's spatial axes into ne[0] before quantizing it --
// [K, IC, OC] becomes [IC*K, OC], [KW, KH, IC, OC] becomes [IC*KH*KW, OC] -- which is exactly the 2-D
// shape the mul_mat already reshapes to, so ne[0] becomes IC*K and 114 of those same 117 kernels align
// (99.99% of the conv BYTES; the three that do not are 1x1x192 duct tape). It does this ONLY for a
// kernel it is about to block-quantize: an F32 or F16 export keeps the declared 3-D/4-D form and runs
// precisely the graph it ran before, bit for bit, including the direct-conv lowering below.
//
// A FOLDED KERNEL NOW REACHES THAT LOWERING TOO (P4.29). It could not when the fold shipped -- ggml's
// GGML_OP_CONV_2D read its geometry off the kernel's `ne`, which the fold erases, and took an F32 or
// F16 kernel besides -- and giving it up, rather than the arithmetic, is where essentially all of a
// quantized model's 2.08x (x86-64) / 2.13x (Cortex-A72) went. `ggml_conv_2d_direct_packed` is told the geometry instead, and the CPU backend
// dequantizes the kernel into the scratch buffer its direct sweep already repacks an F32 one into.
// See the second branch in op_conv_1d.
//
// What the fold costs is the geometry, because the shape it erases is the shape the convolution IS.
// The exporter hands it back on the node instead, as `kernel_k`/`kernel_ic` (CONV_1D) and
// `kernel_kw`/`kernel_kh`/`kernel_ic` (CONV_2D); OC survives as ne[1]. `folded_kernel_geometry` below
// reads them, and its absence is what says a kernel is in the declared layout -- NOT the tensor's rank,
// which cannot tell a folded [IC*K, OC] from a declared [K, IC] with OC left implicit.
//
// WHY THIS IS A MODERATE CHANGE AND NOT A REIMPLEMENTATION OF im2col:
// `ggml_compute_forward_im2col_f32/f16` touch `src1->data` only. `src0` -- the kernel -- is read purely
// for ne[0]/ne[1] (and ne[2] when is_2D), to size the patch matrix. Confirmed by reading both, and it
// is the whole reason the op can be handed a kernel whose contents are somewhere else entirely: what
// im2col needs is a SHAPE, so it is given one.
//
// THE SHAPE CARRIER IS THE KERNEL MINUS ITS OC AXIS, WHICH IS WHY IT IS FREE.
// The obvious carrier is a [K, IC, OC]-shaped tensor, and it is what this item was scoped with -- a
// graph leaf gallocr has to allocate, ~1.7 MB for VITS's largest kernel. But im2col never reads a->ne[3]
// and, in the 1-D case, never reads a->ne[2] either: the OC axis is not part of the patch geometry. So
// the carrier is [K, IC] (1-D) or [KW, KH, IC] (2-D) -- the largest one in VITS is 3x512 floats, 6 KB,
// and the whole model's carriers together are under a megabyte. Nothing writes it and nothing reads it;
// it exists to carry four int64s past a signature that takes a tensor.
struct FoldedKernel {
    bool folded = false;
    int64_t kw = 1, kh = 1, ic = 1, oc = 1;
};

// The fold attrs, validated against BOTH tensors that arrived. Two checks, and the second is the one
// that matters more than it looks: `data_ic` is the activation's own channel count, and comparing it
// against the declared IC is what turns a transposed or mis-ordered fold into a named error here
// instead of a raw GGML_ASSERT inside ggml_im2col -- which is what it aborts with otherwise, with no
// mention of a kernel, an attr or a model. (Verified by sabotage: swapping `kernel_k` and `kernel_ic`
// keeps their product equal to ne[0], so the first check passes and only this one catches it.)
//
// `is_2D` also decides which spelling is looked for: `kernel_k` for a 1-D convolution,
// `kernel_kw`/`kernel_kh` for a 2-D one. A 1-D fold declares no KH because it has none, and accepting
// one silently would let a 2-D export drive a 1-D op.
FoldedKernel folded_kernel_geometry(const char* op, const ggml_tensor* kernel, const Json& attrs,
                                     const SymbolEnv& symbols, bool is_2D, int64_t data_ic) {
    FoldedKernel g;
    if (!attrs.is_object() || !attrs.contains(is_2D ? "kernel_kw" : "kernel_k")) return g;
    g.folded = true;
    g.kw = resolve_attr_int(attrs, is_2D ? "kernel_kw" : "kernel_k", symbols);
    g.kh = is_2D ? resolve_attr_int(attrs, "kernel_kh", symbols) : 1;
    g.ic = resolve_attr_int(attrs, "kernel_ic", symbols);
    g.oc = kernel->ne[1];
    // A fold that does not multiply back out is a file whose attrs and whose tensor disagree, and the
    // failure without this check is a wrong-shaped im2col deep inside ggml rather than a named one.
    if (g.kw < 1 || g.kh < 1 || g.ic < 1 || kernel->ne[0] != g.kw * g.kh * g.ic ||
        kernel->ne[2] != 1 || kernel->ne[3] != 1) {
        throw SchemaError(std::string(op) + ": the topology declares a kernel folded to [IC*" +
                          (is_2D ? "KH*KW" : "K") + ", OC] with K=" + std::to_string(g.kw) +
                          (is_2D ? ("x" + std::to_string(g.kh)) : "") + " IC=" + std::to_string(g.ic) +
                          ", which needs ne[0] == " + std::to_string(g.kw * g.kh * g.ic) +
                          " on a 2-D tensor -- the weight it names is [" +
                          std::to_string(kernel->ne[0]) + ", " + std::to_string(kernel->ne[1]) + ", " +
                          std::to_string(kernel->ne[2]) + ", " + std::to_string(kernel->ne[3]) + "]");
    }
    if (g.ic != data_ic) {
        throw SchemaError(std::string(op) + ": the topology declares a folded kernel with IC=" +
                          std::to_string(g.ic) + " but the activation has " + std::to_string(data_ic) +
                          " channel(s). The fold hands K and IC back as attrs because folding erased "
                          "them, so a disagreement here is the attrs, not the data.");
    }
    return g;
}

// A tensor whose ne carries the patch geometry `ggml_im2col` reads off its kernel operand, and nothing
// else -- see the block comment above for why that is all it has to carry, and why it is small.
ggml_tensor* im2col_shape_carrier(ggml_context* ctx, const ggml_tensor* kernel, const FoldedKernel& g,
                                   bool is_2D) {
    ggml_tensor* carrier = is_2D ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, g.kw, g.kh, g.ic)
                                 : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.kw, g.ic);
    ggml_format_name(carrier, "%s.geom", kernel->name);
    return carrier;
}

Outputs op_conv_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("CONV_1D", in, 2);
    ggml_tensor* kernel = in[0];
    ggml_tensor* data = in[1];
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    const int d0 = static_cast<int>(resolve_attr_int(attrs, "d0", pc.symbols));
    const FoldedKernel fold = folded_kernel_geometry("CONV_1D", kernel, attrs, pc.symbols, /*is_2D=*/false, data->ne[1]);

    // ggml_compute_forward_im2col asserts its `data` operand's fastest-varying axis is densely packed
    // (nb[0] == sizeof(float)) -- true for most producers, but not for a channel-split VIEW feeding
    // straight into a pointwise/depthwise conv (confirmed on Conformer-CTC's GLU-split conv module: the
    // first half of a channel-split tensor is a genuinely strided view, not a fresh contiguous buffer).
    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);

    // ON aarch64, LOWER TO ggml's SINGLE CONV OP INSTEAD, because materialising the whole patch matrix
    // is the wrong trade there. The recipe below writes an [IC*K, OL] im2col matrix to memory and then
    // GEMMs it: for a VITS vocoder that is ~550 MB written and read back per synthesis, and after
    // P4.15 made the GEMM 1.7x faster it accounted for 40% of all convolution time. GGML_OP_CONV_2D
    // (with KH=1, which is what the reshapes below it are for) does the same arithmetic a cache-sized
    // BATCH of patches at a time, so the patches never leave L2 -- but only once ggml's implementation
    // of it is fixed to size its batch for a cache and to write the GEMM straight into the output;
    // both are cmake/patches/ggml-0004-conv2d-cache-blocked.patch, and without them this op measured
    // 0.97x, which is why P4.14 closed it.
    //
    // Measured on a Cortex-A72 over a VITS vocoder's eleven convolution shapes, 4 threads, output
    // BIT-IDENTICAL either way (the batching splits the patch axis, never the reduction): **1.18x**,
    // and 1.10-1.63x on the long-activation convs that dominate.
    //
    // THIS USED TO BE aarch64-ONLY, and the reason it is not any more is worth keeping. On the
    // cache-blocked im2col alone, x86-64 measured **0.87x** -- there the patch matrix is worth
    // materialising, because that machine does in 0.555 s what the Pi takes 1.37 s to do and
    // cache-blocking buys a bandwidth-rich core much less. What changed is that ggml's convolution now
    // has a DIRECT kernel behind a cache-size heuristic (ggml-0006), which does not materialise
    // anything at all: **4.7x** on that machine's long-activation convolutions, and **1.19x**
    // end-to-end on the same synthesis (1.503 -> 1.258 s, two threads pinned). The shapes the
    // heuristic declines still get the batched im2col, and are the only thing this lowering gives up.
    // `scripts/bench10.cpp` is where that decision lives; re-run it on a machine neither of these two
    // represents -- a many-core server, a wide ARM core -- rather than reasoning about it.
    //
    // Eligibility mirrors ggml_compute_forward_conv_2d's own asserts: an F32 or F16 kernel (a
    // quantized one takes the packed path below), contiguous, and a dense (non-grouped) conv, which
    // is what this primitive is. Vulkan, CUDA and Metal all implement GGML_OP_CONV_2D, so this does
    // not strand a graph on the CPU when a backend is present.
    //
    // The `#if` is a DEFAULT, not a wall: `-DLOOM_CONV1D_DIRECT=1` builds this path on any target, and
    // running the suite that way is how it gets tested where the fixtures are -- the conv-bearing
    // gates (kokoro, styletts2, matcha, the NeMo encoders) live on an x86 dev box, and a Raspberry Pi
    // has none of them. Both directions are expected to pass, because the two lowerings agree bit for
    // bit.
#if LOOM_CONV_DIRECT
    // `!fold.folded` is not implied by the dtype test beside it and is not redundant: a folded kernel
    // has lost the [KW, KH, IC, OC] shape ggml_conv_2d_direct reads its geometry from, and the fold is
    // the exporter's decision rather than this op's, so it is checked rather than inferred. In practice
    // only a block-quantized kernel is ever folded, and that one is excluded twice over -- and then
    // caught by the branch immediately below, which is the same lowering told the geometry.
    if (!fold.folded && !conv_kernel_is_packed(kernel) && ggml_is_contiguous(kernel) &&
        (kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16)) {
        const int64_t IC = kernel->ne[1], OC = kernel->ne[2], K = kernel->ne[0];
        const int64_t IL = data->ne[0], N = data->ne[2];   // ggml never leaves an axis at 0; [IL,IC,N]
        GGML_ASSERT(data->ne[1] == IC && data->ne[3] == 1);
        ggml_tensor* kernel_4d = ggml_reshape_4d(pc.ctx, kernel, K, 1, IC, OC);   // [KW, KH=1, IC, OC]
        ggml_tensor* data_4d   = ggml_reshape_4d(pc.ctx, data, IL, 1, IC, N);     // [W, H=1, C, N]
        ggml_tensor* conv = ggml_conv_2d_direct(pc.ctx, kernel_4d, data_4d, s0, 1, p0, 0, d0, 1);
        return {ggml_reshape_3d(pc.ctx, conv, conv->ne[0], OC, N)};               // [OL, OC, N]
    }

    // ... AND THE SAME LOWERING FOR A FOLDED, BLOCK-QUANTIZED KERNEL (BACKLOG P4.29).
    //
    // Everything the branch above buys is what quantizing a convolutional model used to give up, and
    // that is where essentially all of the 2.08x (x86-64) / 2.13x (Cortex-A72) cost of quantizing one
    // lived -- not in the arithmetic. Measured on a VITS vocoder, one synthesis, interleaved A-B-B-A:
    // the direct sweep alone is worth 1.46x / 1.33x, and ggml's batching plus the three CONV_2D
    // fusions the other 1.07x. A quantized kernel could reach none of it, because GGML_OP_CONV_2D took
    // an F32 or F16 kernel and a folded one carries no geometry on its `ne` at all.
    //
    // ggml-0013 fixes both halves: `ggml_conv_2d_direct_packed` is told the geometry the fold erased,
    // and the CPU backend dequantizes the kernel into the scratch buffer its direct sweep already
    // repacks an F32 one into -- so from that point down a quantized convolution executes the SAME
    // instructions as an F32 one, and the expected end state is F32 speed plus the dequantize (+1.8 ms
    // on a ~500 ms synthesis, measured over all 59.5 MB of that model's conv weights at once).
    //
    // ASKED rather than assumed, which is the P4.7e pattern: this is a CPU-only arrangement today. A
    // Vulkan or CUDA backend declines the node -- neither can read a folded kernel, and saying so is
    // part of the same patch -- and the graph keeps the im2col + `mul_mat_kernel_first` lowering below,
    // which is what those backends already run with zero fallback nodes.
    if (fold.folded && fold.kh == 1 && conv_kernel_is_packed(kernel) && ggml_is_contiguous(kernel)) {
        const int64_t IL = data->ne[0], N = data->ne[2];
        GGML_ASSERT(data->ne[1] == fold.ic && data->ne[3] == 1);
        ggml_tensor* data_4d = ggml_reshape_4d(pc.ctx, data, IL, 1, fold.ic, N);   // [W, H=1, C, N]
        ggml_tensor* conv = ggml_conv_2d_direct_packed(pc.ctx, kernel, data_4d, s0, 1, p0, 0, d0, 1,
                                                       static_cast<int>(fold.kw), 1,
                                                       static_cast<int>(fold.ic));
        if (backend_can_run(pc, conv)) {
            return {ggml_reshape_3d(pc.ctx, conv, conv->ne[0], fold.oc, N)};       // [OL, OC, N]
        }
    }
#endif

    // A folded kernel IS the [IC*K, OC] matrix this recipe reshapes to, so it skips the reshape and
    // hands im2col a shape carrier instead of itself; an unfolded one is reshaped exactly as before.
    ggml_tensor* im2col_src = fold.folded ? im2col_shape_carrier(pc.ctx, kernel, fold, /*is_2D=*/false) : kernel;
    ggml_tensor* im2col = ggml_im2col(pc.ctx, im2col_src, data, s0, 0, p0, 0, d0, 0, /*is_2D=*/false, conv_im2col_type(kernel)); // [IC*K, OL, N]
    ggml_tensor* im2col_2d = ggml_reshape_2d(pc.ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]); // [IC*K, N*OL]
    ggml_tensor* kernel_2d = fold.folded
        ? kernel
        : ggml_reshape_2d(pc.ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);      // [IC*K, OC]
    const int64_t oc = fold.folded ? fold.oc : kernel->ne[2];
    ggml_tensor* result = (fold.folded || conv_kernel_is_packed(kernel))
        ? mul_mat_kernel_first(pc.ctx, kernel_2d, im2col_2d)
        : ggml_mul_mat(pc.ctx, im2col_2d, kernel_2d);                                         // [N*OL, OC]
    result = ggml_reshape_3d(pc.ctx, result, im2col->ne[1], oc, im2col->ne[2]);               // [OL, OC, N]
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
    const FoldedKernel fold = folded_kernel_geometry("CONV_2D", kernel, attrs, pc.symbols, /*is_2D=*/true, data->ne[2]);

    if (!ggml_is_contiguous(data)) data = ggml_cont(pc.ctx, data);

#if LOOM_CONV_DIRECT
    // A FOLDED, BLOCK-QUANTIZED KERNEL TAKES THE DIRECT SWEEP HERE TOO (P4.30c step 3), and this is the
    // only line of it that is new: `ggml_conv_2d_direct_packed` has taken a `kh` since ggml-0013 shipped
    // it for the 1-D form, its CPU implementation dequantizes into the DECLARED [KW, KH, IC, OC] layout
    // and re-enters, and every predicate below that point is already the general 2-D one. P4.29 left
    // this form on im2col + `mul_mat_kernel_first` on the argument that nothing in tree had a quantized
    // 2-D convolution hot enough to notice. That argument was WRONG and the census is why: three ASR
    // encoders fold a 2-D subsampling kernel to a block-aligned [IC*KH*KW, OC], and on qwen3-asr-0.6b
    // the two that do are ~10% of an F32 transcription. Quantizing them cost 1.77x and 1.81x -- 974 ->
    // 1720 ms and 251 -> 455 ms on an 11 s utterance, one thread -- which is P4.13's 2.08x surviving in
    // the one op P4.29 did not reach. See Epic-05 P4.30c.
    //
    // The transpose the fallback owes is not paid here either. `mul_mat_kernel_first` has to put the
    // kernel in mul_mat's first operand and transpose the result back, and at these shapes that CONT is
    // 15.2 ms of qwen3-asr's larger stem convolution on its own; the direct sweep writes [OW, OH, OC, N]
    // straight out, which is the layout this op returns.
    //
    // Same two guards as the 1-D branch, for the same two reasons: `fold.folded` because a folded kernel
    // is the only thing that can carry the geometry ggml is being told, and `backend_can_run` because a
    // Vulkan or CUDA backend declines a packed node and must be left the im2col lowering it already runs
    // with zero fallback nodes.
    if (fold.folded && conv_kernel_is_packed(kernel) && ggml_is_contiguous(kernel)) {
        GGML_ASSERT(data->ne[2] == fold.ic);
        ggml_tensor* conv = ggml_conv_2d_direct_packed(pc.ctx, kernel, data, s0, s1, p0, p1, d0, d1,
                                                       static_cast<int>(fold.kw),
                                                       static_cast<int>(fold.kh),
                                                       static_cast<int>(fold.ic));
        if (backend_can_run(pc, conv)) {
            return {conv};                                                        // [OW, OH, OC, N]
        }
    }
#endif

    ggml_tensor* im2col_src = fold.folded ? im2col_shape_carrier(pc.ctx, kernel, fold, /*is_2D=*/true) : kernel;
    ggml_tensor* im2col = ggml_im2col(pc.ctx, im2col_src, data, s0, s1, p0, p1, d0, d1, /*is_2D=*/true, conv_im2col_type(kernel)); // [IC*KH*KW, OW, OH, N]
    ggml_tensor* im2col_2d = ggml_reshape_2d(pc.ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]); // [IC*KH*KW, N*OH*OW]
    ggml_tensor* kernel_2d = fold.folded
        ? kernel
        : ggml_reshape_2d(pc.ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]); // [IC*KH*KW, OC]
    const int64_t oc = fold.folded ? fold.oc : kernel->ne[3];
    ggml_tensor* result = (fold.folded || conv_kernel_is_packed(kernel))
        ? mul_mat_kernel_first(pc.ctx, kernel_2d, im2col_2d)
        : ggml_mul_mat(pc.ctx, im2col_2d, kernel_2d);                                                          // [N*OH*OW, OC]
    result = ggml_reshape_4d(pc.ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], oc);                  // [OW, OH, N, OC]
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
    ggml_tensor* im2col = ggml_im2col(pc.ctx, kernel, data_4d, s0, 0, p0, 0, d0, 0, /*is_2D=*/false, conv_im2col_type(kernel));
    // Free either way here, unlike the dense forms. This mul_mat is BATCHED over channels (ggml_mul_mat
    // pairs index-for-index over ne[2] rather than taking a cross product), so the per-channel matrix is
    // [K, OL] against [K, 1] and swapping the operands moves the length of 1 from ne[1] to ne[0]:
    // [OL, 1, C] becomes [1, OL, C], whose buffer is `ol + c*OL` either way. So the reshape that follows
    // is a pure view in both branches and no transpose is needed.
    ggml_tensor* result = conv_kernel_is_packed(kernel)
        ? ggml_reshape_3d(pc.ctx, ggml_mul_mat(pc.ctx, kernel, im2col), im2col->ne[1], im2col->ne[2], 1)
        : ggml_reshape_3d(pc.ctx, ggml_mul_mat(pc.ctx, im2col, kernel), im2col->ne[1], im2col->ne[2], 1);
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
                                       /*is_2D=*/false, conv_im2col_type(kernel));
    // Same batched-per-channel shape as CONV_1D_DW, so the same free swap -- see the note there. The
    // history concat above is upstream of this and unaffected: it changes what is convolved, not how.
    ggml_tensor* result = conv_kernel_is_packed(kernel)
        ? ggml_reshape_3d(pc.ctx, ggml_mul_mat(pc.ctx, kernel, im2col), im2col->ne[1], im2col->ne[2], 1)
        : ggml_reshape_3d(pc.ctx, ggml_mul_mat(pc.ctx, im2col, kernel), im2col->ne[1], im2col->ne[2], 1);

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

// Whether a 1-D pool can be spelled as a 2-D pool with a one-tall window -- the FALLBACK op_pool_1d
// reaches for when the backend has no `ggml_pool_1d` of its own.
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

// Asks the backend, then chooses -- the same shape as `op_pad_1d_reflect` and for the same reason
// (BACKLOG.md P4.7e). The support for these two is not the same across backends and does not have to
// be reasoned about here: **CUDA has `PAD_REFLECT_1D` but no `POOL_1D`; Metal and SYCL have both;
// Vulkan has neither but does have `POOL_2D`**, which is the only pooling op every GPU backend
// implements.
//
// Keeping the native op wherever the backend runs it matters even though the fallback is exact: a
// topology that says POOL_1D gets POOL_1D on every backend that has one, and a CPU-only build -- the
// default -- is left exactly as it was before any of this. An earlier version of this primitive
// preferred `pool_2d` unconditionally, which quietly changed the graph for every CPU user to work
// around a gap none of them had.
Outputs op_pool_1d(PrimitiveContext& pc, const Inputs& in, const Json& attrs) {
    expect_n_inputs("POOL_1D", in, 1);
    const ggml_op_pool op = parse_pool_op(attrs);
    const int k0 = static_cast<int>(resolve_attr_int(attrs, "k0", pc.symbols));
    const int s0 = static_cast<int>(resolve_attr_int(attrs, "s0", pc.symbols));
    const int p0 = static_cast<int>(resolve_attr_int(attrs, "p0", pc.symbols));
    ggml_tensor* native = ggml_pool_1d(pc.ctx, in[0], op, k0, s0, p0);
    // Not `pool_2d_fallback_is_equivalent` first: the question is what this backend can run, and the
    // fallback is only interesting when the answer is "not this". Where there is no equivalent fallback
    // (a padded average) the native op stays and the scheduler sends it to the CPU -- a correct
    // fallback beats a fast wrong answer.
    if (backend_can_run(pc, native) || !pool_2d_fallback_is_equivalent(op, p0)) {
        return {native};
    }
    return {ggml_pool_2d(pc.ctx, in[0], op, k0, /*k1=*/1, s0, /*s1=*/1,
                          static_cast<float>(p0), /*p1=*/0.0f)};
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
