from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import infer_type_with_broadcast
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

@register_op(namespace="loom")
class loom_fused_attention(Operation):
    """
    Specialized Loom dynamic attention primitive.
    """
    input_spec = InputSpec(
        q=TensorInputType(type_domain="T"),
        k=TensorInputType(type_domain="T"),
        v=TensorInputType(type_domain="T"),
        mask=TensorInputType(optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        return self.q.sym_type

@register_op(namespace="loom")
class loom_spline(Operation):
    """
    Specialized Loom rational-quadratic spline inverse (VITS's StochasticDurationPredictor/ConvFlow).
    See tools/loom_mil_compiler/vits_spline_op.py for the torch-level custom op bridged into this.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        w=TensorInputType(type_domain="T"),
        h=TensorInputType(type_domain="T"),
        d=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type

@register_op(namespace="loom")
class loom_group_norm(Operation):
    """
    Specialized Loom GroupNorm (normalization only, no affine) -- see group_norm_op.py for why this is a
    custom op bridge rather than a generic decomposition-based translation.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        n_groups=TensorInputType(type_domain="U"),
        eps=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.int32, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type

@register_op(namespace="loom")
class loom_rope(Operation):
    """
    Specialized Loom Rotary Position Embedding (RoPE) operation.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        pos=TensorInputType(type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
        "U": (types.int32, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type

@register_op(namespace="loom")
class loom_broadcast_to(Operation):
    """
    Broadcasts `x` up to `like`'s shape. `like` only ever supplies a shape -- its own data is never
    read -- and is always the ORIGINAL other operand of the `add`/`mul` this stands in for, so
    `infer_type_with_broadcast(x, like)` computes exactly the same target shape `add`/`mul`'s own type
    inference would for that same pair.

    A real graph op standing in for what the exporter's emitter used to decide ad hoc: whether an
    `add`/`mul` operand needing MUTUAL (different-axis) broadcast -- each operand size-1 on a
    DIFFERENT axis than the other, so neither is simply "the other's shape with some 1s" -- gets a
    REPEAT node spliced in before it, by comparing rendered shape strings at emission time
    (EXPORT-ROADMAP.md R2a; see `passes.py`'s `insert_explicit_broadcasts`, which inserts this op, and
    `topology_ops.py`'s `loom_broadcast_to` rule, which lowers it 1:1 to the same `REPEAT` primitive
    the old ad hoc code emitted).
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        like=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def type_inference(self):
        return infer_type_with_broadcast(self.x.sym_type, self.like.sym_type, self.x.dtype)


@register_op(namespace="loom")
class loom_replicate_pad(Operation):
    """
    Edge-replicate padding (`nn.functional.pad(x, pad, mode="replicate")`) along the fastest-varying
    (last, MIL-order) axis: `lp` elements copied from the leading edge, `rp` from the trailing edge.

    ggml has no native replicate/edge-pad kernel (unlike `PAD_1D`/`PAD_1D_REFLECT`, which wrap real
    `ggml_pad_ext`/`ggml_pad_reflect_1d` primitives), so this stands in for what the exporter's emitter
    used to compose ad hoc, straight out of `pad`'s own `mode="replicate"` guard: VIEW out the boundary
    column, REPEAT-broadcast it to the pad width, CONCAT it back on (EXPORT-ROADMAP.md R2; see
    `passes.py`'s `canonicalize_replicate_pad`, which inserts this op, and `topology_ops.py`'s
    `loom_replicate_pad` rule, which composes it exactly the way that ad hoc code did). `lp`/`rp` are
    always static -- kernel_size/dilation are architecture constants -- so this never needs a dynamic
    pad width, only (for the right edge) a dynamic byte OFFSET into `x`, which the topology rule derives
    the same way every other dynamic-offset `VIEW` in this exporter already does.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        lp=TensorInputType(const=True, type_domain=types.int32),
        rp=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        shape = list(self.x.shape)
        shape[-1] = shape[-1] + int(self.lp.val) + int(self.rp.val)
        return types.tensor(self.x.dtype, tuple(shape))


@register_op(namespace="loom")
class loom_conv_transpose_dw(Operation):
    """
    A depthwise (`groups == in_channels == out_channels`) 1D `ConvTranspose1d`, always traced with the
    real PyTorch padding/output_padding already folded away to `pad=[0,0]` -- MIL's own tracing of a
    grouped conv_transpose always computes the "valid" (unpadded) result and defers any real crop to a
    separate downstream `slice_by_index` op, which the generic per-op-type path already handles on its
    own turn, so this op never needs to represent a real nonzero pad itself.

    ggml has no native grouped `CONV_TRANSPOSE` primitive at all, so this stands in for what the
    exporter's emitter used to compose ad hoc, straight out of `conv_transpose`'s own `groups != 1`
    guard (EXPORT-ROADMAP.md R2; see `passes.py`'s `canonicalize_conv_transpose_dw`, which inserts this
    op, and `topology_ops.py`'s `loom_conv_transpose_dw` rule, which composes it exactly the way that ad
    hoc code did): the standard "zero-stuff the input by `stride`, then an ordinary stride=1 depthwise
    conv with a kernel-reversed weight" identity -- real `ConvTranspose1d` IS mathematically a
    correlation with a flipped kernel over a zero-stuffed signal. `stride` is always static (an
    architecture constant); `weight` is the ORIGINAL (un-flipped) traced weight -- flipping it needs a
    real constant value, which only the topology rule (via `static_value`) resolves, not this pass.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        weight=TensorInputType(type_domain="T"),
        bias=TensorInputType(optional=True, type_domain="T"),
        stride=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        # x: [N, C, L]; weight: [C, 1, K] (depthwise: out_channels/group == 1). Real ConvTranspose1d
        # output length at pad=0/output_padding=0/dilation=1: (L-1)*stride + K.
        n, c, length = self.x.shape
        k = self.weight.shape[-1]
        stride = int(self.stride.val)
        out_len = (length - 1) * stride + k
        return types.tensor(self.x.dtype, (n, c, out_len))


@register_op(namespace="loom")
class loom_mean(Operation):
    """
    Reduces the fastest-varying (last, MIL-order) axis by averaging, with the reduction COUNT resolved
    at run time rather than baked in at export time -- `ggml_mean` (src/ops/primitives_mean.cpp) always
    reduces ne[0] and divides by its own real element count at `ggml_backend_graph_compute()` time, so
    unlike `loom_scale`'s `reduce_sum`-then-scale-by-a-baked-constant composition, this needs no static
    axis size at all.

    Standing in for `reduce_mean`'s own "reduced axis is ne[0] but its size is only known at run time"
    case (EXPORT-ROADMAP.md R2) -- previously an implicit fall-through to the generic OP_MAP path (no
    `topology_ops.py` rule matched, so the mechanical "map op_type straight to a primitive" default
    applied); this makes that case an explicit graph op instead, so all three `reduce_mean` outcomes
    (static count -> `loom_scale`-based composition, dynamic count on ne[0] -> this op, dynamic count on
    another axis -> unrepresentable) are decided once, by `passes.py`'s `lower_reduce_mean`, rather than
    two of the three being explicit guards and the third an unstated fall-through. First (and so far
    only) needed by StyleTTS2's diffusion `Transformer1d.run()`'s `x.mean(axis=-1)`.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        keep_dims=TensorInputType(const=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        shape = list(self.x.shape)
        axis = len(shape) - 1
        if bool(self.keep_dims.val):
            shape[axis] = 1
        else:
            del shape[axis]
        return types.tensor(self.x.dtype, tuple(shape))


@register_op(namespace="loom")
class loom_scale(Operation):
    """
    Divides `x` by the compile-time-constant integer `n` -- the exact `SCALE` ggml primitive
    (src/ops/primitives_basic.cpp, `s=1/n`) `reduce_mean`'s "reduced axis has a statically-known size"
    case composes into, after a `reduce_sum` (EXPORT-ROADMAP.md R2; see `passes.py`'s
    `lower_reduce_mean`, which inserts both, and `topology_ops.py`'s `loom_scale` rule, which computes
    `1.0/n` itself). A dedicated op rather than a plain `mul` by a constant tensor: ggml has both a
    general elementwise `MUL` and this dedicated `SCALE` primitive, and reusing `SCALE` here (rather
    than composing an equivalent `mul`) keeps the emitted node identical to what the exporter's ad hoc
    emission-time composition already produced.

    Carries the integer `n`, NOT the already-divided `1/n`: MIL casts every float const to fp32 on
    construction regardless of the input's own numpy dtype or the input spec's declared domain (`types
    .fp64` exists as a symbol but isn't actually usable as a stored tensor/const dtype here) -- confirmed
    the hard way, via a real snapshot diff showing `0.005208333333333333` (`1/192` at double precision,
    what the old ad hoc code wrote straight into the JSON attrs dict) silently rounding to
    `0.0052083334885537624` once this op tried to carry the pre-divided float instead. `n` is an exact
    integer either way, so it round-trips through MIL with no such loss, and `topology_ops.py`'s rule
    computes `1.0/n` in plain Python at emission time -- the exact expression the old code used.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        n=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type


@register_pass(namespace="loom")
class FuseLoomAttention(AbstractGraphPass):
    """
    Custom fusion pass scanning standard MIL blocks and fusing them into
    the specialized 'loom_fused_attention' primitive.
    """
    def apply(self, prog):
        for f in prog.functions.values():
            self._fuse_blocks(f)

    def _fuse_blocks(self, block):
        # Placeholder for future pattern matching & replacement logic
        pass
