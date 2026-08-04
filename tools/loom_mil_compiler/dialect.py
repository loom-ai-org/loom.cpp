from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import infer_type_with_broadcast

@register_op(namespace="loom")
class loom_fused_attention(Operation):
    """
    Specialized Loom dynamic attention primitive: one whole scaled-dot-product-attention block, standing
    in for the `mul -> matmul(transpose_y) -> add(mask) -> softmax -> matmul -> transpose -> reshape`
    subgraph a traced causal LM produces per layer (see `passes.py`'s `fuse_loom_attention`, which
    inserts it, and `topology_ops.py`'s rule, which lowers it to the engine's own `ATTENTION` primitive).

    This op is **the only door to the engine's KV cache** (KV-CACHE.md §1.1): `op_attention`
    (src/ops/primitives_attention.cpp) is what reads `n_past`/`n_kv` off SymbolEnv, appends this step's
    K/V at cells `[n_past, n_past + n_tokens)` and reads back `[0, n_kv)`. A topology with no ATTENTION
    node cannot touch a cache at all, which is precisely why a MIL-exported causal LM had none: this op
    was registered and mapped (`exporter.py`'s OP_MAP) from the start, and the pass that was supposed to
    produce it had `pass` for a body.

    `scale` is carried rather than left folded onto q because the engine's ATTENTION applies it inside
    `ggml_soft_max_ext`; `layer` indexes the cache and is assigned by the pass in **attention-block
    occurrence order**, not by torch module index -- see `fuse_loom_attention` for why that distinction
    is load-bearing for architectures like LFM2 that interleave non-attention layers.

    Shape contract, in MIL's own (forward) axis order:
        q     [b, n_head,    seq, head_dim_k]
        k     [b, n_head_kv, seq, head_dim_k]
        v     [b, n_head_kv, seq, head_dim_v]
        mask  [b, 1,         seq, kv]
        out   [b, seq, n_head * head_dim_v]

    The output is the flattened, pre-output-projection context -- i.e. the fusion absorbs the trailing
    `transpose`+`reshape` too. That is not a convenience: `op_attention` itself returns
    `[n_embd, n_tokens]`, so declaring anything else here would make the op's MIL type disagree with what
    the engine actually produces, and every downstream shape would be derived from the wrong one.
    """
    input_spec = InputSpec(
        q=TensorInputType(type_domain="T"),
        k=TensorInputType(type_domain="T"),
        v=TensorInputType(type_domain="T"),
        mask=TensorInputType(optional=True, type_domain="T"),
        scale=TensorInputType(const=True, optional=True, type_domain="T"),
        layer=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        from coremltools.converters.mil.mil.input_type import DefaultInputs
        return DefaultInputs(scale=1.0, layer=0)

    def type_inference(self):
        q_shape = list(self.q.shape)
        v_shape = list(self.v.shape)
        if len(q_shape) != 4 or len(v_shape) != 4:
            raise ValueError(
                f"loom_fused_attention expects rank-4 q/v [b, heads, seq, head_dim], got q={q_shape}, "
                f"v={v_shape}"
            )
        batch, n_head, seq = q_shape[0], q_shape[1], q_shape[2]
        head_dim_v = v_shape[3]
        # n_head and head_dim are architecture constants; only `seq` is ever symbolic here. Guarding
        # rather than assuming, because a symbolic head count would silently produce a symbolic n_embd
        # that every downstream reshape would then inherit.
        if not isinstance(n_head, int) or not isinstance(head_dim_v, int):
            raise ValueError(
                f"loom_fused_attention needs a static head count and head_dim to compute its flattened "
                f"output width, got n_head={n_head}, head_dim_v={head_dim_v}"
            )
        return types.tensor(self.q.dtype, (batch, seq, n_head * head_dim_v))

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


@register_op(namespace="loom")
class loom_short_conv(Operation):
    """
    One causal depthwise convolution that owns its cross-step history -- lowered by `topology_ops.py` to
    the engine's `SHORT_CONV` primitive, the only node type that can reach a `ConvStateCache`
    (BACKLOG.md P4.0.10). Produced by `passes.py`'s `fuse_loom_short_conv`.

    Replaces the traced `conv(pad=[K-1, K-1]) -> slice_by_index(first n_tokens)` pair, which is how
    transformers writes a causal conv with no cache: pad both sides, then throw the trailing K-1 outputs
    away. That form is CORRECT for a prefill and unusable for a decode step, because the K-1 columns a
    length-1 window needs are in the previous call, not in this one. `op_short_conv` keeps them instead.

    `layer` addresses the state slot and is assigned in conv-block OCCURRENCE order, exactly as
    `loom_fused_attention`'s is and for the same reason: LFM2-350M declares 16 hidden layers and has 10
    conv blocks, so a torch module index would address past the end of a 10-slot store.

    The declared output is the SLICE's shape, not the padded conv's -- the fusion absorbs the slice, and
    `op_short_conv` returns exactly n_tokens columns.
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        weight=TensorInputType(type_domain="T"),
        layer=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        from coremltools.converters.mil.mil.input_type import DefaultInputs
        return DefaultInputs(layer=0)

    def type_inference(self):
        x_shape = list(self.x.shape)
        if len(x_shape) != 3:
            raise ValueError(
                f"loom_short_conv expects a rank-3 x [batch, channels, seq], got {x_shape}"
            )
        # Same shape in as out: a causal conv consumes n_tokens columns and produces n_tokens columns.
        # `seq` is the only symbolic entry here, and it passes through untouched, which is what lets the
        # op sit inside a decode loop with no shape special case.
        return types.tensor(self.x.dtype, tuple(x_shape))


# The fusion pass that produces `loom_fused_attention` lives in `passes.py` (`loom::
# fuse_loom_attention`), with every other MIL->MIL rewrite this exporter runs. It was a `pass`-bodied
# `FuseLoomAttention` stub here for the whole of P3/P4.0 -- see KV-CACHE.md §1.3, which measured what
# that cost: 28 SOFTMAX and zero ATTENTION nodes in a MIL-exported Qwen3, hence no reachable KV cache.
