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
