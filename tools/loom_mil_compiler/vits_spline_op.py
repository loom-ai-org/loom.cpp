"""Custom torch op + MIL frontend bridge for VITS's rational-quadratic spline transform.

piper's real `piecewise_rational_quadratic_transform` (transforms.py) uses boolean-mask tensor
indexing (`outputs[inside_interval_mask] = ...`) to blend the "inside the spline's domain" and "outside
(linear tail)" branches. That produces a genuinely DATA-DEPENDENT output shape under `torch.jit.trace`
-- the trace only records whichever elements happened to be inside/outside for the one concrete dummy
input used, and does not generalize to other inputs (confirmed: this is exactly the class of thing
`torch.jit.trace`'s own TracerWarning machinery cannot make safe). There is no dynamic-shape-avoiding
rewrite of the boolean-masked version the way there was for `MultiHeadAttention`'s dynamic pads (see
export_vits_mil.py) -- the mask pattern itself is genuinely data-dependent, not just an artifact of a
particular tracing quirk.

Since RQ_SPLINE_INVERSE (src/ops/primitives_spline.cpp) already exists as a real, independently-verified
ggml primitive that computes the SAME transform via an unconditional elementwise blend (compute both
branches for every element, select via a float mask -- no boolean indexing at all), the correct fix is a
custom torch op that becomes a single OPAQUE node under tracing (never decomposed, so the boolean masking
inside its eager fallback body never gets traced through), bridged to MIL's `loom_spline` op
(tools/loom_mil_compiler/dialect.py -- previously-unwired scaffolding from an earlier prototype) via a
`@register_torch_op` hook. Mirrors the `torch.library.custom_op` precedent already established for the
older `aten_to_loom` pipeline (tools/convert_generic/toy_llm_module.py's `rope_neox`/`attention`), applied
here to coremltools' MIL frontend instead.

x1: [b, half_channels, T] (ConvFlow's own channel-to-transform, always b=half_channels=1 in this
project's single-utterance/mean-only convention). uw/uh: [b, half_channels, T, num_bins]. ud: [b,
half_channels, T, num_bins-1]. Returns the transformed x1, same shape -- logabsdet is dropped (inference
never needs it, same "host logic doesn't need training-only outputs" precedent as every other model here).
"""
import numpy as np
import torch
from torch import Tensor

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

from . import dialect  # noqa: F401  registers the "loom_spline" MIL op (mb.loom_spline) via @register_op

# Real architecture constants (piper's transforms.py DEFAULT_MIN_* / ConvFlow's own tail_bound=5.0,
# confirmed in tools/convert_piper_vits/convert_vits.py's HP table) -- not weights, not derivable from
# any traced tensor, so baked here once and reused by both the eager fallback below and the exporter's
# own RQ_SPLINE_INVERSE translation (exporter.py's `op_type == "loom_spline"` branch).
TAIL_BOUND = 5.0
MIN_BIN_WIDTH = 1e-3
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3


@torch.library.custom_op("loom::spline_inverse", mutates_args=())
def spline_inverse(x1: Tensor, uw: Tensor, uh: Tensor, ud: Tensor) -> Tensor:
    """Eager fallback -- exercised when running the wrapper module directly (not traced), e.g. for a
    pure-PyTorch numeric cross-check against the real `piecewise_rational_quadratic_transform`. Delegates
    to the real piper implementation itself (not a reimplementation) so this can never silently drift
    from ground truth -- the whole point of the custom-op boundary is to keep the TRACER from seeing
    this body, not to avoid calling the real math when actually executed.
    """
    from piper_train.vits.transforms import piecewise_rational_quadratic_transform

    out, _logabsdet = piecewise_rational_quadratic_transform(
        x1, uw, uh, ud, inverse=True, tails="linear", tail_bound=TAIL_BOUND,
        min_bin_width=MIN_BIN_WIDTH, min_bin_height=MIN_BIN_HEIGHT, min_derivative=MIN_DERIVATIVE,
    )
    return out


@spline_inverse.register_fake
def _(x1, uw, uh, ud):
    return torch.empty_like(x1)


@register_torch_op(torch_alias=["loom::spline_inverse"])
def _convert_loom_spline_inverse(context, node):
    inputs = _get_inputs(context, node, expected=4)
    x1, uw, uh, ud = inputs
    res = mb.loom_spline(x=x1, w=uw, h=uh, d=ud, name=node.name)
    context.add(res)
