"""Custom torch op + MIL frontend bridge for `nn.GroupNorm`, needed by Matcha-TTS's Decoder U-Net
(`Block1D`'s real `torch.nn.GroupNorm(8, dim_out)`, borrowed from Kokoro's own istftnet-adjacent style
Conv1d/GroupNorm/Mish stack).

A real `nn.GroupNorm` traces (via coremltools' own torch-frontend decomposition, confirmed directly by
inspecting the raw MIL ops for a standalone `torch.nn.GroupNorm` trace) into a `reshape` (splitting
channels into `[B, G, C/G, T]`) followed by a `reduce_mean(axes=[2,3], keep_dims=True)` -- a genuine
TWO-AXIS joint reduction, over BOTH the per-group channel count (`C/G`, static) AND the spatial/time axis
(`T`, dynamic here). This exporter's existing "reduce_sum"/"layer_norm" dedicated translations (see
exporter.py) only ever handle a SINGLE reduction axis (documented deliberate scope in reduce_sum's own
docstring) -- extending that to a genuine two-axis case where one axis is dynamic would need composing a
runtime-computed divisor (shape-derived count, not a compile-time scalar), a real new capability. Since
ggml ALREADY has a native, independently-verified `GROUP_NORM` primitive (`op_group_norm`,
src/ops/primitives_basic.cpp, wrapping `ggml_group_norm` -- the exact primitive
`tools/convert_matcha/matcha_common.py`'s own bespoke `build_group_norm` already uses successfully) that
computes this exact reduction natively in C++ regardless of dynamic length, the correct fix mirrors
`vits_spline_op.py`'s own precedent: make `nn.GroupNorm.forward` call a custom, OPAQUE-under-tracing torch
op instead of the real decomposition, bridged directly to a native `GROUP_NORM` node -- not because of any
data-dependent-shape issue (unlike the spline case), but to avoid re-deriving a dynamic-axis multi-reduce
composition this exporter doesn't have a general capability for yet, when a real primitive already does
exactly this job.

Only the NORMALIZATION (no affine) goes through the custom op, matching `GROUP_NORM`'s own "affine is a
separate MUL/ADD, never fused in" convention (already established for RMS_NORM/LAYER_NORM too) -- the real
per-channel `weight`/`bias` are applied afterward as ordinary elementwise ops in `_group_norm_traceable`
below, which trace generically with no special handling needed.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

from . import dialect  # noqa: F401  registers the "loom_group_norm" MIL op (mb.loom_group_norm)


@torch.library.custom_op("loom::group_norm_affine_free", mutates_args=())
def group_norm_affine_free(x: Tensor, num_groups: int, eps: float) -> Tensor:
    """Eager fallback -- exercised when running the patched module directly (not traced). Delegates to
    real `F.group_norm` (no affine) so this can never silently drift from ground truth.
    """
    return F.group_norm(x, num_groups, weight=None, bias=None, eps=eps)


@group_norm_affine_free.register_fake
def _(x, num_groups, eps):
    return torch.empty_like(x)


@register_torch_op(torch_alias=["loom::group_norm_affine_free"])
def _convert_loom_group_norm(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x, num_groups, eps = inputs
    res = mb.loom_group_norm(x=x, n_groups=num_groups, eps=eps, name=node.name)
    context.add(res)


def _group_norm_traceable(self, x):
    normed = torch.ops.loom.group_norm_affine_free(x, self.num_groups, self.eps)
    if self.affine:
        shape = [1, -1] + [1] * (x.dim() - 2)
        normed = normed * self.weight.view(*shape) + self.bias.view(*shape)
    return normed


torch.nn.GroupNorm.forward = _group_norm_traceable
