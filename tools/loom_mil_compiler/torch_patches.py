"""
Torch-frontend robustness patches for coremltools' PyTorch->MIL converter.

Neither patch is model-specific: any HF causal-LM traced through this pipeline wants both, so they're
applied once at `import loom_mil_compiler` time (see `__init__.py`) rather than re-pasted into every
export script. Previously duplicated verbatim across export_lfm2_monolithic.py, export_lfm2_atomic.py,
and tools/convert_lfm/make_lfm2_gguf.py (EXPORT-IMPROVEMENT-BACKLOG.md item 1).
"""
import numpy as np

_PATCHED = False


def apply_torch_frontend_patches() -> None:
    """Installs both coremltools torch-frontend patches. Idempotent -- safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from coremltools.converters.mil.mil import Builder as mb

    # 1. Support 1-element numpy array conversion in the cast op, so a compile-time-constant
    #    `.item()`-style cast folds to a const instead of producing a dynamic cast node.
    from coremltools.converters.mil.frontend.torch import ops as mil_ops
    _original_cast = mil_ops._cast

    def _robust_cast(context, node, dtype, dtype_name):
        inputs = mil_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray):
            if x.val.size == 1:
                scalar_val = dtype(x.val.item())
                res = mb.const(val=scalar_val, name=node.name)
                context.add(res, node.name)
                return
        _original_cast(context, node, dtype, dtype_name)

    mil_ops._cast = _robust_cast

    # 2. Pre-tile K/V before SDPA decomposition so grouped-query attention (mismatched Q/K head
    #    counts) traces correctly.
    from coremltools.converters.mil.frontend import _utils as mil_frontend_utils
    _original_decompose_sdpa = mil_frontend_utils._decompose_scaled_dot_product_attention

    def _robust_decompose_sdpa(q, k, v, mask, name, scale=None, before_op=None):
        q_shape = list(q.shape)
        k_shape = list(k.shape)
        rank = len(q_shape)

        if rank == 4:
            q_heads = q_shape[1]
            k_heads = k_shape[1]
            if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
                ratio = q_heads // k_heads
                if ratio > 1:
                    k = mb.tile(x=k, reps=[1, ratio, 1, 1], before_op=before_op)
                    v = mb.tile(x=v, reps=[1, ratio, 1, 1], before_op=before_op)
        elif rank == 3:
            q_heads = q_shape[0]
            k_heads = k_shape[0]
            if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
                ratio = q_heads // k_heads
                if ratio > 1:
                    k = mb.tile(x=k, reps=[ratio, 1, 1], before_op=before_op)
                    v = mb.tile(x=v, reps=[ratio, 1, 1], before_op=before_op)

        return _original_decompose_sdpa(q, k, v, mask, name, scale, before_op)

    mil_frontend_utils._decompose_scaled_dot_product_attention = _robust_decompose_sdpa
