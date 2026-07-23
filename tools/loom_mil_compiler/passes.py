"""
Loom's own MIL->MIL graph-rewrite passes (EXPORT-IMPROVEMENT-BACKLOG.md item 3).

Coremltools' own backend never mixes graph rewriting with serialization: rewrites run as real
`PassPipeline` stages over the pymil graph *before* backend translation
(`coremltools/converters/mil/backend/mil/load.py`'s `MILProtoExporter.translate_generic_op` is purely
mechanical/schema-driven). `exporter.py`'s `generate_graph_topology` used to interleave both -- detecting
and fusing the GQA `repeat_kv()` idiom inline, from inside the same walk that emits Loom JSON nodes. Pulling
that fusion out to run here, as a real MIL->MIL pass over the pymil graph before `generate_graph_topology`
ever sees it, makes the fusion testable directly against pymil graph structure (pattern match + replace)
instead of against Loom's derived JSON node list, and lets plain `common::dead_code_elimination` clean up
the original tile/reshape idiom's now-orphaned dependency chain (the "reps" computation subgraph --
gather/concat/equal/select/div -- HF traces alongside `repeat_kv()`) instead of a hand-rolled
backward-reachability walk over Loom's own node list.

Run via `apply_loom_mil_passes(prog)`, called once by `LoomGGUFExporter.export()` right after
`ct.convert(...)` has produced the `Program` it was constructed with, before any topology generation.
"""

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY, register_pass
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource


def _is_int(d):
    return isinstance(d, (int, np.integer))


@register_pass(namespace="loom")
class fuse_gqa_repeat_kv(AbstractGraphPass):
    """
    Detects HF's standard `repeat_kv()` idiom -- a tile of a size-1 axis immediately merged back into an
    adjacent axis by a reshape, i.e. `unsqueeze -> tile -> reshape` -- and replaces the `tile`/`reshape`
    pair with an equivalent `reshape -> tile -> reshape` sequence built from provably-reliable shape
    information, entirely bypassing the pattern's own poisoned intermediate shape inference.

    Ported from `exporter.py`'s former `_try_fuse_gqa_repeat_kv` (see EXPORT-BACKLOG.md item 3 and
    EXPORT-IMPROVEMENT-BACKLOG.md item 3), with the derivation reworked to operate directly on real MIL
    `Var.shape` tuples in their natural (forward) axis order instead of Loom's ne-order/string-shape
    representation -- that representation only existed to serialize into Loom's JSON topology format, and
    isn't needed to *identify* the pattern.

    Why this pattern needs special handling at all: coremltools reports `tile`'s own `reps` input as
    non-constant here (`reps.val is None`), because `n_rep` -- an architecturally FIXED hyperparameter --
    gets computed via a runtime shape query during tracing anyway. That poisons shape inference for the
    tile's output and (empirically) for every axis of the following reshape's output too, not just the
    tiled one -- so the fusion derives every non-changed axis from `tile`'s own pre-expand *input* shape
    (unaffected by any of this, reliable by construction) rather than trusting the reshape's declared
    output shape anywhere except the one axis whose change we can positively confirm via a concrete-int
    comparison.

    Why a plain single `REPEAT`/`tile` of the original (pre-unsqueeze) tensor is NOT equivalent: a single
    `ggml_repeat`/MIL `tile` block-tiles an axis (`dst[i] = src[i % ne_src]`, i.e. concatenating whole
    copies: kv0,kv1,...,kv7,kv0,kv1,...,kv7), while `repeat_kv()`'s unsqueeze->expand->reshape-merge idiom
    produces an *interleaved* repeat (`dst[i] = src[i // n_rep]`, i.e. kv0,kv0,kv1,kv1,...,kv7,kv7) -- the
    standard GQA head-group convention. These only agree when n_rep==1. The replacement composes three
    ops that reproduce the interleaved semantics exactly: (1) RESHAPE the pre-tile tensor to insert a
    genuine size-1 axis in the position the real (un-collapsed) unsqueeze put it -- a pure relabeling,
    moves no data since the axis being vacated (batch) is already size 1; (2) TILE that size-1 axis up to
    `n_rep` -- always safe regardless of block-tile-vs-interleave semantics, since tiling a *single* source
    element by any tiling scheme yields the same output; (3) RESHAPE again to merge the now-`n_rep`-sized
    axis into the adjacent kv-heads axis, with `n_rep` as the faster-varying component of the pair --
    exactly reproducing `dst[i] = src[i // n_rep]` via a plain contiguous axis-merge.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._fuse_block(f)

    @block_context_manager
    def _fuse_block(self, block):
        for op in list(block.operations):
            # `getattr(..., block)` rather than a bare `op.enclosing_block`: this exporter's own "bespoke"
            # workflow accepts hand-built Programs containing synthetic, duck-typed ops standing in for
            # ops MIL itself doesn't have (see test_compiler.py's MockOperation), which don't carry a real
            # Operation's full attribute set. Defaulting to `block` (truthy) treats "attribute missing" the
            # same as "still present" -- correct, since a mock op was never removed by this pass.
            if getattr(op, "enclosing_block", block) is None:
                # Already removed by an earlier fusion in this same walk.
                continue
            for b in op.blocks:
                self._fuse_block(b)
            if op.op_type != "tile":
                continue
            child_ops = list(op.outputs[0].child_ops) if op.outputs else []
            if len(child_ops) != 1 or child_ops[0].op_type != "reshape":
                continue
            self._try_to_transform(op, child_ops[0], block)

    @staticmethod
    def _try_to_transform(tile_op, reshape_op, block) -> bool:
        pre_tile_x = tile_op.inputs.get("x")
        if pre_tile_x is None:
            return False
        # `tile_op`'s own "x" is itself the output of the preceding unsqueeze (traced as MIL
        # "expand_dims"), NOT the genuine pre-expand tensor -- walk back through it to find the tensor
        # whose rank actually matches the reshape's own (merged) output rank.
        producer = pre_tile_x.op
        if producer is not None and producer.op_type == "expand_dims":
            inner_x = producer.inputs.get("x") or producer.inputs.get("data")
            if inner_x is not None:
                pre_tile_x = inner_x
        if pre_tile_x.shape is None:
            return False
        pre_shape = tuple(pre_tile_x.shape)
        pre_rank = len(pre_shape)

        out_var = reshape_op.outputs[0]
        if out_var.shape is None or len(out_var.shape) != pre_rank:
            return False
        out_shape = tuple(out_var.shape)

        # Find the one axis whose OUTPUT dim is a reliable (concrete-int) value that differs from the
        # pre-expand input's own dim at the same position. Every other axis is derived from `pre_shape`
        # alone below, never from `out_shape` -- see the class docstring for why `out_shape`'s other axes
        # can't be trusted here.
        changed_axis = None
        for i in range(pre_rank):
            d = out_shape[i]
            if _is_int(d) and (not _is_int(pre_shape[i]) or int(d) != int(pre_shape[i])):
                changed_axis = i

        # The changed axis can never be the fastest-varying (last, MIL-order) axis -- repeat_kv() only
        # ever grows the heads axis, never head_dim.
        if changed_axis is None or changed_axis == pre_rank - 1:
            return False
        if not _is_int(pre_shape[changed_axis]):
            return False

        kv_count = int(pre_shape[changed_axis])
        out_count = int(out_shape[changed_axis])
        if kv_count <= 0 or out_count % kv_count != 0:
            return False
        ratio = out_count // kv_count
        if ratio == 1:
            return False

        # ggml caps tensors at 4 dims -- making room for a genuine new axis at `changed_axis` (by
        # dropping the leading axes before it) only works if that dropped prefix's product is 1, i.e. it's
        # just the (always size-1, batch=1 on this roadmap) leading axis. Verify rather than assume.
        if pre_rank != 4 or not _is_int(pre_shape[0]) or int(pre_shape[0]) != 1:
            return False

        def entry(d):
            # Any non-concrete (symbolic/dynamic) dim collapses to -1, delegating to MIL reshape's own
            # numpy-style single-inferred-axis inference -- valid here because this whole exporter only
            # ever targets models with exactly one true dynamic quantity (sequence length; see
            # exporter.py's `get_var_info` for the full invariant), so at most one entry is ever -1.
            return int(d) if _is_int(d) else -1

        tail = [entry(d) for d in pre_shape[changed_axis + 1:]]
        # (1) insert a genuine size-1 axis right after the (unchanged) kv-heads dim, pushing the
        # always-size-1 batch axis out of the shape entirely -- a pure relabeling of the same flat data.
        reshape1_shape = [entry(pre_shape[changed_axis]), 1] + tail
        # (2) grow that new size-1 axis to `ratio`.
        repeat_reps = [1, ratio] + [1] * len(tail)
        # (3) merge (ratio, kv_count) back into one axis of size `ratio*kv_count`, restoring the original
        # leading axes (batch) that were dropped from (1)/(2).
        final_shape = [entry(d) for d in pre_shape[:changed_axis]] + [out_count] + tail

        out_name = out_var.name

        # Preserve the torch-module scope of the op being replaced (if any) on all three new ops --
        # `try_replace_uses_of_var_after_op` below only auto-copies scope onto the LAST new op (the one
        # whose var directly replaces `out_var`), and the atomic-export profile's scope-based partitioning
        # (`exporter.py`'s `apply_atomic_export`) needs every op it walks to carry the correct
        # TORCHSCRIPT_MODULE_NAME to attribute it to the right decoder layer -- relying on the two
        # intermediate ops merely landing in the right slice by positional adjacency would be exactly the
        # class of fragile mis-attribution EXPORT-IMPROVEMENT-BACKLOG.md item 2 already documents two real
        # bugs from.
        tile_scope = tile_op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME) if tile_op.scopes else None
        scope_ctx = (
            mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=list(tile_scope)))
            if tile_scope
            else mb.scope()
        )
        with scope_ctx:
            r1 = mb.reshape(x=pre_tile_x, shape=reshape1_shape, name=out_name + "_gqa_unsqueeze", before_op=tile_op)
            rep = mb.tile(x=r1, reps=repeat_reps, name=out_name + "_gqa_repeat", before_op=tile_op)
            r2 = mb.reshape(x=rep, shape=final_shape, name=out_name, before_op=tile_op)

        if not reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=out_var, new_var=r2,
        ):
            return False
        block.remove_ops([tile_op, reshape_op])
        return True


_LOOM_PASS_NAMES = ["loom::fuse_gqa_repeat_kv", "common::dead_code_elimination"]


def apply_loom_mil_passes(prog) -> None:
    """
    Runs Loom's own MIL->MIL rewrite passes (currently just GQA `repeat_kv()` fusion) plus
    `common::dead_code_elimination` over `prog` in place. Must run before any topology/driver generation
    sees `prog` -- `common::dead_code_elimination` is what actually removes the original tile/reshape
    idiom's now-orphaned dependency chain that the fusion above leaves behind.

    Invokes each registered pass callable directly (`PASS_REGISTRY[name](prog)`) rather than going through
    `PassPipelineManager.apply_pipeline` -- that manager additionally calls `prog.validate()` before/after
    every pass, which is real MIL API surface (`Operation.get_flattened_inputs()` etc.) that this
    exporter's own "bespoke" workflow doesn't need to satisfy: it deliberately accepts hand-built
    `Program`s with synthetic, duck-typed submodule-dispatch ops standing in for ops MIL itself doesn't
    have (see `test_compiler.py`'s `MockOperation`), which are never meant to pass a real MIL validate().
    """
    for pass_name in _LOOM_PASS_NAMES:
        PASS_REGISTRY[pass_name](prog)
