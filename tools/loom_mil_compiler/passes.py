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
from coremltools.converters.mil.mil import Var
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY, register_pass
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource

from . import dialect  # noqa: F401  registers "loom_broadcast_to" etc. (mb.loom_broadcast_to, ...)
from .value_facts import static_ints, static_value


def _is_int(d):
    return isinstance(d, (int, np.integer))


def _scope_ctx_like(op):
    """A `mb.scope(...)` context copying `op`'s own TORCHSCRIPT_MODULE_NAME (if any) onto every op
    built within it, so a rewrite pass's replacement ops keep attributing to the right decoder
    layer/submodule instead of relying on positional adjacency for any future scope-based tooling (see
    EXPORT-IMPROVEMENT-BACKLOG.md item 2's two real mis-attribution bugs)."""
    scope = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME) if op.scopes else None
    if scope:
        return mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=list(scope)))
    return mb.scope()


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
        # whose var directly replaces `out_var`), and downstream scope-based tooling (debugging, any
        # future scope-partitioned discovery aid) needs every op it walks to carry the correct
        # TORCHSCRIPT_MODULE_NAME to attribute it to the right decoder layer -- relying on the two
        # intermediate ops merely landing in the right slice by positional adjacency would be exactly the
        # class of fragile mis-attribution EXPORT-IMPROVEMENT-BACKLOG.md item 2 already documents two real
        # bugs from.
        with _scope_ctx_like(tile_op):
            r1 = mb.reshape(x=pre_tile_x, shape=reshape1_shape, name=out_name + "_gqa_unsqueeze", before_op=tile_op)
            rep = mb.tile(x=r1, reps=repeat_reps, name=out_name + "_gqa_repeat", before_op=tile_op)
            r2 = mb.reshape(x=rep, shape=final_shape, name=out_name, before_op=tile_op)

        if not reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=out_var, new_var=r2,
        ):
            return False
        block.remove_ops([tile_op, reshape_op])
        return True


@register_pass(namespace="loom")
class normalize_matmul(AbstractGraphPass):
    """
    Rewrites `matmul(x, y, transpose_x=True, transpose_y=ty)` into the equivalent
    `matmul(transpose(x), y, transpose_x=False, transpose_y=ty)` -- EXPORT-ROADMAP.md R2a.

    `topology_ops.py`'s matmul rule table only ever composed a correct ggml lowering for
    `transpose_x=False` (see its own comment on `ggml_mul_mat`'s fixed contraction convention): every
    other combination fell through to `_op_matmul_unsupported`, documented there as "only
    transpose_x=False has been needed so far" rather than a real ceiling. Running this pass before
    `generate_graph_topology` ever walks the graph means every matmul it sees already has
    transpose_x=False, so the table's two existing composed rules -- (False, True) and (False, False)
    -- cover every matmul in the program, closing that gap without adding a third composition.

    A pure rewrite, not a new op: `transpose` and `matmul` (with transpose_x=False) are both already
    fully composed by `topology_ops.py`, so this only ever removes a guard, never adds one.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                # Already removed by an earlier rewrite in this same walk.
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "matmul":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        if not bool(static_value(op.inputs.get("transpose_x"), False)):
            return False
        x = op.inputs.get("x")
        y = op.inputs.get("y")
        if x is None or y is None or x.shape is None:
            return False
        rank = len(x.shape)
        if rank < 2:
            # matmul's own "promote 1-D x to a matrix" rule (see its docstring) never sets
            # transpose_x=True for a 1-D operand in practice -- guard rather than assume.
            return False
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        transpose_y = bool(static_value(op.inputs.get("transpose_y"), False))
        out_name = op.outputs[0].name

        with _scope_ctx_like(op):
            xt = mb.transpose(x=x, perm=perm, name=f"{out_name}_normalize_matmul_xt", before_op=op)
            new_out = mb.matmul(x=xt, y=y, transpose_x=False, transpose_y=transpose_y,
                                 name=out_name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_out,
        ):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class insert_explicit_broadcasts(AbstractGraphPass):
    """
    Rewrites an `add`/`mul` whose two operands need MUTUAL (different-axis) broadcasting -- each
    operand is size-1 on a DIFFERENT axis than the other, so neither is simply "the other's shape with
    some 1s" (`ggml_add`/`ggml_mul` only ever let ONE operand broadcast into the other's already-correct
    shape) -- into two explicit `loom_broadcast_to` ops feeding a plain `add`/`mul` whose operands are
    already at matching shape. EXPORT-ROADMAP.md R2a.

    This used to be a shape-string comparison the EMITTER itself performed (`exporter.py`'s add/mul
    case in `transpile_operation`), deciding whether to splice `REPEAT` nodes into the JSON node list
    by rendering both operands' shapes and checking for "1" vs. not. Running this as a real graph
    rewrite before `generate_graph_topology` ever walks the program means the emitter never has to look
    at either operand's shape at all: by the time it sees this op, both operands are already
    broadcast-compatible.

    First confirmed needed on SupertonicTTS's fractional-RoPE angle computation (`theta[d] *
    frac_pos[pos]`, ne=[32,1,1] * ne=[1,L,1] -> ne=[32,L,1], L dynamic) -- see `loom_broadcast_to`'s own
    docstring in `dialect.py` for why lowering to a real graph op (rather than conjuring a `REPEAT` JSON
    node ad hoc at emission time) is what lets `topology_ops.py`'s already-dynamic-shape-aware `REPEAT`
    lowering handle it unmodified.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type not in ("add", "mul"):
                continue
            self._try_transform(op, block)

    @staticmethod
    def _needs_broadcast(shape, out_shape):
        """True iff some axis of `shape` is a literal 1 while the SAME axis of `out_shape` isn't --
        the same "1 vs. not-1" test the old string-comparison code ran on rendered shape expressions,
        but directly on MIL's own shape tuples: a concrete-int axis is unambiguous either way, and a
        genuinely dynamic (symbolic) axis never renders as the literal string "1", so raw-shape ints
        and rendered-shape strings agree on every case this ever needs to distinguish."""
        return any(_is_int(s) and int(s) == 1 and not (_is_int(t) and int(t) == 1)
                    for s, t in zip(shape, out_shape))

    @classmethod
    def _try_transform(cls, op, block) -> bool:
        x = op.inputs.get("x")
        y = op.inputs.get("y")
        if x is None or y is None or x.shape is None or y.shape is None:
            return False
        out_var = op.outputs[0]
        if out_var.shape is None:
            return False
        out_shape = tuple(out_var.shape)
        if len(x.shape) != len(out_shape) or len(y.shape) != len(out_shape):
            return False
        if not (cls._needs_broadcast(x.shape, out_shape) and cls._needs_broadcast(y.shape, out_shape)):
            return False

        # `like=` the ORIGINAL other operand, not `out_var` itself -- `out_var` is produced by `op`,
        # which both new ops are inserted BEFORE, so using it here would be a data-dependency cycle.
        # `infer_type_with_broadcast(x, y)` gives the identical shape `op`'s own type inference already
        # computed for `out_var`, so this loses no information.
        node_tag = op.name
        with _scope_ctx_like(op):
            bx = mb.loom_broadcast_to(x=x, like=y, name=f"{node_tag}_bcast_x", before_op=op)
            by = mb.loom_broadcast_to(x=y, like=x, name=f"{node_tag}_bcast_y", before_op=op)
            builder_fn = mb.add if op.op_type == "add" else mb.mul
            new_out = builder_fn(x=bx, y=by, name=out_var.name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(
            anchor_op=op, old_var=out_var, new_var=new_out,
        ):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class canonicalize_replicate_pad(AbstractGraphPass):
    """
    Rewrites a `pad(mode="replicate")` op into a `loom_replicate_pad` op -- EXPORT-ROADMAP.md R2.

    First (and so far only) needed by SupertonicTTS's `ConvNextBlock` (used by every encoder/decoder in
    that model), which pads via `nn.functional.pad(x, pad, mode="replicate")` before every depthwise
    conv. `topology_ops.py`'s `pad` rule used to decide, at emission time, whether `mode` was
    "replicate" and if so compose VIEW/REPEAT/CONCAT inline; this pass makes that decision once, before
    emission, so the emitter just dispatches on op type like any other.

    Validates the same invariants `topology_ops.py`'s old inline code did -- pad values must be
    compile-time constants, and only the fastest-varying (last, MIL-order) axis may have a non-zero
    replicate pad (the only shape ggml's composed-from-primitives approach here can express) -- raising
    the same errors that code raised, just earlier (right after `ct.convert()`, not mid-emission).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "pad":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        mode = static_value(op.inputs.get("mode"), "constant")
        if mode != "replicate":
            return False
        x = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
        if x is None or x.shape is None:
            raise NotImplementedError(
                f"pad op '{op.name}' has mode='replicate' but an input with no known rank."
            )
        pad_vals = static_ints(op.inputs.get("pad"))
        if pad_vals is None or len(pad_vals) % 2 != 0:
            raise NotImplementedError(
                f"pad op '{op.name}' has mode='replicate' with a non-constant or odd-length 'pad' "
                "input, which this exporter doesn't support."
            )
        n_padded = len(pad_vals) // 2
        rank = len(x.shape)
        lp0 = rp0 = 0
        for i in range(n_padded):
            mil_axis = rank - n_padded + i
            lp, rp = pad_vals[2 * i], pad_vals[2 * i + 1]
            if lp == 0 and rp == 0:
                continue
            if mil_axis != rank - 1:
                raise NotImplementedError(
                    f"pad op '{op.name}' pads MIL axis {mil_axis} (non-zero {lp}/{rp}) with "
                    "mode='replicate', but this exporter only supports replicate-padding the "
                    "fastest-varying axis (ne[0]/MIL's last axis) -- padding any other axis needs a "
                    "new C++ primitive first."
                )
            lp0, rp0 = lp, rp

        out_var = op.outputs[0]
        if lp0 == 0 and rp0 == 0:
            # A genuine identity pad (every entry zero) -- just alias the op away entirely.
            if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=x):
                return False
            block.remove_ops([op])
            return True

        with _scope_ctx_like(op):
            new_out = mb.loom_replicate_pad(x=x, lp=lp0, rp=rp0, name=out_var.name, before_op=op)
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=new_out):
            return False
        block.remove_ops([op])
        return True


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]


@register_pass(namespace="loom")
class canonicalize_conv_transpose_dw(AbstractGraphPass):
    """
    Rewrites a depthwise (`groups == in_channels == out_channels`) `conv_transpose` into a
    `loom_conv_transpose_dw` op -- EXPORT-ROADMAP.md R2.

    First (and so far only) needed by Kokoro's `AdainResBlk1d` upsample "pool" (`ConvTranspose1d
    (kernel=3, stride=2, groups=dim_in, padding=1, output_padding=1)`), also reused by StyleTTS2's
    driver. `topology_ops.py`'s `conv_transpose` rule used to decide, at emission time, whether `groups`
    made this depthwise and if so compose the zero-stuff-then-depthwise-conv identity inline; this pass
    makes that decision once, before emission.

    Validates the same invariants `topology_ops.py`'s old inline code did -- only a true 1D depthwise
    case (`is_2d=False`, `groups == in_channels`, one output channel per group), zero dilation, zero pad
    (every depthwise conv_transpose this exporter has seen traces with `pad=[0,0]`, deferring any real
    crop to a separate downstream `slice_by_index`), and a compile-time-constant weight (needed to flip
    the kernel) -- raising the same errors that code raised for anything else, just earlier (right after
    `ct.convert()`, not mid-emission). A non-depthwise (`groups == 1`) `conv_transpose` is untouched,
    left for `topology_ops.py`'s own `conv_transpose` rule exactly as before.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "conv_transpose":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        groups = static_value(op.inputs.get("groups"), 1)
        g_val = int(_as_list(groups)[0])
        if g_val == 1:
            return False

        pad_type = static_value(op.inputs.get("pad_type"), "valid")
        if pad_type not in ("valid", "custom"):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has pad_type='{pad_type}', which this exporter "
                "doesn't support (only 'valid' and a 'custom' symmetric-crop composition exist)."
            )
        strides_list = _as_list(static_value(op.inputs.get("strides"), [1]))
        is_2d = len(strides_list) == 2
        if is_2d and pad_type == "custom":
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is 2D with pad_type='custom' -- only the 1D "
                "crop composition has been needed/written so far."
            )
        dilations = _as_list(static_value(op.inputs.get("dilations"), [1]))
        if any(int(d) != 1 for d in dilations):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has non-unit 'dilations' {dilations!r}, which "
                "this exporter doesn't support."
            )

        x = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
        weight = op.inputs.get("weight")
        if x is None or x.shape is None or weight is None or weight.shape is None:
            return False
        in_channels = int(x.shape[1])
        out_per_group = int(weight.shape[1])
        if is_2d or g_val != in_channels or out_per_group != 1:
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' has groups={g_val} (in_channels={in_channels}, "
                f"out_channels/group={out_per_group}) -- only a true 1D depthwise case "
                "(groups == in_channels == out_channels) is composed; anything else has no "
                "ggml-side implementation yet."
            )
        pad_list = _as_list(static_value(op.inputs.get("pad"), [0]))
        if any(int(p) != 0 for p in pad_list):
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is depthwise with non-zero pad={pad_list!r} -- "
                "every depthwise conv_transpose this exporter has seen traces with pad=[0,0] "
                "(deferring any real crop to a separate downstream slice_by_index op); this "
                "composition doesn't know how to fold a non-zero pad in directly."
            )
        if static_value(weight) is None:
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' is depthwise but its weight isn't a resolved "
                "constant -- this composition needs to flip the kernel at export time."
            )

        s0 = int(strides_list[0])
        bias = op.inputs.get("bias")
        out_name = op.outputs[0].name
        with _scope_ctx_like(op):
            new_out = mb.loom_conv_transpose_dw(x=x, weight=weight, bias=bias, stride=s0,
                                                 name=out_name, before_op=op)
        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class lower_stack(AbstractGraphPass):
    """
    Rewrites `stack(values, axis)` into `concat([expand_dims(v, axes=[axis]) for v in values], axis)`
    -- EXPORT-ROADMAP.md R2. Unlike every other pass in this module, this introduces no new dialect op:
    `expand_dims` and `concat` are both already-real MIL ops with their own full, general
    `topology_ops.py` rules (`reshape`/`expand_dims` and `concat`), so this is a pure lowering -- once it
    runs, `topology_ops.py` no longer needs a dedicated `stack` composition at all, and gets the exact
    same N-ary CONCAT-chaining logic `concat` already has, instead of a second, parallel copy of it.

    First (and so far only) needed by a hand-rolled conv-based STFT's real/imag parts
    (`torch.stack([real, imag], dim=-1)`, seen when a model computes its DFT via CONV_1D kernels
    directly rather than `torch.stft`, which decomposes differently via coremltools' own
    `lower_complex_dialect_ops`).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "stack":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        values = op.inputs.get("values")
        if not values:
            return False
        real_values = [v for v in values if isinstance(v, Var)]
        if not real_values:
            return False
        out_var = op.outputs[0]
        if out_var.shape is None:
            return False
        out_rank = len(out_var.shape)
        axis_val = int(static_value(op.inputs.get("axis"), 0))
        axis = axis_val + out_rank if axis_val < 0 else axis_val

        with _scope_ctx_like(op):
            if len(real_values) == 1:
                # A single real operand -- still a genuine rank-increasing op, not an identity, so
                # this is the op's own real output name directly rather than an intermediate one.
                new_out = mb.expand_dims(x=real_values[0], axes=[axis], name=out_var.name, before_op=op)
            else:
                expanded = [
                    mb.expand_dims(x=v, axes=[axis], name=f"{out_var.name}_stack_unsq_{i}", before_op=op)
                    for i, v in enumerate(real_values)
                ]
                new_out = mb.concat(values=expanded, axis=axis, name=out_var.name, before_op=op)

        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=out_var, new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class lower_reduce_mean(AbstractGraphPass):
    """
    Rewrites a single-axis `reduce_mean` into whichever of two real ops its reduced axis's countability
    allows -- EXPORT-ROADMAP.md R2:

    * a statically-known reduction count -> `reduce_sum` (already a real, general MIL op with its own
      `topology_ops.py` rule) followed by `loom_scale(n)` (dividing by `n`; see that op's own docstring
      for why it carries `n` rather than the pre-divided `1/n`);
    * a run-time-only count, but on the fastest-varying (last, MIL-order) axis -> `loom_mean`, which
      `ggml_mean` can reduce natively (it supplies its own count at run time);
    * anything else (a run-time-only count on any other axis, or a genuine multi-axis reduction) is
      unrepresentable and raises -- the same errors `topology_ops.py`'s old two guards raised, just
      earlier (right after `ct.convert()`, not mid-emission).

    This is what makes all three outcomes explicit and pass-driven, rather than two of them being
    `topology_ops.py` guards and the third an unstated fall-through to the generic OP_MAP path (which
    happened to already do the right thing for the ne[0] case, but only because nothing else claimed
    "reduce_mean" first).
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._rewrite_block(f)

    @block_context_manager
    def _rewrite_block(self, block):
        for op in list(block.operations):
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._rewrite_block(b)
            if op.op_type != "reduce_mean":
                continue
            self._try_transform(op, block)

    @staticmethod
    def _try_transform(op, block) -> bool:
        x = op.inputs.get("x")
        axes_val = static_ints(op.inputs.get("axes"))
        if x is None or x.shape is None or axes_val is None or len(axes_val) != 1:
            raise NotImplementedError(
                f"reduce_mean op '{op.name}': only a single reduction axis is supported "
                f"(got axes={static_value(op.inputs.get('axes'))!r}); a genuine multi-axis case "
                "(e.g. GroupNorm, see group_norm_op.py) needs its own composition."
            )
        rank = len(x.shape)
        axis = axes_val[0]
        torch_axis = axis + rank if axis < 0 else axis
        if not (0 <= torch_axis < rank):
            return False
        ne_axis = rank - 1 - torch_axis
        n_raw = x.shape[torch_axis]
        keep_dims = bool(static_value(op.inputs.get("keep_dims"), False))
        out_name = op.outputs[0].name

        if _is_int(n_raw):
            n = int(n_raw)
            with _scope_ctx_like(op):
                summed = mb.reduce_sum(x=x, axes=[axis], keep_dims=keep_dims,
                                        name=f"{out_name}_rmean_sum", before_op=op)
                new_out = mb.loom_scale(x=summed, n=n, name=out_name, before_op=op)
        elif ne_axis == 0:
            with _scope_ctx_like(op):
                new_out = mb.loom_mean(x=x, keep_dims=keep_dims, name=out_name, before_op=op)
        else:
            raise NotImplementedError(
                f"reduce_mean op '{op.name}': reducing ne axis {ne_axis} over a count that is only "
                "known at run time has no composition here. REDUCE_SUM + SCALE needs the count at "
                "export time, and ggml_mean supplies its own count only for ne[0]. A dynamic count on "
                "another axis needs its own composition (see loom_group_norm's custom-op bridge for "
                "that case)."
            )

        if not block.try_replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_out):
            return False
        block.remove_ops([op])
        return True


@register_pass(namespace="loom")
class fuse_loom_attention(AbstractGraphPass):
    """
    Replaces each traced scaled-dot-product-attention block with one `loom_fused_attention` op, which
    `topology_ops.py` lowers to the engine's `ATTENTION` primitive -- the only node type that can reach
    a KV cache (KV-CACHE.md stage 2).

    **Opt-in, and that is a correctness requirement rather than caution.** The pattern below is generic
    SDPA, so it matches VITS's/Kokoro's/StyleTTS2's self-attention just as well as a causal LM's -- and
    those are non-autoregressive, so giving them an ATTENTION node (whose `kv_cache` attr defaults to
    TRUE) would hand them a persistent cache they must never have. Only the causal-LM family sets
    `fuse_attention=True`; every other model's topology is untouched, which is also what keeps their
    byte-identity gates meaningful.

    The window, anchored on `softmax` and confirmed against a real trace (a randomly-initialised
    2-layer Llama and, at full size, Qwen3-0.6B -- both produce it identically):

        mul       (q, 1/sqrt(head_dim))            -- scale folded onto Q by HF, not onto the scores
        matmul    (q_scaled, k, transpose_y=True)  -- Q @ K^T
        add       (scores, mask)                   -- mask is a slice_by_index of the graph input
        softmax   (axis=-1)
        matmul    (probs, v)
        transpose (perm=[0, 2, 1, 3])              -- [b, h, s, d] -> [b, s, h, d]
        reshape                                    -- -> [b, s, h*d]

    Both trailing ops are absorbed, because `op_attention` already returns the flattened
    `[n_embd, n_tokens]` context; stopping at the second matmul would leave the op's declared MIL type
    disagreeing with what the engine actually computes.

    **`layer` is assigned in attention-block occurrence order, NOT by torch module index**, and the
    distinction is load-bearing. The index addresses a cache slot, and the cache has one slot per
    ATTENTION block -- so for an architecture that interleaves non-attention layers (LFM2's conv
    blocks), the dense occurrence index is the correct one and the module index would address past the
    end of the cache. It also means `loom.n_layer` for cache sizing is the count of attention blocks,
    which for a uniform decoder like Qwen3 is the same number and for LFM2 is not.

    Anything that does not match is left exactly as it was: an unfused block still exports and still
    runs, just without a cache. A partial match must never half-rewrite.
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._next_layer = 0
            self._fuse_block(f)

    @block_context_manager
    def _fuse_block(self, block):
        for op in list(block.operations):
            # Same guard as fuse_gqa_repeat_kv: `getattr(..., block)` because the bespoke workflow's
            # duck-typed MockOperations carry no `enclosing_block`, and "attribute missing" must read as
            # "still present" rather than as "already removed".
            if getattr(op, "enclosing_block", block) is None:
                continue
            for b in op.blocks:
                self._fuse_block(b)
            if op.op_type != "softmax":
                continue
            if self._try_to_transform(op, block):
                self._next_layer += 1

    @staticmethod
    def _binary_operands(op, want_op_type):
        """`op`'s two operands as (the one produced by `want_op_type`, the other), or None. Written
        order-agnostically because `add`'s operands are commutative and their traced order is not a
        promise -- keying on position would make this pass architecture-sensitive for no reason."""
        x, y = op.inputs.get("x"), op.inputs.get("y")
        if x is None or y is None:
            return None
        if x.op is not None and x.op.op_type == want_op_type:
            return x, y
        if y.op is not None and y.op.op_type == want_op_type:
            return y, x
        return None

    @staticmethod
    def _pre_gqa_repeat(var):
        """The un-repeated tensor behind `fuse_gqa_repeat_kv`'s `reshape -> tile -> reshape` triple, or
        None if `var` is not the output of one.

        Matched structurally rather than by the `_gqa_unsqueeze`/`_gqa_repeat` names that pass gives its
        ops: a name is a debugging aid, and keying on one would make this silently stop working the day
        those strings change.
        """
        reshape_out = var.op
        if reshape_out is None or reshape_out.op_type != "reshape":
            return None
        tile_var = reshape_out.inputs.get("x")
        if tile_var is None or tile_var.op is None or tile_var.op.op_type != "tile":
            return None
        reps = tile_var.op.inputs.get("reps")
        if reps is None or reps.val is None:
            return None
        # repeat_kv() only ever grows ONE axis (the KV-head one); anything else is a different tile.
        if sum(1 for r in np.array(reps.val).ravel() if int(r) != 1) != 1:
            return None
        inner = tile_var.op.inputs.get("x")
        if inner is None or inner.op is None or inner.op.op_type != "reshape":
            return None
        src = inner.op.inputs.get("x")
        if src is None or src.shape is None or len(src.shape) != 4:
            return None
        return src

    @staticmethod
    def _mask_kv_slice_source(mask_var):
        """The tensor behind HF's `mask[..., :kv_len]` slice, or `mask_var` unchanged.

        The traced mask does not reach the attention block directly: transformers slices it to the
        current KV length on the way in, which comes out of the converter as
        `slice_by_index(attention_mask, begin=[0,0,0,0], end=[...], end_mask=[T,T,T,False])` -- full
        extent on every axis but the last, and the last cut to a computed `kv_len`. With no cache in the
        trace, `kv_len == seq_len`, so it is an identity slice that exists only because the traced model
        expected to be given a mask wider than it needed.

        A cached step is the case that slice was written for, and the driver now builds the mask at
        exactly `[n_tokens, n_kv]` (`loom.causal_mask(n_tokens, n_past)`) -- so the slice is not merely
        redundant, it is *wrong*: its extents were baked at trace time and would cut a decode step's
        mask back to the prefill width. Bypassing it is what lets the mask input be declared `["n_kv",
        "n_tokens"]` at all (KV-CACHE.md 3.2), because the retyping is only sound while the input's own
        consumers are all fused-attention nodes -- and a surviving slice is a consumer that is not.

        Every guard bails to "leave it alone", the same rule the rest of this pass follows: an unmatched
        shape leaves a graph that still exports, prefill-only, rather than one rewritten halfway.
        """
        op = getattr(mask_var, "op", None)
        if op is None or op.op_type != "slice_by_index":
            return mask_var
        src = op.inputs.get("x")
        if src is None or src.shape is None or mask_var.shape is None:
            return mask_var
        rank = len(src.shape)
        if rank != len(mask_var.shape) or rank < 2:
            return mask_var

        def mask_bits(name, default):
            var = op.inputs.get(name)
            if var is None or var.val is None:
                return [default] * rank
            bits = list(np.array(var.val).ravel())
            return bits if len(bits) == rank else None

        # Nothing may be squeezed away, nothing strided, and every axis but the last must be taken
        # whole: `begin` at 0 (or ignored via begin_mask) and `end` ignored via end_mask.
        squeeze = mask_bits("squeeze_mask", False)
        stride = mask_bits("stride", 1)
        begin_mask = mask_bits("begin_mask", False)
        end_mask = mask_bits("end_mask", False)
        if squeeze is None or stride is None or begin_mask is None or end_mask is None:
            return mask_var
        if any(bool(b) for b in squeeze) or any(int(st) != 1 for st in stride):
            return mask_var
        begin_var = op.inputs.get("begin")
        begin = list(np.array(begin_var.val).ravel()) if begin_var is not None and begin_var.val is not None else None
        for axis in range(rank):
            if not bool(begin_mask[axis]) and (begin is None or int(begin[axis]) != 0):
                return mask_var
            if axis < rank - 1 and not bool(end_mask[axis]):
                return mask_var
        # The last axis IS sliced (that is the whole point); if it were not, this is some other slice.
        if bool(end_mask[rank - 1]):
            return mask_var
        return src

    def _strip_gqa_repeat(self, k_var, v_var, q_var):
        """`(k, v)` with HF's `repeat_kv()` expansion undone when it is safe to do so, else unchanged.

        `op_attention` reads `n_head_kv` straight off K's own shape and lets `ggml_mul_mat`'s broadcast
        map query head `i` to KV head `i // ratio` -- integer division, i.e. exactly the interleaved
        correspondence `repeat_kv()` materializes (see `fuse_gqa_repeat_kv`'s docstring on why that is
        interleaved and not block-tiled). So attending against the un-repeated K/V is the same
        arithmetic, and it HALVES Qwen3-0.6B's cache: 16 stored heads become the 8 the checkpoint
        actually has.

        Correctness never depends on this. Keeping the repeat is numerically identical, merely wasteful,
        which is why every guard below bails to "leave it alone" rather than raising -- and why K and V
        are stripped only TOGETHER and only to the same head count. Stripping one and not the other
        would leave the cache's K and V widths disagreeing, which no later check would catch.
        """
        k_src, v_src = self._pre_gqa_repeat(k_var), self._pre_gqa_repeat(v_var)
        if k_src is None or v_src is None:
            return k_var, v_var
        n_head, n_head_kv = q_var.shape[1], k_src.shape[1]
        if not isinstance(n_head, int) or not isinstance(n_head_kv, int):
            return k_var, v_var
        if n_head_kv != v_src.shape[1] or n_head_kv <= 0 or n_head % n_head_kv != 0:
            return k_var, v_var
        return k_src, v_src

    def _try_to_transform(self, softmax_op, block) -> bool:
        axis = softmax_op.inputs.get("axis")
        if axis is None or axis.val is None or int(axis.val) not in (-1, 3):
            return False

        scores = softmax_op.inputs.get("x")
        if scores is None or scores.op is None or scores.op.op_type != "add":
            return False
        add_op = scores.op
        operands = self._binary_operands(add_op, "matmul")
        if operands is None:
            return False
        qk_var, mask_var = operands
        qk_op = qk_var.op

        # Q @ K^T, and K must NOT be pre-transposed by a separate op -- `transpose_y` is how the traced
        # graph spells it, and a False here means this is some other matmul that happens to feed a
        # softmax.
        transpose_y = qk_op.inputs.get("transpose_y")
        transpose_x = qk_op.inputs.get("transpose_x")
        if transpose_y is None or transpose_y.val is None or not bool(transpose_y.val):
            return False
        if transpose_x is not None and transpose_x.val is not None and bool(transpose_x.val):
            return False

        q_var = qk_op.inputs.get("x")
        k_var = qk_op.inputs.get("y")
        if q_var is None or k_var is None:
            return False

        # The scale HF folds onto Q. Recovered rather than recomputed from head_dim: a model with a
        # non-default scale (or none) is then still correct, and `scale=1.0` with the `mul` left in
        # place is a valid outcome rather than a silent 1/sqrt(d) that was never in the graph.
        scale = 1.0
        if q_var.op is not None and q_var.op.op_type == "mul":
            factors = (q_var.op.inputs.get("x"), q_var.op.inputs.get("y"))
            const_side = [f for f in factors if f is not None and f.val is not None and f.shape in ((), (1,))]
            other_side = [f for f in factors if f is not None and f.val is None]
            if len(const_side) == 1 and len(other_side) == 1:
                scale = float(np.array(const_side[0].val).ravel()[0])
                q_var = other_side[0]

        # Down the graph: probs @ V, then the transpose+reshape back to [b, seq, n_embd].
        probs_children = list(softmax_op.outputs[0].child_ops)
        if len(probs_children) != 1 or probs_children[0].op_type != "matmul":
            return False
        av_op = probs_children[0]
        if av_op.inputs.get("x") is not softmax_op.outputs[0]:
            return False
        v_var = av_op.inputs.get("y")
        if v_var is None:
            return False
        for flag in ("transpose_x", "transpose_y"):
            f = av_op.inputs.get(flag)
            if f is not None and f.val is not None and bool(f.val):
                return False

        av_children = list(av_op.outputs[0].child_ops)
        if len(av_children) != 1 or av_children[0].op_type != "transpose":
            return False
        transpose_op = av_children[0]
        perm = transpose_op.inputs.get("perm")
        if perm is None or perm.val is None or list(np.array(perm.val).ravel()) != [0, 2, 1, 3]:
            return False

        transpose_children = list(transpose_op.outputs[0].child_ops)
        if len(transpose_children) != 1 or transpose_children[0].op_type != "reshape":
            return False
        reshape_op = transpose_children[0]
        out_var = reshape_op.outputs[0]
        if out_var.shape is None or len(out_var.shape) != 3:
            return False

        # Undo HF's repeat_kv() where it is safe, so the cache stores the checkpoint's real KV heads
        # rather than the expanded ones (KV-CACHE.md 2.3). Purely a size win; see _strip_gqa_repeat.
        k_var, v_var = self._strip_gqa_repeat(k_var, v_var, q_var)

        # Attend against the mask the driver actually builds, not the trace-width slice of it
        # (KV-CACHE.md 3.2). Unlike the GQA strip above this one is a correctness requirement for a
        # cached step, not a size win -- see _mask_kv_slice_source.
        mask_var = self._mask_kv_slice_source(mask_var)

        # Every rank check the op's own type_inference would make, made here first -- a pass that raises
        # from inside mb.loom_fused_attention leaves the block half-rewritten, whereas bailing here
        # leaves a graph that still exports.
        for var in (q_var, k_var, v_var):
            if var.shape is None or len(var.shape) != 4:
                return False
        if not isinstance(q_var.shape[1], int) or not isinstance(v_var.shape[3], int):
            return False

        with _scope_ctx_like(softmax_op):
            fused = mb.loom_fused_attention(
                q=q_var, k=k_var, v=v_var, mask=mask_var,
                scale=np.float32(scale), layer=np.int32(self._next_layer),
                name=out_var.name, before_op=qk_op,
            )

        if not reshape_op.enclosing_block.try_replace_uses_of_var_after_op(
            anchor_op=reshape_op, old_var=out_var, new_var=fused,
        ):
            return False
        # Only the ops this fusion definitively subsumed. Everything upstream (the q `mul`, the mask's
        # own slice chain) is left to dead_code_elimination, which is the pass that knows whether some
        # other consumer still needs it -- this one does not.
        block.remove_ops([reshape_op, transpose_op, av_op, softmax_op, add_op, qk_op])
        return True


_LOOM_PASS_NAMES = [
    "loom::fuse_gqa_repeat_kv",
    "loom::normalize_matmul",
    "loom::insert_explicit_broadcasts",
    "loom::canonicalize_replicate_pad",
    "loom::canonicalize_conv_transpose_dw",
    "loom::lower_stack",
    "loom::lower_reduce_mean",
    "common::dead_code_elimination",
]

# Runs only when the caller asks for it (KV-CACHE.md decision 4). Placed after the GQA fusion, which
# normalizes `repeat_kv()` into a reshape/tile/reshape triple the attention fusion can see past, and
# before dead_code_elimination, which is what removes the subgraph the fusion orphans.
_LOOM_ATTENTION_PASS_NAME = "loom::fuse_loom_attention"


def apply_loom_mil_passes(prog, fuse_attention: bool = False) -> None:
    """
    Runs Loom's own MIL->MIL rewrite passes -- GQA `repeat_kv()` fusion, matmul transpose_x
    normalization (R2a), mutual-broadcast insertion (R2a), replicate-pad and depthwise-conv_transpose
    canonicalization, `stack` and `reduce_mean` lowering (R2) -- plus `common::dead_code_elimination`
    over `prog` in place. Must run before any topology/driver generation sees `prog` --
    `common::dead_code_elimination` is what actually removes each rewrite's now-orphaned dependency
    chain (the original tile/reshape idiom, a stale `transpose_x` bool operand, etc).

    Invokes each registered pass callable directly (`PASS_REGISTRY[name](prog)`) rather than going through
    `PassPipelineManager.apply_pipeline` -- that manager additionally calls `prog.validate()` before/after
    every pass, which is real MIL API surface (`Operation.get_flattened_inputs()` etc.) that this
    exporter's own "bespoke" workflow doesn't need to satisfy: it deliberately accepts hand-built
    `Program`s with synthetic, duck-typed submodule-dispatch ops standing in for ops MIL itself doesn't
    have (see `test_compiler.py`'s `MockOperation`), which are never meant to pass a real MIL validate().
    """
    for pass_name in _LOOM_PASS_NAMES:
        if pass_name == "common::dead_code_elimination" and fuse_attention:
            # Must land between the rewrites and the DCE that cleans up after them, which is why this
            # is spliced here rather than appended to the list.
            PASS_REGISTRY[_LOOM_ATTENTION_PASS_NAME](prog)
        PASS_REGISTRY[pass_name](prog)
