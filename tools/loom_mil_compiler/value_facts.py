"""The exporter's single answer-set for "what is this MIL Var's compile-time value?".

Every op lowering in `topology_ops.py` needs some version of that question -- a `tile`'s `reps`, a
`pad`'s pad amounts, a `reshape`'s target shape, a `slice_by_index`'s per-axis begin/end. Before this
module those answers were re-derived per callsite, in two flavours that had each been reinvented several
times over:

* the *literal* read -- `x.val if x is not None and hasattr(x, "val") and x.val is not None else default`
  -- written out longhand at ~40 places, each free to get the None-handling subtly wrong; and
* the *derived* read -- resolving a value that MIL did NOT constant-fold because it depends on a live
  activation's shape (`gather(shape(x), i)`, arithmetic over such gathers, a `concat` of them building a
  reshape's shape input). Five mutually-recursive helpers grew on the exporter for this, each written for
  one callsite and then partially reused by the next.

Both live here now, behind one object the exporter builds once (`exporter.facts`) and every handler
reads through. The derived family is **memoized per Var**, which is what makes "resolve once, up front"
true in practice rather than just tidier-looking: `scalar_expr` recurses into *both* operands of every
arithmetic op it walks, so on a diamond-shaped expression tree (entirely ordinary -- VITS's
`end = start + 2*length - 1` reaches the same `length` gather down two paths) the un-memoized version
re-walked shared subtrees exponentially. The memo also makes the answers *stable by construction*: two
callsites asking about the same Var can no longer disagree, which is exactly the failure mode the
`slice_axis_value` docstring below records having actually happened.

Nothing here mutates the MIL program or the exporter; it is pure derivation over an SSA graph that is
immutable by the time topology generation runs, so caching on the exporter (rather than per
`generate_graph_topology` call) is safe and lets the cache survive across a model's several topologies.
"""
import numpy as np
from coremltools.converters.mil.mil import Var

# MIL arithmetic ops `scalar_expr` can fold or render as a SymbolEnv expression, and the operator each
# one prints as. `floor_div` shares `real_div`'s slash because it is wrapped in an explicit floor().
_ARITH_OPS = {"add": "+", "sub": "-", "mul": "*", "real_div": "/", "floor_div": "/"}


def static_value(var, default=None):
    """`var`'s compile-time-constant value exactly as MIL stores it (array, scalar, or str), else
    `default`. This one function replaces the
    `x.val if x is not None and hasattr(x, "val") and x.val is not None else default` idiom that was
    written out longhand at every op handler that needed a static operand."""
    if var is None:
        return default
    val = getattr(var, "val", None)
    return default if val is None else val


def static_array(var):
    """`var`'s constant value as a numpy array, else None."""
    val = static_value(var)
    return None if val is None else np.asarray(val)


def static_scalar(var, default=None):
    """`var`'s constant value as a single Python scalar (its first element if it holds several),
    else `default`."""
    arr = static_array(var)
    if arr is None or arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def static_ints(var):
    """`var`'s constant value as a flat list of Python ints, else None."""
    arr = static_array(var)
    return None if arr is None else [int(x) for x in arr.reshape(-1)]


def is_const_producer(var):
    """True iff `var` is produced by a literal `const` op -- a structural question about the graph,
    distinct from `static_value`'s "does this have a folded value at all" (a `shape` op's output can
    have the latter without the former)."""
    return var is not None and getattr(var, "op", None) is not None and var.op.op_type == "const"


class ValueFacts:
    """Memoized compile-time-value queries over one exporter's MIL program."""

    def __init__(self, exporter):
        self.exporter = exporter
        # Keyed by id(var), storing (var, answer) so the referenced Var stays alive and its id can
        # never be recycled onto a different object while the entry is cached.
        self._scalar_expr = {}
        self._gather_shape = {}
        self._range_scalar = {}
        self._slice_axis = {}
        self._reshape_shape = {}

    # -- literal statics ---------------------------------------------------------------------------
    # Bound as methods too, so a handler holding only `self.facts` still reaches them; the real
    # definitions are the module-level functions below, which need no instance.

    value = staticmethod(static_value)
    array = staticmethod(static_array)
    scalar = staticmethod(static_scalar)
    ints = staticmethod(static_ints)
    is_const_producer = staticmethod(is_const_producer)

    # -- derived values ----------------------------------------------------------------------------
    # Values MIL did NOT fold, because they depend on a live activation's shape. Memoized per Var.

    def gather_shape_value(self, var):
        """
        If `var` is exactly `gather(shape(real_tensor), index)` with a constant `index`, derive its
        real symbolic value via `_infer_dynamic_dim_expr` -- needed because `op_range_1d`
        (src/ops/primitives_mil.cpp) can only read a dynamic "end"/"start" bound from a Var's own
        already-computed `.data` at GRAPH-BUILD time, and a `gather(shape(x), ...)` chain's real value
        only exists after `ggml_backend_graph_compute()` runs, strictly AFTER GraphBuilder::build()
        finishes -- so `in[N]->data` is architecturally always null for this exact pattern, silently
        tripping op_range_1d's own "dynamic sequence length fallback" (a hardcoded `n_tokens`)
        regardless of anything this exporter does at the JSON-shape-attribute level. Confirmed as the
        true root cause of Conformer-CTC's length-tracking bug (BACKLOG.md) after every shape-string fix
        upstream of it (conv/matmul/tile/elementwise/expand_dims derivation) still left the bug in place.
        Returns None (meaning: fall back to passing `var` as a normal graph input) if `var` doesn't match
        this exact shape, so every other RANGE_1D use (a real per-batch length input, etc.) is untouched.
        """
        if var is None:
            return None
        key = id(var)
        if key in self._gather_shape:
            return self._gather_shape[key][1]
        result = self._gather_shape_value_uncached(var)
        self._gather_shape[key] = (var, result)
        return result

    def _gather_shape_value_uncached(self, var):
        while var is not None and var.op is not None and var.op.op_type == "cast":
            # A `gather(shape(x), idx)` result routinely gets cast (e.g. i32 -> fp16) before landing in
            # a `concat` that builds a reshape's "shape" input -- peel those off first, same "pure
            # alias" treatment `_infer_dynamic_dim_expr`'s own unary-passthrough case gives `cast`.
            var = var.op.inputs.get("x")
        if var is None or var.op is None or var.op.op_type != "gather":
            return None
        shape_vec_var = var.op.inputs.get("x") or var.op.inputs.get("params")
        indices_var = var.op.inputs.get("indices")
        if shape_vec_var is None or shape_vec_var.op is None or shape_vec_var.op.op_type != "shape":
            return None
        if self.value(indices_var) is None:
            return None
        real_var = shape_vec_var.op.inputs.get("x")
        if real_var is None or real_var.shape is None:
            return None
        idx = int(self.array(indices_var).reshape(-1)[0])
        real_rank = len(real_var.shape)
        torch_axis = idx + real_rank if idx < 0 else idx
        if not (0 <= torch_axis < real_rank):
            return None
        if torch_axis == 0 and real_rank >= 2:
            # `x.shape[0]` (or `x.size(0)`) on a rank>=2 activation is the canonical PyTorch idiom for
            # reading BATCH SIZE -- and this whole exporter's architecture only ever targets batch=1
            # models (every declared model input's own batch axis is a literal 1, stated repeatedly
            # elsewhere in the exporter, e.g. the `tile` case). Short-circuiting straight to "1" here
            # is far more robust than walking the full producer chain (which, for a deep Conformer-CTC
            # encoder layer, runs through `layer_norm`/`linear`/`matmul`/`add` dozens of times before
            # bottoming out at a real input) AND avoids a real correctness gap: without this, the walk
            # frequently gives up at some not-yet-handled op type along that long chain and silently
            # falls back to the SAME "n_tokens" string a genuine batch=1 axis would never actually be,
            # corrupting whatever RESHAPE/RANGE_1D consumes this value. A genuinely non-1 batch axis
            # would surface as a numerical mismatch against the reference model, not a syntax error here.
            return "1"
        return self.exporter._infer_dynamic_dim_expr(real_var, torch_axis)

    def scalar_expr(self, v, _seen=None):
        """
        General best-effort derivation of a SCALAR (0-d/1-element) Var's real symbolic value, walking
        through cast/squeeze aliasing and +-*/floor_div arithmetic over already-resolvable operands --
        needed for scalars computed via a real EXPRESSION over a gather-derived value (e.g.
        RelPositionalEncoding's `start_pos = center_pos - gather(shape(x), 1)`), which
        `gather_shape_value`'s narrower "exactly gather(shape(x), idx)" pattern match can't
        see through at all (it only recognizes gather itself, not gather wrapped in surrounding
        arithmetic). Confirmed needed on Conformer-CTC's positional-encoding table crop: `range_scalar`
        alone left `start_pos`/`end_pos` (both real, resolvable expressions once cast/squeeze/sub
        are walked through) as `None`, causing `slice_by_index`'s own "resolve begin/end directly" case
        to give up on this axis entirely. Returns an int/float when every operand is a compile-time
        literal, else a string SymbolEnv expression, else None if any step can't be resolved.

        `_seen` is accepted and threaded through but unused -- MIL/SSA graphs are acyclic by
        construction (an op's inputs always name EARLIER-defined vars, never itself), so a genuine
        infinite loop here is architecturally impossible, and treating "already visited" as a failure
        was a real bug rather than harmless defensiveness: a DIAMOND dependency (the same upstream var
        reached via two different operand paths) is completely ordinary in an arithmetic expression
        tree -- confirmed on VITS's `end = start + 2*length - 1`, where `start` and the `2*length` term
        both independently reference the same `length`-derived `gather` var. The SECOND reference used
        to hit the visited-set and silently return None, even though nothing about it was actually
        unresolvable -- `slice_by_index`'s "end" bound came back `None` and fell back to the axis's full
        (unsliced) extent, corrupting the sliced relative-position table (silently ~34x too long at a
        real T=62) while `begin` (whose OWN resolution never revisits `gather_0` a second time) looked
        completely fine. The memo below is what makes revisiting cheap instead of exponential.
        """
        if v is None or not isinstance(v, Var):
            return None
        key = id(v)
        if key in self._scalar_expr:
            return self._scalar_expr[key][1]
        result = self._scalar_expr_uncached(v, _seen)
        self._scalar_expr[key] = (v, result)
        return result

    def _scalar_expr_uncached(self, v, _seen):
        if self.value(v) is not None:
            arr = self.array(v).reshape(-1)
            if arr.size == 1:
                f = float(arr[0])
                return int(f) if f.is_integer() else f
            return None
        if v.op is None:
            # A genuine (sub)function input with no producer -- the same "this IS the topology's one
            # true dynamic quantity" case `_infer_dynamic_dim_expr` treats unconditionally as "n_tokens"
            # (see its own docstring). Originally gated to `v.name == "length"` only (NeMo's Conformer-
            # CTC always feeds a real per-utterance length in under that exact name) -- too narrow for
            # VITS's `MultiHeadAttention._get_relative_embeddings`, whose own dynamic `length` scalar
            # traces back to `key.size(2)` (a plain shape query on an ACTIVATION, not a declared
            # "length" input) and bottoms out at some other producer-less var entirely -- confirmed this
            # was exactly why `padded[:, start:end]`'s `start` (`pad + (window_size+1) - length`)
            # resolved to `None` and silently fell back to the slice's full extent (a real element-count
            # bug: the sliced relative-position table came out ~34x too long at T=62, one axis short of
            # crashing GraphBuilder's own RESHAPE element-count check downstream). Every producer-less
            # scalar this whole exporter's single-true-dynamic-axis design ever reaches IS that quantity,
            # matching `_infer_dynamic_dim_expr`'s own unconditional treatment -- not just ones spelled
            # "length".
            return "n_tokens"
        op = v.op
        if op.op_type in ("cast", "squeeze", "identity", "expand_dims"):
            inner = op.inputs.get("x") or op.inputs.get("data")
            return self.scalar_expr(inner, _seen)
        if op.op_type == "select":
            # `torch.where(cond, a, b)` -- NeMo's own `get_seq_len` uses exactly this shape for its
            # "fix for seq_len = 0 for streaming" guard (`torch.where(seq_len == 0, zeros, seq_len_
            # unfixed)`). This exporter's target use (a real, single, non-empty utterance) never hits the
            # degenerate `cond` branch, so -- mirroring the "batch is always 1" style invariant used
            # throughout the exporter -- always take the "b" (false/else) branch rather than trying to
            # resolve `cond` itself.
            return self.scalar_expr(op.inputs.get("b"), _seen)
        if op.op_type == "gather":
            return self.gather_shape_value(v)
        if op.op_type in _ARITH_OPS:
            x_e = self.scalar_expr(op.inputs.get("x"), _seen)
            y_e = self.scalar_expr(op.inputs.get("y"), _seen)
            if x_e is None or y_e is None:
                return None
            if isinstance(x_e, (int, float)) and isinstance(y_e, (int, float)):
                if op.op_type == "add":
                    return x_e + y_e
                if op.op_type == "sub":
                    return x_e - y_e
                if op.op_type == "mul":
                    return x_e * y_e
                if op.op_type == "real_div":
                    return x_e / y_e
                return x_e // y_e
            expr = f"(({x_e}) {_ARITH_OPS[op.op_type]} ({y_e}))"
            return f"(floor({expr}))" if op.op_type == "floor_div" else expr
        return None

    def range_scalar(self, v):
        """
        Resolves one `range_1d` start/end/step operand to either a derived symbolic expression (via
        `gather_shape_value`, then the more general `scalar_expr`) or a literal constant, for use both
        when emitting a RANGE_1D JSON node's own attrs and when inferring the LENGTH of that range's own
        output elsewhere (see the "range_1d" case in `_infer_dynamic_dim_expr`) -- the two need the exact
        same resolution logic.
        """
        if v is None:
            return None
        key = id(v)
        if key in self._range_scalar:
            return self._range_scalar[key][1]
        derived = self.gather_shape_value(v)
        if derived is not None:
            result = derived
        elif self.value(v) is not None:
            result = float(self.array(v).reshape(-1)[0])
        else:
            result = self.scalar_expr(v)
        self._range_scalar[key] = (v, result)
        return result

    def slice_axis_value(self, idx_var, axis):
        """
        Resolves a `slice_by_index` op's "begin"/"end" input at one specific axis -- needed because that
        input can itself be dynamic (a `concat`/`stack` of per-axis gather-derived scalars, same
        structure as a `reshape`'s "shape" input) rather than a plain constant array. Used by BOTH the
        `_infer_dynamic_dim_expr`-level "slice_by_index" case (deriving a symbolic axis's real
        expression for a DOWNSTREAM consumer) and the actual JSON-emitting `slice_by_index` translation
        (building the real VIEW node's own shape/offset) -- the two need the exact same resolution, and
        used to diverge: the translation only ever looked at a literal `.val` array, so whenever
        `begin`/`end` was a live concat (confirmed on the positional-encoding table crop,
        `self.pe[:, center - t + 1 : center + t]`, AND on `rel_shift`'s final
        `matrix_bd[..., : matrix_ac.size(-1)]` crop, whose `end` is a `concat` ending in a `gather` read
        off a DIFFERENT tensor's shape), it silently treated the WHOLE begin/end array as absent and fell
        back to copying the parent's own unsliced extent on every axis -- not just a shape-string
        cosmetic bug, the emitted VIEW's declared shape and its own stride math then genuinely disagreed
        on element count for whichever axis needed the real crop.
        """
        if idx_var is None:
            return None
        key = (id(idx_var), axis)
        if key in self._slice_axis:
            return self._slice_axis[key][1]
        result = self._slice_axis_value_uncached(idx_var, axis)
        self._slice_axis[key] = (idx_var, result)
        return result

    def _slice_axis_value_uncached(self, idx_var, axis):
        if self.value(idx_var) is not None:
            arr = self.array(idx_var).reshape(-1)
            return int(arr[axis]) if axis < len(arr) else None
        if idx_var.op is not None and idx_var.op.op_type in ("concat", "stack"):
            values = idx_var.op.inputs.get("values")
            if values is not None and axis < len(values):
                resolved = self.range_scalar(values[axis])
                if isinstance(resolved, (int, float)):
                    return int(resolved)
                return resolved
        return None

    def reshape_shape(self, op):
        """
        Resolves a `reshape` op's own "shape" input directly, torch-order -- needed because `reshape`
        (unlike `expand_dims`/`squeeze`) has no per-axis correspondence formula `_infer_dynamic_dim_expr`
        can apply to its OUTPUT var, so a symbolic output axis there always falls to that walk's blind
        "n_tokens" substitution, which is wrong whenever the reshape isn't a pure n_tokens pass-through.
        Going straight to the "shape" INPUT's own real value sidesteps that entirely, via two cases:

        1. `shape_var.val` is a genuine compile-time CONSTANT array -- trust it directly. MIL's own
           constant-folding only produces this when the whole "shape" derivation chain is provably
           input-invariant (confirmed on Conformer-CTC's positional-encoding table reshape: `shape` folds
           to a literal `[176, 9999, 1]` because it's derived from a WEIGHT's own static shape, not from
           any activation -- MIL does NOT fold shape-derived-from-an-ACTIVATION chains the same way, e.g.
           Q/K/V's `b, t, _ = x.shape` stays a live, unfolded `concat`/`gather` chain specifically because
           `x` is real per-call data). The output var's own TYPE-level shape inference can still (and
           does, in this exact case) report an opaque, unrelated symbol here regardless -- this bypasses
           that entirely rather than trusting it.
        2. Otherwise, if `shape_var` is a `concat`/`stack` of per-axis scalars (each typically a
           `gather(shape(real_x), idx)`, possibly cast) -- resolve each one via `range_scalar`.
           Needed for the common Q/K/V head-split idiom (`b, t, _ = x.shape; x.view(b, t, h, d)`): two
           genuinely different axes (batch and time) both collapsed to the same bare "n_tokens"
           substitution under the old out_info-based path, producing an invalid RESHAPE target with a
           repeated symbol.

        Returns None (falling back to the existing out_info-based path) unless every element resolves.
        """
        key = id(op)
        if key in self._reshape_shape:
            return self._reshape_shape[key][1]
        result = self._reshape_shape_uncached(op)
        self._reshape_shape[key] = (op, result)
        return result

    def _reshape_shape_uncached(self, op):
        shape_var = op.inputs.get("shape")
        if shape_var is None:
            return None
        if self.value(shape_var) is not None:
            raw = self.array(shape_var).reshape(-1)
            return [-1 if int(x) == -1 else str(int(x)) for x in raw]
        if shape_var.op is None or shape_var.op.op_type not in ("concat", "stack"):
            return None
        values = shape_var.op.inputs.get("values")
        if values is None:
            return None
        resolved = []
        for v in values:
            if not isinstance(v, Var):
                return None
            r = self.range_scalar(v)
            if r is None:
                return None
            if isinstance(r, (int, float)):
                # A literal -1 is PyTorch's own "infer this dim" marker (`x.view(b, t, -1)`) baked
                # directly into the trace, not a computed value -- must stay a real JSON int (not the
                # string "-1") so op_reshape's own infer-idx handling (primitives_basic.cpp) recognizes
                # it; every other numeric constant is just rendered as a plain digit string, same as the
                # rest of this exporter's shape-attr convention.
                resolved.append(-1 if int(r) == -1 else str(int(r)))
            else:
                resolved.append(str(r))
        return resolved
