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

**Derived values are sympy expressions** (see shape_expr.py), not strings: these walks compose real
algebra -- `end - begin`, `total / product(other axes)`, a chain of `add`/`mul`/`floor_div` -- and doing
that by f-string concatenation, on values MIL itself handed over as sympy objects, was throwing away
information at every step. The two exceptions are deliberate and documented at their definitions:
`range_scalar` still returns a plain Python number for a literal operand, and `reshape_shape` still
returns rendered strings, because both feed JSON attributes directly and their existing number-vs-string
distinction is part of what the engine reads back.
"""
import numpy as np
from coremltools.converters.mil.mil import Var

from .shape_expr import as_expr, floor_div, has_dynamic_symbol, render, to_number

# MIL arithmetic ops `scalar_expr` can fold into a real expression, and the sympy operation each maps
# to. `floor_div` is `real_div` wrapped in an explicit floor(), exactly as the engine's evaluator (which
# has no integer division) needs it spelled out.
_ARITH_OPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "real_div": lambda x, y: x / y,
    "floor_div": floor_div,
}


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
        self._dim_expr = {}
        self._reshape_shape_rendered = {}

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
        derived = self.exporter._infer_dynamic_dim_expr(real_var, torch_axis)
        if torch_axis == 0 and real_rank >= 2:
            # `x.shape[0]` (or `x.size(0)`) on a rank>=2 activation is USUALLY the canonical PyTorch
            # idiom for reading BATCH SIZE -- and this whole exporter's architecture only ever targets
            # batch=1 models (every declared model input's own batch axis is a literal 1, stated
            # repeatedly elsewhere in the exporter, e.g. the `tile` case). The real correctness gap this
            # guards is that the walk sometimes gives up at a not-yet-handled op type along a long chain
            # (a deep Conformer-CTC encoder layer runs through `layer_norm`/`linear`/`matmul`/`add`
            # dozens of times before bottoming out at a real input) and silently falls back to the bare
            # root axis -- a value a genuine batch=1 axis would never actually be -- corrupting whatever
            # RESHAPE/RANGE_1D consumes it.
            #
            # **The test is the derivation's provenance, not the axis index** (BACKLOG.md P4.2). This
            # used to short-circuit to 1 for axis 0 unconditionally, which is a claim about the model's
            # LAYOUT, and GigaAM v3 is the counterexample: its
            # `RotaryPositionMultiHeadAttention.forward` transposes to (T, B, H, D) *before* applying
            # the rotary embedding, so `q.shape[0]` there is the sequence length. Read as a batch size
            # it made the rotary cos/sin crop `pe[0:1]` -- one position wide, which ggml then broadcasts
            # over every frame, so every position got position zero's rotation. The graph built, ran,
            # and produced a plausible transcript that was simply wrong; only comparing the encoder's
            # own output against PyTorch's found it. A genuine batch axis is a LITERAL 1 in the traced
            # shape, so the walk answers it exactly and immediately without any of this -- and where the
            # walk has nothing better than the root axis, the batch reading is still the right guess.
            if derived is None or derived == as_expr(self.exporter.root_axis):
                return as_expr(1)
        return derived

    def dim_expr(self, var, torch_axis):
        """The real SymbolEnv expression for one symbolic MIL shape dimension -- the memoized front end
        for `exporter._infer_dynamic_dim_expr_uncached`, which walks `var`'s producer chain backward.

        The memo here is a **correctness-adjacent performance fix, not a tidiness one.** That walk used
        to be a linear producer-chain walk with an `id(var)` cycle guard. Commit a29ffe5 (Kokoro) removed
        the guard -- correctly: the graph is an acyclic DAG, and the guard was silently returning None on
        ordinary *diamonds* (Kokoro's SineGen reaches the same `rad_values` down two paths) -- and, in the
        same commit, added a `concat` case that recurses into *every* operand. Linear walk plus branching
        recursion plus no revisit-suppression is the textbook exponential blow-up, and it is exactly what
        happened: Conformer-CTC's export went from linear in encoder depth to roughly 3x per layer,
        turning a ~2 s export into one that does not finish at all on 16 blocks.

        Caching the answer is the right version of what the deleted guard was reaching for: it suppresses
        the redundant revisit without ever turning a legitimate second visit into a wrong answer. This is
        the same fix, and the same reasoning, as `scalar_expr`'s memo below -- that one was applied when
        its guard was removed for the identical VITS diamond; this walk was simply left behind.

        Keyed on `(id(var), torch_axis)`: the same Var routinely resolves different axes to different
        expressions, so the axis is part of the identity. The Var is stored alongside the answer to keep
        it alive, so its `id` can never be recycled onto another object.
        """
        key = (id(var), torch_axis)
        if key in self._dim_expr:
            return self._dim_expr[key][1]
        result = self.exporter._infer_dynamic_dim_expr_uncached(var, torch_axis)
        self._dim_expr[key] = (var, result)
        return result

    def annotate_dynamic_shapes(self, program) -> None:
        """Eagerly resolves (and memoizes, via `dim_expr`) every dynamic axis of every Var's shape in
        `program`, once, in one pass over the whole graph -- EXPORT-ROADMAP.md R2b.

        Functionally a no-op: `dim_expr` is already memoized per `(id(var), axis)`, so whichever
        consumer touches a given Var's shape first gets the exact same answer this produces, and every
        later touch was already O(1). What changes is *when*: derivation now runs once, in a
        deterministic walk over the whole (already MIL-pass-canonicalized) graph right after
        `apply_loom_mil_passes`, rather than being triggered ad hoc by whichever of a model's several
        topologies (`generate_graph_topology` runs once per submodule/slice) happens to visit a given
        Var first. That turns this module's memo from an incidental cache into the real "annotate once,
        up front" pass R2 asks for -- and is the precondition for ever auditing the C++ "heal
        transposed/permuted layouts" heuristics (BACKLOG.md), which needs every shape already resolved
        rather than derived lazily by whichever consumer happens to ask first.

        Call once per exporter, before any topology/driver generation walks `program` -- mirroring
        `apply_loom_mil_passes`'s own "runs once, up front" contract.
        """
        for func in program.functions.values():
            self._annotate_block(func)

    def _annotate_block(self, block) -> None:
        for op in block.operations:
            for b in op.blocks:
                self._annotate_block(b)
            for var in op.outputs:
                shape = getattr(var, "shape", None)
                if shape is None:
                    continue
                for axis, dim in enumerate(shape):
                    if has_dynamic_symbol(dim):
                        self.dim_expr(var, axis)

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
        return self._scalar_entry(v, _seen)[0]

    def scalar_expr_is_guess(self, v):
        """True iff `scalar_expr(v)`'s answer rests on the producer-less-var fallback below rather than
        on a real derivation -- i.e. it says `n_tokens` because the walk ran out of graph, not because
        it worked out that this value *is* the sequence length.

        Consumers need to tell those two apart and cannot do it by looking at the answer: both are
        exactly `n_tokens`. `slice_by_index`'s derivation is the one that cares (see its own comment in
        `exporter.py`), and before expressions were normalized it accidentally distinguished them by
        *spelling* -- a genuine derivation happened to come out as `floor((n_tokens + 0 - 1)/1) + 1`
        rather than the bare symbol, so a test for the literal string `"n_tokens"` only ever matched the
        fallback. Simplification made those spellings identical and the accident stopped working, in two
        real models at once (see BACKEND.md). This is the same distinction, made on purpose.
        """
        return self._scalar_entry(v)[1]

    def _scalar_entry(self, v, _seen=None):
        """`(expression, is_guess)` for one Var, memoized together so the two can never disagree."""
        if v is None or not isinstance(v, Var):
            return (None, False)
        key = id(v)
        if key in self._scalar_expr:
            return self._scalar_expr[key][1]
        entry = self._scalar_expr_uncached(v, _seen)
        self._scalar_expr[key] = (v, entry)
        return entry

    def _scalar_expr_uncached(self, v, _seen):
        if self.value(v) is not None:
            arr = self.array(v).reshape(-1)
            if arr.size == 1:
                return (as_expr(float(arr[0])), False)
            return (None, False)
        if v.op is None:
            # A genuine (sub)function input with no producer -- the same "this IS the topology's one
            # true dynamic quantity" case `_infer_dynamic_dim_expr` treats unconditionally as this
            # topology's own `root_axis` (see its own docstring in exporter.py -- "n_tokens" unless the
            # caller declared otherwise, e.g. Conformer-CTC/Parakeet's "n_samples"). Originally gated to
            # `v.name == "length"` only (NeMo's Conformer-CTC always feeds a real per-utterance length in
            # under that exact name) -- too narrow for VITS's `MultiHeadAttention._get_relative_embeddings`,
            # whose own dynamic `length` scalar traces back to `key.size(2)` (a plain shape query on an
            # ACTIVATION, not a declared "length" input) and bottoms out at some other producer-less var
            # entirely -- confirmed this was exactly why `padded[:, start:end]`'s `start` (`pad +
            # (window_size+1) - length`) resolved to `None` and silently fell back to the slice's full
            # extent (a real element-count bug: the sliced relative-position table came out ~34x too long
            # at T=62, one axis short of crashing GraphBuilder's own RESHAPE element-count check
            # downstream). Every producer-less scalar this whole exporter's single-true-dynamic-axis
            # design ever reaches IS that quantity, matching `_infer_dynamic_dim_expr`'s own unconditional
            # treatment -- not just ones spelled "length".
            #
            # This is the ONE place the answer is a guess rather than a derivation, which is what the
            # `is_guess` half of this entry marks -- see `scalar_expr_is_guess`.
            return (as_expr(self.exporter.root_axis), True)
        op = v.op
        if op.op_type in ("cast", "squeeze", "identity", "expand_dims"):
            inner = op.inputs.get("x") or op.inputs.get("data")
            return self._scalar_entry(inner, _seen)
        if op.op_type == "select":
            # `torch.where(cond, a, b)` -- NeMo's own `get_seq_len` uses exactly this shape for its
            # "fix for seq_len = 0 for streaming" guard (`torch.where(seq_len == 0, zeros, seq_len_
            # unfixed)`). This exporter's target use (a real, single, non-empty utterance) never hits the
            # degenerate `cond` branch, so -- mirroring the "batch is always 1" style invariant used
            # throughout the exporter -- always take the "b" (false/else) branch rather than trying to
            # resolve `cond` itself.
            return self._scalar_entry(op.inputs.get("b"), _seen)
        if op.op_type == "gather":
            # A real shape query, not a guess -- even when its answer is exactly `n_tokens`.
            return (self.gather_shape_value(v), False)
        if op.op_type in _ARITH_OPS:
            x_e, x_guess = self._scalar_entry(op.inputs.get("x"), _seen)
            y_e, y_guess = self._scalar_entry(op.inputs.get("y"), _seen)
            if x_e is None or y_e is None:
                return (None, False)
            # One code path for literals and symbols alike: sympy folds the all-literal case exactly
            # (and keeps `floor_div`'s floor()), where this used to need a separate int/float branch
            # that quietly disagreed with the string branch about integer division.
            return (_ARITH_OPS[op.op_type](x_e, y_e), x_guess or y_guess)
        return (None, False)

    def range_scalar(self, v):
        """
        Resolves one `range_1d` start/end/step operand to either a derived symbolic expression (via
        `gather_shape_value`, then the more general `scalar_expr`) or a literal constant, for use both
        when emitting a RANGE_1D JSON node's own attrs and when inferring the LENGTH of that range's own
        output elsewhere (see the "range_1d" case in `_infer_dynamic_dim_expr`) -- the two need the exact
        same resolution logic.

        Returns a sympy expression, **or a plain Python float when the operand is a literal constant**.
        That second case is deliberate, not an oversight: RANGE_1D's JSON node emits `start`/`end`/`step`
        as real JSON numbers when they are known and as expression strings only when they are not (the
        engine's `resolve_attr_number` accepts either), so erasing the distinction here would churn every
        RANGE_1D attribute in every exported model to no end. `as_expr` lifts it back into algebra
        wherever the value is composed with others.
        """
        return self._range_entry(v)[0]

    def range_scalar_is_guess(self, v):
        """Whether `range_scalar(v)` rests on `scalar_expr`'s producer-less fallback -- see
        `scalar_expr_is_guess`. A `gather(shape(...))` derivation and a literal constant are never
        guesses, however dynamic their answers look."""
        return self._range_entry(v)[1]

    def _range_entry(self, v):
        if v is None:
            return (None, False)
        key = id(v)
        if key in self._range_scalar:
            return self._range_scalar[key][1]
        derived = self.gather_shape_value(v)
        if derived is not None:
            entry = (derived, False)
        elif self.value(v) is not None:
            entry = (float(self.array(v).reshape(-1)[0]), False)
        else:
            entry = self._scalar_entry(v)
        self._range_scalar[key] = (v, entry)
        return entry

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
        return self._slice_axis_entry(idx_var, axis)[0]

    def slice_axis_is_guess(self, idx_var, axis):
        """Whether this bound rests on `scalar_expr`'s producer-less fallback -- see
        `scalar_expr_is_guess`. `slice_by_index`'s own derivation in `exporter.py` uses this to decide
        whether an `end` of `n_tokens` is a real answer or the walk having run out of graph."""
        return self._slice_axis_entry(idx_var, axis)[1]

    def _slice_axis_entry(self, idx_var, axis):
        if idx_var is None:
            return (None, False)
        key = (id(idx_var), axis)
        if key in self._slice_axis:
            return self._slice_axis[key][1]
        entry = self._slice_axis_value_uncached(idx_var, axis)
        self._slice_axis[key] = (idx_var, entry)
        return entry

    def _slice_axis_value_uncached(self, idx_var, axis):
        if self.value(idx_var) is not None:
            arr = self.array(idx_var).reshape(-1)
            return (int(arr[axis]) if axis < len(arr) else None, False)
        if idx_var.op is not None and idx_var.op.op_type in ("concat", "stack"):
            values = idx_var.op.inputs.get("values")
            if values is not None and axis < len(values):
                resolved, is_guess = self._range_entry(values[axis])
                # A slice bound that resolved to a literal stays a plain int (both callers branch on
                # that to normalize Python-style negative indices against the axis extent); anything
                # else stays a sympy expression.
                number = to_number(resolved)
                return (int(number) if number is not None else resolved, is_guess)
        return (None, False)

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
        Each resolved axis is a **rendered string** (or the int -1); `reshape_shape_exprs` is the same
        answer as sympy, for the callers that go on to compute with it rather than emit it.
        """
        key = id(op)
        if key in self._reshape_shape_rendered:
            return self._reshape_shape_rendered[key][1]
        exprs = self.reshape_shape_exprs(op)
        result = None if exprs is None else [d if d == -1 else render(d) for d in exprs]
        self._reshape_shape_rendered[key] = (op, result)
        return result

    def reshape_shape_exprs(self, op):
        """`reshape_shape` before rendering: a list of sympy expressions, with the int -1 left as the
        literal marker it is. Used by `_infer_dynamic_dim_expr`'s own `reshape`/`fill` case, which
        divides the input's total element count by the other axes and would otherwise have to parse
        back what this just printed."""
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
            return [-1 if int(x) == -1 else as_expr(int(x)) for x in raw]
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
            number = to_number(r)
            if number is not None:
                # A literal -1 is PyTorch's own "infer this dim" marker (`x.view(b, t, -1)`) baked
                # directly into the trace, not a computed value -- it must stay a real int (never the
                # expression -1) all the way to the JSON, so op_reshape's own infer-idx handling
                # (primitives_basic.cpp) recognizes it.
                resolved.append(-1 if int(number) == -1 else as_expr(int(number)))
            else:
                resolved.append(as_expr(r))
        return resolved
