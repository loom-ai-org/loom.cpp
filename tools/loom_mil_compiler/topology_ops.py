"""The MIL-op -> ggml-composite rewrite table used by `LoomGGUFExporter.generate_graph_topology`.

Every MIL op whose ggml lowering is more than "one OP_MAP entry, inputs forwarded verbatim" gets a
handler here, registered against the MIL op types it claims plus an optional *guard predicate*. The
exporter's per-op loop is then a mechanical lookup -- `lookup_topology_rule(op)` returns the first rule
whose op type matches and whose guard accepts, or None to fall through to the generic `OP_MAP` path --
rather than a 2,000-line `if op_type == "..."` chain in which the selection criteria were interleaved
with the composition code that implements them.

The point of the guard being a separate, named thing is auditability: `describe_topology_rules()` prints
the whole table, so "which condition selects which ggml mapping for MIL op X" is answerable by reading
one line per alternative instead of by tracing branches through a handler body. `matmul` (four
transpose_x/transpose_y combinations, only two of which have a correct ggml_mul_mat composition) and
`gelu` (exact-erf vs. the fused tanh approximation, which ggml's GELU primitive cannot compute) are the
two ops where that distinction is load-bearing today.

Handlers take `(self, op, ctx)` where `self` is the `LoomGGUFExporter` (they are plain functions rather
than methods only so this table can live outside `exporter.py`) and `ctx` is the `TopologyContext`
carrying the per-topology output state. A handler returns nothing; it appends to `ctx.nodes`, records
`ctx.aliases` entries, and/or registers `self.weights` entries.
"""
import hashlib
import numpy as np
from coremltools.converters.mil.mil import Var

from .shape_expr import as_expr, render, to_number
from .symbols import DYNAMIC_SYMBOL_RE as _DYNAMIC_SYMBOL_RE
from .value_facts import static_array, static_ints, static_scalar, static_value


def _attr_number_or_expr(value):
    """One resolved quantity as a JSON attribute: a real number when it is one, else the rendered
    expression string. Both are accepted by the engine's `resolve_attr_number`/`resolve_attr_int_array`,
    and keeping the distinction is what makes this refactor a no-op for every already-static attribute."""
    number = to_number(value)
    return number if number is not None else render(value)


class TopologyContext:
    """The mutable state one `generate_graph_topology` call threads through its op handlers.

    Deliberately just the four things the handlers actually touch (verified by a free-variable scan of
    the pre-refactor dispatch chain): the node list being built, the SSA-name alias map, the declared
    input list, and the topology's own name (which namespaces any weight a handler bakes). Everything
    else a handler needs comes off the exporter (`self`) or the op.
    """

    __slots__ = ("func_name", "nodes", "aliases", "topo_inputs")

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.nodes = []
        self.aliases = {}
        self.topo_inputs = []

    def resolve(self, name: str) -> str:
        """Follows `aliases` to the real emitted name. Cycle-guarded: a self-referential entry (e.g. a
        submodule input whose real parameter name already IS the standardized one) would otherwise spin
        forever."""
        seen = set()
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name


class TopologyRule:
    """One (MIL op types, guard) -> ggml composite builder entry of the rewrite table."""

    __slots__ = ("op_types", "guard", "when", "handler")

    def __init__(self, op_types, guard, when, handler):
        self.op_types = op_types
        self.guard = guard
        self.when = when
        self.handler = handler

    def __call__(self, exporter, op, ctx):
        return self.handler(exporter, op, ctx)

    @property
    def name(self) -> str:
        return self.handler.__name__


# {MIL op_type -> [TopologyRule]}, in registration order. Ordering matters only among rules sharing an
# op type: the first accepting guard wins, so an unguarded rule acts as that op type's catch-all and
# must be registered last.
_RULES = {}


def topology_rule(*op_types, guard=None, when="any"):
    """Registers the decorated `(exporter, op, ctx)` function as this table's lowering for `op_types`.

    `guard(exporter, op) -> bool` narrows the rule to a subset of that op type's instances; `when` is
    the human-readable statement of the same condition, used by `describe_topology_rules()`.
    """
    def register(fn):
        rule = TopologyRule(tuple(op_types), guard, when, fn)
        for op_type in op_types:
            _RULES.setdefault(op_type, []).append(rule)
        fn.rule = rule
        return fn

    return register


def lookup_topology_rule(exporter, op):
    """The first registered rule claiming `op`'s type whose guard accepts it, else None (meaning: fall
    through to the exporter's generic OP_MAP lowering)."""
    for rule in _RULES.get(op.op_type, ()):
        if rule.guard is None or rule.guard(exporter, op):
            return rule
    return None


def describe_topology_rules() -> str:
    """The whole table as text, one line per (op type, guard) alternative -- the audit view."""
    out = []
    for op_type in sorted(_RULES):
        for rule in _RULES[op_type]:
            out.append(f"{op_type:<26} when {rule.when:<44} -> {rule.name}")
    return "\n".join(out)


# --- guard helpers -------------------------------------------------------------------------------
# Small pure readers of an op's own static attributes, shared by a rule's guard and its handler so the
# two can never disagree about which case they're in.

def _matmul_transposes(op):
    """MIL matmul's (transpose_x, transpose_y) as plain bools, defaulting to False when absent."""
    return (bool(static_value(op.inputs.get("transpose_x"), False)),
            bool(static_value(op.inputs.get("transpose_y"), False)))


def _gelu_mode(op):
    """MIL gelu's "mode" input (PyTorch's `approximate=`), uppercased; "EXACT" when absent."""
    return str(static_value(op.inputs.get("mode"), "EXACT")).upper()


# --- handlers -----------------------------------------------------------------------------------

@topology_rule('const')
def _op_const(self, op, ctx):
    aliases, func_name = ctx.aliases, ctx.func_name
    val = static_value(op.val)
    weight_name = self.safe_name(op.outputs[0].name)

    # A constant used as a "gather" op's own "indices" input may hold genuine Python-style
    # negative indices (e.g. `x.shape[-1]`, MIL's own convention) -- but `GET_ROWS`
    # (src/ops/primitives_mil.cpp, wrapping ggml_get_rows) has no such convention, it treats
    # the value as a raw non-negative row offset. Left unnormalized, this silently reads
    # garbage/wrapped memory instead of raising -- confirmed as the root cause of Conformer-
    # CTC's length-tracking bug (BACKLOG.md): `gather(shape(x), -1)` extracting a mel-frame
    # count instead read whatever byte pattern a negative row index happened to land on.
    #
    # Special-cased for the concrete pattern actually seen (gathering from a real `shape()`
    # op's own output): that shape-vector is always `[ne0, ne1, ne2, ne3]` -- REVERSED
    # (ne-order) relative to the tensor's own torch-order shape, this whole file's standing
    # convention -- so a torch-order index (post negative-normalization against the QUERIED
    # tensor's real rank, not the shape-vector's fixed length-4) must ALSO be flipped via the
    # same `rank - 1 - axis` this file uses everywhere else, not just offset into
    # non-negative range. (A first attempt that only did the offset -- treating this like an
    # ordinary negative array index modulo 4 -- fixed the out-of-bounds read but silently read
    # the wrong slot: ne3, always the fixed padding value 1 for any tensor of real rank < 4,
    # not the genuine last-torch-axis value ne0 holds.)
    arr = np.asarray(val) if val is not None else None
    if arr is not None and arr.dtype.kind in "iu" and np.any(arr < 0):
        for child in op.outputs[0].child_ops:
            if child.op_type != "gather" or child.inputs.get("indices") is not op.outputs[0]:
                continue
            shape_vec_var = child.inputs.get("x") or child.inputs.get("params")
            shape_op = shape_vec_var.op if shape_vec_var is not None else None
            if shape_op is not None and shape_op.op_type == "shape":
                real_var = shape_op.inputs.get("x")
                if real_var is not None and real_var.shape is not None:
                    real_rank = len(real_var.shape)
                    torch_axis = np.where(arr < 0, arr + real_rank, arr)
                    arr = real_rank - 1 - torch_axis
            else:
                # Not a shape-vector gather -- fall back to a plain same-order negative-index
                # normalization against the gathered axis's own (static) size.
                axis_var = child.inputs.get("axis")
                axis = int(static_value(axis_var, 0))
                if shape_vec_var is not None and shape_vec_var.shape is not None and axis < len(shape_vec_var.shape):
                    dim = shape_vec_var.shape[axis]
                    if isinstance(dim, (int, np.integer)):
                        arr = np.where(arr < 0, arr + int(dim), arr)
            break
        val = arr

    # A flat-namespace export (and `main_topology` itself) writes weights unprefixed
    if func_name == "main_topology" or self.flat_namespace:
        namespaced_name = weight_name
    else:
        namespaced_name = f"{func_name}.{weight_name}"

    # Safe compaction to satisfy GGUF's GGML_MAX_NAME (64 chars) limit
    if len(namespaced_name) >= 64:
        h = hashlib.md5(namespaced_name.encode("utf-8")).hexdigest()[:6]
        namespaced_name = f"{namespaced_name[:30]}_{h}_{namespaced_name[-20:]}"

    arr_val = np.array(val)
    if arr_val.ndim == 0:
        # A genuine 0-D (scalar) constant -- confirmed as a real bug, not just a formality:
        # GGUF/ggml has no 0-D tensor representation, and writing one silently round-trips as
        # a tensor with a ZERO-length dimension (ne[0]=0) instead of a proper length-1 scalar,
        # which then fails every downstream shape check that consumes it (first hit by
        # Kokoro's SineGen phase computation baking a literal `2 * torch.pi` scalar multiply
        # as its own weight). Reshape to a proper 1-element 1-D array, matching ggml's own
        # "scalar = shape [1]" convention used everywhere else in this project.
        arr_val = arr_val.reshape(1)
    self.weights[namespaced_name] = arr_val
    if weight_name != namespaced_name:
        aliases[weight_name] = namespaced_name


@topology_rule('cast')
def _op_cast(self, op, ctx):
    nodes, aliases, resolve = ctx.nodes, ctx.aliases, ctx.resolve
    input_name = self.safe_name(op.inputs["x"].name)
    output_name = self.safe_name(op.outputs[0].name)
    # MIL casts between float precisions (fp16<->fp32) are pure no-ops for this engine, since
    # every ggml op here computes in f32 internally regardless of a tensor's storage dtype --
    # aliasing them away is correct and avoids emitting a mountain of redundant CAST nodes.
    # But a cast that actually changes numeric *kind* (e.g. HF rotary embedding's
    # `position_ids.float()`, MIL's int32->fp32) is a real value reinterpretation: skipping it
    # leaves an integer-typed ggml tensor flowing into float-only ops downstream (MUL_MAT's
    # vec_dot has no integer kernel and null-derefs). Emit a real CAST node for that case.
    in_dtype = self.get_var_info(op.inputs["x"])["dtype"]
    out_dtype = self.get_var_info(op.outputs[0])["dtype"]
    if in_dtype != out_dtype:
        nodes.append({
            "op": "CAST",
            "inputs": [resolve(input_name)],
            "outputs": [output_name],
            "attrs": {"dtype": out_dtype},
        })
    else:
        aliases[output_name] = resolve(input_name)


@topology_rule('linear')
def _op_linear(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Compose linear as MUL_MAT + optional ADD
    x_var_obj = op.inputs.get("x") or op.inputs.get("input")
    x_var = self.safe_name(x_var_obj.name)
    weight_var = self.safe_name(op.inputs["weight"].name)
    bias_var = self.safe_name(op.inputs["bias"].name) if "bias" in op.inputs and hasattr(op.inputs["bias"], "name") else None
    output_var = self.safe_name(op.outputs[0].name)

    # In Loom, MUL_MAT expects [weight, x]
    if bias_var:
        inter_var = output_var + "_matmul"
        nodes.append({
            "op": "MUL_MAT",
            "inputs": [resolve(weight_var), resolve(x_var)],
            "outputs": [inter_var]
        })
        nodes.append({
            "op": "ADD",
            "inputs": [inter_var, resolve(bias_var)],
            "outputs": [output_var]
        })
    else:
        nodes.append({
            "op": "MUL_MAT",
            "inputs": [resolve(weight_var), resolve(x_var)],
            "outputs": [output_var]
        })


# MIL's matmul(x, y, transpose_x, transpose_y) computes X @ Y where X = x^T if transpose_x else x,
# Y = y^T if transpose_y else y (batched over leading dims). This is NOT the same op as
# ggml_mul_mat(A, B), which always contracts over ne0 of both operands and returns
# ne=[A.ne1, B.ne1, ...] -- i.e. it computes B_mat @ A_mat^T, not A_mat @ B_mat. Forwarding MIL's x/y
# straight through as ggml_mul_mat(x, y) silently produces a transposed-but-same-shape (for square
# attention scores) or outright wrong-axis result, exactly the numerical-correctness bug tracked in
# EXPORT-BACKLOG.md item 1 -- confirmed by bisecting the real attention-score matmul (transpose_y=True)
# and the scores@value matmul (transpose_x=transpose_y=False) against HF's own SDPA inputs.
#
# Each transpose combination therefore gets its own rule, guarded on exactly the combination it derives
# (from ggml_mul_mat's result.ne=[A.ne1,B.ne1,B.ne2,B.ne3] formula). Both combinations used by
# scaled_dot_product_attention's decomposition are composed; the unguarded catch-all rejects every other
# combination rather than silently miscomputing it.
#
# `passes.py`'s `normalize_matmul` pass (EXPORT-ROADMAP.md R2a) rewrites every `transpose_x=True` matmul
# into `matmul(transpose(x), y, transpose_x=False, ...)` before this table ever sees it -- so in
# practice only (False, True) and (False, False) ever reach here, and the catch-all below exists purely
# as a defensive backstop, not because transpose_x=True is a real unhandled case anymore.

@topology_rule('matmul', guard=lambda self, op: _matmul_transposes(op) == (False, True),
               when="transpose_x=False, transpose_y=True")
def _op_matmul_x_yt(self, op, ctx):
    # X @ Y^T: both operands already share ne0 (the contracted/embedding axis) in their
    # natural layout, so this is a straight ggml_mul_mat(y, x) -- key-first, matching the
    # llama.cpp attention-score convention.
    ctx.nodes.append({
        "op": "MUL_MAT",
        "inputs": [ctx.resolve(self.safe_name(op.inputs["y"].name)),
                   ctx.resolve(self.safe_name(op.inputs["x"].name))],
        "outputs": [self.safe_name(op.outputs[0].name)]
    })


@topology_rule('matmul', guard=lambda self, op: _matmul_transposes(op) == (False, False),
               when="transpose_x=False, transpose_y=False")
def _op_matmul_x_y(self, op, ctx):
    # X @ Y: Y needs its leading two ne axes swapped (and made contiguous) before it can
    # be used as ggml_mul_mat's first ("A") operand -- see the derivation in the comment
    # above. Composed as PERMUTE + CONT so the C++ side never has to guess this from
    # shapes alone.
    nodes, resolve = ctx.nodes, ctx.resolve
    x_var = self.safe_name(op.inputs["x"].name)
    y_var = self.safe_name(op.inputs["y"].name)
    output_var = self.safe_name(op.outputs[0].name)
    perm_var = output_var + "_mm_y_perm"
    cont_var = output_var + "_mm_y_cont"
    nodes.append({
        "op": "PERMUTE",
        "inputs": [resolve(y_var)],
        "outputs": [perm_var],
        "attrs": {"axes": [1, 0, 2, 3]}
    })
    nodes.append({
        "op": "CONT",
        "inputs": [perm_var],
        "outputs": [cont_var]
    })
    nodes.append({
        "op": "MUL_MAT",
        "inputs": [cont_var, resolve(x_var)],
        "outputs": [output_var]
    })


@topology_rule('matmul', when="any other transpose combination (rejected)")
def _op_matmul_unsupported(self, op, ctx):
    tx, ty = _matmul_transposes(op)
    raise NotImplementedError(
        f"matmul op '{op.name}' has transpose_x={tx}, transpose_y={ty}. `passes.py`'s "
        "normalize_matmul pass should have already rewritten every transpose_x=True matmul into "
        "transpose_x=False before this table ever ran -- reaching here with transpose_x still True "
        "means that pass didn't run, or missed this op."
    )


# MIL's `gelu` carries an extra "mode" string input (PyTorch's `approximate=` arg, "EXACT" or "TANH")
# the generic OP_MAP fallback would otherwise add as a second (bogus, string-typed) ggml node input --
# first hit by VITS's DDSConv (`F.gelu(y)`, no `approximate=` -> PyTorch's own default "none"/exact).
# ggml's own GELU primitive (op_gelu, src/ops/primitives_basic.cpp) always computes the EXACT erf
# formula (`ggml_gelu_erf`, chosen there for reproducibility over the tanh/sigmoid lookup-table
# approximations), so the mode is what decides between a one-node mapping and a full composition -- the
# reason it's a guard rather than a branch inside one handler.

@topology_rule('gelu', guard=lambda self, op: _gelu_mode(op) in ("EXACT", "NONE"),
               when="mode is EXACT/NONE")
def _op_gelu_exact(self, op, ctx):
    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    ctx.nodes.append({
        "op": "GELU",
        "inputs": [ctx.resolve(self.safe_name(x_var_obj.name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
    })


@topology_rule('gelu', guard=lambda self, op: _gelu_mode(op) in ("TANH_APPROXIMATION", "TANH"),
               when="mode is TANH_APPROXIMATION/TANH")
def _op_gelu_tanh_approx(self, op, ctx):
    nodes, resolve, func_name = ctx.nodes, ctx.resolve, ctx.func_name
    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    # HF's "gelu_new"/"NewGELUActivation" (0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) --
    # first hit by Kokoro's CustomAlbert (a real transformers.AlbertModel, hidden_act=
    # "gelu_new"): coremltools' own "fuse_gelu_tanh_approximation" MIL pass recognizes that
    # exact elementwise composition and folds it into one `gelu(mode=TANH_APPROXIMATION)` op,
    # which ggml's GELU primitive can't compute (always the exact erf formula, chosen for
    # reproducibility -- see the EXACT branch's own comment). Composed back out into the
    # identical explicit SQR/SCALE/ADD/MUL/TANH sequence convert_kokoro_albert.py's own
    # bespoke `gelu_new` helper already used for this exact formula, general (any TANH-
    # approximate-GELU model hits this same fused op), not Albert-specific.
    x_name = resolve(self.safe_name(x_var_obj.name))
    out_name = self.safe_name(op.outputs[0].name)
    one_name = "gelu_tanh_approx.one" if (func_name == "main_topology" or self.flat_namespace) \
        else f"{func_name}.gelu_tanh_approx.one"
    if one_name not in self.weights:
        self.weights[one_name] = np.array([1.0], dtype=np.float32)
    sqrt_2_over_pi = float(np.sqrt(2.0 / np.pi))
    x_sq = f"{out_name}_gelu_sq"
    nodes.append({"op": "SQR", "inputs": [x_name], "outputs": [x_sq]})
    cube_term = f"{out_name}_gelu_cubeterm"
    nodes.append({"op": "SCALE", "inputs": [x_sq], "outputs": [cube_term], "attrs": {"s": 0.044715}})
    inner_add = f"{out_name}_gelu_inner_add"
    nodes.append({"op": "ADD", "inputs": [cube_term, one_name], "outputs": [inner_add]})
    inner_mul = f"{out_name}_gelu_inner_mul"
    nodes.append({"op": "MUL", "inputs": [inner_add, x_name], "outputs": [inner_mul]})
    inner_scaled = f"{out_name}_gelu_inner_scaled"
    nodes.append({"op": "SCALE", "inputs": [inner_mul], "outputs": [inner_scaled],
                   "attrs": {"s": sqrt_2_over_pi}})
    tanh_out = f"{out_name}_gelu_tanh"
    nodes.append({"op": "TANH", "inputs": [inner_scaled], "outputs": [tanh_out]})
    tanh_p1 = f"{out_name}_gelu_tanh_p1"
    nodes.append({"op": "ADD", "inputs": [tanh_out, one_name], "outputs": [tanh_p1]})
    mul2 = f"{out_name}_gelu_mul2"
    nodes.append({"op": "MUL", "inputs": [tanh_p1, x_name], "outputs": [mul2]})
    nodes.append({"op": "SCALE", "inputs": [mul2], "outputs": [out_name], "attrs": {"s": 0.5}})


@topology_rule('gelu', when="any other mode (rejected)")
def _op_gelu_unsupported(self, op, ctx):
    raise NotImplementedError(
        f"gelu op '{op.name}' has mode={_gelu_mode(op)!r} -- ggml's GELU primitive only computes "
        "the exact erf formula; only EXACT and TANH_APPROXIMATION are composed here."
    )


@topology_rule('leaky_relu')
def _op_leaky_relu(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # HiFi-GAN vocoder's own activation (Generator/ResBlock2, real slope=0.1/0.01) -- ggml's
    # LEAKY_RELU primitive already exists (src/ops/primitives_basic.cpp) and just needed
    # wiring: MIL's `leaky_relu` op names its slope input "alpha", but op_leaky_relu reads
    # the JSON attr key "slope" specifically.
    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    alpha_var = op.inputs.get("alpha")
    slope = float(static_scalar(alpha_var, 0.01))
    nodes.append({
        "op": "LEAKY_RELU",
        "inputs": [resolve(self.safe_name(x_var_obj.name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"slope": slope},
    })


@topology_rule('reverse')
def _op_reverse(self, op, ctx):
    nodes, resolve, func_name = ctx.nodes, ctx.resolve, ctx.func_name
    # VITS's `Flip` (modules.py: `torch.flip(x, [1])`, the coupling-flow/SDP-flow chains'
    # own channel-axis reversal) -- no native ggml "reverse along an axis" primitive exists,
    # so this composes the same trick tools/convert_piper_vits/convert_vits.py's own
    # `add_flip` already uses: `ggml_get_rows(x, indices)` selects along ne[1] (its "rows"
    # axis, same convention as embedding lookup), so a compile-time-baked REVERSED index
    # array of that axis's own (always-static -- Flip only ever flips VITS's small,
    # architecture-constant channel count, 2 or `inter_channels`) size reverses it exactly.
    # Restricted to a single static ne_axis==1 flip since that's the only pattern any model
    # on this exporter's roadmap has needed; a different axis would need a PERMUTE bridge
    # first (same "cross conventions via PERMUTE+CONT" pattern used throughout this file).
    x_var = self.safe_name(op.inputs["x"].name)
    output_var = self.safe_name(op.outputs[0].name)
    axes_var = op.inputs.get("axes")
    axes_val = static_ints(axes_var)
    if axes_val is None or len(axes_val) != 1:
        raise NotImplementedError(
            f"reverse op '{op.name}' needs exactly one static axis -- multi-axis or "
            "dynamic-axis flip isn't needed by any model on this exporter's roadmap yet."
        )
    x_info = self.get_var_info(op.inputs["x"])
    ne_shape = x_info["shape"]
    rank = len(ne_shape)
    mil_axis = int(axes_val[0])
    if mil_axis < 0:
        mil_axis += rank
    ne_axis = rank - 1 - mil_axis
    if ne_axis != 1:
        raise NotImplementedError(
            f"reverse op '{op.name}' flips ne_axis={ne_axis}, but only ne_axis==1 "
            "(ggml_get_rows' own reversal axis) is composed here yet."
        )
    axis_size_raw = ne_shape[ne_axis]
    if not str(axis_size_raw).lstrip("-").isdigit():
        raise NotImplementedError(
            f"reverse op '{op.name}' flips a non-static axis ({axis_size_raw!r}) -- "
            "GET_ROWS-based reversal needs a compile-time-constant index array."
        )
    axis_size = int(axis_size_raw)

    idx_name = output_var + "_reverse_idx"
    if func_name == "main_topology" or self.flat_namespace:
        idx_full = idx_name
    else:
        idx_full = f"{func_name}.{idx_name}"
    self.weights[idx_full] = np.arange(axis_size - 1, -1, -1, dtype=np.int32)

    nodes.append({
        "op": "GET_ROWS",
        "inputs": [resolve(x_var), idx_full],
        "outputs": [output_var],
    })


@topology_rule('loom_group_norm')
def _op_loom_group_norm(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # `nn.GroupNorm` (Matcha-TTS's `Block1D`), bridged via group_norm_op.py's custom torch
    # op -- see that file's own module docstring for why (avoiding a real dynamic-axis
    # multi-reduce composition this exporter doesn't have a general capability for, when the
    # native GROUP_NORM ggml primitive already does this exact job, same "already-verified
    # primitive" precedent as loom_spline/RQ_SPLINE_INVERSE just below). `x`'s real torch
    # native shape is (B=1,C,T) -> ne=[T,C,1] here (this whole project's "batch is always 1"
    # convention) -- reshape to GROUP_NORM's own required 4D [T,1,C,1] convention (ne[2]
    # holds channels, ne[0]*ne[1] is jointly reduced per group -- see op_group_norm's own
    # comment, src/ops/primitives_basic.cpp), call it, then reshape back to x's original
    # shape (the op's own `type_inference` declares an identical output shape to its input,
    # matching RMS_NORM/LAYER_NORM's "normalization only, no affine" convention -- the real
    # per-channel weight/bias are applied by ordinary MUL/ADD ops afterward, generated by
    # `_group_norm_traceable`'s own plain tensor ops, not this custom op).
    x_var_obj = op.inputs.get("x")
    n_groups_obj = op.inputs.get("n_groups")
    eps_obj = op.inputs.get("eps")
    if x_var_obj is None:
        return
    x_info = self.get_var_info(x_var_obj)
    x_shape = list(x_info["shape"])
    if len(x_shape) < 2:
        raise NotImplementedError(
            f"loom_group_norm op '{op.name}': expected a rank>=2 [T,C,...] input (got shape "
            f"{x_shape!r})."
        )
    channels_raw = x_shape[1]
    if not str(channels_raw).lstrip("-").isdigit():
        raise NotImplementedError(
            f"loom_group_norm op '{op.name}': channel count ({channels_raw!r}) must be a "
            "static architecture constant, never a dynamic quantity."
        )
    channels = int(channels_raw)
    n_groups_val = static_scalar(n_groups_obj)
    n_groups_val = None if n_groups_val is None else int(n_groups_val)
    eps_val = float(static_scalar(eps_obj, 1e-5))
    if n_groups_val is None:
        raise NotImplementedError(
            f"loom_group_norm op '{op.name}': n_groups must be a compile-time constant."
        )

    x_name = resolve(self.safe_name(x_var_obj.name))
    output_var = self.safe_name(op.outputs[0].name)

    x4 = output_var + "_gn_x4"
    nodes.append({"op": "RESHAPE", "inputs": [x_name], "outputs": [x4],
                  "attrs": {"shape": [-1, 1, channels, 1]}})
    normed4 = output_var + "_gn_normed4"
    nodes.append({"op": "GROUP_NORM", "inputs": [x4], "outputs": [normed4],
                  "attrs": {"n_groups": n_groups_val, "eps": eps_val}})
    # Reshape back to x's own rank/shape, but with a fresh "-1" for the leading (dynamic T)
    # axis rather than reusing `x_shape[0]` verbatim: `get_var_info`'s symbolic-expression
    # fallback for a dynamic dim doesn't distinguish which CALL SITE it's resolving (this op
    # fires once per down/mid/up-block stage, each at a genuinely different T after
    # downsampling) and was confirmed to bake a STALE length from a different stage here --
    # same category of bug as the "fill"/`ones_like` one `_decoder_forward_traceable`'s own
    # docstring documents, fixed the same general way: let ggml infer the dynamic axis from
    # the real element count at build time instead of trusting a statically-guessed symbol.
    # Every OTHER axis in `x_shape` (channels, batch=1) is a genuine static architecture
    # constant, safe to reuse verbatim.
    nodes.append({"op": "RESHAPE", "inputs": [normed4], "outputs": [output_var],
                  "attrs": {"shape": [-1] + x_shape[1:]}})


@topology_rule('loom_short_conv')
def _op_loom_short_conv(self, op, ctx):
    # One SHORT_CONV node -- the engine's stateful causal depthwise convolution, and the ONLY node type
    # that can reach a ConvStateCache (BACKLOG.md P4.0.10; see `passes.py`'s `fuse_loom_short_conv`,
    # which produces this op, and src/ops/primitives_conv.cpp's `op_short_conv`, which executes it).
    #
    # No layout work here, unlike the ATTENTION rule: MIL's [b, C, seq] is ne=[seq, C, b] on this side,
    # which already IS the [n_tokens, channels] layout op_short_conv reads, and the depthwise weight's
    # MIL [C, 1, K] is ne=[K, 1, C], already the [kernel, 1, channels] it expects. The only reordering
    # is that the engine takes the kernel FIRST, matching CONV_1D_DW's own input order.
    nodes, resolve = ctx.nodes, ctx.resolve
    out = self.safe_name(op.outputs[0].name)
    x_in = resolve(self.safe_name(op.inputs["x"].name))
    w_in = resolve(self.safe_name(op.inputs["weight"].name))
    layer = int(static_scalar(op.inputs.get("layer"), 0))
    nodes.append({"op": "SHORT_CONV", "inputs": [w_in, x_in], "outputs": [out],
                  "attrs": {"layer": layer}})


@topology_rule('loom_fused_attention')
def _op_loom_fused_attention(self, op, ctx):
    # One ATTENTION node -- the engine's own composite SDPA primitive, and the ONLY node type that can
    # reach a KV cache (KV-CACHE.md stage 2; see `passes.py`'s `fuse_loom_attention`, which produces
    # this op, and src/ops/primitives_attention.cpp's `op_attention`, which executes it).
    #
    # The whole job here is layout. In MIL's forward axis order the traced q/k/v are
    # [b, heads, seq, head_dim], which is ne=[head_dim, seq, heads, b] on this side -- but op_attention
    # wants [n_embd_head, n_head, n_tokens], reading n_head_kv and the head widths straight off k/v's
    # own ne. So each of q/k/v needs its ne1/ne2 swapped, composed as PERMUTE + CONT for exactly the
    # reason `_op_matmul_x_y` gives: the C++ side never has to infer this from shapes alone, and
    # op_attention's own internal ggml_permute assumes a contiguous input in that layout.
    #
    # The mask needs no transform at all: MIL [b, 1, seq, kv] is ne=[kv, seq, 1, 1], which already IS
    # the `kq_mask` [n_kv, n_tokens] contract the bespoke converters declare by hand
    # (convert_qwen3.py's own topology inputs).
    nodes, resolve = ctx.nodes, ctx.resolve
    out = self.safe_name(op.outputs[0].name)

    def head_major(var_obj, label):
        """ne=[head_dim, seq, heads, b] -> [head_dim, heads, seq, b]."""
        name = resolve(self.safe_name(var_obj.name))
        permuted = f"{out}_{label}_hm"
        nodes.append({"op": "PERMUTE", "inputs": [name], "outputs": [permuted],
                      "attrs": {"axes": [0, 2, 1, 3]}})
        cont = f"{out}_{label}_hm_cont"
        nodes.append({"op": "CONT", "inputs": [permuted], "outputs": [cont]})
        return cont

    q_in = head_major(op.inputs["q"], "q")
    k_in = head_major(op.inputs["k"], "k")
    v_in = head_major(op.inputs["v"], "v")

    mask_obj = op.inputs.get("mask")
    if mask_obj is None:
        raise NotImplementedError(
            f"loom_fused_attention op '{op.name}': ATTENTION always takes a kq_mask input, and this op "
            "was built without one. A causal LM's mask is a declared graph input the driver fills in "
            "via loom.causal_mask -- see driver_components.CAUSAL_MASK_INPUT_NAMES."
        )
    mask_in = resolve(self.safe_name(mask_obj.name))

    scale = float(static_scalar(op.inputs.get("scale"), 1.0))
    layer = int(static_scalar(op.inputs.get("layer"), 0))
    nodes.append({
        "op": "ATTENTION",
        "inputs": [q_in, k_in, v_in, mask_in],
        "outputs": [out],
        # `kv_cache` is stated rather than left to op_attention's default-true, because this is the
        # single place in the whole exporter that decides a model gets persistent state, and a reader
        # should not have to know a C++ default to see that.
        "attrs": {"layer": layer, "scale": scale, "kv_cache": True},
    })


@topology_rule('loom_broadcast_to')
def _op_loom_broadcast_to(self, op, ctx):
    # 1:1 REPEAT -- the exact node the exporter's former ad hoc mutual-broadcast detection spliced in
    # by hand at emission time (EXPORT-ROADMAP.md R2a; see `passes.py`'s `insert_explicit_broadcasts`,
    # which inserts this op, and `loom_broadcast_to`'s own docstring in dialect.py). `op`'s own output
    # shape (via its `infer_type_with_broadcast` type inference) IS the real broadcast target, so no
    # special derivation is needed here beyond the same `get_var_info` every other REPEAT-emitting rule
    # already uses.
    ctx.nodes.append({
        "op": "REPEAT",
        "inputs": [ctx.resolve(self.safe_name(op.inputs["x"].name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"shape": self.get_var_info(op.outputs[0])["shape"]},
    })


@topology_rule('loom_spline')
def _op_loom_spline(self, op, ctx):
    nodes, resolve, func_name = ctx.nodes, ctx.resolve, ctx.func_name
    # VITS's rational-quadratic spline inverse (StochasticDurationPredictor's ConvFlow) --
    # see tools/loom_mil_compiler/vits_spline_op.py's module docstring for why this is a
    # custom torch/MIL op at all (the real implementation's boolean-mask tensor indexing
    # can't be traced correctly). Composes down to the already-verified RQ_SPLINE_INVERSE
    # ggml primitive (src/ops/primitives_spline.cpp), which expects `inputs` [n_tokens],
    # `unnormalized_widths`/`unnormalized_heights` [num_bins, n_tokens],
    # `unnormalized_derivatives` [num_bins-1, n_tokens] -- x1/uw/uh/ud's leading torch dims
    # are always [batch=1, half_channels=1, ...] in this project's single-utterance/
    # mean-only convention, so ggml's own reversed-shape convention already puts num_bins
    # (or num_bins-1) at ne0 and n_tokens at ne1 with no permute needed, just a squeeze via
    # RESHAPE. `boundary_deriv_const`/`eps_bump` are conversion-time-baked constants
    # (depend only on num_bins/min_derivative), matching
    # tools/convert_piper_vits/convert_vits.py's own add_conv_flow_reverse construction 1:1.
    from .vits_spline_op import TAIL_BOUND, MIN_BIN_WIDTH, MIN_BIN_HEIGHT, MIN_DERIVATIVE

    x_var = self.safe_name(op.inputs["x"].name)
    w_var = self.safe_name(op.inputs["w"].name)
    h_var = self.safe_name(op.inputs["h"].name)
    d_var = self.safe_name(op.inputs["d"].name)
    output_var = self.safe_name(op.outputs[0].name)

    w_info = self.get_var_info(op.inputs["w"])
    num_bins_raw = w_info["shape"][0]  # get_var_info stringifies every shape entry
    if not str(num_bins_raw).lstrip("-").isdigit():
        raise NotImplementedError(
            f"loom_spline op '{op.name}' has a non-static num_bins ({num_bins_raw!r}) -- "
            "this is always an architecture constant (ConvFlow's own num_bins=10), never "
            "a real dynamic quantity."
        )
    num_bins = int(num_bins_raw)

    x_flat = output_var + "_spl_x"
    nodes.append({"op": "RESHAPE", "inputs": [resolve(x_var)], "outputs": [x_flat],
                  "attrs": {"shape": ["$n_tokens"]}})
    w_flat = output_var + "_spl_w"
    nodes.append({"op": "RESHAPE", "inputs": [resolve(w_var)], "outputs": [w_flat],
                  "attrs": {"shape": [num_bins, "$n_tokens"]}})
    h_flat = output_var + "_spl_h"
    nodes.append({"op": "RESHAPE", "inputs": [resolve(h_var)], "outputs": [h_flat],
                  "attrs": {"shape": [num_bins, "$n_tokens"]}})
    d_flat = output_var + "_spl_d"
    nodes.append({"op": "RESHAPE", "inputs": [resolve(d_var)], "outputs": [d_flat],
                  "attrs": {"shape": [num_bins - 1, "$n_tokens"]}})

    boundary_const = float(np.log(np.exp(1 - MIN_DERIVATIVE) - 1))
    boundary_deriv_const = np.zeros(num_bins + 1, dtype=np.float32)
    boundary_deriv_const[0] = boundary_const
    boundary_deriv_const[-1] = boundary_const
    eps_bump = np.zeros(num_bins, dtype=np.float32)
    eps_bump[-1] = 1e-6

    bdc_name = output_var + "_boundary_deriv_const"
    eps_name = output_var + "_eps_bump"
    if func_name == "main_topology" or self.flat_namespace:
        bdc_full, eps_full = bdc_name, eps_name
    else:
        bdc_full, eps_full = f"{func_name}.{bdc_name}", f"{func_name}.{eps_name}"
    self.weights[bdc_full] = boundary_deriv_const
    self.weights[eps_full] = eps_bump

    spline_out = output_var + "_spl_out"
    nodes.append({
        "op": "RQ_SPLINE_INVERSE",
        "inputs": [x_flat, w_flat, h_flat, d_flat, bdc_full, eps_full],
        "outputs": [spline_out],
        "attrs": {
            "tail_bound": TAIL_BOUND, "min_bin_width": MIN_BIN_WIDTH,
            "min_bin_height": MIN_BIN_HEIGHT, "min_derivative": MIN_DERIVATIVE,
        },
    })

    out_info = self.get_var_info(op.outputs[0])
    nodes.append({"op": "RESHAPE", "inputs": [spline_out], "outputs": [output_var],
                  "attrs": {"shape": list(out_info["shape"])}})


@topology_rule('split')
def _op_split(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Compose split as multiple zero-copy VIEW slices
    x_var = self.safe_name(op.inputs["x"].name)
    axis = static_value(op.inputs.get("axis"), 0)

    # Retrieve input shape info (ne-reversed shape)
    x_info = self.get_var_info(op.inputs["x"])
    ne_shape = x_info["shape"]
    rank = len(ne_shape)

    # Normalize negative axis relative to the tensor rank
    if axis < 0:
        axis = rank + axis

    # Map MIL standard axis to Loom ne-reversed axis
    ne_axis = rank - 1 - axis
    num_splits = len(op.outputs)
    dim_to_split = ne_shape[ne_axis]

    if isinstance(dim_to_split, int):
        split_dim_size = dim_to_split // num_splits
        split_expr = as_expr(split_dim_size)
    else:
        split_expr = as_expr(dim_to_split) / num_splits
        split_dim_size = render(split_expr)

    # Create a VIEW node for each split output
    for idx, out_var in enumerate(op.outputs):
        out_name = self.safe_name(out_var.name)

        slice_shape = list(ne_shape)
        slice_shape[ne_axis] = split_dim_size

        # Calculate byte offset rule
        offset_elements = idx * split_expr
        for prev_ax in range(ne_axis):
            offset_elements = offset_elements * as_expr(ne_shape[prev_ax])
        offset_bytes = render(offset_elements * 4)  # 4 bytes per float element

        nodes.append({
            "op": "VIEW",
            "inputs": [resolve(x_var)],
            "outputs": [out_name],
            "attrs": {
                "shape": slice_shape,
                "offset": offset_bytes
            }
        })


@topology_rule('slice_by_index')
def _op_slice_by_index(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Compose slice_by_index as an optimized zero-copy VIEW node
    x_var = self.safe_name(op.inputs["x"].name)
    output_var = self.safe_name(op.outputs[0].name)

    begin_var = op.inputs.get("begin")
    end_var = op.inputs.get("end")
    begin_mask = static_value(op.inputs.get("begin_mask"))
    end_mask = static_value(op.inputs.get("end_mask"))

    x_info = self.get_var_info(op.inputs["x"])
    ne_shape = x_info["shape"]
    rank = len(ne_shape)

    begin_mask_list = list(begin_mask) if isinstance(begin_mask, (list, tuple, np.ndarray)) else None
    end_mask_list = list(end_mask) if isinstance(end_mask, (list, tuple, np.ndarray)) else None

    # Resolve each MIL-order axis's real (begin, end) via `facts.slice_axis_value` (which
    # handles BOTH a plain constant array and a dynamic `concat`/`stack` of per-axis
    # gather-derived scalars -- see its own docstring), honoring begin_mask/end_mask ("ignore
    # this value, use the full extent on this side" -- MIL's own convention for e.g.
    # `x[1:]`/`x[:-1]`; confirmed via a real op: `end=[1, 0], end_mask=[True, True]` for
    # `waveform[:, 1:]`, where the literal `end=0` is a meaningless placeholder an earlier
    # version of this code used verbatim) and normalizing genuine negative (Python-style)
    # indices for BOTH the concrete-int and symbolic (dynamic-length) dim_size cases.
    # Previously, whenever `begin`/`end` had no plain literal `.val` array at all (a fully
    # dynamic concat, e.g. `rel_shift`'s final `matrix_bd[..., :matrix_ac.size(-1)]` crop, or
    # the positional-encoding table's `pe[:, start:end]`), this code discarded the ENTIRE
    # begin/end array and fell back to copying the PARENT's own unsliced extent on every
    # single axis -- silently turning a real crop into a no-op on whichever axis needed it.
    resolved_begin = [None] * rank
    resolved_end = [None] * rank
    for mil_axis in range(rank):
        ne_axis = rank - 1 - mil_axis
        dim_size = ne_shape[ne_axis]
        is_begin_masked = begin_mask_list is not None and mil_axis < len(begin_mask_list) and bool(begin_mask_list[mil_axis])
        is_end_masked = end_mask_list is not None and mil_axis < len(end_mask_list) and bool(end_mask_list[mil_axis])
        b_val = 0 if is_begin_masked else self.facts.slice_axis_value(begin_var, mil_axis)
        e_val = dim_size if is_end_masked else self.facts.slice_axis_value(end_var, mil_axis)
        if b_val is None:
            b_val = 0
        elif isinstance(b_val, (int, np.integer)) and b_val < 0:
            # A negative (Python-style) index normalizes against the axis extent, which may itself be a
            # symbolic expression -- composed as algebra rather than as an f-string, so a plain
            # `x[..., :-1]` on a dynamic axis comes out as `n_tokens - 1` rather than nested parentheses.
            b_val = (dim_size + int(b_val)) if isinstance(dim_size, int) else as_expr(dim_size) + int(b_val)
        if e_val is None:
            e_val = dim_size
        elif isinstance(e_val, (int, np.integer)) and e_val < 0:
            e_val = (dim_size + int(e_val)) if isinstance(dim_size, int) else as_expr(dim_size) + int(e_val)
        resolved_begin[mil_axis] = b_val
        resolved_end[mil_axis] = e_val

    slice_shape = []
    for i in range(rank):
        mil_axis = rank - 1 - i
        dim_size = ne_shape[i]
        b_val = resolved_begin[mil_axis]
        e_val = resolved_end[mil_axis]

        # `(end - begin)` on EVERY axis, not just ne_axis 0 -- a real, non-fastest-varying
        # axis can be genuinely sliced too (confirmed on Conformer-CTC's `rel_shift`:
        # `x[:, :, 1:]` slices torch axis 2, which is ne_axis 1 on this rank-4 tensor, not
        # ne_axis 0). The previous version special-cased ONLY ne_axis 0 as "the sliced axis"
        # and blindly copied every other axis's size straight from the parent, silently
        # ignoring a real slice on any other axis and emitting the parent's FULL (unsliced)
        # size there instead -- confirmed wrong via the exact same rel_shift slice: the VIEW's
        # own declared shape kept `pos_len+1` (34) instead of the real sliced `pos_len` (33),
        # a genuine element-count bug (not just a shape-string cosmetic issue) since the VIEW
        # itself still only carves out `pos_len` elements' worth of bytes on that axis via its
        # own stride math -- the declared shape and the real memory layout disagreed.
        # `b_val==0 and e_val==dim_size` (the common "no real slicing here" case) computes
        # right back to `dim_size` anyway, so this is a strict generalization, not a behavior
        # change for any axis that was already correct.
        if isinstance(dim_size, int) and isinstance(b_val, int) and isinstance(e_val, int):
            b_val = max(0, min(dim_size, b_val))
            e_val = max(0, min(dim_size, e_val))
            slice_shape.append(e_val - b_val)
        elif b_val == 0 and as_expr(e_val) == as_expr(dim_size):
            # Structural equality on the expressions, not on their printed forms: two spellings of the
            # same length ("n_tokens" vs "(n_tokens)") used to look like a real slice and emit a
            # pointless `(end - begin)` string for an axis that is not sliced at all.
            slice_shape.append(dim_size)
        else:
            slice_shape.append(render(as_expr(e_val) - as_expr(b_val)))

    # Calculate byte offset in C-major MIL layout mapping to ne_shape strides. Uses
    # `resolved_begin` (mask-aware, negative-index-normalized) rather than the raw
    # `begin_list` for the same reason the shape derivation above needs it: an
    # ignored/negative begin must contribute its real (0 or normalized) value, not its raw
    # MIL-op placeholder.
    offset_elements = as_expr(0)
    for i in range(rank):
        b_val = resolved_begin[i]
        if not (isinstance(b_val, int) and b_val == 0):
            stride_product = as_expr(1)
            ne_limit = rank - 1 - i
            for prev_ax in range(ne_limit):
                stride_product = stride_product * as_expr(ne_shape[prev_ax])
            offset_elements = offset_elements + as_expr(b_val) * stride_product

    offset_bytes = render(offset_elements * 4)

    nodes.append({
        "op": "VIEW",
        "inputs": [resolve(x_var)],
        "outputs": [output_var],
        "attrs": {
            "shape": slice_shape,
            "offset": offset_bytes
        }
    })


@topology_rule('fill')
def _op_fill(self, op, ctx):
    nodes, func_name = ctx.nodes, ctx.func_name
    # Compile-time evaluation of constant fill tensors
    shape_val = static_value(op.inputs.get("shape"))
    value_val = static_value(op.inputs.get("value"), 0.0)

    if shape_val is not None:
        shape_list = list(shape_val) if isinstance(shape_val, (list, tuple, np.ndarray)) else [shape_val]
        ne_shape = list(reversed(shape_list))

        array = np.full(ne_shape, value_val, dtype=np.float32)
        weight_name = self.safe_name(op.outputs[0].name)
        self.weights[weight_name] = array
        return
    else:
        # Dynamic fill (`shape` isn't compile-time-constant, e.g. `torch.full` sized off a
        # dynamic-length tensor): compose from a REPEAT-broadcast of a genuinely scalar (all
        # axes size 1) constant, the same mechanism this exporter already uses for every
        # other dynamically-shaped REPEAT target (see the "tile" op_type branch above) --
        # `get_var_info`'s own per-axis shape already mixes literal ints with symbolic
        # expressions ("n_tokens" and derivatives) correctly, one entry per axis.
        #
        # The previous approach here (pre-allocate a `[4096]*rank` constant buffer and VIEW-
        # slice into it, assuming EVERY axis was independently exactly "n_tokens") was only
        # ever exercised at rank<=1 (a 4096-element 1D buffer, 16 KiB): at rank>=2 it
        # allocates `4096**rank` elements outright (256 GiB at rank 3), and even where it
        # doesn't blow up, blindly slicing every axis to "n_tokens" is wrong for any fill
        # whose shape has a mix of static and dynamic axes.
        # Prefer resolving straight from `fill`'s own "shape" input (same mechanism
        # `facts.reshape_shape` already provides for `reshape`, reused here
        # since a dynamic `fill`'s "shape" input has the exact same concat-of-gathers/
        # constant-array structure) -- `get_var_info`'s out_info-based fallback below has no
        # per-axis correspondence formula for `fill` (not in `_infer_dynamic_dim_expr`'s
        # handled op set), so it blindly substitutes EVERY symbolic axis to "n_tokens".
        # Confirmed wrong on Conformer-CTC's length-validity mask fill: a rank-2 `[T, T]`
        # fill (T = the subsampled frame count) got BOTH axes collapsed to the raw sample
        # count instead, producing a wildly oversized mask.
        resolved_torch_shape = self.facts.reshape_shape(op)
        if resolved_torch_shape is not None:
            target_shape = list(reversed(resolved_torch_shape))
        else:
            out_info = self.get_var_info(op.outputs[0])
            target_shape = list(out_info["shape"])
        rank = len(target_shape)

        weight_name = self.safe_name(op.outputs[0].name) + "_fill_scalar"
        if func_name == "main_topology" or self.flat_namespace:
            namespaced_name = weight_name
        else:
            namespaced_name = f"{func_name}.{weight_name}"
        self.weights[namespaced_name] = np.full([1] * rank, value_val, dtype=np.float32)

        nodes.append({
            "op": "REPEAT",
            "inputs": [namespaced_name],
            "outputs": [self.safe_name(op.outputs[0].name)],
            "attrs": {"shape": target_shape}
        })


@topology_rule('pad')
def _op_pad(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # EXPORT-IMPROVEMENT-BACKLOG.md item 4: the one op type STFT's own center-framing
    # (torch.stft(..., center=True)) decomposes into, via coremltools' own
    # common::lower_complex_dialect_ops pass, that this exporter didn't already handle --
    # confirmed by directly tracing a small torch.stft module through the standard
    # ct.convert(..., convert_to="milinternal") pipeline (with thread 1's
    # apply_torch_frontend_patches() active) and inspecting the resulting MIL ops: the whole
    # decomposition is const/cast/reshape/conv/expand_dims/pad, every one of which this
    # exporter already covers except this. Only PAD_1D/PAD_1D_REFLECT (ne[0]-only, i.e. MIL's
    # LAST axis) exist in C++ -- anything padding a different axis, or a non-zero constant
    # fill, or a mode other than constant/reflect, has no ggml-side implementation yet.
    x_var_obj = op.inputs.get("x") or op.inputs.get("data")
    x_var = self.safe_name(x_var_obj.name)
    output_var = self.safe_name(op.outputs[0].name)

    pad_var = op.inputs.get("pad")
    if static_value(pad_var) is None:
        raise NotImplementedError(
            f"pad op '{op.name}' has a non-constant 'pad' input, which this exporter doesn't support."
        )
    pad_vals = static_ints(pad_var)
    if len(pad_vals) % 2 != 0:
        raise NotImplementedError(f"pad op '{op.name}' has an odd-length 'pad' array {pad_vals!r}.")
    n_padded = len(pad_vals) // 2

    if not hasattr(x_var_obj, "shape") or x_var_obj.shape is None:
        raise NotImplementedError(f"pad op '{op.name}' has an input with no known rank.")
    rank = len(x_var_obj.shape)

    mode_var = op.inputs.get("mode")
    mode = static_value(mode_var, "constant")

    # MIL's "pad" only pads the LAST n_padded dims of x: pad[2*i]/pad[2*i+1] apply to MIL
    # axis (rank - n_padded + i). Only a non-zero pad on the FASTEST-varying axis (ne[0],
    # i.e. MIL's last axis) is supported -- exactly what PAD_1D/PAD_1D_REFLECT's ggml kernels
    # (and every real case seen so far, STFT included) operate on.
    lp0 = rp0 = 0
    for i in range(n_padded):
        mil_axis = rank - n_padded + i
        lp, rp = pad_vals[2 * i], pad_vals[2 * i + 1]
        if lp == 0 and rp == 0:
            continue
        if mil_axis != rank - 1:
            raise NotImplementedError(
                f"pad op '{op.name}' pads MIL axis {mil_axis} (non-zero {lp}/{rp}), but this "
                "exporter only supports padding the fastest-varying axis (ne[0]/MIL's last "
                "axis) -- padding any other axis needs a new C++ primitive first."
            )
        lp0, rp0 = lp, rp

    if mode == "constant":
        constant_val_var = op.inputs.get("constant_val")
        constant_val = float(static_scalar(constant_val_var, 0.0))
        if constant_val != 0.0:
            raise NotImplementedError(
                f"pad op '{op.name}' has mode='constant' with a non-zero constant_val="
                f"{constant_val} -- PAD_1D only supports zero-fill."
            )
        mapped_op = "PAD_1D"
    elif mode == "reflect":
        mapped_op = "PAD_1D_REFLECT"
    elif mode == "replicate":
        # `passes.py`'s `canonicalize_replicate_pad` (EXPORT-ROADMAP.md R2) rewrites every
        # mode="replicate" pad into a `loom_replicate_pad` op before this table ever runs -- see that
        # pass and the `loom_replicate_pad` rule just below. Reaching here means it didn't run.
        raise NotImplementedError(
            f"pad op '{op.name}' has mode='replicate' -- passes.py's canonicalize_replicate_pad should "
            "have already rewritten this into a loom_replicate_pad op before this table ever ran."
        )
    else:
        raise NotImplementedError(
            f"pad op '{op.name}' has mode='{mode}', which this exporter doesn't support "
            "(only 'constant', 'reflect', and 'replicate' are)."
        )

    nodes.append({
        "op": mapped_op,
        "inputs": [resolve(x_var)],
        "outputs": [output_var],
        "attrs": {"lp0": lp0, "rp0": rp0},
    })


@topology_rule('loom_replicate_pad')
def _op_loom_replicate_pad(self, op, ctx):
    """1:1 port of the exporter's former ad hoc `pad(mode="replicate")` composition (EXPORT-ROADMAP.md
    R2; see `loom_replicate_pad`'s own docstring in dialect.py and `passes.py`'s
    `canonicalize_replicate_pad`, which inserts this op): VIEW out the boundary column, REPEAT-broadcast
    it to the pad width, CONCAT it back on. `lp`/`rp` are always static; only the VIEW extracting the
    RIGHT edge needs a dynamic byte offset (the left edge is always byte 0), via the same
    `_infer_dynamic_dim_expr` backward walk every other dynamic-offset VIEW in this exporter depends on.
    """
    nodes, resolve = ctx.nodes, ctx.resolve
    x_var_obj = op.inputs["x"]
    x_var = self.safe_name(x_var_obj.name)
    output_var = self.safe_name(op.outputs[0].name)
    lp0 = int(static_value(op.inputs["lp"]))
    rp0 = int(static_value(op.inputs["rp"]))
    rank = len(x_var_obj.shape)

    x_info = self.get_var_info(x_var_obj)
    ne_rest = list(x_info["shape"][1:])
    cur = resolve(x_var)
    if lp0 > 0:
        left_edge = f"{output_var}_replpad_left_edge"
        nodes.append({
            "op": "VIEW", "inputs": [cur], "outputs": [left_edge],
            "attrs": {"shape": [1, *ne_rest], "offset": 0},
        })
        left_tile = f"{output_var}_replpad_left_tile"
        nodes.append({
            "op": "REPEAT", "inputs": [left_edge], "outputs": [left_tile],
            "attrs": {"shape": [lp0, *ne_rest]},
        })
        left_cat = f"{output_var}_replpad_left_cat"
        nodes.append({
            "op": "CONCAT", "inputs": [left_tile, cur], "outputs": [left_cat],
            "attrs": {"dim": 0},
        })
        cur = left_cat
    if rp0 > 0:
        t_expr = self._infer_dynamic_dim_expr(x_var_obj, rank - 1)
        right_edge = f"{output_var}_replpad_right_edge"
        nodes.append({
            "op": "VIEW", "inputs": [resolve(x_var)], "outputs": [right_edge],
            "attrs": {"shape": [1, *ne_rest], "offset": render((t_expr - 1) * 4)},
        })
        right_tile = f"{output_var}_replpad_right_tile"
        nodes.append({
            "op": "REPEAT", "inputs": [right_edge], "outputs": [right_tile],
            "attrs": {"shape": [rp0, *ne_rest]},
        })
        nodes.append({
            "op": "CONCAT", "inputs": [cur, right_tile], "outputs": [output_var],
            "attrs": {"dim": 0},
        })
    else:
        ctx.aliases[output_var] = cur


@topology_rule('band_part')
def _op_band_part(self, op, ctx):
    nodes, aliases, resolve = ctx.nodes, ctx.aliases, ctx.resolve
    # Map band_part (with lower=-1, upper=0) to DIAG_MASK_ZERO. MIL's actual input keys
    # are "lower"/"upper" (see tensor_operation.py's band_part InputSpec) -- NOT
    # "num_lower"/"num_upper", which never matched, so this always silently used the
    # (coincidentally causal-shaped) -1/0 defaults regardless of the op's real attrs. Same
    # bug class as the transpose/"perm" mismatch above; not currently exercised by LFM2
    # (its causal mask gets constant-folded at trace time rather than computed via a live
    # band_part op), but a real latent bug for any model that reaches this path.
    num_lower = static_value(op.inputs.get("lower"), -1)
    num_upper = static_value(op.inputs.get("upper"), 0)

    x_var = self.safe_name(op.inputs["x"].name)
    output_var = self.safe_name(op.outputs[0].name)

    if num_lower == -1 and num_upper == 0:
        # It's a lower-triangle zero mask (causal zero mask)
        nodes.append({
            "op": "DIAG_MASK_ZERO",
            "inputs": [resolve(x_var)],
            "outputs": [output_var],
            "attrs": {"n_past": 0}
        })
    else:
        # Keep all (no-op / alias)
        aliases[output_var] = resolve(x_var)


@topology_rule('transpose')
def _op_transpose(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Map transpose to PERMUTE with ne-reversed axes
    x_var = self.safe_name(op.inputs["x"].name)
    output_var = self.safe_name(op.outputs[0].name)

    x_info = self.get_var_info(op.inputs["x"])
    rank = len(x_info["shape"])

    # MIL's transpose op names its (required) permutation input "perm", not "axes" --
    # checking for "axes" here always missed, silently falling back to an identity
    # permutation for EVERY transpose in the model (confirmed on LFM2: every single
    # PERMUTE node in both the ShortConv and attention layers was emitted as a no-op
    # [0,1,2,3], which is what caused the numerical mismatch against the real model).
    perm_var = op.inputs.get("perm") or op.inputs.get("axes")
    if static_value(perm_var) is None:
        raise ValueError(f"transpose op '{op.name}' has no resolvable 'perm' constant")
    # perm entries may be negative (confirmed on LFM2: e.g. [0, -1, -2] for .transpose(-1,-2)).
    raw_perm = static_ints(perm_var)
    norm_perm = [(p + rank) if p < 0 else p for p in raw_perm]

    # MIL's semantics: output.shape[i] = input.shape[norm_perm[i]] -- destination axis i
    # PULLS FROM source axis norm_perm[i]. ggml_permute's signature is the opposite
    # direction: ggml_permute(x, axis0..axis3) means "source ne[k] MOVES TO dest ne[axis_k]"
    # (see ggml.c: result->ne[axis_k] = a->ne[k]). Converting one to the other needs the
    # INVERSE permutation, not a direct pass-through -- on top of the usual ne-order axis
    # reversal (MIL axis a <-> ne-axis rank-1-a). Passing norm_perm straight through
    # (reversed only) silently permuted every transpose in the model incorrectly.
    inv_perm = [0] * rank
    for i, p in enumerate(norm_perm):
        inv_perm[p] = i
    ne_axes = []
    for k in range(rank):
        a = rank - 1 - k       # MIL input axis feeding ggml source axis k
        b = inv_perm[a]        # MIL output axis that input axis a lands at
        ne_axes.append(rank - 1 - b)
    while len(ne_axes) < 4:
        ne_axes.append(len(ne_axes))

    nodes.append({
        "op": "PERMUTE",
        "inputs": [resolve(x_var)],
        "outputs": [output_var],
        "attrs": {"axes": ne_axes}
    })


@topology_rule('tile')
def _op_tile(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # GQA repeat_kv() fusion (the case where this "tile"'s own `reps` isn't a compile-time
    # constant, which otherwise poisons this node's -- and the following reshape's -- shape
    # inference) is now handled as a real MIL->MIL pass in passes.py, run once over the whole
    # program before this walk ever starts (EXPORT-IMPROVEMENT-BACKLOG.md item 3). By the time
    # a "tile" op reaches this generic per-op handling, either it was never part of that
    # pattern, or it already got rewritten into a plain reshape(shape, 1)->tile(ratio)->reshape
    # sequence with fully reliable (concrete-int-or-single-dynamic-symbol) shapes -- so no
    # special-casing is needed here anymore.
    # Map tile to REPEAT by calculating the target shape
    x_var = self.safe_name(op.inputs["x"].name)
    reps = static_value(op.inputs.get("reps"), [1])

    # Retrieve input shape info (ne-reversed shape)
    x_info = self.get_var_info(op.inputs["x"])
    ne_shape = x_info["shape"]
    rank = len(ne_shape)

    reps_list = list(reps) if isinstance(reps, (list, tuple, np.ndarray)) else [reps]
    if len(reps_list) > rank:
        reps_list = reps_list[-rank:]
    while len(reps_list) < rank:
        reps_list.insert(0, 1)

    # Target shape in ne-order
    target_shape = []
    for i in range(rank):
        mil_axis = rank - 1 - i
        dim_size = ne_shape[i]
        rep_factor = reps_list[mil_axis]
        if rep_factor is None:
            rep_factor = 1

        if rep_factor == 1:
            # A rep factor of 1 is a no-op regardless of dim_size's type -- skip the
            # multiplication wrapper entirely rather than emitting a redundant "(dim * 1)"
            # expression string for a dynamic dim_size (harmless once evaluated by
            # SymbolEnv, but needlessly opaque, and the GQA repeat_kv() fusion pass in
            # passes.py routinely produces exactly this shape -- reps=1 on every
            # unchanged axis, ratio only on the newly-inserted one).
            target_shape.append(dim_size)
            continue
        try:
            dim_int = int(dim_size)
            target_shape.append(str(dim_int * rep_factor))
        except (ValueError, TypeError):
            target_shape.append(render(as_expr(dim_size) * rep_factor))

    # Limit target shape strictly to 4D to satisfy GGML's maximum dimension limits
    while len(target_shape) > 4 and target_shape[-1] == "1":
        target_shape.pop()
    if len(target_shape) > 4:
        target_shape = target_shape[:4]

    nodes.append({
        "op": "REPEAT",
        "inputs": [resolve(x_var)],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"shape": target_shape}
    })


@topology_rule('squeeze')
def _op_squeeze(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Dedicated branch (NOT the generic "reshape"/"expand_dims"/"squeeze" rank-reducing path
    # below): that path's rank-REDUCING case assumes a rank reduction always means "merge
    # several trailing MIL axes (leading ne-order axes) into one" (built for e.g. a multi-head
    # attention output's heads*head_dim merge) and blindly applies `target_shape = [-1] +
    # x_info["shape"][merge_count:]` -- WRONG for squeeze, which drops a SPECIFIC, already-
    # size-1 axis (most commonly ne-order's OWN LAST axis, from squeezing torch axis 0/batch)
    # rather than folding two real-sized axes together. Confirmed on SupertonicTTS's
    # SpeechPromptedCrossAttention (`torch.cat([o0,o1],dim=-1).squeeze(0)`, a (1,1,T,256) ->
    # (1,T,256) squeeze): the generic path's formula computed `target_shape=[-1,1,1]` (merging
    # ne-order axes 0-1 into one flat 2560-element blob) instead of the correct "just drop the
    # LAST ne-order axis, keep [256,T,1] unchanged" -- a real MUL_MAT shape-mismatch crash
    # downstream, not merely cosmetic. Since every squeezed axis is PROVABLY size 1 (squeeze's
    # own contract), no -1 inference is ever needed here: the target shape is always an exact
    # positional deletion from the input's own (reliable) shape.
    x_var_obj = op.inputs.get("x") or op.inputs.get("data")
    x_var = self.safe_name(x_var_obj.name)
    x_info = self.get_var_info(x_var_obj)
    in_rank = len(x_info["shape"])

    axes_var = op.inputs.get("axes")
    if static_value(axes_var) is not None:
        torch_axes = static_ints(axes_var)
    else:
        # No explicit axes -- squeeze every static size-1 axis (mirrors numpy/torch's own
        # "squeeze all size-1 dims" default when `dim` is omitted).
        torch_axes = [in_rank - 1 - i for i, d in enumerate(x_info["shape"]) if str(d) == "1"]
    ne_axes_to_drop = set()
    for a in torch_axes:
        if a < 0:
            a += in_rank
        ne_axes_to_drop.add(in_rank - 1 - a)

    target_shape = [d for i, d in enumerate(x_info["shape"]) if i not in ne_axes_to_drop]
    if not target_shape:
        target_shape = [1]

    nodes.append({
        "op": "RESHAPE",
        "inputs": [resolve(x_var)],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"shape": target_shape}
    })


@topology_rule('reshape', 'expand_dims')
def _op_reshape_expand_dims(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    op_type = op.op_type
    # Map reshape/expand_dims/squeeze to RESHAPE with 1 input and a shape attribute derived
    # from the op's own declared output shape.
    x_var_obj = op.inputs.get("x") or op.inputs.get("data")
    x_var = self.safe_name(x_var_obj.name)

    out_var = op.outputs[0]
    out_info = self.get_var_info(out_var)
    # Same ne-order reversal get_var_info applies, kept aligned 1:1 with out_info["shape"] so
    # raw_dims[i] is the exact MIL-side (pre-substitution) source of out_info["shape"][i].
    raw_dims = list(reversed(out_var.shape)) if hasattr(out_var, "shape") and out_var.shape is not None else []

    x_info = self.get_var_info(x_var_obj) if hasattr(x_var_obj, "shape") and x_var_obj.shape is not None else None
    out_rank = len(out_info["shape"])
    in_rank = len(x_info["shape"]) if x_info is not None else -1

    resolved_torch_shape = self.facts.reshape_shape(op) if op_type == "reshape" else None
    if resolved_torch_shape is not None and len(resolved_torch_shape) == out_rank:
        # The "shape" input's own real per-axis values resolved directly -- strictly more
        # trustworthy than either branch below (both of which only ever look at the output
        # var's OWN, possibly-fresh-and-unrelated, symbolic shape). See
        # `facts.reshape_shape`'s docstring for why that matters.
        target_shape = list(reversed(resolved_torch_shape))
        inferred_count = 0
    elif x_info is not None and in_rank > out_rank:
        # A rank-REDUCING reshape (merging several trailing MIL axes -- i.e. several LEADING
        # ne-order axes -- into one). Coremltools' own output-shape inference for the merged
        # axis is fundamentally untrustworthy here: it isn't just that a single symbol might
        # be a genuinely-different static quantity reported symbolically (confirmed on LFM2's
        # attention-output reshape merging (heads=16, head_dim=64) into hidden_size=1024,
        # reported as a lone symbol that is NOT n_tokens); coremltools also renumbers even
        # perfectly-unchanged pass-through axes (confirmed: this reshape's own INPUT seq axis
        # and OUTPUT seq axis are two DIFFERENT symbol objects, despite being the identical
        # unchanged quantity) -- so there is no symbol-identity check that can tell "reliable"
        # apart from "unreliable" here, from the output's own shape alone. Instead, derive the
        # target POSITIONALLY from the input's OWN (reliable) shape: every ne-order axis
        # except the merged one is a direct, unchanged carry-over from the input at the
        # corresponding position; the merged axis (always ne-order axis 0, since merging
        # collapses the trailing MIL axes = leading ne-order axes) is a literal -1, delegating
        # to op_reshape's own numpy/PyTorch-style inference (src/ops/primitives_basic.cpp) --
        # correct by construction from the input's real total element count at build time.
        merge_count = in_rank - out_rank + 1
        target_shape = [-1] + list(x_info["shape"][merge_count:])
        inferred_count = 1
    else:
        # Rank-preserving (or rank-increasing/split) reshape: a BARE pass-through symbol
        # (exactly one distinct symbol occupying the whole dim, no arithmetic) has
        # consistently been the genuine n_tokens quantity in every such case seen so far
        # (position_ids/cache_position reshapes, RoPE cos/sin, Q/K/V head-splits, etc.) --
        # unlike the rank-REDUCING merge case above, a split/pass-through doesn't fabricate a
        # NEW computed quantity the way a merge does, so there's no reason to distrust it.
        # Only a genuine multi-symbol ARITHMETIC EXPRESSION (2+ distinct symbols) is treated
        # as unreliable here, exactly as originally established.
        target_shape = []
        inferred_count = 0
        for i, d in enumerate(out_info["shape"]):
            raw_str = str(raw_dims[i]) if i < len(raw_dims) else d
            if len(set(_DYNAMIC_SYMBOL_RE.findall(raw_str))) > 1:
                target_shape.append(-1)
                inferred_count += 1
            else:
                target_shape.append(d)
    if inferred_count > 1:
        raise NotImplementedError(
            f"reshape op '{op.name}' needs more than one inferred (multi-symbol) dimension "
            f"in target shape {out_info['shape']!r} -- RESHAPE only supports a single -1 entry."
        )

    # Limit target shape strictly to 4D to satisfy GGML's maximum dimension limits
    while len(target_shape) > 4 and target_shape[-1] == "1":
        target_shape.pop()
    if len(target_shape) > 4:
        target_shape = target_shape[:4]

    nodes.append({
        "op": "RESHAPE",
        "inputs": [resolve(x_var)],
        "outputs": [self.safe_name(out_var.name)],
        "attrs": {"shape": target_shape}
    })


@topology_rule('concat')
def _op_concat(self, op, ctx):
    nodes, aliases, resolve = ctx.nodes, ctx.aliases, ctx.resolve
    # CONCAT in Loom expects strictly 2 inputs.
    # For 3+ inputs, we chain multiple 2-input CONCAT nodes sequentially!
    values_obj = op.inputs.get("values")
    if values_obj:
        inputs = []
        for item in values_obj:
            if isinstance(item, Var):
                inputs.append(resolve(self.safe_name(item.name)))

        if len(inputs) > 2:
            prev_output = inputs[0]
            output_var = self.safe_name(op.outputs[0].name)

            axis = static_value(op.inputs.get("axis"), 0)
            rank = len(self.get_var_info(op.outputs[0])["shape"])
            if axis < 0:
                axis = rank + axis
            ne_axis = rank - 1 - axis

            for i in range(1, len(inputs) - 1):
                inter_output = f"{output_var}_concat_temp_{i}"
                nodes.append({
                    "op": "CONCAT",
                    "inputs": [prev_output, inputs[i]],
                    "outputs": [inter_output],
                    "attrs": {"dim": ne_axis}
                })
                prev_output = inter_output

            nodes.append({
                "op": "CONCAT",
                "inputs": [prev_output, inputs[-1]],
                "outputs": [output_var],
                "attrs": {"dim": ne_axis}
            })
            return
        elif len(inputs) == 2:
            axis = static_value(op.inputs.get("axis"), 0)
            rank = len(self.get_var_info(op.outputs[0])["shape"])
            if axis < 0:
                axis = rank + axis
            ne_axis = rank - 1 - axis
            nodes.append({
                "op": "CONCAT",
                "inputs": inputs,
                "outputs": [self.safe_name(op.outputs[0].name)],
                "attrs": {"dim": ne_axis}
            })
            return
        elif len(inputs) == 1:
            # A single real operand -- e.g. HF's KV-cache update
            # (`torch.cat([past_key_states, key_states], dim=-2)`) traced with an empty/
            # zero-length `past_key_states` (no real cache passed in): MIL's own default
            # pipeline already folds the empty operand away before this walk ever sees it, so
            # `values` legitimately has only one Var left. Concatenating one tensor with
            # nothing is an identity, not a dead op -- alias it away (same pattern as the
            # `cast` branch above) instead of silently dropping it, which previously left
            # every real consumer (found via Qwen3's GQA repeat_kv fusion input) referencing
            # an unresolved, never-produced name.
            aliases[self.safe_name(op.outputs[0].name)] = inputs[0]
            return


@topology_rule('stack')
def _op_stack(self, op, ctx):
    # `passes.py`'s `lower_stack` (EXPORT-ROADMAP.md R2) rewrites every `stack` op into
    # `expand_dims` + `concat` -- both already-real MIL ops with their own full, general rules just
    # above/below -- before this table ever runs. Reaching here means it didn't run.
    raise NotImplementedError(
        f"stack op '{op.name}' reached the topology rule table directly -- passes.py's lower_stack "
        "should have already rewritten this into expand_dims + concat before this table ever ran."
    )


# `passes.py`'s `lower_reduce_mean` (EXPORT-ROADMAP.md R2) rewrites every single-axis `reduce_mean` into
# either a `reduce_sum`+`loom_scale` composition (static count) or a `loom_mean` op (dynamic count on
# ne[0]) before this table ever runs, and raises for anything else (multi-axis, or a dynamic count on any
# other axis) -- see that pass and the `loom_mean`/`loom_scale` rules below. Reaching here means it
# didn't run.

@topology_rule('reduce_mean')
def _op_reduce_mean_unreachable(self, op, ctx):
    raise NotImplementedError(
        f"reduce_mean op '{op.name}' reached the topology rule table directly -- passes.py's "
        "lower_reduce_mean should have already rewritten this into reduce_sum+loom_scale or loom_mean "
        "(or raised) before this table ever ran."
    )


@topology_rule('loom_mean')
def _op_loom_mean(self, op, ctx):
    # `ggml_mean` (src/ops/primitives_mean.cpp) always reduces ne[0] and supplies its own run-time
    # element count -- no attrs needed, matching exactly what the generic OP_MAP fallback this replaces
    # used to emit (see loom_mean's own docstring in dialect.py for why that fallback existed at all).
    x_var_obj = op.inputs["x"]
    x_name = ctx.resolve(self.safe_name(x_var_obj.name))
    if getattr(x_var_obj, "op", None) is not None and x_var_obj.op.op_type == "transpose":
        # ggml_mean (like conv's im2col elsewhere) reduces ne[0] assuming a CONTIGUOUS source -- fed a
        # PERMUTE's own output (a non-contiguous view) directly, it silently reads with the WRONG
        # stride and produces a plausible-looking but WRONG result (no assert, unlike conv's im2col;
        # confirmed via an isolated minimal repro: PERMUTE([4,T]->[T,4])+MEAN gave [10,20,17,14] instead
        # of the correct per-channel means [1,11,21,31] for a hand-computed input -- adding an explicit
        # CONT between the two fixed it exactly). First hit by StyleTTS2's diffusion
        # Transformer1d.run(), whose `x.mean(axis=1)` (reducing over the token axis) traces to exactly
        # this PERMUTE-straight-into-MEAN shape (torch's own `.mean()` needs the reduced axis
        # transposed to ne[0] first, matching this project's own "PERMUTE so the target axis lands on
        # ne[0], THEN reduce" convention for REDUCE_SUM elsewhere). Inserting a CONT here is always
        # safe regardless of whether the source happens to already be contiguous (a CONT of an
        # already-contiguous tensor is a harmless, cheap extra copy), so this cannot regress any
        # existing MEAN usage.
        cont_name = x_name + "_mean_cont"
        ctx.nodes.append({"op": "CONT", "inputs": [x_name], "outputs": [cont_name]})
        x_name = cont_name
    ctx.nodes.append({
        "op": "MEAN",
        "inputs": [x_name],
        "outputs": [self.safe_name(op.outputs[0].name)],
    })


@topology_rule('loom_scale')
def _op_loom_scale(self, op, ctx):
    # `1.0 / n` is computed HERE, in plain Python double precision, rather than carried on the op
    # pre-divided -- see loom_scale's own docstring in dialect.py for why (MIL casts every float const
    # to fp32 on construction, which silently rounds a value like 1/192 well before it ever reaches
    # this rule).
    n = int(static_value(op.inputs["n"]))
    ctx.nodes.append({
        "op": "SCALE",
        "inputs": [ctx.resolve(self.safe_name(op.inputs["x"].name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"s": 1.0 / n},
    })


@topology_rule('reduce_sum')
def _op_reduce_sum(self, op, ctx):
    nodes, aliases, resolve = ctx.nodes, ctx.aliases, ctx.resolve
    # Dedicated branch (not the generic single-input OP_MAP path): op_reduce_sum
    # (src/ops/primitives_mil.cpp) needs a real "axis" (ne-order) + "keep_dims" attr to do a
    # genuine per-axis reduction -- MIL's own "axes"/"keep_dims" inputs are Vars (constants),
    # which the generic attrs-collection loop below only ever captures for non-Var inputs, so
    # they'd otherwise be silently dropped and this would always full-reduce to one scalar
    # (wrong for every real use seen so far: STFT magnitude, CMVN mean/variance -- all reduce
    # over exactly one axis, never every element).
    x_var_obj = op.inputs.get("x")
    axes_obj = op.inputs.get("axes")
    keep_dims_obj = op.inputs.get("keep_dims")
    if x_var_obj is None:
        return
    out_rank = len(self.get_var_info(op.outputs[0])["shape"])
    in_rank = len(self.get_var_info(x_var_obj)["shape"])
    axes_val = static_value(axes_obj)
    keep_dims_val = bool(static_value(keep_dims_obj, False))
    output_var = self.safe_name(op.outputs[0].name)
    x_name = resolve(self.safe_name(x_var_obj.name))
    if axes_val is not None and len(axes_val) > 1:
        # SupertonicTTS's VFTextCrossAttention derives its fractional-RoPE sequence lengths via
        # `mask.sum(dim=[1,2])` on a (B,1,T) mask -- a genuine 2-axis reduce_sum, but axis 1
        # (the mask's own channel dim) is ALWAYS static size 1, i.e. summing over it is a
        # provable no-op, not a real reduction. Unlike GroupNorm's own 2-axis case (both axes
        # genuinely contribute, one of them dynamically-sized -- bridged to a dedicated
        # `loom_group_norm` custom op instead, see group_norm_op.py), this only ever needs
        # dropping the trivial size-1 axes and falling through to the existing single-axis
        # REDUCE_SUM path below -- no new primitive.
        x_shape = self.get_var_info(x_var_obj)["shape"]
        real_axes = []
        for a in axes_val:
            a = int(a)
            if a < 0:
                a += in_rank
            ne_a = in_rank - 1 - a
            size = x_shape[ne_a]
            if str(size) == "1":
                continue
            real_axes.append(a)
        if len(real_axes) == 0:
            # Every reduced axis was static size 1 -- sum is the identity.
            aliases[output_var] = x_name
            return
        if len(real_axes) > 1:
            raise NotImplementedError(
                f"reduce_sum op '{op.name}': multi-axis reduction with more than one "
                f"non-trivial (size>1) axis (got axes={axes_val!r} on shape {x_shape!r}) needs "
                "its own composition (see GroupNorm's loom_group_norm custom-op bridge)."
            )
        axes_val = [real_axes[0]]
    if axes_val is None or len(axes_val) != 1:
        raise NotImplementedError(
            f"reduce_sum op '{op.name}': only a single reduction axis is supported "
            f"(got axes={axes_val!r}); multi-axis reduction needs its own composition."
        )
    axis = int(axes_val[0])
    if axis < 0:
        axis += in_rank
    ne_axis = in_rank - 1 - axis
    nodes.append({
        "op": "REDUCE_SUM",
        "inputs": [x_name],
        "outputs": [output_var],
        "attrs": {"axis": ne_axis, "keep_dims": keep_dims_val}
    })


@topology_rule('clip')
def _op_clip(self, op, ctx):
    # MIL's `clip` is what `torch.clamp(x, min=..., max=...)` becomes, and it carries its two bounds as
    # `alpha`/`beta` INPUT Vars rather than as attributes -- so the generic OP_MAP path would hand
    # op_clamp three inputs and no `min`/`max` attrs at all. (`OP_MAP`'s own "clamp" entry names an op
    # coremltools never emits from the torch frontend; this is the spelling that reaches the table.)
    # Whisper's mel frontend is the first user: `torch.clamp(mel_spec, min=1e-10)` before the log.
    x_var = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    lo = static_scalar(op.inputs.get("alpha"), None)
    hi = static_scalar(op.inputs.get("beta"), None)
    if x_var is None or lo is None or hi is None:
        raise NotImplementedError(
            f"clip op '{op.name}' has non-constant bounds (alpha={lo!r}, beta={hi!r}); op_clamp reads "
            "both as numbers off the node's attrs, so a data-dependent bound needs its own composition."
        )
    ctx.nodes.append({
        "op": "CLAMP",
        "inputs": [ctx.resolve(self.safe_name(x_var.name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"min": float(lo), "max": float(hi)},
    })


def _reduce_max_total(self, op):
    """The element count a `reduce_max` collapses, or None when it does not reduce *every* axis or any
    axis is dynamic. Shared by the guard and the handler so they cannot disagree about which case they
    are in -- the same arrangement `_matmul_transposes` and `_reduce_mean_plan` use."""
    x_var = op.inputs.get("x")
    if x_var is None:
        return None
    shape = self.get_var_info(x_var)["shape"]
    axes = static_ints(op.inputs.get("axes"))
    rank = len(shape)
    # `axes=None` is MIL's own spelling of "every axis"; an explicit list has to name them all.
    if axes is not None and {a % rank for a in axes} != set(range(rank)):
        return None
    total = 1
    for size in shape:
        if not str(size).isdigit():
            return None
        total *= int(size)
    return total


@topology_rule('reduce_max', guard=lambda self, op: _reduce_max_total(self, op) is not None,
               when="it reduces every axis and all of them are static")
def _op_reduce_max_global(self, op, ctx):
    # ggml has no reduce-max primitive, but POOL_1D already does max-pooling natively -- so a *global*
    # maximum is one pool whose kernel spans the whole flattened tensor, which is exactly the reduction
    # `tools/convert_whisper/convert_whisper_encoder.py` hand-wrote for this same operation. The mel
    # frontend's `log_spec.max()` is what needs it: Whisper floors every bin at 8 dB below the loudest
    # bin in the entire 30 s clip, so this single scalar reaches every output element.
    total = _reduce_max_total(self, op)
    out_name = self.safe_name(op.outputs[0].name)
    flat_name = f"{out_name}_reduce_max_flat"
    ctx.nodes.append({
        "op": "RESHAPE",
        "inputs": [ctx.resolve(self.safe_name(op.inputs["x"].name))],
        "outputs": [flat_name],
        "attrs": {"shape": [total]},
    })
    ctx.nodes.append({
        "op": "POOL_1D",
        "inputs": [flat_name],
        "outputs": [out_name],
        "attrs": {"op": "max", "k0": total, "s0": total, "p0": 0},
    })


@topology_rule('reduce_max', when="it reduces only some axes, or a dynamic one (rejected)")
def _op_reduce_max_unsupported(self, op, ctx):
    raise NotImplementedError(
        f"reduce_max op '{op.name}' does not reduce every axis of a statically-shaped tensor "
        f"(shape={self.get_var_info(op.inputs['x'])['shape']!r}, "
        f"axes={static_ints(op.inputs.get('axes'))!r}). Only the global maximum is composed here, as a "
        "POOL_1D spanning the flattened tensor; a per-axis maximum needs a real ggml reduction that "
        "does not exist yet."
    )


@topology_rule('cumsum')
def _op_cumsum(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # MIL's `cumsum(x, axis, exclusive, reverse)` -> ggml's existing native CUMSUM primitive
    # (op_cumsum, wraps ggml_cumsum -- always along ne[0], already used by the RQ spline
    # primitive). Only the plain inclusive/forward case (exclusive=False, reverse=False) is
    # composed here -- Kokoro's SineGen phase accumulation is the only real user so far and
    # needs exactly that; ggml_cumsum itself has no exclusive/reverse variant to fall back on.
    x_var_obj = op.inputs.get("x")
    axis_obj = op.inputs.get("axis")
    excl_obj = op.inputs.get("exclusive")
    rev_obj = op.inputs.get("reverse")
    if x_var_obj is None:
        return
    in_rank = len(self.get_var_info(x_var_obj)["shape"])
    axis_val = int(static_value(axis_obj, 0))
    if axis_val < 0:
        axis_val += in_rank
    if axis_val != in_rank - 1:
        raise NotImplementedError(
            f"cumsum op '{op.name}': only cumulative sum over the trailing (ne[0]) axis is "
            f"supported (got axis={axis_val!r} for rank {in_rank}) -- ggml_cumsum only ever "
            "sums over ne[0]."
        )
    excl_val = bool(static_value(excl_obj, False))
    rev_val = bool(static_value(rev_obj, False))
    if excl_val or rev_val:
        raise NotImplementedError(
            f"cumsum op '{op.name}' has exclusive={excl_val}/reverse={rev_val} -- ggml_cumsum "
            "only implements the plain inclusive/forward case."
        )
    nodes.append({
        "op": "CUMSUM",
        "inputs": [resolve(self.safe_name(x_var_obj.name))],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {}
    })


@topology_rule('layer_norm')
def _op_layer_norm(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Dedicated branch (not the generic single-input OP_MAP path): MIL's `layer_norm(x, axes,
    # gamma, beta, epsilon)` bundles the learned affine into the op itself, but ggml_norm
    # (op_layer_norm, src/ops/primitives_basic.cpp) deliberately only does the mean/variance
    # normalization over ne[0] and leaves gamma/beta to separate MUL/ADD nodes (same
    # convention as RMS_NORM/GROUP_NORM) -- so this composes LAYER_NORM -> MUL(gamma) ->
    # ADD(beta) explicitly, mirroring the "stack" composition's own multi-node/aliases pattern.
    # Falling through to the generic path instead (as this exporter did before Conformer-CTC,
    # which never previously exercised a real MIL `layer_norm` op) passed gamma/beta/axes Vars
    # straight through as extra positional inputs, which op_layer_norm's `expect_n_inputs`
    # correctly rejects rather than silently mishandling.
    x_var_obj = op.inputs.get("x")
    axes_obj = op.inputs.get("axes")
    gamma_obj = op.inputs.get("gamma")
    beta_obj = op.inputs.get("beta")
    eps_obj = op.inputs.get("epsilon")
    if x_var_obj is None:
        return
    in_rank = len(self.get_var_info(x_var_obj)["shape"])
    axes_val = static_value(axes_obj)
    norm_axes = sorted(int(a) + in_rank if a < 0 else int(a) for a in axes_val) if axes_val is not None else None
    if norm_axes != [in_rank - 1]:
        raise NotImplementedError(
            f"layer_norm op '{op.name}': only normalization over the single trailing (ne[0]) "
            f"axis is supported (got axes={axes_val!r} for rank {in_rank}) -- ggml_norm only "
            "ever normalizes over ne[0]."
        )
    eps_val = float(static_scalar(eps_obj, 1e-5))

    out_name = self.safe_name(op.outputs[0].name)
    cur = out_name if gamma_obj is None and beta_obj is None else f"{out_name}_ln_raw"
    nodes.append({
        "op": "LAYER_NORM",
        "inputs": [resolve(self.safe_name(x_var_obj.name))],
        "outputs": [cur],
        "attrs": {"eps": eps_val}
    })
    if gamma_obj is not None:
        nxt = out_name if beta_obj is None else f"{out_name}_ln_scaled"
        nodes.append({
            "op": "MUL",
            "inputs": [cur, resolve(self.safe_name(gamma_obj.name))],
            "outputs": [nxt],
            "attrs": {}
        })
        cur = nxt
    if beta_obj is not None:
        nodes.append({
            "op": "ADD",
            "inputs": [cur, resolve(self.safe_name(beta_obj.name))],
            "outputs": [out_name],
            "attrs": {}
        })


@topology_rule('instance_norm')
def _op_instance_norm(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Dedicated branch, same shape as "layer_norm" just above: MIL's `instance_norm(x, gamma,
    # beta, epsilon)` (rank 3-4, normalizes over every axis AFTER the channel axis -- for the
    # rank-3 case this exporter has actually seen so far, Kokoro's AdaIN1d on a (B,C,T)
    # channel-first tensor, that's exactly the trailing torch axis == ne[0] in this project's
    # T-fast convention) bundles the learned affine into the op itself; ggml_norm
    # (op_layer_norm) only does the mean/variance normalization over ne[0] and leaves
    # gamma/beta to separate MUL/ADD nodes. Unlike layer_norm, instance_norm has no `axes`
    # input to validate at all -- its spatial-dims-only normalization is exactly ne[0] by
    # construction for rank 3, so no separate axis check is needed (rank 4 -- 2 spatial dims
    # -- would need a real multi-axis ggml_norm and isn't supported here, same "not yet hit"
    # bound as this exporter's other narrow-by-design branches).
    x_var_obj = op.inputs.get("x")
    gamma_obj = op.inputs.get("gamma")
    beta_obj = op.inputs.get("beta")
    eps_obj = op.inputs.get("epsilon")
    if x_var_obj is None:
        return
    in_rank = len(self.get_var_info(x_var_obj)["shape"])
    if in_rank != 3:
        raise NotImplementedError(
            f"instance_norm op '{op.name}': only rank-3 (B,C,T) input is supported (got "
            f"rank {in_rank}) -- rank 4 (2 spatial dims) needs a real multi-axis ggml_norm, "
            "not yet needed by any model this exporter has targeted."
        )
    eps_val = float(static_scalar(eps_obj, 1e-5))
    # Unlike layer_norm's own gamma/beta (which index ne[0], the SAME axis being normalized),
    # instance_norm's gamma/beta index the CHANNEL axis -- ne[1] here, a DIFFERENT axis than
    # the one ggml_norm just normalized (ne[0]=T) -- so each needs an explicit RESHAPE to
    # [1,C,1] before the MUL/ADD to broadcast against the right axis (confirmed the hard way:
    # a raw [C]-shaped MUL against a [T,C,1] tensor raises ggml's own incompatible-shapes
    # error, since a bare [C] naturally broadcasts against ne[0], not ne[1]). Mirrors the
    # "conv" branch's own established bias RESHAPE-to-[1,oc,1] convention just above.
    channels = int(op.outputs[0].shape[1])

    out_name = self.safe_name(op.outputs[0].name)
    cur = out_name if gamma_obj is None and beta_obj is None else f"{out_name}_in_raw"
    nodes.append({
        "op": "LAYER_NORM",
        "inputs": [resolve(self.safe_name(x_var_obj.name))],
        "outputs": [cur],
        "attrs": {"eps": eps_val}
    })
    if gamma_obj is not None:
        gamma_r = f"{out_name}_in_gamma_r"
        nodes.append({
            "op": "RESHAPE",
            "inputs": [resolve(self.safe_name(gamma_obj.name))],
            "outputs": [gamma_r],
            "attrs": {"shape": [1, channels, 1]}
        })
        nxt = out_name if beta_obj is None else f"{out_name}_in_scaled"
        nodes.append({
            "op": "MUL",
            "inputs": [cur, gamma_r],
            "outputs": [nxt],
            "attrs": {}
        })
        cur = nxt
    if beta_obj is not None:
        beta_r = f"{out_name}_in_beta_r"
        nodes.append({
            "op": "RESHAPE",
            "inputs": [resolve(self.safe_name(beta_obj.name))],
            "outputs": [beta_r],
            "attrs": {"shape": [1, channels, 1]}
        })
        nodes.append({
            "op": "ADD",
            "inputs": [cur, beta_r],
            "outputs": [out_name],
            "attrs": {}
        })


@topology_rule('upsample_nearest_neighbor', 'upsample_bilinear')
def _op_upsample(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    op_type = op.op_type
    # MIL's `upsample_nearest_neighbor`/`upsample_bilinear` (core ops -- NOT the "torch_"
    # dialect ops of the same name, which the "torch_upsample_to_core_upsample" SSA pass
    # already rewrites into these before this exporter ever sees the graph) always operate on
    # the LAST TWO axes ("height"=axis -2, "width"=axis -1). Every real usage this exporter
    # has hit so far (Kokoro's `nn.Upsample`/f0_upsamp on a genuine rank-3 (B,1,T) tensor, and
    # this project's own "unsqueeze a dummy axis, F.interpolate(mode='linear'), squeeze it back
    # off" rank-4 trick for coremltools' separate 1D-linear-needs-rank-4 restriction) puts the
    # tensor's REAL dynamic axis at torch axis -1 == this project's ne[0] (T-fast) convention,
    # with axis -2 always a static, unscaled size (either a real channel axis or the trick's
    # own dummy axis) -- so only `scale_factor_width` is expected to ever be non-1, mapped
    # directly to ggml's existing INTERPOLATE_1D primitive (which already resizes ne[0] only,
    # added for exactly this Kokoro use case per its own primitives_basic.cpp comment).
    sfh_obj = op.inputs.get("scale_factor_height")
    sfw_obj = op.inputs.get("scale_factor_width")
    sfh = float(static_scalar(sfh_obj, 1.0))
    sfw = float(static_scalar(sfw_obj, 1.0))
    if sfh != 1.0 and sfw != 1.0:
        raise NotImplementedError(
            f"{op_type} op '{op.name}' has non-1.0 scale factors on BOTH axes "
            f"(height={sfh}, width={sfw}) -- a genuine 2D resize, which this exporter's "
            "INTERPOLATE_1D composition (ne[0]-only) can't represent."
        )
    # Whichever of {height, width} is actually non-1.0 is the axis this exporter cares about
    # (T, this project's ne[0]) -- torch's own promotion of a genuinely rank-3 (B,C,T) 1D
    # interpolate call onto this rank>=3-only 2D op is NOT consistent about which of the two
    # trailing axes ends up "height" vs "width" (confirmed empirically: Kokoro's f0_upsamp
    # puts T at width/axis-1, but `UpSample1d`'s plain `F.interpolate(scale_factor=2,
    # mode='nearest')` puts it at height/axis-2 instead) -- so this picks by VALUE, not by a
    # fixed axis position.
    sfw = sfh if sfh != 1.0 else sfw
    x_var_obj = op.inputs.get("x")
    t_expr = self.get_var_info(x_var_obj)["shape"][0]
    out_name = self.safe_name(op.outputs[0].name)
    nodes.append({
        "op": "INTERPOLATE_1D",
        "inputs": [resolve(self.safe_name(x_var_obj.name))],
        "outputs": [out_name],
        "attrs": {
            "ne0": f"floor(({t_expr})*{sfw})",
            "mode": "nearest" if op_type == "upsample_nearest_neighbor" else "linear",
        }
    })


@topology_rule('range_1d')
def _op_range_1d(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Dedicated branch (not the generic OP_MAP path, and not the old input-ordering-only fix
    # below the main OP_MAP dispatch): op_range_1d's C++ side can only read a dynamic "end"/
    # "start" bound from a Var's own already-BUILT `.data`, which a `gather(shape(x), ...)`
    # chain's value never is at build time (see facts.gather_shape_value's own
    # docstring for why -- this is the actual root cause of Conformer-CTC's length-tracking
    # bug, not anything about shape-string derivation, which was already correct upstream of
    # this exact point). Prefer a real symbolic "attrs" expression (which op_range_1d already
    # natively supports, evaluated via SymbolEnv) over a data-dependent graph input wherever
    # one can be derived, for each of start/end/step independently -- falling back to the
    # previous positional-Vars behavior only when NONE of the three can be resolved this way,
    # since op_range_1d's own input-reading is strictly positional (in[0]=start/in[1]=end/
    # in[2]=step) and can't be safely mixed with only SOME slots given via attrs.
    start_obj = op.inputs.get("start")
    end_obj = op.inputs.get("end")
    step_obj = op.inputs.get("step")

    start_resolved = self.facts.range_scalar(start_obj)
    end_resolved = self.facts.range_scalar(end_obj)
    step_resolved = self.facts.range_scalar(step_obj)

    range_node = {"op": "RANGE_1D", "outputs": [self.safe_name(op.outputs[0].name)]}
    if start_resolved is not None and end_resolved is not None and step_resolved is not None:
        range_node["inputs"] = []
        range_node["attrs"] = {
            "start": _attr_number_or_expr(start_resolved),
            "end": _attr_number_or_expr(end_resolved),
            "step": _attr_number_or_expr(step_resolved),
        }
    else:
        range_inputs = []
        for v in (start_obj, end_obj, step_obj):
            if v is not None and isinstance(v, Var):
                range_inputs.append(resolve(self.safe_name(v.name)))
        range_node["inputs"] = range_inputs
    nodes.append(range_node)


@topology_rule('conv')
def _op_conv(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Map conv to CONV_1D or CONV_2D and extract static attributes
    strides = static_value(op.inputs.get("strides"), [1])
    pad = static_value(op.inputs.get("pad"), [0])
    dilations = static_value(op.inputs.get("dilations"), [1])
    groups = static_value(op.inputs.get("groups"), 1)

    # Format to standard integers
    is_2d = isinstance(strides, (list, tuple, np.ndarray)) and len(strides) == 2
    s0 = int(strides[0]) if isinstance(strides, (list, tuple, np.ndarray)) else int(strides)
    p0 = int(pad[0]) if isinstance(pad, (list, tuple, np.ndarray)) else int(pad)
    d0 = int(dilations[0]) if isinstance(dilations, (list, tuple, np.ndarray)) else int(dilations)
    g_val = int(groups[0]) if isinstance(groups, (list, tuple, np.ndarray)) else int(groups)

    # Check if it is a depthwise convolution (groups > 1)
    is_dw = (g_val > 1)
    if is_dw:
        mapped_op = "CONV_2D_DW" if is_2d else "CONV_1D_DW"
    else:
        mapped_op = "CONV_2D" if is_2d else "CONV_1D"

    # Extract main inputs [x, weight]
    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    x_var = self.safe_name(x_var_obj.name)
    weight_var = self.safe_name(op.inputs["weight"].name)

    attrs = {"s0": s0, "p0": p0, "d0": d0, "groups": g_val}
    if is_2d:
        # 2D conv's `pad` is [top, bottom, left, right] (2 entries per spatial axis,
        # before+after) -- op_conv_2d/op_conv_2d_dw (src/ops/primitives_conv.cpp) only take
        # one padding value per axis (symmetric padding), so this uses `pad[2]` (the second
        # spatial axis's "before" entry), matching s1/d1's own per-axis (not per-side) index.
        attrs["s1"] = int(strides[1])
        attrs["p1"] = int(pad[2]) if len(pad) > 2 else p0
        attrs["d1"] = int(dilations[1]) if len(dilations) > 1 else d0

    # MIL's "conv" has an OPTIONAL per-output-channel "bias" input (torch Conv1d/2d's own
    # `bias=True`) -- unlike this file's own "conv_transpose" translation just below (which
    # explicitly rejects a non-zero bias rather than silently dropping it), this branch used
    # to never even look at "bias" at all, silently omitting it entirely. Confirmed as a real,
    # general correctness bug (not Conformer-CTC-specific -- every model this exporter has
    # produced with a biased conv loses that bias) via Conformer-CTC-small's own subsampling
    # `pre_encode.conv.{0,2}` (real, non-tiny biases, mean ~0.4) -- see BACKLOG.md for the full
    # diagnostic trail (a standalone multi-channel CONV_2D unit test first ruled out the ggml
    # primitive itself, then diffing the exported topology's own JSON directly against the
    # bespoke conversion's hand-built one showed the missing ADD node). Composed the same way
    # the "linear" case above does (MUL_MAT + ADD), reshaped to broadcast against the conv
    # output's real ne-order channel axis (ne[2] for CONV_2D's [OW,OH,OC,N], ne[1] for
    # CONV_1D's [OL,OC,N] -- matches convert_conformer_ctc.py's own established
    # RESHAPE-to-[1,1,C,1]-then-ADD convention for the exact same bias).
    bias_var_obj = op.inputs.get("bias")
    output_var = self.safe_name(op.outputs[0].name)
    if static_value(bias_var_obj) is not None and np.any(static_array(bias_var_obj)):
        bias_var = self.safe_name(bias_var_obj.name)
        # Read the MIL var's own RAW (torch-order) shape directly -- NOT get_var_info's
        # ne-order-reversed one -- a conv's output is torch-shaped [N, OC, ...spatial...], so
        # OC is plain axis 1. OC is always statically known (never a symbolic dynamic dim) for
        # every real conv this exporter targets.
        oc = int(op.outputs[0].shape[1])
        bias_shape = [1, 1, oc, 1] if is_2d else [1, oc, 1]
        conv_out_var = output_var + "_conv_raw"
        bias_reshaped_var = output_var + "_bias_r"
        nodes.append({
            "op": mapped_op,
            "inputs": [resolve(weight_var), resolve(x_var)],
            "outputs": [conv_out_var],
            "attrs": attrs
        })
        nodes.append({
            "op": "RESHAPE",
            "inputs": [resolve(bias_var)],
            "outputs": [bias_reshaped_var],
            "attrs": {"shape": bias_shape}
        })
        nodes.append({
            "op": "ADD",
            "inputs": [conv_out_var, bias_reshaped_var],
            "outputs": [output_var]
        })
    else:
        nodes.append({
            "op": mapped_op,
            "inputs": [resolve(weight_var), resolve(x_var)],
            "outputs": [output_var],
            "attrs": attrs
        })


@topology_rule('conv_transpose')
def _op_conv_transpose(self, op, ctx):
    nodes, resolve = ctx.nodes, ctx.resolve
    # Map conv_transpose to CONV_TRANSPOSE_1D/2D. Loom's C++ primitives
    # (src/ops/primitives_conv.cpp's op_conv_transpose_1d/2d, backing ggml_conv_transpose_1d /
    # ggml_conv_transpose_2d_p0) only ever compute the UNPADDED ("valid") result -- no
    # padding, no dilation, no grouping. `pad_type="valid"` (this exporter's own ISTFT
    # module, Kokoro's hand-built Generator upsampling) needs nothing more. `pad_type=
    # "custom"` with a real non-zero symmetric-or-not `pad` -- first hit by HiFi-GAN's own
    # upsample stages (VITS's `dec.ups.*`, real `nn.ConvTranspose1d(..., padding=(kernel-
    # stride)//2)`) -- is composed instead as the mathematically equivalent "valid conv_
    # transpose, then crop `pad_before`/`pad_after` off each end of the spatial axis"
    # (real ConvTranspose1d padding semantics: `L_out = (L_in-1)*stride - 2*pad + kernel`,
    # exactly "valid"'s own `(L_in-1)*stride + kernel` minus the crop). Anything else (2D,
    # non-unit dilation, grouped) still raises rather than silently dropping configuration.
    strides = static_value(op.inputs.get("strides"), [1])
    pad = static_value(op.inputs.get("pad"), [0])
    dilations = static_value(op.inputs.get("dilations"), [1])
    groups = static_value(op.inputs.get("groups"), 1)
    output_shape = op.inputs.get("output_shape")
    bias_var = op.inputs.get("bias")
    pad_type_var = op.inputs.get("pad_type")
    pad_type = static_value(pad_type_var, "valid")

    if pad_type not in ("valid", "custom"):
        raise NotImplementedError(
            f"conv_transpose op '{op.name}' has pad_type='{pad_type}', which this exporter "
            "doesn't support (only 'valid' and a 'custom' symmetric-crop composition exist)."
        )
    pad_list = list(pad) if isinstance(pad, (list, tuple, np.ndarray)) else [pad]
    is_2d = isinstance(strides, (list, tuple, np.ndarray)) and len(strides) == 2
    if is_2d and pad_type == "custom":
        raise NotImplementedError(
            f"conv_transpose op '{op.name}' is 2D with pad_type='custom' -- only the 1D "
            "crop composition has been needed/written so far."
        )
    pad_before = int(pad_list[0]) if pad_type == "custom" and len(pad_list) > 0 else 0
    pad_after = int(pad_list[1]) if pad_type == "custom" and len(pad_list) > 1 else 0
    # Torch-traced conv_transpose always carries an explicit 'output_shape' -- it's
    # redundant with the real formula above (valid, or valid-minus-crop for custom), not a
    # genuine extra constraint, so it's fine to ignore rather than reject; just sanity-check
    # it agrees rather than trusting it blindly.
    if output_shape is not None and getattr(output_shape, "val", None) is not None:
        x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
        weight_var_obj = op.inputs["weight"]
        strides_list = list(strides) if isinstance(strides, (list, tuple, np.ndarray)) else [strides]
        n_spatial = len(strides_list)
        in_spatial = list(x_var_obj.shape[-n_spatial:])
        kernel_spatial = list(weight_var_obj.shape[-n_spatial:])
        expected = [(int(d) - 1) * int(s) + int(k) - pad_before - pad_after
                    for d, s, k in zip(in_spatial, strides_list, kernel_spatial)]
        actual = static_ints(output_shape)[-n_spatial:]
        if expected != actual:
            raise NotImplementedError(
                f"conv_transpose op '{op.name}' declares output_shape={static_ints(output_shape)!r}, "
                f"which doesn't match this composition's own {expected!r} -- this combination "
                "isn't supported."
            )
    if any(int(d) != 1 for d in (dilations if isinstance(dilations, (list, tuple, np.ndarray)) else [dilations])):
        raise NotImplementedError(
            f"conv_transpose op '{op.name}' has non-unit 'dilations' {dilations!r}, which "
            "this exporter doesn't support."
        )
    g_val = int(groups[0]) if isinstance(groups, (list, tuple, np.ndarray)) else int(groups)
    s0 = int(strides[0]) if isinstance(strides, (list, tuple, np.ndarray)) else int(strides)
    if g_val != 1:
        # `passes.py`'s `canonicalize_conv_transpose_dw` (EXPORT-ROADMAP.md R2) rewrites every
        # depthwise (groups != 1) conv_transpose into a `loom_conv_transpose_dw` op before this table
        # ever runs -- see that pass and the `loom_conv_transpose_dw` rule just below. Reaching here
        # means it didn't run.
        raise NotImplementedError(
            f"conv_transpose op '{op.name}' has groups={g_val} != 1 -- passes.py's "
            "canonicalize_conv_transpose_dw should have already rewritten this into a "
            "loom_conv_transpose_dw op before this table ever ran."
        )

    mapped_op = "CONV_TRANSPOSE_2D" if is_2d else "CONV_TRANSPOSE_1D"

    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
    x_var = self.safe_name(x_var_obj.name)
    weight_var = self.safe_name(op.inputs["weight"].name)
    output_var = self.safe_name(op.outputs[0].name)
    has_bias = static_value(bias_var) is not None and np.any(static_array(bias_var))
    needs_crop = pad_before != 0 or pad_after != 0

    if getattr(x_var_obj, "op", None) is not None and x_var_obj.op.op_type == "transpose":
        # Same non-contiguous-input danger already fixed for MEAN just above: ggml's
        # `ggml_conv_transpose_1d` (like its im2col-based plain-conv sibling) requires a
        # CONTIGUOUS source and has no assert to catch a stale/wrong stride the way plain
        # conv's im2col does -- it just asserts `nb10 == sizeof(float)` and aborts outright.
        # First hit by Matcha-TTS's Decoder U-Net: `rearrange(x, "b t c -> b c t")` (a real
        # `transpose`) sits directly upstream of every `Upsample1D`'s real ConvTranspose1d.
        # Inserting a CONT here is always safe regardless of whether the source happens to
        # already be contiguous (a CONT of an already-contiguous tensor is a harmless, cheap
        # extra copy), so this cannot regress any existing conv_transpose usage (HiFi-GAN's
        # own upsample stages never feed it a bare transpose directly).
        cont_name = x_var + "_convt_cont"
        nodes.append({"op": "CONT", "inputs": [resolve(x_var)], "outputs": [cont_name]})
        x_var = cont_name

    # Bias-add composition (mirrors the "conv" branch's own established RESHAPE-then-ADD
    # pattern just above): ggml_conv_transpose_1d/2d have no bias-add of their own -- first
    # hit here since HiFi-GAN's upsample ("dec.ups.{stage}") transposed convs are the first
    # model on this exporter's roadmap to use a BIASED conv_transpose at all (this exporter's
    # only prior conv_transpose consumers -- its own ISTFT module, Kokoro's hand-built
    # Generator upsampling -- are both bias-free).
    raw_var = (output_var + "_convt_raw") if (has_bias or needs_crop) else output_var
    nodes.append({
        "op": mapped_op,
        "inputs": [resolve(weight_var), resolve(x_var)],
        "outputs": [raw_var],
        "attrs": {"s0": s0}
    })
    biased_var = raw_var
    if has_bias:
        bias_var_name = self.safe_name(bias_var.name)
        oc = int(op.outputs[0].shape[1])
        bias_shape = [1, 1, oc, 1] if mapped_op == "CONV_TRANSPOSE_2D" else [1, oc, 1]
        bias_reshaped_var = output_var + "_bias_r"
        biased_var = (output_var + "_convt_biased") if needs_crop else output_var
        nodes.append({
            "op": "RESHAPE",
            "inputs": [resolve(bias_var_name)],
            "outputs": [bias_reshaped_var],
            "attrs": {"shape": bias_shape}
        })
        nodes.append({
            "op": "ADD",
            "inputs": [raw_var, bias_reshaped_var],
            "outputs": [biased_var]
        })
    if needs_crop:
        # Crop `pad_before`/`pad_after` elements off ne[0] (the spatial axis, fastest-
        # varying -- CONV_TRANSPOSE_1D's own convention, matching CONV_1D's) via a zero-copy
        # VIEW. Target shape's ne0 comes from `op.outputs[0]`'s own real (already-cropped)
        # MIL-inferred size -- resolved through the new "conv_transpose" case in
        # `_infer_dynamic_dim_expr` above rather than re-derived here, so this stays correct
        # even chained across multiple upsample stages.
        out_info = self.get_var_info(op.outputs[0])
        crop_shape = list(out_info["shape"])
        nodes.append({
            "op": "VIEW",
            "inputs": [biased_var],
            "outputs": [output_var],
            "attrs": {"shape": crop_shape, "offset": pad_before * 4}
        })


@topology_rule('loom_conv_transpose_dw')
def _op_loom_conv_transpose_dw(self, op, ctx):
    """1:1 port of the exporter's former ad hoc depthwise-conv_transpose composition
    (EXPORT-ROADMAP.md R2; see `loom_conv_transpose_dw`'s own docstring in dialect.py and
    `passes.py`'s `canonicalize_conv_transpose_dw`, which inserts this op): zero-stuff the input by
    `stride`, then an ordinary stride=1 depthwise conv with a kernel-reversed weight. Reuses the exact
    node sequence already verified in tools/convert_kokoro/convert_kokoro_f0n.py's
    `add_depthwise_conv_transpose_upsample` (itself checked against test_primitive_registry.cpp's
    test_depthwise_conv_transpose_1d_via_composition before ever being used there), generalized to a
    real traced op's own stride/kernel_size/channel count and a REAL dynamic-length expression (via
    get_var_info) instead of that script's own hardcoded "$n_tokens".
    """
    nodes, resolve, func_name = ctx.nodes, ctx.resolve, ctx.func_name
    x_var_obj = op.inputs["x"]
    weight_var_obj = op.inputs["weight"]
    bias_var = op.inputs.get("bias")
    s0 = int(static_value(op.inputs["stride"]))
    in_channels = int(x_var_obj.shape[1])
    kernel_size = int(weight_var_obj.shape[-1])

    weight_val = static_value(weight_var_obj)
    flipped = np.ascontiguousarray(np.asarray(weight_val)[:, :, ::-1])
    flipped_name = self.safe_name(weight_var_obj.name) + "_dwt_flip"
    if func_name == "main_topology" or self.flat_namespace:
        namespaced_flipped = flipped_name
    else:
        namespaced_flipped = f"{func_name}.{flipped_name}"
    if len(namespaced_flipped) >= 64:
        h = hashlib.md5(namespaced_flipped.encode("utf-8")).hexdigest()[:6]
        namespaced_flipped = f"{namespaced_flipped[:30]}_{h}_{namespaced_flipped[-20:]}"
    self.weights[namespaced_flipped] = flipped

    x_var = self.safe_name(x_var_obj.name)
    output_var = self.safe_name(op.outputs[0].name)
    t_expr = self.get_var_info(x_var_obj)["shape"][0]
    channels = in_channels

    d3 = f"{output_var}_dwt_d3"
    nodes.append({"op": "RESHAPE", "inputs": [resolve(x_var)], "outputs": [d3],
                  "attrs": {"shape": [1, t_expr, channels]}})
    stuffed3 = f"{output_var}_dwt_stuffed3"
    nodes.append({"op": "PAD_1D", "inputs": [d3], "outputs": [stuffed3],
                  "attrs": {"lp0": 0, "rp0": s0 - 1}})
    overstuffed = f"{output_var}_dwt_overstuffed"
    nodes.append({"op": "RESHAPE", "inputs": [stuffed3], "outputs": [overstuffed],
                  "attrs": {"shape": [render(as_expr(t_expr) * s0), channels]}})
    std_len = render((as_expr(t_expr) - 1) * s0 + 1)
    trunc_v = f"{output_var}_dwt_trunc_v"
    nodes.append({"op": "VIEW", "inputs": [overstuffed], "outputs": [trunc_v],
                  "attrs": {"shape": [std_len, channels]}})
    trunc = f"{output_var}_dwt_trunc"
    nodes.append({"op": "CONT", "inputs": [trunc_v], "outputs": [trunc]})
    pad_each = kernel_size - 1
    padded = f"{output_var}_dwt_padded"
    nodes.append({"op": "PAD_1D", "inputs": [trunc], "outputs": [padded],
                  "attrs": {"lp0": pad_each, "rp0": pad_each}})

    has_bias = bias_var is not None and static_value(bias_var) is not None and np.any(static_array(bias_var))
    raw_var = (output_var + "_dwt_raw") if has_bias else output_var
    nodes.append({"op": "CONV_1D_DW", "inputs": [namespaced_flipped, padded], "outputs": [raw_var],
                  "attrs": {"s0": 1, "p0": 0, "d0": 1}})
    if has_bias:
        bias_var_name = self.safe_name(bias_var.name)
        bias_reshaped = output_var + "_dwt_bias_r"
        nodes.append({"op": "RESHAPE", "inputs": [resolve(bias_var_name)], "outputs": [bias_reshaped],
                      "attrs": {"shape": [1, channels, 1]}})
        nodes.append({"op": "ADD", "inputs": [raw_var, bias_reshaped], "outputs": [output_var]})


# `less` is the one op type whose rule is genuinely conditional on a *derivation*, not on a static
# attribute: only a NeMo-style length-validity comparison that provably compares a quantity against
# itself gets replaced by a baked all-true mask; every other `less` must stay a real comparison and is
# therefore left to the generic OP_MAP path (which maps it to the LESS primitive). Expressing that as a
# guard is what makes the fall-through explicit -- before this table it was an `if` the block simply ran
# off the end of, the single easiest thing in the whole dispatcher to misread as "handled".

def _less_is_always_valid_mask(self, op):
    """True iff this `less` is the length-validity mask idiom whose result is all-true by
    construction. See the derivation below for why that is a correctness fix rather than an
    optimization."""
    # Recognize NeMo-style length-validity masking (`torch.arange(T) < length`, comparing a
    # bare position index against a value derived from the graph's own "length" input) and
    # bake its result as a constant all-true (1.0) tensor instead of translating the real
    # comparison -- rather than a shape-derivation fix, this sidesteps a genuine, deeply
    # rooted correctness bug in the traced MIL graph itself: NeMo's own `calc_length()`
    # formula (used to compute the comparison's RHS bound) traces with a wrong constant baked
    # in for `all_paddings` (confirmed directly by reading the exported GGUF's own stored
    # weight values and reconstructing the exact arithmetic: `all_paddings - kernel_size`
    # computes as `1 - 3` instead of the real `2 - 3`, since coremltools' own optimizer had
    # ALSO already eliminated every standalone `torch.floor()` call in this chain as a
    # provable no-op for the specific dummy trace length used -- confirmed via `grep`, there
    # are zero raw `FLOOR` ops anywhere in the exported topology -- so the wrong constant
    # can't be fixed by recovering a dropped floor, the arithmetic itself is wrong). See
    # BACKLOG.md for the full derivation.
    #
    # Rather than reverse-engineer and re-derive NeMo's exact (buggy-under-tracing) formula,
    # this exploits an invariant this WHOLE exporter already assumes everywhere else (the
    # always-1 batch axis, the "single utterance" driver-script shape): every model this
    # exporter targets is run with `length` set to the REAL, exact length of `waveform` --
    # there is never any actual padding. Under that guarantee, `torch.arange(T) < length` is
    # true for every position BY CONSTRUCTION (T is itself derived from that same real
    # length), regardless of what value NeMo's own traced arithmetic computes for the
    # comparison's RHS. Scoped narrowly (only "less", the one comparison op actually seen in
    # this exact idiom) rather than every comparison type, matching this file's own
    # "not implemented since nothing here has needed it yet" convention -- extend if/when a
    # different comparison op is found doing the same thing.
    # NOT every `arange(T) < f(length)` this narrow structural pattern matches is actually
    # this bug: NeMo's real mel-frontend (`FilterbankFeatures.forward`/`normalize_batch`,
    # traced for real here -- unlike Conformer-CTC-small's own MIL export, which never had a
    # test comparing its numeric output against a reference and so never caught this) uses
    # the IDENTICAL `arange(T) < f(length)` shape for a DELIBERATELY different comparison:
    # `get_seq_len()`'s `floor((length + pad_amount - n_fft) / hop_length)` is genuinely ONE
    # LESS than the real STFT frame count (`T`, the same "last frame is always invalid"
    # off-by-one already root-caused and hand-replicated in convert_conformer_ctc.py/
    # convert_parakeet_tdt.py's own CMVN section -- see valid_frames_expr() there) -- NOT a
    # tracing artifact, a real, intentional NeMo convention that must NOT be forced to
    # all-true. Structurally this looks IDENTICAL to the genuine calc_length tracing bug
    # (both are "arange(T) < floor((length + C1 - C2) / C3)"), so telling them apart needs an
    # actual identity check: derive T's own real formula (`_find_range_1d_var` + the
    # `range_1d` case of `_infer_dynamic_dim_expr`, the SAME resolution RANGE_1D's own node
    # emission uses) and the comparison bound's real formula (`facts.scalar_expr`, walking
    # through the exact `select`/arithmetic chain `get_seq_len` traces to) and compare them AS
    # STRINGS -- only bypass when they're the identical expression (proving T and the bound
    # are the SAME quantity, so any real length must make the comparison true by construction
    # -- the calc_length case). When they differ (the CMVN case: T is the raw STFT frame
    # count, the bound is deliberately T-1), leave the real comparison/select chain in place
    # instead -- every primitive it needs (FLOOR_DIV, promote_i32_to_f32, op_select's
    # mul_broadcast) already exists, proven by Conformer-CTC's OWN encoder needing them for
    # other parts of its graph.
    # A pure STRING-equality check here (an earlier version of this fix) turned out to be
    # too CONSERVATIVE, not too permissive: Conformer-CTC-small's own encoder/subsampling-
    # level masks (MaskedConvSequential's per-stage `_create_mask`, and the encoder's own
    # top-level `_create_masks`) are fed a "length" that ALREADY passes through the mel-
    # frontend's own `get_seq_len` (T-1) convention, so their OWN "T vs. bound" formulas
    # come out structurally unequal too (propagated through further conv-shape arithmetic,
    # off by exactly 1 at SOME lengths and by 0 at others depending on integer-halving
    # parity) -- structurally indistinguishable from CMVN's own case by pure string
    # comparison, but NOT the same thing to force-bypass or not: confirmed empirically (via
    # a real, controlled experiment: force-bypassing EVERY "less" match here, including this
    # one, dropped Conformer-CTC-small's own encoder-output max abs diff from 2.09 to 0.13 --
    # i.e. leaving these encoder-level comparisons real, un-bypassed, was making things WORSE,
    # not more faithful, presumably because they still route through NeMo's own separately-
    # documented `calc_length`/`all_paddings` tracing bug). Only CMVN's own comparison is
    # reliably, structurally DIFFERENT in a way worth preserving: unlike the encoder-level
    # case, it's off by EXACTLY 1 at every possible length, never 0, because it's the raw
    # "T vs T-1" relationship itself, not something derived further from it. Distinguish via a
    # numeric probe (several concrete `n_tokens` values, not just one, so a coincidental match
    # at a single probe can't fool this) rather than a syntactic one: only refuse to bypass
    # when range == length + 1 at EVERY probe; default to bypass otherwise (matching the
    # empirically-correct, more permissive behavior for everything else, including the
    # structurally-similar-looking but NOT-off-by-exactly-1-always encoder-level case).
    def _eval_expr(expr, n_tokens_value):
        """`expr` at a concrete sequence length. Substituting into the sympy expression evaluates it
        exactly, where the previous version round-tripped the expression through a string and `eval`
        with `/` as float division. The decision this feeds is unchanged on every current model -- the
        snapshot diff across all 12 shows no LESS node appearing or disappearing -- but there is no
        re-parsing and no `eval` any more.

        Substitutes EVERY free symbol in `expr` with `n_tokens_value`, not just the literal `N_TOKENS`
        (as an earlier version of this did) -- needed once a topology's root axis can be named
        something other than "n_tokens" (EXPORT-ROADMAP.md R1, axes.py): Conformer-CTC/Parakeet, the
        one family this exact bypass exists for, now declares its root axis "n_samples". Hardcoding
        `N_TOKENS` here would silently substitute nothing for those models, making every probe evaluate
        to a still-symbolic (non-numeric) expression -- `float()` would then raise, `_eval_expr` would
        return None for every probe, and the caller's loop reads that as "not always off by exactly
        one", flipping this bypass from correctly refused (the CMVN case) to wrongly permitted. Sound
        for the same reason `compare_snapshots.py`'s own probe substitution is: this whole exporter
        targets models with exactly one true dynamic quantity per topology, so any free symbol reaching
        here IS that topology's one true axis, whatever it happens to be named."""
        if expr is None:
            return None
        try:
            expr = as_expr(expr)
            return float(expr.subs({s: n_tokens_value for s in expr.free_symbols}))
        except (TypeError, ValueError):
            return None

    x_var = op.inputs.get("x")
    y_var = op.inputs.get("y")
    x_is_length = isinstance(x_var, Var) and self._traces_to_length_input(x_var)
    y_is_length = isinstance(y_var, Var) and self._traces_to_length_input(y_var)
    range_var = None
    length_side_var = None
    if isinstance(x_var, Var) and y_is_length:
        range_var = self._find_range_1d_var(x_var)
        length_side_var = y_var
    elif isinstance(y_var, Var) and x_is_length:
        range_var = self._find_range_1d_var(y_var)
        length_side_var = x_var
    bypass_ok = False
    if range_var is not None and length_side_var is not None:
        range_expr = self._infer_dynamic_dim_expr(range_var, 0)
        length_expr = self.facts.scalar_expr(length_side_var)
        bypass_ok = True
        if range_expr is not None and length_expr is not None:
            always_off_by_exactly_one = True
            for probe in (1600, 8000, 10240, 16000, 16001, 20000, 31999, 320000):
                r = _eval_expr(range_expr, probe)
                l = _eval_expr(length_expr, probe)
                if r is None or l is None or r != l + 1:
                    always_off_by_exactly_one = False
                    break
            if always_off_by_exactly_one:
                bypass_ok = False
    return bypass_ok


@topology_rule('less', guard=_less_is_always_valid_mask,
               when="it is a provably-all-true length-validity mask")
def _op_less_always_valid(self, op, ctx):
    func_name = ctx.func_name
    out_info = self.get_var_info(op.outputs[0])
    target_shape = list(out_info["shape"])
    rank = len(target_shape)
    weight_name = self.safe_name(op.outputs[0].name) + "_always_valid_scalar"
    namespaced_name = (weight_name if (func_name == "main_topology" or self.flat_namespace)
                        else f"{func_name}.{weight_name}")
    self.weights[namespaced_name] = np.full([1] * rank, 1.0, dtype=np.float32)
    ctx.nodes.append({
        "op": "REPEAT",
        "inputs": [namespaced_name],
        "outputs": [self.safe_name(op.outputs[0].name)],
        "attrs": {"shape": target_shape}
    })


@topology_rule('batch_norm')
def _op_batch_norm(self, op, ctx):
    nodes, resolve, func_name = ctx.nodes, ctx.resolve, ctx.func_name
    # SupertonicTTS's SpeechDecoder.final_norm (nn.BatchNorm1d, always eval mode -- this
    # project never traces training-mode graphs). `mean`/`variance`/`gamma`/`beta` are all
    # real CONSTANT Vars once traced (baked from the module's own running-stats buffers and
    # learned affine params), so this folds to a plain per-channel scale+shift at CONVERSION
    # time -- same "fold at conversion time" precedent as weight-norm/Snake's reciprocal
    # elsewhere in this project, not a new runtime primitive.
    x_var_obj = op.inputs["x"]
    mean_var = op.inputs.get("mean")
    var_var = op.inputs.get("variance")
    gamma_var = op.inputs.get("gamma")
    beta_var = op.inputs.get("beta")
    eps_var = op.inputs.get("epsilon")
    if any(static_value(v) is None for v in (mean_var, var_var)):
        raise NotImplementedError(
            f"batch_norm op '{op.name}' has a non-constant mean/variance -- only eval-mode "
            "(real running-stats buffers) BatchNorm is supported."
        )
    mean_np = static_array(mean_var).astype(np.float32)
    var_np = static_array(var_var).astype(np.float32)
    gamma_np = (static_array(gamma_var).astype(np.float32)
                if static_value(gamma_var) is not None
                else np.ones_like(mean_np))
    beta_np = (static_array(beta_var).astype(np.float32)
               if static_value(beta_var) is not None
               else np.zeros_like(mean_np))
    eps = float(static_scalar(eps_var, 1e-5))

    scale_np = gamma_np / np.sqrt(var_np + eps)
    shift_np = beta_np - mean_np * scale_np

    in_rank = len(self.get_var_info(x_var_obj)["shape"])
    channel_ne_axis = in_rank - 1 - 1  # MIL's channel axis is always torch axis 1
    bcast_shape = [1] * in_rank
    bcast_shape[channel_ne_axis] = scale_np.shape[0]

    x_name = resolve(self.safe_name(x_var_obj.name))
    output_var = self.safe_name(op.outputs[0].name)
    weight_base = f"{self.safe_name(op.name)}_bn"
    if func_name == "main_topology" or self.flat_namespace:
        scale_name, shift_name = f"{weight_base}_scale", f"{weight_base}_shift"
    else:
        scale_name, shift_name = f"{func_name}.{weight_base}_scale", f"{func_name}.{weight_base}_shift"
    self.weights[scale_name] = scale_np.reshape(bcast_shape)
    self.weights[shift_name] = shift_np.reshape(bcast_shape)

    scaled = f"{output_var}_bn_scaled"
    nodes.append({"op": "MUL", "inputs": [x_name, scale_name], "outputs": [scaled]})
    nodes.append({"op": "ADD", "inputs": [scaled, shift_name], "outputs": [output_var]})


@topology_rule('random_normal', 'random_uniform', 'random_bernoulli', 'random_categorical')
def _op_random(self, op, ctx):
    op_type = op.op_type
    # EXPORT-IMPROVEMENT-BACKLOG.md item 4: ggml has no RNG-capable compute op at all (no
    # ggml_rand* of any kind exists) -- this isn't a missing-ggml-mapping gap the way most
    # other NotImplementedError cases in this file are, it's a hard architectural boundary:
    # a static per-submodule topology (what this function produces) is a pure, deterministic
    # dataflow graph GraphBuilder builds once and reuses; there is no way to make one of its
    # nodes produce fresh randomness on each call. Fail with a message pointing at the actual
    # fix (already-existing infrastructure, not something new to build) rather than the
    # generic "missing a ggml mapping" message below, which would suggest this is just an
    # unimplemented translation rather than something this function can never do.
    raise NotImplementedError(
        f"MIL op '{op_type}' (from '{op.name}') can't be translated into a static topology "
        "node -- ggml has no RNG-capable compute op. The fix is to hoist whatever "
        "torch.randn()/torch.rand()-style call produced this op OUTSIDE this submodule's own "
        "trace boundary (e.g. via a 'prefix'/'aux' split in a ModularExportSpec, see "
        "modular_export.py) and feed the pre-sampled noise in as an explicit input instead "
        "-- sampled at runtime via the existing loom.gaussian_array/loom.uniform_array/"
        "loom.seed_rng Lua host functions (src/core/lua_bridge.cpp), the same host-side-RNG "
        "pattern every hand-written driver (VitsDriver/SupertonicDriver/MatchaDriver/...) "
        "already uses."
    )



if __name__ == "__main__":
    # `python3 -m loom_mil_compiler.topology_ops` prints the whole rewrite table -- the audit view this
    # refactor exists to make possible.
    print(describe_topology_rules())
