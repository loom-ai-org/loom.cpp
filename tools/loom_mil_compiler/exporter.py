import json
import math
import re
import os
import numpy as np
import sympy
from coremltools.converters.mil.mil import Block, Function, Operation, Var

from . import axes
from .driver_ir import (
    Argmax, Assign, BinOp, Break, Call, FieldAccess, If, Index, Len, Lit, Local, LocalDecl,
    LuaCodegen, OutputRef, RawBlock, RawExpr, Return, SubgraphCall, UnaryOp, Var as IRVar, While,
    check_subgraph_calls, validate,
)
from .driver_ir import Function as IRFunction
from .driver_builder import DriverContext, DriverScript
from .driver_components import (
    CALLER, CAUSAL_MASK_INPUT_NAMES, HOST_COMPUTED_INPUT_NAMES, MASK, POSITION,
    POSITION_INPUT_NAMES, SYNTHESIZED_BUILDERS, ArgmaxEpilogue, ChainStage, CtcGreedyEpilogue,
    DriverInputs, ModularChain,
    MonolithicCall, PrefillDecodeLoop,
)
from .passes import apply_loom_mil_passes
from .shape_expr import (
    as_expr, floor_div, has_dynamic_symbol, render, sub_dynamic_symbols,
)
from .symbols import DYNAMIC_SYMBOL_RE
from .topology_ops import TopologyContext, lookup_topology_rule
from .value_facts import ValueFacts, is_const_producer, static_array, static_scalar, static_value

def _binding_kind(name: str) -> str:
    """How the driver obtains one traced-model input: computed host-side, or read from the caller.

    The two host-computed sets live in `driver_components.py` alongside the component that acts on
    them (P4.0.6/C.2). A traced model's own `cache_position`/`position_ids`/`attention_mask` inputs
    exist because passing them explicitly is what keeps the sequence length genuinely dynamic under
    `torch.jit.trace`; the driver knows `n_tokens`/`n_past` and fills them in, so a caller never has to
    know they are there."""
    if name in POSITION_INPUT_NAMES:
        return POSITION
    if name in CAUSAL_MASK_INPUT_NAMES:
        return MASK
    return CALLER


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LoomGGUFExporter:
    # A mapping from standard/custom MIL op_types to Loom's C++ register_op primitives.
    OP_MAP = {
        "add": "ADD",
        "sub": "SUB",
        "mul": "MUL",
        "div": "DIV",
        "real_div": "DIV",
        # `floor_div` (PyTorch `//`) is genuinely NOT the same op as `real_div` -- it floors the
        # quotient. Mapping it to the same plain "DIV" primitive as `real_div` silently dropped that
        # floor -- confirmed as a real, load-bearing bug on Conformer-CTC's NeMo-style `calc_length()`
        # subsampled-length formula (BACKLOG.md): the resulting fractional "current_lengths" value
        # (32.5 instead of 33) fed a `LESS` comparison that incorrectly zeroed out the last valid
        # subsampled frame via the CNN-subsampling padding mask, corrupting the entire encoder output
        # from the very first conv stage onward. See op_floor_div (src/ops/primitives_basic.cpp) for why
        # this can't just be a MIL-graph-level FLOOR node instead: coremltools' own tracing had already
        # eliminated the standalone `torch.floor()` op as a no-op for the specific dummy trace length
        # used, so `floor_div` is the only place this exporter can still recover the real semantics.
        "floor_div": "FLOOR_DIV",
        "matmul": "MUL_MAT",
        "relu": "RELU",
        "gelu": "GELU",
        "silu": "SILU",
        "softmax": "SOFTMAX",
        "reshape": "RESHAPE",
        "transpose": "PERMUTE",
        "concat": "CONCAT",
        "gather": "GET_ROWS",
        "reduce_mean": "MEAN",
        "layer_norm": "LAYER_NORM",
        "rms_norm": "RMS_NORM",
        "sigmoid": "SIGMOID",
        "tanh": "TANH",
        "exp": "EXP",
        "sin": "SIN",
        "cos": "COS",
        "atan": "ATAN",
        "atan2": "ATAN2",
        "floor": "FLOOR",
        "clamp": "CLAMP",
        "pow": "POW",
        "square": "SQR",  # MIL's dedicated unary x**2 op (e.g. SnakeBeta's torch.pow(x,2)) -- ggml
                          # already has this exact primitive (SQR), just wasn't wired into OP_MAP yet
                          # (same category of gap as the pre-existing sqrt/log entries above).
        "rsqrt": "RSQRT",
        "sqrt": "SQRT",
        "softplus": "SOFTPLUS",  # ggml already has this primitive (used by e.g. Mish = x*tanh(softplus(x))
                                 # in Matcha-TTS's/SupertonicTTS's own bespoke conversions); just wasn't
                                 # wired into OP_MAP yet, same category of gap as "square"/sqrt/log above.
        "log": "LOG",
        "logical_and": "MUL",
        "shape": "SHAPE",
        "range_1d": "RANGE_1D",
        "expand_dims": "RESHAPE",
        "squeeze": "RESHAPE",
        "less_equal": "LESS_EQUAL",
        "greater_equal": "GREATER_EQUAL",
        "less": "LESS",
        "greater": "GREATER",
        "equal": "EQUAL",
        "not_equal": "NOT_EQUAL",
        "logical_not": "NOT",
        "select": "SELECT",
        "abs": "ABS",
        "neg": "NEG",
        "sign": "SIGN",
        "minimum": "MINIMUM",
        "maximum": "MAXIMUM",
        "reduce_sum": "REDUCE_SUM",
        "identity": "IDENTITY",
        # Specialized Loom dialect ops:
        "loom_fused_attention": "ATTENTION",
        "loom_spline": "RQ_SPLINE_INVERSE",
        "loom_rope": "ROPE",
        "loom_group_norm": "GROUP_NORM",
    }

    def __init__(self, program, **kwargs):
        import os
        self.program = program
        self.kwargs = kwargs
        self.weights = {}
        self.topologies = {}
        # The traced programs behind `topologies` when this exporter has no `program` of its own -- the
        # multi-phase case, where each phase was converted by its own exporter and only the finished
        # topologies were handed over. Set by `decomposition.MultiPhase`; read by `_fused_ops`, which is
        # how a KV-cached phase's cache geometry still reaches the GGUF. Empty for every other path.
        self.phase_programs = []
        # {declared input name: window}, filled by `_route_windowed_masks` as each topology is
        # generated and read by the driver assembly, which is what turns a synthesized mask input into
        # a `loom.causal_mask(n_tokens, n_past, window)` call (BACKLOG.md P4.0.11a).
        self.mask_windows = {}
        # The built driver (`driver_builder.DriverScript`: top-level prelude chunks + the entry
        # function), set by whichever of the three paths `export()` dispatches to. Was a bare
        # `IRFunction` until P4.0.6/C.2 -- a driver is a Lua module, not a function, and the two
        # synthesized paths now reach it through a `DriverBuilder` rather than building it inline.
        self.driver_script = None
        # Whether this export writes weights into ONE flat namespace instead of prefixing each with its
        # own topology's name (`{func_name}.{weight}`). Read by `topology_ops.py` in 8 places, always as
        # `func_name == "main_topology" or self.flat_namespace`.
        #
        # Was `profile` ("monolithic"/None) until BACKLOG.md P4.0.3's rename. That name described the
        # caller's export shape rather than the switch's effect, and it carried a second, unrelated
        # meaning: `profile is None` also used to enable the bespoke hand-built-Program path, which is
        # now decided structurally by `is_bespoke` alone (no caller ever passed a profile to suppress
        # it -- both are checked in `export()`). The correlation is real but partial: a flattened
        # single-topology export wants a flat namespace, while a modular or multi-phase export needs its
        # per-function prefixes, which is why those paths simply leave this False.
        self.flat_namespace = bool(kwargs.get("flat_namespace"))
        self.output_path = kwargs.get("output_path") or os.environ.get("LOOM_OUTPUT_PATH", "model.gguf")
        self.quantize = kwargs.get("quantize") or os.environ.get("LOOM_QUANTIZE", None)
        # This topology's ONE true dynamic quantity's real name (EXPORT-ROADMAP.md R1, axes.py) --
        # "n_tokens" unless the caller says otherwise. The engine's own dynamic-shape support is
        # genuinely single-axis (see get_var_info's own docstring): every symbolic dim ordinarily
        # collapses to whichever single symbol this names, which SymbolEnv resolves at build time.
        # Conformer-CTC/Parakeet declare "n_samples" here (raw audio samples, never a token count);
        # Kokoro's decoder_vocoder phase declares "n_enc_frames" (see `declared_axes` below for why that
        # phase ALSO needs more than just this one name).
        self.root_axis = kwargs.get("root_axis") or "n_tokens"
        # {raw MIL symbol string (e.g. "is531") -> replacement expression (e.g. "2*n_enc_frames")},
        # resolved from the human-facing `declared_axes` kwarg ({input name: {torch axis: expression}})
        # against `program`'s own real traced Vars -- see `_resolve_declared_axes`. Needed whenever a
        # topology has more than one independently-varying dynamic axis (first hit by Kokoro's
        # "decoder_vocoder" phase: asr's own frame count vs. f0_curve/n_curve's fixed-2x/noise_in's
        # fixed-600x/wsum's fixed-600x+20 lengths -- none of these are op-derived from asr, they're
        # independently-traced LEAF inputs, so there's no data-flow path this exporter could use to
        # infer the ratio on its own). A caller who DOES know the real ratio (because it's inherent to
        # how their own wrapper module's forward() signature is shaped, not something recoverable from
        # the graph) declares it per input name and axis position; this exporter looks up the input's
        # own real MIL symbol so nothing here is guessed from string patterns.
        self._axis_overrides = self._resolve_declared_axes(program, kwargs.get("declared_axes"))
        self._validate_input_axes(program)
        # The one place any "what is this Var's compile-time value?" question gets answered, for both
        # this class and the op-handler table -- see value_facts.py's own module docstring.
        self.facts = ValueFacts(self)
        # Set by `_ensure_mil_passes_applied` the first time it runs (or decides not to, for the
        # bespoke/MockOperation workflow) -- see that method's own docstring.
        self._mil_passes_applied = False

    def _resolve_declared_axes(self, program, declared_axes):
        """`{input name: {torch axis: expression}}` -> `{raw MIL symbol name: expression}`, by reading
        each named input's own real traced shape out of `program`'s main function. Moves the "read the
        real symbol off the traced Var" step (`export_kokoro_mil.py`'s own former `root_symbol()`
        helper) inside the exporter, so a caller declares an axis by input name and position -- a
        first-class fact about the model, matching R1's design sketch -- rather than by a raw
        coremltools-internal symbol string it has to extract itself."""
        if not declared_axes or program is None:
            return {}
        main_func = getattr(program, "functions", {}).get("main")
        if main_func is None:
            return {}
        overrides = {}
        for input_name, per_axis in declared_axes.items():
            input_var = main_func.inputs.get(input_name)
            if input_var is None or input_var.shape is None:
                raise KeyError(
                    f"declared_axes: no input {input_name!r} in the traced program's main function"
                )
            for axis, expr in per_axis.items():
                dim = input_var.shape[axis]
                # A declaration for an axis that isn't actually dynamic is dead code that reads as
                # live: `str(4000)` is a perfectly good dict key that no MIL symbol will ever match, so
                # without this the entry would silently do nothing (BACKLOG.md P4.0.2).
                if not has_dynamic_symbol(dim):
                    raise ValueError(
                        f"declared_axes: {input_name!r} axis {axis} is static ({dim}), so declaring it "
                        f"as {expr!r} would have no effect -- either make that axis a ct.RangeDim in "
                        f"this phase's mil_inputs, or drop the declaration"
                    )
                overrides[str(dim)] = expr
        self._reject_shared_symbol_overrides(program, declared_axes, overrides)
        return overrides

    def _reject_shared_symbol_overrides(self, program, declared_axes, overrides):
        """An override keyed on a symbol that ANOTHER, undeclared input also carries rewrites that input
        too, silently.

        `_sub_symbol` substitutes per raw MIL symbol, not per input -- so declaring one input's axis
        moves every input sharing its `ct.RangeDim` instance. For the axes `declared_axes` was written
        for (Kokoro's f0_curve/n_curve/noise_in/wsum) the question never arises: those are independently
        traced leaves with their own range dims and their own symbols.

        Where it does arise is the causal-LM family, whose `tokens`/`cache_position`/`attention_mask`
        share ONE `ct.RangeDim` deliberately (see `causal_lm_export.build_trace`) -- so the obvious way
        to give the fused mask its own `n_kv` axis is to declare it here, and the obvious way is wrong:
        it would retype `tokens` and `cache_position` as well. That is why `axes.N_KV` is applied to the
        emitted topology instead, and why this raises rather than letting the attempt look like it
        worked."""
        if not overrides:
            return
        declared_sites = {
            (input_name, axis)
            for input_name, per_axis in declared_axes.items()
            for axis in per_axis
        }
        collisions = {}
        for input_name, axis, symbol_name in self._input_axis_symbols(program):
            if symbol_name in overrides and (input_name, axis) not in declared_sites:
                collisions.setdefault(symbol_name, []).append(f"{input_name}[{axis}]")
        if not collisions:
            return
        detail = "; ".join(
            f"{sym} (declared as {overrides[sym]!r}) is also carried by {', '.join(where)}"
            for sym, where in sorted(collisions.items())
        )
        raise ValueError(
            f"declared_axes: {detail}. Substitution is per MIL symbol, not per input, so this "
            f"declaration would silently retype those inputs too -- inputs sharing one ct.RangeDim "
            f"instance cannot be given different axes here. If this is a fused causal LM's mask, its "
            f"'n_kv' axis is applied to the emitted topology after fusion instead (axes.N_KV); "
            f"otherwise give the input its own ct.RangeDim in this phase's mil_inputs."
        )

    def _input_axis_symbols(self, program):
        """`[(input name, axis index, raw MIL symbol name)]` for every dynamic axis of every declared
        input of `program`'s main function.

        Returns `[]` rather than raising for anything that has no such function to read: no program at
        all (the write-only exporter `multi_phase_export.export()` builds to merge already-generated
        topologies), or a modular-blueprint Program, which has one Function per submodule and no "main"
        (see `export()`). The modular path therefore gets no axis validation -- a real limit of this
        check, not an oversight: `apply_modular_export` synthesizes its own leaf inputs and their axes
        rather than taking them from a caller."""
        main_func = getattr(program, "functions", {}).get("main") if program is not None else None
        if main_func is None:
            return []
        found = []
        for input_name, input_var in (main_func.inputs or {}).items():
            shape = getattr(input_var, "shape", None)
            if shape is None:
                continue
            for axis, dim in enumerate(shape):
                for free in as_expr(dim).free_symbols:
                    if DYNAMIC_SYMBOL_RE.fullmatch(free.name):
                        found.append((input_name, axis, free.name))
        return found

    def _validate_input_axes(self, program):
        """Every dynamic input axis must be accounted for: either declared via `declared_axes`, or
        carrying the SAME MIL symbol as every other undeclared one, in which case it is this topology's
        `root_axis`.

        This is the rule the exporter has always relied on and never checked (BACKLOG.md P4.0.2).
        `_sub_symbol` rewrites any symbol it wasn't given an override for into `root_axis`, so two
        genuinely independent dynamic quantities silently collapse into one name and the emitted shape
        expressions are wrong -- not malformed, just wrong, which no downstream gate catches. The fix
        callers already use is one of two things, and the error says both: share a single `ct.RangeDim`
        INSTANCE across inputs that really do move together (coremltools then gives them one symbol, as
        `causal_lm_export`'s tokens/cache_position/attention_mask do deliberately), or declare the real
        relationship (as Kokoro's `decoder_vocoder` phase does for f0_curve/n_curve/noise_in/wsum, whose
        lengths are fixed multiples of `asr`'s and are not derivable from the graph).

        **`axes.N_KV` is deliberately invisible here, and that is not a hole.** A fused causal LM's mask
        carries `n_kv` in the *emitted topology*, applied after `fuse_loom_attention` by
        `_retype_fused_mask_input`. It cannot come from the trace at all -- two independent `ct.RangeDim`s
        over one attention block fail coremltools' type inference (KV-CACHE.md §2) -- so the traced
        program a fused causal LM presents to this check still has exactly one dynamic symbol, and passes
        it for the same reason it always did."""
        uncovered = {}
        for input_name, axis, symbol_name in self._input_axis_symbols(program):
            if symbol_name in self._axis_overrides:
                continue
            uncovered.setdefault(symbol_name, []).append(f"{input_name}[{axis}]")
        if len(uncovered) <= 1:
            return
        groups = "; ".join(f"{sym} -> {', '.join(where)}" for sym, where in sorted(uncovered.items()))
        raise ValueError(
            f"this topology has {len(uncovered)} independent dynamic input axes, but only one can be "
            f"the root axis ({self.root_axis!r}); the rest would silently collapse onto it. Found: "
            f"{groups}. Either share one ct.RangeDim instance across inputs whose lengths always match, "
            f"or declare the others via declared_axes={{input: {{axis: expression}}}}."
        )

    def _sub_symbol(self, dim):
        """Replaces every symbolic MIL dim (e.g. "is531") in `dim` with its `self._axis_overrides`
        entry if the caller declared one for that exact raw symbol, else this topology's own
        `self.root_axis`. See `root_axis`/`declared_axes`' own docstrings in __init__.

        Substitution happens on the sympy expression MIL itself handed us -- coremltools' own `Symbol`
        subclasses `sympy.Symbol`, so a compound dim like `4*is2 + 20` keeps its algebra rather than
        being rebuilt from its printed form (see shape_expr.py's module docstring)."""
        return sub_dynamic_symbols(dim, self._axis_overrides, default=self.root_axis)

    def safe_name(self, name: str) -> str:
        """
        Sanitizes MIL SSA variable/op names to be safe for Lua identifiers
        by replacing characters like %, ., / and prepending _ if starts with a digit.
        """
        for c in "%./-+$#@!&*()[]{}|<>?;:":
            name = name.replace(c, "_")
        if name and name[0].isdigit():
            name = "_" + name
        return name

    def _infer_dynamic_dim_expr(self, var, torch_axis, _seen=None):
        """Memoizing entry point for the shape-expression walk below, returning a **sympy expression**
        (or None) -- see `ValueFacts.dim_expr` for why the memo is load-bearing rather than an
        optimization, and shape_expr.py for why the walk composes algebra rather than strings. `_seen`
        is accepted and ignored (every recursive call site still threads it; the walk itself has had no
        cycle guard since a29ffe5)."""
        return self.facts.dim_expr(var, torch_axis)

    def _infer_dynamic_dim_expr_uncached(self, var, torch_axis, _seen=None):
        """
        Best-effort derivation of the REAL SymbolEnv expression for one symbolic MIL shape dimension, by
        walking `var`'s own producer chain backward through ops that are known to either preserve or
        transform that dimension via a real formula -- rather than `get_var_info`'s default of collapsing
        every symbolic dim straight to the bare string "n_tokens".

        That default is correct when the symbol is a pure pass-through of the topology's one true dynamic
        input (the common case, and the only one exercised before this), but wrong when a real
        *derived* quantity is involved -- confirmed on Conformer-CTC's STFT-via-CONV_1D mel frontend:
        `torch.stft`'s conv-based framing turns the raw sample count ("n_tokens") into a frame count via
        `floor((n_tokens + 2*pad - kernel) / stride) + 1`, a materially different number
        (101 vs. 16000 for a 1-second clip) that a bare substitution silently gets wrong -- not a syntax
        error, a wrong shape fed straight into a RESHAPE/CONV_1D_DW/etc. downstream.

        Deliberately narrow and safe to fail out of: only handles `cast` (pure alias, recurse unchanged)
        and `conv` (real 1D/2D stride/pad/kernel formula, recursing into the same spatial axis of its own
        `x` input) -- any other producer, or a producer this walk can't fully explain, falls back to
        `get_var_info`'s original bare-substitution behavior, so every case this doesn't specifically
        understand is byte-identical to before this method existed.

        Returns a **sympy expression** (see shape_expr.py), not a string: MIL hands this walk sympy
        objects to begin with (`coremltools...mil.Symbol` subclasses `sympy.Symbol`), and composing the
        conv/pool/pad formulas below as algebra rather than as f-strings is what lets an expression like
        `floor(1 * n_tokens * 512 / 512)` collapse back to `n_tokens` instead of accumulating into the
        unreadable nested-`floor` strings this used to emit. Rendering into the engine's expression
        language happens once, where a shape attribute is actually emitted (`get_var_info`), so an
        expression the engine could not parse raises there -- naming the construct -- rather than
        shipping inside a GGUF and failing at model load.
        """
        # No cycle guard: MIL/SSA graphs are acyclic by construction (an op's inputs always name EARLIER-
        # defined vars, never itself), so a genuine infinite loop here isn't possible. A guard keyed on
        # `id(var)` alone WAS added once, then found to actively corrupt correct answers instead -- the
        # "concat" case just above is the first branching walk in this function (recursing into multiple
        # operands from one call), and two operands legitimately sharing a common upstream ancestor (a
        # real DAG diamond, not a cycle -- confirmed on Kokoro's SineGen: `rad0.unsqueeze(1)` and
        # `rad_values[:,1:,:]` both trace back to the SAME `rad_values` var) would hit a per-walk `_seen`
        # set's SECOND visit and silently return None, exactly the bug already root-caused and fixed the
        # identical way in `facts.scalar_expr`'s own cycle guard for VITS (see BACKLOG.md).
        if var.shape is None or torch_axis >= len(var.shape):
            return None
        dim = var.shape[torch_axis]
        if not has_dynamic_symbol(dim):
            return as_expr(dim)

        op = var.op
        if op is None:
            # A genuine (sub)function input with no producer -- ordinarily this IS the topology's own
            # `root_axis` (e.g. "waveform" itself), but not always: a topology with more than one
            # independently-traced dynamic LEAF input (Kokoro's "decoder_vocoder" -- see
            # `declared_axes`' own docstring in __init__) has NO data-flow path here to tell that apart
            # from the ordinary case, so an explicit caller-declared axis always wins when present.
            return self._sub_symbol(dim)

        _UNARY_PASSTHROUGH_OPS = {
            "cast", "log", "exp", "sqrt", "rsqrt", "abs", "neg", "sign", "floor", "clamp", "clip",
            "tanh", "sigmoid", "relu", "gelu", "softplus", "identity", "softmax", "logical_not", "silu",
            "leaky_relu", "cumsum", "atan", "sin", "cos", "square",
        }
        if op.op_type in _UNARY_PASSTHROUGH_OPS:
            # Pure unary, shape-preserving ops -- the axis's real expression is whatever its single
            # input's already is. Needed for the SAME reason the elementwise-broadcast case is: a chain
            # like log(x+eps) sitting between the true dynamic source and a "shape" op that reads this
            # var's real (built) tensor dimensions back out (confirmed on Conformer-CTC's mel-frontend
            # length tracking -- `gather(shape(real_div(...(log(...(matmul with the STFT conv's own
            # output)...)))))` -- every one of those intermediate unary ops needed to be walked through,
            # not just the ones this file happened to hit first).
            #
            # `square` and `clip` were the next two, found by GigaAM v3 (BACKLOG.md P4.2) and both from
            # the same three lines of mel frontend. `spec.abs()` on a COMPLEX tensor lowers to
            # `sqrt(square(re) + square(im))`, so a `.abs()`-based magnitude puts a `square` immediately
            # downstream of the STFT conv whose frame-count formula this walk exists to derive -- and
            # `torch.clamp` converts to MIL's `clip`, not to `clamp`, so the entry above it never
            # matched anything. Both gaps were invisible until now because NeMo's own preprocessor
            # spells the same magnitude `view_as_real(...).pow(2).sum(-1)` (`pow` is elementwise-binary,
            # `reduce_sum` has its own case) and never calls either. The failure they produced is worth
            # recording: not an error, a silent fallback to the bare root axis, so the encoder's frame
            # count came out as `floor(floor((n_samples - 1)/2)/2) + 1` -- the subsampling formula with
            # the STFT's own /160 simply missing -- and the export succeeded. It failed at RUN time, in
            # the engine, as a rotary-table VIEW asking for 44000 rows of a 10000-row constant.
            inner = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
            if inner is not None:
                return self._infer_dynamic_dim_expr(inner, torch_axis, _seen)
            return None

        if op.op_type in ("layer_norm", "instance_norm", "loom_group_norm"):
            # Shape-preserving over EVERY axis (normalizes in place, never changes rank or size) -- the
            # real expression for any symbolic output axis is just its input's own, unchanged. Needed
            # for the same reason the elementwise-broadcast case is: `layer_norm`'s own composed
            # LAYER_NORM+MUL+ADD translation (see the "layer_norm" op_type branch below) sits directly
            # upstream of every Q/K/V linear projection's reshape in Conformer-CTC's encoder, and a
            # `gather(shape(...), 0)` reading the (always-1) batch axis back out needs to walk straight
            # through it rather than giving up and falling back to a bare "n_tokens" substitution.
            # `instance_norm` (added alongside `layer_norm` here, not as its own separate case -- same
            # formula, no per-axis distinction needed since NEITHER changes any dim) is Kokoro's own
            # `AdaIN1d`/`AdainResBlk1d` normalization, used inside every Generator resblock/noise_res
            # call -- without this, a length several conv_transpose/upsample hops downstream of an
            # instance_norm output fell through to bare substitution, silently corrupting the SECOND
            # (i=1) Generator upsample stage's own input length (confirmed: produced literal "31"
            # instead of the real "20*n_tokens"-derived 601, a `x = x + x_source` shape mismatch two
            # ops later at `ups[1]`'s own conv_transpose). `loom_group_norm` (Matcha-TTS's `Block1D`,
            # see group_norm_op.py) sits directly upstream of the Decoder U-Net's own `conv_transpose`
            # (every `ResnetBlock1D` has two GroupNorms, always ahead of the next downsample/upsample) --
            # without this case, the SAME class of bug hit again: the crop-target shape derived for
            # `up_blocks[0]`'s real ConvTranspose1d silently fell back to bare substitution once the
            # backward walk hit this custom op and stopped, confirmed via a real `ggml_conv_transpose_1d`
            # VIEW-crop-out-of-bounds crash (target shape assumed a longer input than the real one).
            inner = op.inputs.get("x")
            if inner is not None:
                return self._infer_dynamic_dim_expr(inner, torch_axis, _seen)

        if op.op_type in ("reshape", "fill"):
            # Unlike `expand_dims`/`squeeze`, a general `reshape` (or a dynamically-shaped `fill`, which
            # has the exact same "shape" INPUT structure -- see the "fill" op_type translation branch
            # above) has no per-axis correspondence formula
            # to the INPUT at all (elements get freely redistributed) -- the only reliable source of a
            # symbolic output axis's true value is the op's own "shape" INPUT, resolved the exact same
            # way the "reshape" translation branch (above, in `export()`) already does when building this
            # very node's JSON. Reusing `facts.reshape_shape` here (rather than duplicating
            # that logic) is what makes THAT already-correct answer visible to a completely different
            # consumer: a `gather(shape(q), 2)` reading Q's own post-reshape sequence length back out one
            # step further downstream (confirmed on Conformer-CTC's `rel_shift`, whose `b, h, qlen,
            # pos_len = x.size()` queries `matrix_bd`'s shape, which recurses through `matmul` into the
            # Q/K/V reshape's own output -- without this case, that walk gave up at "reshape" (no case
            # existed for it at all) and fell back to the same "n_tokens" substitution the Q reshape
            # itself needed `facts.reshape_shape` to avoid).
            resolved = self.facts.reshape_shape_exprs(op)
            if resolved is not None and torch_axis < len(resolved):
                if resolved[torch_axis] != -1:
                    return resolved[torch_axis]
                # A literal -1 at this axis is PyTorch's own "infer this dim" marker (real, correctly
                # left as-is for op_reshape's OWN build-time element-count inference when THIS node's
                # JSON gets built -- see the "reshape" translation branch above) -- but a caller here
                # needs an actual FORMULA, not a runtime-only placeholder. The general, always-correct
                # answer (by definition of what "-1" means) is `total_elements(x) / product(every OTHER
                # resolved target axis)` -- NOT a per-axis positional correspondence to `x`'s own shape:
                # an earlier version of this code assumed the `-1` axis maps to `x`'s SAME-position axis,
                # which is only true when the reshape doesn't reorder which physical axis holds which
                # logical quantity. Confirmed wrong on Conformer-CTC's `rel_shift`: `x_29` (shape
                # `(b,h,qlen,pos_len+1)`) reshaped to `(b,h,-1,qlen)` PUTS `pos_len+1` at the "-1"
                # position, swapping `x_29`'s own trailing two axes rather than passing either through
                # positionally -- the same-position guess silently returned `qlen` (WRONG, `x_29`'s own
                # axis 2) instead of `pos_len+1` (`x_29`'s own axis 3). The total/other formula sidesteps
                # needing to know WHICH input axis correspond to which output axis at all.
                x_var = op.inputs.get("x") or op.inputs.get("data")
                if x_var is not None and x_var.shape is not None:
                    # Each axis is resolved with its OWN fresh cycle-guard set, not the shared `_seen`
                    # threaded down from the caller -- these are independent explorations of `x`'s
                    # several axes, not a single recursive path, and reusing `_seen` across them made an
                    # already-visited-by-a-SIBLING-axis id(var) look like a genuine cycle, incorrectly
                    # returning None and poisoning the whole total-element-count computation.
                    x_axis_exprs = [self._infer_dynamic_dim_expr(x_var, a) for a in range(len(x_var.shape))]
                    other_exprs = [resolved[a] for a in range(len(resolved)) if a != torch_axis]
                    if all(e is not None for e in x_axis_exprs) and all(e != -1 for e in other_exprs):
                        total_expr = math.prod(x_axis_exprs, start=as_expr(1))
                        other_expr = math.prod(other_exprs, start=as_expr(1))
                        # The division is exact whenever the reshape is (it redistributes the same
                        # elements), so this is where carrying algebra pays for itself most visibly:
                        # StyleTTS2's diffusion axis, `floor(1 * n_tokens * 512 / 512)`, cancels back
                        # to plain `n_tokens` instead of nesting another floor() around the last one.
                        return floor_div(total_expr, other_expr)

        if op.op_type == "range_1d" and torch_axis == 0:
            # `range_1d`'s own output LENGTH is a real formula over its start/end/step, using the exact
            # same resolution `facts.range_scalar`/`facts.gather_shape_value` already apply when
            # emitting this op's own JSON node -- needed one level further downstream than that node
            # itself: a `reshape`/`repeat` consuming THIS range's real output (e.g. broadcasting a
            # length-validity arange up before a comparison) queries ITS OWN declared shape, which is
            # this range's element count, not a bare "n_tokens" substitution.
            start_e = self.facts.range_scalar(op.inputs.get("start"))
            end_e = self.facts.range_scalar(op.inputs.get("end"))
            step_e = self.facts.range_scalar(op.inputs.get("step"))
            if start_e is not None and end_e is not None and step_e is not None:
                start_e, end_e, step_e = as_expr(start_e), as_expr(end_e), as_expr(step_e)
                if start_e == 0 and step_e == 1:
                    return end_e
                return floor_div(end_e - start_e, step_e)

        if op.op_type == "conv":
            x_var = op.inputs.get("x")
            weight_var = op.inputs.get("weight")
            strides = static_value(op.inputs.get("strides"))
            pad = static_value(op.inputs.get("pad"))
            dilations = static_value(op.inputs.get("dilations"))
            if x_var is None or weight_var is None or strides is None or pad is None or x_var.shape is None:
                return None
            rank = len(var.shape)
            # MIL conv is NC(D...) -- axis 0 is batch, axis 1 is out-channels (from the weight, not
            # derived from `x`, genuinely unresolvable here), and the remaining `rank - 2` axes are the
            # real spatial ones this formula applies to, in the same order for `x` and its output
            # (conv never permutes spatial axes).
            #
            # **The batch axis RECURSES rather than answering 1**, which is the same correction P4.2
            # made to `gather_shape_value` and for the same reason. Conv preserves the batch axis, so
            # "whatever `x`'s axis 0 is" is a real formula; `1` was an architectural assumption that
            # happens to hold for every model whose batch is genuinely one -- and those still resolve
            # to a literal 1 through the recursion, so nothing about them changes. Family 3's audio
            # encoder is the counterexample: it folds the CHUNK COUNT into the batch axis so the
            # convolutional stem sees a fixed 100-frame window per chunk, and reading that as 1 did not
            # fail -- it silently made the post-stem sequence length 13 instead of 13 per chunk, which
            # surfaced hundreds of ops later as a mask whose two sides had different lengths
            # (BACKLOG.md P4.3).
            #
            # The old fallback is preserved exactly: this function's own trailing "torch_axis == 0 ->
            # batch = 1" answer still applies when the recursion has nothing better, which is the case
            # the comment this replaces was really recording (Conformer-CTC's GLU-split VIEW, whose
            # `x` operand's own axis-0 walk bottoms out here).
            if torch_axis == 0:
                batch_expr = self._infer_dynamic_dim_expr(x_var, 0, _seen)
                return batch_expr if batch_expr is not None else as_expr(1)
            if torch_axis == 1:
                return None
            spatial_idx = torch_axis - 2
            n_spatial = rank - 2
            if n_spatial not in (1, 2) or spatial_idx >= n_spatial:
                return None
            stride = int(strides[spatial_idx]) if spatial_idx < len(strides) else 1
            dilation = int(dilations[spatial_idx]) if dilations is not None and spatial_idx < len(dilations) else 1
            kernel = int(weight_var.shape[2 + spatial_idx])
            eff_kernel = dilation * (kernel - 1) + 1
            # `pad` is [before, after] per spatial axis, flattened (2D conv: [top, bottom, left, right]).
            pad_before = int(pad[2 * spatial_idx]) if 2 * spatial_idx < len(pad) else 0
            pad_after = int(pad[2 * spatial_idx + 1]) if 2 * spatial_idx + 1 < len(pad) else 0
            in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
            if in_expr is None:
                return None
            return floor_div(in_expr + (pad_before + pad_after) - eff_kernel, stride) + 1

        if op.op_type in ("upsample_nearest_neighbor", "upsample_bilinear"):
            # MIL's `upsample_nearest_neighbor`/`upsample_bilinear` (core ops, see the matching op_type
            # branch in `export()`'s main loop for the node-emission side of this) always operate on the
            # LAST TWO axes (height=-2, width=-1), scaling each independently by its own constant
            # scale_factor_height/scale_factor_width -- floor(H1*sfh)/floor(W1*sfw). Without this case,
            # get_var_info's default fallback for this op's OWN output var collapsed to a bare "n_tokens"
            # substitution regardless of the real scale, silently corrupting every consumer downstream
            # (confirmed on Kokoro's f0_upsamp: a real "600*n_tokens" length read back as literal
            # "n_tokens", a 600x error that produced a genuinely zero-length tensor a few ops later once
            # SineGen's own 1/300 downsample floor()'d it back down).
            x_var = op.inputs.get("x")
            if x_var is None or x_var.shape is None:
                return None
            rank = len(var.shape)
            if torch_axis not in (rank - 2, rank - 1):
                # Any other axis passes straight through unscaled.
                return self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
            sfh_obj = op.inputs.get("scale_factor_height")
            sfw_obj = op.inputs.get("scale_factor_width")
            sfh = float(static_scalar(sfh_obj, 1.0))
            sfw = float(static_scalar(sfw_obj, 1.0))
            scale = sfh if torch_axis == rank - 2 else sfw
            in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
            if in_expr is None:
                return None
            if scale == 1.0:
                return in_expr
            return sympy.floor(in_expr * as_expr(scale))

        if op.op_type == "conv_transpose":
            # Mirrors the "conv" case just above, using ConvTranspose1d/2d's own inverse length
            # formula: L_out = (L_in-1)*stride - (pad_before+pad_after) + eff_kernel (dilation=1,
            # output_padding=0 always -- this exporter's own conv_transpose translation already rejects
            # anything else). First needed by HiFi-GAN's upsample stages (VITS): stage 0's real output
            # length is "(n_tokens-1)*8 - 8 + 16" = "n_tokens*8", NOT a bare "n_tokens" substitution --
            # confirmed wrong via the same class of bug this whole method exists to avoid. `weight`'s
            # layout for conv_transpose is [in_channels, out_channels/groups, *kernel] -- kernel starts
            # at axis 2, same offset as "conv"'s own weight layout.
            x_var = op.inputs.get("x")
            weight_var = op.inputs.get("weight")
            strides = static_value(op.inputs.get("strides"))
            pad = static_value(op.inputs.get("pad"))
            dilations = static_value(op.inputs.get("dilations"))
            if x_var is None or weight_var is None or strides is None or x_var.shape is None:
                return None
            rank = len(var.shape)
            if torch_axis == 0:
                return as_expr(1)
            if torch_axis == 1:
                return None
            spatial_idx = torch_axis - 2
            n_spatial = rank - 2
            if n_spatial not in (1, 2) or spatial_idx >= n_spatial:
                return None
            stride = int(strides[spatial_idx]) if spatial_idx < len(strides) else 1
            dilation = int(dilations[spatial_idx]) if dilations is not None and spatial_idx < len(dilations) else 1
            kernel = int(weight_var.shape[2 + spatial_idx])
            eff_kernel = dilation * (kernel - 1) + 1
            pad_before = int(pad[2 * spatial_idx]) if pad is not None and 2 * spatial_idx < len(pad) else 0
            pad_after = int(pad[2 * spatial_idx + 1]) if pad is not None and 2 * spatial_idx + 1 < len(pad) else 0
            in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
            if in_expr is None:
                return None
            return (in_expr - 1) * stride - (pad_before + pad_after) + eff_kernel

        if op.op_type == "matmul":
            # `matmul(x, y, transpose_x, transpose_y)`'s output rank-2 axes: the last axis comes from
            # `y`'s own last axis (or second-to-last if transpose_y), the second-to-last comes from `x`'s
            # own second-to-last axis (or last if transpose_x); any earlier (batch) axes broadcast from
            # whichever operand is symbolic there, same as the elementwise case below. Needed for
            # Conformer-CTC's mel-filterbank projection (`mel_spec = filterbank_const @ power_spectrum`,
            # a real matmul with one static and one dynamic operand) sitting in the middle of the same
            # length-tracking chain the unary-passthrough case above documents.
            x_var = op.inputs.get("x")
            y_var = op.inputs.get("y")
            tx_var = op.inputs.get("transpose_x")
            ty_var = op.inputs.get("transpose_y")
            transpose_x = bool(static_value(tx_var, False))
            transpose_y = bool(static_value(ty_var, False))
            rank = len(var.shape)
            if x_var is not None and y_var is not None and torch_axis == rank - 1:
                y_axis = len(y_var.shape) - 2 if transpose_y else len(y_var.shape) - 1
                return self._infer_dynamic_dim_expr(y_var, y_axis, _seen)
            if x_var is not None and y_var is not None and torch_axis == rank - 2:
                x_axis = len(x_var.shape) - 1 if transpose_x else len(x_var.shape) - 2
                return self._infer_dynamic_dim_expr(x_var, x_axis, _seen)
            if x_var is not None and y_var is not None and torch_axis < rank - 2:
                # A leading (batch) axis -- MIL matmul batch-broadcasts these numpy-style, right-aligned
                # against the trailing 2 "real" matmul axes each operand always has. Every batched matmul
                # actually seen so far (Conformer-CTC's per-head attention Q@K^T/attn@V) keeps BOTH
                # operands at the model's one true batch rank with no broadcast-insertion, so a direct
                # same-`torch_axis` correspondence (whichever operand has that many dims) is sufficient;
                # a genuinely differently-ranked-batch matmul would need the fuller right-aligned formula,
                # not implemented since nothing here has needed it yet.
                for operand in (x_var, y_var):
                    if operand.shape is not None and torch_axis < len(operand.shape) - 2:
                        inferred = self._infer_dynamic_dim_expr(operand, torch_axis, _seen)
                        if inferred is not None:
                            return inferred

        if op.op_type == "slice_by_index":
            # `slice_by_index` over a dynamic axis, for the one shape this walk can state exactly: a
            # CONSTANT begin/end at stride 1 with nothing squeezed, whose output length is
            # `end - begin` with negative bounds taken from the input's own (symbolic) length.
            #
            # Needed by every mel frontend that is genuinely dynamic. Whisper's drops the final STFT
            # frame the same way (`(stft.abs() ** 2)[..., :-1]`) and never needed this because its clip
            # is always 30 s, so every dim downstream is a literal; family 3's is variable-length, and
            # without this case the walk gave up exactly here and returned -1 -- which is not an error
            # anywhere, just a wrong number that reached a POOL_1D span as `-128` (BACKLOG.md P4.3).
            #
            # Deliberately narrow, and it falls through rather than guessing: a strided or
            # rank-changing slice has a real formula too, and inventing one here that nothing in this
            # tree exercises is how a silently wrong length gets shipped.
            x_var = op.inputs.get("x")
            # Explicit `is None` throughout: these are numpy arrays, whose truthiness raises rather
            # than answering, so an `or []` fallback is a crash and not a default.
            squeeze_mask = static_array(op.inputs.get("squeeze_mask"))
            begins = static_array(op.inputs.get("begin"))
            ends = static_array(op.inputs.get("end"))
            strides = static_array(op.inputs.get("stride"))
            begin_mask = static_array(op.inputs.get("begin_mask"))
            begin_mask = [] if begin_mask is None else list(begin_mask)
            end_mask = static_array(op.inputs.get("end_mask"))
            end_mask = [] if end_mask is None else list(end_mask)
            squeeze_mask = [] if squeeze_mask is None else list(squeeze_mask)
            if (x_var is not None and begins is not None and ends is not None
                    and not any(bool(s) for s in squeeze_mask)
                    and torch_axis < len(begins) and torch_axis < len(ends)
                    and (strides is None or int(strides[torch_axis]) == 1)):
                in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                if in_expr is not None:
                    masked_begin = torch_axis < len(begin_mask) and bool(begin_mask[torch_axis])
                    masked_end = torch_axis < len(end_mask) and bool(end_mask[torch_axis])
                    begin = as_expr(0) if masked_begin else as_expr(int(begins[torch_axis]))
                    end = in_expr if masked_end else as_expr(int(ends[torch_axis]))
                    # MIL keeps torch's negative-index convention: measured back from this axis's own
                    # length, which here is symbolic rather than known.
                    if not masked_begin and int(begins[torch_axis]) < 0:
                        begin = in_expr + int(begins[torch_axis])
                    if not masked_end and int(ends[torch_axis]) < 0:
                        end = in_expr + int(ends[torch_axis])
                    return end - begin

        if op.op_type == "linear":
            # `linear(x, weight, bias)` computes `x @ weight.T + bias`: same rank as `x`, every axis
            # unchanged except the last (which becomes `weight`'s static D_out) -- a direct passthrough
            # for every OTHER axis. Needed for the exact same reason `layer_norm` was: Conformer-CTC's
            # Q/K/V linear projections sit directly between a `gather(shape(...), 0)` reading the
            # (always-1) batch axis and the LAYER_NORM chain upstream of them -- without this, the walk
            # gave up at `linear` too.
            x_var = op.inputs.get("x")
            rank = len(var.shape)
            if x_var is not None and x_var.shape is not None and torch_axis < rank - 1 and torch_axis < len(x_var.shape):
                return self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)

        if op.op_type == "tile":
            # A real, constant `reps` (unlike the GQA repeat_kv case passes.py's fuse_gqa_repeat_kv
            # already handles, where `reps.val` is poisoned to None under tracing) needs the SAME
            # care as conv: `reps[axis] == 1` is a no-op for that axis (the value is unchanged, so the
            # correct expression is the INPUT's own, not a fresh "n_tokens" collapse) -- confirmed on
            # Conformer-CTC's `torch.arange(T).unsqueeze(0).tile(B, 1)` batch-broadcast: this engine's
            # "batch is always 1" design means axis 0's `reps` is always 1, but MIL's own shape algebra
            # still mints a genuine opaque symbol for that axis instead of simplifying it to the
            # literal 1 -- get_var_info's bare-substitution default would otherwise collapse THAT
            # symbol to "n_tokens" too, indistinguishable from the genuine time axis, corrupting any
            # downstream RESHAPE that (correctly) expects this axis to stay whatever it started as.
            reps_var = op.inputs.get("reps")
            x_var = op.inputs.get("x")
            reps_val = static_value(reps_var)
            if reps_val is not None and x_var is not None and torch_axis < len(reps_val):
                rep = int(reps_val[torch_axis])
                in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                if in_expr is not None:
                    return in_expr if rep == 1 else in_expr * rep
            elif x_var is not None and x_var.shape is not None and torch_axis < len(x_var.shape):
                # `reps` itself is unavailable (poisoned by the same "computed via a runtime shape
                # query" tracing artifact as the GQA case), so the real multiplier can't be read for
                # EITHER axis of a `tile`. Two sub-cases, both heuristic but bounded:
                if x_var.shape[torch_axis] == 1:
                    # The input axis is a literal, static 1 -- exactly the shape a batch-broadcast tile
                    # has, and this exporter's whole design never targets real multi-batch inference
                    # (every declared model input's own batch axis is a literal 1) -- resolves to "1".
                    return as_expr(1)
                # The input axis is ALREADY dynamic (not a static 1) -- a genuine multiplicative tile of
                # an already-dynamic axis is exactly what passes.py's dedicated `fuse_gqa_repeat_kv` MIL
                # pass exists to intercept and compose correctly (see EXPORT-BACKLOG.md); any plain
                # `tile` MIL op an exporter walk still sees here, with unreadable `reps`, is therefore
                # overwhelmingly likely NOT that case (it would have been fused away already) -- treat
                # `reps[axis]` as 1 (identity passthrough) and recurse. Confirmed on Conformer-CTC's
                # `arange(T).unsqueeze(0).tile(B, 1)`: axis 1 (the real, already-dynamic time axis) has
                # `reps[1]=1` structurally (only axis 0, the newly-unsqueezed one, is ever multiplied by
                # this idiom), but MIL's own shape algebra still mints a fresh, unrelated opaque symbol
                # for it regardless of the value actually changing.
                return self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)

        if op.op_type in ("expand_dims", "squeeze"):
            # Both insert/remove only genuine size-1 axes at known positions -- every OTHER axis has a
            # direct 1:1 correspondence to the input, just shifted by the inserted/removed position(s).
            # Needed alongside the elementwise-broadcast case above: `expand_dims`'s own declared output
            # shape is what generate_graph_topology's "reshape" branch actually queries (via
            # `get_var_info(out_var)`, not the input), so a symbol on an expand_dims/squeeze output needs
            # its OWN producer-chain walk here too, one level removed from wherever the real value comes
            # from (confirmed on Conformer-CTC's `valid_mask.unsqueeze(1)`).
            x_var = op.inputs.get("x") or op.inputs.get("data")
            axes_var = op.inputs.get("axes") or op.inputs.get("axis")
            axes_val = static_value(axes_var)
            if x_var is not None and axes_val is not None and x_var.shape is not None:
                out_rank = len(var.shape)
                in_rank = len(x_var.shape)
                norm_axes = sorted((int(a) + out_rank if a < 0 else int(a)) for a in axes_val) if op.op_type == "expand_dims" else \
                            sorted((int(a) + in_rank if a < 0 else int(a)) for a in axes_val)
                if op.op_type == "expand_dims":
                    if torch_axis in norm_axes:
                        return as_expr(1)
                    shift = sum(1 for a in norm_axes if a < torch_axis)
                    in_axis = torch_axis - shift
                    if 0 <= in_axis < in_rank:
                        return self._infer_dynamic_dim_expr(x_var, in_axis, _seen)
                else:  # squeeze: output axis maps back to input by re-inserting the removed positions
                    in_axis = torch_axis
                    for a in norm_axes:
                        if a <= in_axis:
                            in_axis += 1
                    if 0 <= in_axis < in_rank:
                        return self._infer_dynamic_dim_expr(x_var, in_axis, _seen)

        if op.op_type == "reduce_sum":
            # Same "output axis maps back to input by re-inserting the removed position(s)" logic as
            # `squeeze` above -- a `keep_dims=False` reduction genuinely drops the reduced axis, so
            # every axis AFTER it shifts down by one. Needed for the SAME STFT-magnitude chain the
            # elementwise-broadcast/pow cases document: `sum((real,imag)**2, axis=-1)` sits directly
            # between the STFT conv (whose frame-count formula this whole file exists to derive) and the
            # log/matmul/div chain a length-tracking `gather(shape(...))` reads back out.
            x_var = op.inputs.get("x")
            axes_var = op.inputs.get("axes")
            keep_dims_var = op.inputs.get("keep_dims")
            axes_val = static_value(axes_var)
            keep_dims_val = bool(static_value(keep_dims_var, False))
            if x_var is not None and x_var.shape is not None and axes_val is not None and len(axes_val) == 1 and not keep_dims_val:
                in_rank = len(x_var.shape)
                reduced_axis = int(axes_val[0])
                if reduced_axis < 0:
                    reduced_axis += in_rank
                in_axis = torch_axis
                if reduced_axis <= in_axis:
                    in_axis += 1
                if 0 <= in_axis < in_rank:
                    return self._infer_dynamic_dim_expr(x_var, in_axis, _seen)

        if op.op_type == "pad":
            # MIL `pad` only ever pads the LAST n_padded dims of `x` (see the "pad" translation branch
            # above, which also REJECTS anything but a constant `pad` array and anything padding other
            # than the single trailing/fastest-varying axis -- so by the time a real "pad" op reaches
            # here, its shape delta is always a known constant on exactly one axis). Every other axis is
            # an unchanged passthrough; the padded axis's real expression is the input's own plus the
            # constant `lp+rp`. Needed for Conformer-CTC's `rel_shift` (`F.pad(matrix_bd, pad=(1, 0))`
            # immediately before the `x.view(b, h, -1, qlen)` this file's "reshape" case already
            # resolves) -- without this, a `gather(shape(x_29), ...)` reading the PADDED tensor's own
            # shape back out gave up at "pad" and fell back to "n_tokens".
            x_var = op.inputs.get("x") or op.inputs.get("data")
            pad_var = op.inputs.get("pad")
            pad_val = static_value(pad_var)
            if x_var is not None and x_var.shape is not None and pad_val is not None:
                rank = len(x_var.shape)
                n_padded = len(pad_val) // 2
                padded_axis = rank - 1  # the only axis the "pad" translation branch ever allows non-zero
                if torch_axis != padded_axis or n_padded == 0:
                    if torch_axis < len(var.shape):
                        return self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                else:
                    lp, rp = int(pad_val[-2]), int(pad_val[-1])
                    in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                    if in_expr is not None:
                        return in_expr if lp == 0 and rp == 0 else in_expr + (lp + rp)

        if op.op_type == "slice_by_index":
            # A `pe[:, start:end]`-style dynamic slice: the real length on `torch_axis` is `end - begin`,
            # resolved independently for `begin` (only a literal, non-negative int or a mask is
            # understood -- MIL's own "ignore this value, use the full extent" convention) and `end`
            # (a mask, a literal, or a gather-derived concat element). Needed for several distinct real
            # cases: `self.pe[:, start:end]` (positional-encoding table crop, begin/end both dynamic but
            # begin resolves via mask), `rel_shift`'s `x[:, :, 1:]` (literal non-zero begin=1, end
            # masked), and `att_mask = fill[0:current_lengths, 0:current_lengths]` (literal begin=0, end
            # a genuinely data-dependent value -- see the "n_tokens" special-case below). Falls through to
            # the blind substitution if EITHER side can't be resolved this way, same as before this case
            # existed.
            x_var = op.inputs.get("x")
            begin_var = op.inputs.get("begin")
            end_var = op.inputs.get("end")
            begin_mask_var = op.inputs.get("begin_mask")
            end_mask_var = op.inputs.get("end_mask")
            begin_mask_val = static_value(begin_mask_var)
            end_mask_val = static_value(end_mask_var)
            is_begin_masked = begin_mask_val is not None and len(begin_mask_val) > torch_axis and bool(begin_mask_val[torch_axis])
            is_end_masked = end_mask_val is not None and len(end_mask_val) > torch_axis and bool(end_mask_val[torch_axis])

            # `begin` may resolve to either a plain non-negative literal (the common case) or a genuine
            # SymbolEnv expression string (e.g. RelPositionalEncoding's `start_pos = center_pos -
            # gather(shape(x), 1)`, resolved via `facts.scalar_expr`'s arithmetic-walk) -- both are
            # equally valid as the subtrahend in `end - begin` below, so both are kept, not just ints.
            if is_begin_masked:
                begin_expr = as_expr(0)
            else:
                resolved_begin = self.facts.slice_axis_value(begin_var, torch_axis)
                if isinstance(resolved_begin, int) and resolved_begin >= 0:
                    begin_expr = as_expr(resolved_begin)
                elif isinstance(resolved_begin, sympy.Basic):
                    begin_expr = resolved_begin
                else:
                    begin_expr = None

            end_expr = None
            end_is_guess = False
            if x_var is not None and x_var.shape is not None and torch_axis < len(x_var.shape):
                if is_end_masked:
                    end_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                else:
                    resolved_end = self.facts.slice_axis_value(end_var, torch_axis)
                    if resolved_end is not None:
                        end_expr = as_expr(resolved_end)
                    end_is_guess = self.facts.slice_axis_is_guess(end_var, torch_axis)
                if end_expr is None or end_is_guess:
                    # An `end` that resolved to nothing, or only to `facts.scalar_expr`'s producer-less
                    # fallback, is this walk bottoming out inside the `end` chain rather than a genuine
                    # answer -- confirmed on two distinct real cases: Conformer-CTC's
                    # `att_mask = fill[0:current_lengths, 0:current_lengths]` (`current_lengths` derived
                    # from the REAL "length" graph INPUT's own runtime VALUE via ADD/DIV/floor_div,
                    # architecturally impossible to resolve into a SymbolEnv shape expression at all --
                    # SymbolEnv only ever binds compile-time shape quantities like n_tokens, never a
                    # tensor's actual data, the same "value only exists after graph compute" limit
                    # `facts.gather_shape_value`'s docstring documents for RANGE_1D) and the
                    # positional-encoding table crop (`self.pe[:, start:end]`, whose `end` is a real
                    # ARITHMETIC EXPRESSION over a gather -- `center + t` -- that `facts.range_scalar`'s
                    # narrower "exact gather(shape(x), idx)" pattern match can't see through at all,
                    # returning None outright). But this whole exporter already assumes single-utterance,
                    # no-padding inference everywhere else (e.g. the always-1 batch axis) -- under that
                    # assumption BOTH values are ALWAYS numerically equal to `x`'s own real (allocated)
                    # extent, so trusting `x`'s own extent here is correct for every case this exporter
                    # targets, not just a guess.
                    #
                    # The test is `facts.slice_axis_is_guess`, i.e. the *provenance* of the answer, not
                    # its spelling. It used to be a comparison against the literal string "n_tokens",
                    # which worked only because a genuinely-derived length happened to come out spelled
                    # `floor((n_tokens + 0 - 1) / 1) + 1` instead. Normalizing expressions made those
                    # two identical and broke the accident in two models at once: SupertonicTTS's VFE
                    # (a const `text_attn.increments` table of declared shape (1, 1000, 1) cropped to
                    # `[:, :n_tokens]`, which then took the literal 1000) and VITS's relative-position
                    # reshape (whose `x` extent is `n_tokens + 1`, one too many). Both are real
                    # derivations off a `gather(shape(...))`, and both must be kept. See BACKEND.md.
                    x_full_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                    # Compared against THIS topology's own root axis, not the literal "n_tokens" --
                    # Conformer-CTC (the CMVN case this whole block's comment is about) now declares
                    # "n_samples" here (EXPORT-ROADMAP.md R1). A bare re-derivation of the root axis
                    # itself is exactly as uninformative as it always was; it just isn't spelled
                    # "n_tokens" for every model any more.
                    if x_full_expr is not None and x_full_expr != as_expr(self.root_axis):
                        end_expr = x_full_expr
            if end_expr is not None and begin_expr is not None:
                return end_expr if begin_expr == 0 else end_expr - begin_expr

        if op.op_type == "split":
            # `split(x, axis, num_splits/split_sizes)` divides `x` into N outputs along `axis` -- every
            # OTHER axis is a direct, unchanged passthrough to `x`'s own corresponding axis. Needed for
            # Conformer-CTC's combined Q/K/V (or similar) linear projection split before `rel_shift`'s
            # `matrix_bd`/`matrix_ac` computation queries its own shape back out -- only the passthrough
            # case is implemented (the split axis ITSELF, if queried, falls through to the blind
            # substitution below; nothing seen so far has needed it).
            x_var = op.inputs.get("x")
            axis_var = op.inputs.get("axis")
            axis_val = static_value(axis_var)
            if x_var is not None and x_var.shape is not None and axis_val is not None and torch_axis < len(x_var.shape):
                rank = len(x_var.shape)
                split_axis = int(axis_val) + rank if int(axis_val) < 0 else int(axis_val)
                if torch_axis != split_axis:
                    return self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)

        if op.op_type == "concat":
            # Any axis OTHER than the concat axis has a direct 1:1 correspondence to ANY ONE of the
            # operands (they must all agree there by construction); the concat axis itself is the SUM of
            # every operand's own real expression, not any single operand's. Needed for Kokoro's SineGen:
            # `torch.cat([rad0.unsqueeze(1), rad_values[:,1:,:]], dim=1)` rebuilds the SAME total length
            # (a 1-element slice + an (L-1)-element slice) by replacing just the first row -- without this
            # case the walk gave up at "concat" and fell back to a bare "n_tokens" substitution for what
            # is actually a 600x-derived length, corrupting SineGen's own downsample factor.
            values = op.inputs.get("values")
            axis_var = op.inputs.get("axis")
            axis_val = static_value(axis_var)
            if values is not None and axis_val is not None and len(values) > 0:
                first = values[0]
                if isinstance(first, Var) and first.shape is not None:
                    out_rank = len(var.shape)
                    cat_axis = int(axis_val) + out_rank if axis_val < 0 else int(axis_val)
                    if torch_axis != cat_axis:
                        return self._infer_dynamic_dim_expr(first, torch_axis, _seen)
                    parts = []
                    for operand in values:
                        if not (isinstance(operand, Var) and operand.shape is not None and torch_axis < len(operand.shape)):
                            return None
                        part_expr = self._infer_dynamic_dim_expr(operand, torch_axis, _seen)
                        if part_expr is None:
                            return None
                        parts.append(part_expr)
                    return sum(parts, as_expr(0))

        if op.op_type == "transpose":
            # `output.shape[i] = input.shape[perm[i]]` (MIL's own semantics, same as the "transpose"
            # translation branch above uses) -- a symbolic output axis's real expression is just its
            # corresponding INPUT axis under `perm`, recursed into directly (no formula of its own).
            # Needed for the mel-frontend's `x_19 = processed_signal.permute(1, 0, 2, 3)` immediately
            # before the subsampling reshape: without this, the walk gave up at `transpose` and fell back
            # to a bare "n_tokens" substitution for the STFT frame-count axis (which is NOT the raw
            # sample count `n_tokens` -- see the `conv`/`range_1d` cases' own formulas), producing an
            # invalid multi-"n_tokens" RESHAPE target downstream.
            x_var = op.inputs.get("x")
            perm_var = op.inputs.get("perm") or op.inputs.get("axes")
            perm_val = static_value(perm_var)
            if x_var is not None and perm_val is not None and x_var.shape is not None:
                rank = len(var.shape)
                norm_perm = [(int(p) + rank) if int(p) < 0 else int(p) for p in perm_val]
                if torch_axis < len(norm_perm):
                    in_axis = norm_perm[torch_axis]
                    if 0 <= in_axis < len(x_var.shape):
                        return self._infer_dynamic_dim_expr(x_var, in_axis, _seen)

        if op.op_type in ("gather", "gather_along_axis"):
            # `gather(x, indices, axis=a)` replaces x's axis `a` with the INDICES' own shape:
            # `out.shape = x.shape[:a] + indices.shape + x.shape[a+1:]`. So an output axis inside the
            # indices' block takes its expression from `indices`, and every other one passes through
            # from `x` at its shifted position.
            #
            # The case that needs this is the most ordinary one imaginable and had simply never been
            # hit: an **embedding lookup at the start of a topology**. `nn.Embedding(vocab, C)` traces
            # to `gather(weight, tokens, axis=0)`, whose output length IS the token count -- and with no
            # case here the walk gave up, so the dynamic axis fell out of a downstream RESHAPE target as
            # a literal, and the topology failed to build with "target shape [1,512,1] has 512 elements
            # but input has 3584". Found exporting Kokoro's `text_encoder_cnn`; the same shape as the
            # `leaky_relu`/`conv_transpose` gaps `vits_export.py`'s own docstring records.
            x_var = op.inputs.get("x")
            idx_var = op.inputs.get("indices")
            axis = static_scalar(op.inputs.get("axis"))
            if x_var is not None and idx_var is not None and x_var.shape is not None \
                    and idx_var.shape is not None:
                a = int(axis) if axis is not None else 0
                if a < 0:
                    a += len(x_var.shape)
                idx_rank = len(idx_var.shape)
                if a <= torch_axis < a + idx_rank:
                    return self._infer_dynamic_dim_expr(idx_var, torch_axis - a, _seen)
                src_axis = torch_axis if torch_axis < a else torch_axis - idx_rank + 1
                if 0 <= src_axis < len(x_var.shape):
                    return self._infer_dynamic_dim_expr(x_var, src_axis, _seen)
            return None

        if op.op_type == "stack":
            # Inserts one new axis (like expand_dims) but from N same-shaped operands rather than one --
            # any axis OTHER than the new one has a direct 1:1 correspondence to (any one of, they're
            # all identical there by construction) the stacked operands, just shifted. Needed for the
            # SAME STFT-magnitude chain: `stack([real, imag], axis=-1)` is the exporter's own composed
            # RESHAPE+CONCAT (see the "stack" op_type translation), but the ORIGINAL MIL op this walk
            # sees is still the real `stack`.
            values = op.inputs.get("values")
            axis_var = op.inputs.get("axis")
            axis_val = static_value(axis_var)
            if values is not None and axis_val is not None and len(values) > 0:
                first = values[0]
                if isinstance(first, Var) and first.shape is not None:
                    out_rank = len(var.shape)
                    stack_axis = int(axis_val) + out_rank if axis_val < 0 else int(axis_val)
                    if torch_axis == stack_axis:
                        return as_expr(len(values))
                    shift = 1 if stack_axis < torch_axis else 0
                    in_axis = torch_axis - shift
                    if 0 <= in_axis < len(first.shape):
                        return self._infer_dynamic_dim_expr(first, in_axis, _seen)

        _ELEMENTWISE_BROADCAST_OPS = {
            "less", "greater", "less_equal", "greater_equal", "equal", "not_equal",
            "add", "sub", "mul", "real_div", "floor_div", "mod", "logical_and", "logical_or",
            "maximum", "minimum", "pow",
        }
        if op.op_type in _ELEMENTWISE_BROADCAST_OPS or op.op_type == "select":
            # Elementwise binary ops (and `select(cond, a, b)`, a ternary op with the exact same
            # broadcast-and-preserve-axes semantics over its `a`/`b` operands) preserve per-axis
            # correspondence between operands and output (only ever broadcasting a size-1 operand up,
            # never reshuffling axes) -- the real expression for a symbolic output axis is whichever
            # operand ISN'T just a static/size-1 broadcast target there. Needed because get_var_info's
            # blind substitution has no way to tell "this symbol is the genuine dynamic axis, inherited
            # from one real operand" apart from "this symbol is an unrelated one CoreML happened to mint
            # for a different (e.g. always-1 batch) axis" -- confirmed on Conformer-CTC's length-validity
            # mask (`torch.arange(T) < length`, broadcast across an always-1 batch axis via an upstream
            # `tile`): both axes of the `less` op's own output are bare symbols, but only one of them is
            # genuinely `n_tokens`. The `select` case specifically was needed for the CMVN per-utterance
            # std-dev's `torch.where(mask, std, fallback)`, whose own output feeds an `expand_dims` whose
            # declared shape a downstream RESHAPE trusts verbatim (see the `expand_dims`/`squeeze` case
            # above) -- without this, the walk gave up at `select` and fell back to a bare "n_tokens"
            # substitution for what is actually always-1 batch axis, producing an invalid RESHAPE target.
            operand_keys = ("a", "b") if op.op_type == "select" else ("x", "y")
            # Both operands can independently report a DYNAMIC symbol at this axis under MIL's own type
            # inference (it doesn't always manage to prove one side is a literal 1, even when it truly
            # broadcasts from one) -- so picking whichever operand happens to be checked FIRST, as an
            # earlier version of this code did, is a real bug whenever the two operands DISAGREE on what
            # this axis truly is: confirmed on Conformer-CTC's `att_mask = pad_mask_for_att_mask *
            # att_mask_3`, where `pad_mask_for_att_mask` (a `tile`-broadcast) reports opaque dynamic
            # symbols on EVERY axis (including its own genuinely-1 batch axis), while `att_mask_3`'s
            # SAME axis is the real, non-degenerate quantity -- checking "x" first silently returned the
            # batch-1 operand's own (wrong) value instead of ever considering "y". Instead, resolve every
            # matching operand and prefer whichever ISN'T a plain literal "1" (a real broadcast target
            # never carries useful information at its own broadcast axis) -- falling back to the first
            # resolved value only if EVERY operand bottoms out at "1".
            first_resolved = None
            for operand_key in operand_keys:
                operand = op.inputs.get(operand_key)
                if operand is None or not isinstance(operand, Var) or operand.shape is None or torch_axis >= len(operand.shape):
                    continue
                if not has_dynamic_symbol(operand.shape[torch_axis]):
                    continue
                inferred = self._infer_dynamic_dim_expr(operand, torch_axis, _seen)
                if inferred is not None and inferred != 1:
                    return inferred
                if first_resolved is None:
                    first_resolved = inferred
            if first_resolved is not None:
                return first_resolved

        if torch_axis == 0 and len(var.shape) >= 2:
            # Bottomed out (no case above understood the full producer chain) on what -- by this whole
            # exporter's own stated architecture (every declared model input's own batch axis is a
            # literal 1, the same assumption `facts.gather_shape_value`'s own dedicated
            # torch_axis==0 shortcut already relies on) -- can ONLY be a batch axis. Needed for
            # Conformer-CTC's CMVN std-dev: `x_std`'s own axis 0 walk runs through a long, twisted
            # select/sub/pow/tile chain this file doesn't have (and doesn't need) a specific case for,
            # bottoming out here; blindly substituting "n_tokens" there was flatly wrong (confirmed: it
            # produced an element-count-changing RESHAPE target, not just an imprecise one) since axis 0
            # is never genuinely the sequence-length axis for any real input this exporter targets. A
            # model that genuinely needed a non-1 batch axis 0 would surface as a numerical mismatch
            # against the reference model here, not a syntax error.
            return as_expr(1)

        # Any other producer (pad/expand_dims/squeeze/etc.): not a transform this walk understands, but
        # also not necessarily wrong to keep walking from -- fall back to a bare symbol substitution
        # (identical to get_var_info's own long-standing default) rather than giving up outright. This
        # matters for e.g. a reflect-pad immediately upstream of a conv (confirmed on Conformer-CTC's
        # STFT: the pre-conv `expand_dims` var's own shape is the COMPOSITE expression "is1639 + 512",
        # not a bare symbol) -- substituting the base symbol there ("n_tokens + 512") and letting the
        # conv branch above apply its own formula on top gives the right answer without this walk needing
        # to specifically understand every intermediate op type between the true input and the first conv.
        return self._sub_symbol(dim)

    def _find_range_1d_var(self, var, depth=0, _seen=None):
        """
        Backward-walks `var`'s producer chain through shape-preserving passthrough ops, returning the
        `range_1d` Var itself iff `var` is (an alias/broadcast of) a bare `torch.arange(N)`-style index
        vector, else None. Shares its walk set with (and is the basis of) `_traces_to_range_1d` --
        deliberately a SMALL, conservative set of ops (never arithmetic), since anything beyond a pure
        alias/broadcast would mean this var is no longer just "the position index," which is the one
        thing this check needs to be sure of.
        """
        if _seen is None:
            _seen = set()
        if depth > 8 or var is None or not isinstance(var, Var) or id(var) in _seen:
            return None
        _seen.add(id(var))
        if var.op is None:
            return None
        if var.op.op_type == "range_1d":
            return var
        if var.op.op_type in ("cast", "reshape", "expand_dims", "squeeze", "identity", "tile"):
            for v in var.op.inputs.values():
                if isinstance(v, Var):
                    found = self._find_range_1d_var(v, depth + 1, _seen)
                    if found is not None:
                        return found
        return None

    def _traces_to_range_1d(self, var, depth=0, _seen=None):
        """
        Returns True iff `var` (an alias/broadcast of) a bare `torch.arange(N)`-style index vector --
        see `_find_range_1d_var` (which this wraps) for the walk itself. Used alongside
        `_traces_to_length_input` to recognize the `torch.arange(T) < length`-style length-validity
        comparison pattern (see the "less" op_type translation below).
        """
        return self._find_range_1d_var(var, depth, _seen) is not None

    def _traces_to_length_input(self, var, depth=0, _seen=None):
        """
        Backward-walks `var`'s producer chain through arithmetic/passthrough ops, returning True iff it
        reaches the graph's own declared "length" INPUT directly (a genuine function input with no
        producer, named "length"). Used alongside `_traces_to_range_1d` to recognize the
        `torch.arange(T) < length`-style length-validity comparison pattern -- see the "less" op_type
        translation below for why this pattern's result is always-true for this exporter's target use
        case (single-utterance, no real padding), regardless of whatever arithmetic chain computed the
        comparison's actual bound. Walks a broader op set than `_traces_to_range_1d` (real arithmetic is
        expected here -- NeMo's own `calc_length()`-style formula), checking every Var-typed input of a
        walkable op rather than specific named inputs, so it doesn't need to special-case `select`'s
        `cond`/`a`/`b` vs. a binary op's `x`/`y`.
        """
        if _seen is None:
            _seen = set()
        if depth > 16 or var is None or not isinstance(var, Var) or id(var) in _seen:
            return False
        _seen.add(id(var))
        if var.op is None:
            return var.name == "length"
        _WALKABLE = {
            "cast", "reshape", "expand_dims", "squeeze", "identity", "select",
            "add", "sub", "mul", "real_div", "floor_div", "equal", "not_equal",
            "less", "greater", "less_equal", "greater_equal",
        }
        if var.op.op_type not in _WALKABLE:
            return False
        for v in var.op.inputs.values():
            if isinstance(v, Var) and self._traces_to_length_input(v, depth + 1, _seen):
                return True
        return False

    def get_var_info(self, var):
        """
        Extracts dtype ('f32' or 'i32') and shape in fastest-varying (ne-order) reversed list.
        """
        name = self.safe_name(var.name)
        dtype_str = str(var.dtype)
        if "fp" in dtype_str or "float" in dtype_str or "double" in dtype_str:
            dtype = "f32"
        elif "int" in dtype_str:
            dtype = "i32"
        else:
            dtype = "f32"

        shape = []
        if hasattr(var, "shape") and var.shape is not None:
            for dim in var.shape:
                # All shape entries in Loom topologies must be serialized as string expressions for C++
                # parsing. Every symbolic dim (CoreML stringifies these as "isN") collapses to the single
                # "n_tokens" symbol GraphBuilder/SymbolEnv resolve at build time -- the engine's
                # dynamic-shape support is genuinely single-axis (EXPORT-BACKLOG.md item 3), so this
                # exporter only ever targets models with exactly one true dynamic quantity (sequence
                # length), matching every model on the current roadmap (batch/hidden/heads are always
                # architecturally static). A topology routinely contains SEVERAL distinct "isN" names for
                # that one same quantity, not just one: CoreML's shape algebra mints a fresh opaque symbol
                # at any derivation step it can't simplify back to the original input symbol (confirmed
                # empirically -- an LFM2 ShortConv layer's causal pad+conv+slice, which provably preserves
                # sequence length, produces 4 distinct "isN" names downstream of its one "is0" input; the
                # modular-export path's own inter-slice boundaries surface even more of these as separate
                # declared inputs of a single slice). There is no cheap, reliable way to tell "several
                # names, one true quantity" apart from "two genuinely independent dynamic axes" from the
                # dim strings alone -- CoreML doesn't expose symbol-equality at this level, and an
                # input-count-based heuristic tried here produced false positives on real, correct
                # modular slices. If a future model genuinely needs a second independent dynamic axis, that would
                # surface as a numerical mismatch against the reference model, not a syntactic error here.
                #
                # Substituting each symbol OCCURRENCE (not the whole dim string) matters: a dim isn't
                # always bare "is936" -- coremltools' own shape inference can report a full ARITHMETIC
                # EXPRESSION over several such symbols, e.g. a reshape's "-1"-inferred axis surfacing as
                # "is936*is937*is938*is939*is940/1024" (confirmed on LFM2's GQA key/value reshape, which
                # crashed ggml_reshape_4d's element-count assertion once this was collapsed to the bare
                # string "n_tokens", discarding the "/1024" entirely). SymbolEnv's own expression evaluator
                # (src/core/symbol_env.cpp) supports `+ - * / ()`, so substituting every symbol occurrence
                # with "n_tokens" and keeping the surrounding arithmetic intact lets GraphBuilder evaluate
                # the whole expression correctly at build time instead.
                if has_dynamic_symbol(dim):
                    # Always try the real conv/cast/pad/concat/upsample-aware derivation first (see
                    # _infer_dynamic_dim_expr's own docstring for why blind substitution isn't always
                    # correct) -- this used to be gated behind "only for a BARE lone symbol, or an
                    # explicit per-op-type allowlist" on the theory that an already-composite MIL shape
                    # expression (e.g. "is936*is937.../1024") was safer left to plain substitution. That
                    # theory doesn't hold: `_infer_dynamic_dim_expr` ITSELF falls back to the exact same
                    # `self._sub_symbol(str(dim))` substitution for any op type (or any axis) it doesn't
                    # specifically understand -- so calling it unconditionally can never produce a WORSE
                    # answer than the gated version did, only occasionally a better one. The gate was
                    # real, growing technical debt in practice: every one of "pad"/"concat"/
                    # "upsample_nearest_neighbor"/"upsample_bilinear" needed adding to it by hand after
                    # each one's OWN correct `_infer_dynamic_dim_expr` case was silently unreachable from
                    # here (confirmed on Kokoro's STFT reflect-pad: `_infer_dynamic_dim_expr`'s "pad" case
                    # already computed the right answer, but this function's gate never called it, since
                    # MIL's own composite "isN + 20" string for a pad output isn't a bare fullmatch and
                    # "pad" wasn't yet on the allowlist -- blind-substituting just the embedded symbol gave
                    # "n_tokens + 20" instead of the real "600*n_tokens + 20").
                    torch_axis = list(var.shape).index(dim)
                    inferred = self._infer_dynamic_dim_expr(var, torch_axis)
                    shape.append(render(inferred if inferred is not None else self._sub_symbol(dim)))
                else:
                    shape.append(str(dim))
        
        # Loom expects fast-varying dimension first, so reverse standard shape order
        reversed_shape = list(reversed(shape))
        return {"name": name, "dtype": dtype, "shape": reversed_shape}

    def _ensure_mil_passes_applied(self):
        """Idempotently runs `apply_loom_mil_passes` (EXPORT-IMPROVEMENT-BACKLOG.md item 3) plus
        `annotate_dynamic_shapes` (EXPORT-ROADMAP.md R2b) over `self.program`, exactly once.

        `generate_graph_topology` calls this itself, at the top -- so it no longer matters whether a
        caller reaches it through `export()`'s own monolithic/modular dispatch, or (like every small
        TTS model's export script: Kokoro/VITS/StyleTTS2/Supertonic/Matcha's own `_build_topology`
        helpers) constructs a `LoomGGUFExporter` directly and calls `generate_graph_topology` without
        ever calling `export()` at all. Before this method existed, only the first group actually got
        the passes: `insert_explicit_broadcasts` (R2a) silently never ran for the second, since nothing
        there ever called `apply_loom_mil_passes` -- confirmed the hard way, via a snapshot diff against
        a pre-R2a baseline showing a genuinely dropped mutual-broadcast REPEAT in Matcha's encoder_logw
        mask computation. Memoized so a caller building several topologies off the same exporter/program
        only pays for the walk once.

        Skipped for the bespoke/advanced workflow (`export()`'s own comment on why): that path exists
        specifically to accept hand-built Programs (see test_compiler.py's MockOperation) that were
        never traced through `ct.convert()`'s standard pipeline and may contain synthetic ops standing
        in for ones MIL itself doesn't have -- a real MIL pass (`dead_code_elimination` in particular,
        which insists on internally-consistent var/op child-tracking) isn't meaningful there and isn't
        safe to run over a graph assembled by directly splicing Python op lists rather than through
        MIL's own block-mutation API. The same "is this bespoke" test `export()` used to run once at its
        own top now lives here, so every caller -- not just `export()` -- gets the same skip.
        """
        if self._mil_passes_applied or self.program is None:
            return
        is_bespoke = len(self.program.functions) > 1 and "main" in self.program.functions
        if not is_bespoke:
            # `fuse_attention` is opt-in per export (KV-CACHE.md decision 4): the SDPA pattern is
            # generic, so running it unconditionally would give the non-autoregressive TTS families an
            # ATTENTION node -- and a KV cache -- they must never have. `fuse_conv` is opt-in for the
            # identical reason, one op family over (BACKLOG.md P4.0.10): `conv_state` also defaults to
            # true, so a non-autoregressive model matching the causal-conv pattern would acquire
            # persistent state it must never have.
            apply_loom_mil_passes(self.program,
                                   fuse_attention=bool(self.kwargs.get("fuse_attention")),
                                   fuse_conv=bool(self.kwargs.get("fuse_conv")))
            self.facts.annotate_dynamic_shapes(self.program)
        self._mil_passes_applied = True

    def export(self):
        """
        Traverses the MIL program:
          - 'main' function becomes the embedded Lua driver script.
          - Other functions represent heavy submodules and become static topologies.
          - Weights and assets are serialized to GGUF.
        """
        is_bespoke = len(self.program.functions) > 1 and "main" in self.program.functions
        self._ensure_mil_passes_applied()

        if is_bespoke:
            # 1. Advanced / Bespoke Exporting Workflow
            print("Exporting via Advanced/Bespoke workflow...")
            for func_name, func in self.program.functions.items():
                if func_name == "main":
                    # MIL's own function name on the left, the emitted Lua entry point's on the right
                    # -- they were the same string until KV-CACHE.md's N.1 and are unrelated concepts.
                    self.transpile_to_lua(func, name="infer")
                else:
                    self.topologies[func_name] = self.generate_graph_topology(func, func_name)
            driver_script = self._finalize_driver()
        elif self.kwargs.get("modular_layout") is not None:
            # Modular-export blueprint (EXPORT-IMPROVEMENT-BACKLOG.md item 2): `self.program` here is
            # NOT a single flattened trace -- it has no "main" function at all, just one Function per
            # independently-traced submodule (prefix/aux/layer_i/suffix_i) -- so there is no monolithic
            # fallback to degrade to on failure; a bug here should fail loudly, not silently produce a
            # working-looking but wrong export.
            self.apply_modular_export()
            driver_script = self._finalize_driver()
        else:
            self.apply_monolithic_export()
            driver_script = self._finalize_driver()

        # 3. Serialization Phase
        self.write_gguf(driver_script)
        return self.output_path

    def _finalize_driver(self) -> str:
        """Runs the driver IR's own two checks over whichever path built the script, then codegens it.

        Every path lands here. The two synthesized ones now build their script through a
        `DriverBuilder`, which has already run both of these -- they are idempotent, and keeping them
        here is what holds the bespoke transpile path (which has no builder, because it lowers a MIL
        `main` function op by op rather than assembling components) to exactly the same two checks."""
        validate(self.driver_script.entry)
        check_subgraph_calls(self.driver_script.entry, self.topologies)
        return self.driver_script.render()

    def _driver_context(self, topologies=None) -> DriverContext:
        """What a `DriverComponent` emits against. Every topology this exporter produces shares one
        root axis (`self.root_axis`) -- unlike a multi-phase export, where each phase declares its
        own -- so the map is built by filling that in for each."""
        names = self.topologies if topologies is None else topologies
        return DriverContext(
            topologies=self.topologies,
            axes={name: self.root_axis for name in names},
            weights=None,
        )

    def apply_monolithic_export(self):
        print("Exporting via Automatic Monolithic path...")
        main_func = self.program.functions["main"]
        self.topologies["main_topology"] = self.generate_graph_topology(main_func, "main_topology")

        first_input = "tokens"
        feature_scale = 1
        if main_func.inputs:
            first_input_var = list(main_func.inputs.values())[0]
            first_input = self.safe_name(list(main_func.inputs.keys())[0])
            if hasattr(first_input_var, "shape") and len(first_input_var.shape) == 3:
                # For 3D shapes [batch, seq, feature], scale tokens by the last dimension (feature size)
                try:
                    feature_scale = int(first_input_var.shape[2])
                except (ValueError, TypeError):
                    pass

        n_tokens_expr = Len(first_input)
        if feature_scale > 1:
            n_tokens_expr = BinOp("floordiv", Len(first_input), Lit(feature_scale))

        # The traced function's own declared-input order IS the emission order here: the host-computed
        # bindings read `n_tokens_expr`, which reads the first input, so anything that reordered this
        # would produce a driver reading a symbol before it is bound -- which `driver_ir.validate`
        # catches, but only after the fact.
        bindings = tuple(
            (self.safe_name(name), _binding_kind(name)) for name in main_func.inputs.keys()
        )
        # The synthesized windowed masks have no MIL var, so they are not in `main_func.inputs` -- they
        # exist only on the emitted topology (`_route_windowed_masks`). Appended rather than merged in
        # traced order because they are host-computed like every other MASK binding and read only
        # `n_tokens`/`n_past`, which the traced inputs above have already bound.
        bindings = bindings + tuple(
            (name, MASK) for name in sorted(self.mask_windows) if name not in dict(bindings)
        )
        input_names = tuple(name for name, _ in bindings)

        # Through `SYNTHESIZED_BUILDERS` rather than by naming the class, so the table P4.0.7's
        # catalogue attributes components to models with is the one the exporter really builds from.
        # A second entry, `infer_with_past`, only for a topology whose ATTENTION nodes carry a cache
        # (KV-CACHE.md 3.3). Derived from the emitted graph rather than from `fuse_attention`, for the
        # same reason `GraphTopology::uses_kv_cache()` is derived on the engine side (decision 5): the
        # request to fuse and the presence of a fused node are two different facts, and a block the
        # pattern declined to match would otherwise get a decode loop with nothing to decode against.
        # A frame-wise classifier reduces its whole output rather than one row of it, so it gets its own
        # builder rather than a mode of the causal-LM one (BACKLOG.md P4.0.17). NAMED by the family
        # through `backend_kwargs()` rather than inferred here, because this is the one case where the
        # decomposition does not determine the orchestration -- Conformer-CTC is a `Flattened` export
        # like Qwen3 and shares none of its host-side shape. Naming it is also what keeps P4.0.7's
        # catalogue honest: `component_registry.usage()` reads the same declaration, so it cannot
        # attribute `argmax_epilogue` to a model that does not use it.
        if self.kwargs.get("driver_builder") == "CtcGreedy":
            self.apply_ctc_greedy_export(bindings, input_names, n_tokens_expr)
            return

        cached = self._topology_uses_kv_cache(self.topologies["main_topology"])
        decode = None
        if cached:
            blockers = self._non_cached_sequence_state(self.topologies["main_topology"])
            if blockers:
                print(f"  no infer_with_past: {', '.join(blockers)} mixes across the token axis with "
                      f"state the KV cache does not hold, so a decode step at n_tokens=1 cannot see "
                      f"its own history. Exporting infer (prefill) only.")
            else:
                decode = PrefillDecodeLoop(topology="main_topology", bindings=bindings,
                                           inputs=input_names, mask_windows=self.mask_windows)
        # Retain and reduce by name for KV-cached topologies -- the causal LMs, whose vocab is what
        # makes the Lua marshalling cap reachable at all (BACKLOG.md P4.0.14). Every other family keeps
        # returning its tensor, so no ASR/TTS driver text moves. One `cached` for both halves: which
        # component binds the logits and which one reads them is a single decision, and splitting it
        # across two independently-computed conditions is how the two ends of an edge drift apart.
        self.driver_script = SYNTHESIZED_BUILDERS["Flattened"](
            inputs=DriverInputs(bindings=bindings, n_tokens=n_tokens_expr,
                                 mask_windows=self.mask_windows),
            call=MonolithicCall(topology="main_topology", inputs=input_names, n_tokens=n_tokens_expr,
                                 retained=cached),
            epilogue=ArgmaxEpilogue(out_var="_mono_out", shape_var="_mono_shape",
                                    n_tokens=n_tokens_expr,
                                    retained_module="main_topology" if cached else None),
            decode=decode,
        ).build(self._driver_context())

    def apply_ctc_greedy_export(self, bindings, input_names, n_tokens_expr):
        """The NeMo CTC driver: one forward pass over the waveform, then greedy decode (P4.0.17).

        Takes the pieces `apply_monolithic_export` has already computed rather than recomputing them --
        the declared-input bindings and the root-axis expression are properties of the traced graph, not
        of what the host does with its output, and the two paths must not be able to disagree about
        them.
        """
        if self.kwargs.get("ctc_blank_id") is None:
            raise ValueError(
                "driver_builder='CtcGreedy' was requested without `ctc_blank_id`. The blank is the "
                "head's last class and only the checkpoint knows how many there are, so the family "
                "must supply it (ASRNemoEncoderExportConfig reads it during build_trace)."
            )
        blank_id = int(self.kwargs["ctc_blank_id"])
        self.driver_script = SYNTHESIZED_BUILDERS["CtcGreedy"](
            inputs=DriverInputs(bindings=bindings, n_tokens=n_tokens_expr),
            # Retained for the same reason a large-vocab LM retains: the reduction is engine-side, so
            # the [n_classes, n_frames] logits never become a Lua table.
            call=MonolithicCall(topology="main_topology", inputs=input_names, n_tokens=n_tokens_expr,
                                 retained=True),
            epilogue=CtcGreedyEpilogue(retained_module="main_topology", blank_id=blank_id),
        ).build(self._driver_context())

    # Ops that mix along the TOKEN axis and carry their own cross-step state, which the KV cache does
    # not hold: it stores K/V per attention block and nothing else. A topology containing one of these
    # cannot be stepped a token at a time, because the op would be handed a length-1 window with no
    # history -- semantically wrong even where the shapes happen to work out.
    #
    # Found by running it: LFM2-350M is a hybrid, 6 attention blocks and 10 ShortConv ones, and its
    # `infer_with_past` failed inside the first conv layer's own slice ("VIEW: resolved shape
    # [1,1024,1,] ... needs 16380 bytes but parent has 12288"). The shape error is the symptom; the
    # cause is that a causal depthwise convolution is stateful across steps and has no cache.
    #
    # An allow-list by exclusion rather than by enumeration, because the safe set is the open one:
    # everything else in a decoder block (MUL_MAT, ADD, the norms, ROPE, SILU, ...) is position-wise or
    # head-wise and computes the same thing for token t whether or not tokens 0..t-1 are present.
    # This set SHRINKS -- one entry per op that gains a state slot. `SHORT_CONV` is deliberately absent
    # rather than listed-and-excepted: it mixes along the token axis AND carries its own history, which
    # is exactly the property that makes an op safe here, so the rule reads the same for it as for
    # MUL_MAT. `CONV_1D_DW` stays, because an UNFUSED causal conv is still stateless -- a topology gets
    # a decode loop only if the fusion actually replaced its convs (BACKLOG.md P4.0.10).
    _NON_CACHED_SEQUENCE_STATE_OPS = frozenset({
        "CONV_1D", "CONV_1D_DW", "CONV_2D", "CONV_2D_DW", "CONV_TRANSPOSE_1D", "CONV_TRANSPOSE_2D",
        "CONV_FLOW_REVERSE", "SSM_CONV", "SSM_SCAN", "RWKV_WKV6", "RWKV_WKV7",
    })

    @classmethod
    def _non_cached_sequence_state(cls, topo: dict) -> list:
        """The op types in `topo` that make a token-at-a-time decode step invalid, sorted."""
        return sorted({
            node["op"] for node in topo.get("nodes", [])
            if node["op"] in cls._NON_CACHED_SEQUENCE_STATE_OPS
        })

    @staticmethod
    def _topology_uses_kv_cache(topo: dict) -> bool:
        """Does this topology contain an `ATTENTION` node with a cache -- the engine-side
        `GraphTopology::uses_kv_cache()` question, asked on the exporter's own dict.

        `kv_cache` defaults to TRUE in `op_attention`, so reading a missing attr as false would report
        exactly the models that need a cache as not needing one (the same trap KV-CACHE.md 1.2 records
        on the C++ side)."""
        return any(
            node["op"] == "ATTENTION" and node.get("attrs", {}).get("kv_cache", True)
            for node in topo.get("nodes", [])
        )

    def apply_modular_export(self):
        """
        Synthesizes the driver for a modular-export blueprint (`kwargs["modular_layout"]`, a
        `ModularExportResult` from modular_export.py): one real, independently-traced Function per
        prefix/aux/layer_i/suffix_i submodule. Every function is self-contained by construction (no
        cross-slice variable leakage to detect, unlike partitioning a single flattened trace by scope
        would), so this only has to generate each function's topology directly
        (`generate_graph_topology(func, name)`, no ops_list/inputs_dict reconstruction) and chain
        SubgraphCalls prefix -> [aux] -> layer_0..N-1 -> suffix_0..M-1 -> argmax.
        """
        print("Exporting via Modular-Blueprint path...")
        layout = self.kwargs["modular_layout"]
        functions = self.program.functions
        special_names = set(HOST_COMPUTED_INPUT_NAMES)

        def is_aux_input(name):
            return layout.aux_kwarg and (name == layout.aux_kwarg or name.startswith(layout.aux_kwarg + "_"))

        has_aux = layout.aux_output_names is not None
        stage_names = ["prefix"] + (["aux"] if has_aux else []) \
            + [f"layer_{i}" for i in range(layout.num_layers)] + layout.suffix_names

        # 1. Generate topologies for every submodule function. generate_graph_topology drops any
        # declared input no node actually reads (post dead-node-pruning) -- e.g. LFM2's conv-type
        # layers never touch `position_embeddings`, only its attention-type layers do, so which inputs
        # survive can differ PER LAYER even though they were traced with an identical call signature.
        # Wiring below must therefore consult each stage's own post-filter declared inputs, never a
        # single shared name list.
        for name in stage_names:
            self.topologies[name] = self.generate_graph_topology(functions[name], name)

        def declared_inputs(name):
            return [inp["name"] for inp in self.topologies[name]["inputs"]]

        prefix_input_names = declared_inputs("prefix")
        chain_in_names = [n for n in prefix_input_names if n not in special_names]
        if len(chain_in_names) != 1:
            raise ValueError(f"prefix submodule must declare exactly one non-special input, got {prefix_input_names}")
        first_input = self.safe_name(chain_in_names[0])
        n_tokens_expr = Len(first_input)

        # 2. `first_input` (the caller-supplied token-ids input) must be bound before anything below
        # reads n_tokens_expr, mirroring apply_monolithic_export's own ordering. The host-computed
        # names follow it, sorted -- unlike the monolithic path there is no single traced function
        # whose declared-input order could be used, since each stage declares its own subset.
        special_needed = set()
        for name in stage_names:
            special_needed.update(n for n in declared_inputs(name) if n in special_names)
        bindings = [(first_input, CALLER)] + [
            (self.safe_name(name), _binding_kind(name)) for name in sorted(special_needed)
        ]

        # 3. Prefix.
        #
        # From here on every stage RETAINS its output (BACKLOG.md P4.0.12): a chain edge is a
        # `[n_embd, n_tokens]` hidden state the driver only threads onward, and marshalling it made two
        # copies of a value nobody looks at -- a device->host->device round trip per edge per step once
        # a second backend exists. Each stage's consumer therefore names the producing MODULE
        # (`OutputRef`) instead of a Lua local, and `driver_ir.check_subgraph_calls` is what checks that
        # the two agree, since a module name is invisible to `validate`.
        stages = [ChainStage(
            topology="prefix", outputs=(), retained=True,
            inputs={self.safe_name(n): IRVar(self.safe_name(n)) for n in prefix_input_names},
        )]
        chain_src = OutputRef("prefix")

        # 4. Auxiliary submodule (computed once, shared across every repeated-block call below) -- e.g.
        # LFM2's rotary-embedding table, computed once in Lfm2Model.forward and threaded into every
        # decoder layer as `position_embeddings=(cos, sin)`.
        aux_refs = None
        if has_aux:
            aux_input_names = declared_inputs("aux")
            aux_chain_names = [n for n in aux_input_names if n not in special_names]
            if len(aux_chain_names) > 1:
                raise ValueError(f"aux submodule must declare at most one non-special input, got {aux_input_names}")
            aux_inputs_tbl = {}
            for n in aux_input_names:
                safe_n = self.safe_name(n)
                # The aux submodule's own non-special input (if it has one -- it may not, e.g. LFM2's
                # pos_emb is called with the real hidden_states value purely for its dtype/device, which
                # the traced graph never ends up actually depending on) plays the same "current chain
                # tensor" role a repeated-block call's primary input does -- feed it the same value.
                aux_inputs_tbl[safe_n] = chain_src if n in aux_chain_names else IRVar(safe_n)
            # The aux submodule's i-th declared OUTPUT, addressed positionally exactly as before -- the
            # index is now the store's 1-based one rather than a local's position in a capture list.
            aux_refs = [OutputRef("aux", index=i + 1) for i in range(len(layout.aux_output_names))]
            stages.append(ChainStage(
                topology="aux", outputs=(), retained=True, inputs=aux_inputs_tbl,
            ))

        # 5. Repeated block, threading `chain_var` (hidden_states) from one layer's output into the
        # next's input. Each layer's OWN declared inputs are consulted independently (see the comment on
        # step 1).
        for i in range(layout.num_layers):
            layer_name = f"layer_{i}"
            layer_inputs = declared_inputs(layer_name)
            chain_names_i = [n for n in layer_inputs if n not in special_names and not is_aux_input(n)]
            if len(chain_names_i) != 1:
                raise ValueError(f"repeated block must declare exactly one non-special, non-aux input, got {layer_inputs}")
            chain_name_i = chain_names_i[0]

            inputs_tbl = {}
            for n in layer_inputs:
                safe_n = self.safe_name(n)
                if n == chain_name_i:
                    inputs_tbl[safe_n] = chain_src
                elif is_aux_input(n):
                    idx = 0 if n == layout.aux_kwarg else int(n[len(layout.aux_kwarg) + 1:])
                    inputs_tbl[safe_n] = aux_refs[idx]
                else:
                    inputs_tbl[safe_n] = IRVar(safe_n)

            stages.append(ChainStage(
                topology=layer_name, outputs=(), retained=True, inputs=inputs_tbl,
            ))
            chain_src = OutputRef(layer_name)

        # 6. Suffix chain (e.g. final norm + lm_head).
        for name in layout.suffix_names:
            in_names = declared_inputs(name)
            if len(in_names) != 1:
                raise ValueError(f"suffix submodule '{name}' must declare exactly one input, got {in_names}")
            stages.append(ChainStage(
                topology=name, outputs=(), retained=True,
                inputs={self.safe_name(in_names[0]): chain_src},
            ))
            chain_src = OutputRef(name)

        # 7. The last stage is no exception (BACKLOG.md P4.0.14). Its output is not an intermediate --
        # it is the logits the epilogue argmaxes, a genuinely host-side control decision -- but the
        # *decision* is one integer, and marshalling a [n_vocab, n_tokens] table to compute it is what
        # capped this path at ~2048 prompt tokens on a 65536-wide vocab. The epilogue reduces the
        # retained output by module name instead, so the only value this chain ever moves across the
        # boundary is the token id itself.
        #
        # `check_subgraph_calls` is what holds the two halves together: the epilogue names the module
        # the last stage retained, and a mismatch is an export-time error rather than a read of
        # something that was never stored.
        final_module = stages[-1].topology

        # 8. Same argmax epilogue the monolithic path uses -- two of this builder's three components
        # are shared with it, which is the smallest real instance of P4.0.7's reuse claim, and since
        # P4.0.14 they are shared in the same MODE as well: both paths retain and reduce by name.
        self.driver_script = SYNTHESIZED_BUILDERS["Modular"](
            inputs=DriverInputs(bindings=tuple(bindings), n_tokens=n_tokens_expr),
            chain=ModularChain(stages=tuple(stages), n_tokens=n_tokens_expr),
            epilogue=ArgmaxEpilogue(n_tokens=n_tokens_expr, retained_module=final_module),
        ).build(self._driver_context())

    def transpile_to_lua(self, func: Function, name="infer"):
        """
        Transpiles the main MIL orchestration function to a Lua JIT driver script. `name` is the
        emitted Lua entry point (`DriverBuilder.entry_name`'s equivalent for the path that has no
        builder), not the MIL function's own name.
        """
        # Track the first input variable name to dynamically derive n_tokens
        self.first_input = "tokens"
        if func.inputs:
            self.first_input = self.safe_name(list(func.inputs.keys())[0])

        body = []
        # Unpack incoming inputs
        for inp_name in func.inputs.keys():
            safe_inp = self.safe_name(inp_name)
            body.append(Local(safe_inp, FieldAccess("inputs", safe_inp)))

        # Transpile operations inside function block
        body.extend(self.transpile_block(func))

        # Unpack and return outputs
        output_names = [self.safe_name(v.name) for v in func.outputs]
        body.append(Return([IRVar(n) for n in output_names]))

        # No builder: this path lowers a hand-built MIL `main` function op by op rather than assembling
        # components, which is what `Decomposition.driver_builder` returning None records. It still
        # produces the same artifact, so `_finalize_driver` holds it to the same two checks.
        self.driver_script = DriverScript(prelude=[], entry=IRFunction(name, ["inputs"], body))

    def transpile_block(self, block: Block) -> list:
        stmts = []
        for op in block.operations:
            stmts.extend(self.transpile_operation(op))
        return stmts

    def transpile_operation(self, op: Operation) -> list:
        op_type = op.op_type
        output_names = [self.safe_name(v.name) for v in op.outputs]

        # A. Constant Serialization
        if op_type == "const":
            val = op.val.val
            if isinstance(val, np.ndarray) and val.size > 100:
                weight_name = self.safe_name(op.outputs[0].name)
                self.weights[weight_name] = val
                return [RawBlock([f"-- Weight {weight_name} packaged in GGUF"])]
            return [Local(output_names[0], RawExpr(self.format_lua_val(val)))]

        # B. Autoregressive / Loop Control Flow
        if op_type == "while_loop":
            cond_block = op.blocks[0]
            body_block = op.blocks[1]

            cond_stmts = self.transpile_block(cond_block)
            cond_var = self.safe_name(cond_block.outputs[0].name)
            body_stmts = self.transpile_block(body_block)
            loop_body = cond_stmts + [If(UnaryOp("not", IRVar(cond_var)), [Break()])] + body_stmts
            return [While(Lit(True), loop_body)]

        # C. Conditional Branches
        if op_type == "cond":
            true_block = op.blocks[0]
            false_block = op.blocks[1]
            pred_var = self.safe_name(op.inputs["pred"].name)

            true_stmts = self.transpile_block(true_block)
            false_stmts = self.transpile_block(false_block)

            # `cond`'s own output(s) are each block's own returned var (true_block.outputs[i] /
            # false_block.outputs[i] map positionally to op.outputs[i]) -- neither branch's own `local`
            # declarations survive past the if/else in Lua (block-scoped), so the result name is declared
            # OUTSIDE the if/else via LocalDecl and plain-assigned (Assign, no `local`) from inside each
            # arm. Previously this case didn't bind op.outputs at all, silently leaving any use of the
            # cond's result reading an undeclared global (nil) -- caught by validate() once the IR
            # rewrite made "read before defined" a mechanical, export-time check instead of a runtime bug.
            decls = []
            for i, out_var in enumerate(op.outputs):
                out_name = self.safe_name(out_var.name)
                decls.append(LocalDecl(out_name))
                true_stmts = true_stmts + [Assign(out_name, IRVar(self.safe_name(true_block.outputs[i].name)))]
                false_stmts = false_stmts + [Assign(out_name, IRVar(self.safe_name(false_block.outputs[i].name)))]

            return decls + [If(IRVar(pred_var), true_stmts, false_stmts)]

        # D. Submodule Dispatch
        if op_type in self.program.functions:
            inputs_tbl = {k: IRVar(self.safe_name(v.name)) for k, v in op.inputs.items() if hasattr(v, "name")}
            n_tokens_expr = Len(self.first_input)
            n_past_expr = Lit(0)
            if "n_tokens" in op.inputs:
                n_tokens_expr = IRVar(self.safe_name(op.inputs["n_tokens"].name))
            if "n_past" in op.inputs:
                n_past_expr = IRVar(self.safe_name(op.inputs["n_past"].name))
            return [SubgraphCall(outputs=output_names, module=op_type,
                                  axes={self.root_axis: n_tokens_expr, "n_past": n_past_expr},
                                  inputs=inputs_tbl)]

        # E. Fast Host Math Mapping
        if op_type == "argmax":
            x_name = self.safe_name(op.inputs["x"].name)
            # Retrieve shape to get vocabulary size (ne0 dimension)
            x_info = self.get_var_info(op.inputs["x"])
            n_vocab = int(x_info["shape"][0])
            # Row index is the last row (seq_len - 1), which is #first_input - 1
            row_expr = BinOp("-", Len(self.first_input), Lit(1))
            return [Argmax(output_names[0], x_name, Lit(n_vocab), row_expr)]

        if op_type == "range":
            start = self.safe_name(op.inputs["start"].name)
            end = self.safe_name(op.inputs["end"].name)
            return [Local(output_names[0], Call("loom.range", [IRVar(start), IRVar(end)]))]

        if op_type == "causal_mask":
            n_tokens = self.safe_name(op.inputs["n_tokens"].name)
            n_past = self.safe_name(op.inputs["n_past"].name)
            return [Local(output_names[0], Call("loom.causal_mask", [IRVar(n_tokens), IRVar(n_past)]))]

        if op_type in ("random_normal", "random_uniform"):
            # Unlike generate_graph_topology's own static-topology walk (which can NEVER satisfy a
            # random op -- see that function's own comment), this driver-level walk CAN: `main`'s body
            # is genuine Lua, already calling host functions mid-script (range/causal_mask above), so a
            # random op here maps directly onto the existing loom.gaussian_array/loom.uniform_array host
            # RNG functions (src/core/lua_bridge.cpp) real hand-written drivers already use.
            shape_var = op.inputs.get("shape")
            if static_value(shape_var) is None:
                raise NotImplementedError(
                    f"{op_type} op '{op.name}' has a non-constant 'shape' input, which this exporter "
                    "doesn't support (the sample count passed to loom.gaussian_array/uniform_array must "
                    "be a compile-time constant)."
                )
            n = 1
            for d in shape_var.val:
                n *= int(d)

            if op_type == "random_normal":
                mean_var, stddev_var = op.inputs.get("mean"), op.inputs.get("stddev")
                mean = float(static_scalar(mean_var, 0.0))
                stddev = float(static_scalar(stddev_var, 1.0))
                if mean != 0.0 or stddev != 1.0:
                    raise NotImplementedError(
                        f"random_normal op '{op.name}' has mean={mean}/stddev={stddev} -- this exporter "
                        "only supports the standard N(0,1) case loom.gaussian_array itself draws "
                        "(compose an explicit MUL/ADD after it for any other mean/stddev)."
                    )
                return [Local(output_names[0], Call("loom.gaussian_array", [Lit(n)]))]

            low_var, high_var = op.inputs.get("low"), op.inputs.get("high")
            low = float(static_scalar(low_var, 0.0))
            high = float(static_scalar(high_var, 1.0))
            if low != 0.0 or high != 1.0:
                raise NotImplementedError(
                    f"random_uniform op '{op.name}' has low={low}/high={high} -- this exporter only "
                    "supports the standard U(0,1) case loom.uniform_array itself draws (compose an "
                    "explicit MUL/ADD after it for any other range)."
                )
            return [Local(output_names[0], Call("loom.uniform_array", [Lit(n)]))]

        # F. Fallback for generic SSA arithmetic
        inputs = [self.safe_name(v.name) for k, v in op.inputs.items() if hasattr(v, "name")]
        if len(inputs) == 2:
            op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(op_type)
            if op_symbol:
                return [Local(output_names[0], BinOp(op_symbol, IRVar(inputs[0]), IRVar(inputs[1])))]
        return [RawBlock([f"-- Fallback: host math implementation for {op_type}"])]

    def format_lua_val(self, val):
        if isinstance(val, (int, float, bool)):
            return str(val).lower()
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, np.ndarray):
            return "{" + ", ".join(map(str, val.flatten())) + "}"
        return "nil"

    def generate_graph_topology(self, func: Function, func_name: str, ops_list=None, inputs_dict=None) -> dict:
        """
        Walks a heavy submodule MIL graph and serializes it to a static graph topology.
        """
        self._ensure_mil_passes_applied()
        ctx = TopologyContext(func_name)
        nodes = ctx.nodes
        topo_inputs = ctx.topo_inputs
        aliases = ctx.aliases

        inputs = inputs_dict if inputs_dict is not None else (func.inputs if func else {})
        operations = ops_list if ops_list is not None else (func.operations if func else [])

        # EXPORT-IMPROVEMENT-BACKLOG.md item 4: an "lstm" op can't be translated into ordinary static
        # topology nodes the way every other op_type below is -- ggml has no native LSTM/GRU op, and
        # unlike e.g. `linear`/`matmul`, correct recurrence needs a genuine host-side per-timestep loop
        # (see recurrent.py's own module docstring and tools/loom_mil_compiler/test_recurrent.py's
        # verified topology-generation logic), not a fixed sequence of graph nodes. `generate_graph_topology`
        # only ever returns ONE static topology for the whole `operations` list it's given; splitting a
        # function at an "lstm" op boundary into "pre-LSTM topology -> recurrent stepper call ->
        # post-LSTM topology" driver segments is real, unimplemented follow-up work, not something this
        # call can do safely.
        # Fail loudly and specifically here rather than either silently mistranslating (there's no OP_MAP
        # entry for "lstm" so it would otherwise hit the generic "missing a ggml mapping" message, which
        # doesn't explain why or point at the real fix) or leaving a caller to guess.
        if any(op.op_type in ("lstm", "gru") for op in operations):
            raise NotImplementedError(
                f"generate_graph_topology('{func_name}'): contains an '{next(op.op_type for op in operations if op.op_type in ('lstm', 'gru'))}' "
                "op. ggml has no native LSTM/GRU op -- correct recurrence needs a real per-timestep "
                "stepper (see tools/loom_mil_compiler/recurrent.py and the new "
                "LoomLuaBridge::l_run_recurrent C++ binding, which generalizes the existing "
                "BiLstmStepper), not a static topology. Auto-wiring this into the generic export "
                "profiles (monolithic/modular-blueprint driver synthesis) is unimplemented "
                "follow-up work -- for now, build the per-timestep topologies directly via "
                "recurrent.build_lstm_cell_topologies() and drive them with loom.run_recurrent() in a "
                "hand-written driver script, the same way tools/convert_kokoro/kokoro_driver.lua does "
                "today via BiLstmStepper."
            )

        resolve = ctx.resolve

        # Track inputs to the submodule and standardize the first input name to "hidden_states" for decoder layers
        first_input_var = None
        for name, var in inputs.items():
            if first_input_var is None:
                first_input_var = var
                
        if first_input_var is not None:
            orig_name = self.safe_name(first_input_var.name)
            
            if func_name.startswith("layer_"):
                # A submodule traced standalone with its real parameter name already literally
                # "hidden_states" (e.g. the modular-export blueprint, EXPORT-IMPROVEMENT-BACKLOG.md
                # item 2) would otherwise alias it to itself here -- `resolve()` walks `aliases` in a
                # `while name in aliases` loop, so a self-referential entry spins forever.
                if orig_name != "hidden_states":
                    aliases[orig_name] = "hidden_states"
                var_info = self.get_var_info(first_input_var)
                var_info["name"] = "hidden_states"
                topo_inputs.append(var_info)
            else:
                topo_inputs.append(self.get_var_info(first_input_var))
            
            for name, var in inputs.items():
                if var != first_input_var:
                    topo_inputs.append(self.get_var_info(var))
        else:
            for name, var in inputs.items():
                topo_inputs.append(self.get_var_info(var))

        for op in operations:
            op_type = op.op_type

            # The declarative rewrite table (topology_ops.py) owns every MIL op whose ggml lowering
            # is more than "one OP_MAP entry, inputs forwarded verbatim": lookup is by (op type,
            # guard predicate), so which condition selects which composition is stated in the rule
            # registration rather than buried in a branch. Everything it doesn't claim falls through
            # to the generic OP_MAP path below.
            rule = lookup_topology_rule(self, op)
            if rule is not None:
                rule(self, op, ctx)
                continue

            mapped_op = self.OP_MAP.get(op_type)
            if mapped_op is None:
                raise NotImplementedError(f"MIL op '{op_type}' is missing a ggml mapping.")

            inputs = []
            if mapped_op == "GET_ROWS":
                # GET_ROWS in Loom C++ strictly expects 2 inputs: [weights, indices]
                # Any 3rd input (like axis) must be pruned.
                x_val_obj = op.inputs.get("x") or op.inputs.get("params")
                indices_val_obj = op.inputs.get("indices")
                if x_val_obj and indices_val_obj:
                    indices_name = resolve(self.safe_name(indices_val_obj.name))
                    # An index Var traced through elementwise arithmetic (e.g. HF's `zeros_like(input_ids)`
                    # idiom, which coremltools decomposes to `input_ids - input_ids` rather than a plain
                    # fill) is NOT guaranteed to still be int-typed at the ggml level even though it's int
                    # at the MIL level: this project's generic elementwise primitives (op_add/op_sub/...)
                    # unconditionally `promote_i32_to_f32` BOTH operands before computing (see
                    # primitives_basic.cpp), so an all-int32 SUB still produces an F32 result -- fed
                    # straight into ggml_get_rows, which hard-asserts its index operand is GGML_TYPE_I32.
                    # First hit by Kokoro's CustomAlbert (`token_type_ids = zeros_like(input_ids)`,
                    # decomposed by coremltools to `input_ids - input_ids` rather than a plain fill) --
                    # not Albert-specific, so this checks the PRODUCER's op_type, not the (unreliable --
                    # MIL itself correctly types a `sub` of two int32 vars as int32; only THIS project's
                    # own runtime compute silently loses that) declared MIL dtype. A block INPUT or a
                    # `const`/weight producer is genuinely already int (GraphBuilder/GGUF both preserve
                    # declared int dtypes outside of arithmetic), so only the known-promoting elementwise
                    # op family needs the cast.
                    producer_op_type = indices_val_obj.op.op_type if getattr(indices_val_obj, "op", None) is not None else None
                    if producer_op_type in ("add", "sub", "mul", "div", "real_div", "floor_div", "pow"):
                        cast_name = f"{self.safe_name(op.outputs[0].name)}_indices_i32"
                        nodes.append({
                            "op": "CAST",
                            "inputs": [indices_name],
                            "outputs": [cast_name],
                            "attrs": {"dtype": "i32"},
                        })
                        indices_name = cast_name
                    inputs = [resolve(self.safe_name(x_val_obj.name)), indices_name]
            elif mapped_op == "MUL_MAT":
                # MUL_MAT strictly expects exactly 2 inputs: [x, y]
                # Any other trailing transpose variables must be pruned.
                x_val_obj = op.inputs.get("x")
                y_val_obj = op.inputs.get("y")
                if x_val_obj and y_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name)), resolve(self.safe_name(y_val_obj.name))]
            elif mapped_op in ("PERMUTE", "SOFTMAX", "CLAMP", "RSQRT", "RESHAPE", "VIEW", "LOG", "SQRT"):
                # Unary reduction/metadata operations in Loom C++ strictly expect exactly 1 input tensor.
                # ("MEAN" used to be handled here too, via the generic `reduce_mean` -> MEAN OP_MAP entry
                # -- EXPORT-ROADMAP.md R2's `lower_reduce_mean` pass now rewrites every `reduce_mean` into
                # a `loom_mean`/`loom_scale`-based composition before this generic path ever runs, so
                # "MEAN" can no longer appear as `mapped_op` here; its own CONT-before-MEAN fix moved to
                # `topology_ops.py`'s `loom_mean` rule.)
                x_val_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                if x_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name))]
            elif mapped_op in ("CONV_1D", "CONV_2D"):
                # Convolutions strictly expect exactly 2 inputs: [x, weight]
                # Strides, padding, dilation, groups are passed as JSON attributes.
                x_val_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                weight_val_obj = op.inputs.get("weight")
                if x_val_obj and weight_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name)), resolve(self.safe_name(weight_val_obj.name))]
            elif op_type in ("add", "mul"):
                # Swap commutative inputs to ensure the larger/dynamic tensor is first,
                # preventing GGML broadcast repetition failures.
                #
                # Mutual (different-axis) broadcast -- ggml_mul/ggml_add only ever let ONE operand
                # broadcast INTO the other's ALREADY-correct shape, so a genuine "outer product" case
                # (each operand size-1 on a DIFFERENT axis than the other) isn't representable that way
                # at all -- no longer needs handling here: `passes.py`'s `insert_explicit_broadcasts`
                # pass (EXPORT-ROADMAP.md R2a) rewrites any such `add`/`mul` before this walk ever runs,
                # splicing in two explicit `loom_broadcast_to` ops so both operands already match the
                # output shape by the time this branch sees them.
                x_var_obj = op.inputs.get("x")
                y_var_obj = op.inputs.get("y")
                inp1 = resolve(self.safe_name(x_var_obj.name)) if x_var_obj is not None and hasattr(x_var_obj, "name") else None
                inp2 = resolve(self.safe_name(y_var_obj.name)) if y_var_obj is not None and hasattr(y_var_obj, "name") else None

                if inp1 and inp2:
                    if inp1 in self.weights and inp2 not in self.weights:
                        inputs = [inp2, inp1]
                    else:
                        inputs = [inp1, inp2]
                elif inp1:
                    inputs = [inp1]
                elif inp2:
                    inputs = [inp2]
            else:
                for k, v in op.inputs.items():
                    if isinstance(v, Var):
                        inputs.append(resolve(self.safe_name(v.name)))
                    elif isinstance(v, (list, tuple)):
                        for item in v:
                            if isinstance(item, Var):
                                inputs.append(resolve(self.safe_name(item.name)))

            outputs = [self.safe_name(v.name) for v in op.outputs]

            attrs = {}
            for k, v in op.inputs.items():
                if not isinstance(v, Var) and not isinstance(v, (list, tuple)):
                    if hasattr(v, "val"):
                        attrs[k] = v.val
                    else:
                        attrs[k] = v

            node = {
                "op": mapped_op,
                "inputs": inputs,
                "outputs": outputs
            }
            if attrs:
                # Filter out complex objects
                serializable_attrs = {}
                for ak, av in attrs.items():
                    if isinstance(av, (int, float, str, bool, list, dict)):
                        serializable_attrs[ak] = av
                if serializable_attrs:
                    node["attrs"] = serializable_attrs
            nodes.append(node)

        func_outputs = func.outputs if func else (ops_list[-1].outputs if ops_list else [])
        # A topology can declare more than one co-equal output symbol now (EXPORT-ROADMAP.md P2 --
        # BACKLOG.md's implementation sequence): every one of `func`'s own declared outputs, not just
        # the first, is a real topology output. Single-output functions (every model on the roadmap as
        # of P2) get a one-element list, so nothing downstream that only ever looked at "the" output
        # changes behavior.
        output_symbols = [resolve(self.safe_name(v.name)) for v in func_outputs] if func_outputs else ["output"]

        pruned_nodes = self._prune_dead_nodes(nodes, output_symbols)

        # A declared input with no node (post-pruning) that actually reads it is unreachable from this
        # topology's own output(s) -- GraphBuilder's ggml_gallocr_alloc_graph only allocates a backend
        # buffer for tensors reachable from a declared output, so an orphan input tensor is created
        # but never given a buffer. If the driver ever supplied a value for it anyway (e.g. a
        # modular-export blueprint's per-layer function still nominally "declaring" a
        # cache_position/position_ids input that its own real call never ends up depending on once
        # past_key_values is forced to None -- see modular_export.py's _CACHE_KWARG_NAMES comment),
        # setting data into that unallocated tensor is a ggml hard-crash ("tensor buffer not set"), not
        # a graceful error. Dropping it from the declared list here means check_subgraph_calls treats
        # the driver still supplying it as an undeclared-input validation error instead -- catching the
        # mismatch at export time rather than as a runtime crash.
        referenced = {name for node in pruned_nodes for name in node["inputs"]}
        topo_inputs = [inp for inp in topo_inputs if inp["name"] in referenced]

        pruned_nodes, output_symbols = self._materialize_view_outputs(pruned_nodes, output_symbols)

        self._retype_fused_mask_input(topo_inputs, pruned_nodes, func_name)
        # After the retyping, never before: that check asserts the fused mask's only consumers are
        # cached ATTENTION nodes, and it has to see the original wiring to mean anything.
        self.mask_windows.update(self._route_windowed_masks(topo_inputs, pruned_nodes, func_name))

        # "output" (singular string) is both the original schema and what every single-output topology
        # still serializes -- byte-identical to pre-P2 output. "outputs" (plural array) is new, used only
        # when a function genuinely declares more than one output; see graph_topology.h's own comment on
        # the C++ side of this same distinction.
        topo = {"version": 1, "inputs": topo_inputs}
        if len(output_symbols) == 1:
            topo["output"] = output_symbols[0]
        else:
            topo["outputs"] = output_symbols
        topo["nodes"] = pruned_nodes
        return topo

    def _retype_fused_mask_input(self, topo_inputs, nodes, func_name):
        """A cached `ATTENTION` node's mask spans the whole cache, so its declared shape is
        `[n_kv, n_tokens]` -- not the `[n_tokens, n_tokens]` the trace produced (KV-CACHE.md 3.2).

        The trace cannot say this. HF computes scores of shape `[1, h, s, s]` with no cache, and a
        second independent `ct.RangeDim` for the mask's key axis fails coremltools' type inference at
        conversion time (KV-CACHE.md §2), so `attention_mask` shares one symbol with `tokens` and
        `cache_position` by design. The axis therefore arrives here, on the emitted topology, and the
        one thing that makes that sound is a property of the fused graph rather than an assumption:
        **after fusion the mask input's only consumers are cached ATTENTION nodes**, so no other node's
        shape derives from it. `fuse_loom_attention._mask_kv_slice_source` is what makes it true, by
        bypassing the trace-width `mask[..., :kv_len]` slice; this checks it.

        At `n_past = 0` the retyping changes nothing numerically -- `n_kv == n_tokens`, and
        `loom.causal_mask(n_tokens, 0)` already returns exactly that many values -- so a prefill is the
        same computation before and after. What it buys is a decode step, where the driver passes
        `n_past > 0` and the engine sizes this input to the real cache extent.

        Silent no-ops are the failure mode here, so both are raises: a topology with cached ATTENTION
        nodes whose masks are NOT declared inputs means the fusion left something between them (which is
        exactly the state this step exists to remove), and a mask input with any other consumer means
        retyping it would move a shape something else derives from.
        """
        cached = [
            node for node in nodes
            if node["op"] == "ATTENTION" and node.get("attrs", {}).get("kv_cache", True)
        ]
        if not cached:
            return
        declared = {inp["name"]: inp for inp in topo_inputs}
        # The ATTENTION rule emits [q, k, v, mask] -- see topology_ops._op_loom_fused_attention.
        mask_names = [node["inputs"][3] for node in cached if len(node["inputs"]) > 3]
        retypable = sorted({name for name in mask_names if name in declared})
        if not retypable:
            raise ValueError(
                f"topology '{func_name}': {len(cached)} cached ATTENTION node(s), but none of their "
                f"masks ({sorted(set(mask_names))}) is a declared input of this topology, so the "
                f"'n_kv' axis has nowhere to be declared. The mask must reach the fused node directly "
                f"-- fuse_loom_attention bypasses the traced mask[..., :kv_len] slice for exactly this "
                f"reason (see _mask_kv_slice_source); something between them survived."
            )
        for name in retypable:
            others = sorted({
                node["op"] for node in nodes
                if name in node["inputs"] and not (
                    node["op"] == "ATTENTION" and len(node["inputs"]) > 3 and node["inputs"][3] == name
                )
            })
            if others:
                raise ValueError(
                    f"topology '{func_name}': cannot declare input '{name}' as "
                    f"['n_kv', '{self.root_axis}'] because it is also read by {others}. Retyping is "
                    f"only sound while a fused mask's ONLY consumers are cached ATTENTION nodes -- any "
                    f"other node's shape would be derived from an axis the trace never had."
                )
            declared[name]["shape"] = [axes.N_KV.name, self.root_axis]

    def _route_windowed_masks(self, topo_inputs, nodes, func_name) -> dict:
        """Give each sliding-window attention block its own mask input, and say which window it wants.

        Interleaved local/global models (Gemma 3) need two different masks over the same keys: the full
        blocks attend to `[0, p]`, the sliding ones to `(p - window, p]`. The trace cannot express that
        -- it hands every block the one `attention_mask` input this family passes in (see
        `LMCausalModelExportConfig._attention_windows`) -- so the second mask is SYNTHESIZED here, as a
        declared topology input with no MIL var behind it, and the driver fills both in with
        `loom.causal_mask(n_tokens, n_past, window)`.

        Putting the window in the MASK rather than on the `ATTENTION` node is what keeps the engine out
        of it: `ggml_soft_max_ext` already takes an arbitrary `[n_kv, n_tokens]` mask, so a banded one
        needs no new primitive, no new attr and no runtime branch. The cost is one extra host-built
        array per distinct window per step, which is the same order as the mask already being built.

        One input per DISTINCT window, not per block: Gemma 3's 15 sliding blocks all want 512, so they
        share one. Returns `{input_name: window}` for the driver.
        """
        windows = self.kwargs.get("attention_windows")
        if not windows:
            return {}
        cached = [
            node for node in nodes
            if node["op"] == "ATTENTION" and node.get("attrs", {}).get("kv_cache", True)
            and len(node["inputs"]) > 3
        ]
        if not cached:
            return {}
        if len(cached) != len(windows):
            # The list is indexed by the `layer` attr, which `fuse_loom_attention` assigns in
            # attention-block occurrence order. Those agree for a uniform decoder and would not for a
            # hybrid whose conv blocks also count in `layer_types` -- better to refuse than to band the
            # wrong blocks.
            raise ValueError(
                f"topology '{func_name}': the checkpoint declares {len(windows)} layer window(s) but "
                f"the fusion produced {len(cached)} cached ATTENTION block(s). `attention_windows` is "
                "indexed by the block's own `layer` attr (occurrence order), so the two must agree."
            )

        declared = {inp["name"]: inp for inp in topo_inputs}
        routed = {}
        for node in cached:
            window = int(windows[int(node["attrs"]["layer"])])
            if window <= 0:
                continue
            base = node["inputs"][3]
            if base not in declared:
                raise ValueError(
                    f"topology '{func_name}': block {node['attrs']['layer']} wants a {window}-token "
                    f"window but its mask '{base}' is not a declared input, so there is nothing to "
                    "give it a windowed sibling of. Fusion must leave the mask reaching the node "
                    "directly (see _retype_fused_mask_input)."
                )
            name = f"{base}_sw{window}"
            if name not in declared:
                declared[name] = {"name": name, "dtype": declared[base]["dtype"],
                                   "shape": [axes.N_KV.name, self.root_axis]}
                topo_inputs.append(declared[name])
                routed[name] = window
            node["inputs"][3] = name
        return routed

    # Ops whose ggml result is a live VIEW of their input rather than a fresh contiguous buffer.
    # `ggml_backend_tensor_get` -- how every declared output is read back, by the Lua bridge and by
    # every reference test -- does a raw contiguous byte copy, so reading one of these back returns the
    # data in PRE-op order and silently ignores what the op did.
    _VIEW_PRODUCING_OPS = ("PERMUTE", "TRANSPOSE")

    def _materialize_view_outputs(self, nodes: list, output_symbols):
        """Appends a `CONT` after any declared output produced by a view op, so a topology's output is
        always a real contiguous buffer.

        **This is a general correctness fix, not a convenience.** A traced module whose last operation
        is a transpose emits a bare `PERMUTE` as the topology's declared output; `torch`'s own
        `.contiguous()` cannot prevent it, because MIL has no notion of contiguity and drops the call
        entirely. The result builds, runs, and returns the *untransposed* data.

        The hazard was already known and, until now, only ever avoided: `matcha_export.py`'s module
        docstring and `vits_export.StatsWrapper` both record deliberately NOT returning a transposed
        output for exactly this reason, and every hand-built converter that needs one writes
        `PERMUTE + CONT` by hand (`topology_ops.py` does the same internally in several places). What
        forced the fix rather than another workaround is a topology whose consumer *requires* the
        transposed layout: StyleTTS2's `bert_encoder`, whose driver reads `d_en_flat[c*T + t]`. Avoiding
        the transpose there is not an option, so the exporter has to emit the copy the bespoke converter
        always did.

        Caught by `test_e2e_kokoro_mil_topology_equivalence` as mean_abs_diff=0.717 against a reference
        whose values only reach 2.23 -- which is what a transpose looks like when nothing crashes.
        """
        by_output = {}
        for node in nodes:
            for name in node.get("outputs", []):
                by_output[name] = node
        out_nodes = list(nodes)
        new_symbols = []
        for symbol in output_symbols:
            producer = by_output.get(symbol)
            if producer is None or producer["op"] not in self._VIEW_PRODUCING_OPS:
                new_symbols.append(symbol)
                continue
            cont_name = f"{symbol}_cont"
            out_nodes.append({"op": "CONT", "inputs": [symbol], "outputs": [cont_name]})
            new_symbols.append(cont_name)
        return out_nodes, new_symbols

    def _prune_dead_nodes(self, nodes: list, output_symbols) -> list:
        """
        Removes topology nodes whose output is never consumed, directly or transitively, by any of the
        topology's own declared outputs (`output_symbols`: an iterable of symbol names -- a plain string
        is also accepted for older direct callers and treated as a single-element set). GraphBuilder
        builds and COMPUTES every node unconditionally regardless of whether anything uses its result --
        so an orphaned subgraph isn't just wasted compute, it can still crash (confirmed empirically on
        an orphaned chain that segfaulted during ggml_backend_graph_compute despite having zero real
        consumers, because nothing ever validates an unused node's own shapes/values are sane). Keeps
        only nodes reachable backward from any declared output.

        The GQA repeat_kv() fusion's own orphaned dependency chain (the original tile's now-unused
        "reps"-computation subgraph -- gather/concat/equal/select/div) no longer needs this: that fusion
        now runs as a real MIL->MIL pass (passes.py) with `common::dead_code_elimination` run right after
        it, so the orphan never survives into this walk at all (EXPORT-IMPROVEMENT-BACKLOG.md item 3).
        Kept as a general safety net for any topology-generation path that can still leave dead nodes
        behind (a pre-partitioned modular slice, an advanced/bespoke hand-built Program) rather than
        relying on every future caller to prove it never will.
        """
        needed = {output_symbols} if isinstance(output_symbols, str) else set(output_symbols)
        live = []
        for node in reversed(nodes):
            if any(o in needed for o in node["outputs"]):
                live.append(node)
                needed.update(node["inputs"])
        live.reverse()
        return live

    def _fused_ops(self, op_type: str):
        """Every op of `op_type` in the program(s) this GGUF is being written from.

        Usually that is `self.program` -- one traced graph, whether flattened or bespoke. A
        **multi-phase** export has no program of its own (`MultiPhase` builds each phase's topology with
        its own exporter and hands this one the finished topologies), so it sets `phase_programs`
        instead, and both geometries below then read the fused nodes from the phases that produced them.
        Without that, a multi-phase export with a KV-cached phase writes no cache geometry at all and the
        artifact is unloadable -- which is exactly what the first Whisper export did (BACKLOG.md P4.1).
        """
        programs = [self.program] if self.program is not None else list(self.phase_programs)
        for program in programs:
            for func in getattr(program, "functions", {}).values():
                for op in func.operations:
                    if op.op_type == op_type:
                        yield op

    def _kv_cache_geometry(self) -> dict:
        """The five facts `loom::make_kv_cache` needs, read off the fused ATTENTION nodes themselves --
        or `{}` when this export produced none (KV-CACHE.md 2.2b).

        Fusing changes a topology's RUNTIME REQUIREMENTS, not just its shape: an ATTENTION node with
        `kv_cache=true` makes `op_attention` throw unless the host registered a KvCache, and until this
        method existed the exporter wrote exactly one hparam (`loom.architecture`), so a fused model was
        unloadable. Stage 1 gave the bespoke converters this; the MIL exporter needs it too.

        Read from the graph rather than from the config, for the same reason `uses_kv_cache()` is
        derived: the number of cache slots must equal the number of ATTENTION blocks, and only the graph
        knows how many the fusion actually produced. That is NOT the model's layer count in general --
        LFM2 interleaves conv layers, so its attention-block count is smaller, and `fuse_loom_attention`
        assigns its `layer` indices densely to match (see that pass on why occurrence order is the
        correct addressing).
        """
        n_head_kv = head_dim_k = head_dim_v = None
        n_blocks = 0
        for op in self._fused_ops("loom_fused_attention"):
            n_blocks += 1
            k_shape, v_shape = list(op.inputs["k"].shape), list(op.inputs["v"].shape)
            geom = (int(k_shape[1]), int(k_shape[3]), int(v_shape[3]))
            if n_head_kv is None:
                n_head_kv, head_dim_k, head_dim_v = geom
            elif geom != (n_head_kv, head_dim_k, head_dim_v):
                # One cache is allocated for the whole model with one width per layer, so a model
                # whose blocks disagree cannot be served by it. Better to say so than to write the
                # first block's geometry and corrupt the rest.
                raise NotImplementedError(
                    f"loom_fused_attention op '{op.name}' has K/V geometry {geom}, but an earlier "
                    f"block declared {(n_head_kv, head_dim_k, head_dim_v)}. A KvCache is allocated "
                    "with ONE per-layer width, so per-block variation is unsupported."
                )
        if not n_blocks:
            return {}

        kv_cache_size = self.kwargs.get("kv_cache_size")
        if not kv_cache_size:
            raise ValueError(
                "this export fused attention into ATTENTION nodes, which need a KV cache at run time, "
                "but no `kv_cache_size` was passed to the backend -- so the GGUF would declare a cache "
                "it never sizes and make_kv_cache would reject it. Pass the capacity in tokens (the "
                "causal-LM family passes its own `max_seq_len`)."
            )
        return {
            "n_layer": n_blocks,
            "n_head_kv": n_head_kv,
            "n_embd_head_k": head_dim_k,
            "n_embd_head_v": head_dim_v,
            "kv_cache_size": int(kv_cache_size),
        }

    def _conv_state_geometry(self) -> dict:
        """The three facts `loom::make_conv_state_cache` needs, read off the fused SHORT_CONV nodes
        themselves -- or `{}` when this export produced none (BACKLOG.md P4.0.10).

        Exactly the arrangement `_kv_cache_geometry` uses, and for the same reason: a SHORT_CONV node
        with `conv_state=true` makes `op_short_conv` throw unless the host registered a store, so fusing
        changes the topology's runtime requirements and the file has to say so. Read from the graph
        rather than from the config because the slot count must equal the number of conv blocks the
        fusion actually produced -- for LFM2-350M that is 10, against a declared `num_hidden_layers` of
        16, and `fuse_loom_short_conv` numbers its slots densely to match.

        Unlike the KV cache there is no capacity to pass in: a conv slot holds `kernel - 1` columns and
        nothing else, so its size is a property of the weights and never of the context length.
        """
        n_state = n_embd_conv = None
        n_blocks = 0
        for op in self._fused_ops("loom_short_conv"):
            n_blocks += 1
            # MIL x is [batch, channels, seq]; weight is [channels, 1, kernel].
            geom = (int(op.inputs["weight"].shape[-1]) - 1, int(op.inputs["x"].shape[1]))
            if n_state is None:
                n_state, n_embd_conv = geom
            elif geom != (n_state, n_embd_conv):
                # One store is allocated for the whole model with one slot shape, so blocks that
                # disagree cannot be served by it -- say so rather than write the first block's
                # geometry and corrupt the rest.
                raise NotImplementedError(
                    f"loom_short_conv op '{op.name}' has geometry {geom} (kernel-1, channels), but "
                    f"an earlier block declared {(n_state, n_embd_conv)}. A ConvStateCache is "
                    "allocated with ONE slot shape, so per-block variation is unsupported."
                )
        if not n_blocks:
            return {}
        return {
            "n_conv_layer": n_blocks,
            "n_conv_state": n_state,
            "n_embd_conv": n_embd_conv,
        }

    def _collect_mul_mat_weight_names(self) -> set:
        """Every MUL_MAT node's *first* input, across all topologies -- the weight-first argument per
        loom's convention (src/ops/primitives_basic.cpp's op_mul_mat is a bare ggml_mul_mat(a, b) wrap
        with `a` as the weight). Mirrors tools/quantize/quantize_gguf_q8_0.py's topology-driven tensor
        selection (proven end-to-end against a real quantized model, see BACKLOG.md's "quantized weight
        support" milestone) rather than tensor-name pattern matching -- this exporter never emits
        "repeat_for" blocks (unlike that script's GGUF-KV-driven input), so no expansion pass is needed."""
        names = set()
        for topo in self.topologies.values():
            for node in topo.get("nodes", []):
                if node.get("op") == "MUL_MAT" and node.get("inputs"):
                    names.add(node["inputs"][0])
        return names

    def _prune_dead_weights(self) -> None:
        """
        Drops any `self.weights` entry never referenced as an input by any surviving topology node.
        `generate_graph_topology`'s "const" handling unconditionally serializes EVERY MIL const op
        (below the >100-element inline threshold) as a GGUF weight tensor -- including incidental
        attribute-only constants (e.g. a "matmul" op's own `transpose_x`/`transpose_y` boolean flags,
        already consumed directly via `.val` in Python by the dedicated "matmul" composition and never
        emitted as a node input reference) that are not real model weights at all and have no consumer
        anywhere in the topology. Confirmed one such orphan (`_80_transpose_x_0`, GGUF-declared with an
        empty `shape=[]`, i.e. a genuine zero-rank scalar tensor) sitting immediately before a real,
        actually-used weight (`const_2_to_fp16`, LFM2's RoPE inverse-frequency table) in tensor-declaration
        order -- and a real, reproducible crash computing a MUL_MAT against that exact real weight,
        consistent with a degenerate zero-byte allocation for the dead scalar corrupting the next
        tensor's backing memory. Pruning these (same principle as `_prune_dead_nodes`) is correct
        regardless of whether it's this crash's full root cause: a weight nothing reads should never be
        written to the GGUF at all.
        """
        referenced = set()
        for topo in self.topologies.values():
            for node in topo.get("nodes", []):
                referenced.update(node.get("inputs", []))
        dead = [name for name in self.weights if name not in referenced]
        for name in dead:
            del self.weights[name]

    def _write_tokenizer(self, w, tokenizer_dir: str):
        """Dispatches to the right vocab writer for `tokenizer_dir`'s real HF tokenizer family, auto-
        detecting both the family ("bpe"/"wordpiece"/"sentencepiece_proto") and, for "bpe", the
        pretokenizer regex shape (`tokenizer.ggml.pre`) unless explicitly overridden via
        `tokenizer_family`/`tokenizer_pre` kwargs -- see tokenizer_detect.py's own module docstring for
        the detection recipes."""
        from .tokenizer_detect import detect_vocab_family, detect_loom_pre_type

        family = self.kwargs.get("tokenizer_family") or detect_vocab_family(tokenizer_dir)

        if family == "bpe":
            from .bpe_tokenizer_export import write_bpe_vocab
            pre_type = self.kwargs.get("tokenizer_pre")
            if pre_type is None:
                from transformers import AutoTokenizer
                pre_type = detect_loom_pre_type(AutoTokenizer.from_pretrained(tokenizer_dir))
            write_bpe_vocab(w, tokenizer_dir, pre_type=pre_type)
        elif family == "wordpiece":
            from .wordpiece_tokenizer_export import write_wordpiece_vocab
            write_wordpiece_vocab(w, tokenizer_dir)
        elif family == "sentencepiece_proto":
            from pathlib import Path
            # Every entry point that imports loom_mil_compiler (export_hf_causal_lm.py,
            # export_lfm2_*.py) inserts tools/ itself (not its parent) onto sys.path, so convert_nemo/ is
            # importable as a top-level package the same way loom_mil_compiler is -- not "tools.convert_nemo".
            from .spm_tokenizer_export import write_sentencepiece_vocab
            proto_path = next(p for p in (Path(tokenizer_dir) / "tokenizer.model",
                                           Path(tokenizer_dir) / "spiece.model") if p.exists())
            write_sentencepiece_vocab(w, proto_path.read_bytes())
        elif family == "byte":
            from .byt5_tokenizer_export import write_byt5_vocab
            write_byt5_vocab(w, tokenizer_dir)
        else:
            raise NotImplementedError(f"_write_tokenizer: no vocab writer for tokenizer family {family!r}")

    def write_gguf(self, driver_script: str):
        from gguf import GGUFWriter
        import hashlib
        import os

        self._prune_dead_weights()

        arch = self.kwargs.get("architecture") or os.environ.get("LOOM_ARCH", "mil_model")
        w = GGUFWriter(self.output_path, f"loom-{arch}")
        w.add_string("loom.architecture", arch)

        tokenizer_dir = self.kwargs.get("tokenizer_dir") or os.environ.get("LOOM_TOKENIZER_DIR")
        if tokenizer_dir:
            self._write_tokenizer(w, tokenizer_dir)

        # Embed the Lua driver orchestration script
        w.add_string("model.driver_script", driver_script)

        # The KV-cache geometry, when this export produced ATTENTION nodes (KV-CACHE.md 2.2b). Same
        # five "loom.*" keys the bespoke converters write and `loom::make_kv_cache` reads, so a host
        # allocates from the file alone rather than from a per-model C++ struct. Absent entirely for
        # every unfused export, which is every model but the causal LMs.
        for key, value in self._kv_cache_geometry().items():
            w.add_uint32(f"loom.{key}", int(value))

        # The conv-state geometry, on exactly the same terms, when this export produced SHORT_CONV
        # nodes (BACKLOG.md P4.0.10). Absent for every model without stateful convolutions, which today
        # is everything but a fused LFM2.
        for key, value in self._conv_state_geometry().items():
            w.add_uint32(f"loom.{key}", int(value))

        # The family's own declared host-facing numbers (`LoomExportConfig.hparams()`), in the same
        # "loom.*" namespace and for the same reason as the two geometries above: a host that has to
        # size an input before it can call `infer` reads the file instead of carrying a per-model C++
        # struct. Unlike those two this is DECLARED by the config rather than derived from the emitted
        # graph -- the two geometries answer "what did this export produce", these answer "what does a
        # caller have to know", and only the config knows the second.
        for key, value in (self.kwargs.get("hparams") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"hparams[{key!r}] is {value!r} ({type(value).__name__}). `hparams()` writes GGUF "
                    f"scalars a host reads back with hparam_u32/hparam_f32, so only int and float are "
                    f"representable -- anything structured belongs in a topology or in the driver."
                )
            if isinstance(value, int):
                w.add_uint32(f"loom.{key}", value)
            else:
                w.add_float32(f"loom.{key}", float(value))

        # Embed each static submodule topology JSON string
        for submodule_name, topo in self.topologies.items():
            w.add_string(f"model.graph_topology.{submodule_name}", json.dumps(topo, cls=NumpyEncoder))

        qtype = None
        quantizable = set()
        block_size = 1
        if self.quantize:
            from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
            qtype = GGMLQuantizationType[self.quantize]
            block_size, _ = GGML_QUANT_SIZES[qtype]
            quantizable = self._collect_mul_mat_weight_names()
        n_quantized = 0

        # Content-address weight payloads (BACKLOG.md P0.2): a split export can legitimately declare the
        # SAME weight under two different names (LFM2's tied embedding is both `prefix.module_weight` and
        # `suffix_1.module_weight` under the modular profile), and a single traced model routinely
        # mints many distinctly-named constants that happen to hold the same value (e.g. a bare `1.0`
        # scalar reused across unrelated ops). Write the bytes once and alias every later name to the
        # first, instead of paying for (and storing) a second copy. Hashed on the FINAL on-disk
        # shape/dtype/bytes (post dtype-cast, post quantization), not just the raw array's bytes: two
        # names sharing byte content but differing in shape (confirmed empirically -- a rank-1 `[1]`
        # scalar and a rank-3 `[1, 1, 1]` one holding the identical value) or in quantization eligibility
        # (`name in quantizable`, e.g. one is a tied embedding used as a MUL_MAT weight in one topology's
        # slice but only via GET_ROWS in another's) must NOT be merged, or the alias would silently hand
        # one consumer the other's shape or dtype. A name only ever becomes an alias if the tensor a
        # consumer would see -- shape, dtype, and data -- is indistinguishable from an earlier name's.
        payload_hash_to_name = {}
        alias_names = []
        alias_targets = []

        # Quantize & write weights / tensors
        for name, array in self.weights.items():
            if array.dtype == bool or array.dtype == np.bool_:
                array = array.astype(np.int32)
            elif np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
            elif array.dtype == np.int64:
                array = array.astype(np.int32)
            elif not np.issubdtype(array.dtype, np.number):
                continue

            # Only MUL_MAT weight tensors are quantized (matching llama.cpp's own convention: norm/bias
            # 1D tensors have negligible size benefit and real accuracy cost). Tensors whose last
            # (fastest-varying) dimension isn't block-aligned are left F32 rather than erroring, same
            # graceful behavior as the standalone quantize_gguf_q8_0.py POC.
            if (qtype is not None and name in quantizable and array.ndim >= 2 and array.dtype == np.float32
                    and array.shape[-1] % block_size == 0):
                from gguf import quants
                array_to_write = quants.quantize(np.ascontiguousarray(array), qtype)
                raw_dtype = qtype
            else:
                array_to_write = array
                raw_dtype = None

            # The dtype+shape tag guards against two same-bytes-different-meaning collisions a pure byte
            # hash would miss: an all-zero I32 array vs an all-zero F32 array of the same byte length,
            # and (found empirically, see BACKLOG.md) a rank-1 `[1]` scalar constant vs a rank-3
            # `[1, 1, 1]` one holding the identical single value -- MIL mints these at different ranks
            # for different consumers, and aliasing them would silently hand one consumer the other's
            # shape, not just its bytes. Two names only ever become aliases of each other when the
            # tensor a consumer would see -- shape, dtype, AND data -- is indistinguishable either way.
            hasher = hashlib.sha256()
            hasher.update(str(array_to_write.dtype).encode("ascii"))
            hasher.update(str(array_to_write.shape).encode("ascii"))
            hasher.update(np.ascontiguousarray(array_to_write).tobytes())
            digest = hasher.hexdigest()
            canonical = payload_hash_to_name.get(digest)
            if canonical is not None:
                alias_names.append(name)
                alias_targets.append(canonical)
                continue
            payload_hash_to_name[digest] = name

            # No `raw_shape` -- add_tensor's raw_shape (when given) is a *byte*-shape fed straight
            # into quant_shape_from_byte_shape, not the pre-quantization logical shape; omitting it
            # lets it default to the quantized array's own (correct) byte-shape.
            if raw_dtype is not None:
                w.add_tensor(name, array_to_write, raw_dtype=raw_dtype)
                n_quantized += 1
            else:
                w.add_tensor(name, array_to_write)

        # add_array() is a no-op when given an empty list, so this KV pair is simply absent for every
        # model with no duplicate payloads -- GgufModel::load treats that as zero aliases, not an error.
        w.add_array("loom.tensor_alias.names", alias_names)
        w.add_array("loom.tensor_alias.targets", alias_targets)

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        suffix = f", {n_quantized} tensor(s) quantized to {self.quantize}" if self.quantize else ""
        if alias_names:
            suffix += f", {len(alias_names)} duplicate weight name(s) aliased instead of re-stored"
        print(f"wrote GGUF with driver_script and {len(self.topologies)} topologies to {self.output_path}{suffix}")
