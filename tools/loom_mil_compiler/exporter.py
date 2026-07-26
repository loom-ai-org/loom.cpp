import json
import math
import re
import os
import sys
import numpy as np
from coremltools.converters.mil.mil import Block, Function, Operation, Var

from .driver_ir import (
    Argmax, Assign, BinOp, Break, Call, FieldAccess, If, Index, Len, Lit, Local, LocalDecl,
    LuaCodegen, RawBlock, RawExpr, Return, SubgraphCall, UnaryOp, Var as IRVar, While, check_subgraph_calls,
    validate,
)
from .driver_ir import Function as IRFunction
from .passes import apply_loom_mil_passes

# Traced-model input names auto-computed by the driver (via loom.range) rather than unpacked from the
# caller's `inputs` table -- see apply_monolithic_export/apply_atomic_export's own comment.
_POSITION_INPUT_NAMES = {"cache_position", "position_ids"}

# Traced-model input names auto-computed by the driver via loom.causal_mask (an already-prepared 4D
# additive mask, the same "pass it explicitly so the traced model skips computing it internally" fix as
# _POSITION_INPUT_NAMES -- see export_lfm2_*.py's own _causal_mask() comment).
_CAUSAL_MASK_INPUT_NAMES = {"attention_mask"}

# CoreML's own naming convention for a symbolic (dynamic) shape dimension -- always "is" followed by
# digits (e.g. "is0", "is936"). Matched with \b so it only substitutes whole symbol tokens, never a
# coincidental "is" substring inside a longer identifier.
_DYNAMIC_SYMBOL_RE = re.compile(r"\bis\d+\b")

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
        self.ir_function = None
        self.profile = kwargs.get("profile") or os.environ.get("LOOM_PROFILE", None)
        self.output_path = kwargs.get("output_path") or os.environ.get("LOOM_OUTPUT_PATH", "model.gguf")
        self.quantize = kwargs.get("quantize") or os.environ.get("LOOM_QUANTIZE", None)
        # {raw MIL symbol string (e.g. "is531") -> replacement expression (e.g. "2*n_tokens")}. The
        # engine's own dynamic-shape support is genuinely single-axis (see get_var_info's own docstring):
        # every symbolic dim ordinarily collapses to the bare "n_tokens" SymbolEnv resolves at build
        # time, which is correct whenever several distinct "isN" names all really mean the SAME one true
        # dynamic quantity (the common case) but wrong the one time a topology genuinely has more than
        # one independently-varying dynamic axis (first hit by Kokoro's "decoder_vocoder" phase: asr's
        # own frame count vs. f0_curve/n_curve's fixed-2x/noise_in's fixed-600x/wsum's fixed-600x+20
        # lengths -- none of these are op-derived from asr, they're independently-traced LEAF inputs, so
        # there's no data-flow path this exporter could use to infer the ratio on its own). This dict
        # lets a caller who DOES know the real ratio (because it's inherent to how their own wrapper
        # module's forward() signature is shaped, not something recoverable from the graph) override
        # specific symbols by their raw name -- populated from the real traced MIL Vars' own shape
        # symbols, not guessed from string patterns.
        self.symbol_overrides = kwargs.get("symbol_overrides") or {}

    def _sub_symbol(self, s: str) -> str:
        """Replaces every occurrence of a symbolic MIL dim (e.g. "is531") in `s` with its
        `self.symbol_overrides` entry if the caller registered one for that exact raw symbol, else the
        usual bare "n_tokens" fallback. See `symbol_overrides`' own docstring in __init__."""
        return _DYNAMIC_SYMBOL_RE.sub(lambda m: self.symbol_overrides.get(m.group(0), "n_tokens"), s)

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
        """
        # No cycle guard: MIL/SSA graphs are acyclic by construction (an op's inputs always name EARLIER-
        # defined vars, never itself), so a genuine infinite loop here isn't possible. A guard keyed on
        # `id(var)` alone WAS added once, then found to actively corrupt correct answers instead -- the
        # "concat" case just above is the first branching walk in this function (recursing into multiple
        # operands from one call), and two operands legitimately sharing a common upstream ancestor (a
        # real DAG diamond, not a cycle -- confirmed on Kokoro's SineGen: `rad0.unsqueeze(1)` and
        # `rad_values[:,1:,:]` both trace back to the SAME `rad_values` var) would hit a per-walk `_seen`
        # set's SECOND visit and silently return None, exactly the bug already root-caused and fixed the
        # identical way in `_resolve_scalar_expr`'s own cycle guard for VITS (see BACKLOG.md).
        if var.shape is None or torch_axis >= len(var.shape):
            return None
        dim = var.shape[torch_axis]
        if not _DYNAMIC_SYMBOL_RE.search(str(dim)):
            return str(dim)

        op = var.op
        if op is None:
            # A genuine (sub)function input with no producer -- ordinarily this IS the actual dynamic
            # quantity "n_tokens" derives from (e.g. "waveform" itself), but not always: a topology with
            # more than one independently-traced dynamic LEAF input (Kokoro's "decoder_vocoder" -- see
            # symbol_overrides' own docstring) has NO data-flow path here to tell that apart from the
            # ordinary case, so an explicit caller-registered override always wins when present.
            return self._sub_symbol(str(dim))

        _UNARY_PASSTHROUGH_OPS = {
            "cast", "log", "exp", "sqrt", "rsqrt", "abs", "neg", "sign", "floor", "clamp",
            "tanh", "sigmoid", "relu", "gelu", "softplus", "identity", "softmax", "logical_not", "silu",
            "leaky_relu", "cumsum", "atan", "sin", "cos",
        }
        if op.op_type in _UNARY_PASSTHROUGH_OPS:
            # Pure unary, shape-preserving ops -- the axis's real expression is whatever its single
            # input's already is. Needed for the SAME reason the elementwise-broadcast case is: a chain
            # like log(x+eps) sitting between the true dynamic source and a "shape" op that reads this
            # var's real (built) tensor dimensions back out (confirmed on Conformer-CTC's mel-frontend
            # length tracking -- `gather(shape(real_div(...(log(...(matmul with the STFT conv's own
            # output)...)))))` -- every one of those intermediate unary ops needed to be walked through,
            # not just the ones this file happened to hit first).
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
            # very node's JSON. Reusing `_try_resolve_reshape_shape_input` here (rather than duplicating
            # that logic) is what makes THAT already-correct answer visible to a completely different
            # consumer: a `gather(shape(q), 2)` reading Q's own post-reshape sequence length back out one
            # step further downstream (confirmed on Conformer-CTC's `rel_shift`, whose `b, h, qlen,
            # pos_len = x.size()` queries `matrix_bd`'s shape, which recurses through `matmul` into the
            # Q/K/V reshape's own output -- without this case, that walk gave up at "reshape" (no case
            # existed for it at all) and fell back to the same "n_tokens" substitution the Q reshape
            # itself needed `_try_resolve_reshape_shape_input` to avoid).
            resolved = self._try_resolve_reshape_shape_input(op)
            if resolved is not None and torch_axis < len(resolved):
                if resolved[torch_axis] != -1:
                    return str(resolved[torch_axis])
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
                        total_expr = " * ".join(f"({e})" for e in x_axis_exprs)
                        other_expr = " * ".join(f"({e})" for e in other_exprs) if other_exprs else "1"
                        return f"(floor(({total_expr}) / ({other_expr})))"

        if op.op_type == "range_1d" and torch_axis == 0:
            # `range_1d`'s own output LENGTH is a real formula over its start/end/step, using the exact
            # same resolution `_resolve_range_scalar`/`_try_derive_gather_shape_value` already apply when
            # emitting this op's own JSON node -- needed one level further downstream than that node
            # itself: a `reshape`/`repeat` consuming THIS range's real output (e.g. broadcasting a
            # length-validity arange up before a comparison) queries ITS OWN declared shape, which is
            # this range's element count, not a bare "n_tokens" substitution.
            start_e = self._resolve_range_scalar(op.inputs.get("start"))
            end_e = self._resolve_range_scalar(op.inputs.get("end"))
            step_e = self._resolve_range_scalar(op.inputs.get("step"))
            if start_e is not None and end_e is not None and step_e is not None:
                if start_e in (0, 0.0) and step_e in (1, 1.0):
                    return str(end_e)
                return f"(floor((({end_e}) - ({start_e})) / ({step_e})))"

        if op.op_type == "conv":
            x_var = op.inputs.get("x")
            weight_var = op.inputs.get("weight")
            strides = op.inputs.get("strides").val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else None
            pad = op.inputs.get("pad").val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else None
            dilations = op.inputs.get("dilations").val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else None
            if x_var is None or weight_var is None or strides is None or pad is None or x_var.shape is None:
                return None
            rank = len(var.shape)
            # MIL conv is NC(D...) -- axis 0 is batch (always a literal 1 for every real input this
            # exporter targets, same architectural assumption used throughout this file -- confirmed
            # needed on Conformer-CTC's GLU-split VIEW, whose `x` operand's own axis-0 walk bottoms out
            # here: an earlier version returned `None` unconditionally for torch_axis<2, which skips
            # straight past this whole function's OWN final "torch_axis==0 -> batch=1" fallback, since a
            # bare `return None` inside an `if` block returns from the ENCLOSING function, not just this
            # branch), axis 1 is out-channels (from the weight, not derived from `x`, genuinely
            # unresolvable here), the remaining `rank - 2` axes are the real spatial ones this formula
            # applies to, in the same order for `x` and its output (conv never permutes spatial axes).
            if torch_axis == 0:
                return "1"
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
            return f"(floor((({in_expr}) + {pad_before + pad_after} - {eff_kernel}) / {stride}) + 1)"

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
            sfh = float(sfh_obj.val) if sfh_obj is not None and hasattr(sfh_obj, "val") and sfh_obj.val is not None else 1.0
            sfw = float(sfw_obj.val) if sfw_obj is not None and hasattr(sfw_obj, "val") and sfw_obj.val is not None else 1.0
            scale = sfh if torch_axis == rank - 2 else sfw
            in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
            if in_expr is None:
                return None
            if scale == 1.0:
                return in_expr
            return f"(floor(({in_expr})*{scale}))"

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
            strides = op.inputs.get("strides").val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else None
            pad = op.inputs.get("pad").val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else None
            dilations = op.inputs.get("dilations").val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else None
            if x_var is None or weight_var is None or strides is None or x_var.shape is None:
                return None
            rank = len(var.shape)
            if torch_axis == 0:
                return "1"
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
            return f"((({in_expr}) - 1) * {stride} - {pad_before + pad_after} + {eff_kernel})"

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
            transpose_x = bool(tx_var.val) if tx_var is not None and hasattr(tx_var, "val") and tx_var.val is not None else False
            transpose_y = bool(ty_var.val) if ty_var is not None and hasattr(ty_var, "val") and ty_var.val is not None else False
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
            reps_val = reps_var.val if reps_var is not None and hasattr(reps_var, "val") else None
            if reps_val is not None and x_var is not None and torch_axis < len(reps_val):
                rep = int(reps_val[torch_axis])
                in_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                if in_expr is not None:
                    return in_expr if rep == 1 else f"({in_expr} * {rep})"
            elif x_var is not None and x_var.shape is not None and torch_axis < len(x_var.shape):
                # `reps` itself is unavailable (poisoned by the same "computed via a runtime shape
                # query" tracing artifact as the GQA case), so the real multiplier can't be read for
                # EITHER axis of a `tile`. Two sub-cases, both heuristic but bounded:
                if x_var.shape[torch_axis] == 1:
                    # The input axis is a literal, static 1 -- exactly the shape a batch-broadcast tile
                    # has, and this exporter's whole design never targets real multi-batch inference
                    # (every declared model input's own batch axis is a literal 1) -- resolves to "1".
                    return "1"
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
            axes_val = axes_var.val if axes_var is not None and hasattr(axes_var, "val") else None
            if x_var is not None and axes_val is not None and x_var.shape is not None:
                out_rank = len(var.shape)
                in_rank = len(x_var.shape)
                norm_axes = sorted((int(a) + out_rank if a < 0 else int(a)) for a in axes_val) if op.op_type == "expand_dims" else \
                            sorted((int(a) + in_rank if a < 0 else int(a)) for a in axes_val)
                if op.op_type == "expand_dims":
                    if torch_axis in norm_axes:
                        return "1"
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
            axes_val = axes_var.val if axes_var is not None and hasattr(axes_var, "val") else None
            keep_dims_val = bool(keep_dims_var.val) if keep_dims_var is not None and hasattr(keep_dims_var, "val") and keep_dims_var.val is not None else False
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
            pad_val = pad_var.val if pad_var is not None and hasattr(pad_var, "val") else None
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
                        return in_expr if lp == 0 and rp == 0 else f"(({in_expr}) + {lp + rp})"

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
            begin_mask_val = begin_mask_var.val if begin_mask_var is not None and hasattr(begin_mask_var, "val") else None
            end_mask_val = end_mask_var.val if end_mask_var is not None and hasattr(end_mask_var, "val") else None
            is_begin_masked = begin_mask_val is not None and len(begin_mask_val) > torch_axis and bool(begin_mask_val[torch_axis])
            is_end_masked = end_mask_val is not None and len(end_mask_val) > torch_axis and bool(end_mask_val[torch_axis])

            # `begin` may resolve to either a plain non-negative literal (the common case) or a genuine
            # SymbolEnv expression string (e.g. RelPositionalEncoding's `start_pos = center_pos -
            # gather(shape(x), 1)`, resolved via `_resolve_scalar_expr`'s arithmetic-walk) -- both are
            # equally valid as the subtrahend in `end - begin` below, so both are kept, not just ints.
            if is_begin_masked:
                begin_expr = "0"
            else:
                resolved_begin = self._resolve_slice_axis_value(begin_var, torch_axis)
                if isinstance(resolved_begin, int) and resolved_begin >= 0:
                    begin_expr = str(resolved_begin)
                elif isinstance(resolved_begin, str):
                    begin_expr = resolved_begin
                else:
                    begin_expr = None

            end_expr = None
            if x_var is not None and x_var.shape is not None and torch_axis < len(x_var.shape):
                if is_end_masked:
                    end_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                else:
                    resolved_end = self._resolve_slice_axis_value(end_var, torch_axis)
                    if resolved_end is not None:
                        end_expr = str(resolved_end) if isinstance(resolved_end, int) else resolved_end
                if end_expr is None or end_expr == "n_tokens":
                    # An unresolved (None) or bare-"n_tokens" `end` is almost certainly this walk
                    # bottoming out somewhere inside the `end` chain, NOT a genuine answer -- confirmed on
                    # two distinct real cases: Conformer-CTC's `att_mask = fill[0:current_lengths,
                    # 0:current_lengths]` (`current_lengths` derived from the REAL "length" graph INPUT's
                    # own runtime VALUE via ADD/DIV/floor_div, architecturally impossible to resolve into
                    # a SymbolEnv shape expression at all -- SymbolEnv only ever binds compile-time shape
                    # quantities like n_tokens, never a tensor's actual data, the same "value only exists
                    # after graph compute" limit `_try_derive_gather_shape_value`'s docstring documents
                    # for RANGE_1D) and the positional-encoding table crop (`self.pe[:, start:end]`, whose
                    # `end` is a real ARITHMETIC EXPRESSION over a gather -- `center + t` -- that
                    # `_resolve_range_scalar`'s narrower "exact gather(shape(x), idx)" pattern match can't
                    # see through at all, returning None outright). But this whole exporter already
                    # assumes single-utterance, no-padding inference everywhere else (e.g. the always-1
                    # batch axis) -- under that assumption BOTH values are ALWAYS numerically equal to
                    # `x`'s own real (allocated) extent, so trusting `x`'s own extent here is correct for
                    # every case this exporter targets, not just a guess.
                    x_full_expr = self._infer_dynamic_dim_expr(x_var, torch_axis, _seen)
                    if x_full_expr is not None and x_full_expr != "n_tokens":
                        end_expr = x_full_expr
            if end_expr is not None and begin_expr is not None:
                return end_expr if begin_expr == "0" else f"(({end_expr}) - ({begin_expr}))"

        if op.op_type == "split":
            # `split(x, axis, num_splits/split_sizes)` divides `x` into N outputs along `axis` -- every
            # OTHER axis is a direct, unchanged passthrough to `x`'s own corresponding axis. Needed for
            # Conformer-CTC's combined Q/K/V (or similar) linear projection split before `rel_shift`'s
            # `matrix_bd`/`matrix_ac` computation queries its own shape back out -- only the passthrough
            # case is implemented (the split axis ITSELF, if queried, falls through to the blind
            # substitution below; nothing seen so far has needed it).
            x_var = op.inputs.get("x")
            axis_var = op.inputs.get("axis")
            axis_val = axis_var.val if axis_var is not None and hasattr(axis_var, "val") else None
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
            axis_val = axis_var.val if axis_var is not None and hasattr(axis_var, "val") else None
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
                        parts.append(f"({part_expr})")
                    return "(" + "+".join(parts) + ")"

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
            perm_val = perm_var.val if perm_var is not None and hasattr(perm_var, "val") else None
            if x_var is not None and perm_val is not None and x_var.shape is not None:
                rank = len(var.shape)
                norm_perm = [(int(p) + rank) if int(p) < 0 else int(p) for p in perm_val]
                if torch_axis < len(norm_perm):
                    in_axis = norm_perm[torch_axis]
                    if 0 <= in_axis < len(x_var.shape):
                        return self._infer_dynamic_dim_expr(x_var, in_axis, _seen)

        if op.op_type == "stack":
            # Inserts one new axis (like expand_dims) but from N same-shaped operands rather than one --
            # any axis OTHER than the new one has a direct 1:1 correspondence to (any one of, they're
            # all identical there by construction) the stacked operands, just shifted. Needed for the
            # SAME STFT-magnitude chain: `stack([real, imag], axis=-1)` is the exporter's own composed
            # RESHAPE+CONCAT (see the "stack" op_type translation), but the ORIGINAL MIL op this walk
            # sees is still the real `stack`.
            values = op.inputs.get("values")
            axis_var = op.inputs.get("axis")
            axis_val = axis_var.val if axis_var is not None and hasattr(axis_var, "val") else None
            if values is not None and axis_val is not None and len(values) > 0:
                first = values[0]
                if isinstance(first, Var) and first.shape is not None:
                    out_rank = len(var.shape)
                    stack_axis = int(axis_val) + out_rank if axis_val < 0 else int(axis_val)
                    if torch_axis == stack_axis:
                        return str(len(values))
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
                if not _DYNAMIC_SYMBOL_RE.search(str(operand.shape[torch_axis])):
                    continue
                inferred = self._infer_dynamic_dim_expr(operand, torch_axis, _seen)
                if inferred is not None and inferred != "1":
                    return inferred
                if first_resolved is None:
                    first_resolved = inferred
            if first_resolved is not None:
                return first_resolved

        if torch_axis == 0 and len(var.shape) >= 2:
            # Bottomed out (no case above understood the full producer chain) on what -- by this whole
            # exporter's own stated architecture (every declared model input's own batch axis is a
            # literal 1, the same assumption `_try_derive_gather_shape_value`'s own dedicated
            # torch_axis==0 shortcut already relies on) -- can ONLY be a batch axis. Needed for
            # Conformer-CTC's CMVN std-dev: `x_std`'s own axis 0 walk runs through a long, twisted
            # select/sub/pow/tile chain this file doesn't have (and doesn't need) a specific case for,
            # bottoming out here; blindly substituting "n_tokens" there was flatly wrong (confirmed: it
            # produced an element-count-changing RESHAPE target, not just an imprecise one) since axis 0
            # is never genuinely the sequence-length axis for any real input this exporter targets. A
            # model that genuinely needed a non-1 batch axis 0 would surface as a numerical mismatch
            # against the reference model here, not a syntax error.
            return "1"

        # Any other producer (pad/expand_dims/squeeze/etc.): not a transform this walk understands, but
        # also not necessarily wrong to keep walking from -- fall back to a bare symbol substitution
        # (identical to get_var_info's own long-standing default) rather than giving up outright. This
        # matters for e.g. a reflect-pad immediately upstream of a conv (confirmed on Conformer-CTC's
        # STFT: the pre-conv `expand_dims` var's own shape is the COMPOSITE expression "is1639 + 512",
        # not a bare symbol) -- substituting the base symbol there ("n_tokens + 512") and letting the
        # conv branch above apply its own formula on top gives the right answer without this walk needing
        # to specifically understand every intermediate op type between the true input and the first conv.
        return self._sub_symbol(str(dim))

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

    def _try_derive_gather_shape_value(self, var):
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
        if indices_var is None or not hasattr(indices_var, "val") or indices_var.val is None:
            return None
        real_var = shape_vec_var.op.inputs.get("x")
        if real_var is None or real_var.shape is None:
            return None
        idx = int(np.asarray(indices_var.val).reshape(-1)[0])
        real_rank = len(real_var.shape)
        torch_axis = idx + real_rank if idx < 0 else idx
        if not (0 <= torch_axis < real_rank):
            return None
        if torch_axis == 0 and real_rank >= 2:
            # `x.shape[0]` (or `x.size(0)`) on a rank>=2 activation is the canonical PyTorch idiom for
            # reading BATCH SIZE -- and this whole exporter's architecture only ever targets batch=1
            # models (every declared model input's own batch axis is a literal 1, stated repeatedly
            # elsewhere in this file, e.g. the `tile` case above). Short-circuiting straight to "1" here
            # is far more robust than walking the full producer chain (which, for a deep Conformer-CTC
            # encoder layer, runs through `layer_norm`/`linear`/`matmul`/`add` dozens of times before
            # bottoming out at a real input) AND avoids a real correctness gap: without this, the walk
            # frequently gives up at some not-yet-handled op type along that long chain and silently
            # falls back to the SAME "n_tokens" string a genuine batch=1 axis would never actually be,
            # corrupting whatever RESHAPE/RANGE_1D consumes this value. A genuinely non-1 batch axis
            # would surface as a numerical mismatch against the reference model, not a syntax error here.
            return "1"
        return self._infer_dynamic_dim_expr(real_var, torch_axis)

    def _resolve_scalar_expr(self, v, _seen=None):
        """
        General best-effort derivation of a SCALAR (0-d/1-element) Var's real symbolic value, walking
        through cast/squeeze aliasing and +-*/floor_div arithmetic over already-resolvable operands --
        needed for scalars computed via a real EXPRESSION over a gather-derived value (e.g.
        RelPositionalEncoding's `start_pos = center_pos - gather(shape(x), 1)`), which
        `_try_derive_gather_shape_value`'s narrower "exactly gather(shape(x), idx)" pattern match can't
        see through at all (it only recognizes gather itself, not gather wrapped in surrounding
        arithmetic). Confirmed needed on Conformer-CTC's positional-encoding table crop: `_resolve_range_
        scalar` alone left `start_pos`/`end_pos` (both real, resolvable expressions once cast/squeeze/sub
        are walked through) as `None`, causing `slice_by_index`'s own "resolve begin/end directly" case
        to give up on this axis entirely. Returns an int/float when every operand is a compile-time
        literal, else a string SymbolEnv expression, else None if any step can't be resolved.
        """
        if _seen is None:
            _seen = set()
        if v is None or not isinstance(v, Var):
            return None
        # NOT a linear-chain cycle guard (this function recurses into TWO operands per arithmetic op,
        # unlike `_infer_dynamic_dim_expr`'s single-input producer-chain walk) -- MIL/SSA graphs are
        # acyclic by construction (an op's inputs always name EARLIER-defined vars, never itself), so a
        # genuine infinite loop here is architecturally impossible. Treating "already visited" as a
        # failure was a real bug, not just defensive-and-harmless: a DIAMOND dependency (the same
        # upstream var reached via two different operand paths) is completely ordinary in an arithmetic
        # expression tree -- confirmed on VITS's `end = start + 2*length - 1`, where `start` and the
        # `2*length` term both independently reference the same `length`-derived `gather` var. The
        # SECOND reference used to hit `id(v) in _seen` and silently return None, even though nothing
        # about it was actually unresolvable -- `slice_by_index`'s "end" bound came back `None` and fell
        # back to the axis's full (unsliced) extent, corrupting the sliced relative-position table
        # (silently ~34x too long at a real T=62) while `begin` (whose OWN resolution never revisits
        # `gather_0` a second time) looked completely fine. `_seen` is kept (threaded through, unused)
        # rather than dropped from the signature, to keep every call site below unchanged.
        if getattr(v, "val", None) is not None:
            arr = np.asarray(v.val).reshape(-1)
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
            return self._resolve_scalar_expr(inner, _seen)
        if op.op_type == "select":
            # `torch.where(cond, a, b)` -- NeMo's own `get_seq_len` uses exactly this shape for its
            # "fix for seq_len = 0 for streaming" guard (`torch.where(seq_len == 0, zeros, seq_len_
            # unfixed)`). This exporter's target use (a real, single, non-empty utterance) never hits the
            # degenerate `cond` branch, so -- mirroring the "batch is always 1" style invariant used
            # throughout this file -- always take the "b" (false/else) branch rather than trying to
            # resolve `cond` itself.
            return self._resolve_scalar_expr(op.inputs.get("b"), _seen)
        if op.op_type == "gather":
            derived = self._try_derive_gather_shape_value(v)
            return derived
        _ARITH_OPS = {"add": "+", "sub": "-", "mul": "*", "real_div": "/", "floor_div": "/"}
        if op.op_type in _ARITH_OPS:
            x_e = self._resolve_scalar_expr(op.inputs.get("x"), _seen)
            y_e = self._resolve_scalar_expr(op.inputs.get("y"), _seen)
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

    def _resolve_range_scalar(self, v):
        """
        Resolves one `range_1d` start/end/step operand to either a derived symbolic expression (via
        `_try_derive_gather_shape_value`, then the more general `_resolve_scalar_expr`) or a literal
        constant, for use both when emitting a RANGE_1D JSON node's own attrs and when inferring the
        LENGTH of that range's own output elsewhere (see the "range_1d" case in
        `_infer_dynamic_dim_expr`) -- the two need the exact same resolution logic.
        """
        if v is None:
            return None
        derived = self._try_derive_gather_shape_value(v)
        if derived is not None:
            return derived
        if hasattr(v, "val") and v.val is not None:
            return float(np.asarray(v.val).reshape(-1)[0])
        return self._resolve_scalar_expr(v)

    def _resolve_slice_axis_value(self, idx_var, axis):
        """
        Resolves a `slice_by_index` op's "begin"/"end" input at one specific axis -- needed because that
        input can itself be dynamic (a `concat`/`stack` of per-axis gather-derived scalars, same
        structure as a `reshape`'s "shape" input) rather than a plain constant array. Used by BOTH the
        `_infer_dynamic_dim_expr`-level "slice_by_index" case above (deriving a symbolic axis's real
        expression for a DOWNSTREAM consumer) and the actual JSON-emitting `slice_by_index` translation
        branch (building the real VIEW node's own shape/offset) -- the two need the exact same
        resolution, and used to diverge: the translation branch only ever looked at a literal `.val`
        array, so whenever `begin`/`end` was a live concat (confirmed on the positional-encoding table
        crop, `self.pe[:, center - t + 1 : center + t]`, AND on `rel_shift`'s final `matrix_bd[..., :
        matrix_ac.size(-1)]` crop, whose `end` is a `concat` ending in a `gather` read off a DIFFERENT
        tensor's shape), it silently treated the WHOLE begin/end array as absent and fell back to
        copying the parent's own unsliced extent on every axis -- not just a shape-string cosmetic bug,
        the emitted VIEW's declared shape and its own stride math then genuinely disagreed on element
        count for whichever axis needed the real crop.
        """
        if idx_var is None:
            return None
        if getattr(idx_var, "val", None) is not None:
            arr = np.asarray(idx_var.val).reshape(-1)
            return int(arr[axis]) if axis < len(arr) else None
        if idx_var.op is not None and idx_var.op.op_type in ("concat", "stack"):
            values = idx_var.op.inputs.get("values")
            if values is not None and axis < len(values):
                resolved = self._resolve_range_scalar(values[axis])
                if isinstance(resolved, (int, float)):
                    return int(resolved)
                return resolved
        return None

    def _try_resolve_reshape_shape_input(self, op):
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
           `gather(shape(real_x), idx)`, possibly cast) -- resolve each one via `_resolve_range_scalar`.
           Needed for the common Q/K/V head-split idiom (`b, t, _ = x.shape; x.view(b, t, h, d)`): two
           genuinely different axes (batch and time) both collapsed to the same bare "n_tokens"
           substitution under the old out_info-based path, producing an invalid RESHAPE target with a
           repeated symbol.

        Returns None (falling back to the existing out_info-based path) unless every element resolves.
        """
        shape_var = op.inputs.get("shape")
        if shape_var is None:
            return None
        if getattr(shape_var, "val", None) is not None:
            raw = np.asarray(shape_var.val).reshape(-1)
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
            r = self._resolve_range_scalar(v)
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
                # atomic-export path's own inter-slice boundaries surface even more of these as separate
                # declared inputs of a single slice). There is no cheap, reliable way to tell "several
                # names, one true quantity" apart from "two genuinely independent dynamic axes" from the
                # dim strings alone -- CoreML doesn't expose symbol-equality at this level, and an
                # input-count-based heuristic tried here produced false positives on real, correct atomic
                # slices. If a future model genuinely needs a second independent dynamic axis, that would
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
                dim_str = str(dim)
                if _DYNAMIC_SYMBOL_RE.search(dim_str):
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
                    shape.append(inferred if inferred is not None else self._sub_symbol(dim_str))
                else:
                    shape.append(dim_str)
        
        # Loom expects fast-varying dimension first, so reverse standard shape order
        reversed_shape = list(reversed(shape))
        return {"name": name, "dtype": dtype, "shape": reversed_shape}

    def export(self):
        """
        Traverses the MIL program:
          - 'main' function becomes the embedded Lua driver script.
          - Other functions represent heavy submodules and become static topologies.
          - Weights and assets are serialized to GGUF.
        """
        is_bespoke = len(self.program.functions) > 1 and "main" in self.program.functions

        if not (is_bespoke and self.profile is None):
            # EXPORT-IMPROVEMENT-BACKLOG.md item 3: graph rewrites (currently just GQA repeat_kv()
            # fusion) run as real MIL->MIL passes over the pymil graph here, before any of the workflows
            # below ever walk it -- not interleaved into generate_graph_topology's own translation walk.
            # Skipped for the bespoke/advanced workflow below: that path exists specifically to accept
            # hand-built Programs (see test_compiler.py's MockOperation) that were never traced through
            # ct.convert()'s standard pipeline and may contain synthetic ops standing in for ones MIL
            # itself doesn't have -- a real MIL pass (dead_code_elimination in particular, which insists
            # on internally-consistent var/op child-tracking) isn't meaningful there and isn't safe to run
            # over a graph that was assembled by directly splicing Python op lists rather than through
            # MIL's own block-mutation API.
            apply_loom_mil_passes(self.program)

        if is_bespoke and self.profile is None:
            # 1. Advanced / Bespoke Exporting Workflow
            print("Exporting via Advanced/Bespoke workflow...")
            for func_name, func in self.program.functions.items():
                if func_name == "main":
                    self.transpile_to_lua(func, name="main")
                else:
                    self.topologies[func_name] = self.generate_graph_topology(func, func_name)
            driver_script = self._finalize_driver()
        elif self.kwargs.get("submodule_layout") is not None:
            # Submodule-export blueprint (EXPORT-IMPROVEMENT-BACKLOG.md item 2): `self.program` here is
            # NOT a single flattened trace -- it has no "main" function at all, just one Function per
            # independently-traced submodule (prefix/aux/layer_i/suffix_i) -- so there is no monolithic
            # fallback to degrade to on failure the way "atomic" has; a bug here should fail loudly, not
            # silently produce a working-looking but wrong export.
            self.apply_submodule_export()
            driver_script = self._finalize_driver()
        else:
            profile = self.profile or "monolithic"
            if profile == "atomic":
                try:
                    # Validation/codegen must run INSIDE this try block, not after it: atomic
                    # partitioning is a best-effort heuristic (scope-boundary guessing), and an IR that
                    # fails validation (e.g. a spurious/undefined subgraph input the heuristic
                    # mis-attributed to the wrong slice) is exactly the same class of "atomic partitioning
                    # didn't actually work" failure as an exception raised during partitioning itself --
                    # both should fall back to the monolithic profile rather than crashing the export.
                    self.apply_atomic_export()
                    driver_script = self._finalize_driver()
                except Exception as e:
                    print(f"Warning: Automated atomic partitioning failed: {e}. Falling back to monolithic profile.")
                    self.topologies = {}
                    self.ir_function = None
                    self.apply_monolithic_export()
                    driver_script = self._finalize_driver()
            else:
                self.apply_monolithic_export()
                driver_script = self._finalize_driver()

        # 3. Serialization Phase
        self.write_gguf(driver_script)
        return self.output_path

    def _finalize_driver(self) -> str:
        """Validates the built driver IR and codegens it to Lua source text."""
        validate(self.ir_function)
        check_subgraph_calls(self.ir_function, self.topologies)
        return "\n".join(LuaCodegen().emit_function(self.ir_function))

    def apply_monolithic_export(self):
        print("Exporting via Automatic Monolithic path...")
        main_func = self.program.functions["main"]
        self.topologies["main_topo"] = self.generate_graph_topology(main_func, "main_topo")

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

        body = []
        for name in main_func.inputs.keys():
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                # A traced model's own "cache_position"/"position_ids" input (see export_lfm2_*.py's own
                # comment: passing this explicitly rather than letting the model derive it internally
                # from a Python-level `.shape[1]` query is what keeps it genuinely dynamic under
                # torch.jit.trace) is host-computed here, not unpacked from the caller's `inputs` table --
                # the driver already knows n_tokens/n_past, and callers shouldn't need to know this is an
                # LFM2-specific implementation detail of the traced graph.
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            elif name in _CAUSAL_MASK_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))
            else:
                body.append(Local(safe_inp, BinOp("or", FieldAccess("inputs", safe_inp), FieldAccess("inputs", "tokens"))))

        inputs_tbl = {self.safe_name(k): IRVar(self.safe_name(k)) for k in main_func.inputs.keys()}

        body.append(SubgraphCall(
            outputs=["_mono_out"],
            extra_outputs=["_mono_shape"],
            module="main_topo",
            n_tokens=n_tokens_expr,
            n_past=Lit(0),
            inputs=inputs_tbl,
        ))

        # Argmax the logits row for the active (last real) token rather than returning the raw output
        # array -- _mono_shape[1] is the output's ne0 (vocab size), the same convention
        # transpile_operation's own "argmax" case and apply_atomic_export's final slice rely on.
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar("_mono_out")]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar("_mono_out"), Index(IRVar("_mono_shape"), 1), row_expr])])],
            else_=[Return([IRVar("_mono_out")])],
        ))

        self.ir_function = IRFunction("main", ["inputs"], body)

    def apply_atomic_export(self):
        print("Exporting via Automatic Atomic path...")
        import re
        main_func = self.program.functions["main"]
        operations = list(main_func.operations)
        
        # 1. Scope-Based Partitioning.
        #
        # Preferred signal: coremltools attaches the ORIGINAL PyTorch module hierarchy to every op
        # converted from a traced/exported model, under ScopeSource.TORCHSCRIPT_MODULE_NAME (e.g.
        # ('model', 'model', '5', 'self_attn') for `model.model.layers[5].self_attn`). A digit segment
        # that is NOT the tuple's last element marks indexing into a repeated submodule
        # (nn.ModuleList/Sequential) -- exactly the atomic layer boundary we want. Trailing digits are
        # NOT a boundary signal: they are just auto-generated per-op SSA name suffixes coremltools
        # invents when the original PyTorch value had no better name (e.g. a bare intermediate named
        # "1823"), and treating those as boundaries mis-partitions the graph into one slice per op
        # instead of one slice per real decoder layer.
        #
        # Fallback signal: hand-built MIL programs (e.g. this module's own unit tests) have no torch
        # scope metadata at all, so we fall back to the previous heuristic of regex-matching the op's
        # own output name for "layer_N"/"embed"/"output_head"-shaped names.
        try:
            from coremltools.converters.mil.mil.scope import ScopeSource
        except ImportError:
            ScopeSource = None

        def torch_scope_key(op):
            if ScopeSource is None or not hasattr(op, "scopes"):
                return None
            mn = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, ())
            if not mn:
                return None
            for i in range(len(mn) - 1):
                if mn[i].isdigit():
                    return tuple(mn[:i + 1])
            return tuple(mn[:-1]) if len(mn) > 1 else tuple(mn)

        def name_regex_key(op):
            op_name = op.outputs[0].name if op.outputs else op.name
            match = re.search(r'(layers?|blk|blocks?|modules?|linear|dense|fc|conv)_(\d+)', op_name, re.IGNORECASE)
            if match:
                return f"layer_{match.group(2)}"
            if any(x in op_name.lower() for x in ("embed", "emb", "wte")):
                return "embedding"
            if any(x in op_name.lower() for x in ("lm_head", "output_head", "logits", "pred", "output")):
                return "output_head"
            return None

        def boundary_key(op):
            return torch_scope_key(op) or name_regex_key(op)

        def label_for(key):
            if isinstance(key, str):
                return key
            if key[-1].isdigit():
                return f"layer_{key[-1]}"
            # Match on the LEAF scope segment only (the immediate submodule attribute name), and
            # exactly rather than by substring: a broad "'embed' in ..." match on the whole path
            # also matches LFM2's final-output RMSNorm, which the model confusingly calls
            # "embedding_norm" despite having nothing to do with the token-embedding lookup --
            # colliding both onto the same "embedding" label and silently discarding one topology.
            leaf = key[-1].lower()
            if leaf in ("embed_tokens", "embedding", "wte", "tok_embeddings"):
                return "embedding"
            if leaf in ("lm_head", "output_head", "logits"):
                return "output_head"
            return self.safe_name("_".join(key))

        # Seed the initial slice with the first identifiable boundary (ops before it, e.g. leading
        # const/cast setup with no scope opinion of their own, join that first slice).
        initial_key = None
        for op in operations:
            k = boundary_key(op)
            if k is not None:
                initial_key = k
                break

        slices = [] # list of tuples: (slice_name, ops_list)
        if initial_key is not None:
            current_key = initial_key
            current_ops = []
            for op in operations:
                k = boundary_key(op)
                if k is not None and k != current_key:
                    if current_ops:
                        slices.append((current_key, current_ops))
                    current_ops = []
                    current_key = k
                current_ops.append(op)
            if current_ops:
                slices.append((current_key, current_ops))

            # Fold metadata-only slices (e.g. precomputed rotary-embedding tables, which show up as
            # their own scope but are pure const/cast) forward into the next slice that actually
            # consumes them -- they have no compute/output of their own to serve as a standalone topology.
            merged = []
            pending = []
            for key, ops in slices:
                if all(op.op_type in ("const", "cast") for op in ops):
                    pending.extend(ops)
                    continue
                merged.append((key, pending + ops))
                pending = []
            if pending:
                if merged:
                    merged[-1] = (merged[-1][0], merged[-1][1] + pending)
                else:
                    merged = []
            slices = [(label_for(key), ops) for key, ops in merged]

        if len(slices) <= 1:
            raise ValueError("No distinct scope-based layer boundaries could be identified in the graph.")

        # 2. Extract inputs/outputs interfaces for each sliced topology
        # Replicate consumed constants locally in each slice to decouple them,
        # then extract only non-constant variable inputs.
        #
        # A SubgraphCall only ever exposes ONE slice's output(s) as `last_op.outputs` (see
        # `output_names = ... last_op.outputs` below) -- the single-output-per-topology convention
        # the driver/engine actually supports. So a var is only reachable by a LATER slice via
        # legitimate SubgraphCall input-wiring if its producer op is the designated LAST op of
        # whichever slice originally owns it (e.g. layer_(N-1)'s final hidden_states, threaded into
        # layer_N). Any op that is NOT its own slice's last op -- whether because it's genuinely
        # ungoverned (boundary_key is None, swept into whichever slice happened to be "current" in
        # iteration order purely by happenstance) or because it's a real but non-final interior op
        # of a shared multi-output slice (e.g. a RoPE cos/sin precompute under its own
        # "model_model_pos_emb" scope, where only ONE of the two sibling cos/sin ops can ever be the
        # slice's single exposed output) -- can NEVER be read this way. Any later slice that
        # references such a var's name directly sees it as an "external input" nothing upstream ever
        # actually provides: the mis-attribution bug, previously only caught (not fixed) by
        # validate()'s atomic->monolithic fallback, surfacing as a SubgraphCall reading an input no
        # earlier statement ever defined.
        #
        # Fix: recursively pull each such not-properly-exposed producer (and its own transitive
        # dependencies, stopping at consts or at another op that IS its slice's legitimate exposed
        # output) into EVERY slice that consumes it, instead of leaving it live in only the one slice
        # that happened to inherit it. This is safe to duplicate freely: every such op is a pure
        # function of consts/already-available inputs (that's exactly why it carries no real
        # cross-slice state of its own), so recomputing it per consuming slice is redundant compute,
        # never a correctness change. Any resulting now-unused copy left behind in the original
        # "accidental host" slice is harmless: item 3's `_prune_dead_nodes` already drops any node
        # unreachable from that topology's own declared output.
        exposed_ops = {ops[-1] for _, ops in slices if ops}

        def _is_legitimate_external_ref(op):
            return op in exposed_ops

        # Some MIL ops (e.g. "concat"/"stack") take a LIST of Vars under one input key (e.g.
        # `values`) rather than one Var per key -- generate_graph_topology's own input-extraction
        # already knows to flatten these (see its "elif isinstance(v, (list, tuple))" branch further
        # below); this partitioning code must walk the exact same shape or it silently never visits
        # a list-valued input's real producer op at all. Confirmed as a second, real bug this round:
        # without this, replicating the pos_emb cos/sin closure for an attention layer pulled in the
        # `concat((freqs, freqs))` node (reached via `cos`'s own bare-Var "x" input) but never its
        # OWN `freqs` producer (only reachable through `concat`'s list-valued `values` input), leaving
        # a node referencing a never-defined `freqs` var in the emitted topology.
        def _iter_input_vars(op):
            for v in op.inputs.values():
                if isinstance(v, Var):
                    yield v
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, Var):
                            yield item

        def _collect_replica_closure(op, ops_set, acc, visited):
            if op in ops_set or op in visited:
                return
            visited.add(op)
            for v in _iter_input_vars(op):
                if v.op is None or v in op.outputs:
                    continue
                producer = v.op
                if producer in ops_set or producer in visited:
                    continue
                if producer.op_type == "const":
                    if producer not in acc:
                        acc.append(producer)
                elif not _is_legitimate_external_ref(producer):
                    _collect_replica_closure(producer, ops_set, acc, visited)
                    if producer not in acc:
                        acc.append(producer)

        for idx, (name, ops) in enumerate(slices):
            ops_set = set(ops)
            extra = []
            visited = set()
            for op in ops:
                for v in _iter_input_vars(op):
                    if v.op and v not in op.outputs:
                        producer = v.op
                        if producer in ops_set:
                            continue
                        if producer.op_type == "const":
                            if producer not in extra:
                                extra.append(producer)
                        elif not _is_legitimate_external_ref(producer):
                            _collect_replica_closure(producer, ops_set, extra, visited)
                            if producer not in extra:
                                extra.append(producer)
            local_ops = (extra + list(ops)) if extra else list(ops)
            slices[idx] = (name, local_ops)

        slice_inputs = {}

        for name, ops in slices:
            # A var is an external input of THIS slice iff the op that produced it is not itself
            # part of this slice's own op list -- i.e. it's either a top-level function input (its
            # producing op is a placeholder, not in `ops`) or another slice's output. Checking
            # membership against a running "seen so far" set (as a previous version of this code did)
            # is wrong: it also catches purely-internal intermediates produced earlier in the SAME
            # slice, misclassifying them as external inputs (observed on LFM2: 30-60 bogus "inputs"
            # per layer, some with >4 MIL dims, crashing ggml_new_tensor at runtime).
            ops_set = set(ops)
            slice_in = {}
            for op in ops:
                for v in _iter_input_vars(op):
                    if v not in op.outputs and v.op not in ops_set:
                        if v.op and v.op.op_type == "const":
                            continue
                        slice_in[self.safe_name(v.name)] = v
            slice_inputs[name] = slice_in
            
        # 3. Generate topologies for all sliced sub-graphs
        for name, ops in slices:
            inputs_dict = slice_inputs[name]
            self.topologies[name] = self.generate_graph_topology(None, name, ops_list=ops, inputs_dict=inputs_dict)
            
        # 4. Synthesize the automatic looping driver IR
        first_input = "tokens"
        feature_scale = 1
        if main_func.inputs:
            first_input_var = list(main_func.inputs.values())[0]
            first_input = self.safe_name(list(main_func.inputs.keys())[0])
            if hasattr(first_input_var, "shape") and len(first_input_var.shape) == 3:
                try:
                    feature_scale = int(first_input_var.shape[2])
                except (ValueError, TypeError):
                    pass

        n_tokens_expr = Len(first_input)
        if feature_scale > 1:
            n_tokens_expr = BinOp("floordiv", Len(first_input), Lit(feature_scale))

        body = []
        for name in main_func.inputs.keys():
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                # See apply_monolithic_export's identical case: host-computed, not caller-supplied.
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            elif name in _CAUSAL_MASK_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))
            else:
                body.append(Local(safe_inp, BinOp("or", FieldAccess("inputs", safe_inp), FieldAccess("inputs", "tokens"))))

        for idx, (name, ops) in enumerate(slices):
            inputs_dict = slice_inputs[name]

            # Map input keys, standardizing the first input name to "hidden_states" for decoder layers
            is_layer = name.startswith("layer_")
            first_key = list(inputs_dict.keys())[0] if inputs_dict else None

            slice_inputs_tbl = {}
            for k in inputs_dict.keys():
                safe_k = self.safe_name(k)
                if is_layer and k == first_key:
                    slice_inputs_tbl["hidden_states"] = IRVar(safe_k)
                else:
                    slice_inputs_tbl[safe_k] = IRVar(safe_k)

            last_op = ops[-1]
            output_names = [self.safe_name(v.name) for v in last_op.outputs]

            if idx == len(slices) - 1:
                # Final slice: also capture the output shape so the driver can argmax the last
                # sequence position's logits row instead of returning a raw output value (matching
                # the "argmax" convention transpile_operation's bespoke path and
                # apply_monolithic_export both use for causal-LM next-token generation).
                body.append(SubgraphCall(
                    outputs=output_names, extra_outputs=["_atomic_final_shape"],
                    module=name, n_tokens=n_tokens_expr, n_past=Lit(0), inputs=slice_inputs_tbl,
                ))
            else:
                body.append(SubgraphCall(
                    outputs=output_names, module=name, n_tokens=n_tokens_expr, n_past=Lit(0), inputs=slice_inputs_tbl,
                ))

        final_last_op = slices[-1][1][-1]
        final_output_names = [self.safe_name(v.name) for v in final_last_op.outputs]
        final_out = final_output_names[0]
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar(final_out)]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar(final_out), Index(IRVar("_atomic_final_shape"), 1), row_expr])])],
            else_=[Return([IRVar(final_out)])],
        ))

        self.ir_function = IRFunction("main", ["inputs"], body)

    def apply_submodule_export(self):
        """
        Synthesizes the driver for a submodule-export blueprint (`kwargs["submodule_layout"]`, a
        `SubmoduleExportResult` from submodule_export.py): one real, independently-traced Function per
        prefix/aux/layer_i/suffix_i submodule. Every function is self-contained by construction (no
        cross-slice variable leakage to detect, unlike apply_atomic_export's scope-partitioned single
        trace), so this only has to generate each function's topology directly
        (`generate_graph_topology(func, name)`, no ops_list/inputs_dict reconstruction) and chain
        SubgraphCalls prefix -> [aux] -> layer_0..N-1 -> suffix_0..M-1 -> argmax.
        """
        print("Exporting via Submodule-Blueprint path...")
        layout = self.kwargs["submodule_layout"]
        functions = self.program.functions
        special_names = set(_POSITION_INPUT_NAMES) | set(_CAUSAL_MASK_INPUT_NAMES)

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

        # 2. `first_input` (the caller-supplied token-ids input) must be defined before anything below
        # reads n_tokens_expr, mirroring apply_monolithic_export/apply_atomic_export's own ordering.
        body = [Local(first_input, BinOp("or", FieldAccess("inputs", first_input), FieldAccess("inputs", "tokens")))]

        special_needed = set()
        for name in stage_names:
            special_needed.update(n for n in declared_inputs(name) if n in special_names)
        for name in sorted(special_needed):
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            else:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))

        # 3. Prefix.
        chain_var = "_sub_chain_0"
        body.append(SubgraphCall(
            outputs=[chain_var], module="prefix", n_tokens=n_tokens_expr, n_past=Lit(0),
            inputs={self.safe_name(n): IRVar(self.safe_name(n)) for n in prefix_input_names},
        ))

        # 4. Auxiliary submodule (computed once, shared across every repeated-block call below) -- e.g.
        # LFM2's rotary-embedding table, computed once in Lfm2Model.forward and threaded into every
        # decoder layer as `position_embeddings=(cos, sin)`.
        aux_out_vars = None
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
                aux_inputs_tbl[safe_n] = IRVar(chain_var) if n in aux_chain_names else IRVar(safe_n)
            aux_out_vars = [f"_sub_aux_{i}" for i in range(len(layout.aux_output_names))]
            body.append(SubgraphCall(
                outputs=aux_out_vars, module="aux", n_tokens=n_tokens_expr, n_past=Lit(0), inputs=aux_inputs_tbl,
            ))

        # 5. Repeated block, threading `chain_var` (hidden_states) from one layer's output into the
        # next's input, exactly like apply_atomic_export's own layer chain. Each layer's OWN declared
        # inputs are consulted independently (see the comment on step 1).
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
                    inputs_tbl[safe_n] = IRVar(chain_var)
                elif is_aux_input(n):
                    idx = 0 if n == layout.aux_kwarg else int(n[len(layout.aux_kwarg) + 1:])
                    inputs_tbl[safe_n] = IRVar(aux_out_vars[idx])
                else:
                    inputs_tbl[safe_n] = IRVar(safe_n)

            next_chain_var = f"_sub_chain_{i + 1}"
            body.append(SubgraphCall(
                outputs=[next_chain_var], module=layer_name, n_tokens=n_tokens_expr, n_past=Lit(0),
                inputs=inputs_tbl,
            ))
            chain_var = next_chain_var

        # 6. Suffix chain (e.g. final norm + lm_head).
        for idx, name in enumerate(layout.suffix_names):
            in_names = declared_inputs(name)
            if len(in_names) != 1:
                raise ValueError(f"suffix submodule '{name}' must declare exactly one input, got {in_names}")
            is_last = idx == len(layout.suffix_names) - 1
            next_chain_var = f"_sub_suffix_{idx}"
            call_kwargs = dict(
                outputs=[next_chain_var], module=name, n_tokens=n_tokens_expr, n_past=Lit(0),
                inputs={self.safe_name(in_names[0]): IRVar(chain_var)},
            )
            if is_last:
                call_kwargs["extra_outputs"] = ["_submodule_final_shape"]
            body.append(SubgraphCall(**call_kwargs))
            chain_var = next_chain_var

        # 7. Argmax the logits row for the active (last real) token -- same convention
        # apply_monolithic_export/apply_atomic_export use for causal-LM next-token generation.
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar(chain_var)]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar(chain_var), Index(IRVar("_submodule_final_shape"), 1), row_expr])])],
            else_=[Return([IRVar(chain_var)])],
        ))

        self.ir_function = IRFunction("main", ["inputs"], body)

    def transpile_to_lua(self, func: Function, name="main"):
        """
        Transpiles the main MIL orchestration function to a Lua JIT driver script.
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

        self.ir_function = IRFunction(name, ["inputs"], body)

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
            return [SubgraphCall(outputs=output_names, module=op_type, n_tokens=n_tokens_expr, n_past=n_past_expr,
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
            if shape_var is None or not hasattr(shape_var, "val") or shape_var.val is None:
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
                mean = float(mean_var.val) if mean_var is not None and hasattr(mean_var, "val") and mean_var.val is not None else 0.0
                stddev = float(stddev_var.val) if stddev_var is not None and hasattr(stddev_var, "val") and stddev_var.val is not None else 1.0
                if mean != 0.0 or stddev != 1.0:
                    raise NotImplementedError(
                        f"random_normal op '{op.name}' has mean={mean}/stddev={stddev} -- this exporter "
                        "only supports the standard N(0,1) case loom.gaussian_array itself draws "
                        "(compose an explicit MUL/ADD after it for any other mean/stddev)."
                    )
                return [Local(output_names[0], Call("loom.gaussian_array", [Lit(n)]))]

            low_var, high_var = op.inputs.get("low"), op.inputs.get("high")
            low = float(low_var.val) if low_var is not None and hasattr(low_var, "val") and low_var.val is not None else 0.0
            high = float(high_var.val) if high_var is not None and hasattr(high_var, "val") and high_var.val is not None else 1.0
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
        nodes = []
        topo_inputs = []
        aliases = {}
        
        inputs = inputs_dict if inputs_dict is not None else (func.inputs if func else {})
        operations = ops_list if ops_list is not None else (func.operations if func else [])

        # EXPORT-IMPROVEMENT-BACKLOG.md item 4: an "lstm" op can't be translated into ordinary static
        # topology nodes the way every other op_type below is -- ggml has no native LSTM/GRU op, and
        # unlike e.g. `linear`/`matmul`, correct recurrence needs a genuine host-side per-timestep loop
        # (see recurrent.py's own module docstring and tools/loom_mil_compiler/test_recurrent.py's
        # verified topology-generation logic), not a fixed sequence of graph nodes. `generate_graph_topology`
        # only ever returns ONE static topology for the whole `operations` list it's given; splitting a
        # function at an "lstm" op boundary into "pre-LSTM topology -> recurrent stepper call ->
        # post-LSTM topology" driver segments -- the way apply_atomic_export splits at torch-scope
        # boundaries -- is real, unimplemented follow-up work, not something this call can do safely.
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
                "profiles (monolithic/atomic/submodule-blueprint driver synthesis) is unimplemented "
                "follow-up work -- for now, build the per-timestep topologies directly via "
                "recurrent.build_lstm_cell_topologies() and drive them with loom.run_recurrent() in a "
                "hand-written driver script, the same way tools/convert_kokoro/kokoro_driver.lua does "
                "today via BiLstmStepper."
            )

        def resolve(name):
            seen = set()
            while name in aliases and name not in seen:
                seen.add(name)
                name = aliases[name]
            return name
        
        # Track inputs to the submodule and standardize the first input name to "hidden_states" for decoder layers
        first_input_var = None
        for name, var in inputs.items():
            if first_input_var is None:
                first_input_var = var
                
        if first_input_var is not None:
            orig_name = self.safe_name(first_input_var.name)
            
            if func_name.startswith("layer_"):
                # A submodule traced standalone with its real parameter name already literally
                # "hidden_states" (e.g. the submodule-export blueprint, EXPORT-IMPROVEMENT-BACKLOG.md
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
            if op_type == "const":
                val = op.val.val
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
                            axis = int(axis_var.val) if axis_var is not None and hasattr(axis_var, "val") and axis_var.val is not None else 0
                            if shape_vec_var is not None and shape_vec_var.shape is not None and axis < len(shape_vec_var.shape):
                                dim = shape_vec_var.shape[axis]
                                if isinstance(dim, (int, np.integer)):
                                    arr = np.where(arr < 0, arr + int(dim), arr)
                        break
                    val = arr

                # For monolithic profiles, skip namespace prefixing
                if func_name == "main_topo" or self.profile == "monolithic":
                    namespaced_name = weight_name
                else:
                    namespaced_name = f"{func_name}.{weight_name}"

                # Safe compaction to satisfy GGUF's GGML_MAX_NAME (64 chars) limit
                if len(namespaced_name) >= 64:
                    import hashlib
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
                continue

            if op_type == "cast":
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
                continue

            if op_type == "linear":
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
                continue

            if op_type == "matmul":
                # MIL's matmul(x, y, transpose_x, transpose_y) computes X @ Y where X = x^T if
                # transpose_x else x, Y = y^T if transpose_y else y (batched over leading dims). This is
                # NOT the same op as ggml_mul_mat(A, B), which always contracts over ne0 of both operands
                # and returns ne=[A.ne1, B.ne1, ...] -- i.e. it computes B_mat @ A_mat^T, not A_mat @
                # B_mat. Forwarding MIL's x/y straight through as ggml_mul_mat(x, y) silently produces a
                # transposed-but-same-shape (for square attention scores) or outright wrong-axis result,
                # exactly the numerical-correctness bug tracked in EXPORT-BACKLOG.md item 1 -- confirmed
                # by bisecting the real attention-score matmul (transpose_y=True) and the
                # scores@value matmul (transpose_x=transpose_y=False) against HF's own SDPA inputs.
                #
                # Both combinations used by scaled_dot_product_attention's decomposition are handled
                # explicitly below (derived from ggml_mul_mat's result.ne=[A.ne1,B.ne1,B.ne2,B.ne3]
                # formula); any other combination is intentionally unsupported rather than silently wrong.
                x_var_obj = op.inputs["x"]
                y_var_obj = op.inputs["y"]
                x_var = self.safe_name(x_var_obj.name)
                y_var = self.safe_name(y_var_obj.name)
                output_var = self.safe_name(op.outputs[0].name)

                tx_var = op.inputs.get("transpose_x")
                ty_var = op.inputs.get("transpose_y")
                tx = bool(tx_var.val) if tx_var is not None and hasattr(tx_var, "val") else False
                ty = bool(ty_var.val) if ty_var is not None and hasattr(ty_var, "val") else False

                if not tx and ty:
                    # X @ Y^T: both operands already share ne0 (the contracted/embedding axis) in their
                    # natural layout, so this is a straight ggml_mul_mat(y, x) -- key-first, matching the
                    # llama.cpp attention-score convention.
                    nodes.append({
                        "op": "MUL_MAT",
                        "inputs": [resolve(y_var), resolve(x_var)],
                        "outputs": [output_var]
                    })
                elif not tx and not ty:
                    # X @ Y: Y needs its leading two ne axes swapped (and made contiguous) before it can
                    # be used as ggml_mul_mat's first ("A") operand -- see the derivation in the comment
                    # above. Composed as PERMUTE + CONT so the C++ side never has to guess this from
                    # shapes alone.
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
                else:
                    raise NotImplementedError(
                        f"matmul op '{op.name}' has transpose_x={tx}, transpose_y={ty}, which no "
                        "exporter composition handles yet (only transpose_x=False has been needed so far)."
                    )
                continue

            if op_type == "gelu":
                # MIL's `gelu` carries an extra "mode" string input (PyTorch's `approximate=` arg,
                # "EXACT" or "TANH") the generic OP_MAP fallback below would otherwise add as a second
                # (bogus, string-typed) ggml node input -- first hit by VITS's DDSConv (`F.gelu(y)`, no
                # `approximate=` -> PyTorch's own default "none"/exact). ggml's own GELU primitive
                # (op_gelu, src/ops/primitives_basic.cpp) always computes the EXACT erf formula
                # (`ggml_gelu_erf`, chosen there for reproducibility over the tanh/sigmoid lookup-table
                # approximations) -- correct for this case, but reject rather than silently mismatch if a
                # future model traces the "TANH" approximate variant instead.
                mode_var = op.inputs.get("mode")
                mode = (mode_var.val if mode_var is not None and hasattr(mode_var, "val")
                        and mode_var.val is not None else "EXACT")
                x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                if str(mode).upper() in ("EXACT", "NONE"):
                    nodes.append({
                        "op": "GELU",
                        "inputs": [resolve(self.safe_name(x_var_obj.name))],
                        "outputs": [self.safe_name(op.outputs[0].name)],
                    })
                    continue
                if str(mode).upper() not in ("TANH_APPROXIMATION", "TANH"):
                    raise NotImplementedError(
                        f"gelu op '{op.name}' has mode={mode!r} -- ggml's GELU primitive only computes "
                        "the exact erf formula; only EXACT and TANH_APPROXIMATION are composed here."
                    )
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
                one_name = "gelu_tanh_approx.one" if (func_name == "main_topo" or self.profile == "monolithic") \
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
                continue

            if op_type == "leaky_relu":
                # HiFi-GAN vocoder's own activation (Generator/ResBlock2, real slope=0.1/0.01) -- ggml's
                # LEAKY_RELU primitive already exists (src/ops/primitives_basic.cpp) and just needed
                # wiring: MIL's `leaky_relu` op names its slope input "alpha", but op_leaky_relu reads
                # the JSON attr key "slope" specifically.
                x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                alpha_var = op.inputs.get("alpha")
                slope = (float(alpha_var.val) if alpha_var is not None and hasattr(alpha_var, "val")
                         and alpha_var.val is not None else 0.01)
                nodes.append({
                    "op": "LEAKY_RELU",
                    "inputs": [resolve(self.safe_name(x_var_obj.name))],
                    "outputs": [self.safe_name(op.outputs[0].name)],
                    "attrs": {"slope": slope},
                })
                continue

            if op_type == "reverse":
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
                axes_val = (list(axes_var.val) if axes_var is not None and hasattr(axes_var, "val")
                            and axes_var.val is not None else None)
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
                if func_name == "main_topo" or self.profile == "monolithic":
                    idx_full = idx_name
                else:
                    idx_full = f"{func_name}.{idx_name}"
                self.weights[idx_full] = np.arange(axis_size - 1, -1, -1, dtype=np.int32)

                nodes.append({
                    "op": "GET_ROWS",
                    "inputs": [resolve(x_var), idx_full],
                    "outputs": [output_var],
                })
                continue

            if op_type == "loom_group_norm":
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
                    continue
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
                n_groups_val = int(n_groups_obj.val) if n_groups_obj is not None and hasattr(n_groups_obj, "val") else None
                eps_val = float(eps_obj.val) if eps_obj is not None and hasattr(eps_obj, "val") else 1e-5
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
                continue

            if op_type == "loom_spline":
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
                if func_name == "main_topo" or self.profile == "monolithic":
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
                continue

            if op_type == "split":
                # Compose split as multiple zero-copy VIEW slices
                x_var = self.safe_name(op.inputs["x"].name)
                axis = op.inputs["axis"].val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
                
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
                else:
                    split_dim_size = f"({dim_to_split} / {num_splits})"
                    
                # Create a VIEW node for each split output
                for idx, out_var in enumerate(op.outputs):
                    out_name = self.safe_name(out_var.name)
                    
                    slice_shape = list(ne_shape)
                    slice_shape[ne_axis] = split_dim_size
                    
                    # Calculate byte offset rule
                    offset_elements = f"{idx} * {split_dim_size}"
                    for prev_ax in range(ne_axis):
                        offset_elements = f"({offset_elements} * {ne_shape[prev_ax]})"
                    offset_bytes = f"({offset_elements} * 4)" # 4 bytes per float element
                    
                    nodes.append({
                        "op": "VIEW",
                        "inputs": [resolve(x_var)],
                        "outputs": [out_name],
                        "attrs": {
                            "shape": slice_shape,
                            "offset": offset_bytes
                        }
                    })
                continue

            if op_type == "slice_by_index":
                # Compose slice_by_index as an optimized zero-copy VIEW node
                x_var = self.safe_name(op.inputs["x"].name)
                output_var = self.safe_name(op.outputs[0].name)

                begin_var = op.inputs.get("begin")
                end_var = op.inputs.get("end")
                begin_mask = op.inputs["begin_mask"].val if "begin_mask" in op.inputs and hasattr(op.inputs["begin_mask"], "val") else None
                end_mask = op.inputs["end_mask"].val if "end_mask" in op.inputs and hasattr(op.inputs["end_mask"], "val") else None

                x_info = self.get_var_info(op.inputs["x"])
                ne_shape = x_info["shape"]
                rank = len(ne_shape)

                begin_mask_list = list(begin_mask) if isinstance(begin_mask, (list, tuple, np.ndarray)) else None
                end_mask_list = list(end_mask) if isinstance(end_mask, (list, tuple, np.ndarray)) else None

                # Resolve each MIL-order axis's real (begin, end) via `_resolve_slice_axis_value` (which
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
                    b_val = 0 if is_begin_masked else self._resolve_slice_axis_value(begin_var, mil_axis)
                    e_val = dim_size if is_end_masked else self._resolve_slice_axis_value(end_var, mil_axis)
                    if b_val is None:
                        b_val = 0
                    elif isinstance(b_val, (int, np.integer)) and b_val < 0:
                        b_val = (dim_size + int(b_val)) if isinstance(dim_size, int) else f"({dim_size} + ({int(b_val)}))"
                    if e_val is None:
                        e_val = dim_size
                    elif isinstance(e_val, (int, np.integer)) and e_val < 0:
                        e_val = (dim_size + int(e_val)) if isinstance(dim_size, int) else f"({dim_size} + ({int(e_val)}))"
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
                    elif b_val == 0 and str(e_val) == str(dim_size):
                        slice_shape.append(dim_size)
                    else:
                        slice_shape.append(f"({e_val} - {b_val})")

                # Calculate byte offset in C-major MIL layout mapping to ne_shape strides. Uses
                # `resolved_begin` (mask-aware, negative-index-normalized) rather than the raw
                # `begin_list` for the same reason the shape derivation above needs it: an
                # ignored/negative begin must contribute its real (0 or normalized) value, not its raw
                # MIL-op placeholder.
                offset_elements = "0"
                for i in range(rank):
                    b_val = resolved_begin[i]
                    if not (isinstance(b_val, int) and b_val == 0):
                        stride_product = "1"
                        ne_limit = rank - 1 - i
                        for prev_ax in range(ne_limit):
                            stride_product = f"({stride_product} * {ne_shape[prev_ax]})"
                        offset_elements = f"({offset_elements} + ({b_val} * {stride_product}))"

                offset_bytes = f"({offset_elements} * 4)"
                
                nodes.append({
                    "op": "VIEW",
                    "inputs": [resolve(x_var)],
                    "outputs": [output_var],
                    "attrs": {
                        "shape": slice_shape,
                        "offset": offset_bytes
                    }
                })
                continue

            if op_type == "fill":
                # Compile-time evaluation of constant fill tensors
                shape_val = op.inputs["shape"].val if "shape" in op.inputs and hasattr(op.inputs["shape"], "val") else None
                value_val = op.inputs["value"].val if "value" in op.inputs and hasattr(op.inputs["value"], "val") else 0.0
                
                if shape_val is not None:
                    shape_list = list(shape_val) if isinstance(shape_val, (list, tuple, np.ndarray)) else [shape_val]
                    ne_shape = list(reversed(shape_list))
                    
                    array = np.full(ne_shape, value_val, dtype=np.float32)
                    weight_name = self.safe_name(op.outputs[0].name)
                    self.weights[weight_name] = array
                    continue
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
                    # `_try_resolve_reshape_shape_input` already provides for `reshape`, reused here
                    # since a dynamic `fill`'s "shape" input has the exact same concat-of-gathers/
                    # constant-array structure) -- `get_var_info`'s out_info-based fallback below has no
                    # per-axis correspondence formula for `fill` (not in `_infer_dynamic_dim_expr`'s
                    # handled op set), so it blindly substitutes EVERY symbolic axis to "n_tokens".
                    # Confirmed wrong on Conformer-CTC's length-validity mask fill: a rank-2 `[T, T]`
                    # fill (T = the subsampled frame count) got BOTH axes collapsed to the raw sample
                    # count instead, producing a wildly oversized mask.
                    resolved_torch_shape = self._try_resolve_reshape_shape_input(op)
                    if resolved_torch_shape is not None:
                        target_shape = list(reversed(resolved_torch_shape))
                    else:
                        out_info = self.get_var_info(op.outputs[0])
                        target_shape = list(out_info["shape"])
                    rank = len(target_shape)

                    weight_name = self.safe_name(op.outputs[0].name) + "_fill_scalar"
                    if func_name == "main_topo" or self.profile == "monolithic":
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
                    continue

            if op_type == "pad":
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
                if pad_var is None or not hasattr(pad_var, "val") or pad_var.val is None:
                    raise NotImplementedError(
                        f"pad op '{op.name}' has a non-constant 'pad' input, which this exporter doesn't support."
                    )
                pad_vals = [int(v) for v in pad_var.val]
                if len(pad_vals) % 2 != 0:
                    raise NotImplementedError(f"pad op '{op.name}' has an odd-length 'pad' array {pad_vals!r}.")
                n_padded = len(pad_vals) // 2

                if not hasattr(x_var_obj, "shape") or x_var_obj.shape is None:
                    raise NotImplementedError(f"pad op '{op.name}' has an input with no known rank.")
                rank = len(x_var_obj.shape)

                mode_var = op.inputs.get("mode")
                mode = (mode_var.val if mode_var is not None and hasattr(mode_var, "val")
                        and mode_var.val is not None else "constant")

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
                    constant_val = (float(constant_val_var.val)
                                     if constant_val_var is not None and hasattr(constant_val_var, "val")
                                     and constant_val_var.val is not None else 0.0)
                    if constant_val != 0.0:
                        raise NotImplementedError(
                            f"pad op '{op.name}' has mode='constant' with a non-zero constant_val="
                            f"{constant_val} -- PAD_1D only supports zero-fill."
                        )
                    mapped_op = "PAD_1D"
                elif mode == "reflect":
                    mapped_op = "PAD_1D_REFLECT"
                elif mode == "replicate":
                    # SupertonicTTS's ConvNextBlock (used by EVERY encoder/decoder in that model) pads via
                    # `nn.functional.pad(x, pad, mode="replicate")` before every depthwise conv -- ggml has
                    # no native replicate/edge-pad kernel (unlike PAD_1D/PAD_1D_REFLECT, which wrap real
                    # ggml_pad_ext/ggml_pad_reflect_1d primitives), so this composes it purely from
                    # already-existing primitives instead of adding a new C++ op: VIEW out the single
                    # boundary column (ne[0]=1, full ne[1:]), REPEAT-broadcast it to the pad width (both are
                    # already used together for exactly this "materialize the broadcast, then CONCAT"
                    # pattern -- see REPEAT's own docstring, built for StyleTTS2's diffusion sampler), then
                    # CONCAT it onto the appropriate side. lp0/rp0 are always static (kernel_size/dilation
                    # are architecture constants), so only the VIEW extracting the RIGHT edge needs a
                    # dynamic offset (the left edge is always byte 0) -- reuses the same
                    # `_infer_dynamic_dim_expr` backward walk every other dynamic-offset VIEW in this
                    # exporter already depends on, rather than any new shape-inference machinery.
                    if lp0 == 0 and rp0 == 0:
                        aliases[output_var] = resolve(x_var)
                        continue
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
                            "attrs": {"shape": [1, *ne_rest], "offset": f"(({t_expr} - 1) * 4)"},
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
                        aliases[output_var] = cur
                    continue
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
                continue

            if op_type == "band_part":
                # Map band_part (with lower=-1, upper=0) to DIAG_MASK_ZERO. MIL's actual input keys
                # are "lower"/"upper" (see tensor_operation.py's band_part InputSpec) -- NOT
                # "num_lower"/"num_upper", which never matched, so this always silently used the
                # (coincidentally causal-shaped) -1/0 defaults regardless of the op's real attrs. Same
                # bug class as the transpose/"perm" mismatch above; not currently exercised by LFM2
                # (its causal mask gets constant-folded at trace time rather than computed via a live
                # band_part op), but a real latent bug for any model that reaches this path.
                num_lower = op.inputs["lower"].val if "lower" in op.inputs and hasattr(op.inputs["lower"], "val") else -1
                num_upper = op.inputs["upper"].val if "upper" in op.inputs and hasattr(op.inputs["upper"], "val") else 0
                
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
                continue

            if op_type == "transpose":
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
                if perm_var is None or not hasattr(perm_var, "val"):
                    raise ValueError(f"transpose op '{op.name}' has no resolvable 'perm' constant")
                # perm entries may be negative (confirmed on LFM2: e.g. [0, -1, -2] for .transpose(-1,-2)).
                raw_perm = [int(p) for p in perm_var.val]
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
                continue

            if op_type == "tile":
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
                reps = op.inputs["reps"].val if "reps" in op.inputs and hasattr(op.inputs["reps"], "val") else [1]
                
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
                        target_shape.append(f"({dim_size} * {rep_factor})")
                        
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
                continue

            if op_type == "squeeze":
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
                if axes_var is not None and hasattr(axes_var, "val") and axes_var.val is not None:
                    torch_axes = [int(a) for a in axes_var.val]
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
                continue

            if op_type in ["reshape", "expand_dims"]:
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

                resolved_torch_shape = self._try_resolve_reshape_shape_input(op) if op_type == "reshape" else None
                if resolved_torch_shape is not None and len(resolved_torch_shape) == out_rank:
                    # The "shape" input's own real per-axis values resolved directly -- strictly more
                    # trustworthy than either branch below (both of which only ever look at the output
                    # var's OWN, possibly-fresh-and-unrelated, symbolic shape). See
                    # `_try_resolve_reshape_shape_input`'s docstring for why that matters.
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
                continue

            if op_type == "concat":
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
                        
                        axis = op.inputs.get("axis").val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
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
                        continue
                    elif len(inputs) == 2:
                        axis = op.inputs.get("axis").val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
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
                        continue
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
                        continue
                continue

            if op_type == "stack":
                # MIL `stack(values, axis)` joins N same-shape tensors along a genuinely NEW axis
                # (unlike `concat`, which joins along an EXISTING one) -- e.g. a hand-rolled
                # conv-based STFT's real/imag parts, `torch.stack([real, imag], dim=-1)`, seen when a
                # model computes its DFT via CONV_1D kernels directly rather than `torch.stft` (which
                # decomposes differently, via coremltools' own `lower_complex_dialect_ops`). No new
                # ggml primitive needed: compose as RESHAPE (insert a size-1 axis) on each operand, then
                # the same CONCAT-along-that-axis this file already emits for `concat`.
                values_obj = op.inputs.get("values")
                axis_val = int(op.inputs["axis"].val) if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
                out_var = op.outputs[0]
                out_rank = len(self.get_var_info(out_var)["shape"])
                axis = axis_val + out_rank if axis_val < 0 else axis_val
                ne_axis = out_rank - 1 - axis

                reshaped = []
                for i, item in enumerate(values_obj):
                    if not isinstance(item, Var):
                        continue
                    v_name = resolve(self.safe_name(item.name))
                    v_shape = list(self.get_var_info(item)["shape"])
                    new_shape = v_shape[:ne_axis] + ["1"] + v_shape[ne_axis:]
                    unsq_name = f"{self.safe_name(out_var.name)}_stack_unsq_{i}"
                    nodes.append({
                        "op": "RESHAPE",
                        "inputs": [v_name],
                        "outputs": [unsq_name],
                        "attrs": {"shape": new_shape}
                    })
                    reshaped.append(unsq_name)

                if len(reshaped) < 2:
                    if reshaped:
                        aliases[self.safe_name(out_var.name)] = reshaped[0]
                    continue
                prev_output = reshaped[0]
                for i in range(1, len(reshaped) - 1):
                    inter_output = f"{self.safe_name(out_var.name)}_stack_temp_{i}"
                    nodes.append({
                        "op": "CONCAT",
                        "inputs": [prev_output, reshaped[i]],
                        "outputs": [inter_output],
                        "attrs": {"dim": ne_axis}
                    })
                    prev_output = inter_output
                nodes.append({
                    "op": "CONCAT",
                    "inputs": [prev_output, reshaped[-1]],
                    "outputs": [self.safe_name(out_var.name)],
                    "attrs": {"dim": ne_axis}
                })
                continue

            if op_type == "reduce_mean":
                # Dedicated branch (not the generic single-input OP_MAP path, which maps "reduce_mean"
                # straight to the raw "MEAN" primitive -- `ggml_mean` unconditionally reduces ne[0] ONLY,
                # silently wrong whenever the real reduction axis isn't ne[0]). First hit by Matcha-TTS's
                # own hand-rolled `text_encoder.py::LayerNorm` (glow-tts-derived, NOT `nn.LayerNorm`):
                # `torch.mean(x, 1, keepdim=True)` on a (B,C,T) tensor reduces the CHANNEL axis (torch
                # axis 1), which under this exporter's axis-reversal convention is ne[1], not ne[0] --
                # confirmed via a real end-to-end numeric mismatch (encoder_mu/encoder_logw both wildly
                # wrong, traced back to this exact LayerNorm's own mean/variance being computed over the
                # wrong axis). Composed as REDUCE_SUM (a real, axis-aware ggml primitive, same one
                # "reduce_sum" itself uses below) over the real ne-order axis, then SCALE by 1/N -- valid
                # generally (not just for this one case) since REDUCE_SUM already handles ne[0] correctly
                # too, so this is a strict generalization, not a special case, of every previously-working
                # reduce_mean usage (STFT/CMVN, all of which happen to reduce ne[0]).
                x_var_obj = op.inputs.get("x")
                axes_obj = op.inputs.get("axes")
                keep_dims_obj = op.inputs.get("keep_dims")
                if x_var_obj is None:
                    continue
                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                axes_val = axes_obj.val if axes_obj is not None and hasattr(axes_obj, "val") else None
                if axes_val is None or len(axes_val) != 1:
                    raise NotImplementedError(
                        f"reduce_mean op '{op.name}': only a single reduction axis is supported "
                        f"(got axes={axes_val!r}); a genuine multi-axis case (e.g. GroupNorm, see "
                        "group_norm_op.py) needs its own composition."
                    )
                axis = int(axes_val[0])
                if axis < 0:
                    axis += in_rank
                ne_axis = in_rank - 1 - axis
                keep_dims_val = bool(keep_dims_obj.val) if keep_dims_obj is not None and hasattr(keep_dims_obj, "val") else False
                x_shape = self.get_var_info(x_var_obj)["shape"]
                n_raw = x_shape[ne_axis]
                if not str(n_raw).lstrip("-").isdigit():
                    raise NotImplementedError(
                        f"reduce_mean op '{op.name}': reduction axis size ({n_raw!r}) must be a static "
                        "architecture constant -- a genuinely dynamic reduction count needs its own "
                        "composition (see loom_group_norm's custom-op bridge for that case)."
                    )
                n = int(n_raw)
                x_name = resolve(self.safe_name(x_var_obj.name))
                output_var = self.safe_name(op.outputs[0].name)
                sum_name = output_var + "_rmean_sum"
                nodes.append({
                    "op": "REDUCE_SUM",
                    "inputs": [x_name],
                    "outputs": [sum_name],
                    "attrs": {"axis": ne_axis, "keep_dims": keep_dims_val}
                })
                nodes.append({
                    "op": "SCALE",
                    "inputs": [sum_name],
                    "outputs": [output_var],
                    "attrs": {"s": 1.0 / n}
                })
                continue

            if op_type == "reduce_sum":
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
                    continue
                out_rank = len(self.get_var_info(op.outputs[0])["shape"])
                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                axes_val = axes_obj.val if axes_obj is not None and hasattr(axes_obj, "val") else None
                keep_dims_val = bool(keep_dims_obj.val) if keep_dims_obj is not None and hasattr(keep_dims_obj, "val") else False
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
                        continue
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
                continue

            if op_type == "cumsum":
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
                    continue
                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                axis_val = int(axis_obj.val) if axis_obj is not None and hasattr(axis_obj, "val") and axis_obj.val is not None else 0
                if axis_val < 0:
                    axis_val += in_rank
                if axis_val != in_rank - 1:
                    raise NotImplementedError(
                        f"cumsum op '{op.name}': only cumulative sum over the trailing (ne[0]) axis is "
                        f"supported (got axis={axis_val!r} for rank {in_rank}) -- ggml_cumsum only ever "
                        "sums over ne[0]."
                    )
                excl_val = bool(excl_obj.val) if excl_obj is not None and hasattr(excl_obj, "val") and excl_obj.val is not None else False
                rev_val = bool(rev_obj.val) if rev_obj is not None and hasattr(rev_obj, "val") and rev_obj.val is not None else False
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
                continue

            if op_type == "layer_norm":
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
                    continue
                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                axes_val = axes_obj.val if axes_obj is not None and hasattr(axes_obj, "val") else None
                norm_axes = sorted(int(a) + in_rank if a < 0 else int(a) for a in axes_val) if axes_val is not None else None
                if norm_axes != [in_rank - 1]:
                    raise NotImplementedError(
                        f"layer_norm op '{op.name}': only normalization over the single trailing (ne[0]) "
                        f"axis is supported (got axes={axes_val!r} for rank {in_rank}) -- ggml_norm only "
                        "ever normalizes over ne[0]."
                    )
                eps_val = float(eps_obj.val) if eps_obj is not None and hasattr(eps_obj, "val") and eps_obj.val is not None else 1e-5

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
                continue

            if op_type == "instance_norm":
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
                    continue
                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                if in_rank != 3:
                    raise NotImplementedError(
                        f"instance_norm op '{op.name}': only rank-3 (B,C,T) input is supported (got "
                        f"rank {in_rank}) -- rank 4 (2 spatial dims) needs a real multi-axis ggml_norm, "
                        "not yet needed by any model this exporter has targeted."
                    )
                eps_val = float(eps_obj.val) if eps_obj is not None and hasattr(eps_obj, "val") and eps_obj.val is not None else 1e-5
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
                continue

            if op_type in ("upsample_nearest_neighbor", "upsample_bilinear"):
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
                sfh = float(sfh_obj.val) if sfh_obj is not None and hasattr(sfh_obj, "val") and sfh_obj.val is not None else 1.0
                sfw = float(sfw_obj.val) if sfw_obj is not None and hasattr(sfw_obj, "val") and sfw_obj.val is not None else 1.0
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
                continue

            if op_type == "range_1d":
                # Dedicated branch (not the generic OP_MAP path, and not the old input-ordering-only fix
                # below the main OP_MAP dispatch): op_range_1d's C++ side can only read a dynamic "end"/
                # "start" bound from a Var's own already-BUILT `.data`, which a `gather(shape(x), ...)`
                # chain's value never is at build time (see _try_derive_gather_shape_value's own
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

                start_resolved = self._resolve_range_scalar(start_obj)
                end_resolved = self._resolve_range_scalar(end_obj)
                step_resolved = self._resolve_range_scalar(step_obj)

                range_node = {"op": "RANGE_1D", "outputs": [self.safe_name(op.outputs[0].name)]}
                if start_resolved is not None and end_resolved is not None and step_resolved is not None:
                    range_node["inputs"] = []
                    range_node["attrs"] = {"start": start_resolved, "end": end_resolved, "step": step_resolved}
                else:
                    range_inputs = []
                    for v in (start_obj, end_obj, step_obj):
                        if v is not None and isinstance(v, Var):
                            range_inputs.append(resolve(self.safe_name(v.name)))
                    range_node["inputs"] = range_inputs
                nodes.append(range_node)
                continue

            if op_type == "conv":
                # Map conv to CONV_1D or CONV_2D and extract static attributes
                strides = op.inputs["strides"].val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else [1]
                pad = op.inputs["pad"].val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else [0]
                dilations = op.inputs["dilations"].val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else [1]
                groups = op.inputs["groups"].val if "groups" in op.inputs and hasattr(op.inputs["groups"], "val") else 1
                
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
                if bias_var_obj is not None and getattr(bias_var_obj, "val", None) is not None and np.any(bias_var_obj.val):
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
                continue

            if op_type == "conv_transpose":
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
                strides = op.inputs["strides"].val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else [1]
                pad = op.inputs["pad"].val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else [0]
                dilations = op.inputs["dilations"].val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else [1]
                groups = op.inputs["groups"].val if "groups" in op.inputs and hasattr(op.inputs["groups"], "val") else 1
                output_shape = op.inputs.get("output_shape")
                bias_var = op.inputs.get("bias")
                pad_type_var = op.inputs.get("pad_type")
                pad_type = (pad_type_var.val if pad_type_var is not None and hasattr(pad_type_var, "val")
                            and pad_type_var.val is not None else "valid")

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
                    actual = [int(v) for v in output_shape.val[-n_spatial:]]
                    if expected != actual:
                        raise NotImplementedError(
                            f"conv_transpose op '{op.name}' declares output_shape={list(output_shape.val)!r}, "
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
                    # Depthwise conv_transpose (Kokoro's AdainResBlk1d upsample "pool":
                    # ConvTranspose1d(kernel=3, stride=2, groups=dim_in, padding=1, output_padding=1)) --
                    # ggml has no native grouped CONV_TRANSPOSE primitive at all, so this composes the
                    # standard "zero-stuff the input, then an ordinary (stride=1) depthwise conv with a
                    # kernel-reversed weight" identity instead (real ConvTranspose1d IS mathematically a
                    # correlation with a flipped kernel over a zero-stuffed signal). Confirmed (empirically,
                    # via get_var_info dumps on this exact op) that MIL ALWAYS traces a grouped
                    # conv_transpose with pad=[0,0] regardless of the real PyTorch padding/output_padding
                    # -- it computes the "valid" (unpadded) result and defers any real crop to a SEPARATE
                    # downstream slice_by_index op, which the generic per-op-type loop already handles on
                    # its own turn -- so this composition only ever needs to reproduce the "valid" formula
                    # `(L_in-1)*stride + kernel_size` (symmetric `kernel_size-1` padding both sides of the
                    # zero-stuffed signal, i.e. padding=0/output_padding=0 in the general formula), never a
                    # real nonzero pad itself. Reuses the exact node sequence already verified in
                    # tools/convert_kokoro/convert_kokoro_f0n.py's `add_depthwise_conv_transpose_upsample`
                    # (itself checked against test_primitive_registry.cpp's
                    # test_depthwise_conv_transpose_1d_via_composition before ever being used there),
                    # generalized to a real traced op's own stride/kernel_size/channel count and a REAL
                    # dynamic-length expression (via get_var_info) instead of that script's own hardcoded
                    # "$n_tokens".
                    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                    weight_var_obj = op.inputs["weight"]
                    in_channels = int(x_var_obj.shape[1])
                    out_per_group = int(weight_var_obj.shape[1])
                    kernel_size = int(weight_var_obj.shape[-1])
                    if is_2d or g_val != in_channels or out_per_group != 1:
                        raise NotImplementedError(
                            f"conv_transpose op '{op.name}' has groups={g_val} (in_channels={in_channels}, "
                            f"out_channels/group={out_per_group}) -- only a true 1D depthwise case "
                            "(groups == in_channels == out_channels) is composed; anything else has no "
                            "ggml-side implementation yet."
                        )
                    if any(int(p) != 0 for p in pad_list):
                        raise NotImplementedError(
                            f"conv_transpose op '{op.name}' is depthwise with non-zero pad={pad_list!r} -- "
                            "every depthwise conv_transpose this exporter has seen traces with pad=[0,0] "
                            "(deferring any real crop to a separate downstream slice_by_index op); this "
                            "composition doesn't know how to fold a non-zero pad in directly."
                        )
                    weight_val = weight_var_obj.val
                    if weight_val is None:
                        raise NotImplementedError(
                            f"conv_transpose op '{op.name}' is depthwise but its weight isn't a resolved "
                            "constant -- this composition needs to flip the kernel at export time."
                        )
                    flipped = np.ascontiguousarray(np.asarray(weight_val)[:, :, ::-1])
                    flipped_name = self.safe_name(op.inputs["weight"].name) + "_dwt_flip"
                    if func_name == "main_topo" or self.profile == "monolithic":
                        namespaced_flipped = flipped_name
                    else:
                        namespaced_flipped = f"{func_name}.{flipped_name}"
                    if len(namespaced_flipped) >= 64:
                        import hashlib
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
                                  "attrs": {"shape": [f"({t_expr})*{s0}", channels]}})
                    std_len = f"(({t_expr})-1)*{s0}+1"
                    trunc_v = f"{output_var}_dwt_trunc_v"
                    nodes.append({"op": "VIEW", "inputs": [overstuffed], "outputs": [trunc_v],
                                  "attrs": {"shape": [std_len, channels]}})
                    trunc = f"{output_var}_dwt_trunc"
                    nodes.append({"op": "CONT", "inputs": [trunc_v], "outputs": [trunc]})
                    pad_each = kernel_size - 1
                    padded = f"{output_var}_dwt_padded"
                    nodes.append({"op": "PAD_1D", "inputs": [trunc], "outputs": [padded],
                                  "attrs": {"lp0": pad_each, "rp0": pad_each}})

                    has_bias = bias_var is not None and getattr(bias_var, "val", None) is not None and np.any(bias_var.val)
                    raw_var = (output_var + "_dwt_raw") if has_bias else output_var
                    nodes.append({"op": "CONV_1D_DW", "inputs": [namespaced_flipped, padded], "outputs": [raw_var],
                                  "attrs": {"s0": 1, "p0": 0, "d0": 1}})
                    if has_bias:
                        bias_var_name = self.safe_name(bias_var.name)
                        bias_reshaped = output_var + "_dwt_bias_r"
                        nodes.append({"op": "RESHAPE", "inputs": [resolve(bias_var_name)], "outputs": [bias_reshaped],
                                      "attrs": {"shape": [1, channels, 1]}})
                        nodes.append({"op": "ADD", "inputs": [raw_var, bias_reshaped], "outputs": [output_var]})
                    continue

                mapped_op = "CONV_TRANSPOSE_2D" if is_2d else "CONV_TRANSPOSE_1D"

                x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                x_var = self.safe_name(x_var_obj.name)
                weight_var = self.safe_name(op.inputs["weight"].name)
                output_var = self.safe_name(op.outputs[0].name)
                has_bias = bias_var is not None and getattr(bias_var, "val", None) is not None and np.any(bias_var.val)
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
                continue

            if op_type == "less":
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
                # emission uses) and the comparison bound's real formula (`_resolve_scalar_expr`, walking
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
                def _norm_expr(e):
                    return re.sub(r"\s+", "", str(e)) if e is not None else None

                def _eval_expr(expr, n_tokens_value):
                    if expr is None:
                        return None
                    try:
                        return float(eval(expr.replace("n_tokens", str(n_tokens_value)),
                                           {"__builtins__": {}}, {"floor": math.floor}))
                    except Exception:
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
                    range_expr = _norm_expr(self._infer_dynamic_dim_expr(range_var, 0))
                    length_expr = _norm_expr(self._resolve_scalar_expr(length_side_var))
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
                if bypass_ok:
                    out_info = self.get_var_info(op.outputs[0])
                    target_shape = list(out_info["shape"])
                    rank = len(target_shape)
                    weight_name = self.safe_name(op.outputs[0].name) + "_always_valid_scalar"
                    namespaced_name = (weight_name if (func_name == "main_topo" or self.profile == "monolithic")
                                        else f"{func_name}.{weight_name}")
                    self.weights[namespaced_name] = np.full([1] * rank, 1.0, dtype=np.float32)
                    nodes.append({
                        "op": "REPEAT",
                        "inputs": [namespaced_name],
                        "outputs": [self.safe_name(op.outputs[0].name)],
                        "attrs": {"shape": target_shape}
                    })
                    continue

            if op_type == "batch_norm":
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
                if any(v is None or not hasattr(v, "val") or v.val is None for v in (mean_var, var_var)):
                    raise NotImplementedError(
                        f"batch_norm op '{op.name}' has a non-constant mean/variance -- only eval-mode "
                        "(real running-stats buffers) BatchNorm is supported."
                    )
                mean_np = np.asarray(mean_var.val, dtype=np.float32)
                var_np = np.asarray(var_var.val, dtype=np.float32)
                gamma_np = (np.asarray(gamma_var.val, dtype=np.float32)
                            if gamma_var is not None and hasattr(gamma_var, "val") and gamma_var.val is not None
                            else np.ones_like(mean_np))
                beta_np = (np.asarray(beta_var.val, dtype=np.float32)
                           if beta_var is not None and hasattr(beta_var, "val") and beta_var.val is not None
                           else np.zeros_like(mean_np))
                eps = (float(eps_var.val) if eps_var is not None and hasattr(eps_var, "val")
                       and eps_var.val is not None else 1e-5)

                scale_np = gamma_np / np.sqrt(var_np + eps)
                shift_np = beta_np - mean_np * scale_np

                in_rank = len(self.get_var_info(x_var_obj)["shape"])
                channel_ne_axis = in_rank - 1 - 1  # MIL's channel axis is always torch axis 1
                bcast_shape = [1] * in_rank
                bcast_shape[channel_ne_axis] = scale_np.shape[0]

                x_name = resolve(self.safe_name(x_var_obj.name))
                output_var = self.safe_name(op.outputs[0].name)
                weight_base = f"{self.safe_name(op.name)}_bn"
                if func_name == "main_topo" or self.profile == "monolithic":
                    scale_name, shift_name = f"{weight_base}_scale", f"{weight_base}_shift"
                else:
                    scale_name, shift_name = f"{func_name}.{weight_base}_scale", f"{func_name}.{weight_base}_shift"
                self.weights[scale_name] = scale_np.reshape(bcast_shape)
                self.weights[shift_name] = shift_np.reshape(bcast_shape)

                scaled = f"{output_var}_bn_scaled"
                nodes.append({"op": "MUL", "inputs": [x_name, scale_name], "outputs": [scaled]})
                nodes.append({"op": "ADD", "inputs": [scaled, shift_name], "outputs": [output_var]})
                continue

            if op_type in ("random_normal", "random_uniform", "random_bernoulli", "random_categorical"):
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
                    "trace boundary (e.g. via a 'prefix'/'aux' split in a SubmoduleExportSpec, see "
                    "submodule_export.py) and feed the pre-sampled noise in as an explicit input instead "
                    "-- sampled at runtime via the existing loom.gaussian_array/loom.uniform_array/"
                    "loom.seed_rng Lua host functions (src/core/lua_bridge.cpp), the same host-side-RNG "
                    "pattern every hand-written driver (VitsDriver/SupertonicDriver/MatchaDriver/...) "
                    "already uses."
                )

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
            elif mapped_op in ("MEAN", "PERMUTE", "SOFTMAX", "CLAMP", "RSQRT", "RESHAPE", "VIEW", "LOG", "SQRT"):
                # Unary reduction/metadata operations in Loom C++ strictly expect exactly 1 input tensor
                x_val_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                if x_val_obj:
                    x_name = resolve(self.safe_name(x_val_obj.name))
                    if (mapped_op == "MEAN" and getattr(x_val_obj, "op", None) is not None
                            and x_val_obj.op.op_type == "transpose"):
                        # ggml_mean (like conv's im2col elsewhere) reduces ne[0] assuming a CONTIGUOUS
                        # source -- fed a PERMUTE's own output (a non-contiguous view) directly, it
                        # silently reads with the WRONG stride and produces a plausible-looking but
                        # WRONG result (no assert, unlike conv's im2col; confirmed via an isolated
                        # minimal repro: PERMUTE([4,T]->[T,4])+MEAN gave [10,20,17,14] instead of the
                        # correct per-channel means [1,11,21,31] for a hand-computed input -- adding an
                        # explicit CONT between the two fixed it exactly). First hit by StyleTTS2's
                        # diffusion Transformer1d.run(), whose `x.mean(axis=1)` (reducing over the token
                        # axis) traces to exactly this PERMUTE-straight-into-MEAN shape (torch's own
                        # `.mean()` needs the reduced axis transposed to ne[0] first, matching this
                        # project's own "PERMUTE so the target axis lands on ne[0], THEN reduce"
                        # convention for REDUCE_SUM elsewhere). Inserting a CONT here is always safe
                        # regardless of whether the source happens to already be contiguous (a CONT of an
                        # already-contiguous tensor is a harmless, cheap extra copy), so this cannot
                        # regress any existing MEAN usage.
                        cont_name = x_name + "_mean_cont"
                        nodes.append({"op": "CONT", "inputs": [x_name], "outputs": [cont_name]})
                        x_name = cont_name
                    inputs = [x_name]
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
                x_var_obj = op.inputs.get("x")
                y_var_obj = op.inputs.get("y")
                inp1 = resolve(self.safe_name(x_var_obj.name)) if x_var_obj is not None and hasattr(x_var_obj, "name") else None
                inp2 = resolve(self.safe_name(y_var_obj.name)) if y_var_obj is not None and hasattr(y_var_obj, "name") else None

                # Mutual (different-axis) broadcast: ggml_mul/ggml_add only ever let ONE operand
                # broadcast INTO the other's ALREADY-correct shape -- a genuine "outer product" case
                # (each operand is size-1 on a DIFFERENT axis than the other, so BOTH need widening to
                # reach the real output shape) isn't representable that way at all. Confirmed on
                # SupertonicTTS's VFTextCrossAttention fractional-RoPE angle computation (`theta[d] *
                # frac_pos[pos]`, ne=[32,1,1] * ne=[1,L,1] -> ne=[32,L,1] -- neither operand's shape
                # divides evenly into the other's) -- the bespoke conversion's own supertonic_common.py
                # independently worked around the identical operation via a dedicated MUL_MAT-based outer
                # product; this is the general exporter-level fix instead, reusing the already-existing
                # REPEAT primitive (built for StyleTTS2's own broadcast needs) rather than a new op.
                if (x_var_obj is not None and y_var_obj is not None
                        and getattr(x_var_obj, "shape", None) is not None
                        and getattr(y_var_obj, "shape", None) is not None):
                    out_shape = self.get_var_info(op.outputs[0])["shape"]

                    def _needs_repeat(var_obj):
                        shape = self.get_var_info(var_obj)["shape"]
                        return len(shape) == len(out_shape) and any(
                            str(s) == "1" and str(t) != "1" for s, t in zip(shape, out_shape)
                        )

                    if _needs_repeat(x_var_obj) and _needs_repeat(y_var_obj):
                        node_tag = self.safe_name(op.name)
                        x_rep = f"{node_tag}_bcast_x"
                        nodes.append({"op": "REPEAT", "inputs": [inp1], "outputs": [x_rep],
                                      "attrs": {"shape": out_shape}})
                        inp1 = x_rep
                        y_rep = f"{node_tag}_bcast_y"
                        nodes.append({"op": "REPEAT", "inputs": [inp2], "outputs": [y_rep],
                                      "attrs": {"shape": out_shape}})
                        inp2 = y_rep

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
        output_symbol = resolve(self.safe_name(func_outputs[0].name)) if func_outputs else "output"

        pruned_nodes = self._prune_dead_nodes(nodes, output_symbol)

        # A declared input with no node (post-pruning) that actually reads it is unreachable from this
        # topology's own output -- GraphBuilder's ggml_gallocr_alloc_graph only allocates a backend
        # buffer for tensors reachable from the declared output, so an orphan input tensor is created
        # but never given a buffer. If the driver ever supplied a value for it anyway (e.g. a
        # submodule-export blueprint's per-layer function still nominally "declaring" a
        # cache_position/position_ids input that its own real call never ends up depending on once
        # past_key_values is forced to None -- see submodule_export.py's _CACHE_KWARG_NAMES comment),
        # setting data into that unallocated tensor is a ggml hard-crash ("tensor buffer not set"), not
        # a graceful error. Dropping it from the declared list here means check_subgraph_calls treats
        # the driver still supplying it as an undeclared-input validation error instead -- catching the
        # mismatch at export time rather than as a runtime crash.
        referenced = {name for node in pruned_nodes for name in node["inputs"]}
        topo_inputs = [inp for inp in topo_inputs if inp["name"] in referenced]

        return {
            "version": 1,
            "inputs": topo_inputs,
            "output": output_symbol,
            "nodes": pruned_nodes
        }

    def _prune_dead_nodes(self, nodes: list, output_symbol: str) -> list:
        """
        Removes topology nodes whose output is never consumed, directly or transitively, by the
        topology's own declared output. GraphBuilder builds and COMPUTES every node unconditionally
        regardless of whether anything uses its result -- so an orphaned subgraph isn't just wasted
        compute, it can still crash (confirmed empirically on an orphaned chain that segfaulted during
        ggml_backend_graph_compute despite having zero real consumers, because nothing ever validates an
        unused node's own shapes/values are sane). Keeps only nodes reachable backward from the output.

        The GQA repeat_kv() fusion's own orphaned dependency chain (the original tile's now-unused
        "reps"-computation subgraph -- gather/concat/equal/select/div) no longer needs this: that fusion
        now runs as a real MIL->MIL pass (passes.py) with `common::dead_code_elimination` run right after
        it, so the orphan never survives into this walk at all (EXPORT-IMPROVEMENT-BACKLOG.md item 3).
        What this still exists for: `apply_atomic_export`'s own `_collect_replica_closure` deliberately
        replicates a producer op (and its transitive dependencies) into EVERY slice that consumes it,
        which leaves a now-unused copy behind in whichever slice originally "accidentally hosted" it --
        see that method's own comment. That's a Python-level list-slicing artifact of partitioning a
        single flattened trace after the fact, not something any MIL-level pass over the pre-partitioned
        `main` function could see or clean up, so this backward-reachability walk is still required for
        the atomic profile specifically.
        """
        needed = {output_symbol}
        live = []
        for node in reversed(nodes):
            if any(o in needed for o in node["outputs"]):
                live.append(node)
                needed.update(node["inputs"])
        live.reverse()
        return live

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
            from convert_nemo.tokenizer_common import write_sentencepiece_vocab
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
                q = quants.quantize(np.ascontiguousarray(array), qtype)
                # No `raw_shape` -- add_tensor's raw_shape (when given) is a *byte*-shape fed straight
                # into quant_shape_from_byte_shape, not the pre-quantization logical shape; omitting it
                # lets it default to the quantized array's own (correct) byte-shape.
                w.add_tensor(name, q, raw_dtype=qtype)
                n_quantized += 1
            else:
                w.add_tensor(name, array)

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        suffix = f", {n_quantized} tensor(s) quantized to {self.quantize}" if self.quantize else ""
        print(f"wrote GGUF with driver_script and {len(self.topologies)} topologies to {self.output_path}{suffix}")
