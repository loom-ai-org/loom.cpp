"""
Traces each declared submodule of an HF causal-LM standalone and assembles the results into one
multi-Function MIL Program -- EXPORT-IMPROVEMENT-BACKLOG.md item 2's "modular-export blueprint",
replacing scope-based partitioning of a single flattened trace.

This is the same assembly pattern tools/convert_lfm/make_lfm2_gguf.py used by hand
(`master_prog.functions[name] = mil.functions["main"]` per submodule), generalized and automated: no
hardcoded layer count, no hand-derived dummy shapes, no per-model wrapper subclasses. The one piece
that genuinely can't be recovered from module structure alone -- where "prefix"/"repeated"/"suffix"
boundaries fall, and which repeated-block kwarg a once-computed auxiliary tensor (e.g. a shared
rotary-embedding table) feeds -- is a short declarative `ModularExportSpec`, verified at export time
(a wrong attribute name raises `AttributeError` immediately, not a silent wrong export).
"""
import inspect
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.mil import Program, Function

from .modular_discovery import find_repeated_blocks, get_by_path, capture_calls

# HF's near-universal name for the stateful KV/conv-cache object threaded through every causal-LM
# submodule call. Replaying a captured call verbatim would bake in whatever cache state existed by the
# time the hook fired during the one real forward pass used for capture (e.g. a decoder layer captured
# mid-way through a 16-layer loop carries every EARLIER layer's cached K/V) -- silently wrong for a
# submodule that must behave like a fresh, history-free call once traced and invoked standalone (the
# modular driver supplies its own KV-cache bookkeeping at the C++ level). Forcing this one kwarg to None
# whenever present reproduces the affected model's own documented "no cache" default behavior.
_CACHE_KWARG_NAMES = {"past_key_values", "past_key_value"}


@dataclass
class ModularExportSpec:
    """Declares an HF causal-LM's prefix/repeated/suffix boundary -- the one piece of structure that
    isn't recoverable from `named_modules()` alone, since a `forward()` method is imperative Python,
    not a static graph. Attribute paths are dotted (e.g. "model.embed_tokens")."""
    prefix_attr: str
    repeated_attr: str
    suffix_attrs: list
    # A submodule computed ONCE (not per repeated-block call) whose output tensor(s) are threaded into
    # EVERY repeated-block invocation under `aux_kwarg` -- e.g. LFM2's `model.pos_emb`, computed once
    # in Lfm2Model.forward and passed to every decoder layer as `position_embeddings=(cos, sin)`.
    aux_attr: Optional[str] = None
    aux_kwarg: Optional[str] = None


@dataclass
class ModularExportResult:
    program: Program
    num_layers: int
    layer_input_names: list
    suffix_names: list
    aux_output_names: Optional[list] = None
    aux_kwarg: Optional[str] = None


@dataclass
class _LeafPath:
    name: str
    kind: str  # "arg" | "kwarg" | "kwarg_tuple_item"
    key: object  # int for "arg"; str for "kwarg"; (str, index) for "kwarg_tuple_item"


def _positional_param_names(module: nn.Module, count: int):
    sig = inspect.signature(module.forward)
    names = [p.name for p in sig.parameters.values() if p.name != "self"]
    return names[:count]


def _flatten_call(module: nn.Module, args: tuple, kwargs: dict):
    """Flattens a captured (args, kwargs) call into a name-ordered list of real tensor leaves, forcing
    known stateful-cache kwargs to None (see _CACHE_KWARG_NAMES). A tuple/list-valued kwarg (e.g.
    `position_embeddings=(cos, sin)`) becomes one leaf PER element, not one concatenated leaf -- P2
    (EXPORT-ROADMAP.md / BACKLOG.md's implementation sequence) gave `GraphBuilder`/`loom.run_subgraph`
    real multi-input/multi-output support, so a once-computed shared value (e.g. LFM2's rotary table,
    traced as its own "aux" submodule in export_modular) no longer has to be flattened into a single
    tensor to cross the run_subgraph boundary intact.

    Leaf names follow exporter.py's `apply_modular_export`/`is_aux_input` naming convention exactly:
    element 0 of a tuple-valued kwarg named `k` is named plain `k`; element i>=1 is named `f"{k}_{i}"`.
    `apply_modular_export` positionally maps the aux submodule's own i-th declared OUTPUT to the i-th
    such input across every repeated-block call, so both this function and `_replay` below must walk
    the tuple in the same order every real caller already does (a plain, untouched Python tuple) --
    neither side ever has to know or re-derive what index a name like "position_embeddings_1" means."""
    kwargs = dict(kwargs)
    for k in _CACHE_KWARG_NAMES:
        kwargs.pop(k, None)

    arg_names = _positional_param_names(module, len(args))
    paths, values = [], []

    for i, (pname, val) in enumerate(zip(arg_names, args)):
        if isinstance(val, torch.Tensor):
            paths.append(_LeafPath(pname, "arg", i))
            values.append(val)

    for k, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            paths.append(_LeafPath(k, "kwarg", k))
            values.append(val)
        elif isinstance(val, (tuple, list)) and len(val) > 0 and all(isinstance(x, torch.Tensor) for x in val):
            for i, item in enumerate(val):
                leaf_name = k if i == 0 else f"{k}_{i}"
                paths.append(_LeafPath(leaf_name, "kwarg_tuple_item", (k, i)))
                values.append(item)

    return list(args), kwargs, paths, values


def _replay(module, args_template, kwargs_template, paths, tensors):
    args = list(args_template)
    kwargs = dict(kwargs_template)
    tuple_kwargs = {}  # base kwarg name -> {index: tensor}, regrouped after the loop below
    for path, tensor in zip(paths, tensors):
        if path.kind == "arg":
            args[path.key] = tensor
        elif path.kind == "kwarg":
            kwargs[path.key] = tensor
        else:
            k, idx = path.key
            tuple_kwargs.setdefault(k, {})[idx] = tensor
    for k, items in tuple_kwargs.items():
        kwargs[k] = tuple(items[i] for i in range(len(items)))
    result = module(*args, **kwargs)
    if isinstance(result, (tuple, list)):
        # A module that itself returns a tuple (e.g. LFM2's rotary embedding returning (cos, sin)) is
        # traced with that tuple returned AS-IS (P2) -- torch.jit.trace + ct.convert turn a
        # tuple-of-tensors return into that many real MIL Function outputs, no longer squashed into one
        # concatenated tensor. See this function's own docstring note in `_flatten_call` for how the
        # ORDER here matches the consuming side's own index-by-position convention.
        if not all(isinstance(r, torch.Tensor) for r in result):
            raise ValueError(f"{type(module).__name__} returned a tuple with non-tensor element(s): "
                              f"{[type(r).__name__ for r in result]}")
        result = tuple(result)
    return result


class _ReplayWrapper(nn.Module):
    """Re-exposes `module`'s real captured (args, kwargs) call as a flat `forward(*tensors)` -- one
    positional tensor per real tensor leaf found in the captured call, in `paths` order -- so
    torch.jit.trace sees a plain tensors-in/tensors-out signature regardless of the wrapped module's
    own (possibly kwarg-heavy, possibly tuple-valued) call convention."""

    def __init__(self, module, args_template, kwargs_template, paths):
        super().__init__()
        self.module = module
        self._args_template = args_template
        self._kwargs_template = kwargs_template
        self._paths = paths

    def forward(self, *tensors):
        return _replay(self.module, self._args_template, self._kwargs_template, self._paths, tensors)


def _trace_module(name: str, module: nn.Module, args: tuple, kwargs: dict, seq_len: int, seq_len_dim) -> Function:
    """Traces `module` standalone against its own real captured call, declaring a dynamic seq-len
    RangeDim on whichever axis of each tensor leaf actually measures `seq_len` in the capture -- no
    axis is ever hardcoded to a fixed position or a specific argument name."""
    t0 = time.monotonic()
    args_t, kwargs_t, paths, values = _flatten_call(module, args, kwargs)
    if not values:
        raise ValueError(f"submodule '{name}' ({type(module).__name__}) has no real tensor input to trace against")

    wrapper = _ReplayWrapper(module, args_t, kwargs_t, paths).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(values))
    print(f"  [{name}] traced {type(module).__name__} in {time.monotonic() - t0:.1f}s (inputs={[p.name for p in paths]})")

    tensor_inputs = []
    for path, val in zip(paths, values):
        shape = list(val.shape)
        for ax, size in enumerate(shape):
            if size == seq_len:
                shape[ax] = seq_len_dim
        dtype = np.int32 if val.dtype in (torch.int64, torch.int32, torch.long) else np.float32
        tensor_inputs.append(ct.TensorType(name=path.name, shape=tuple(shape), dtype=dtype))

    # compute_precision=ct.precision.FLOAT32: ct.convert()'s own default (None) FP16-casts every constant
    # weight even under convert_to="milinternal" -- see export_hf_causal_lm.py's own comment / BACKLOG.md
    # for the full root-cause (found via Conformer-CTC's CONV_2D subsampling stage, but applies here too).
    mil_prog = ct.convert(traced, inputs=tensor_inputs, convert_to="milinternal",
                          compute_precision=ct.precision.FLOAT32)
    return mil_prog.functions["main"]


def export_modular(model: nn.Module, spec: ModularExportSpec, dummy_inputs: dict,
                    seq_len: int, max_seq_len: int = 4096) -> ModularExportResult:
    """Runs one real eager forward pass of `model` to capture every declared submodule's real call,
    then traces each standalone, assembling the results into one multi-Function Program. Returns the
    Program alongside the layout metadata `apply_modular_export` (exporter.py) needs to synthesize
    the driver: which input name is the repeated block's hidden-states chain input, which input names
    the auxiliary submodule's outputs feed, and how many layers/suffix stages there are."""
    prefix_module = get_by_path(model, spec.prefix_attr)
    # find_repeated_blocks structurally re-derives which attributes are qualifying repeated blocks
    # (an nn.ModuleList/Sequential with >1 child) independent of spec.repeated_attr's own claim --
    # cross-checking against it here means a typo'd or non-repeated attribute path raises immediately
    # with a clear error, rather than silently proceeding against whatever get_by_path happened to find.
    repeated_blocks = find_repeated_blocks(model)
    if spec.repeated_attr not in repeated_blocks:
        raise ValueError(
            f"'{spec.repeated_attr}' is not a qualifying repeated block (an nn.ModuleList/Sequential "
            f"with more than one child); discovered repeated blocks: {sorted(repeated_blocks)}"
        )
    children = repeated_blocks[spec.repeated_attr]
    suffix_modules = [get_by_path(model, a) for a in spec.suffix_attrs]

    targets = {"prefix": prefix_module, "layer": children[0]}
    for i, m in enumerate(suffix_modules):
        targets[f"suffix_{i}"] = m
    if spec.aux_attr:
        targets["aux"] = get_by_path(model, spec.aux_attr)

    captured = capture_calls(model, dummy_inputs, targets)

    seq_len_dim = ct.RangeDim(1, max_seq_len)
    prog = Program()

    prefix_func = _trace_module("prefix", prefix_module, *captured["prefix"], seq_len, seq_len_dim)
    prog.functions["prefix"] = prefix_func

    aux_output_names = None
    if spec.aux_attr:
        aux_func = _trace_module("aux", targets["aux"], *captured["aux"], seq_len, seq_len_dim)
        prog.functions["aux"] = aux_func
        aux_output_names = [v.name for v in aux_func.outputs]

    layer_args, layer_kwargs = captured["layer"]
    layer_input_names = None
    for i, child in enumerate(children):
        func = _trace_module(f"layer_{i}", child, layer_args, layer_kwargs, seq_len, seq_len_dim)
        prog.functions[f"layer_{i}"] = func
        if layer_input_names is None:
            layer_input_names = list(func.inputs.keys())

    suffix_names = []
    for i, m in enumerate(suffix_modules):
        name = f"suffix_{i}"
        func = _trace_module(name, m, *captured[name], seq_len, seq_len_dim)
        prog.functions[name] = func
        suffix_names.append(name)

    return ModularExportResult(
        program=prog,
        num_layers=len(children),
        layer_input_names=layer_input_names,
        suffix_names=suffix_names,
        aux_output_names=aux_output_names,
        aux_kwarg=spec.aux_kwarg,
    )
