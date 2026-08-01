"""The driver components (BACKLOG.md P4.0.6, `EXPORT-PREPARATION.md` stages C.2-C.3).

Two groups, and the split is the migration itself: the components the two *synthesized* paths are made
of, and `RawLuaDriver`, which adopts one of the five hand-written TTS drivers whole so its family moves
onto the builder in a step whose gate is byte-identity -- and only then gets peeled into real
components one at a time (C.4-C.8), each peel independently revertable.

`LoomGGUFExporter.apply_monolithic_export` and `apply_modular_export` already built `driver_ir` nodes
directly, which is why the plan migrates them first: the `DriverBuilder` API gets proved against code
that already works, and the gate is unambiguous (byte-identical driver text for all six models that use
them). What they did *not* have is a name for the pieces, and the pieces are largely the same pieces --
`EXPORT-PREPARATION.md` §1.5 counts "prefill prologue/epilogue" as **one** inventory item spanning both
paths, and that is what this module makes true rather than observed:

    PrefillArgmaxBuilder : DriverInputs -> MonolithicCall  -> ArgmaxEpilogue
    ModularChainBuilder  : DriverInputs -> ModularChain    -> ArgmaxEpilogue

Two of the three components are shared. That is the whole claim of P4.0.7's "marketplace" in its
smallest instance, and it is worth having demonstrated before the TTS families arrive with components
nobody has written yet.

**What these components deliberately do not do.** They take IR pieces the exporter has already computed
(safe names, the length expression, each stage's own declared inputs) rather than reaching into the
traced program themselves. Which inputs a traced graph declares is the exporter's question -- it owns
`safe_name`, the MIL function and `generate_graph_topology`'s own post-pruning input list. A component
that re-derived any of that would be a second authority on it, and the two would drift.

**On the honesty of the links here.** `MonolithicCall`/`ModularChain` declare `TopologyName` and
`TopologyInput`, and for the six models on these two paths both are close to tautological *today*:
the exporter generates the topology and the call from the same `main_func`/`layout`, so the two sides
cannot disagree. They are declared anyway for one reason that is not tautological -- `ModularChain`'s
stage list is assembled from `layout.num_layers`/`layout.suffix_names`, which is a claim about what was
traced rather than a reading of it, and a drift there currently surfaces as a bare `KeyError` on a
dict of MIL functions. `TopologyInput(exact=True)` also adds the direction `check_subgraph_calls` has
never checked: an input the topology declares that the driver never supplies, which the engine reads
back as an uninitialised tensor rather than as an error.
"""
import re
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Tuple

from .driver_ir import (
    BinOp, Call, FieldAccess, If, Index, Lit, Local, RawBlock, Return, SubgraphCall, Var,
)
from .driver_ir import Function as IRFunction
from .driver_builder import DriverBuilder, DriverComponent, DriverContext, DriverScript
from .spec_protocol import (
    TOPOLOGIES, ConfigDerived, FieldRef, NestedSpec, TopologyInput, TopologyName, Unchecked, WhenSet,
)

# Traced-model input names auto-computed by the driver (via loom.range) rather than unpacked from the
# caller's `inputs` table -- see `DriverInputs` on why the driver, not the caller, owns these.
POSITION_INPUT_NAMES = {"cache_position", "position_ids"}

# Traced-model input names auto-computed by the driver via loom.causal_mask (an already-prepared 4D
# additive mask, the same "pass it explicitly so the traced model skips computing it internally" fix as
# POSITION_INPUT_NAMES -- see export_lfm2_*.py's own _causal_mask() comment).
CAUSAL_MASK_INPUT_NAMES = {"attention_mask"}

# Every input name the driver computes itself rather than reading from the caller.
HOST_COMPUTED_INPUT_NAMES = POSITION_INPUT_NAMES | CAUSAL_MASK_INPUT_NAMES

# The generic name a caller may always use for a driver's primary input, whatever the traced graph
# happens to call it (`input_ids`, `audio_signal`, ...).
GENERIC_PRIMARY_INPUT = "tokens"

# `DriverInputs.bindings` kinds.
CALLER, POSITION, MASK = "caller", "position", "mask"


def caller_input(name: str):
    """Reads one input from the driver's `inputs` table, with `tokens` accepted as an alias for it.

    The alias is what lets a caller pass the primary input as `inputs.tokens` without knowing the traced
    graph's own name for it. When that name IS `tokens` there is nothing to alias, and emitting the
    fallback anyway produced `inputs.tokens or inputs.tokens` -- same expression on both sides of the
    `or`, which is what the generated Lua read like for every causal LM."""
    field_access = FieldAccess("inputs", name)
    if name == GENERIC_PRIMARY_INPUT:
        return field_access
    return BinOp("or", field_access, FieldAccess("inputs", GENERIC_PRIMARY_INPUT))


@dataclass
class DriverInputs(DriverComponent):
    """Binds every name the topologies below will be called with: read from the caller's table, or
    computed host-side.

    The host-computed half is the load-bearing part and the reason this is a component rather than a
    loop. A traced model's own `cache_position`/`position_ids` input exists because passing it
    explicitly is what keeps the sequence length genuinely dynamic under `torch.jit.trace` (letting the
    model derive it internally from a Python-level `.shape[1]` query bakes the trace length in), and
    `attention_mask` is there for the same reason. Neither is something a *caller* should have to know
    about: the driver already knows `n_tokens`/`n_past`, so it fills them in.

    `bindings` is an ordered `(name, kind)` list rather than three sets, because the order is the
    emission order and the two paths genuinely differ in it -- the monolithic path binds in the traced
    function's own declared-input order, the modular path binds the chain input first and then the
    host-computed ones sorted, since its stages each declare their own subset.
    """

    bindings: Tuple[Tuple[str, str], ...]
    # The length expression the host-computed bindings are built at. Reads a name this component binds
    # earlier in the same list, which is why `driver_ir.validate` is the authority on it.
    n_tokens: object

    __unchecked__ = {
        "bindings": Unchecked(
            "the traced graph's own declared input names, READ off the MIL function (and off each "
            "stage's post-pruning topology inputs) rather than claimed about it. A link runs the other "
            "way -- there is no second authority to compare these against, and the call sites that use "
            "them are checked by TopologyInput on MonolithicCall/ModularChain."
        ),
        "n_tokens": Unchecked(
            "a driver_ir expression over names this component itself binds. `driver_ir.validate` is "
            "the authority on whether it reads a symbol defined before it, and it runs over the "
            "assembled function, which is the only place the question is answerable."
        ),
    }

    def emit(self, ctx: DriverContext) -> List:
        out = []
        for name, kind in self.bindings:
            if kind == POSITION:
                out.append(Local(name, Call("loom.range", [Lit(0), self.n_tokens])))
            elif kind == MASK:
                out.append(Local(name, Call("loom.causal_mask", [self.n_tokens, Lit(0)])))
            else:
                out.append(Local(name, caller_input(name)))
        return out


@dataclass
class MonolithicCall(DriverComponent):
    """The single `run_subgraph` call a flattened export's driver makes.

    Captures the output's shape alongside its data (`extra_outputs`) because `ArgmaxEpilogue` needs the
    vocab size, which is the output's own ne0 and is not otherwise knowable to the driver.
    """

    topology: str = "main_topo"
    inputs: Tuple[str, ...] = ()
    n_tokens: object = None
    out_var: str = "_mono_out"
    shape_var: str = "_mono_shape"

    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology"), exact=True),
    }
    __unchecked__ = {
        "n_tokens": Unchecked("the driver_ir expression bound to this topology's root axis; see "
                              "DriverInputs.n_tokens for why validate() is its authority"),
        "out_var": Unchecked("the local this component binds -- it CREATES the name ArgmaxEpilogue "
                             "reads, and that read is checked by driver_ir.validate over the "
                             "assembled function"),
        "shape_var": Unchecked("same: a local this component binds rather than one it refers to"),
    }

    def emit(self, ctx: DriverContext) -> List:
        return [SubgraphCall(
            outputs=[self.out_var],
            extra_outputs=[self.shape_var],
            module=self.topology,
            # This topology's own declared root axis (EXPORT-ROADMAP.md R1) -- "n_tokens" unless the
            # caller declared otherwise (e.g. Conformer-CTC/Parakeet's "n_samples"); the VALUE is still
            # the first input's own length regardless of what the axis is named.
            axes={ctx.root_axis(self.topology): self.n_tokens, "n_past": Lit(0)},
            inputs={name: Var(name) for name in self.inputs},
        )]


@dataclass
class ChainStage:
    """One `run_subgraph` in a modular chain: which topology, and where each of its declared inputs
    comes from -- the chain variable, an aux output, or a driver-bound local.

    A record rather than a component: the stages of a chain are not independently orderable or
    reusable, and `ModularChain` is the thing a builder assembles.
    """

    topology: str
    # {input name -> the IR expression supplying it}, over exactly the topology's declared inputs.
    inputs: dict
    outputs: Tuple[str, ...]
    extra_outputs: Tuple[str, ...] = ()

    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology"), exact=True),
    }
    __unchecked__ = {
        "outputs": Unchecked("the locals this stage binds; the reads of them are checked by "
                             "driver_ir.validate over the assembled function"),
        "extra_outputs": Unchecked("same -- shape locals this stage binds. Whether capturing them is "
                                   "legal at all is driver_ir.check_subgraph_calls' question, and it "
                                   "answers it against the topology's real declared output count"),
    }

    def link_label(self) -> str:
        """A chain is 20-plus stages, so "ChainStage" alone does not say which one failed."""
        return f"ChainStage({self.topology!r})"


@dataclass
class ModularChain(DriverComponent):
    """Threads one tensor through an independently-traced submodule chain: prefix -> [aux] -> layer_0..N
    -> suffix_0..M.

    Every stage is self-contained by construction (each was traced standalone, so there is no
    cross-slice variable leakage to detect the way partitioning one flattened trace would need), which
    is why this is a flat list of calls and not a dataflow analysis. What it is *not* is a loop over a
    single shared input-name list: `generate_graph_topology` drops any declared input no node actually
    reads, and LFM2's conv-type layers never touch `position_embeddings` while its attention-type layers
    do -- so which inputs survive differs per layer even though every layer was traced with an identical
    call signature. Each `ChainStage` therefore carries its own resolved input map.
    """

    stages: Tuple[ChainStage, ...] = ()
    n_tokens: object = None

    __links__ = {
        "stages": NestedSpec(
            where="DriverBuilder.build, which registers every ChainStage with the export's own "
                  "checker (DriverComponent.sub_specs) so a failure names the stage that failed "
                  "rather than the chain it was in"
        ),
    }
    __unchecked__ = {
        "n_tokens": Unchecked("the driver_ir expression bound to every stage's root axis; see "
                              "DriverInputs.n_tokens"),
    }

    def sub_specs(self):
        return list(self.stages)

    def emit(self, ctx: DriverContext) -> List:
        return [
            SubgraphCall(
                outputs=list(stage.outputs),
                extra_outputs=list(stage.extra_outputs),
                module=stage.topology,
                axes={ctx.root_axis(stage.topology): self.n_tokens, "n_past": Lit(0)},
                inputs=dict(stage.inputs),
            )
            for stage in self.stages
        ]


@dataclass
class ArgmaxEpilogue(DriverComponent):
    """Returns the next token rather than the raw logits array.

    Argmaxes the logits row for the active (last real) token: `shape_var[1]` is the output's ne0 (vocab
    size), the same convention `transpile_operation`'s own "argmax" case relies on. The `type(...) ==
    'table'` guard is what keeps this correct for a topology whose output is not an array -- the engine
    hands back a scalar there, and argmaxing it is meaningless.
    """

    out_var: str
    shape_var: str
    n_tokens: object

    __unchecked__ = {
        "out_var": Unchecked("a local an earlier component bound. The read is checked by "
                             "driver_ir.validate over the assembled function, which is where a "
                             "cross-component symbol read is answerable and nowhere else"),
        "shape_var": Unchecked("same -- the shape local the calling component captured"),
        "n_tokens": Unchecked("the driver_ir expression for the active row; see DriverInputs.n_tokens"),
    }

    def emit(self, ctx: DriverContext) -> List:
        row = BinOp("-", self.n_tokens, Lit(1))
        return [If(
            cond=BinOp("==", Call("type", [Var(self.out_var)]), Lit("table")),
            then=[Return([Call("loom.argmax_row",
                               [Var(self.out_var), Index(Var(self.shape_var), 1), row])])],
            else_=[Return([Var(self.out_var)])],
        )]


# -- builders ----------------------------------------------------------------------------------------

# Every field of a builder holds one of its components, and a component declares and runs its own
# links -- so the builder's own declaration says where that happens rather than restating them. This is
# `NestedSpec`'s ordinary use, with one difference worth noting: unlike `ASRNemoEncoderExportConfig.
# output`, whose links are only checkable inside the traced wrapper's forward, these ARE checkable at
# the outer site, and `DriverBuilder.build` does exactly that.
_BUILDER_FIELDS_CHECKED_IN = (
    "DriverBuilder.build, which registers every component this builder returns from components() -- "
    "and each component's own sub_specs() -- with the export's checker, before any of them emits"
)


@dataclass
class PrefillArgmaxBuilder(DriverBuilder):
    """One traced graph, run once over the whole prompt, argmax the last row.

    This is what "export any causal LM" means today, and the limit is worth stating where the builder
    is rather than only in the roadmap: `n_past` is bound to `Lit(0)` and the mask is built for a fresh
    sequence, so the exported artifact is a *prefill*, not a generation loop. The engine has a KV cache
    and the bespoke drivers use it; the MIL path cannot yet, because a MIL-exported causal LM has no
    `ATTENTION` node to reach it through (`EXPORT-PREPARATION.md` §4).
    """

    inputs: DriverInputs
    call: MonolithicCall
    epilogue: ArgmaxEpilogue

    __links__ = {name: NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN)
                 for name in ("inputs", "call", "epilogue")}

    def components(self):
        return [self.inputs, self.call, self.epilogue]


@dataclass
class ModularChainBuilder(DriverBuilder):
    """Independently-traced submodules chained prefix -> [aux] -> layers -> suffix, then the same
    argmax epilogue. Shares two of its three components with `PrefillArgmaxBuilder`."""

    inputs: DriverInputs
    chain: ModularChain
    epilogue: ArgmaxEpilogue

    __links__ = {name: NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN)
                 for name in ("inputs", "chain", "epilogue")}

    def components(self):
        return [self.inputs, self.chain, self.epilogue]


# -- adopting a hand-written driver (C.3) ------------------------------------------------------------
#
# The five multi-phase TTS families ship a hand-written `.lua` that `render_driver` substitutes
# generated samplers into. `RawLuaDriver` adopts one whole, unchanged, so the family moves onto the
# builder in a step whose gate is byte-identity -- and only then gets peeled into real components one
# at a time (C.4-C.8), each peel independently revertable.
#
# **Wrapping the text in a `RawBlock` would, on its own, check nothing.** `check_subgraph_calls` walks
# `SubgraphCall` nodes, and raw text has none: the export would be byte-identical and the five drivers
# would gain exactly the check they had before, which is none. That is precisely the ambiguity the
# plan's negative gate exists to resolve, so the adoption *parses* its own `loom.run_subgraph` call
# sites and declares each one as a `RunSubgraphCall` -- routed through the P4.0.5 protocol rather than
# through a second copy of `check_subgraph_calls`, which is what gets `TopologyInput`'s bidirectional
# message and its "which inputs are missing" half for free.


@dataclass
class RunSubgraphCall:
    """One `loom.run_subgraph` call site found in hand-written Lua, as a checkable declaration.

    The same shape as `EstimatorSpec` -- a topology name plus the exact set of inputs supplied to it --
    and deliberately so: a hand-written call has the identical failure mode wherever it appears, and
    the message should not depend on whether a human or a template wrote the call. What this adds is
    the *site*: a driver has 30-odd call sites, so the label carries the file and line, which is the
    difference between a one-line fix and a search.
    """

    topology: str
    # The argument table's keys, or None when the third argument is not a table literal -- e.g.
    # `render_sampler`'s own generated call, which passes a prepared `args` variable. The topology name
    # is still checkable there; the input set is not, and saying so is the point of the `WhenSet`.
    inputs: Optional[Tuple[str, ...]]
    line: int
    origin: str = "driver"

    __links__ = {
        "topology": TopologyName(),
        "inputs": WhenSet(TopologyInput(FieldRef("topology"), exact=True)),
    }
    __unchecked__ = {
        "line": Unchecked("where the call was found, for the label. Read off the source, not claimed "
                          "about it"),
        "origin": Unchecked("the driver file the call was found in, for the label -- same"),
    }

    def link_label(self) -> str:
        return f"{self.origin}:{self.line} loom.run_subgraph({self.topology!r})"


# The Lua fragment every call site starts with.
_RUN_SUBGRAPH = "loom.run_subgraph("
# A table-literal entry's key: `name =`, but not `name ==`.
_TABLE_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)")
_STRING_LITERAL = re.compile(r"""^(?:"([^"]*)"|'([^']*)')$""")


def _split_top_level(text: str) -> List[str]:
    """`text` split on commas that are not inside brackets or a string literal."""
    parts, depth, start, quote = [], 0, 0, None
    for i, ch in enumerate(text):
        if quote is not None:
            if ch == quote and text[i - 1] != "\\":
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _table_keys(text: str) -> Optional[Tuple[str, ...]]:
    """The keys of a Lua table literal, or None if `text` is not one (a variable, a call, ...) or
    carries a positional entry -- in which case the set of inputs is not statically known and claiming
    otherwise would be worse than declaring nothing."""
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    keys = []
    for item in _split_top_level(text[1:-1]):
        if not item.strip():
            continue
        match = _TABLE_KEY.match(item)
        if match is None:
            return None
        keys.append(match.group(1))
    return tuple(keys)


def parse_run_subgraph_calls(source: str, origin: str = "driver"):
    """Every `loom.run_subgraph` call site in `source`, as `(checkable, unresolved)`.

    `unresolved` carries the sites whose topology argument is a computed Lua expression rather than a
    string literal -- `namespace_ .. "_h_fwd"`, `name_prefix .. "_block" .. i`, a bare variable. Those
    cannot be checked from the text, and they are *returned* rather than dropped: an adoption that
    silently ignored a third of a driver's call sites while reporting the rest as checked would be the
    "validated where convenient" failure the whole protocol is built to prevent. They are also not a
    permanent gap -- every one of them belongs to the BiLSTM/resblock stepping loops that D.2 turns
    into a registered component, which declares its topologies as data.
    """
    checkable, unresolved = [], []
    index = source.find(_RUN_SUBGRAPH)
    while index != -1:
        line = source.count("\n", 0, index) + 1
        args_text, end = _balanced_args(source, index + len(_RUN_SUBGRAPH))
        index = source.find(_RUN_SUBGRAPH, end)
        args = _split_top_level(args_text)
        if len(args) != 3:
            unresolved.append((line, args_text.strip().splitlines()[0]))
            continue
        literal = _STRING_LITERAL.match(args[0].strip())
        if literal is None:
            unresolved.append((line, args[0].strip()))
            continue
        checkable.append(RunSubgraphCall(
            topology=literal.group(1) if literal.group(1) is not None else literal.group(2),
            inputs=_table_keys(args[2]), line=line, origin=origin,
        ))
    return checkable, unresolved


def _balanced_args(source: str, start: int):
    """`(argument text, index just past the closing paren)` for a call whose `(` is at `start - 1`."""
    depth, i, quote = 1, start, None
    while depth and i < len(source):
        ch = source[i]
        if quote is not None:
            if ch == quote and source[i - 1] != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        i += 1
    return source[start:i - 1], i


@dataclass
class RawLuaDriver(DriverComponent):
    """A hand-written driver, adopted whole: everything above its entry function becomes the prelude,
    the entry function's body becomes one verbatim `RawBlock`, and anything after it the postlude.

    Nothing here is a semantic change and the gate is byte-identity, which is the point -- the family
    is now assembled by a `DriverBuilder`, so the next commit can peel one block out of the raw body
    into a real component without also having to move the family onto the builder in the same step.

    The split is by line rather than by regex over the whole file because the reconstruction has to be
    exact: `render()` re-joins prelude + `function <entry>(...)` + body + `end` + postlude with single
    newlines, and `verbatim=True` keeps the body's own indentation rather than adding the enclosing
    block's. Both are checked by `assert_round_trip`, which is not a belt-and-braces test but the thing
    that makes this commit's gate provable before an export is run at all.
    """

    source: str
    entry: str = "synthesize"
    origin: str = "driver"
    # `{topology name: where it comes from}` -- see BaseMultiPhaseModelExportConfig.external_topologies.
    external: dict = dataclass_field(default_factory=dict)

    __links__ = {
        "source": NestedSpec(
            where="DriverBuilder.build, via sub_specs() -- every loom.run_subgraph call site in this "
                  "text with a literal topology name is parsed out and declared as a RunSubgraphCall, "
                  "which is what gives the five hand-written drivers their first cross-check against "
                  "the topologies they are actually shipped with"
        ),
        # An external declaration is only honest if it cannot rot in either direction, so both are
        # checked -- and the two need different context, which is exactly what deferral is for.
        "external": ConfigDerived(
            claim=lambda spec, ctx: (),
            measured=lambda spec, ctx: tuple(sorted(set(spec.external) & set(ctx.topologies))),
            detail=lambda spec, ctx: sorted(set(spec.external) & set(ctx.topologies)),
            message=(
                "{label} declares topolog(ies) {detail} as coming from outside this export, but this "
                "export produces them. A stale external declaration silently suppresses the very check "
                "it was added to make honest -- drop it from external_topologies()."
            ),
            needs=(TOPOLOGIES,),
        ),
        "_external_is_called": ConfigDerived(
            claim=lambda spec, ctx: (),
            measured=lambda spec, ctx: tuple(sorted(set(spec.external) - spec.called_topologies())),
            detail=lambda spec, ctx: sorted(set(spec.external) - spec.called_topologies()),
            message=(
                "{label} declares topolog(ies) {detail} as external, but its driver never calls them. "
                "A dead declaration is worse than none: it reads as an accounted-for dependency."
            ),
            needs=(),
        ),
    }
    __unchecked__ = {
        "entry": Unchecked(
            "the Lua function name the host calls. Not checkable against the topologies -- but not "
            "unchecked either, in the sense that matters: `_split` raises if the source has no such "
            "function, which is the only way to get it wrong"
        ),
        "origin": Unchecked("the driver file's name, used to label a failing call site"),
    }

    def __post_init__(self):
        self._prelude, self._body, self._postlude = _split_lua_module(self.source, self.entry)
        self._calls, self.unresolved_calls = parse_run_subgraph_calls(self.source, self.origin)

    def link_label(self) -> str:
        return f"RawLuaDriver({self.origin!r})"

    def called_topologies(self) -> set:
        """The topology names this driver names literally. Not the whole set it calls -- the computed
        ones are in `unresolved_calls` -- which is why the "external but never called" check can only
        ever be a check on the literal half."""
        return {call.topology for call in self._calls}

    def sub_specs(self):
        # A call into a topology this export deliberately does not produce is checked by the
        # `external` declaration above instead: TopologyName would report it as missing, which is true
        # and useless, and TopologyInput has nothing to compare against at all.
        return [call for call in self._calls if call.topology not in self.external]

    def prelude(self, ctx):
        return list(self._prelude)

    def emit(self, ctx):
        return [RawBlock(list(self._body), verbatim=True)]

    def postlude(self, ctx):
        return list(self._postlude)

    def coverage(self) -> str:
        """One line for the export log: how much of this driver the adoption actually checks.

        Printed rather than merely available, and reporting *two* numbers rather than one, because
        "checked" covers two different amounts here: a call whose third argument is a table literal
        has its full input set compared against the topology, while one that passes a prepared
        variable (`render_sampler`'s own generated call) only has its topology name checked. Reporting
        those as one figure would be the "validated where convenient" reading the whole protocol
        exists to prevent, one level down."""
        checked = self.sub_specs()
        total = len(self._calls) + len(self.unresolved_calls)
        with_inputs = sum(1 for call in checked if call.inputs is not None)
        notes = []
        if self.external:
            external_calls = sorted({c.topology for c in self._calls if c.topology in self.external})
            notes.append(f"{len(external_calls)} call topolog(ies) this export does not produce and "
                         f"the config declares external ({', '.join(external_calls)})")
        if self.unresolved_calls:
            shown = self.unresolved_calls[:2]
            examples = ", ".join(f"line {line}: {text}" for line, text in shown)
            more = ", ..." if len(self.unresolved_calls) > len(shown) else ""
            notes.append(f"{len(self.unresolved_calls)} name their topology with a computed expression "
                         f"and cannot be checked from the text ({examples}{more})")
        note = "".join(f"; {n}" for n in notes)
        return (f"  {self.origin}: {len(checked)}/{total} loom.run_subgraph call sites checked "
                f"against the exported topologies ({with_inputs} with their full input set){note}")

    def assert_round_trip(self) -> None:
        """The rendered driver must be the source, byte for byte.

        Called by the builder rather than only by a test: the split-and-rejoin is the one thing this
        component does that can silently corrupt a working driver, and a 30-line indentation shift is
        exactly the kind of change a reviewer reads past."""
        rebuilt = DriverScript(prelude=list(self._prelude),
                               entry=IRFunction(self.entry, ["inputs"],
                                                [RawBlock(list(self._body), verbatim=True)]),
                               postlude=list(self._postlude)).render()
        if rebuilt != self.source:
            raise ValueError(
                f"RawLuaDriver({self.origin!r}) does not reproduce its own source: the adoption is "
                f"supposed to be byte-identical and is not. First difference at offset "
                f"{_first_difference(rebuilt, self.source)}."
            )


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _split_lua_module(source: str, entry: str):
    """`(prelude lines, entry body lines, postlude lines)` for a Lua module whose entry point is a
    top-level `function <entry>(inputs)` closed by a column-0 `end`."""
    lines = source.split("\n")
    header = f"function {entry}(inputs)"
    try:
        start = lines.index(header)
    except ValueError:
        raise ValueError(
            f"driver source has no top-level {header!r} line to adopt as its entry function"
        ) from None
    for stop in range(start + 1, len(lines)):
        if lines[stop] == "end":
            return lines[:start], lines[start + 1:stop], lines[stop + 1:]
    raise ValueError(f"driver source's {header!r} is never closed by a column-0 'end'")


@dataclass
class MultiPhaseDriverBuilder(DriverBuilder):
    """Every multi-phase (TTS) family's driver, for as long as it is still hand-written.

    Holds exactly one component today. That is not a placeholder for a missing design -- it is what a
    family looks like at the moment it moves onto the builder, and C.4-C.8 add components beside the
    raw one as they peel blocks out of it. The `entry_name` differs from every other builder's
    (`synthesize`, not `main`) because these drivers are called by the TTS host, and that is a fact
    about the family rather than about the decomposition.
    """

    driver: RawLuaDriver

    __links__ = {"driver": NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN)}

    entry_name = "synthesize"

    def components(self):
        return [self.driver]

    def build(self, ctx, checker=None):
        self.driver.assert_round_trip()
        print(self.driver.coverage())
        return super().build(ctx, checker)
