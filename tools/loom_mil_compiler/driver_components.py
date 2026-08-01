"""The driver components the synthesized paths are made of (BACKLOG.md P4.0.6, `EXPORT-PREPARATION.md`
stage C.2).

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
from dataclasses import dataclass
from typing import List, Tuple

from .driver_ir import BinOp, Call, FieldAccess, If, Index, Lit, Local, Return, SubgraphCall, Var
from .driver_builder import DriverBuilder, DriverComponent, DriverContext
from .spec_protocol import FieldRef, NestedSpec, TopologyInput, TopologyName, Unchecked

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
