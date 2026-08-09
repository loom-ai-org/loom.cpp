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
import dataclasses
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import List, Optional, Tuple

from .driver_ir import (
    ArrayLit, Argmax, Assign, BinOp, Break, Call, CallStmt, FieldAccess, If, Index, Len, Lit, Local,
    LocalDecl, NumericFor, OutputRef, RawBlock, RetainedArgmax, RetainedArgmaxRows, Return,
    SubgraphCall, TableLit, Var, While,
)
from .driver_ir import Function as IRFunction
from .driver_builder import (
    DriverBuilder, DriverComponent, DriverContext, DriverEntry, DriverScript,
)
from .spec_protocol import (
    TOPOLOGIES, ConfigDerived, CoveredBy, FieldRef, NestedSpec, TopologyInput, TopologyName, Unchecked,
    WhenSet, declared_links,
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
    # {input name: sliding-window width} for the masks that are banded rather than full-causal
    # (BACKLOG.md P4.0.11a). Empty for every model that has no windowed attention, which is every model
    # on this roadmap but Gemma 3 -- and an absent entry means "full causal", so the emitted call is
    # byte-identical to what it was before this field existed.
    mask_windows: dict = dataclasses.field(default_factory=dict)

    __unchecked__ = {
        "mask_windows": Unchecked(
            "read off the EMITTED topology by `LoomGGUFExporter._route_windowed_masks`, which "
            "synthesizes the windowed mask inputs in the first place -- so the exporter is the "
            "authority and there is no separate claim here for a checkpoint to disagree with."
        ),

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
                args = [self.n_tokens, Lit(0)]
                window = self.mask_windows.get(name)
                if window:
                    args.append(Lit(window))
                out.append(Local(name, Call("loom.causal_mask", args)))
            else:
                out.append(Local(name, caller_input(name)))
        return out


@dataclass
class MonolithicCall(DriverComponent):
    """The single `run_subgraph` call a flattened export's driver makes.

    Captures the output's shape alongside its data (`extra_outputs`) because `ArgmaxEpilogue` needs the
    vocab size, which is the output's own ne0 and is not otherwise knowable to the driver -- unless the
    call `retained`, in which case neither local exists and the epilogue reduces by module name instead.
    """

    topology: str = "main_topology"
    inputs: Tuple[str, ...] = ()
    n_tokens: object = None
    out_var: str = "_mono_out"
    shape_var: str = "_mono_shape"
    # When true, the output stays in the module's own buffer and this call binds nothing (P4.0.14).
    retained: bool = False

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
        "retained": Unchecked(
            "whether the epilogue reads this output by module name or by local -- one decision the "
            "exporter makes for both ends at once, exactly as ChainStage.retained is, and "
            "driver_ir.check_subgraph_calls is what catches the two disagreeing: a reduction naming a "
            "module nothing retained is an error there, and a retaining call that also binds locals is "
            "too."
        ),
    }

    def emit(self, ctx: DriverContext) -> List:
        if self.retained:
            # The logits never become a Lua table: they stay in the module's own OutputStore and
            # `ArgmaxEpilogue` asks for the row it wants by module name. That is what removes the
            # ~512-token prefill ceiling a 262144-wide vocab hits (BACKLOG.md P4.0.14).
            return [SubgraphCall(
                outputs=[],
                module=self.topology,
                axes={ctx.root_axis(self.topology): self.n_tokens, "n_past": Lit(0)},
                inputs={name: Var(name) for name in self.inputs},
                retain=True,
            )]
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
    # When true, this stage's outputs stay in the engine (BACKLOG.md P4.0.12) and `outputs`/
    # `extra_outputs` are empty: the next stage reaches them with an `OutputRef` instead of a local.
    # Per stage rather than per chain because a chain need not be uniform, though every synthesized one
    # now is: since P4.0.14 the LAST stage retains too, and the epilogue reduces its logits by module
    # name rather than marshalling them to argmax a single row.
    retained: bool = False

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
        "retained": Unchecked(
            "whether this stage's consumer reads it by module name or by local -- one decision the "
            "exporter makes for both ends of the edge at once, and driver_ir.check_subgraph_calls is "
            "what catches the two disagreeing: a reference to a module nothing retained is an error "
            "there, and a retained call that also binds locals is too."
        ),
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

    **This is the case module-owned output buffers exist for** (BACKLOG.md P4.0.12). Every edge here is
    an intermediate the driver merely threads onward -- a `[n_embd, n_tokens]` hidden state nobody looks
    at -- and before retention each one was read into a Lua table and written straight back: two copies
    per edge per step on CPU, and a device->host->device round trip per edge per step the moment a
    second backend lands. A stage marked `retained` leaves its output in the engine and the next stage
    names it, so nothing tensor-shaped crosses the boundary at all: since P4.0.14 the last stage retains
    too, and the only value the chain produces host-side is the one integer `ArgmaxEpilogue` returns.
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
                retain=stage.retained,
            )
            for stage in self.stages
        ]


@dataclass
class ArgmaxEpilogue(DriverComponent):
    """Returns the next token rather than the raw logits array.

    Argmaxes the logits row for the active (last real) token, in one of two modes that differ only in
    where the row is read from:

    * `retained_module` set -- the producing call left its output in that module's own buffer, so the
      reduction is `loom.argmax_row('<module>', row)` and nothing tensor-shaped crosses the boundary.
      This is the mode every causal LM uses (BACKLOG.md P4.0.14): a marshalled logits table lands in
      LuaJIT's array part, which caps a 262144-wide vocab at ~512 prompt tokens.
    * otherwise -- the producing call marshalled, and `shape_var[1]` is the output's ne0 (vocab size),
      the same convention `transpile_operation`'s own "argmax" case relies on. The `type(...) ==
      'table'` guard is what keeps this correct for a topology whose output is not an array -- the
      engine hands back a scalar there, and argmaxing it is meaningless.
    """

    out_var: Optional[str] = None
    shape_var: Optional[str] = None
    n_tokens: object = None
    # When set, the row is read out of this module's retained output rather than out of a Lua table, and
    # `out_var`/`shape_var` are unused because the producing call bound no locals at all.
    retained_module: Optional[str] = None

    __links__ = {
        "retained_module": WhenSet(TopologyName()),
    }
    __unchecked__ = {
        "out_var": Unchecked("a local an earlier component bound. The read is checked by "
                             "driver_ir.validate over the assembled function, which is where a "
                             "cross-component symbol read is answerable and nowhere else"),
        "shape_var": Unchecked("same -- the shape local the calling component captured"),
        "n_tokens": Unchecked("the driver_ir expression for the active row; see DriverInputs.n_tokens"),
    }

    def emit(self, ctx: DriverContext) -> List:
        row = BinOp("-", self.n_tokens, Lit(1))
        if self.retained_module is not None:
            # No `type(...) == 'table'` guard, because there is no value here to ask the question of:
            # the engine reduces the tensor it already holds, and a topology whose output is not
            # reducible fails in the bridge naming the module rather than silently returning it. That
            # the module really did retain is checked at export time by
            # driver_ir.check_subgraph_calls, which is the only checker that knows what a module is.
            return [Return([RetainedArgmax(self.retained_module, row)])]
        return [If(
            cond=BinOp("==", Call("type", [Var(self.out_var)]), Lit("table")),
            then=[Return([Call("loom.argmax_row",
                               [Var(self.out_var), Index(Var(self.shape_var), 1), row])])],
            else_=[Return([Var(self.out_var)])],
        )]


@dataclass
class CtcGreedyEpilogue(DriverComponent):
    """Greedy CTC decode: per-frame argmax, then collapse consecutive duplicates and drop the blank.

    The ASR counterpart to `ArgmaxEpilogue`, and the reason the two are different components rather than
    modes of one: a causal LM reduces ONE row of its output and returns one token, while a CTC model
    reduces every row and returns a sequence. `ArgmaxEpilogue` applied to a CTC encoder is not merely
    unhelpful, it raises -- it argmaxes row `n_tokens - 1`, and for these topologies `n_tokens` is the
    *sample* count while the output has one row per subsampled frame (BACKLOG.md P4.0.17).

    **This is `src/core/ctc_decode.cpp` retired, not relocated.** The reduction is engine-side because
    the logits are (`loom.argmax_rows`, one call, n_frames numbers back, the tensor never marshalled);
    the collapse is here because blank-and-duplicate handling is this family's own convention, and a
    `loom.ctc_greedy_decode` binding would have put family-specific logic in an engine whose whole claim
    is that a new family costs Python and not C++.

    Returns token ids, not text: detokenization is the caller's, exactly as it is for `whisper_driver`
    and every causal LM here.
    """

    retained_module: str = "main_topology"
    blank_id: object = None

    # Locals this component binds. Prefixed so they cannot collide with a traced input's own safe_name.
    frames_var: str = "_ctc_frames"
    out_var: str = "_ctc_out"
    prev_var: str = "_ctc_prev"
    index_var: str = "_ctc_i"
    id_var: str = "_ctc_id"

    __links__ = {
        "retained_module": TopologyName(),
    }
    __unchecked__ = {
        "blank_id": Unchecked(
            "the blank class index, which for a NeMo CTC head is the LAST class -- derived by the "
            "exporter from the traced output's own channel count (`decoder.num_classes_with_blank`, "
            "read off the checkpoint by EncoderOutput.expected_channels) rather than claimed here. "
            "Nothing else in the artifact states it, so there is no second authority to check against."
        ),
        "frames_var": Unchecked("a local this component binds; every read of it is inside this "
                                "component's own emitted block and is checked by driver_ir.validate "
                                "over the assembled function"),
        "out_var": Unchecked("same"),
        "prev_var": Unchecked("same"),
        "index_var": Unchecked("same -- the loop variable, scoped to the loop body by NumericFor"),
        "id_var": Unchecked("same"),
    }

    def emit(self, ctx: DriverContext) -> List:
        blank = Lit(self.blank_id)
        return [
            Local(self.frames_var, RetainedArgmaxRows(self.retained_module)),
            Local(self.out_var, ArrayLit([])),
            # Seeded with the blank so a leading real token is not deduplicated away -- the same seed,
            # for the same reason, as the C++ implementation this replaces.
            Local(self.prev_var, blank),
            NumericFor(
                var=self.index_var, start=Lit(1), stop=Len(self.frames_var),
                body=[
                    Local(self.id_var, Index(Var(self.frames_var), Var(self.index_var))),
                    If(
                        cond=BinOp("and",
                                    BinOp("~=", Var(self.id_var), Var(self.prev_var)),
                                    BinOp("~=", Var(self.id_var), blank)),
                        then=[CallStmt(Call("table.insert", [Var(self.out_var), Var(self.id_var)]))],
                    ),
                    Assign(self.prev_var, Var(self.id_var)),
                ],
            ),
            Return([Var(self.out_var)]),
        ]


@dataclass
class PrefillDecodeLoop(DriverComponent):
    """The generation loop `infer_with_past` is: prefill the prompt, then decode one token at a time
    against the KV cache, until `max_new_tokens` or `eos_token` (KV-CACHE.md 3.3).

    **One loop, not a prefill followed by a decode loop**, and that is the whole reason this is short.
    A cached `ATTENTION` node appends this step's K/V at `[n_past, n_past + n_tokens)` and attends over
    `[0, n_kv)` -- so a prefill *is* the first iteration, at `n_tokens = #prompt, n_past = 0`, and every
    later iteration is the same call at `n_tokens = 1`. The engine supplies the past; there is no second
    traced graph and no `decoder_with_past` (KV-CACHE.md §2, which is where that step was measured away).

    **Why the loop lives here rather than in the host** (decision 3): a caller-supplied `n_past` on a
    single cached step would put generation back in whoever embeds the engine, which is the per-model
    C++ driver shape this architecture is retiring. `infer` is untouched beside it -- one prefill, argmax
    the last row, one token -- so a host that only wants logits-shaped behaviour still has it.

    `max_new_tokens` defaults to 16 and `eos_token` to -1, the "negative disables the check" convention
    `WhisperConfig::eot_token` and `whisper_driver.lua` already use, so `infer_with_past{tokens = ...}`
    is a complete call.
    """

    topology: str = "main_topology"
    # (name, kind) in the traced function's own declared-input order, the same form DriverInputs takes.
    bindings: Tuple[Tuple[str, str], ...] = ()
    inputs: Tuple[str, ...] = ()
    default_max_new_tokens: int = 16
    # What `inputs.eos_token` falls back to. -1 keeps the "negative disables the check" convention for
    # every family that has always let the caller name it; a family whose checkpoint STATES its eos
    # binds it here instead, so a host does not carry a per-model token id (BACKLOG.md P4.3).
    default_eos_token: int = -1
    # Further ids that also end generation. A chat-formatted checkpoint has two -- the base model's
    # end-of-text and the chat turn's end -- and a loop knowing only the first runs to
    # `max_new_tokens` on every utterance. Empty for every family with a single eos, which emits
    # exactly the Lua it emitted before this field existed.
    extra_eos_tokens: Tuple[int, ...] = ()
    # {input name: sliding-window width} for the masks that are banded rather than full-causal
    # (BACKLOG.md P4.0.11a). Empty for every model that has no windowed attention, which is every model
    # on this roadmap but Gemma 3 -- and an absent entry means "full causal", so the emitted call is
    # byte-identical to what it was before this field existed.
    mask_windows: dict = dataclasses.field(default_factory=dict)
    # {input name: the IR expression supplying it}, for inputs that are neither host-computed from
    # n_tokens/n_past nor the step's own tokens -- an earlier component's output, held constant across
    # every iteration (BACKLOG.md P4.1).
    #
    # **This is what makes the loop a cross-attention decode loop**, and it is one field rather than a
    # new component because nothing else about the loop changes: a Whisper decoder step is the same
    # cached call at `n_tokens = 1`, with `xa` bound to the encoder phase's single run. Empty for a
    # plain causal LM, whose only input that is not host-computed IS the step's tokens.
    bound: dict = dataclasses.field(default_factory=dict)
    # The IR expression the loop's first iteration starts from, defaulting to the caller's own
    # `inputs.tokens`. An earlier component may instead have BUILT the prompt -- Whisper's is a
    # checkpoint-dependent prefix, with a language the driver may have had to detect -- in which case
    # this names that local and the caller never sees a token prefix at all (BACKLOG.md P4.1 follow-up).
    prompt: object = None
    # Tokens an earlier component already obtained from the model, which the loop's output starts from
    # rather than an empty array. Whisper's is the case: when timestamps are asked for, the token after
    # the task must be a timestamp, so the driver picks that one itself with a restricted argmax --
    # the model produced it, so it belongs in what the loop returns, even though it is also fed back in
    # as part of the prompt (BACKLOG.md P4.1).
    generated_prefix: object = None
    # The topology that turns this step's TOKEN IDS into the embeddings the decoder actually takes, for
    # a family whose decoder is traced on `inputs_embeds` rather than on ids (BACKLOG.md P4.3, family
    # 3). Empty for every family whose decoder takes ids, which is every family before this one.
    #
    # It is a topology rather than a flag because the embedding table is a traced graph like any other:
    # the step's tokens go in, its output is retained, and the decoder's embedding input is bound to it
    # by `OutputRef` -- so a token id is still the only thing that crosses the Lua boundary per step.
    embed_topology: Optional[str] = None
    # The topology that turns the decoder's HIDDEN STATES into logits, when the decoder phase stops
    # short of its own head (BACKLOG.md P4.3). Empty when the decoder emits logits directly.
    #
    # **This exists for a cost reason, not a structural one.** A family-3 prompt is dominated by audio
    # rows -- 143 of Qwen3-ASR's 158 for eleven seconds of speech -- and a head inside the decoder graph
    # would project every one of them through a 151936-wide vocabulary that nothing reads. Splitting it
    # off lets the driver run the head only where a token is genuinely needed.
    head_topology: Optional[str] = None
    # Where the loop's first iteration starts in the KV cache. Defaults to 0, which is the plain
    # prefill: the loop's own prompt is the whole prompt. A family whose prompt was fed to the cache in
    # SEGMENTS before the loop (`PromptSegments`) passes the running total it left behind, so the
    # loop's first iteration is simply the last segment.
    initial_n_past: object = None

    # Locals this component binds. Prefixed so they cannot collide with a traced input's own safe_name.
    generated_var: str = "_gen"
    step_var: str = "_step_tokens"
    n_past_var: str = "_n_past"
    n_tokens_var: str = "_n_tokens"

    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology"), exact=True),
        # Optional, so `WhenSet`: a family whose decoder takes token ids and emits its own logits sets
        # neither, and must not be asked to name topologies it has no reason to have.
        "embed_topology": WhenSet(TopologyName()),
        "head_topology": WhenSet(TopologyName()),
    }
    __unchecked__ = {
        "bindings": Unchecked(
            "the traced graph's own declared input names and kinds, READ off the MIL function rather "
            "than claimed about it -- same as DriverInputs.bindings, and the call site built from them "
            "is what `inputs` checks."
        ),
        "mask_windows": Unchecked(
            "read off the EMITTED topology by `LoomGGUFExporter._route_windowed_masks`, which "
            "synthesizes the windowed mask inputs in the first place -- so the exporter is the "
            "authority and there is no separate claim here for a checkpoint to disagree with."
        ),
        "bound": Unchecked(
            "driver_ir expressions over locals earlier components bind, exactly like "
            "SubgraphCallComponent.length -- validate() over the assembled function is their authority. "
            "The input NAMES they are keyed by are checked, by this component's own `inputs` link, "
            "which is exact against the topology's real declared inputs."
        ),
        "prompt": Unchecked(
            "same: a driver_ir expression over a local an earlier component bound, whose reads "
            "driver_ir.validate resolves over the assembled function -- a name nothing binds fails the "
            "export rather than reading nil at run time."
        ),
        "generated_prefix": Unchecked("same -- an expression over an earlier component's local, "
                                       "resolved by driver_ir.validate"),
        "initial_n_past": Unchecked(
            "a driver_ir expression over a local an earlier component bound, exactly like `prompt`; "
            "driver_ir.validate resolves it over the assembled function, so a name nothing binds "
            "fails the export rather than reading nil at run time"
        ),
        "default_max_new_tokens": Unchecked(
            "a default for a caller-supplied argument, not a claim about the model. Nothing in the "
            "checkpoint could disagree with it."
        ),
        "default_eos_token": Unchecked(
            "READ off the checkpoint's own generation config by the family that binds it, and a "
            "default for a caller-supplied argument otherwise. The one thing a link could compare it "
            "against is that same generation config, which is where the value came from."
        ),
        "extra_eos_tokens": Unchecked("same -- the remaining ids the checkpoint's generation config "
                                      "lists, read rather than claimed"),
        "generated_var": Unchecked("a local this component binds; reads of it are inside its own loop "
                                   "and are checked by driver_ir.validate over the assembled function"),
        "step_var": Unchecked("same"),
        "n_past_var": Unchecked("same"),
        "n_tokens_var": Unchecked("same"),
    }

    def _call_inputs(self, ctx: DriverContext):
        """The `run_subgraph` input table: the step's tokens for the caller-supplied input, and the
        host-computed pair rebuilt per iteration -- which is the difference from `DriverInputs`, whose
        `cache_position`/`attention_mask` are built once for one prefill."""
        out = {}
        for name, kind in self.bindings:
            if name in self.bound:
                # Checked before `kind` deliberately: an explicitly bound input wins over what its name
                # would otherwise imply, so a model whose encoder output happens to be called
                # `attention_mask` is still bound to the encoder rather than to loom.causal_mask.
                out[name] = self.bound[name]
            elif kind == CALLER and self.embed_topology:
                # The step's tokens reach a `inputs_embeds`-traced decoder through the embedding
                # topology this loop just ran, backend-side. `OutputRef` rather than a marshalled
                # table for the same reason Whisper's `xa` is one: a hidden-size-wide row per token
                # would otherwise cross the Lua boundary on every step.
                out[name] = OutputRef(self.embed_topology)
            elif kind == POSITION:
                out[name] = Call("loom.range", [Var(self.n_past_var), Var(self.n_tokens_var)])
            elif kind == MASK:
                args = [Var(self.n_tokens_var), Var(self.n_past_var)]
                window = self.mask_windows.get(name)
                if window:
                    args.append(Lit(window))
                out[name] = Call("loom.causal_mask", args)
            else:
                out[name] = Var(self.step_var)
        return out

    def _step_body(self, ctx: DriverContext) -> List:
        """The calls one iteration makes, before the token is read.

        One call for a family whose decoder takes ids and emits logits; up to three for a composition
        (`embed` -> `decoder` -> `lm_head`), each retained so that only the token id itself is ever
        marshalled. The middle one is the same cached call in both cases -- what changes is where its
        input comes from and where its output goes, which is exactly what the two topology fields say.
        """
        body = []
        if self.embed_topology:
            body.append(SubgraphCall(
                outputs=[], module=self.embed_topology,
                axes={ctx.root_axis(self.embed_topology): Var(self.n_tokens_var), "n_past": Lit(0)},
                inputs={ctx.primary_input(self.embed_topology): Var(self.step_var)},
                retain=True,
            ))
        body.append(SubgraphCall(
            outputs=[], module=self.topology,
            axes={ctx.root_axis(self.topology): Var(self.n_tokens_var),
                  "n_past": Var(self.n_past_var)},
            inputs=self._call_inputs(ctx),
            retain=True,
        ))
        if self.head_topology:
            body.append(SubgraphCall(
                outputs=[], module=self.head_topology,
                axes={ctx.root_axis(self.head_topology): Var(self.n_tokens_var), "n_past": Lit(0)},
                inputs={ctx.primary_input(self.head_topology): OutputRef(self.topology)},
                retain=True,
            ))
        return body

    def emit(self, ctx: DriverContext) -> List:
        next_var = "_next_token"
        max_new = "_max_new_tokens"
        eos = "_eos_token"
        # Whichever module actually holds the logits: the decoder itself, or the head phase it was
        # split into. One name, so the argmax and the call that produced it cannot disagree.
        logits_module = self.head_topology or self.topology
        return [
            Local(self.generated_var,
                  self.generated_prefix if self.generated_prefix is not None else ArrayLit([])),
            Local(self.step_var,
                  self.prompt if self.prompt is not None
                  else FieldAccess("inputs", GENERIC_PRIMARY_INPUT)),
            Local(self.n_past_var,
                  self.initial_n_past if self.initial_n_past is not None else Lit(0)),
            Local(self.n_tokens_var, Len(self.step_var)),
            Local(max_new, BinOp("or", FieldAccess("inputs", "max_new_tokens"),
                                 Lit(self.default_max_new_tokens))),
            Local(eos, BinOp("or", FieldAccess("inputs", "eos_token"),
                             Lit(self.default_eos_token))),
            While(cond=Lit(True), body=[
                # Retain, then reduce by name: the logits stay in the module's own buffer and the only
                # thing a decode step ships across the boundary is the token id. That is what removes
                # the ~512-token prefill ceiling a 262144-wide vocab hits on the first iteration
                # (BACKLOG.md P4.0.14) -- and every later iteration, at n_tokens = 1, gets the same
                # reduction for free. Two statements rather than one fused call because they are two
                # facts, and `loom.run_subgraph_and_retain` composes with every other way a retained
                # output is read; a second, fused spelling of the same reduction is what P4.0.14
                # retired.
                *self._step_body(ctx),
                Local(next_var, RetainedArgmax(logits_module,
                                               BinOp("-", Var(self.n_tokens_var), Lit(1)))),
                CallStmt(Call("table.insert", [Var(self.generated_var), Var(next_var)])),
                Assign(self.n_past_var, BinOp("+", Var(self.n_past_var), Var(self.n_tokens_var))),
                If(cond=BinOp(">=", Len(self.generated_var), Var(max_new)), then=[Break()]),
                If(cond=BinOp("==", Var(next_var), Var(eos)), then=[Break()]),
                *[If(cond=BinOp("==", Var(next_var), Lit(alt)), then=[Break()])
                  for alt in self.extra_eos_tokens],
                # Every later step is the same call at n_tokens = 1: the cache holds the past, so the
                # graph only ever computes K/V for what it is handed.
                Assign(self.step_var, ArrayLit([Var(next_var)])),
                Assign(self.n_tokens_var, Lit(1)),
            ]),
            Return([Var(self.generated_var)]),
        ]


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
    """One traced graph, run once over the whole prompt, argmax the last row -- and, when the export is
    fused, a `infer_with_past` generation loop beside it.

    `infer` is the same three components it has always been: `n_past` bound to `Lit(0)`, the mask built
    for a fresh sequence, one token out. That is a *prefill*, and it stays one -- a host wanting a
    single forward pass has it unchanged.

    `decode` is set only for a topology whose ATTENTION nodes carry a KV cache (KV-CACHE.md stage 2 is
    what produces those; before it, a MIL-exported causal LM had none and the loop had nothing to loop
    over). Its statements land in a second entry function, not in `infer`, so which one a caller reaches
    for is a call-site decision rather than an export-time one.
    """

    inputs: DriverInputs
    call: MonolithicCall
    epilogue: ArgmaxEpilogue
    decode: Optional[PrefillDecodeLoop] = None

    __links__ = {name: NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN)
                 for name in ("inputs", "call", "epilogue")}
    __unchecked__ = {
        "decode": CoveredBy(
            "the same NestedSpec reasoning as the three fields above -- DriverBuilder.build registers "
            "it with the export's checker like any other component. It is declared separately only "
            "because it is optional: a WhenSet wrapper would read as a weaker claim than the one that "
            "actually runs."
        ),
    }

    def components(self):
        return [self.inputs, self.call, self.epilogue]

    def extra_entries(self):
        if self.decode is None:
            return []
        return [DriverEntry(name="infer_with_past", params=("inputs",), components=[self.decode])]


@dataclass
class CtcGreedyBuilder(DriverBuilder):
    """One traced graph over the whole waveform, then greedy CTC decode -- the NeMo Conformer-CTC shape
    (BACKLOG.md P4.0.17).

    Shares `DriverInputs` and `MonolithicCall` with `PrefillArgmaxBuilder` and differs only in the
    epilogue, which is the honest statement of how these two families relate: the same single forward
    pass, a different reduction over its output. That the difference is exactly one component is why the
    ASR encoders needed a builder rather than a special case inside the causal-LM one.

    `MonolithicCall` retains here for the reason it retains for a large-vocab LM: the reduction happens
    engine-side, so the `[n_classes, n_frames]` logits never become a Lua table.
    """

    inputs: DriverInputs
    call: MonolithicCall
    epilogue: CtcGreedyEpilogue

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


# Which builder each decomposition's *synthesized* driver path uses, by decomposition class name.
# `LoomGGUFExporter.apply_monolithic_export`/`apply_modular_export` construct through this table rather
# than naming the classes directly, so P4.0.7's catalogue can attribute components to models -- "which
# models use `argmax_epilogue`" -- without a second, hand-maintained copy of the mapping to drift from.
# `MultiPhase` is absent by construction: its builder is selected per family by
# `MultiPhase.driver_builder`, because a peeled family's component list is the family's own.
SYNTHESIZED_BUILDERS = {
    "Flattened": PrefillArgmaxBuilder,
    "Modular": ModularChainBuilder,
    # Keyed by decomposition like the two above, EXCEPT this one, which is keyed by orchestration --
    # and the exception is the finding P4.0.17 records rather than a shortcut. Conformer-CTC IS a
    # `Flattened` export; what differs is what its host does with the one output, and
    # `Decomposition.driver_builder`'s premise ("the orchestration shape a driver has is a property of
    # how the model was decomposed") simply does not hold for it. The ASR config asks for this by name
    # through `backend_kwargs()`, so the request travels with the family that has it rather than being
    # inferred from a topology.
    "CtcGreedy": CtcGreedyBuilder,
}


# -- adopting a hand-written driver (C.3) ------------------------------------------------------------
#
# The five multi-phase TTS families shipped a hand-written `.lua` apiece. `RawLuaDriver` adopts one
# whole, unchanged, so the family moves onto the builder in a step whose gate is byte-identity -- and
# only then gets peeled into real components one at a time (C.4-C.8), each peel independently
# revertable. All five are peeled, so nothing constructs this today; it is the first step the *next*
# hand-written driver takes, and its registry entry is where that is argued.
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
# Both spellings that run a registered module from hand-written Lua. `run_subgraph_and_retain` was
# invisible to this parser until Parakeet's decode fragment used it -- the fragment's `joint` call site
# simply was not seen, and the coverage check reported the topology as named by nobody. A blind spot in
# the checker reads exactly like a driver that does not call something, which is the failure mode this
# whole parse exists to prevent (BACKLOG.md P4.0.12 added the retaining form; nothing taught D.2's
# machinery about it).
_RUN_SUBGRAPH_FORMS = ("loom.run_subgraph(", "loom.run_subgraph_and_retain(")
_RUN_SUBGRAPH = _RUN_SUBGRAPH_FORMS[0]
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
    sites = []
    for form in _RUN_SUBGRAPH_FORMS:
        index = source.find(form)
        while index != -1:
            args_text, end = _balanced_args(source, index + len(form))
            sites.append((index, args_text))
            index = source.find(form, end)
    # `run_subgraph(` is a prefix of nothing else here, but `run_subgraph_and_retain(` is NOT found by
    # the plain form, so the two scans are independent and their results interleave by position.
    for index, args_text in sorted(sites):
        line = source.count("\n", 0, index) + 1
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
    entry: str = "infer"
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


# -- peeling a driver into components (C.4-C.8) ------------------------------------------------------
#
# `RawLuaDriver` adopts a driver whole; these are what its blocks become one at a time. The rule the
# author set for where the peeled Lua goes is that it stays **Lua, in `.lua` files** -- a family's
# driver becomes a directory of small fragments plus a component list that orders them and declares
# what each one reads and defines. The alternative (Lua as Python string literals) puts the
# hand-written half of every TTS model behind a quoting layer, and the point of the exercise is to make
# these drivers easier to reason about, not harder to read.


_NOTE_IS_COSMETIC = Unchecked(
    "a comment emitted above this component's statement. Cosmetic to the engine and load-bearing to a "
    "reader: the embedded driver_script is what someone inspecting a GGUF sees, and a peel that "
    "dropped every explanatory comment from the hand-written driver would be a real loss even though "
    "no test could fail for it"
)


def _note_block(note):
    """A `note` as Lua comment lines, or nothing.

    Led by a blank line, because a peeled driver is read in sections and every hand-written driver in
    this tree separates them that way. A fragment that wants the same spacing carries it in its own
    `.lua` file, where it is data rather than a rule this function would have to guess."""
    if not note:
        return []
    return [RawBlock([""] + [f"-- {line}".rstrip() for line in note.strip().split("\n")])]


@dataclass
class HelperCall:
    """One call to a `loom_lua` helper that drives topologies whose names the Lua computes (D.2).

    `run_bi_lstm("text_encoder_lstm", ...)` runs four `loom.run_subgraph` calls, one per cell topology,
    with the name built at run time inside the library function. Nothing could check those: the parser
    reads literals, and the call sites are not even in this fragment. What the *caller* knows, and only
    the caller, is the namespace -- so it declares it, and the library's `DrivenTopologies` supplies the
    shape (`_h_fwd`/`_c_fwd`/`_h_bwd`/`_c_bwd`, and the input table each call supplies).

    The expansion is an ordinary `RunSubgraphCall` per topology, so a namespace whose cells the export
    did not produce fails with `TopologyName`'s own message and the list of names that do exist -- the
    same failure a mistyped literal gets, which is the point: after D.2 there is no second class of call
    site with weaker checking.

    `written` is the first argument exactly as the fragment writes it, and is what keeps the declaration
    from rotting: a namespace renamed in the Lua and not here (or the reverse) is caught by
    `LuaFragment`'s own both-way check. It defaults to the quoted namespace, which is what a
    single-namespace call site looks like; a loop passes its expression (`'"duration_lstm_" .. i'`).
    """

    helper: str
    # One namespace, or the several a loop passes.
    namespaces: object
    written: Optional[str] = None
    origin: str = "driver"
    line: int = 0

    __links__ = {
        "helper": ConfigDerived(
            claim=lambda spec, ctx: True,
            measured=lambda spec, ctx: spec.driven() is not None,
            detail=lambda spec, ctx: spec.helper,
            message=(
                "{label} names loom_lua function {detail!r}, which either does not exist or declares "
                "no `drives`. Only a library function that calls loom.run_subgraph with a computed "
                "name has topologies to declare -- see lua_library.DrivenTopologies."
            ),
            needs=(),
        ),
        "namespaces": NestedSpec(
            where="LuaFragment.sub_specs, which expands each namespace into one RunSubgraphCall per "
                  "topology the helper drives, so each is checked against the real exported topology "
                  "and its declared inputs"
        ),
    }
    __unchecked__ = {
        "written": Unchecked(
            "the call's first argument as written. Checked, but by the fragment that contains it and "
            "in both directions -- a declaration whose text is not in the fragment, and a call site in "
            "the fragment that no declaration covers, are both failures there. Restating it as a link "
            "here would only see one of those"
        ),
        "origin": Unchecked("the fragment file this call was declared in, for the label"),
        "line": Unchecked("where in that fragment, for the label. Filled in by LuaFragment from the "
                          "parsed call site rather than declared by hand"),
    }

    def link_label(self) -> str:
        where = f"{self.origin}:{self.line} " if self.line else f"{self.origin} "
        return f"{where}{self.helper}({self.written or self.names()[0]!r})"

    def names(self) -> Tuple[str, ...]:
        return (self.namespaces,) if isinstance(self.namespaces, str) else tuple(self.namespaces)

    def as_written(self) -> str:
        if self.written is not None:
            return self.written
        return f'"{self.names()[0]}"'

    def driven(self):
        from .lua_library import LIBRARY

        entry = LIBRARY.get(self.helper)
        return None if entry is None else entry.drives

    def expand(self) -> List["RunSubgraphCall"]:
        driven = self.driven()
        if driven is None:
            # The `helper` link reports this with the real reason; expanding to nothing here keeps that
            # message the one a reader sees instead of an AttributeError from underneath it.
            return []
        return [
            RunSubgraphCall(topology=topology, inputs=tuple(driven.inputs),
                            line=self.line, origin=f"{self.origin} via {self.helper}")
            for namespace in self.names()
            for topology in driven.topologies(namespace)
        ]


@dataclass
class ComputedCall:
    """One `loom.run_subgraph` whose topology name the fragment itself computes (D.2).

    The same declaration as `HelperCall` for the case with no helper in between: Kokoro's and
    StyleTTS2's duration encoders call `loom.run_subgraph("duration_adaln_" .. i, ...)` inside a Lua
    `for`, so the name exists only at run time. The loop is the right way to write that -- three
    unrolled copies would be worse -- and this is what makes it checkable anyway.

    `inputs` is stated rather than parsed even though the table literal is right there, because the
    parse is what has already failed: `parse_run_subgraph_calls` reports these sites as unresolved and
    declares nothing about them. Stating both halves keeps one declaration to compare against the real
    topology.
    """

    topologies: Tuple[str, ...]
    inputs: Tuple[str, ...]
    written: str
    origin: str = "driver"
    line: int = 0

    __links__ = {
        "topologies": NestedSpec(
            where="LuaFragment.sub_specs, which declares one RunSubgraphCall per name so each is "
                  "checked against the real topology and its declared inputs"
        ),
    }
    __unchecked__ = {
        "inputs": CoveredBy("topologies"),
        "written": Unchecked(
            "the topology-name expression as written. Checked by the fragment that contains it, in "
            "both directions -- see HelperCall.written"
        ),
        "origin": Unchecked("the fragment file this call was declared in, for the label"),
        "line": Unchecked("where in that fragment, for the label"),
    }

    def link_label(self) -> str:
        where = f"{self.origin}:{self.line} " if self.line else f"{self.origin} "
        return f"{where}loom.run_subgraph({self.written})"

    def as_written(self) -> str:
        return self.written

    def expand(self) -> List["RunSubgraphCall"]:
        return [
            RunSubgraphCall(topology=topology, inputs=tuple(self.inputs),
                            line=self.line, origin=self.origin)
            for topology in self.topologies
        ]


def _helper_call_sites(text: str):
    """`[(helper name, first-argument text, line)]` for every call in `text` to a `loom_lua` function
    that drives topologies.

    Read off the fragment rather than declared, because this is the half that has to find what the
    declarations *miss*: a helper call nobody declared is a call site with no check on it, which is
    precisely the state D.2 exists to end."""
    from .lua_library import LIBRARY

    sites = []
    for name, entry in LIBRARY.items():
        if entry.drives is None:
            continue
        marker = f"{name}("
        index = text.find(marker)
        while index != -1:
            # A definition is not a call site. `local function run_bi_lstm(` only appears in the
            # library file itself, but a fragment that inlined one would otherwise read as a caller.
            prefix = text[:index].rstrip()
            if not prefix.endswith("function"):
                args_text, _ = _balanced_args(text, index + len(marker))
                args = _split_top_level(args_text)
                sites.append((name, args[0].strip(), text.count("\n", 0, index) + 1))
            index = text.find(marker, index + len(marker))
    return sites


@dataclass
class LuaFragment(DriverComponent):
    """One hand-written block of a peeled driver, kept as its own `.lua` file.

    `reads`/`defines` are the declaration that makes a fragment composable: `driver_ir.validate` runs
    over the assembled function, so a fragment placed before the component that binds what it reads
    fails at export time rather than inside the engine. That is a check no hand-written driver has ever
    had, and it is the one that actually matters while blocks are being moved around.

    **What the declaration is and is not.** Both lists are checked for *presence* in the fragment's own
    text, which catches the rot that actually happens -- renaming a local in the `.lua` and leaving the
    declaration behind. Neither is derived from a Lua parse, so an *under*-declared `reads` is not
    caught here: it makes `validate` blind to that one ordering, and the family's e2e test is what
    covers it. Saying so is the point; a declaration that claimed more than it checks would be worse
    than none.
    """

    path: object  # Path
    reads: Tuple[str, ...] = ()
    defines: Tuple[str, ...] = ()
    # Emit as top-level Lua before the entry function rather than as statements inside it. Every
    # driver's header comment and its `local function` helpers are this.
    top_level: bool = False
    # `{topology name: where it comes from}` for calls inside this fragment that this export does not
    # produce -- see `BaseMultiPhaseModelExportConfig.external_topologies`.
    external: dict = dataclass_field(default_factory=dict)
    # `HelperCall`/`ComputedCall` declarations for this fragment's call sites whose topology name is
    # computed at run time (D.2). Empty for a fragment that names every topology literally.
    drives: Tuple[object, ...] = ()

    __links__ = {
        "drives": [
            ConfigDerived(
                claim=lambda spec, ctx: (),
                measured=lambda spec, ctx: tuple(spec.undeclared_call_sites()),
                detail=lambda spec, ctx: [f"line {line}: {text}"
                                          for _, text, line in spec.undeclared_call_sites()],
                message=(
                    "{label} has computed call site(s) {detail} that no `drives` declaration covers, "
                    "so the topologies they run are checked by nothing. Declare them with a "
                    "HelperCall (a loom_lua helper that drives topologies) or a ComputedCall (a "
                    "loom.run_subgraph whose name this fragment computes)."
                ),
                needs=(),
            ),
            ConfigDerived(
                claim=lambda spec, ctx: (),
                measured=lambda spec, ctx: tuple(spec.stale_drives_declarations()),
                detail=lambda spec, ctx: list(spec.stale_drives_declarations()),
                message=(
                    "{label} declares call site(s) {detail} that its text does not contain. A "
                    "declaration whose call site was renamed or deleted checks a topology nothing "
                    "runs, which reads as coverage and is not."
                ),
                needs=(),
            ),
        ],
        "defines": ConfigDerived(
            claim=lambda spec, ctx: (),
            measured=lambda spec, ctx: tuple(spec.names_missing_from_text(spec.defines)),
            detail=lambda spec, ctx: list(spec.names_missing_from_text(spec.defines)),
            message=(
                "{label} declares that it defines {detail}, but its text never mentions those names. "
                "A stale defines list makes driver_ir.validate accept an ordering the driver does not "
                "actually support."
            ),
            needs=(),
        ),
        "reads": ConfigDerived(
            claim=lambda spec, ctx: (),
            measured=lambda spec, ctx: tuple(spec.names_missing_from_text(spec.reads)),
            detail=lambda spec, ctx: list(spec.names_missing_from_text(spec.reads)),
            message=(
                "{label} declares that it reads {detail}, but its text never mentions those names. "
                "A stale reads list makes driver_ir.validate reject orderings that are in fact fine, "
                "and hides the ones that are not."
            ),
            needs=(),
        ),
    }
    __unchecked__ = {
        "path": Unchecked(
            "the fragment file. `read_text()` reports a missing one with the path and the errno, "
            "which is strictly better than a link saying it does not exist"
        ),
        "top_level": Unchecked(
            "whether this fragment is a Lua statement or a top-level definition. Getting it wrong "
            "produces Lua that does not load, which the family's e2e test catches immediately -- "
            "there is no authority to check it against beyond the language itself"
        ),
        "external": Unchecked(
            "which of this fragment's calls come from another GGUF, forwarded from the config's "
            "external_topologies(). Checked there, in both directions, against the whole driver -- "
            "restating it per fragment would report a name as dead merely because a different "
            "fragment is the one that calls it"
        ),
    }

    def __post_init__(self):
        self.lines = Path(self.path).read_text().rstrip("\n").split("\n")
        origin = Path(self.path).name
        self._calls, self.unresolved_calls = parse_run_subgraph_calls("\n".join(self.lines), origin)
        # A declaration carries the file and line of the site it covers, so a failure names where to
        # look -- read off the text rather than written out per declaration, since the fragment is the
        # thing that knows.
        self.drives = tuple(
            dataclasses.replace(declaration, origin=origin,
                                line=self._line_of(declaration.as_written()))
            for declaration in self.drives
        )

    def _line_of(self, written: str) -> int:
        for number, line in enumerate(self.lines, start=1):
            if written in line:
                return number
        return 0

    def link_label(self) -> str:
        return f"LuaFragment({Path(self.path).name!r})"

    def sub_specs(self):
        """`run_subgraph` calls still written by hand inside this fragment, literal and computed alike.

        A peel must never *reduce* checking, and without this it would: `RawLuaDriver` parses the whole
        adopted driver, so a block moved into a fragment would take its call sites out of reach. The
        calls that survive a peel are the ones a component cannot express -- a computed topology name,
        or a call inside a Lua loop -- and they are exactly the ones worth still parsing.

        Since D.2 the computed half is here too: a `HelperCall`/`ComputedCall` declaration expands into
        one `RunSubgraphCall` per topology it really runs, which is what closes the last call sites in
        Kokoro's and StyleTTS2's drivers that no check could reach.
        """
        declared = [call for declaration in self.drives for call in declaration.expand()]
        return ([call for call in self._calls if call.topology not in self.external]
                + [call for call in declared if call.topology not in self.external]
                + list(self.drives))

    def declared_call_sites(self) -> set:
        """The first-argument text of every call site this fragment's `drives` declarations claim."""
        return {declaration.as_written() for declaration in self.drives}

    def computed_call_sites(self) -> list:
        """`[(what, first-argument text, line)]` for every call site in this fragment whose topology
        name is not a literal: a `loom.run_subgraph` the fragment computes, and a call to a `loom_lua`
        helper that drives topologies of its own."""
        text = "\n".join(self.lines)
        sites = [("loom.run_subgraph", written, line) for line, written in self.unresolved_calls]
        sites.extend((helper, written, line) for helper, written, line in _helper_call_sites(text))
        return sites

    def undeclared_call_sites(self) -> list:
        declared = self.declared_call_sites()
        return [site for site in self.computed_call_sites() if site[1] not in declared]

    def stale_drives_declarations(self) -> list:
        found = {written for _, written, _ in self.computed_call_sites()}
        return [declaration.as_written() for declaration in self.drives
                if declaration.as_written() not in found]

    def names_missing_from_text(self, names) -> list:
        text = "\n".join(self.lines)
        return [n for n in names if not re.search(rf"\b{re.escape(n)}\b", text)]

    def prelude(self, ctx):
        return list(self.lines) + [""] if self.top_level else []

    def emit(self, ctx):
        if self.top_level:
            return []
        return [RawBlock(list(self.lines), verbatim=True,
                         reads_=list(self.reads), defines_=list(self.defines))]


@dataclass
class SubgraphCallComponent(DriverComponent):
    """One `loom.run_subgraph` call, as IR rather than as text.

    This is what a peel buys structurally: the call stops being a string that a regex can read and
    becomes a `SubgraphCall` node, so `driver_ir.check_subgraph_calls` covers it directly and the
    output arity is checked too -- neither of which a parsed `RunSubgraphCall` can do, since it cannot
    see how many values the surrounding Lua binds.
    """

    topology: str
    outputs: Tuple[str, ...]
    # {input name -> the IR expression supplying it}.
    inputs: dict
    # {axis name -> IR expression}. Defaults to the topology's own root axis bound to `length`, plus
    # `n_past = 0`, which is what every hand-written TTS call does.
    length: object = None
    extra_outputs: Tuple[str, ...] = ()
    axes: Optional[dict] = None
    note: Optional[str] = None
    multiline: bool = False
    # Keep this call's outputs in the module's own `OutputStore` instead of marshalling them into Lua
    # tables (BACKLOG.md P4.0.12). A later call reaches them with an `OutputRef` -- a backend-side
    # tensor copy -- so a chain edge never becomes a list of Lua doubles. Whisper's encoder is the case
    # that needs it: its output is `n_audio_ctx * d_model` floats (1.15M for whisper-small), read once
    # per decode step, and marshalling them would put that table through the boundary on every token.
    retain: bool = False

    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology"), exact=True),
    }
    __unchecked__ = {
        "note": _NOTE_IS_COSMETIC,
        "multiline": Unchecked("whether to render the input table one entry per line. Cosmetic, and "
                               "the same category as `note`: what a reader of the embedded driver "
                               "sees, which for a seven-input call is the difference between a block "
                               "and a 200-column line"),
        "outputs": Unchecked("the locals this call binds; reads of them are checked by "
                             "driver_ir.validate over the assembled function"),
        "extra_outputs": Unchecked("same, for the shape locals -- and whether capturing them is legal "
                                   "at all is check_subgraph_calls' question, answered against the "
                                   "topology's real declared output count"),
        "length": Unchecked("the driver_ir expression bound to this topology's root axis, over locals "
                            "earlier components bind; validate() is its authority"),
        "axes": Unchecked("an explicit axis table for the rare call that needs more than the root "
                          "axis; the names in it are checked by ExportPhase's own Axis links, which "
                          "run against the declaration that created them"),
        "retain": Unchecked(
            "whether this call keeps its outputs in the module's OutputStore. Not a claim about the "
            "model at all -- and the half that IS checkable is checked where it means something: "
            "driver_ir.check_subgraph_calls rejects an `OutputRef` naming a module no earlier call "
            "retained, which is the failure a wrong value here would cause."
        ),
    }

    def link_label(self) -> str:
        return f"loom.run_subgraph({self.topology!r})"

    def emit(self, ctx):
        axes = self.axes
        if axes is None:
            axes = {ctx.root_axis(self.topology): self.length, "n_past": Lit(0)}
        return _note_block(self.note) + [SubgraphCall(
            outputs=list(self.outputs), extra_outputs=list(self.extra_outputs),
            module=self.topology, axes=axes, inputs=dict(self.inputs), multiline=self.multiline,
            retain=self.retain,
        )]


def _names_a_topology(link) -> bool:
    """Does `link` assert that its field holds a topology name -- including through a wrapper?

    `MultiPhaseDriverBuilder.called_topologies` reads this off the declarations rather than off a list
    of component classes, precisely so a new component is counted the day it declares its link. A bare
    `isinstance(link, TopologyName)` undoes half of that: an OPTIONAL topology field is declared
    `WhenSet(TopologyName())`, whose wrapper is not a `TopologyName`, so the field is checked and yet
    invisible to the count. That is the same failure the comment at the call site describes, one
    wrapper deeper -- it reported family 3's `embed` and `lm_head` as "topologies no call site names"
    while the loop was calling both every step (BACKLOG.md P4.3).
    """
    return isinstance(link, TopologyName) or (
        isinstance(link, WhenSet) and _names_a_topology(link.inner)
    )


@dataclass
class PromptSegments(DriverComponent):
    """A prompt made of alternating text and non-text pieces, fed to a KV-cached decoder as one cached
    call per piece (BACKLOG.md P4.3, `EXPORT-ROADMAP.md` family 3's "embedding-injection driver").

    **Why this is a walk over segments rather than a concatenation.** A speech-LM prompt is text
    embeddings with the audio encoder's own output rows substituted in where a placeholder token sits.
    Building that as one tensor would need a backend-side concatenation of two retained values -- an
    engine op that does not exist, and one `OutputStore` has no shape for. It is not needed: attention
    is causal and the decoder is cached, so a call at `n_past = k` over `n` rows writes cells
    `[k, k+n)` and attends over `[0, k+n)`, which makes N successive cached calls the same arithmetic
    as one call over the concatenation. Measured against HF on Qwen3-ASR-0.6B before this component was
    written: 2.3e-04 on hidden states whose absmax is 95.7, and the same first token.

    **This component deliberately stops one segment short.** It emits the calls for every segment but
    the last and leaves `n_past_var` holding the running total, which `PrefillDecodeLoop.initial_n_past`
    picks up -- so the final text segment is the loop's own first iteration, exactly as a plain causal
    LM's prefill is its first iteration. Two components, one prompt, and no third spelling of "run the
    decoder over some rows".

    A `"text"` segment's expression is a Lua array of token ids, which this runs `embed_topology` over;
    a `"bound"` segment's is an `OutputRef` to a module that has already run and retained, whose row
    count the driver computes from the audio geometry rather than reading back.
    """

    topology: str = "main_topology"
    bindings: Tuple[Tuple[str, str], ...] = ()
    embed_topology: Optional[str] = None
    # ((kind, expression), ...) in prompt order, kind in {"text", "bound"}.
    segments: Tuple[Tuple[str, object], ...] = ()
    # How many decoder positions one chunk of audio occupies, and how many waveform samples one chunk
    # is. Both READ off the model in `phases()` and cross-checked against the encoder's real output
    # there -- the driver needs them because a `bound` segment's length is not something it can ask the
    # retained tensor for.
    audio_rows_per_chunk: int = 0
    samples_per_chunk: int = 0

    n_past_var: str = "_n_past"
    n_tokens_var: str = "_seg_tokens"

    __links__ = {
        "topology": TopologyName(),
        "embed_topology": WhenSet(TopologyName()),
    }
    __unchecked__ = {
        "bindings": Unchecked(
            "the traced decoder's own declared input names and kinds, READ off the MIL function "
            "through `exporter._binding_kind` -- the same list PrefillDecodeLoop is built from, so the "
            "two cannot disagree about this decoder's inputs"
        ),
        "segments": Unchecked(
            "driver_ir expressions over locals earlier components bind, plus an OutputRef per bound "
            "segment. validate() resolves the former over the assembled function and "
            "check_subgraph_calls rejects the latter if nothing retained that module, which are the "
            "two ways this list can be wrong."
        ),
        "audio_rows_per_chunk": Unchecked(
            "READ off the audio config and CROSS-CHECKED in BaseSpeechLMExportConfig.phases() against "
            "the number of rows the traced encoder actually emits for a known chunk count -- which is "
            "the check that matters, because a wrong value here places every later segment at the "
            "wrong n_past with no shape mismatch to reveal it"
        ),
        "samples_per_chunk": Unchecked("same cross-check: the sample count whose mel is exactly one "
                                       "encoder chunk"),
        "n_past_var": Unchecked("a local this component binds and PrefillDecodeLoop reads; "
                                "driver_ir.validate resolves that read over the assembled function"),
        "n_tokens_var": Unchecked("same"),
    }

    def link_label(self) -> str:
        return f"prompt segments -> loom.run_subgraph({self.topology!r})"

    def _call_inputs(self, embeds):
        """The decoder's input table for one segment: the embeddings, and the host-computed pair."""
        out = {}
        for name, kind in self.bindings:
            if kind == POSITION:
                out[name] = Call("loom.range", [Var(self.n_past_var), Var(self.n_tokens_var)])
            elif kind == MASK:
                out[name] = Call("loom.causal_mask", [Var(self.n_tokens_var), Var(self.n_past_var)])
            else:
                out[name] = embeds
        return out

    def emit(self, ctx: DriverContext) -> List:
        body: List = [
            _note_block(
                "Prompt: one cached decoder call per segment. The KV cache makes this identical to "
                "one call over the concatenated prompt, which is why no tensor is ever joined."
            )[0],
            Local(self.n_past_var, Lit(0)),
            LocalDecl(self.n_tokens_var),
        ]
        for kind, expr in self.segments:
            if kind == "text":
                body.append(Assign(self.n_tokens_var, Len(expr)))
                body.append(SubgraphCall(
                    outputs=[], module=self.embed_topology,
                    axes={ctx.root_axis(self.embed_topology): Var(self.n_tokens_var), "n_past": Lit(0)},
                    inputs={ctx.primary_input(self.embed_topology): expr},
                    retain=True,
                ))
                embeds = OutputRef(self.embed_topology)
            elif kind == "bound":
                # The audio segment's length: one row per `audio_rows_per_chunk` per chunk of waveform.
                # Computed rather than read back because a retained tensor's shape does not cross the
                # boundary -- and it is exactly the arithmetic the host already did to pad the audio.
                # `floordiv`, not `/`: Lua's `/` is float division, so a whole number of chunks would
                # still make this a float and the axis table would carry 143.0 where the engine binds
                # an integer extent. The waveform is a whole number of chunks by the encoder's own
                # contract, so the floor changes no value -- only its type.
                body.append(Assign(self.n_tokens_var, BinOp(
                    "*",
                    BinOp("floordiv", Len(FieldAccess("inputs", "waveform")),
                          Lit(self.samples_per_chunk)),
                    Lit(self.audio_rows_per_chunk),
                )))
                embeds = expr
            else:
                raise ValueError(
                    f"unknown prompt segment kind {kind!r}; this component understands 'text' (token "
                    f"ids to run the embedding topology over) and 'bound' (another module's retained "
                    f"output, used as embeddings directly)."
                )
            body.append(SubgraphCall(
                outputs=[], module=self.topology,
                axes={ctx.root_axis(self.topology): Var(self.n_tokens_var),
                      "n_past": Var(self.n_past_var)},
                inputs=self._call_inputs(embeds),
                retain=True,
            ))
            body.append(Assign(self.n_past_var,
                               BinOp("+", Var(self.n_past_var), Var(self.n_tokens_var))))
        return body


@dataclass
class FlowMatchingSampler(DriverComponent):
    """A `FlowMatchingSpec`'s generated sampler function, plus the one line that calls it.

    The spec and its codegen already existed (`flow_matching_export.py`); what this adds is that both
    ends now go through the builder instead of a marker substitution into hand-written text. That
    closes the gap `FlowMatchingSpec.func_name`'s own `Unchecked` note predicted: the call site is IR,
    so the emitted function name is a real `DriverSymbol` rather than a string nobody could check.
    """

    spec: object  # FlowMatchingSpec
    result: str
    # The `(length, n_elems, n_steps)` arguments, as IR expressions, and the fixed-input table.
    length: object
    n_elems: object
    n_steps: object
    step_inputs: dict
    note: Optional[str] = None

    __links__ = {
        "spec": NestedSpec(
            where="DriverBuilder.build, via sub_specs() -- the spec's own TopologyName/"
                  "TopologyOutputArity/TopologyInput links run against the export's real topologies"
        ),
    }
    __unchecked__ = {
        "note": _NOTE_IS_COSMETIC,
        "result": Unchecked("the local this component binds; reads of it are checked by "
                            "driver_ir.validate over the assembled function"),
        "length": Unchecked("the n_tokens the estimator's graph is built at, as an IR expression over "
                            "locals earlier components bind; validate() is its authority"),
        "n_elems": Unchecked("same -- the element count of the loop-carried state"),
        "n_steps": Unchecked("same -- the caller's step count, read off the inputs table"),
        "step_inputs": Unchecked(
            "the fixed inputs handed to every step, by name. The NAMES are checked, as part of the "
            "spec's own TopologyInput link over its supplied_inputs -- what is unchecked is the "
            "expression each maps to, which is a driver local like any other"
        ),
    }

    def link_label(self) -> str:
        return f"FlowMatchingSampler({self.spec.func_name!r})"

    def sub_specs(self):
        return [self.spec]

    def prelude(self, ctx):
        from .flow_matching_export import render_sampler

        return render_sampler(self.spec).split("\n") + [""]

    def emit(self, ctx):
        return _note_block(self.note) + [Local(self.result, Call(self.spec.func_name, [
            self.length, self.n_elems, self.n_steps, TableLit(dict(self.step_inputs)),
        ]))]


@dataclass
class ExportConstants(DriverComponent):
    """Values a driver needs that only the CHECKPOINT knows -- bound as ordinary locals at the top of
    the entry function.

    A driver reads no GGUF metadata: `loom` gives it topologies and host math, not hparams. So a family
    whose orchestration depends on numbers read off the model at export time (Parakeet's blank id, its
    TDT duration set, the prediction stack's width and depth) has to have them *written into* the driver.

    **The alternative was interpolating them into hand-written Lua through a marker, and this exists to
    avoid exactly that** (BACKLOG.md P4.0.18). A marker substitution is a `str.replace`: the injected
    text is opaque, so a body that misspells one of these reads `nil`, and in Lua `id ~= nil` is quietly
    true -- a TDT decoder would emit every blank as a token and the first sign would be a garbage
    transcript. Emitted as IR, each name is a real `Local`, so `driver_ir.validate` fails the export
    naming the symbol instead.

    Numbers and lists of numbers only: this binds facts, not behaviour. Anything with a branch in it is
    a component or a `loom_lua` function, not a constant.
    """

    values: dict = dataclass_field(default_factory=dict)

    __unchecked__ = {
        "values": Unchecked(
            "read off the restored checkpoint by the family's own config (Parakeet's blank id is its "
            "embedding's row count minus one, its durations are `model_defaults.tdt_durations`), so "
            "there is no second authority here to compare them against -- the config is where a "
            "cross-check belongs, and ASRParakeetExportConfig.phases does run one. What IS checked here "
            "is the other half: every read of these names goes through driver_ir.validate, which is the "
            "whole reason they are IR rather than substituted text."
        ),
    }

    def emit(self, ctx: DriverContext) -> List:
        out = []
        for name, value in self.values.items():
            if isinstance(value, (list, tuple)):
                out.append(Local(name, ArrayLit([Lit(v) for v in value])))
            else:
                out.append(Local(name, Lit(value)))
        return out


@dataclass
class DriverReturn(DriverComponent):
    """What the entry function hands back to the host."""

    values: Tuple[str, ...]

    __unchecked__ = {
        "values": Unchecked("locals earlier components bind. driver_ir.validate is the authority that "
                            "they exist; what the host does with them is the family's contract, and "
                            "the family's e2e test is what checks it"),
    }

    def emit(self, ctx):
        return [Return([Var(name) for name in self.values])]


@dataclass
class MultiPhaseDriverBuilder(DriverBuilder):
    """Every multi-phase (TTS) family's driver.

    Two shapes, and the second replaces the first one family at a time: `driver` holds a whole
    hand-written `.lua` adopted unchanged (C.3), or `peeled` holds the component list a family has
    been broken into (C.4-C.8). The `entry_name` differs from every other builder's (`synthesize`, not
    `main`) because these drivers are called by the TTS host, which is a fact about the family rather
    than about the decomposition.
    """

    driver: Optional[RawLuaDriver] = None
    peeled: Optional[list] = None

    __links__ = {
        "driver": NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN),
        "peeled": NestedSpec(where=_BUILDER_FIELDS_CHECKED_IN),
    }

    entry_name = "infer"

    def __post_init__(self):
        if (self.driver is None) == (self.peeled is None):
            raise ValueError(
                "MultiPhaseDriverBuilder takes exactly one of `driver` (a whole hand-written .lua, "
                "adopted unchanged) or `peeled` (the component list a family has been broken into)"
            )

    def components(self):
        return [self.driver] if self.driver is not None else list(self.peeled)

    def build(self, ctx, checker=None):
        if self.driver is not None:
            self.driver.assert_round_trip()
            print(self.driver.coverage())
            return super().build(ctx, checker)
        script = super().build(ctx, checker)
        print(self.coverage(ctx))
        # A peeled driver ends with a newline, the way every hand-written one in this tree does and
        # every text file should. Not on `DriverScript.render` itself: the two synthesized paths do
        # not end with one, and changing that would move six models' driver text for no reason.
        script.postlude = list(script.postlude) + [""]
        return script

    def called_topologies(self) -> set:
        """Every topology this peeled driver runs, from all three kinds of call site it now has: IR,
        a literal parsed out of a fragment, and a computed name declared as data (D.2)."""
        names = set()
        for component in self.components():
            # Any field a component declares as a `TopologyName` names a topology it runs -- read off
            # the declaration rather than by listing the component classes that have one, so a new
            # component is counted the day it declares its link instead of the day someone remembers to
            # add it here. `PrefillDecodeLoop` is what proved the difference: it declares `topology`
            # exactly like `SubgraphCallComponent` does, and an isinstance list reported Whisper's
            # decoder as a topology no call site names.
            for field, links in declared_links(component).items():
                if any(_names_a_topology(link) for link in links):
                    value = getattr(component, field, None)
                    if isinstance(value, str):
                        names.add(value)
            if isinstance(component, FlowMatchingSampler):
                # The estimator is called from inside the generated sampler function, not from the
                # entry function -- a fourth kind of call site, and one that has always been checked
                # (the spec's own TopologyName link), which is why it belongs in the count.
                names.add(component.spec.estimator)
            for spec in getattr(component, "sub_specs", list)():
                names.update(call.topology for call in getattr(spec, "expand", list)())
                if isinstance(spec, RunSubgraphCall):
                    names.add(spec.topology)
        return names

    def coverage(self, ctx) -> str:
        """One line for the export log: how much of this driver is checked, and what it leaves behind.

        The second half is the part worth printing rather than merely computing. Before D.2 the honest
        answer for Kokoro and StyleTTS2 was "most of it is not", and a number nobody prints is a number
        nobody notices moving -- which is how the eleven duplicated `loom_lua` functions survived a
        whole peeling stage with their own comments admitting it.
        """
        ir = sum(1 for c in self.components() if isinstance(c, SubgraphCallComponent))
        literal = sum(len(getattr(c, "_calls", ())) for c in self.components())
        declared = [d for c in self.components() for d in getattr(c, "drives", ())]
        computed = sum(len(d.expand()) for d in declared)
        called = self.called_topologies()
        uncalled = sorted(set(ctx.topologies) - called)
        note = (f"; {len(uncalled)} exported topolog(ies) no call site names: {uncalled}"
                if uncalled else "; every exported topology is named by one")
        return (f"  driver: {ir} call(s) as IR, {literal} parsed literal, {len(declared)} computed "
                f"site(s) declared -> {computed} topolog(ies){note}")
