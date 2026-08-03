"""`DriverBuilder` + `DriverComponent` -- how exported topologies become the embedded Lua driver
(BACKLOG.md P4.0.6, `EXPORT-PREPARATION.md` §3/§6 stage C).

The graph side of this exporter already has the shape this module gives the driver side:

    Decomposition : how the model becomes topologies
    DriverBuilder : how those topologies become a driver

`decomposition.py` answers the first question for every family. The second had no owner at all: two
orchestration shapes are synthesized straight into `driver_ir` nodes inside `LoomGGUFExporter`
(`apply_monolithic_export`, `apply_modular_export`), while the five TTS families ship a hand-written
`.lua` that `flow_matching_export.render_driver` performs a string substitution into. Same artifact,
four assembly mechanisms, no common calling convention -- which is what makes adding a family feel
bespoke even though the pieces to build one already exist (`EXPORT-PREPARATION.md` §1.5).

**What a component is.** One contribution to a driver: some statements in the entry function, some
top-level Lua before it (a generated sampler function is exactly this), or both. It declares its links
the way every other spec in this tree does -- a class-level `__links__` dict read by `spec_protocol` --
and `DriverBuilder.build` checks them *before* calling `emit`, so a component that names a topology the
export never produced fails with that link's own message rather than emitting Lua that fails later
inside the engine.

**Why the builder, and not just a list of components.** `build()` is where the three checks that
already exist finally all run over the same artifact, in the one order that makes each of them
meaningful:

    check links -> emit -> driver_ir.validate() -> check_subgraph_calls() -> DriverSymbol links

The last step is why `spec_protocol.DriverSymbol` was written in stage B with no call site: a component
that *contributes* statements to a driver can declare the symbols it expects the surrounding driver to
have already bound, and that is only answerable once every other component has emitted. It reaches the
checker through the ordinary deferral ledger -- `provide(driver=...)` -- so a component whose driver
link never becomes checkable is reported by `finish()` rather than passing silently.

**Selected by the decomposition, not owned by the family** (`EXPORT-PREPARATION.md` §5 decision 2).
`Decomposition.driver_builder(config)` is the hook: a fourth decomposition (the cross-attention AR
decode shape, families 2 + 6) brings its own builder without reopening the component API, and the
component API is therefore shaped by the six components that exist today rather than by a speculative
guess at loop-carried decode state.

**Migration is incremental by construction.** `driver_ir.RawBlock` is an escape hatch for Lua this IR
does not model, so a family moves onto the builder by wrapping its current hand-written driver in one
raw block -- and then peels blocks out into real components one at a time, each peel independently
revertable and independently gated.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

from .driver_ir import Function as IRFunction
from .driver_ir import LuaCodegen, Stmt, check_subgraph_calls, validate
from .spec_protocol import LinkChecker, NestedSpec, Unchecked, declared_links


@dataclasses.dataclass
class DriverContext:
    """The real things a component emits against: the exported topologies, each one's declared root
    axis, and the merged weights.

    Exactly what `MultiPhase.export` already assembles before it calls `render_driver`, and what
    `LoomGGUFExporter` already holds by the time it synthesizes a driver -- named once here so a
    component can be handed it rather than being handed the exporter (which would make every component
    depend on the 2000-line class it happens to be called from today).

    `axes` maps a topology name to the root axis it was exported with -- "n_tokens" unless the phase
    declared otherwise (Conformer-CTC/Parakeet's "n_samples", Kokoro's "n_enc_frames"). A component
    binding a `run_subgraph` call's axis table reads it from here instead of assuming the default, which
    is the R1 mistake in its most tempting form.
    """

    topologies: Dict[str, dict]
    axes: Dict[str, str] = dataclasses.field(default_factory=dict)
    weights: Optional[dict] = None

    def root_axis(self, topology: str) -> str:
        return self.axes.get(topology, "n_tokens")

    def link_slots(self) -> dict:
        """The `spec_protocol` context slots this context can populate, omitting the ones it has
        nothing for -- a slot provided as an empty dict reads as present (`LinkCheckContext.has`), so
        an export with no merged weights would run `WeightName` links against `{}` and fail them
        instead of deferring."""
        slots = {"topologies": self.topologies}
        if self.weights is not None:
            slots["weights"] = self.weights
        return slots


class DriverComponent:
    """One contribution to a driver.

    Subclasses override whichever half they have: `prelude` for top-level Lua emitted before the entry
    function (a generated sampler is this), `emit` for statements inside it. Both are pure functions of
    the context, so the builder can check links between them without a component having cached state
    from an earlier call.

    Links are declared the way every spec in this tree declares them -- a class-level `__links__` dict
    (`spec_protocol`) -- rather than through a bespoke `links()` return value. That is what makes a
    component subject to the standing rule (`every field is either checkable or documented unchecked`)
    with no extra machinery, and what lets P4.0.7's catalogue be generated from the declarations rather
    than transcribed from them.
    """

    def links(self) -> dict:
        """`{field or check name: [Link]}` this component declares. Introspection for the P4.0.7
        catalogue; the builder itself hands the component to the `LinkChecker`, which reads the same
        declarations."""
        return declared_links(self)

    def sub_specs(self) -> List:
        """Specs this component owns, registered with the same checker it is.

        `spec_protocol.NestedSpec` deliberately does not auto-recurse, because a nested spec's links
        are frequently only checkable at a site the outer checker cannot reach. A driver component is
        the case where they *are*: `ModularChain` holds one `ChainStage` per submodule, each declaring
        its own topology and input set, and they are all checkable against the same context the chain
        is. Registering them individually rather than collapsing them into one link is what lets a
        failure name the stage -- `ChainStage('layer_14')` rather than the chain it was in.
        """
        return []

    def prelude(self, ctx: DriverContext) -> List[str]:
        """Top-level Lua lines emitted before the entry function, verbatim. A component owns the blank
        lines around its own contribution -- see `DriverScript`."""
        return []

    def emit(self, ctx: DriverContext) -> List[Stmt]:
        """Statements appended to the entry function's body, in component order."""
        return []

    def postlude(self, ctx: DriverContext) -> List[str]:
        """Top-level Lua lines emitted after the entry function."""
        return []


@dataclasses.dataclass
class DriverEntry:
    """One entry function beside the builder's own `entry_name`.

    A driver is a Lua module and a host resolves entries as globals, so exposing a second way in is
    adding a function, not changing one. `infer` keeps its exact behaviour -- one prefill, argmax the
    last row, one token -- and `infer_with_past` (KV-CACHE.md 3.3) is the generation loop beside it,
    because a model that can do both should not make the caller pick at export time.

    Its components go through the same `LinkChecker`, the same `validate()` and the same
    `check_subgraph_calls()` as the main entry's. The entry it belongs to is a property of where the
    statements land, not of how much they are trusted.
    """

    name: str
    params: tuple
    components: List["DriverComponent"]

    __links__ = {
        "components": NestedSpec(
            where="DriverBuilder.build, which registers every entry's components with the export's "
                  "checker alongside the main entry's -- one ledger, one finish(), no second pipeline"
        ),
    }
    __unchecked__ = {
        "name": Unchecked(
            "the global a host resolves this entry as. There is no second authority to check it "
            "against: the driver script IS the declaration, and `entry_name` (the one every host may "
            "assume) is the builder's, not this record's."
        ),
        "params": Unchecked(
            "the entry's Lua parameter list. Whether the body reads a name it was not given is "
            "driver_ir.validate's question, and it runs over the assembled function."
        ),
    }


@dataclasses.dataclass
class DriverScript:
    """A built driver: top-level lines, the entry function, then any top-level lines after it.

    Three parts rather than one `Function`, because a real driver is a Lua *module*, not a function --
    every TTS driver in this tree is a preamble comment plus zero or more generated sampler functions
    plus the entry point the host calls. Modelling that as "one IRFunction" would have forced the
    sampler functions to become nested closures, which is a semantic change (the host resolves
    `synthesize` as a global) dressed up as a refactor.

    **Lines, not chunks, and joined with a single newline.** A component therefore owns the blank lines
    around its own contribution rather than the script imposing a separator between contributions. That
    is what makes adopting an existing hand-written driver byte-exact: its spacing is data, not
    something to be re-derived. `postlude` exists for the same reason -- a file ending in a newline
    after its last `end` is one trailing empty line, and reconstructing it is not optional when the
    gate is a byte comparison.
    """

    prelude: List[str]
    entry: IRFunction
    postlude: List[str] = dataclasses.field(default_factory=list)
    # Additional top-level entry functions, emitted after `entry` (KV-CACHE.md 3.3).
    extra_entries: List[IRFunction] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        lines = list(self.prelude) + LuaCodegen().emit_function(self.entry)
        for fn in self.extra_entries:
            # The ONE separator this script imposes rather than leaving to a component, and the reason
            # is that an entry function is the module's own structure rather than any component's
            # contribution: two top-level `function`/`end` pairs with no blank line between them are
            # valid Lua and unreadable, and the embedded driver_script is something people read out of
            # the GGUF. Nothing that existed before this had extra entries, so no gate moves.
            lines.append("")
            lines.extend(LuaCodegen().emit_function(fn))
        return "\n".join(lines + list(self.postlude))

    def functions(self) -> List[IRFunction]:
        """Every entry, main first -- what a check that must run over the whole driver iterates."""
        return [self.entry] + list(self.extra_entries)


class DriverBuilder:
    """Assembles one family's (or one orchestration shape's) components into a checked driver.

    `entry_name`/`entry_params` are the function the host calls: `infer(inputs)`, for every family and
    every path. It was `main(inputs)` for the synthesized causal-LM/ASR paths and `synthesize(inputs)`
    for the TTS families until KV-CACHE.md's N.1 -- three names for one concept, each describing the
    model rather than the call, which is why the whole tree now uses the one generic name. A driver may
    expose additional entries beside it (`infer_with_past`, KV-CACHE.md stage 3); `entry_name` is the
    one every host can assume.
    """

    entry_name = "infer"
    entry_params = ("inputs",)

    def components(self) -> List[DriverComponent]:
        raise NotImplementedError

    def extra_entries(self) -> List[DriverEntry]:
        """Entry functions beside `entry_name`, each with its own component list.

        Empty for every builder but the fused causal LM's, which exposes `infer_with_past` next to
        `infer` (KV-CACHE.md 3.3). A builder that returns entries here does not get a second, weaker
        pipeline: every component below is registered with the same checker and every function is
        validated and cross-checked against the topologies exactly as the main entry is.
        """
        return []

    def build(self, ctx: DriverContext, checker: Optional[LinkChecker] = None) -> DriverScript:
        """Check every component's links, emit, then run the driver IR's own two checks over the result.

        `checker` lets the caller pass the export-wide `LinkChecker` -- `MultiPhase.export` does, so a
        component declared here and a spec declared anywhere else in the same export share one deferral
        ledger and one `finish()`. Without one, a local checker is built and finished here.

        The ordering is the point, not an implementation detail:

        1. **links before emit.** A component that names a topology the export did not produce should
           say so with `TopologyName`'s message, not emit a `run_subgraph` call that fails later.
        2. **`validate()` then `check_subgraph_calls()`** over the assembled function, which is where a
           symbol read across a component boundary (component B reading a local component A stopped
           emitting) is caught -- the class of bug that only exists once a driver is composed.
        3. **`provide(driver=...)` last**, so `DriverSymbol` links resolve against the finished
           function. A component cannot check what the *whole* driver defines until the whole driver
           exists, and deferral is exactly the mechanism for that.
        """
        owned = checker is None
        checker = checker if checker is not None else LinkChecker()

        components = list(self.components())
        extra = list(self.extra_entries())
        for component in components + [c for entry in extra for c in entry.components]:
            checker.check(component)
            for sub in component.sub_specs():
                checker.check(sub)
        checker.provide(**ctx.link_slots())

        # Imported here rather than at module scope: the registry names every component class, and each
        # of those modules imports this one.
        from .component_registry import PRELUDE, POSTLUDE, STATEMENTS, check_emission

        prelude: List[str] = []
        postlude: List[str] = []

        def emit_body(group: List[DriverComponent]) -> List[Stmt]:
            body: List[Stmt] = []
            for component in group:
                contributions = {
                    PRELUDE: component.prelude(ctx),
                    STATEMENTS: component.emit(ctx),
                    POSTLUDE: component.postlude(ctx),
                }
                # P4.0.7/D.1: the component is looked up in the registry as it emits, and what it
                # really contributed is compared against what its entry claims. A shipped component
                # with no entry raises here -- a piece of a driver that the catalogue does not account
                # for is exactly the "inventory, not a shelf" failure the registry exists to prevent.
                check_emission(component, {slot for slot, value in contributions.items() if value})
                prelude.extend(contributions[PRELUDE])
                body.extend(contributions[STATEMENTS])
                postlude.extend(contributions[POSTLUDE])
            return body

        entry = IRFunction(self.entry_name, list(self.entry_params), emit_body(components))
        extra_functions = [
            IRFunction(spec.name, list(spec.params), emit_body(spec.components)) for spec in extra
        ]
        for function in [entry] + extra_functions:
            validate(function)
            check_subgraph_calls(function, ctx.topologies)

        # The whole script, not just the entry function: a driver is a Lua module, and a generated
        # sampler is a top-level `local function` in the prelude, so a `DriverSymbol` link naming one
        # is only answerable against the script.
        script = DriverScript(prelude=prelude, entry=entry, postlude=postlude,
                              extra_entries=extra_functions)
        checker.provide(driver=script)
        if owned:
            checker.finish()
        return script

    def render(self, ctx: DriverContext, checker: Optional[LinkChecker] = None) -> str:
        """The embedded `model.driver_script` text."""
        return self.build(ctx, checker).render()
