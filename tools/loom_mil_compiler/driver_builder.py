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
from .spec_protocol import LinkChecker, declared_links


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

    def render(self) -> str:
        return "\n".join(
            list(self.prelude) + LuaCodegen().emit_function(self.entry) + list(self.postlude)
        )


class DriverBuilder:
    """Assembles one family's (or one orchestration shape's) components into a checked driver.

    `entry_name`/`entry_params` are the function the host calls -- `main(inputs)` for the synthesized
    causal-LM and ASR paths, `synthesize(inputs)` for every TTS family.
    """

    entry_name = "main"
    entry_params = ("inputs",)

    def components(self) -> List[DriverComponent]:
        raise NotImplementedError

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
        for component in components:
            checker.check(component)
            for sub in component.sub_specs():
                checker.check(sub)
        checker.provide(**ctx.link_slots())

        prelude: List[str] = []
        body: List[Stmt] = []
        postlude: List[str] = []
        for component in components:
            prelude.extend(component.prelude(ctx))
            body.extend(component.emit(ctx))
            postlude.extend(component.postlude(ctx))

        entry = IRFunction(self.entry_name, list(self.entry_params), body)
        validate(entry)
        check_subgraph_calls(entry, ctx.topologies)

        # The whole script, not just the entry function: a driver is a Lua module, and a generated
        # sampler is a top-level `local function` in the prelude, so a `DriverSymbol` link naming one
        # is only answerable against the script.
        script = DriverScript(prelude=prelude, entry=entry, postlude=postlude)
        checker.provide(driver=script)
        if owned:
            checker.finish()
        return script

    def render(self, ctx: DriverContext, checker: Optional[LinkChecker] = None) -> str:
        """The embedded `model.driver_script` text."""
        return self.build(ctx, checker).render()
