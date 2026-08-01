"""Checks `driver_builder.py` (BACKLOG.md P4.0.6, `EXPORT-PREPARATION.md` stage C.1).

C.1 writes no components and touches no export path, so what is testable here is exactly the contract
the later steps are built on, and each of these is a property a plausible-looking implementation could
fail while still producing correct Lua for the two paths that exist today:

* **links are checked before `emit` runs.** A component that names a topology the export did not
  produce must fail with that link's message and never have emitted -- otherwise the ordering §2's
  "a component's declared links are checked before it emits anything" describes is decorative.
* **the driver IR's own two checks run over the *assembled* function**, which is the only place a
  cross-component symbol read can be caught. That bug class does not exist until a driver is composed
  from parts, so nothing before P4.0.6 could have caught it.
* **`DriverSymbol` resolves through deferral, not through a special case.** Stage B wrote that link
  with no call site precisely for this; if `build` checked it eagerly it would fail for every component,
  and if it never provided the slot the link would sit in the ledger and `finish()` would raise.
* **an unfinished shared checker stays the caller's.** `MultiPhase.export` owns one checker for a whole
  export; a builder that finished it would turn every later `provide()` into a no-op.
"""
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.driver_builder import (
    DriverBuilder, DriverComponent, DriverContext, DriverScript,
)
from loom_mil_compiler.driver_ir import (
    DriverIRError, Lit, Local, Return, SubgraphCall, Var,
)
from loom_mil_compiler.spec_protocol import (
    DriverSymbol, LinkChecker, LinkError, TopologyName, Unchecked, WeightName,
)


@dataclass
class NeedsWeights(DriverComponent):
    """Declares a link whose context (`weights`) the builder never populates on its own -- the deferral
    case both `TestCheckerOwnership` tests turn on."""

    names: tuple = ("w",)

    __links__ = {"names": WeightName()}


def _topo(inputs=(), output="out"):
    return {"version": 1, "inputs": [{"name": n} for n in inputs], "nodes": [], "output": output}


def _ctx(**kwargs):
    kwargs.setdefault("topologies", {"encoder": _topo(["tokens"]), "vocoder": _topo(["mel"])})
    return DriverContext(**kwargs)


# -- components used by the tests --------------------------------------------------------------------


@dataclass
class RecordingComponent(DriverComponent):
    """Emits one `local`, and records that it was asked to."""

    name: str
    emitted: list = field(default_factory=list)

    __unchecked__ = {
        "name": Unchecked("the local this test component binds; a fixture, not a declaration"),
        "emitted": Unchecked("the call record this test asserts on"),
    }

    def prelude(self, ctx):
        return [f"-- prelude of {self.name}"]

    def emit(self, ctx):
        self.emitted.append(ctx)
        return [Local(self.name, Lit(1))]


@dataclass
class CallComponent(DriverComponent):
    """Emits one `run_subgraph` call against a named topology, declaring the topology as a link."""

    topology: str
    inputs: tuple = ("tokens",)
    emitted: list = field(default_factory=list)

    __links__ = {"topology": TopologyName()}
    __unchecked__ = {
        "inputs": Unchecked("what this fixture passes; the real check is TopologyInput, exercised in "
                            "test_spec_protocol.py against the link kind itself"),
        "emitted": Unchecked("the call record this test asserts on"),
    }

    def emit(self, ctx):
        self.emitted.append(ctx)
        return [SubgraphCall(
            outputs=["y"], module=self.topology,
            axes={ctx.root_axis(self.topology): Lit(1)},
            inputs={n: Lit(0) for n in self.inputs},
        )]


@dataclass
class SymbolReadingComponent(DriverComponent):
    """Declares that the assembled driver must define some symbol, without emitting anything."""

    expects: str

    __links__ = {"expects": DriverSymbol()}


class _Builder(DriverBuilder):
    def __init__(self, components, entry_name="main"):
        self._components = list(components)
        self.entry_name = entry_name

    def components(self):
        return self._components


# -- the tests ---------------------------------------------------------------------------------------


class TestAssembly(unittest.TestCase):
    def test_prelude_and_body_keep_component_order(self):
        script = _Builder([RecordingComponent("a"), RecordingComponent("b")]).build(_ctx())
        self.assertIsInstance(script, DriverScript)
        self.assertEqual(script.prelude, ["-- prelude of a", "-- prelude of b"])
        self.assertEqual([s.name for s in script.entry.body], ["a", "b"])

    def test_render_puts_the_prelude_above_the_entry_function(self):
        """Joined with a single newline: a component owns the blank lines around its own contribution,
        which is what lets an existing hand-written driver be adopted byte-exactly."""
        text = _Builder([RecordingComponent("a")]).render(_ctx())
        self.assertEqual(text, "-- prelude of a\nfunction main(inputs)\n    local a = 1\nend")

    def test_a_postlude_survives_to_the_end_of_the_script(self):
        """A file ending in a newline after its last `end` is one trailing empty line -- and every
        hand-written driver in this tree does."""
        class _Trailing(DriverComponent):
            def postlude(self, ctx):
                return [""]

        self.assertEqual(_Builder([RecordingComponent("a"), _Trailing()]).render(_ctx()),
                         "-- prelude of a\nfunction main(inputs)\n    local a = 1\nend\n")

    def test_a_component_with_no_prelude_renders_the_function_alone(self):
        """The synthesized causal-LM/ASR shape: no top-level Lua at all, so `render` must not introduce
        a leading blank line -- byte-identity with `"\\n".join(emit_function(...))` is C.2's gate."""
        text = _Builder([CallComponent("encoder")]).render(_ctx())
        self.assertTrue(text.startswith("function main(inputs)\n"))

    def test_entry_name_and_params_are_the_builder_s(self):
        builder = _Builder([RecordingComponent("a")], entry_name="synthesize")
        self.assertEqual(builder.build(_ctx()).entry.name, "synthesize")
        self.assertEqual(builder.build(_ctx()).entry.params, ["inputs"])


class TestLinksRunBeforeEmit(unittest.TestCase):
    def test_a_failing_link_stops_the_component_emitting(self):
        component = CallComponent("decodr")
        with self.assertRaises(LinkError) as raised:
            _Builder([component]).build(_ctx())
        self.assertEqual(
            str(raised.exception),
            "CallComponent names topology 'decodr', which is not among the exported topologies "
            "['encoder', 'vocoder'].",
        )
        self.assertEqual(component.emitted, [], "emit() ran despite the component's link failing")

    def test_a_holding_link_lets_every_component_emit(self):
        components = [CallComponent("encoder"), RecordingComponent("a")]
        _Builder(components).build(_ctx())
        self.assertEqual(len(components[0].emitted), 1)
        self.assertEqual(len(components[1].emitted), 1)


class TestTheIRChecksRunOverTheAssembledFunction(unittest.TestCase):
    def test_a_symbol_read_across_a_component_boundary_is_caught(self):
        """The bug class that does not exist until a driver is composed from parts: component B reads a
        local that component A no longer emits. `driver_ir.validate` has always been able to catch it;
        before the builder there was no assembled function to run it over."""
        class _Reader(DriverComponent):
            def emit(self, ctx):
                return [Return([Var("gone")])]

        with self.assertRaises(DriverIRError) as raised:
            _Builder([RecordingComponent("a"), _Reader()]).build(_ctx())
        self.assertIn("symbol 'gone' is read", str(raised.exception))

    def test_an_undeclared_run_subgraph_input_is_caught(self):
        with self.assertRaises(DriverIRError) as raised:
            _Builder([CallComponent("encoder", inputs=("tokens", "mel"))]).build(_ctx())
        self.assertIn("passes undeclared input(s) ['mel']", str(raised.exception))

    def test_the_root_axis_comes_from_the_context_not_a_default(self):
        """Binding "n_tokens" on a topology exported with "n_samples" is the R1 mistake in its most
        tempting form -- the call still runs, against a symbol the topology never uses."""
        ctx = _ctx(axes={"encoder": "n_samples"})
        script = _Builder([CallComponent("encoder")]).build(ctx)
        self.assertEqual(list(script.entry.body[0].axes), ["n_samples"])
        self.assertEqual(
            list(_Builder([CallComponent("vocoder", inputs=("mel",))]).build(ctx).entry.body[0].axes),
            ["n_tokens"])


class TestDriverSymbolResolvesThroughDeferral(unittest.TestCase):
    def test_a_symbol_the_assembled_driver_defines_passes(self):
        _Builder([RecordingComponent("a"), SymbolReadingComponent("a")]).build(_ctx())

    def test_a_symbol_nothing_defines_fails_with_the_link_s_message(self):
        with self.assertRaises(LinkError) as raised:
            _Builder([RecordingComponent("a"), SymbolReadingComponent("b")]).build(_ctx())
        self.assertIn("reads driver symbol(s) ['b'], which the emitted driver 'main' never defines",
                      str(raised.exception))


class TestCheckerOwnership(unittest.TestCase):
    def test_a_caller_s_checker_is_not_finished_by_build(self):
        """`MultiPhase.export` owns one checker for the whole export and calls `finish()` itself, after
        the driver is built -- a builder that finished it would report every link the rest of the
        export has not yet been able to check."""
        checker = LinkChecker()
        _Builder([NeedsWeights()]).build(_ctx(), checker=checker)
        self.assertEqual(len(checker.deferred), 1)
        checker.provide(weights={"w": None})
        self.assertEqual(checker.deferred, [])

    def test_an_owned_checker_reports_a_link_that_never_became_checkable(self):
        with self.assertRaises(LinkError) as raised:
            _Builder([NeedsWeights()]).build(_ctx())
        self.assertIn("were never checked", str(raised.exception))

    def test_weights_are_only_provided_when_the_context_has_them(self):
        """A slot provided as `{}` reads as present, so an export with no merged weights would *fail*
        WeightName links instead of deferring them."""
        self.assertNotIn("weights", _ctx().link_slots())
        self.assertIn("weights", _ctx(weights={}).link_slots())


class TestTheDecompositionHook(unittest.TestCase):
    def test_the_base_hook_answers_none(self):
        from loom_mil_compiler.decomposition import Decomposition

        self.assertIsNone(Decomposition().driver_builder(config=None))

    def test_a_component_reports_its_own_links_for_the_catalogue(self):
        """P4.0.7's catalogue is generated from the declarations rather than transcribed from them."""
        self.assertEqual(list(CallComponent("encoder").links()), ["topology"])
        self.assertEqual(RecordingComponent("a").links(), {})


if __name__ == "__main__":
    unittest.main()
