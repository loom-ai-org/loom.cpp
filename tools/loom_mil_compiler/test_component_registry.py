"""Checks the driver-component registry (BACKLOG.md P4.0.7, `EXPORT-PREPARATION.md` stage D.1).

The registry's whole claim is that it is a shelf rather than a directory, and a directory passes every
test that only asks "is the table well-formed". So what is tested here is each direction the manifest
can rot in, and each one is a property that would otherwise be discovered by reading the catalogue and
believing it:

* a shipped component with **no entry** fails an export rather than shipping unaccounted for;
* an entry whose **`emits` is too narrow** fails an export, because the catalogue's emission column is
  generated from it;
* an entry **nobody uses** must say why it is still here, the same rule `loom_lua` applies to a library
  function with no caller;
* the **usage** column is derived from the real component lists and the real builder fields, so it
  cannot claim a model that does not use the component.

The last one is the reason `usage()` reports what it could not read rather than returning a shorter
dict: a catalogue that silently omits a model understates exactly what it exists to account for.
"""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler import component_registry as cr
from loom_mil_compiler.driver_builder import DriverBuilder, DriverComponent, DriverContext
from loom_mil_compiler.driver_components import (
    SYNTHESIZED_BUILDERS, DriverReturn, LuaFragment, ModularChainBuilder, PrefillArgmaxBuilder,
)
from loom_mil_compiler.driver_ir import RawBlock


def _ctx(**kwargs):
    kwargs.setdefault("topologies", {})
    return DriverContext(**kwargs)


@dataclass
class OneOffComponent(DriverComponent):
    """Defined in a `test_` module, so it is deliberately *not* a shipped component -- see
    `component_registry.is_shipped`."""

    def emit(self, ctx):
        return [RawBlock(["-- nothing"])]


class _OneComponent(DriverBuilder):
    def __init__(self, component):
        self._component = component

    def components(self):
        return [self._component]


class TestTheManifest(unittest.TestCase):
    def test_every_shipped_component_class_is_registered(self):
        """The direction that keeps the catalogue complete. Enforced by discovery over the package, so
        a component added to a family module fails here without anyone remembering to extend a list."""
        self.assertEqual(cr.unregistered_component_classes(), [])

    def test_the_scan_reaches_the_classes_it_is_supposed_to_reach(self):
        """Without this the test above passes vacuously the moment the scan stops finding anything."""
        registered = {entry.cls.__name__ for entry in cr.registry().values()}
        for name in ("DriverInputs", "MonolithicCall", "ModularChain", "ArgmaxEpilogue",
                     "RawLuaDriver", "LuaFragment", "SubgraphCallComponent", "FlowMatchingSampler",
                     "DriverReturn", "LuaLibrary"):
            self.assertIn(name, registered)

    def test_a_test_local_component_is_not_a_shipped_one(self):
        self.assertFalse(cr.is_shipped(OneOffComponent))
        self.assertTrue(cr.is_shipped(DriverReturn))

    def test_an_entry_no_model_uses_must_say_why_it_is_still_here(self):
        used, _ = cr.usage()
        for name, models in used.items():
            if not models:
                reason = cr.get(name).no_user_reason
                self.assertTrue(reason and len(reason) > 40,
                                f"{name} has no user and no reason for still being registered")

    def test_a_reason_is_only_allowed_where_there_is_no_user(self):
        """The other half: a component that has acquired a user must lose the excuse, or the catalogue
        keeps printing an explanation that is no longer true."""
        used, _ = cr.usage()
        for name, models in used.items():
            if models:
                self.assertIsNone(cr.get(name).no_user_reason, name)

    def test_unknown_emission_slot_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            cr.ComponentEntry("x", DriverReturn, ("preamble",), "…")
        self.assertIn("preamble", str(raised.exception))

    def test_a_non_component_cannot_be_registered(self):
        with self.assertRaises(TypeError):
            cr.ComponentEntry("x", dict, ("statements",), "…")

    def test_get_names_the_alternatives(self):
        with self.assertRaises(KeyError) as raised:
            cr.get("no_such_component")
        self.assertIn("subgraph_call", str(raised.exception))


class TestTheBuilderConsultsIt(unittest.TestCase):
    def test_an_unregistered_shipped_component_fails_the_build(self):
        """The negative gate for D.1, as a test: a component the registry does not know cannot reach a
        GGUF. Faked by lying about the class's module, since every real one is registered -- which is
        the state the test above enforces."""
        component = OneOffComponent()
        original = type(component).__module__
        try:
            type(component).__module__ = "loom_mil_compiler.some_new_family"
            with self.assertRaises(KeyError) as raised:
                _OneComponent(component).build(_ctx())
        finally:
            type(component).__module__ = original
        message = str(raised.exception)
        self.assertIn("some_new_family.OneOffComponent", message)
        self.assertIn("component registry", message)

    def test_a_component_that_emits_more_than_its_entry_claims_fails_the_build(self):
        """`emits` is what the catalogue's column is generated from, so a claim narrower than the truth
        is a false published account, not a cosmetic slip."""
        entry = cr.get("driver_return")
        narrowed = cr.ComponentEntry(entry.name, entry.cls, (cr.PRELUDE,), entry.summary)
        cr.registry()
        original = cr._BY_CLASS[entry.cls]
        try:
            cr._BY_CLASS[entry.cls] = narrowed
            with self.assertRaises(ValueError) as raised:
                _OneComponent(DriverReturn(values=())).build(_ctx())
        finally:
            cr._BY_CLASS[entry.cls] = original
        message = str(raised.exception)
        self.assertIn("'driver_return'", message)
        self.assertIn("statements", message)
        self.assertIn("['prelude']", message)

    def test_a_component_may_emit_less_than_its_entry_allows(self):
        """`LuaFragment` emits a prelude when `top_level` is set and statements otherwise; neither
        instance is a drift, which is why the check is a subset one."""
        fragment = LuaFragment(
            Path(__file__).resolve().parent.parent / "convert_supertonic" / "supertonic_driver"
            / "00_header.lua", top_level=True)
        script = _OneComponent(fragment).build(_ctx())
        self.assertTrue(script.prelude)
        self.assertEqual(script.entry.body, [])


class TestUsageIsDerived(unittest.TestCase):
    def test_the_synthesized_paths_are_read_off_the_builders_own_fields(self):
        self.assertEqual(cr.builder_components(PrefillArgmaxBuilder),
                         ["driver_inputs", "monolithic_call", "argmax_epilogue"])
        self.assertEqual(cr.builder_components(ModularChainBuilder),
                         ["driver_inputs", "modular_chain", "argmax_epilogue"])

    def test_the_exporter_builds_through_the_same_table_the_attribution_reads(self):
        """What makes the attribution non-circular: `apply_monolithic_export`/`apply_modular_export`
        construct through `SYNTHESIZED_BUILDERS`, so "qwen3 uses argmax_epilogue" is a statement about
        the code that runs rather than about a hand-kept mapping."""
        self.assertIs(SYNTHESIZED_BUILDERS["Flattened"], PrefillArgmaxBuilder)
        self.assertIs(SYNTHESIZED_BUILDERS["Modular"], ModularChainBuilder)
        source = (Path(__file__).resolve().parent / "exporter.py").read_text()
        self.assertIn('SYNTHESIZED_BUILDERS["Flattened"](', source)
        self.assertIn('SYNTHESIZED_BUILDERS["Modular"](', source)

    def test_the_tts_half_comes_from_the_real_component_lists(self):
        used, _ = cr.usage()
        self.assertEqual(used["flow_matching_sampler"], ["matcha", "supertonic"])
        self.assertEqual(used["modular_chain"], ["lfm2-modular"])
        self.assertIn("qwen3", used["monolithic_call"])
        # Supertonic is the one peeled family that declares no loom_lua function, which is a real
        # property of it (its remaining hand-written block is scalar arithmetic) rather than an
        # oversight -- and the kind of thing a hand-maintained usage column would get wrong.
        self.assertNotIn("supertonic", used["lua_library"])

    def test_every_registered_recognizer_is_actually_counted(self):
        """`usage()` reports what it could not read instead of returning a shorter dict, and today it
        reads all of them -- including P4.0.4's generic `hf-causal-lm` fallback, whose config is
        inferred from a real `config.json` and which resolves to no overrides for a path that does not
        exist. An empty report is the claim that the *used by* column is complete."""
        used, unavailable = cr.usage()
        self.assertEqual(unavailable, {})
        self.assertIn("hf-causal-lm", used["monolithic_call"])


if __name__ == "__main__":
    unittest.main()
