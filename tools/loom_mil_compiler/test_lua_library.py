"""Checks `lua_library.py` -- the driver-side standard library.

The library exists because stage C's peel left 11 functions totalling 112 lines shipped byte-identical
in two families, with their own comments saying so. What is worth testing is not that the files exist
but the two properties that make a shared library safer than the copies it replaces:

* **its dependency declarations cannot drift from the Lua**, in either direction -- a call the manifest
  does not declare emits a driver referencing an undefined function, and a declaration nothing calls
  bloats every driver that depends on it;
* **a driver carries only what it uses**, so a reader of an embedded `driver_script` can assume every
  function in it is reachable, and Matcha's GGUF does not ship StyleTTS2's ADPM2 sampler.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.driver_builder import DriverContext
from loom_mil_compiler.lua_library import (
    LIBRARY, LUA_DIR, DrivenTopologies, LuaFunction, LuaLibrary, catalogue, drives_mismatches,
    resolve, undeclared_calls, unused_requires,
)
from loom_mil_compiler.spec_protocol import LinkError


class TestTheManifestMatchesTheLua(unittest.TestCase):
    def test_every_call_is_declared(self):
        """The direction that breaks a driver: a function calls a sibling the manifest does not list,
        so `resolve` never pulls the sibling in and the emitted Lua references nothing."""
        self.assertEqual(undeclared_calls(), {})

    def test_every_declaration_is_called(self):
        """The direction that bloats one: a requirement nothing calls travels into every driver that
        depends on it."""
        self.assertEqual(unused_requires(), {})

    def test_every_entry_has_exactly_one_function_in_its_own_file(self):
        for name, fn in LIBRARY.items():
            self.assertTrue(fn.path.is_file(), name)
            defs = re.findall(r"^local function (\w+)", fn.source(), flags=re.MULTILINE)
            self.assertEqual(defs, [name], f"{name}.lua should define exactly `{name}`")

    def test_no_lua_file_is_missing_from_the_manifest(self):
        """The scan direction: a file added to `lua/` and forgotten in `_FUNCTIONS` is invisible to
        every check above, which is the one way they pass vacuously."""
        on_disk = {p.stem for p in LUA_DIR.glob("*.lua")}
        self.assertEqual(on_disk - set(LIBRARY), set())


class TestResolution(unittest.TestCase):
    def test_dependencies_come_before_their_callers(self):
        """Lua binds a `local function` by lexical position, so emitting a caller first would produce a
        driver that loads and then fails on the first call."""
        names = [fn.name for fn in resolve(["predict_durations"])]
        self.assertEqual(names, ["sigmoid", "round_half_to_even", "predict_durations"])

    def test_order_does_not_depend_on_the_order_declared(self):
        forwards = [fn.name for fn in resolve(["run_proj1x1", "run_resblk_stack"])]
        backwards = [fn.name for fn in resolve(["run_resblk_stack", "run_proj1x1"])]
        self.assertEqual(forwards, backwards)

    def test_an_unknown_name_raises_naming_the_library(self):
        with self.assertRaises(KeyError) as raised:
            resolve(["array_summ"])
        self.assertIn("no such loom_lua function(s): ['array_summ']", str(raised.exception))

    def test_a_bad_declaration_fails_the_link_check(self):
        from loom_mil_compiler.spec_protocol import check_links

        with self.assertRaises(LinkError) as raised:
            check_links(LuaLibrary(uses=("array_summ",)))
        self.assertEqual(
            str(raised.exception),
            "LuaLibrary declares loom_lua function(s) ['array_summ'], which do not exist. The library "
            "is tools/loom_mil_compiler/lua/, one file per function.",
        )


class TestADriverCarriesOnlyWhatItUses(unittest.TestCase):
    def test_the_prelude_is_the_transitive_closure_and_nothing_else(self):
        text = "\n".join(LuaLibrary(uses=("run_proj1x1",)).prelude(DriverContext(topologies={})))
        self.assertIn("local function to_layout_a", text)
        self.assertIn("local function run_proj1x1", text)
        self.assertNotIn("adpm2", text)
        self.assertNotIn("run_bi_lstm", text)

    def test_unreferenced_reports_a_declaration_nothing_calls(self):
        library = LuaLibrary(uses=("array_sum", "compute_wsum"))
        driver = "local x = array_sum({1, 2})"
        self.assertEqual(library.unreferenced(driver), ["compute_wsum"])

    def test_a_function_called_only_by_another_library_function_is_not_dead(self):
        """`sigmoid` has no fragment caller in either family -- `predict_durations` calls it. A dead-code
        check that looked only at fragments would report it and be wrong."""
        library = LuaLibrary(uses=("predict_durations",))
        driver = "\n".join(library.prelude(DriverContext(topologies={}))) + "\nlocal d = predict_durations(x, 1)"
        self.assertEqual(library.unreferenced(driver), [])


class TestTheRealFamilies(unittest.TestCase):
    """Driven from the shipped configs, so a family declaring a function it stopped calling -- or
    calling one it stopped declaring -- fails here rather than at export time."""

    def _configs(self):
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig
        from loom_mil_compiler.styletts2_export import TTSStyleTTS2ExportConfig
        from loom_mil_compiler.vits_export import TTSVitsExportConfig

        return (
            TTSMatchaExportConfig(model_dir="/u", output_path="/u", architecture="matcha"),
            TTSVitsExportConfig(checkpoint_path="/u", output_path="/u", architecture="vits"),
            TTSKokoroExportConfig(model_dir="/u", output_path="/u", architecture="kokoro"),
            TTSStyleTTS2ExportConfig(checkpoint_path="/u", output_path="/u", architecture="styletts2"),
        )

    def test_no_family_declares_a_function_its_driver_never_calls(self):
        from loom_mil_compiler.driver_components import MultiPhaseDriverBuilder

        for config in self._configs():
            components = config.driver_components()
            library = next(c for c in components if isinstance(c, LuaLibrary))
            # The whole driver text, since a function may be called from a fragment, from a component's
            # emitted statement, or from another library function.
            text = "\n".join(
                line
                for component in components
                for line in (component.prelude(DriverContext(topologies={}))
                             + getattr(component, "lines", []))
            )
            self.assertEqual(library.unreferenced(text), [], type(config).__name__)

    def test_the_eleven_duplicated_functions_now_have_exactly_one_definition(self):
        """The measurement this library was built from: 11 functions were byte-identical in Kokoro's and
        StyleTTS2's headers. Their definitions must now exist once, in `lua/`, and nowhere else."""
        shared = ("run_bi_lstm", "run_resblk_stack", "run_proj1x1", "to_row_major", "from_row_major",
                  "to_layout_a", "from_layout_a", "sigmoid", "round_half_to_even",
                  "predict_durations", "compute_wsum")
        fragments = sorted(Path(__file__).resolve().parents[1].glob("convert_*/*_driver/*.lua"))
        self.assertTrue(fragments)
        text = "\n".join(p.read_text() for p in fragments)
        for name in shared:
            self.assertNotIn(f"local function {name}", text,
                             f"{name} is defined in a family fragment as well as in lua/")
            self.assertIn(name, LIBRARY)

    def test_every_topology_driving_function_declares_what_it_drives(self):
        """D.2. The three functions that call `loom.run_subgraph` with a computed name are the only
        ones whose call sites nothing could reach; `drives` is what a family expands against."""
        driving = {name for name, fn in LIBRARY.items() if fn.drives is not None}
        self.assertEqual(driving, {"run_bi_lstm", "run_resblk_stack", "run_proj1x1"})

    def test_no_declaration_disagrees_with_the_body_it_describes(self):
        """The direction a family cannot check: the declaration drifting from the Lua, after which
        every family checks its namespaces faithfully against the wrong shape."""
        self.assertEqual(drives_mismatches(), {})

    def test_a_renamed_suffix_is_caught_in_both_directions(self):
        import loom_mil_compiler.lua_library as lua_library

        original = lua_library._FUNCTIONS
        broken = LuaFunction("run_bi_lstm", drives=DrivenTopologies(
            suffixes=("_h_forward", "_c_fwd", "_h_bwd", "_c_bwd"),
            inputs=("layer_input", "h_prev", "c_prev")))
        try:
            lua_library._FUNCTIONS = (broken,)
            complaints = drives_mismatches()["run_bi_lstm"]
        finally:
            lua_library._FUNCTIONS = original
        self.assertIn("declares suffix '_h_forward'", complaints[0])
        self.assertIn("body concatenates '_h_fwd'", complaints[1])

    def test_a_function_that_drives_topologies_without_saying_so_is_reported(self):
        import loom_mil_compiler.lua_library as lua_library

        original = lua_library._FUNCTIONS
        try:
            lua_library._FUNCTIONS = (LuaFunction("run_proj1x1", requires=("to_layout_a",)),)
            complaints = drives_mismatches()["run_proj1x1"]
        finally:
            lua_library._FUNCTIONS = original
        self.assertIn("declares no `drives`", complaints[0])

    def test_the_catalogue_is_generated_rather_than_transcribed(self):
        table = catalogue()
        self.assertIn("| `predict_durations` | `sigmoid`, `round_half_to_even` |", table)
        self.assertIn("kokoro, styletts2", table)


if __name__ == "__main__":
    unittest.main()
