"""Checks `driver_components.py` (BACKLOG.md P4.0.6, `EXPORT-PREPARATION.md` stage C.2).

The real gate for C.2 is byte-identical `model.driver_script` across the six models on the two
synthesized paths, and that is a snapshot diff, not a unit test. What these cover is the half a
snapshot cannot: the components in isolation, at the level where a future change would be *reviewed*
rather than re-exported. The rendered Lua is asserted whole -- a `match=` fragment would pass through
exactly the spacing and naming changes the snapshot gate exists to catch.

These run with no coremltools program: the components take IR pieces the exporter has already computed,
which is precisely what makes them testable without a traced model, and is why the exporter keeps
ownership of `safe_name`/the MIL function rather than a component reaching for it.
"""
import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.driver_builder import DriverContext
from loom_mil_compiler.driver_components import (
    CALLER, MASK, POSITION, ArgmaxEpilogue, ChainStage, DriverInputs, ModularChain,
    FlowMatchingSampler, LuaFragment, ModularChainBuilder, MonolithicCall, MultiPhaseDriverBuilder,
    PrefillArgmaxBuilder, RawLuaDriver, SubgraphCallComponent, caller_input, parse_run_subgraph_calls,
)
from loom_mil_compiler.driver_ir import (
    BinOp, DriverIRError, Len, Lit, LuaCodegen, SubgraphCall, Var,
)
from loom_mil_compiler.spec_protocol import LinkError


def _topo(inputs=(), output="out"):
    return {"version": 1, "inputs": [{"name": n} for n in inputs], "nodes": [], "output": output}


def _render(builder, ctx):
    return builder.render(ctx)


class TestCallerInput(unittest.TestCase):
    def test_a_named_input_is_aliased_to_tokens(self):
        self.assertEqual(caller_input("input_ids").render(), "(inputs.input_ids or inputs.tokens)")

    def test_the_generic_name_is_not_aliased_to_itself(self):
        """`inputs.tokens or inputs.tokens` -- the same expression on both sides of the `or` -- is what
        every causal LM's driver read like before this case existed."""
        self.assertEqual(caller_input("tokens").render(), "inputs.tokens")


class TestPrefillArgmaxBuilder(unittest.TestCase):
    def _builder(self, bindings, inputs, n_tokens):
        return PrefillArgmaxBuilder(
            inputs=DriverInputs(bindings=bindings, n_tokens=n_tokens),
            call=MonolithicCall(topology="main_topology", inputs=inputs, n_tokens=n_tokens),
            epilogue=ArgmaxEpilogue(out_var="_mono_out", shape_var="_mono_shape", n_tokens=n_tokens),
        )

    def test_the_plain_causal_lm_shape(self):
        ctx = DriverContext(topologies={"main_topology": _topo(["tokens"])},
                            axes={"main_topology": "n_tokens"})
        text = self._builder((("tokens", CALLER),), ("tokens",), Len("tokens")).render(ctx)
        self.assertEqual(text, "\n".join([
            "function infer(inputs)",
            "    local tokens = inputs.tokens",
            "    local _mono_out, _mono_shape = loom.run_subgraph('main_topology', "
            "{n_tokens = #tokens, n_past = 0}, {tokens = tokens})",
            "    if (type(_mono_out) == 'table') then",
            "        return loom.argmax_row(_mono_out, _mono_shape[1], (#tokens - 1))",
            "    else",
            "        return _mono_out",
            "    end",
            "end",
        ]))

    def test_host_computed_position_and_mask_inputs(self):
        """LFM2's traced graph declares `cache_position` and `attention_mask`; the driver fills both in
        rather than making a caller know they exist."""
        ctx = DriverContext(
            topologies={"main_topology": _topo(["input_ids", "cache_position", "attention_mask"])},
            axes={"main_topology": "n_tokens"})
        bindings = (("input_ids", CALLER), ("cache_position", POSITION), ("attention_mask", MASK))
        text = self._builder(bindings, tuple(n for n, _ in bindings), Len("input_ids")).render(ctx)
        self.assertIn("    local input_ids = (inputs.input_ids or inputs.tokens)\n"
                      "    local cache_position = loom.range(0, #input_ids)\n"
                      "    local attention_mask = loom.causal_mask(#input_ids, 0)\n", text)

    def test_the_root_axis_is_the_topology_s_own(self):
        """Conformer-CTC/Parakeet declare "n_samples" -- raw audio samples, never a token count. The
        VALUE is still the first input's own length; only the axis it binds differs (R1)."""
        ctx = DriverContext(topologies={"main_topology": _topo(["audio_signal"])},
                            axes={"main_topology": "n_samples"})
        n_tokens = BinOp("floordiv", Len("audio_signal"), Lit(80))
        text = self._builder((("audio_signal", CALLER),), ("audio_signal",), n_tokens).render(ctx)
        self.assertIn("{n_samples = math.floor(#audio_signal / 80), n_past = 0}", text)


class TestModularChainBuilder(unittest.TestCase):
    def _ctx(self):
        return DriverContext(
            topologies={
                "prefix": _topo(["input_ids"]),
                "layer_0": _topo(["hidden_states"]),
                "layer_1": _topo(["hidden_states"]),
                "norm": _topo(["hidden_states"]),
            },
            axes={n: "n_tokens" for n in ("prefix", "layer_0", "layer_1", "norm")},
        )

    def _builder(self):
        n_tokens = Len("input_ids")
        stages = [
            ChainStage(topology="prefix", outputs=("_mod_chain_0",),
                       inputs={"input_ids": Var("input_ids")}),
            ChainStage(topology="layer_0", outputs=("_mod_chain_1",),
                       inputs={"hidden_states": Var("_mod_chain_0")}),
            ChainStage(topology="layer_1", outputs=("_mod_chain_2",),
                       inputs={"hidden_states": Var("_mod_chain_1")}),
            ChainStage(topology="norm", outputs=("_mod_suffix_0",),
                       inputs={"hidden_states": Var("_mod_chain_2")},
                       extra_outputs=("_modular_final_shape",)),
        ]
        return ModularChainBuilder(
            inputs=DriverInputs(bindings=(("input_ids", CALLER),), n_tokens=n_tokens),
            chain=ModularChain(stages=tuple(stages), n_tokens=n_tokens),
            epilogue=ArgmaxEpilogue(out_var="_mod_suffix_0", shape_var="_modular_final_shape",
                                    n_tokens=n_tokens),
        )

    def test_the_chain_threads_one_variable_through_every_stage(self):
        text = self._builder().render(self._ctx())
        self.assertEqual(text, "\n".join([
            "function infer(inputs)",
            "    local input_ids = (inputs.input_ids or inputs.tokens)",
            "    local _mod_chain_0 = loom.run_subgraph('prefix', {n_tokens = #input_ids, n_past = 0}, "
            "{input_ids = input_ids})",
            "    local _mod_chain_1 = loom.run_subgraph('layer_0', {n_tokens = #input_ids, n_past = 0}, "
            "{hidden_states = _mod_chain_0})",
            "    local _mod_chain_2 = loom.run_subgraph('layer_1', {n_tokens = #input_ids, n_past = 0}, "
            "{hidden_states = _mod_chain_1})",
            "    local _mod_suffix_0, _modular_final_shape = loom.run_subgraph('norm', "
            "{n_tokens = #input_ids, n_past = 0}, {hidden_states = _mod_chain_2})",
            "    if (type(_mod_suffix_0) == 'table') then",
            "        return loom.argmax_row(_mod_suffix_0, _modular_final_shape[1], (#input_ids - 1))",
            "    else",
            "        return _mod_suffix_0",
            "    end",
            "end",
        ]))

    def test_a_stage_naming_a_topology_that_was_not_exported_fails_naming_the_stage(self):
        """A chain is 20-plus stages, so the message has to say which one. Before the builder this
        surfaced as a bare KeyError on the dict of MIL functions."""
        builder = self._builder()
        builder.chain.stages[1].topology = "layer_zero"
        with self.assertRaises(LinkError) as raised:
            builder.build(self._ctx())
        self.assertEqual(
            str(raised.exception),
            "ChainStage('layer_zero') names topology 'layer_zero', which is not among the exported "
            "topologies ['layer_0', 'layer_1', 'norm', 'prefix'].",
        )

    def test_a_stage_leaving_a_declared_input_unsupplied_is_caught(self):
        """The direction `driver_ir.check_subgraph_calls` has never checked. An input the topology
        declares but the driver never supplies is not an error to the engine -- it is an uninitialised
        tensor, and the model merely produces wrong output."""
        ctx = self._ctx()
        ctx.topologies["layer_0"] = _topo(["hidden_states", "position_embeddings"])
        with self.assertRaises(LinkError) as raised:
            self._builder().build(ctx)
        self.assertEqual(
            str(raised.exception),
            "ChainStage('layer_0') does not match topology 'layer_0': leaves declared input(s) "
            "unsupplied: ['position_embeddings']; topology declares ['hidden_states', "
            "'position_embeddings'], spec supplies ['hidden_states'].",
        )


class TestTheTwoPathsShareComponents(unittest.TestCase):
    def test_the_prologue_and_epilogue_are_the_same_two_classes(self):
        """`EXPORT-PREPARATION.md` §1.5 counts "prefill prologue/epilogue" as ONE inventory item across
        both synthesized paths. This is that claim, as a test rather than an observation -- two of the
        three components in each builder are literally the same class."""
        prefill = set(PrefillArgmaxBuilder.__annotations__)
        modular = set(ModularChainBuilder.__annotations__)
        self.assertEqual(
            {PrefillArgmaxBuilder.__annotations__[n] for n in prefill & modular},
            {DriverInputs, ArgmaxEpilogue},
        )

    def test_the_epilogue_renders_identically_for_both(self):
        epilogue = ArgmaxEpilogue(out_var="x", shape_var="s", n_tokens=Len("t"))
        emitted = epilogue.emit(DriverContext(topologies={}))
        rendered = LuaCodegen()._emit_stmt(emitted[0], 0)
        self.assertEqual(rendered[1], "    return loom.argmax_row(x, s[1], (#t - 1))")


if __name__ == "__main__":
    unittest.main()


# -- C.3: adopting a hand-written driver -------------------------------------------------------------


_DRIVER = """-- a hand-written driver
local function helper(x)
    return x + 1
end

function infer(inputs)
    local a = loom.run_subgraph("encoder", {n_tokens = #inputs.tokens, n_past = 0}, {
        tokens = inputs.tokens,
        style = inputs.style,
    })
    local b = loom.run_subgraph("vocoder", {n_enc_frames = 4, n_past = 0}, { mel = a })
    return b
end
"""


def _called_topologies(config) -> set:
    """Every topology a family's driver names literally, whether it is still one adopted `.lua` or has
    been peeled into fragments and components. Written this way so the declaration check survives each
    peel rather than having to be rewritten by it."""
    path = config.driver_script_path
    if path.is_file():
        return RawLuaDriver(source=path.read_text(), origin=path.name).called_topologies()
    names = set()
    for component in config.driver_components():
        if isinstance(component, SubgraphCallComponent):
            names.add(component.topology)
        elif isinstance(component, LuaFragment):
            names.update(call.topology for call in component._calls)
    return names


def _adoption_ctx():
    return DriverContext(
        topologies={"encoder": _topo(["tokens", "style"]), "vocoder": _topo(["mel"])},
        axes={"encoder": "n_tokens", "vocoder": "n_enc_frames"},
    )


class TestRawLuaDriverIsByteExact(unittest.TestCase):
    def test_the_adoption_reproduces_its_own_source(self):
        """The split-and-rejoin is the one thing this component does that can silently corrupt a
        working driver, and a whole-file indentation shift is exactly what a reviewer reads past."""
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(source=_DRIVER, origin="d.lua"))
        self.assertEqual(builder.render(_adoption_ctx()), _DRIVER)

    def test_the_body_keeps_its_own_indentation(self):
        """`RawBlock` normally takes the enclosing block's indentation; an adopted body already has
        its own, and re-indenting it would move every line of the embedded driver_script."""
        script = MultiPhaseDriverBuilder(
            driver=RawLuaDriver(source=_DRIVER, origin="d.lua")).build(_adoption_ctx())
        self.assertTrue(script.entry.body[0].verbatim)
        self.assertIn("    local b = loom.run_subgraph(\"vocoder\", {n_enc_frames = 4, n_past = 0}, "
                      "{ mel = a })", script.render())

    def test_everything_above_the_entry_function_stays_above_it(self):
        script = MultiPhaseDriverBuilder(
            driver=RawLuaDriver(source=_DRIVER, origin="d.lua")).build(_adoption_ctx())
        self.assertEqual(script.prelude[0], "-- a hand-written driver")
        self.assertEqual(script.entry.name, "infer")
        self.assertEqual(script.postlude, [""], "the source's trailing newline")

    def test_a_source_with_no_entry_function_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            RawLuaDriver(source="-- nothing here\n", origin="d.lua")
        self.assertIn("no top-level 'function infer(inputs)' line", str(raised.exception))

    def test_a_corrupted_round_trip_is_caught_before_anything_is_written(self):
        driver = RawLuaDriver(source=_DRIVER, origin="d.lua")
        driver._body[0] = driver._body[0].lstrip()
        with self.assertRaises(ValueError) as raised:
            driver.assert_round_trip()
        self.assertIn("does not reproduce its own source", str(raised.exception))


class TestParsingRunSubgraphCallSites(unittest.TestCase):
    """The half of C.3 that makes the step more than bookkeeping. Wrapping a driver in a `RawBlock`
    checks *nothing* on its own -- `check_subgraph_calls` walks `SubgraphCall` nodes and raw text has
    none -- so the adoption parses its own call sites and declares them."""

    def test_a_multi_line_table_literal_is_read(self):
        calls, unresolved = parse_run_subgraph_calls(_DRIVER, "d.lua")
        self.assertEqual(unresolved, [])
        self.assertEqual([(c.topology, c.inputs, c.line) for c in calls],
                         [("encoder", ("tokens", "style"), 7), ("vocoder", ("mel",), 11)])

    def test_a_computed_topology_name_is_reported_rather_than_dropped(self):
        """Every one of these in the real drivers is a BiLSTM/resblock stepping loop -- D.2's
        component. An adoption that silently ignored them while reporting the rest as checked would be
        "validated where convenient" one level down."""
        source = _DRIVER.replace('"vocoder"', 'prefix .. "_vocoder"')
        calls, unresolved = parse_run_subgraph_calls(source, "d.lua")
        self.assertEqual([c.topology for c in calls], ["encoder"])
        self.assertEqual(unresolved, [(11, 'prefix .. "_vocoder"')])

    def test_a_non_literal_argument_table_leaves_the_input_set_unknown(self):
        """`render_sampler` emits exactly this: a prepared `args` variable. The topology name is still
        checkable; the input set is not, and `WhenSet` is what says so rather than pretending."""
        source = _DRIVER.replace("{ mel = a }", "args")
        calls, _ = parse_run_subgraph_calls(source, "d.lua")
        self.assertIsNone(calls[1].inputs)

    def test_coverage_reports_the_two_numbers_separately(self):
        driver = RawLuaDriver(source=_DRIVER.replace("{ mel = a }", "args"), origin="d.lua")
        self.assertEqual(
            driver.coverage(),
            "  d.lua: 2/2 loom.run_subgraph call sites checked against the exported topologies "
            "(1 with their full input set)")


class TestTheAdoptedDriverIsActuallyChecked(unittest.TestCase):
    def test_a_call_naming_a_topology_the_export_did_not_produce_fails_with_its_line(self):
        source = _DRIVER.replace('"vocoder"', '"vocodr"')
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(source=source, origin="d.lua"))
        with self.assertRaises(LinkError) as raised:
            builder.build(_adoption_ctx())
        self.assertEqual(
            str(raised.exception),
            "d.lua:11 loom.run_subgraph('vocodr') names topology 'vocodr', which is not among the "
            "exported topologies ['encoder', 'vocoder'].",
        )

    def test_a_call_supplying_an_input_the_topology_does_not_declare_fails(self):
        source = _DRIVER.replace("style = inputs.style", "styl = inputs.style")
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(source=source, origin="d.lua"))
        with self.assertRaises(LinkError) as raised:
            builder.build(_adoption_ctx())
        self.assertEqual(
            str(raised.exception),
            "d.lua:7 loom.run_subgraph('encoder') does not match topology 'encoder': supplies input(s) "
            "it does not declare: ['styl']; leaves declared input(s) unsupplied: ['style']; topology "
            "declares ['tokens', 'style'], spec supplies ['tokens', 'styl'].",
        )

    def test_the_adoption_still_works_on_a_real_peeled_family_s_fragments(self):
        """All five families are peeled as of C.8, so there is no shipped whole-driver `.lua` left to
        round-trip. The adoption is not dead code -- it is how the *next* family arrives -- so it is
        exercised against a real driver reassembled from a peeled family's own fragments, which is the
        closest thing to the shape it was written for."""
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig

        config = TTSMatchaExportConfig(model_dir="/unused", output_path="/unused",
                                       architecture="matcha")
        source = MultiPhaseDriverBuilder(peeled=config.driver_components()).render(
            DriverContext(
                topologies={"encoder_mu": _topo(["tokens"]), "encoder_logw": _topo(["tokens"]),
                            "decoder": _topo(["z", "mu", "t"]), "vocoder": _topo(["mel"])},
                axes={n: "n_tokens" for n in ("encoder_mu", "encoder_logw", "decoder", "vocoder")},
            ))
        RawLuaDriver(source=source, origin="rebuilt.lua").assert_round_trip()

    def test_no_whole_driver_lua_is_still_shipped(self):
        """C.4-C.8 peeled all five. Asserted rather than assumed, so a family reverting to one file is
        a deliberate act rather than a silent one."""
        leftovers = sorted(Path(__file__).resolve().parents[1].glob("convert_*/*_driver_mil.lua"))
        self.assertEqual(leftovers, [])


class TestExternalTopologies(unittest.TestCase):
    """Kokoro and StyleTTS2 are *partial* MIL exports: their drivers call topologies still loaded from
    the pre-MIL `.gguf` alongside the exported one. C.3's own gate is what surfaced that -- it was
    recorded only in a C++ test -- and `external_topologies()` is the finding, turned into a
    declaration that cannot rot in either direction."""

    def _ctx(self):
        return DriverContext(topologies={"encoder": _topo(["tokens", "style"])},
                             axes={"encoder": "n_tokens"})

    def test_a_declared_external_call_is_not_reported_as_missing(self):
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(
            source=_DRIVER, origin="d.lua", external={"vocoder": "the pre-MIL gguf"}))
        self.assertEqual(builder.render(self._ctx()), _DRIVER)

    def test_an_undeclared_one_still_fails(self):
        """The reason this is a declaration rather than "skip anything not exported": a typo and a
        cross-GGUF dependency would otherwise be indistinguishable, which is the whole class of bug
        the check exists for."""
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(source=_DRIVER, origin="d.lua"))
        with self.assertRaises(LinkError) as raised:
            builder.build(self._ctx())
        self.assertIn("names topology 'vocoder', which is not among the exported topologies",
                      str(raised.exception))

    def test_a_stale_declaration_fails(self):
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(
            source=_DRIVER, origin="d.lua",
            external={"encoder": "wrong -- this export produces it", "vocoder": "the pre-MIL gguf"}))
        with self.assertRaises(LinkError) as raised:
            builder.build(self._ctx())
        self.assertEqual(
            str(raised.exception),
            "RawLuaDriver('d.lua') declares topolog(ies) ['encoder'] as coming from outside this "
            "export, but this export produces them. A stale external declaration silently suppresses "
            "the very check it was added to make honest -- drop it from external_topologies().",
        )

    def test_a_dead_declaration_fails(self):
        builder = MultiPhaseDriverBuilder(driver=RawLuaDriver(
            source=_DRIVER, origin="d.lua",
            external={"vocoder": "the pre-MIL gguf", "nobody_calls_this": "nor this"}))
        with self.assertRaises(LinkError) as raised:
            builder.build(self._ctx())
        self.assertEqual(
            str(raised.exception),
            "RawLuaDriver('d.lua') declares topolog(ies) ['nobody_calls_this'] as external, but its "
            "driver never calls them. A dead declaration is worse than none: it reads as an "
            "accounted-for dependency.",
        )

    def test_coverage_counts_external_calls_apart_from_checked_ones(self):
        driver = RawLuaDriver(source=_DRIVER, origin="d.lua", external={"vocoder": "the pre-MIL gguf"})
        self.assertEqual(
            driver.coverage(),
            "  d.lua: 1/2 loom.run_subgraph call sites checked against the exported topologies "
            "(1 with their full input set); 1 call topolog(ies) this export does not produce and the "
            "config declares external (vocoder)")

    def test_no_family_is_partial_any_more(self):
        """Both families that WERE partial now export every topology their driver calls.

        `external_topologies()` existed because Kokoro's and StyleTTS2's MIL exports were incomplete:
        their drivers ran against a mix of MIL topologies and pre-MIL ones from a second GGUF. P4.0.7
        closed that -- the six BiLSTMs became `RecurrentPhase`s and the rest ordinary traced phases --
        so both declare nothing.

        The declaration machinery stays and is still tested above, because the next partial family will
        need it. What is NOT asserted here is that every driver call resolves: that needs the real
        checkpoint, and `test_e2e_{kokoro,styletts2}_mil_lua_driver.cpp` are the authority, since they
        now load exactly one GGUF and a missing topology fails them outright."""
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig
        from loom_mil_compiler.styletts2_export import TTSStyleTTS2ExportConfig

        kokoro = TTSKokoroExportConfig(model_dir="/u", output_path="/u", architecture="kokoro")
        styletts2 = TTSStyleTTS2ExportConfig(checkpoint_path="/u", output_path="/u",
                                             architecture="styletts2")
        self.assertEqual(kokoro.external_topologies(), {})
        self.assertEqual(styletts2.external_topologies(), {})


# -- C.4: peeling a family into components ------------------------------------------------------------


def _ctx():
    """A context with no topologies: a fragment emits Lua text and calls nothing."""
    return DriverContext(topologies={})


class TestLuaFragment(unittest.TestCase):
    """A fragment's `reads`/`defines` are what make it composable -- `driver_ir.validate` runs over the
    assembled function, so a fragment placed before what it reads fails at export time. The declaration
    is therefore worth checking against the fragment's own text, since the rot that actually happens is
    renaming a local in the `.lua` and leaving the declaration behind."""

    def setUp(self):
        self.dir = Path(__file__).resolve().parent / "_fragment_fixture"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "block.lua"
        self.path.write_text("    local total = 0\n    for i = 1, #xs do total = total + xs[i] end\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_body_fragment_keeps_its_text_verbatim_and_carries_its_declaration(self):
        block, = LuaFragment(self.path, reads=("xs",), defines=("total",)).emit(_ctx())
        self.assertTrue(block.verbatim)
        self.assertEqual(block.reads(), ["xs"])
        self.assertEqual(block.defines(), ["total"])
        self.assertEqual(block.lines[0], "    local total = 0")

    def test_a_top_level_fragment_goes_to_the_prelude_instead(self):
        fragment = LuaFragment(self.path, top_level=True)
        self.assertEqual(fragment.emit(_ctx()), [])
        self.assertEqual(fragment.prelude(_ctx())[-1], "", "a blank line separates it from what follows")

    def test_a_stale_defines_declaration_fails(self):
        fragment = LuaFragment(self.path, reads=("xs",), defines=("tota",))
        with self.assertRaises(LinkError) as raised:
            _Peeled([fragment]).build(_ctx())
        self.assertEqual(
            str(raised.exception),
            "LuaFragment('block.lua') declares that it defines ['tota'], but its text never mentions "
            "those names. A stale defines list makes driver_ir.validate accept an ordering the driver "
            "does not actually support.",
        )

    def test_a_stale_reads_declaration_fails(self):
        fragment = LuaFragment(self.path, reads=("ys",), defines=("total",))
        with self.assertRaises(LinkError) as raised:
            _Peeled([fragment]).build(_ctx())
        self.assertIn("declares that it reads ['ys']", str(raised.exception))

    def test_an_out_of_order_fragment_is_caught_by_validate(self):
        """The check the peel exists for: a block moved above what it reads is an export-time error
        rather than a runtime one -- and no hand-written driver has ever had it."""
        with self.assertRaises(DriverIRError) as raised:
            _Peeled([LuaFragment(self.path, reads=("xs",), defines=("total",))]).build(_ctx())
        self.assertIn("symbol 'xs' is read", str(raised.exception))


class _Peeled(MultiPhaseDriverBuilder):
    def __init__(self, components):
        super().__init__(peeled=list(components))


class TestPeeledMatcha(unittest.TestCase):
    """The first peeled family, driven from the real config and the real `.lua` fragments."""

    def _ctx(self):
        return DriverContext(
            topologies={"encoder_mu": _topo(["tokens"]), "encoder_logw": _topo(["tokens"]),
                        "decoder": _topo(["z", "mu", "t"]), "vocoder": _topo(["mel"])},
            axes={n: "n_tokens" for n in ("encoder_mu", "encoder_logw", "decoder", "vocoder")},
        )

    def _config(self):
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig

        return TTSMatchaExportConfig(model_dir="/unused", output_path="/unused", architecture="matcha")

    def _render(self):
        return MultiPhaseDriverBuilder(peeled=self._config().driver_components()).render(self._ctx())

    def test_it_assembles_into_a_driver_that_validates(self):
        text = self._render()
        self.assertIn("function infer(inputs)", text)
        self.assertTrue(text.endswith("    return waveform\nend\n"),
                        "a peeled driver ends with a newline, like every hand-written one")

    def test_the_generated_sampler_lands_above_the_entry_function(self):
        """`FlowMatchingSampler` sits at its CALL site in the component list -- placing it earlier to
        match where its function appears is the ordering validate() rejects, since the call reads
        `t_mel`. Its prelude is collected separately, so the function still comes out on top."""
        text = self._render()
        self.assertLess(text.index("local function sample_decoder"),
                        text.index("function infer(inputs)"))
        self.assertLess(text.index("function infer(inputs)"),
                        text.index("local z = sample_decoder("))

    def test_every_run_subgraph_call_is_ir_rather_than_text(self):
        """What the peel buys structurally: `check_subgraph_calls` sees these directly, including their
        output arity, which a call site parsed out of raw text cannot."""
        script = MultiPhaseDriverBuilder(peeled=self._config().driver_components()).build(self._ctx())
        calls = [s for s in script.entry.body if isinstance(s, SubgraphCall)]
        self.assertEqual([c.module for c in calls], ["encoder_mu", "encoder_logw", "vocoder"])

    def test_func_name_is_now_a_real_driver_symbol(self):
        """The gap FlowMatchingSpec's own Unchecked note predicted P4.0.6 would close: the sampler's
        name was a string matched by a marker, and is now a symbol the built script either defines or
        does not.

        Renaming `func_name` would not test it -- one component emits both the definition and the
        call, so they move together. What the link guards is a driver that *calls* the sampler without
        the definition being emitted at all, which is precisely what the marker-substitution form could
        produce silently: a `.lua` calling `sample_decoder` whose marker line was deleted."""
        class _CallOnly(FlowMatchingSampler):
            def prelude(self, ctx):
                return []

        components = []
        for component in self._config().driver_components():
            if isinstance(component, FlowMatchingSampler):
                component = _CallOnly(**{f.name: getattr(component, f.name)
                                         for f in dataclasses.fields(FlowMatchingSampler)})
            components.append(component)
        with self.assertRaises(LinkError) as raised:
            MultiPhaseDriverBuilder(peeled=components).build(self._ctx())
        self.assertIn("reads driver symbol(s) ['sample_decoder']", str(raised.exception))

    def test_a_call_whose_inputs_drift_from_the_topology_fails(self):
        components = self._config().driver_components()
        call = next(c for c in components
                    if isinstance(c, SubgraphCallComponent) and c.topology == "vocoder")
        call.inputs = {"mels": call.inputs["mel"]}
        with self.assertRaises(LinkError) as raised:
            MultiPhaseDriverBuilder(peeled=components).build(self._ctx())
        self.assertEqual(
            str(raised.exception),
            "loom.run_subgraph('vocoder') does not match topology 'vocoder': supplies input(s) it does "
            "not declare: ['mels']; leaves declared input(s) unsupplied: ['mel']; topology declares "
            "['mel'], spec supplies ['mels'].",
        )


class TestPeeledSupertonic(unittest.TestCase):
    """The second peeled family, and the only thing worth asserting about it separately: it is built
    from the same classes as the first, differing only in data.

    That is P4.0.7's reuse claim tested rather than asserted, and it is why the plan orders Supertonic
    straight after Matcha instead of after the harder families -- if the second family had needed a new
    component, the component API would have been shaped by one example."""

    def _config(self):
        from loom_mil_compiler.supertonic_export import TTSSupertonicExportConfig

        return TTSSupertonicExportConfig(model_dir="/unused", output_path="/unused",
                                         architecture="supertonic")

    def _matcha_config(self):
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig

        return TTSMatchaExportConfig(model_dir="/unused", output_path="/unused", architecture="matcha")

    def test_it_introduces_no_component_class_matcha_did_not_already_use(self):
        supertonic = {type(c) for c in self._config().driver_components()}
        matcha = {type(c) for c in self._matcha_config().driver_components()}
        self.assertEqual(supertonic - matcha, set())

    def test_it_assembles_into_a_driver_that_validates(self):
        ctx = DriverContext(
            topologies={"dp": _topo(["txt_ids", "stl_emb"]), "ttl_text": _topo(["txt_ids", "stl_emb"]),
                        "vfe": _topo(["z_t", "txt_emb", "stl_emb", "t"]), "decoder": _topo(["latent"])},
            axes={n: "n_tokens" for n in ("dp", "ttl_text", "vfe", "decoder")},
        )
        text = MultiPhaseDriverBuilder(peeled=self._config().driver_components()).render(ctx)
        self.assertLess(text.index("local function sample_vfe"), text.index("function infer"))
        self.assertTrue(text.endswith("    return waveform\nend\n"))


class TestPeeledVits(unittest.TestCase):
    """The first peeled family with no sampler at all: VITS's stochastic duration predictor is traced
    into the `logw` topology itself, so the only host-side randomness is two `loom.gaussian_array`
    draws. Worth its own test because it is the case that could have needed a new component and did
    not."""

    def _components(self):
        from loom_mil_compiler.vits_export import TTSVitsExportConfig

        return TTSVitsExportConfig(checkpoint_path="/unused", output_path="/unused",
                                   architecture="vits").driver_components()

    def test_it_needs_no_sampler_component(self):
        self.assertEqual([c for c in self._components() if isinstance(c, FlowMatchingSampler)], [])

    def test_it_introduces_no_new_component_class(self):
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig

        matcha = {type(c) for c in TTSMatchaExportConfig(
            model_dir="/unused", output_path="/unused", architecture="matcha").driver_components()}
        self.assertEqual({type(c) for c in self._components()} - matcha, set())

    def test_the_frame_expansion_stays_hand_written_and_declares_what_it_touches(self):
        """Genuine host control flow over a data-dependent frame count -- BACKEND.md's own conclusion
        about what stays host-side. What the peel adds is that it now says what it reads."""
        expand = next(c for c in self._components()
                      if isinstance(c, LuaFragment) and "expand_z_p" in str(c.path))
        self.assertEqual(set(expand.reads), {"stats", "w_ceil", "y_length", "T"})
        self.assertIn("z_p", expand.defines)


class TestPeeledKokoro(unittest.TestCase):
    """A deliberately thin peel, and the property worth pinning is that it did not *reduce* checking.

    Kokoro makes eleven run_subgraph calls and only two can become IR: seven name their topology with
    a computed expression and two sit inside Lua `for` loops. The rest stay in fragments -- which parse
    their own call sites for exactly this reason."""

    def _components(self):
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig

        return TTSKokoroExportConfig(model_dir="/unused", output_path="/unused",
                                     architecture="kokoro").driver_components()

    def test_the_two_mil_calls_became_ir(self):
        calls = [c for c in self._components() if isinstance(c, SubgraphCallComponent)]
        self.assertEqual([c.topology for c in calls], ["albert_bert_encoder", "decoder_vocoder"])

    def test_the_fragments_still_parse_the_calls_that_could_not(self):
        """A peel must never take a call site out of reach of the parser. `duration_proj` sits inside
        a Lua loop and `text_encoder_cnn` is external -- both are still seen."""
        fragments = [c for c in self._components() if isinstance(c, LuaFragment)]
        seen = {call.topology for f in fragments for call in f._calls}
        self.assertIn("duration_proj", seen)
        self.assertIn("text_encoder_cnn", seen)

    def test_a_fragment_call_is_now_checked_rather_than_declared_external(self):
        """The other side of the `external_topologies()` story. While the export was partial these two
        were *excluded* from checking, because the topologies they name were not in the file. Now that
        the export produces them, the exclusion is gone and the calls are link-checked like any other
        -- which is strictly better, and is what "self-contained" buys beyond packaging."""
        fragments = [c for c in self._components() if isinstance(c, LuaFragment)]
        # `sub_specs` carries the `drives` declarations themselves (D.2) alongside the calls they
        # expand into, since a declaration has links of its own; only the calls name a topology.
        declared = {getattr(call, "topology", None) for f in fragments for call in f.sub_specs()}
        self.assertIn("text_encoder_cnn", declared)
        self.assertIn("duration_proj", declared)

    def test_the_seven_input_call_is_rendered_as_a_block(self):
        """Cosmetic and load-bearing: the embedded driver_script is what someone inspecting a GGUF
        reads, and seven inputs on one line is a 200-column line."""
        call = next(c for c in self._components()
                    if isinstance(c, SubgraphCallComponent) and c.topology == "decoder_vocoder")
        self.assertTrue(call.multiline)


class TestPeeledStyleTTS2(unittest.TestCase):
    """The last family, and the one the plan predicted would "stay partly raw". It does, and the
    boundary is exactly where the plan said: the ADPM2 sampler's `diffusion` call lives inside
    `denoise_fn`, a closure the sampler invokes twice per step, so it cannot be a statement in the
    entry function at all. That call is what `estimators()`' `EstimatorSpec` is for -- checked without
    being generated."""

    def _config(self):
        from loom_mil_compiler.styletts2_export import TTSStyleTTS2ExportConfig

        return TTSStyleTTS2ExportConfig(checkpoint_path="/unused", output_path="/unused",
                                        architecture="styletts2")

    def test_only_the_two_top_level_mil_calls_became_ir(self):
        calls = [c for c in self._config().driver_components()
                 if isinstance(c, SubgraphCallComponent)]
        self.assertEqual([c.topology for c in calls], ["albert", "decoder_vocoder"])

    def test_the_closure_bound_diffusion_call_is_covered_by_an_estimator_spec(self):
        """Not by a component and not by a fragment link: `EstimatorSpec` is the declaration that
        checks a call without generating it, and this family is the reason that split exists."""
        specs = self._config().estimators()
        self.assertEqual([s.topology for s in specs], ["diffusion"])

    def test_the_adpm2_helpers_stay_hand_written_lua(self):
        """Two network evaluations per step, Karras preconditioning, per-step noise injection. No
        template emits that without becoming a worse thing to read than the loop.

        They live in `loom_lua` rather than in this family's header -- not because they are shared (only
        StyleTTS2 calls them) but because the library is where hand-written Lua functions live at all.
        Being there is what makes "only StyleTTS2 ships them" a checked property instead of a
        side effect of which file they happened to be pasted into."""
        from loom_mil_compiler.lua_library import LuaLibrary, resolve

        library = next(c for c in self._config().driver_components() if isinstance(c, LuaLibrary))
        emitted = {fn.name for fn in resolve(library.uses)}
        self.assertEqual(emitted & {"karras_schedule", "adpm2_step", "adpm2_sample"},
                         {"karras_schedule", "adpm2_step", "adpm2_sample"})

    def test_no_other_family_ships_the_adpm2_sampler(self):
        from loom_mil_compiler.lua_library import LuaLibrary, resolve
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig

        matcha = TTSMatchaExportConfig(model_dir="/u", output_path="/u", architecture="matcha")
        library = next(c for c in matcha.driver_components() if isinstance(c, LuaLibrary))
        self.assertEqual({fn.name for fn in resolve(library.uses)} & {"adpm2_step"}, set())


class TestComputedCallSitesAreDeclared(unittest.TestCase):
    """D.2: the call sites whose topology name is built at run time.

    Before this, Kokoro's and StyleTTS2's drivers had two kinds of call site no check could reach --
    one built in a Lua `for` loop, and one built *inside* a `loom_lua` helper, a level below the
    fragment entirely. `parse_run_subgraph_calls` reported the first as unresolved and never saw the
    second at all. What is tested here is that declaring them as data restores the ordinary checks
    rather than merely recording the gap: the expansion is a `RunSubgraphCall` like any other, so a
    namespace whose topologies were not exported fails with `TopologyName`'s own message.
    """

    def _fragments(self, config):
        return [c for c in config.driver_components() if isinstance(c, LuaFragment)]

    def _kokoro(self):
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig

        return TTSKokoroExportConfig(model_dir="/unused", output_path="/unused",
                                     architecture="kokoro")

    def test_a_bilstm_namespace_expands_to_its_four_cell_topologies(self):
        fragment = next(f for f in self._fragments(self._kokoro())
                        if f.path.name == "03_frame_expansion.lua")
        call, = fragment.drives
        self.assertEqual([c.topology for c in call.expand()],
                         ["text_encoder_lstm_h_fwd", "text_encoder_lstm_c_fwd",
                          "text_encoder_lstm_h_bwd", "text_encoder_lstm_c_bwd"])
        self.assertEqual(call.expand()[0].inputs, ("layer_input", "h_prev", "c_prev"))

    def test_a_loop_declares_every_namespace_it_iterates(self):
        fragment = next(f for f in self._fragments(self._kokoro())
                        if f.path.name == "02_duration_encoder.lua")
        lstm = next(d for d in fragment.drives if getattr(d, "helper", None) == "run_bi_lstm"
                    and len(d.names()) == 3)
        self.assertEqual([c.topology for c in lstm.expand()][:2],
                         ["duration_lstm_0_h_fwd", "duration_lstm_0_c_fwd"])
        adaln = next(d for d in fragment.drives if not hasattr(d, "helper"))
        self.assertEqual([c.topology for c in adaln.expand()],
                         ["duration_adaln_0", "duration_adaln_1", "duration_adaln_2"])

    def test_every_computed_call_site_in_every_family_is_declared(self):
        """The completeness direction, over the real fragments: a call site nobody declared is a
        topology nothing checks, which is exactly the state D.2 ends."""
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig
        from loom_mil_compiler.matcha_export import TTSMatchaExportConfig
        from loom_mil_compiler.styletts2_export import TTSStyleTTS2ExportConfig
        from loom_mil_compiler.supertonic_export import TTSSupertonicExportConfig
        from loom_mil_compiler.vits_export import TTSVitsExportConfig

        configs = (
            self._kokoro(),
            TTSStyleTTS2ExportConfig(checkpoint_path="/u", output_path="/u", architecture="styletts2"),
            TTSMatchaExportConfig(model_dir="/u", output_path="/u", architecture="matcha"),
            TTSVitsExportConfig(checkpoint_path="/u", output_path="/u", architecture="vits"),
            TTSSupertonicExportConfig(model_dir="/u", output_path="/u", architecture="supertonic"),
        )
        for config in configs:
            for fragment in self._fragments(config):
                self.assertEqual(fragment.undeclared_call_sites(), [],
                                 f"{type(config).__name__} {fragment.path.name}")
                self.assertEqual(fragment.stale_drives_declarations(), [],
                                 f"{type(config).__name__} {fragment.path.name}")

    def test_a_declaration_whose_call_site_was_renamed_is_reported(self):
        """The direction that turns a declaration into decoration: the Lua changes, the declaration
        stays, and it keeps checking a topology nothing runs."""
        from loom_mil_compiler.driver_components import HelperCall

        fragment = next(f for f in self._fragments(self._kokoro())
                        if f.path.name == "04_f0n.lua")
        stale = dataclasses.replace(fragment, drives=fragment.drives + (
            HelperCall("run_proj1x1", "f0n_gone"),))
        self.assertEqual(stale.stale_drives_declarations(), ['"f0n_gone"'])

    def test_a_call_site_nobody_declared_is_reported_with_its_line(self):
        fragment = next(f for f in self._fragments(self._kokoro())
                        if f.path.name == "04_f0n.lua")
        thinned = dataclasses.replace(fragment, drives=fragment.drives[:-1])
        missing = thinned.undeclared_call_sites()
        self.assertEqual([site[:2] for site in missing], [("run_proj1x1", '"f0n_n_proj"')])
        self.assertGreater(missing[0][2], 0)

    def test_the_declarations_fail_the_link_check_against_a_topology_set_without_them(self):
        """End to end through the protocol rather than through `expand()`: this is the message a real
        export produces, and it is the ordinary `TopologyName` one."""
        from loom_mil_compiler.spec_protocol import LinkChecker

        fragment = next(f for f in self._fragments(self._kokoro())
                        if f.path.name == "03_frame_expansion.lua")
        checker = LinkChecker()
        for spec in fragment.sub_specs():
            checker.check(spec)
        with self.assertRaises(LinkError) as raised:
            checker.provide(topologies={"text_encoder_cnn": _topo(["tokens"])})
        message = str(raised.exception)
        self.assertIn("text_encoder_lstm_h_fwd", message)
        self.assertIn("via run_bi_lstm", message)

    def test_an_unknown_helper_is_reported_rather_than_expanding_to_nothing(self):
        from loom_mil_compiler.driver_components import HelperCall
        from loom_mil_compiler.spec_protocol import check_links

        with self.assertRaises(LinkError) as raised:
            check_links(HelperCall("array_sum", "whatever"))
        self.assertIn("declares no `drives`", str(raised.exception))


class TestPeeledDriverCoverage(unittest.TestCase):
    def test_every_exported_topology_is_named_by_a_checked_call_site(self):
        """Kokoro's own numbers, without an export: 39 topologies, and the driver names all of them
        once the computed sites are declared. Four before D.2."""
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig

        config = TTSKokoroExportConfig(model_dir="/unused", output_path="/unused",
                                       architecture="kokoro")
        builder = MultiPhaseDriverBuilder(peeled=config.driver_components())
        called = builder.called_topologies()
        self.assertEqual(len(called), 39)
        for name in ("albert_bert_encoder", "decoder_vocoder", "text_encoder_cnn", "duration_proj",
                     "duration_adaln_2", "top_lstm_c_bwd", "f0n_f0_block2", "f0n_n_proj"):
            self.assertIn(name, called)
