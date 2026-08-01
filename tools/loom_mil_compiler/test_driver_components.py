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
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.driver_builder import DriverContext
from loom_mil_compiler.driver_components import (
    CALLER, MASK, POSITION, ArgmaxEpilogue, ChainStage, DriverInputs, ModularChain,
    ModularChainBuilder, MonolithicCall, MultiPhaseDriverBuilder, PrefillArgmaxBuilder, RawLuaDriver,
    caller_input, parse_run_subgraph_calls,
)
from loom_mil_compiler.driver_ir import BinOp, Len, Lit, LuaCodegen, Var
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
            call=MonolithicCall(topology="main_topo", inputs=inputs, n_tokens=n_tokens),
            epilogue=ArgmaxEpilogue(out_var="_mono_out", shape_var="_mono_shape", n_tokens=n_tokens),
        )

    def test_the_plain_causal_lm_shape(self):
        ctx = DriverContext(topologies={"main_topo": _topo(["tokens"])},
                            axes={"main_topo": "n_tokens"})
        text = self._builder((("tokens", CALLER),), ("tokens",), Len("tokens")).render(ctx)
        self.assertEqual(text, "\n".join([
            "function main(inputs)",
            "    local tokens = inputs.tokens",
            "    local _mono_out, _mono_shape = loom.run_subgraph('main_topo', "
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
            topologies={"main_topo": _topo(["input_ids", "cache_position", "attention_mask"])},
            axes={"main_topo": "n_tokens"})
        bindings = (("input_ids", CALLER), ("cache_position", POSITION), ("attention_mask", MASK))
        text = self._builder(bindings, tuple(n for n, _ in bindings), Len("input_ids")).render(ctx)
        self.assertIn("    local input_ids = (inputs.input_ids or inputs.tokens)\n"
                      "    local cache_position = loom.range(0, #input_ids)\n"
                      "    local attention_mask = loom.causal_mask(#input_ids, 0)\n", text)

    def test_the_root_axis_is_the_topology_s_own(self):
        """Conformer-CTC/Parakeet declare "n_samples" -- raw audio samples, never a token count. The
        VALUE is still the first input's own length; only the axis it binds differs (R1)."""
        ctx = DriverContext(topologies={"main_topo": _topo(["audio_signal"])},
                            axes={"main_topo": "n_samples"})
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
            "function main(inputs)",
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

function synthesize(inputs)
    local a = loom.run_subgraph("encoder", {n_tokens = #inputs.tokens, n_past = 0}, {
        tokens = inputs.tokens,
        style = inputs.style,
    })
    local b = loom.run_subgraph("vocoder", {n_enc_frames = 4, n_past = 0}, { mel = a })
    return b
end
"""


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
        self.assertEqual(script.entry.name, "synthesize")
        self.assertEqual(script.postlude, [""], "the source's trailing newline")

    def test_a_source_with_no_entry_function_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            RawLuaDriver(source="-- nothing here\n", origin="d.lua")
        self.assertIn("no top-level 'function synthesize(inputs)' line", str(raised.exception))

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

    def test_the_five_real_drivers_all_round_trip(self):
        """Not a fixture: the actual shipped `.lua` files, which is the only thing C.3's byte-identity
        gate is a claim about."""
        drivers = sorted(Path(__file__).resolve().parents[1].glob("convert_*/*_driver_mil.lua"))
        self.assertEqual(len(drivers), 5, [d.name for d in drivers])
        for path in drivers:
            RawLuaDriver(source=path.read_text(), origin=path.name).assert_round_trip()


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

    def test_the_two_partial_families_declare_exactly_what_their_drivers_call(self):
        """Driven from the real `.lua` files and the real configs, so a driver gaining a call into the
        bespoke gguf without a matching declaration fails here rather than at export time."""
        from loom_mil_compiler.kokoro_export import TTSKokoroExportConfig
        from loom_mil_compiler.styletts2_export import TTSStyleTTS2ExportConfig

        for config_class, exported in ((TTSKokoroExportConfig, {"albert_bert_encoder",
                                                                "decoder_vocoder"}),
                                       (TTSStyleTTS2ExportConfig, {"albert", "decoder_vocoder",
                                                                   "diffusion"})):
            path = config_class.__dataclass_fields__["driver_script_path"].default
            external = config_class.external_topologies(config_class)
            driver = RawLuaDriver(source=path.read_text(), origin=path.name, external=external)
            self.assertEqual(driver.called_topologies() - exported, set(external), path.name)
