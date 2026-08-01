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
    ModularChainBuilder, MonolithicCall, PrefillArgmaxBuilder, caller_input,
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
