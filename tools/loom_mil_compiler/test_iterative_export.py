"""
Checks `iterative_export.py` (EXPORT-IMPROVEMENT.md item 4). Two things matter here:

* the generated Euler sampler must reproduce, exactly, the loop the Matcha/Supertonic drivers hand-wrote
  -- those two are numerically pinned end-to-end by `test_e2e_{matcha,supertonic}_mil_lua_driver.cpp`,
  so what is worth testing at this level is the *shape* of what gets emitted and, above all,
* the export-time validation, which is the actual reason these are specs rather than hand-written Lua.
  A `run_subgraph` call whose argument names don't match the topology's declared inputs is otherwise
  only caught deep inside the engine at run time, with nothing pointing back at the driver line.
"""
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.iterative_export import (
    SAMPLER_MARKER, EstimatorSpec, IterativeRefinementSpec, render_driver, render_sampler,
)


def _topology(*input_names):
    return {"inputs": [{"name": n} for n in input_names]}


MATCHA = IterativeRefinementSpec(func_name="sample_decoder", estimator="decoder",
                                 carried_input="z", fixed_inputs=["mu"])
SUPERTONIC = IterativeRefinementSpec(func_name="sample_vfe", estimator="vfe", carried_input="z_t",
                                     fixed_inputs=["txt_emb", "stl_emb"])


class TestRenderSampler(unittest.TestCase):
    def test_emits_the_declared_call_with_every_input(self):
        lua = render_sampler(SUPERTONIC)
        self.assertIn("local function sample_vfe(length, n_elems, n_steps, step_inputs)", lua)
        self.assertIn('loom.run_subgraph("vfe", length, 0, args)', lua)
        for line in ("z_t = z,", "txt_emb = step_inputs.txt_emb,",
                     "stl_emb = step_inputs.stl_emb,", "t = { t },"):
            self.assertIn(line, lua)

    def test_emits_forward_euler_with_uniform_dt(self):
        lua = render_sampler(MATCHA)
        self.assertIn("local dt = 1.0 / n_steps", lua)
        self.assertIn("for step = 0, n_steps - 1 do", lua)
        self.assertIn("local t = step / n_steps", lua)
        self.assertIn("z[i] = z[i] + v[i] * dt", lua)

    def test_uses_only_double_quotes_so_the_lua_stays_valid(self):
        """The generated code is spliced into a Lua file; a stray Python repr quote would break it."""
        for spec in (MATCHA, SUPERTONIC):
            for line in render_sampler(spec).splitlines():
                if line.lstrip().startswith("--"):
                    continue  # prose comments may legitimately contain apostrophes
                self.assertNotIn("'", line, line)


class TestValidation(unittest.TestCase):
    def test_accepts_a_spec_matching_its_topology(self):
        MATCHA.validate_against_topology(_topology("z", "mu", "t"))
        SUPERTONIC.validate_against_topology(_topology("z_t", "txt_emb", "stl_emb", "t"))

    def test_rejects_an_input_the_topology_never_declared(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "t"))
        self.assertIn("'mu'", str(cm.exception))
        self.assertIn("sample_decoder", str(cm.exception))

    def test_rejects_leaving_a_declared_input_unsupplied(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "mu", "t", "cond"))
        self.assertIn("'cond'", str(cm.exception))

    def test_render_driver_rejects_an_unknown_estimator(self):
        with self.assertRaises(ValueError) as cm:
            render_driver(SAMPLER_MARKER, [MATCHA], topologies={"vocoder": _topology("mel")})
        self.assertIn("decoder", str(cm.exception))

    def test_render_driver_requires_a_marker_when_it_has_something_to_emit(self):
        with self.assertRaises(ValueError):
            render_driver("-- no marker here\n", [MATCHA])

    def test_render_driver_substitutes_the_marker(self):
        out = render_driver(f"-- head\n{SAMPLER_MARKER}\n-- tail\n", [MATCHA],
                            topologies={"decoder": _topology("z", "mu", "t")})
        self.assertNotIn(SAMPLER_MARKER, out)
        self.assertIn("local function sample_decoder", out)
        self.assertTrue(out.startswith("-- head\n") and out.endswith("-- tail\n"))


class TestBespokeEstimator(unittest.TestCase):
    """StyleTTS2's ADPM2 sampler generates nothing but is still checked -- the codegen and the
    validation generalize to different extents, which is why EstimatorSpec exists separately."""

    def test_a_bare_estimator_declaration_validates_without_a_marker(self):
        source = "-- hand-written ADPM2 driver, nothing generated\n"
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "embedding"])
        out = render_driver(source, topologies={"diffusion": _topology("x_in", "time", "embedding")},
                            estimators=[spec])
        self.assertEqual(out, source)

    def test_a_mismatched_bespoke_call_is_still_caught(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "attn_mask"])
        with self.assertRaises(ValueError) as cm:
            render_driver("-- driver\n", topologies={"diffusion": _topology("x_in", "time", "embedding")},
                          estimators=[spec])
        self.assertIn("attn_mask", str(cm.exception))
        self.assertIn("embedding", str(cm.exception))

    def test_both_spec_kinds_share_one_validation_implementation(self):
        """A refinement spec's own per-step call is an EstimatorSpec, so the two cannot drift apart."""
        self.assertEqual(MATCHA.estimator_spec(),
                         EstimatorSpec(topology="decoder", inputs=["z", "mu", "t"]))


if __name__ == "__main__":
    unittest.main()
