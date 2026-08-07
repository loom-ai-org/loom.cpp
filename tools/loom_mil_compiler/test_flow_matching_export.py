"""
Checks `flow_matching_export.py` (EXPORT-IMPROVEMENT.md item 4). Two things matter here:

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

from loom_mil_compiler.flow_matching_export import EstimatorSpec, FlowMatchingSpec, render_sampler
from loom_mil_compiler.spec_protocol import check_links


def _check(spec, **topologies):
    """`spec`'s links against a fake export's topologies, the way `DriverBuilder.build` runs them.

    `strict=False` because `FlowMatchingSpec.func_name` is a `DriverSymbol`: it defers until the built
    driver exists, which is the builder's business and not this module's. What is being exercised here
    is the half answerable from the topologies alone -- and that the deferral is *reported* rather than
    skipped is `test_a_deferred_link_is_reported_rather_than_skipped` below."""
    return check_links(spec, topologies=topologies, strict=False)


def _topology(*input_names, outputs=("v",)):
    """A traced topology as `generate_graph_topology` emits it. The declared output is not decoration:
    the generated Euler loop indexes `run_subgraph`'s single return value, so `FlowMatchingSpec` now
    declares a `TopologyOutputArity` link against it (P4.0.5) -- a two-output estimator would bind `v`
    to the first output's data and integrate the wrong tensor."""
    return {"inputs": [{"name": n} for n in input_names], "outputs": list(outputs)}


MATCHA = FlowMatchingSpec(func_name="sample_decoder", estimator="decoder",
                                 carried_input="z", fixed_inputs=["mu"])
SUPERTONIC = FlowMatchingSpec(func_name="sample_vfe", estimator="vfe", carried_input="z_t",
                                     fixed_inputs=["txt_emb", "stl_emb"])


class TestRenderSampler(unittest.TestCase):
    def test_emits_the_declared_call_with_every_input(self):
        lua = render_sampler(SUPERTONIC)
        self.assertIn("local function sample_vfe(length, n_elems, n_steps, step_inputs)", lua)
        self.assertIn('loom.run_subgraph("vfe", {n_tokens = length, n_past = 0}, args)', lua)
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

    def test_rejects_an_estimator_the_export_never_produced(self):
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, vocoder=_topology("mel"))
        self.assertIn("decoder", str(cm.exception))


class TestBespokeEstimator(unittest.TestCase):
    """A hand-written sampler generates nothing but is still checked -- the codegen and the validation
    generalize to different extents, which is why EstimatorSpec exists separately.

    StyleTTS2 is the family this was written for, and since P4.0.18 it reaches the same two links
    through `LuaFragment`'s parse of its own `.lua` rather than through a declaration beside it -- see
    `test_driver_components.TestPeeledStyleTTS2`. What is exercised here is the spec itself, which
    is what `FlowMatchingSpec.estimator_spec()` still reduces to."""

    def test_a_bare_estimator_declaration_checks_clean(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "embedding"])
        self.assertEqual(_check(spec, diffusion=_topology("x_in", "time", "embedding")), [])

    def test_a_mismatched_bespoke_call_is_still_caught(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "attn_mask"])
        with self.assertRaises(ValueError) as cm:
            _check(spec, diffusion=_topology("x_in", "time", "embedding"))
        self.assertIn("attn_mask", str(cm.exception))
        self.assertIn("embedding", str(cm.exception))

    def test_both_spec_kinds_share_one_validation_implementation(self):
        """A refinement spec's own per-step call is an EstimatorSpec, so the two cannot drift apart."""
        self.assertEqual(MATCHA.estimator_spec(),
                         EstimatorSpec(topology="decoder", inputs=["z", "mu", "t"]))


class TestSpecProtocolRetrofit(unittest.TestCase):
    """P4.0.5 stage B.2. Two questions: did the messages survive the move onto `spec_protocol`, and
    does the protocol catch anything the hand-written validator did not."""

    def test_the_mismatch_message_is_preserved_verbatim(self):
        # The whole acceptance criterion of the protocol (`EXPORT-PREPARATION.md` §2): a generic checker
        # that degrades a specific message into "validation failed" is a regression, not a refactor. So
        # this asserts the exact string, not that *a* ValueError was raised.
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "mu", "t", "cond"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') does not match topology 'decoder': "
            "leaves declared input(s) unsupplied: ['cond']; "
            "topology declares ['z', 'mu', 't', 'cond'], spec supplies ['z', 'mu', 't'].",
        )

    def test_both_halves_of_the_message_still_appear_together(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "cond", "t"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') does not match topology 'decoder': "
            "supplies input(s) it does not declare: ['mu']; "
            "leaves declared input(s) unsupplied: ['cond']; "
            "topology declares ['z', 'cond', 't'], spec supplies ['z', 'mu', 't'].",
        )

    def test_the_unknown_topology_message_is_preserved_verbatim(self):
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, vocoder=_topology("mel"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') names topology 'decoder', which is not among the "
            "exported topologies ['vocoder'].",
        )

    def test_a_bespoke_estimator_is_still_named_by_its_topology(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in"])
        with self.assertRaises(ValueError) as cm:
            _check(spec, albert=_topology("tokens"))
        self.assertEqual(
            str(cm.exception),
            "EstimatorSpec('diffusion') names topology 'diffusion', which is not among the exported "
            "topologies ['albert'].",
        )

    def test_a_multi_output_estimator_is_now_rejected(self):
        """The check the hand-written validator never made. `render_sampler` emits
        `local v = loom.run_subgraph(...)` and indexes `v[i]`; against a two-output topology `v` binds
        the first output's DATA and the loop integrates the wrong tensor -- valid Lua, plausible
        shapes, wrong audio, and nothing reports it."""
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, decoder=_topology("z", "mu", "t", outputs=("v", "logdet")))
        self.assertIn("is built for a topology declaring 1 output(s)", str(cm.exception))
        self.assertIn("'decoder' declares 2: ['v', 'logdet']", str(cm.exception))

    def test_supplied_inputs_is_what_the_generated_lua_actually_passes(self):
        """The link's subject is a derived property, so the declaration cannot drift from the emission:
        every name checked here is a name `render_sampler` writes into the per-step table."""
        lua = render_sampler(SUPERTONIC)
        self.assertEqual(SUPERTONIC.supplied_inputs, ["z_t", "txt_emb", "stl_emb", "t"])
        for name in SUPERTONIC.supplied_inputs:
            self.assertIn(f"{name} = ", lua)

    def test_every_field_of_both_specs_is_declared(self):
        """The standing rule, on the first two specs to adopt the protocol: each field is either
        link-checked, covered by another field's link, or documented as uncheckable."""
        from loom_mil_compiler.spec_protocol import dangling_coverage, undeclared_fields

        for cls in (EstimatorSpec, FlowMatchingSpec):
            self.assertEqual(undeclared_fields(cls), [], cls.__name__)
            self.assertEqual(dangling_coverage(cls), [], cls.__name__)

    def test_a_deferred_link_is_reported_rather_than_skipped(self):
        """A spec registered with a checker that never gets the topologies must be REPORTED, not
        silently skipped. What must not happen is a caller believing the spec was validated."""
        from loom_mil_compiler.spec_protocol import LinkChecker, LinkError

        checker = LinkChecker()
        checker.check(MATCHA)
        with self.assertRaises(LinkError) as cm:
            checker.finish()
        self.assertIn("were never checked", str(cm.exception))
        self.assertIn("FlowMatchingSpec('sample_decoder').estimator", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
