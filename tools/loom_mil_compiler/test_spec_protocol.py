"""Checks `spec_protocol.py` (BACKLOG.md P4.0.5, `EXPORT-PREPARATION.md` stage B.1).

Two things are worth testing here and they are not the same thing:

* **the link kinds hold what they claim** -- each one is a predicate lifted out of a hand-written
  validator, so the test that matters is that it still catches what that validator caught, with a
  message that still names the specifics. Retrofit-specific message fidelity lives in each family's own
  test file (B.2-B.5); what is tested here is the vocabulary in isolation.
* **deferral cannot become a silent opt-out.** A link whose context never arrives must be *reported*,
  and that is the single design detail the plan says must not be skipped -- because a skipped check and
  a passing check are indistinguishable from outside, so nothing else in the suite can catch it.
"""
import unittest

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loom_mil_compiler.spec_protocol import (
    Axis, ConfigDerived, DriverSymbol, EachOf, FieldRef, LinkChecker, LinkCheckContext, LinkError,
    ModuleAttrPath, TopologyInput, TopologyName, TopologyOutputArity, Unchecked, WeightName, WhenSet,
    check_links, declared_links, spec_label, undeclared_fields,
)


def _topology(*input_names, outputs=("out",)):
    return {"inputs": [{"name": n} for n in input_names], "outputs": list(outputs)}


TOPOLOGIES = {
    "decoder": _topology("z", "mu", "t"),
    "two_out": _topology("x", outputs=("a", "b")),
    "singular": {"inputs": [{"name": "x"}], "output": "var_9"},
}


@dataclass
class _CallSpec:
    topology: str
    inputs: list
    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology")),
    }


class TestTopologyLinks(unittest.TestCase):
    def test_topology_name_names_the_real_alternatives(self):
        with self.assertRaises(LinkError) as cm:
            check_links(_CallSpec("nope", []), topologies=TOPOLOGIES)
        msg = str(cm.exception)
        self.assertIn("names topology 'nope'", msg)
        # The list of names that DO exist is the whole value of this message -- it turns a typo into a
        # one-line fix instead of a grep.
        self.assertIn("['decoder', 'singular', 'two_out']", msg)

    def test_topology_input_reports_both_directions(self):
        with self.assertRaises(LinkError) as cm:
            check_links(_CallSpec("decoder", ["z", "bogus"]), topologies=TOPOLOGIES)
        msg = str(cm.exception)
        self.assertIn("supplies input(s) it does not declare: ['bogus']", msg)
        self.assertIn("leaves declared input(s) unsupplied: ['mu', 't']", msg)
        self.assertIn("topology declares ['z', 'mu', 't'], spec supplies ['z', 'bogus']", msg)

    def test_topology_input_exact_false_allows_a_subset(self):
        @dataclass
        class Partial:
            topology: str
            inputs: list
            __links__ = {"inputs": TopologyInput("decoder", exact=False)}

        check_links(Partial("decoder", ["z"]), topologies=TOPOLOGIES)
        with self.assertRaises(LinkError):
            check_links(Partial("decoder", ["z", "nope"]), topologies=TOPOLOGIES)

    def test_output_arity_counts_both_topology_spellings(self):
        @dataclass
        class Arity:
            topology: str
            __links__ = {"topology": [TopologyName(), TopologyOutputArity(FieldRef("topology"), 1)]}

        # "output" (singular string) and "outputs" (plural array) are the two real spellings; a
        # single-output topology written either way must count as one.
        check_links(Arity("singular"), topologies=TOPOLOGIES)
        check_links(Arity("decoder"), topologies=TOPOLOGIES)
        with self.assertRaises(LinkError) as cm:
            check_links(Arity("two_out"), topologies=TOPOLOGIES)
        self.assertIn("declares 2: ['a', 'b']", str(cm.exception))


class _Leaf:
    pass


class _Model:
    def __init__(self):
        self.model = _Inner()
        self.lm_head = _Leaf()


class _Inner:
    def __init__(self):
        self.embed_tokens = _Leaf()
        self.layers = [1, 2]


class TestModuleAttrPath(unittest.TestCase):
    def test_resolves_a_dotted_path(self):
        @dataclass
        class Paths:
            prefix_attr: str
            suffix_attrs: list = field(default_factory=list)
            aux_attr: object = None
            __links__ = {
                "prefix_attr": ModuleAttrPath(),
                "suffix_attrs": EachOf(ModuleAttrPath()),
                "aux_attr": WhenSet(ModuleAttrPath()),
            }

        check_links(Paths("model.embed_tokens", ["lm_head"]), model=_Model())

    def test_names_the_path_the_component_and_the_type_it_failed_on(self):
        @dataclass
        class Paths:
            prefix_attr: str
            __links__ = {"prefix_attr": ModuleAttrPath()}

        with self.assertRaises(LinkError) as cm:
            check_links(Paths("model.embed_token"), model=_Model())
        msg = str(cm.exception)
        # All three specifics: which field, which path, and -- the one that says whether this is a typo
        # or the wrong nesting level -- what type it actually walked into.
        self.assertIn("prefix_attr = 'model.embed_token'", msg)
        self.assertIn("_Inner (at model) has no attribute 'embed_token'", msg)

    def test_each_of_names_the_offending_index(self):
        @dataclass
        class Paths:
            suffix_attrs: list
            __links__ = {"suffix_attrs": EachOf(ModuleAttrPath())}

        with self.assertRaises(LinkError) as cm:
            check_links(Paths(["lm_head", "nope"]), model=_Model())
        self.assertIn("suffix_attrs[1] = 'nope'", str(cm.exception))

    def test_when_set_skips_none_without_deferring(self):
        @dataclass
        class Paths:
            aux_attr: object = None
            __links__ = {"aux_attr": WhenSet(ModuleAttrPath())}

        # Neither a failure nor a deferral: the field is simply not in use. If this deferred, every
        # family that leaves an optional field unset would trip `finish()`.
        self.assertEqual(check_links(Paths(), model=_Model()), [])


class TestAxis(unittest.TestCase):
    @dataclass
    class Phase:
        root_axis: str = "n_tokens"
        declared_axes: object = None
        __links__ = {
            "root_axis": Axis(),
            "declared_axes": Axis(form="declaration_table"),
        }

    def test_accepts_the_vocabulary_and_expressions_over_it(self):
        check_links(self.Phase("n_enc_frames", {
            "f0_curve": {1: "2*n_enc_frames"},
            "wsum": {0: "600*n_enc_frames+20"},
        }))

    def test_rejects_a_typo_that_would_otherwise_substitute_silently(self):
        # The failure this catches: "n_token" is a perfectly good dict key, so before this link it
        # produced shape expressions over a symbol nothing else in the model uses -- wrong, not
        # malformed, and no downstream gate looks at it.
        with self.assertRaises(LinkError) as cm:
            check_links(self.Phase("n_token"))
        msg = str(cm.exception)
        self.assertIn("root_axis = 'n_token'", msg)
        self.assertIn("['batch', 'n_codes', 'n_enc_frames', 'n_latent', 'n_samples', 'n_tokens']", msg)

    def test_rejects_an_unknown_symbol_inside_a_declared_expression(self):
        with self.assertRaises(LinkError) as cm:
            check_links(self.Phase("n_enc_frames", {"f0_curve": {1: "2*n_frames"}}))
        self.assertIn("declared_axes['f0_curve'][1] = '2*n_frames'", str(cm.exception))
        self.assertIn("['n_frames']", str(cm.exception))

    def test_rejects_an_expression_outside_the_engine_grammar(self):
        # `parse` is exactly `symbol_env.cpp`'s grammar, so a declaration that parses here is one the
        # engine can read back -- which makes this link a grammar check as well as a vocabulary one.
        with self.assertRaises(LinkError) as cm:
            check_links(self.Phase("n_tokens", {"x": {0: "n_tokens % 3"}}))
        self.assertIn("is not a valid shape expression", str(cm.exception))

    def test_needs_no_context_at_all(self):
        # Axis is the one link kind checkable from the declaration alone, so it can never defer.
        self.assertEqual(check_links(self.Phase()), [])


class TestConfigDerived(unittest.TestCase):
    @dataclass
    class Claim:
        family: str = "ctc"
        source: str = "decoder.num_classes_with_blank"
        __links__ = {
            "channels": ConfigDerived(
                claim=lambda spec, ctx: ctx.model["expected"],
                measured=lambda spec, ctx: ctx.model["actual"],
                detail=lambda spec, ctx: ctx.model["shape"],
                message="declares family={spec.family}, whose channel axis must be {claimed} "
                        "(the checkpoint's own {spec.source}), but the model produced {actual} "
                        "({detail}).",
            ),
        }

    def test_passes_when_the_claim_matches(self):
        check_links(self.Claim(), model={"expected": 512, "actual": 512, "shape": (1, 8, 512)})

    def test_message_template_keeps_every_specific(self):
        with self.assertRaises(LinkError) as cm:
            check_links(self.Claim(), model={"expected": 512, "actual": 176, "shape": (1, 8, 176)})
        msg = str(cm.exception)
        # The acceptance criterion of the whole protocol: the generic checker must not degrade a
        # message that names the config field its number came from.
        self.assertEqual(
            msg,
            "declares family=ctc, whose channel axis must be 512 (the checkpoint's own "
            "decoder.num_classes_with_blank), but the model produced 176 ((1, 8, 176)).",
        )

    def test_extra_needs_slots_defer_until_provided(self):
        @dataclass
        class NeedsOutputs:
            __links__ = {
                "arity": ConfigDerived(
                    claim=lambda spec, ctx: 3,
                    measured=lambda spec, ctx: len(ctx.outputs),
                    message="expected {claimed} outputs, got {actual}",
                    needs=("model", "outputs"),
                ),
            }

        checker = LinkChecker()
        checker.check(NeedsOutputs())
        checker.provide(model=object())
        self.assertEqual(len(checker.deferred), 1)
        with self.assertRaises(LinkError) as cm:
            checker.provide(outputs=(1, 2))
        self.assertIn("expected 3 outputs, got 2", str(cm.exception))


class TestWeightAndDriverLinks(unittest.TestCase):
    def test_weight_name_reports_the_merged_dict_size(self):
        @dataclass
        class W:
            weights: list
            __links__ = {"weights": WeightName()}

        check_links(W(["a.weight"]), weights={"a.weight": 1, "b.weight": 2})
        with self.assertRaises(LinkError) as cm:
            check_links(W(["a.weight", "missing"]), weights={"a.weight": 1, "b.weight": 2})
        msg = str(cm.exception)
        self.assertIn("names weight(s) ['missing']", msg)
        self.assertIn("2 tensors", msg)

    def test_driver_symbol_checks_against_a_real_ir_function(self):
        from loom_mil_compiler.driver_ir import Function, Local, Lit

        @dataclass
        class D:
            reads: list
            __links__ = {"reads": DriverSymbol()}

        fn = Function(name="main", params=["inputs"], body=[Local("z", Lit(1))])
        check_links(D(["z", "inputs"]), driver=fn)
        with self.assertRaises(LinkError) as cm:
            check_links(D(["nope"]), driver=fn)
        self.assertIn("reads driver symbol(s) ['nope']", str(cm.exception))


class TestDeferralIsNotAnOptOut(unittest.TestCase):
    """The design detail the plan says must not be skipped, tested directly."""

    def test_finish_raises_naming_what_was_never_checked(self):
        checker = LinkChecker()
        checker.check(_CallSpec("decoder", ["z"]))
        with self.assertRaises(LinkError) as cm:
            checker.finish()
        msg = str(cm.exception)
        self.assertIn("2 declared link(s) were never checked", msg)
        self.assertIn("_CallSpec.topology", msg)
        self.assertIn("needs ['topologies']", msg)
        # The report must say what to do, since the fix is a judgement call between the two options.
        self.assertIn("__unchecked__", msg)

    def test_providing_context_clears_the_deferral(self):
        checker = LinkChecker()
        checker.check(_CallSpec("decoder", ["z", "mu", "t"]))
        self.assertEqual(len(checker.deferred), 2)
        checker.provide(topologies=TOPOLOGIES)
        self.assertEqual(checker.deferred, [])
        checker.finish()

    def test_a_deferred_link_still_fails_when_it_finally_runs(self):
        checker = LinkChecker()
        checker.check(_CallSpec("decoder", ["z", "wrong"]))
        with self.assertRaises(LinkError):
            checker.provide(topologies=TOPOLOGIES)

    def test_one_shot_check_links_is_strict_by_default(self):
        with self.assertRaises(LinkError):
            check_links(_CallSpec("decoder", ["z"]))
        self.assertEqual(len(check_links(_CallSpec("decoder", ["z"]), strict=False)), 2)

    def test_unknown_context_slot_raises_rather_than_being_stored(self):
        with self.assertRaises(LinkError) as cm:
            LinkCheckContext().provide(topolgies={})
        self.assertIn("unknown link-check context slot 'topolgies'", str(cm.exception))


class TestStandingRule(unittest.TestCase):
    def test_undeclared_fields_are_reported(self):
        @dataclass
        class Half:
            topology: str
            note: str = ""
            forgotten: int = 0
            __links__ = {"topology": TopologyName()}
            __unchecked__ = {"note": Unchecked("cosmetic; rendered into a driver comment")}

        self.assertEqual(undeclared_fields(Half), ["forgotten"])

    def test_a_fully_declared_class_has_none(self):
        self.assertEqual(undeclared_fields(_CallSpec), [])

    def test_label_defaults_to_the_class_name_and_is_overridable(self):
        @dataclass
        class Named:
            func_name: str
            __links__ = {}

            def link_label(self):
                return f"FlowMatchingSpec({self.func_name!r})"

        self.assertEqual(spec_label(_CallSpec("decoder", [])), "_CallSpec")
        self.assertEqual(spec_label(Named("sample_decoder")), "FlowMatchingSpec('sample_decoder')")

    def test_declared_links_normalizes_a_single_link_to_a_list(self):
        self.assertEqual(list(declared_links(_CallSpec("decoder", []))), ["topology", "inputs"])
        self.assertEqual(len(declared_links(_CallSpec("decoder", []))["topology"]), 1)


if __name__ == "__main__":
    unittest.main()
