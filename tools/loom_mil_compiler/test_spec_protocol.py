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
    check_links, dangling_coverage, declared_links, spec_label, undeclared_fields,
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
        self.assertIn("['batch', 'n_codes', 'n_enc_frames', 'n_kv', 'n_latent', 'n_samples', 'n_tokens']", msg)

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


# -- B.6: the standing rule, enforced across the whole package ---------------------------------------
#
# "Every spec field must be either checkable against the real model/topology, or explicitly documented
# as unchecked." Up to here that is a convention each retrofit happened to follow. What makes it a
# protocol is this: the rule is enforced by DISCOVERY, not by a list of classes someone remembers to
# extend. A new spec class in a family module fails until it declares, and a new field on an existing
# one fails the same way.
#
# It is cheap now and expensive later, which is why the plan puts it in stage B rather than after
# families 2/6/10/11 exist.

# Modules whose dataclasses are infrastructure rather than specs: they describe no model and there is
# nothing real to check them against. Exempted per module, with the reason, rather than per class --
# adding an IR node or a link kind should not require touching this test.
_INFRASTRUCTURE_MODULES = {
    "driver_ir": "the Lua IR's own expression/statement node types -- a syntax tree, not a declaration "
                 "about any model",
    "spec_protocol": "this protocol's own vocabulary (the link kinds, LinkSite, FieldRef, Unchecked, "
                     "CoveredBy, NestedSpec). Declaring links on the link kinds is not a fixed point "
                     "worth having",
    "registry": "ModelRecognizer/TaskRegistryEntry describe how a checkpoint is RECOGNIZED, which is "
                "checked by detect() running against real checkpoints, not by a link",
    "tasks": "TaskSpec is the canonical vocabulary itself; TaskRegistry.register validates against it",
    "checkpoint_probe": "a pickle-opcode probe's result. Read FROM a real checkpoint rather than "
                        "claimed about one -- the direction a link runs in is the other way",
}

# Individual classes that are results or internals rather than declarations.
_NOT_SPECS = {
    "modular_export.ModularExportResult": "an export RESULT -- what export_modular produced, not what "
                                          "a caller declared. Its fields are outputs; checking them "
                                          "against the model they were derived FROM is circular",
    "modular_export._LeafPath": "an internal bookkeeping record for one captured tensor argument, "
                                "built and consumed inside _flatten_call/_replay",
    "driver_builder.DriverContext": "the real topologies/axes/weights a component emits against -- the "
                                    "same category as spec_protocol's own LinkCheckContext. It is what "
                                    "links are checked AGAINST, so checking it against something else "
                                    "has no second authority to appeal to",
    "driver_builder.DriverScript": "a build RESULT (the emitted prelude chunks and entry function), "
                                   "checked by driver_ir.validate/check_subgraph_calls at the moment "
                                   "DriverBuilder.build produces it rather than by a declaration",
    "flow_matching_export._TextDriver": "a driver that is still text, standing in for a DriverScript "
                                        "so DriverSymbol can read it. It is what links are checked "
                                        "AGAINST, in the same category as DriverScript itself",
}

# Modules the scan is allowed to fail to import. Both are standalone scripts that re-run `dialect.py`'s
# op registration at import time, so whether importing them raises "op already registered" depends on
# what else the process imported first -- they load cleanly under pytest and not from a bare script.
# The allowance is therefore a subset check, not an equality one: whether these two appear varies, but
# a THIRD unimportable module must fail, because any spec class inside it escapes the scan entirely.
_MAY_NOT_IMPORT = {"compare_snapshots", "export_lstm_test_fixture"}


def _package_dataclasses():
    """Every dataclass defined in `loom_mil_compiler`, by `module.ClassName`, plus the modules that
    could not be imported."""
    import dataclasses
    import importlib
    import inspect
    import pkgutil

    import loom_mil_compiler as package

    found, unimportable = {}, set()
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("test_"):
            continue
        try:
            module = importlib.import_module(f"loom_mil_compiler.{module_info.name}")
        except Exception:
            unimportable.add(module_info.name)
            continue
        for obj in vars(module).values():
            if (inspect.isclass(obj) and dataclasses.is_dataclass(obj)
                    and obj.__module__ == module.__name__):
                found.setdefault(f"{module_info.name}.{obj.__name__}", obj)
    return found, unimportable


class TestStandingRuleAcrossThePackage(unittest.TestCase):
    def test_every_spec_class_declares_every_field(self):
        found, _ = _package_dataclasses()
        offenders = {}
        for qualname, cls in sorted(found.items()):
            module_name = qualname.split(".", 1)[0]
            if module_name in _INFRASTRUCTURE_MODULES or qualname in _NOT_SPECS:
                continue
            missing = undeclared_fields(cls)
            if missing:
                offenders[qualname] = missing
        self.assertEqual(offenders, {}, (
            "these fields are neither link-declared nor explicitly __unchecked__. A field with no "
            "declaration reads as validated and is not -- add a link, a CoveredBy, or an Unchecked "
            "with the reason; if the class is not a spec at all, add it to _NOT_SPECS here"
        ))

    def test_no_coverage_claim_points_at_a_link_that_does_not_exist(self):
        """`CoveredBy` is the one declaration that can rot silently: deleting the link a field defers
        to would turn its declaration into an exemption."""
        found, _ = _package_dataclasses()
        dangling = {q: dangling_coverage(c) for q, c in found.items() if dangling_coverage(c)}
        self.assertEqual(dangling, {})

    def test_the_scan_reaches_the_classes_it_is_supposed_to_reach(self):
        """The rule is only as good as the discovery. If a refactor moved these, the test above would
        pass vacuously -- so name the ones that must be found."""
        found, _ = _package_dataclasses()
        for qualname in ("flow_matching_export.FlowMatchingSpec", "modular_export.ModularExportSpec",
                         "multi_phase_export.ExportPhase", "nemo_asr_export.ASRNemoEncoderExportConfig",
                         "causal_lm_export.LMCausalModelExportConfig", "decomposition.Modular",
                         "kokoro_export.TTSKokoroExportConfig", "vits_export.TTSVitsExportConfig",
                         "styletts2_export.TTSStyleTTS2ExportConfig",
                         "matcha_export.TTSMatchaExportConfig",
                         "supertonic_export.TTSSupertonicExportConfig"):
            self.assertIn(qualname, found)

    def test_no_unexpected_module_escapes_the_scan_by_failing_to_import(self):
        _, unimportable = _package_dataclasses()
        self.assertEqual(unimportable - _MAY_NOT_IMPORT, set(), (
            "a module the scan could not import -- any spec class in it escapes the standing rule "
            "silently, which is the one way this test can pass vacuously"
        ))

    def test_every_registered_config_class_is_covered(self):
        """The other direction: driven by what is actually registered, so a family that registers a
        config class the scan somehow missed still fails here."""
        from loom_mil_compiler.registry import default_registry

        for task, entry in default_registry()._entries.items():
            self.assertEqual(undeclared_fields(entry.config_class), [], f"{task}/{entry.config_class}")

    def test_an_exemption_reason_is_required_rather_than_a_bare_name(self):
        """Both exemption tables map to prose, and the prose is the point -- "not a spec" and "nobody
        got around to it" are different statements and only one should survive review."""
        for table in (_INFRASTRUCTURE_MODULES, _NOT_SPECS):
            for name, reason in table.items():
                self.assertGreater(len(reason), 40, name)


if __name__ == "__main__":
    unittest.main()
