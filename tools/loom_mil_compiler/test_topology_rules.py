"""
Structural checks on `topology_ops.py`'s rewrite table (EXPORT-IMPROVEMENT.md item 1). These are the
invariants that stop being obvious once dispatch is a table instead of a top-to-bottom `if` chain:

* an unguarded rule is its op type's catch-all, so any rule registered *after* one can never fire;
* every rule that claims an op type must still be reachable for some input;
* the guarded families (`matmul`, `gelu`, `less`) must select the composition their `when` text claims.

The `less` case additionally pins the fall-through that a mechanical extraction of the old `if` chain
got wrong: a `less` that is NOT the length-validity mask idiom must match *no* rule at all, so the
generic OP_MAP path emits the real comparison. That regression (three dropped `LESS` nodes in Kokoro's
decoder_vocoder topology) was invisible to every existing test.
"""
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loom_mil_compiler  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_mil_compiler.topology_ops import _RULES, describe_topology_rules, lookup_topology_rule


class _FakeVar:
    """Enough of a MIL Var for a guard to read: just a static `.val`."""

    def __init__(self, val):
        self.val = val


class _FakeOp:
    """Enough of a MIL Operation for `lookup_topology_rule` and the static-attribute guards."""

    def __init__(self, op_type, **inputs):
        self.op_type = op_type
        self.name = f"fake_{op_type}"
        self.inputs = {k: _FakeVar(v) for k, v in inputs.items()}


class TestTopologyRuleTable(unittest.TestCase):
    def test_no_rule_is_shadowed_by_an_earlier_catch_all(self):
        """An unguarded rule swallows every remaining instance of its op type, so nothing may follow
        it. This is the one ordering constraint the table has, and it is silent when violated."""
        for op_type, rules in _RULES.items():
            for i, rule in enumerate(rules[:-1]):
                self.assertIsNotNone(
                    rule.guard,
                    f"{op_type!r}: unguarded rule {rule.name} at position {i} makes "
                    f"{[r.name for r in rules[i + 1:]]} unreachable",
                )

    def test_every_op_type_maps_to_at_least_one_rule(self):
        for op_type, rules in _RULES.items():
            self.assertTrue(rules, f"{op_type!r} registered with no rules")

    def test_describe_lists_every_registration(self):
        text = describe_topology_rules()
        for op_type, rules in _RULES.items():
            for rule in rules:
                self.assertIn(rule.name, text)
                self.assertIn(op_type, text)

    def test_matmul_transpose_guards_select_the_right_composition(self):
        cases = {
            (False, True): "_op_matmul_x_yt",
            (False, False): "_op_matmul_x_y",
            (True, False): "_op_matmul_unsupported",
            (True, True): "_op_matmul_unsupported",
        }
        for (tx, ty), expected in cases.items():
            op = _FakeOp("matmul", transpose_x=tx, transpose_y=ty)
            self.assertEqual(lookup_topology_rule(None, op).name, expected, f"tx={tx} ty={ty}")

    def test_matmul_defaults_to_no_transpose_when_the_inputs_are_absent(self):
        op = _FakeOp("matmul")
        self.assertEqual(lookup_topology_rule(None, op).name, "_op_matmul_x_y")

    def test_gelu_mode_guards_select_the_right_composition(self):
        cases = {
            "EXACT": "_op_gelu_exact",
            "NONE": "_op_gelu_exact",
            "TANH_APPROXIMATION": "_op_gelu_tanh_approx",
            "TANH": "_op_gelu_tanh_approx",
            "SOMETHING_NEW": "_op_gelu_unsupported",
        }
        for mode, expected in cases.items():
            self.assertEqual(lookup_topology_rule(None, _FakeOp("gelu", mode=mode)).name, expected, mode)
        # An absent mode input is PyTorch's own default: exact erf.
        self.assertEqual(lookup_topology_rule(None, _FakeOp("gelu")).name, "_op_gelu_exact")

    def test_reduce_mean_routes_by_axis_and_whether_its_count_is_static(self):
        """The three-way split that the StyleTTS2 regression was hiding in. The middle row is the one
        that broke: a dynamic count on ne[0] must reach *no* rule, so the generic MEAN lowering runs and
        supplies its own run-time count."""

        class _Exporter:
            """Reports whatever ne-order shape the case under test needs."""

            def __init__(self, shape):
                self.shape = shape

            def get_var_info(self, var):
                return {"shape": self.shape}

        # ne-order shape, torch axis -> expected rule name (None means "falls through")
        cases = [
            # Matcha's hand-rolled LayerNorm: reduces ne[1], statically sized.
            (["n_tokens", "192", "1"], 1, "_op_reduce_mean_scaled_sum"),
            # StyleTTS2's diffusion: reduces ne[0] over a live n_tokens-derived length.
            (["(floor((n_tokens) * (512)))", "1024", "1"], -1, None),
            # Statically-sized ne[0] (STFT/CMVN): still the composed path.
            (["512", "n_tokens", "1"], -1, "_op_reduce_mean_scaled_sum"),
            # Dynamic count on an axis ggml_mean cannot reduce: genuinely unrepresentable.
            (["512", "n_tokens", "1"], 1, "_op_reduce_mean_unsupported"),
        ]
        for shape, torch_axis, expected in cases:
            op = _FakeOp("reduce_mean", axes=[torch_axis], keep_dims=False)
            op.inputs["x"] = _FakeVar(None)
            rule = lookup_topology_rule(_Exporter(shape), op)
            got = None if rule is None else rule.name
            self.assertEqual(got, expected, f"shape={shape} torch_axis={torch_axis}")

    def test_multi_axis_reduce_mean_is_rejected_not_silently_dropped(self):
        class _Exporter:
            def get_var_info(self, var):
                return {"shape": ["4", "8", "1"]}

        op = _FakeOp("reduce_mean", axes=[0, 1], keep_dims=False)
        op.inputs["x"] = _FakeVar(None)
        rule = lookup_topology_rule(_Exporter(), op)
        self.assertIsNotNone(rule, "a multi-axis reduce_mean must not fall through to ne[0]-only MEAN")
        self.assertEqual(rule.name, "_op_reduce_mean_unsupported")

    def test_an_ordinary_less_falls_through_to_the_generic_path(self):
        """`less` has exactly one rule, guarded on the length-validity-mask idiom. A comparison that
        isn't that idiom must match nothing, so the generic OP_MAP `LESS` lowering runs instead."""
        self.assertEqual(len(_RULES["less"]), 1)
        self.assertIsNotNone(_RULES["less"][0].guard)

        class _NotAMask:
            """An exporter whose length/range probes all decline -- i.e. an ordinary comparison."""

            def _traces_to_length_input(self, var, *a, **k):
                return False

            def _find_range_1d_var(self, var, *a, **k):
                return None

        op = _FakeOp("less", x=1.0, y=2.0)
        self.assertIsNone(lookup_topology_rule(_NotAMask(), op))


if __name__ == "__main__":
    unittest.main()
