"""Unit tests for driver_ir.py's check_subgraph_calls, independent of coremltools/torch -- the module is
pure Python and only reads plain dict topologies, so these exercise it directly with hand-built
SubgraphCall/Function IR rather than tracing a real model."""
import unittest

from driver_ir import (
    ArrayLit, BinOp, Call, CallStmt, DriverIRError, Function, If, Len, Lit, Local, LuaCodegen,
    OutputRef, RetainedArgmax, Return, SubgraphCall, Var, While, check_subgraph_calls, validate,
)


def _topo(inputs=(), outputs=None, output=None):
    topo = {"version": 1, "inputs": [{"name": n} for n in inputs], "nodes": []}
    if outputs is not None:
        topo["outputs"] = list(outputs)
    elif output is not None:
        topo["output"] = output
    return topo


class TestCheckSubgraphCalls(unittest.TestCase):
    def test_single_output_data_and_shape_ok(self):
        """The pre-P2 shape every model still uses: one data local + one shape local (extra_outputs)
        against a single-output topology."""
        call = SubgraphCall(outputs=["out"], extra_outputs=["shape"], module="m",
                             axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        check_subgraph_calls(fn, {"m": _topo(output="logits")})  # must not raise

    def test_multi_output_data_only_ok(self):
        """A caller only wanting the data of both of a 2-output topology's outputs (no shapes) --
        Lua's own "capture fewer than every return value" semantics."""
        call = SubgraphCall(outputs=["a", "b"], module="m", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        check_subgraph_calls(fn, {"m": _topo(outputs=["y", "z"])})  # must not raise

    def test_multi_output_partial_data_capture_ok(self):
        """Capturing fewer data outputs than declared (here 1 of 2) is legal -- Lua discards the rest."""
        call = SubgraphCall(outputs=["a"], module="m", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        check_subgraph_calls(fn, {"m": _topo(outputs=["y", "z"])})  # must not raise

    def test_multi_output_full_data_plus_shapes_ok(self):
        call = SubgraphCall(outputs=["a", "b"], extra_outputs=["sa", "sb"], module="m", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        check_subgraph_calls(fn, {"m": _topo(outputs=["y", "z"])})  # must not raise

    def test_too_many_data_outputs_raises(self):
        call = SubgraphCall(outputs=["a", "b", "c"], module="m", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        with self.assertRaises(DriverIRError) as ctx:
            check_subgraph_calls(fn, {"m": _topo(outputs=["y", "z"])})
        self.assertIn("captures 3 data output(s)", str(ctx.exception))

    def test_shapes_without_full_data_capture_raises(self):
        """Requesting a shape while only having captured 1 of 2 data outputs would silently bind a
        SHAPE local to the second output's DATA instead (run_subgraph returns all data before any
        shape) -- must be rejected at export time, not left to a mismatched-looking Lua array at
        runtime."""
        call = SubgraphCall(outputs=["a"], extra_outputs=["sa"], module="m", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        with self.assertRaises(DriverIRError) as ctx:
            check_subgraph_calls(fn, {"m": _topo(outputs=["y", "z"])})
        self.assertIn("requires capturing every data output first", str(ctx.exception))

    def test_undeclared_input_still_raises(self):
        """Pre-existing input validation must still work unchanged alongside the new output checks."""
        call = SubgraphCall(outputs=["a"], module="m", axes={}, inputs={"bogus": Var("x")})
        fn = Function(name="main", params=["inputs", "x"], body=[call])
        with self.assertRaises(DriverIRError) as ctx:
            check_subgraph_calls(fn, {"m": _topo(inputs=["real"], outputs=["y"])})
        self.assertIn("undeclared input(s)", str(ctx.exception))

    def test_unregistered_topology_skipped(self):
        call = SubgraphCall(outputs=["a", "b", "c"], module="not_in_dict", axes={}, inputs={})
        fn = Function(name="main", params=["inputs"], body=[call])
        check_subgraph_calls(fn, {})  # must not raise -- topology absent, skipped entirely


def _retain(module, inputs=None):
    return SubgraphCall(outputs=[], module=module, axes={}, inputs=inputs or {}, retain=True)


class TestRetainedOutputAdjacency(unittest.TestCase):
    """The static half of the staleness guard (BACKLOG.md P4.0.12).

    An `OutputRef` names a MODULE, so `validate()` -- which knows only about symbols -- reports nothing
    about it either way. These are the checks that replace what it cannot do, and the reason they live
    in `check_subgraph_calls` rather than beside it: only this checker has the topologies.
    """

    def _topos(self):
        return {
            "a": _topo(outputs=["cos", "sin"]),
            "b": _topo(inputs=["x"], output="y"),
        }

    def test_a_reference_to_an_earlier_retained_run_is_fine(self):
        fn = Function("infer", ["inputs"], [
            _retain("a"),
            SubgraphCall(outputs=["out"], module="b", axes={}, inputs={"x": OutputRef("a")}),
        ])
        check_subgraph_calls(fn, self._topos())  # must not raise

    def test_a_reference_to_a_module_nothing_retained_raises(self):
        """The failure retention introduces that marshalling could not have: the read succeeds at
        runtime and returns whatever the module last produced -- or nothing at all."""
        fn = Function("infer", ["inputs"], [
            SubgraphCall(outputs=["out"], module="b", axes={}, inputs={"x": OutputRef("a")}),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("no earlier loom.run_subgraph_and_retain('a', ...)", str(raised.exception))

    def test_a_reference_produced_only_inside_a_branch_is_rejected(self):
        """Conservative on purpose: whatever a nested block retains does not escape it, so a producer
        that only runs on one arm is not assumed to have run. A driver that genuinely needs this pins
        the generation number at run time instead."""
        fn = Function("infer", ["inputs"], [
            If(cond=Lit(True), then=[_retain("a")]),
            SubgraphCall(outputs=["out"], module="b", axes={}, inputs={"x": OutputRef("a")}),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("straight-line block", str(raised.exception))

    def test_a_reference_inside_a_loop_sees_the_enclosing_producer(self):
        """The other direction: a retained run BEFORE the loop is visible inside it, which is the shape
        a decode loop reading a prefill's output would have."""
        fn = Function("infer", ["inputs"], [
            _retain("a"),
            While(cond=Lit(True), body=[
                SubgraphCall(outputs=["out"], module="b", axes={}, inputs={"x": OutputRef("a")}),
            ]),
        ])
        check_subgraph_calls(fn, self._topos())  # must not raise

    def test_an_index_past_the_declared_outputs_raises(self):
        fn = Function("infer", ["inputs"], [
            _retain("a"),
            SubgraphCall(outputs=["out"], module="b", axes={}, inputs={"x": OutputRef("a", index=3)}),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("which declares 2 output(s)", str(raised.exception))

    def test_a_retaining_call_that_also_binds_a_local_raises(self):
        """The two halves of one decision -- `retain` here and the consumer's `OutputRef` -- disagreeing
        would otherwise show up as a local silently bound to the generation number."""
        fn = Function("infer", ["inputs"], [
            SubgraphCall(outputs=["oops"], module="a", axes={}, inputs={}, retain=True),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("read back by module name", str(raised.exception))

    def test_an_output_reference_renders_as_a_self_describing_table(self):
        self.assertEqual(OutputRef("prefix").render(), "{from = 'prefix'}")
        self.assertEqual(OutputRef("aux", index=2).render(), "{from = 'aux', index = 2}")
        self.assertEqual(OutputRef("aux").reads(), [])

    def test_a_retaining_call_renders_as_a_statement_binding_nothing(self):
        stmt = _retain("prefix", {"x": OutputRef("tok")})
        self.assertEqual(
            LuaCodegen()._emit_stmt(stmt, 1),
            ["    loom.run_subgraph_and_retain('prefix', {}, {x = {from = 'tok'}})"],
        )
        self.assertEqual(stmt.defines(), [])


class TestRetainedArgmax(unittest.TestCase):
    """The other reference by module name (BACKLOG.md P4.0.14) -- the epilogue's reduction rather than
    an inter-module edge. Same blind spot in `validate()`, so the same checker has to cover it."""

    def _topos(self):
        return {"a": _topo(inputs=["tok"], output="logits")}

    def test_it_renders_as_the_module_form_of_argmax_row(self):
        expr = RetainedArgmax("a", BinOp("-", Len("tokens"), Lit(1)))
        self.assertEqual(expr.render(), "loom.argmax_row('a', (#tokens - 1))")

    def test_it_reads_the_row_expression_and_not_the_module(self):
        """A module is not a symbol: reporting one as read would make `validate()` demand a local of
        that name, and the row expression is the only thing here an earlier statement really binds."""
        self.assertEqual(RetainedArgmax("a", Len("tokens")).reads(), ["tokens"])

    def test_reducing_a_module_that_retained_is_fine(self):
        fn = Function("infer", ["inputs"], [
            _retain("a", {"tok": Var("inputs")}),
            Return([RetainedArgmax("a", Lit(0))]),
        ])
        check_subgraph_calls(fn, self._topos())  # must not raise

    def test_reducing_a_module_nothing_retained_raises(self):
        """The half of P4.0.14's decision the exporter could get wrong on its own: an epilogue naming a
        module whose producing call still marshals. At runtime the bridge raises "has no retained
        outputs"; this is the same error, at export time."""
        fn = Function("infer", ["inputs"], [
            SubgraphCall(outputs=["out"], module="a", axes={}, inputs={"tok": Var("inputs")}),
            Return([RetainedArgmax("a", Lit(0))]),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("loom.argmax_row('a', ...)", str(raised.exception))

    def test_a_reduction_inside_a_loop_sees_the_producer_in_the_same_body(self):
        """`infer_with_past`'s exact shape: retain and reduce are two statements in one loop body, and
        the producer is the statement right before the read on every iteration."""
        fn = Function("infer", ["inputs"], [
            While(cond=Lit(True), body=[
                _retain("a", {"tok": Var("inputs")}),
                Local("next", RetainedArgmax("a", Lit(0))),
            ]),
        ])
        check_subgraph_calls(fn, self._topos())  # must not raise

    def test_a_producer_only_inside_a_branch_does_not_escape_it(self):
        """Same conservatism `OutputRef` gets, and for the same reason -- this is one walk, not two."""
        fn = Function("infer", ["inputs"], [
            If(cond=Lit(True), then=[_retain("a", {"tok": Var("inputs")})]),
            Return([RetainedArgmax("a", Lit(0))]),
        ])
        with self.assertRaises(DriverIRError) as raised:
            check_subgraph_calls(fn, self._topos())
        self.assertIn("straight-line block", str(raised.exception))


if __name__ == "__main__":
    unittest.main()


class TestArrayLitAndCallStmt(unittest.TestCase):
    """The two nodes the decode loop needed (KV-CACHE.md 3.3). Both exist rather than being spelled
    with `RawExpr`/`RawBlock` for the same reason: those report no reads, so a symbol referenced inside
    one is invisible to `validate()` -- and both of these carry symbols an earlier statement bound."""

    def test_an_array_literal_renders_and_reports_its_reads(self):
        expr = ArrayLit([Var("a"), Lit(1)])
        self.assertEqual(expr.render(), "{a, 1}")
        self.assertEqual(expr.reads(), ["a"])

    def test_an_empty_array_literal(self):
        self.assertEqual(ArrayLit([]).render(), "{}")

    def test_a_call_statement_renders_with_no_binding(self):
        stmt = CallStmt(Call("table.insert", [Var("t"), Var("v")]))
        self.assertEqual(LuaCodegen()._emit_stmt(stmt, 1), ["    table.insert(t, v)"])
        self.assertEqual(stmt.defines(), [])

    def test_validate_sees_through_both(self):
        """The point of having them as nodes: an undefined symbol inside either is caught."""
        fn = Function("infer", ["inputs"], [CallStmt(Call("table.insert", [Var("gen"), Var("x")]))])
        with self.assertRaises(DriverIRError) as raised:
            validate(fn)
        self.assertIn("'gen'", str(raised.exception))

        fn = Function("infer", ["inputs"], [Local("gen", ArrayLit([Var("missing")]))])
        with self.assertRaises(DriverIRError) as raised:
            validate(fn)
        self.assertIn("'missing'", str(raised.exception))
