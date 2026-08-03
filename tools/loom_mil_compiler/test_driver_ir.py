"""Unit tests for driver_ir.py's check_subgraph_calls, independent of coremltools/torch -- the module is
pure Python and only reads plain dict topologies, so these exercise it directly with hand-built
SubgraphCall/Function IR rather than tracing a real model."""
import unittest

from driver_ir import (
    ArrayLit, Call, CallStmt, DriverIRError, Function, Lit, Local, LuaCodegen, SubgraphCall, Var,
    check_subgraph_calls, validate,
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
