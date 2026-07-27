"""
Locks in `scripted_loop.py`'s finding (EXPORT-IMPROVEMENT.md item 3): a fixed-step iterative-refinement
loop reaches MIL as a genuine `while_loop` when the wrapper is SCRIPTED, and as an unrolled copy-per-step
when it is TRACED. Both directions are asserted, because the whole point is the contrast -- a future
coremltools bump that silently starts unrolling the scripted form again would otherwise go unnoticed.

Also pins the constraint that actually bites: the trip count must be a TorchScript compile-time constant.
A plain `int` attribute does not convert at all, and the error it produces is about `less`/`cast` dtypes,
with nothing pointing at the loop -- so it is asserted here explicitly rather than left as a comment.
"""
import unittest

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loom_mil_compiler  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
import coremltools as ct
from loom_mil_compiler.scripted_loop import RefinementLoop, convert_scripted_loop, count_mil_ops

N_STEPS = 4
WIDTH = 8


class Estimator(nn.Module):
    """Stand-in for a per-step vector-field/denoiser network: one linear, so it is trivially countable
    in the resulting MIL op histogram."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH)

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(state)) * t


class RuntimeTripCount(nn.Module):
    """Same loop, but with the trip count as an ordinary (non-`__constants__`) int attribute -- the
    formulation coremltools cannot convert."""

    n_steps: int

    def __init__(self, step: nn.Module, n_steps: int):
        super().__init__()
        self.step = step
        self.n_steps = n_steps

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(1)
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            state = state + self.step(state, t) * dt
            t = t + dt
        return state


class TestScriptedLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = RefinementLoop(Estimator(), N_STEPS).eval()
        self.state = torch.randn(1, WIDTH)

    def test_traced_loop_is_unrolled(self):
        """The status quo: torch.jit.trace flattens the loop into one copy of the estimator per step."""
        traced = torch.jit.trace(self.module, (self.state,))
        prog = ct.convert(traced,
                          inputs=[ct.TensorType(name="state", shape=(1, WIDTH), dtype=np.float32)],
                          convert_to="milinternal", compute_precision=ct.precision.FLOAT32)
        hist = count_mil_ops(prog)
        self.assertNotIn("while_loop", hist)
        self.assertEqual(hist.get("linear", 0), N_STEPS)

    def test_scripted_loop_becomes_a_mil_while_loop(self):
        """The item 3 result: scripting keeps one estimator, wrapped in a real MIL while_loop."""
        prog = convert_scripted_loop(self.module, (1, WIDTH))
        hist = count_mil_ops(prog)
        self.assertEqual(hist.get("while_loop", 0), 1)
        self.assertEqual(hist.get("linear", 0), 1, "estimator should appear once, inside the loop body")

    def test_scripted_loop_is_numerically_equivalent_to_eager(self):
        """Capturing the loop must not change what it computes."""
        with torch.no_grad():
            expected = self.module(self.state)
        traced = torch.jit.trace(self.module, (self.state,))
        with torch.no_grad():
            self.assertTrue(torch.allclose(traced(self.state), expected, atol=1e-6))
        scripted = torch.jit.script(self.module)
        with torch.no_grad():
            self.assertTrue(torch.allclose(scripted(self.state), expected, atol=1e-6))

    def test_runtime_trip_count_is_not_convertible(self):
        """Constraint 1 in scripted_loop.py's docstring: a non-constant trip count fails conversion.
        If a future coremltools makes this work, this test fails and the docstring should be relaxed."""
        module = RuntimeTripCount(Estimator(), N_STEPS).eval()
        with self.assertRaises(Exception):
            convert_scripted_loop(module, (1, WIDTH))


if __name__ == "__main__":
    unittest.main()
