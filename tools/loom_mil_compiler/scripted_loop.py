"""Capturing a source model's iterative-refinement loop as a real MIL `while_loop` instead of letting
`torch.jit.trace` unroll it -- EXPORT-IMPROVEMENT.md item 3.

`torch.jit.trace` records one concrete execution, so a plain Python `for` over N refinement steps
reaches MIL as N copies of the step network with no loop structure at all. `torch.jit.script` keeps the
loop as a TorchScript `prim::Loop`, which coremltools' torch frontend does lower to a genuine MIL
`while_loop`. `RefinementLoop` below is that, packaged: wrap the per-step estimator, script the wrapper,
convert.

Three constraints were established empirically against coremltools 9.0 / torch 2.8, and they are the
reason this module exists rather than a one-line note saying "just script it":

1. **The trip count must be a TorchScript compile-time constant.** `__constants__` (what
   `RefinementLoop` uses) or a source literal both work. A plain `n: int` attribute, or a `while i < n`
   with an `int` counter, does NOT: coremltools' frontend mis-types the loop bound as `str` and the
   conversion dies inside its own `less`/`cast` lowering with a dtype-mismatch error that says nothing
   about loops. So the step count is bakeable per export, but a genuinely *runtime*-determined trip
   count -- VITS's and SupertonicTTS's duration-driven loops -- is not reachable this way today.
2. **Loop-carried state must be tensors.** A Python float accumulator (`t = t + dt`) becomes a
   TorchScript `float` loop carry; keep it a 1-element tensor so it stays a real MIL loop variable.
3. **Scripting is all-or-nothing over the wrapper.** Everything the scripted `forward` touches must
   itself be scriptable, so the step network has to be script-compatible (no `*args`, no data-dependent
   Python control flow it can't type). This is the practical cost of the approach and the reason it is
   not applied to any shipping export here.

**No shipping export uses this yet, deliberately.** `generate_graph_topology` produces a *static* node
list and so cannot contain a loop at all; a `while_loop` is only consumable on the driver-IR path
(`exporter.transpile_operation` lowers `while_loop` to a Lua `While`/`Break`), which transpiles the loop
body op-by-op into host Lua -- fine for scalar bookkeeping, not for a tensor estimator network. Making a
`while_loop` body become its own callable topology is real unimplemented work. Until then the host-side
loop (`iterative_export.py`) is the better shape anyway: it supports a runtime step count, which
constraint 1 above says this path cannot. See BACKEND.md for the full comparison.
"""
import numpy as np
import torch
import torch.nn as nn


class RefinementLoop(nn.Module):
    """Runs `step(state, t)` for a fixed number of iterations with forward-Euler-style loop-carried
    state, in a form `torch.jit.script` preserves as a `prim::Loop`.

    `step` must accept `(state, t)` and return the state's update direction, with both `state` and `t`
    tensors. `n_steps` is listed in `__constants__` so TorchScript inlines it as a real integer literal
    -- see this module's docstring, constraint 1.
    """

    __constants__ = ["n_steps"]

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


def count_mil_ops(program) -> dict:
    """Histogram of op types across a MIL Program's main function *and every nested block*, so a loop
    body's own ops are counted rather than hidden behind the `while_loop` that owns them."""
    hist = {}

    def walk(block):
        for op in block.operations:
            hist[op.op_type] = hist.get(op.op_type, 0) + 1
            for sub in getattr(op, "blocks", None) or []:
                walk(sub)

    walk(program.functions["main"])
    return hist


def convert_scripted_loop(module: nn.Module, state_shape, input_name: str = "state"):
    """Scripts `module` (rather than tracing it) and converts to a MIL Program, so any loop in its
    `forward` survives as a `while_loop`. Returns the Program."""
    import coremltools as ct

    scripted = torch.jit.script(module.eval())
    return ct.convert(
        scripted,
        inputs=[ct.TensorType(name=input_name, shape=tuple(state_shape), dtype=np.float32)],
        convert_to="milinternal",
        compute_precision=ct.precision.FLOAT32,
    )
