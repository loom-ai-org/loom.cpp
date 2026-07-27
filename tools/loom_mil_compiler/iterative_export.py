"""Generalizes the "N-step iterative refinement over loop-carried tensor state" driver family --
EXPORT-IMPROVEMENT.md item 4, the second family template alongside `submodule_export.py`'s
`SubmoduleExportSpec`.

Matcha-TTS and SupertonicTTS are not two unrelated problems: both trace a per-step *estimator* network
into one topology and then integrate it host-side with forward Euler over a loop-carried state tensor,
and both hand-wrote the identical Lua loop to do it:

    local z = loom.gaussian_array(<n_elems>)
    local dt = 1.0 / inputs.n_steps
    for step = 0, inputs.n_steps - 1 do
        local t = step / inputs.n_steps
        local v = loom.run_subgraph("<estimator>", <length>, 0, { <carried> = z, ...fixed..., t = { t } })
        for i = 1, #z do z[i] = z[i] + v[i] * dt end
    end

An `IterativeRefinementSpec` declares the six things that actually differ between them -- the estimator
topology's name, which of its inputs carries the state, which carries the scalar time, which are fixed
across steps, how many elements the state has, and what `n_tokens` to build the estimator's graph at --
and `render_sampler` emits that loop as a named Lua function the hand-written driver calls in one line.

The point is not the line count. It is that the spec is **checked against the real traced topology** at
export time (`validate_against_topology`): naming an input the estimator doesn't declare, or forgetting
one it does, raises immediately with the actual declared-input list. A hand-written Lua loop with the
same mistake fails much later, inside the engine, as an undeclared-input or missing-input error with no
connection back to the line that got it wrong. This mirrors `SubmoduleExportSpec`'s own "a wrong
attribute path raises `AttributeError` immediately, not a silent wrong export" property.

Deliberately NOT generalized here: the *integration rule*. Both current models use plain deterministic
forward Euler with uniform `dt = 1/n_steps`; StyleTTS2's diffusion sampler is ADPM2 over a Karras sigma
schedule (second-order, two network evaluations per step, per-step noise injection, and real
preconditioning math around the call), and Kokoro/VITS's duration loops are a different shape entirely
(a scatter over predicted durations, not an ODE). Those stay bespoke, the same way
`submodule_export.py` concedes non-causal-LM architectures.

The *validation*, though, generalizes further than the codegen does -- a hand-written `run_subgraph`
call fails the same way a generated one would. So `EstimatorSpec` (below) is its own declaration: a
bespoke sampler can still declare which topology it calls with which inputs and get the export-time
check, generating nothing. See BACKEND.md for why the loop itself stays host-side rather than becoming
a MIL `while_loop`.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EstimatorSpec:
    """One traced per-step network a driver calls, plus the exact set of topology inputs it is handed.

    Separate from `IterativeRefinementSpec` because the *validation* generalizes further than the
    *codegen* does. StyleTTS2's diffusion sampler is ADPM2 over a Karras sigma schedule -- two network
    evaluations per step, per-step noise injection, and real preconditioning math
    (`c_skip`/`c_out`/`c_in`/`c_noise`) wrapped around the call -- so no template can emit its loop
    without becoming a worse thing to read than the loop. But its `run_subgraph` call has the identical
    failure mode as every other one: a name that does not match the topology's declared inputs is only
    caught deep inside the engine at run time. Declaring the call alone still buys that check.
    """

    topology: str
    # Topology input names the driver supplies, in argument-table order.
    inputs: list

    def validate_against_topology(self, topology: dict, label: Optional[str] = None):
        """Cross-checks against the topology's real declared inputs, raising ValueError naming the exact
        mismatch. This is the whole reason these specs exist rather than a hand-written call."""
        declared = [inp["name"] for inp in topology.get("inputs", [])]
        missing = [n for n in self.inputs if n not in declared]
        unsupplied = [n for n in declared if n not in self.inputs]
        if missing or unsupplied:
            raise ValueError(
                f"{label or 'EstimatorSpec'} does not match topology {self.topology!r}: "
                + (f"supplies input(s) it does not declare: {missing}; " if missing else "")
                + (f"leaves declared input(s) unsupplied: {unsupplied}; " if unsupplied else "")
                + f"topology declares {declared}, spec supplies {self.inputs}."
            )


@dataclass
class IterativeRefinementSpec:
    """Declares one "sample by integrating an estimator network N times" driver phase.

    `func_name` is the Lua function `render_sampler` emits; the driver calls it as

        local z = <func_name>(<length>, <n_elems>, <n_steps>, { <fixed input> = <value>, ... })

    where `length` is the `n_tokens` the estimator's graph is built at and `n_elems` is the element
    count of the loop-carried state (which `loom.gaussian_array` samples).
    """

    func_name: str
    # The traced per-step estimator's topology name, as registered in the exporter's `topologies` dict.
    estimator: str
    # The estimator input that receives the loop-carried state (Matcha "z", Supertonic "z_t").
    carried_input: str
    # The estimator input that receives the current scalar time, passed as a 1-element Lua array.
    time_input: str = "t"
    # Estimator inputs that are constant across every step, supplied by the caller's table. Order is
    # only cosmetic (it fixes the order they're written into the per-step argument table).
    fixed_inputs: list = field(default_factory=list)
    # Human-readable note rendered above the generated function, so the embedded driver script still
    # explains itself to someone reading the GGUF rather than this file.
    note: Optional[str] = None

    def estimator_spec(self) -> EstimatorSpec:
        """This spec's per-step call, as the plain declaration a bespoke sampler would write by hand --
        so both share one validation implementation and cannot drift apart."""
        return EstimatorSpec(topology=self.estimator,
                              inputs=[self.carried_input, *self.fixed_inputs, self.time_input])

    def validate_against_topology(self, topology: dict):
        self.estimator_spec().validate_against_topology(
            topology, label=f"IterativeRefinementSpec({self.func_name!r})")


def render_sampler(spec: IterativeRefinementSpec) -> str:
    """The Lua source for `spec`'s sampler function.

    Emits deterministic forward-Euler integration: `z_{k+1} = z_k + v(z_k, t_k) * dt` with
    `t_k = k/n_steps` and `dt = 1/n_steps`, matching both bespoke drivers this replaces exactly (and,
    through them, the reference `loom::MatchaDriver` / `loom::SupertonicDriver` C++ oracles).
    """
    fixed = ", ".join(f'"{n}"' for n in spec.fixed_inputs)
    lines = []
    if spec.note:
        for para in spec.note.strip().splitlines():
            lines.append(f"-- {para.rstrip()}")
    lines += [
        "-- Generated from IterativeRefinementSpec (tools/loom_mil_compiler/iterative_export.py):",
        f'--   estimator="{spec.estimator}", carried="{spec.carried_input}", '
        f'time="{spec.time_input}", fixed=[{fixed}]',
        f"local function {spec.func_name}(length, n_elems, n_steps, step_inputs)",
        "    local z = loom.gaussian_array(n_elems)",
        "    local dt = 1.0 / n_steps",
        "    for step = 0, n_steps - 1 do",
        "        local t = step / n_steps",
        "        local args = {",
        f"            {spec.carried_input} = z,",
    ]
    for name in spec.fixed_inputs:
        lines.append(f"            {name} = step_inputs.{name},")
    lines += [
        f"            {spec.time_input} = {{ t }},",
        "        }",
        f'        local v = loom.run_subgraph("{spec.estimator}", length, 0, args)',
        "        for i = 1, #z do",
        "            z[i] = z[i] + v[i] * dt",
        "        end",
        "    end",
        "    return z",
        "end",
    ]
    return "\n".join(lines)


# The marker a hand-written driver puts on its own line where the generated sampler(s) should land.
SAMPLER_MARKER = "--@loom:samplers"


def _check(spec, topology_name, topologies, label):
    if topology_name not in topologies:
        raise ValueError(
            f"{label} names topology {topology_name!r}, which is not among the exported topologies "
            f"{sorted(topologies)}."
        )
    spec.validate_against_topology(topologies[topology_name])


def render_driver(driver_source: str, specs=(), topologies=None, estimators=()) -> str:
    """Substitutes `SAMPLER_MARKER` in a hand-written driver with `specs`' generated sampler functions.

    When `topologies` is given (the exporter's own `topologies` dict), every spec is validated against
    its estimator's real declared inputs first -- so a mismatch is an export-time error naming the
    offending input, not a run-time engine error. `estimators` carries plain `EstimatorSpec`s for calls
    the driver still writes by hand (StyleTTS2's ADPM2 sampler): they are checked but generate nothing.

    With no `specs`, no marker is required -- a driver can opt into the validation alone.
    """
    if topologies is not None:
        for spec in specs:
            _check(spec, spec.estimator, topologies, f"IterativeRefinementSpec({spec.func_name!r})")
        for spec in estimators:
            _check(spec, spec.topology, topologies, f"EstimatorSpec({spec.topology!r})")
    if not specs:
        return driver_source
    if SAMPLER_MARKER not in driver_source:
        raise ValueError(
            f"driver source has no {SAMPLER_MARKER!r} line for the generated sampler(s) to replace"
        )
    return driver_source.replace(SAMPLER_MARKER, "\n\n".join(render_sampler(s) for s in specs))
