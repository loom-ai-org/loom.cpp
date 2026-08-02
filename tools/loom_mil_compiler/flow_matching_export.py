"""Generalizes the "Euler integration of a learned vector field over loop-carried tensor state" driver
family -- EXPORT-IMPROVEMENT.md item 4, the second family template alongside `modular_export.py`'s
`ModularExportSpec`. Renamed from `IterativeRefinementSpec`/`iterative_export.py` (BACKLOG.md P3.3):
what this declares is flow matching specifically (Matcha's own code calls its sampler "CFM"/Conditional
Flow Matching; Supertonic's estimator is literally `VectorFieldEstimator.compute_velocity`), not generic
"iterative refinement" -- a name that would blur why StyleTTS2's real diffusion sampler (ADPM2, see
below) is deliberately NOT part of this family.

Matcha-TTS and SupertonicTTS are not two unrelated problems: both trace a per-step *estimator* network
into one topology and then integrate it host-side with forward Euler over a loop-carried state tensor,
and both hand-wrote the identical Lua loop to do it:

    local z = loom.gaussian_array(<n_elems>)
    local dt = 1.0 / inputs.n_steps
    for step = 0, inputs.n_steps - 1 do
        local t = step / inputs.n_steps
        local v = loom.run_subgraph("<estimator>", {n_tokens = <length>, n_past = 0}, { <carried> = z, ...fixed..., t = { t } })
        for i = 1, #z do z[i] = z[i] + v[i] * dt end
    end

A `FlowMatchingSpec` declares the six things that actually differ between them -- the estimator
topology's name, which of its inputs carries the state, which carries the scalar time, which are fixed
across steps, how many elements the state has, and what `n_tokens` to build the estimator's graph at --
and `render_sampler` emits that loop as a named Lua function the hand-written driver calls in one line.

The point is not the line count. It is that the spec is **checked against the real traced topology** at
export time (`validate_against_topology`): naming an input the estimator doesn't declare, or forgetting
one it does, raises immediately with the actual declared-input list. A hand-written Lua loop with the
same mistake fails much later, inside the engine, as an undeclared-input or missing-input error with no
connection back to the line that got it wrong. This mirrors `ModularExportSpec`'s own "a wrong
attribute path raises `AttributeError` immediately, not a silent wrong export" property.

Deliberately NOT generalized here: the *integration rule*. Both current models use plain deterministic
forward Euler with uniform `dt = 1/n_steps`; StyleTTS2's diffusion sampler is ADPM2 over a Karras sigma
schedule (second-order, two network evaluations per step, per-step noise injection, and real
preconditioning math around the call), and Kokoro/VITS's duration loops are a different shape entirely
(a scatter over predicted durations, not an ODE). Those stay bespoke, the same way
`modular_export.py` concedes non-causal-LM architectures.

The *validation*, though, generalizes further than the codegen does -- a hand-written `run_subgraph`
call fails the same way a generated one would. So `EstimatorSpec` (below) is its own declaration: a
bespoke sampler can still declare which topology it calls with which inputs and get the export-time
check, generating nothing. See BACKEND.md for why the loop itself stays host-side rather than becoming
a MIL `while_loop`.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .spec_protocol import (
    CoveredBy, DriverSymbol, FieldRef, LinkChecker, TopologyInput, TopologyName, TopologyOutputArity,
    Unchecked,
)


@dataclass
class EstimatorSpec:
    """One traced per-step network a driver calls, plus the exact set of topology inputs it is handed.

    Separate from `FlowMatchingSpec` because the *validation* generalizes further than the
    *codegen* does. StyleTTS2's diffusion sampler is ADPM2 over a Karras sigma schedule -- two network
    evaluations per step, per-step noise injection, and real preconditioning math
    (`c_skip`/`c_out`/`c_in`/`c_noise`) wrapped around the call -- so no template can emit its loop
    without becoming a worse thing to read than the loop. But its `run_subgraph` call has the identical
    failure mode as every other one: a name that does not match the topology's declared inputs is only
    caught deep inside the engine at run time. Declaring the call alone still buys that check.

    Both of its checks are `spec_protocol` links as of P4.0.5 (`EXPORT-PREPARATION.md` stage B.2). The
    hand-written `validate_against_topology` below is now a thin adapter kept for its callers; the
    predicate and the message live in `TopologyName`/`TopologyInput`.
    """

    topology: str
    # Topology input names the driver supplies, in argument-table order.
    inputs: list

    __links__ = {
        "topology": TopologyName(),
        "inputs": TopologyInput(FieldRef("topology"), exact=True),
    }

    def link_label(self) -> str:
        """A driver can declare several hand-written estimator calls, so the topology name is what
        tells a reader which one failed -- matching the label `render_driver` used to pass by hand."""
        return f"EstimatorSpec({self.topology!r})"

    def validate_against_topology(self, topology: dict, label: Optional[str] = None):
        """Cross-checks against the topology's real declared inputs, raising ValueError naming the exact
        mismatch. This is the whole reason these specs exist rather than a hand-written call.

        Retained as the single-topology entry point (callers hand it one topology dict, not the whole
        export's), but the checking is `spec_protocol`'s: `LinkError` subclasses `ValueError`, and the
        message is byte-identical to the one this method used to build itself."""
        from .spec_protocol import check_links

        check_links(self, topologies={self.topology: topology}, label=label or "EstimatorSpec")


@dataclass
class FlowMatchingSpec:
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

    __links__ = {
        "estimator": [
            TopologyName(),
            # New in P4.0.5, and a real gap rather than a restatement. `render_sampler` emits
            # `local v = loom.run_subgraph(...)` and then indexes `v[i]` -- correct only for a
            # single-output estimator. With two declared outputs `v` silently binds the FIRST output's
            # data and the loop integrates the wrong tensor, with nothing anywhere reporting it: the
            # engine is happy, the shapes are plausible and the audio is merely wrong.
            TopologyOutputArity(FieldRef("estimator"), count=1),
        ],
        # The per-step argument table, checked as one set against the estimator's declared inputs so the
        # message can name both what is supplied-but-undeclared and what is declared-but-unsupplied.
        "supplied_inputs": TopologyInput(FieldRef("estimator"), exact=True),
        "carried_input": CoveredBy("supplied_inputs"),
        "time_input": CoveredBy("supplied_inputs"),
        "fixed_inputs": CoveredBy("supplied_inputs"),
        # Was `Unchecked` until P4.0.6/C.4, with the reason "checkable as a DriverSymbol only once the
        # driver is IR rather than text". It now is, for both driver shapes, because `DriverSymbol`
        # resolves against the built *script* rather than the entry function alone: a generated sampler
        # is a top-level `local function` in the prelude either way -- emitted by `FlowMatchingSampler`
        # for a peeled family, substituted into the adopted text for one that is not.
        "func_name": DriverSymbol(),
    }
    __unchecked__ = {
        "note": Unchecked("cosmetic: rendered as a comment above the generated sampler."),
    }

    @property
    def supplied_inputs(self) -> List[str]:
        """The per-step argument table's keys, in the order `render_sampler` writes them.

        A property rather than a field so the declaration and the emission cannot disagree: this is the
        list the generated Lua actually passes, and it is also the list the link checks. `__links__`
        resolves it through plain `getattr`, so a derived value is a first-class link subject."""
        return [self.carried_input, *self.fixed_inputs, self.time_input]

    def link_label(self) -> str:
        """A model may declare more than one sampler, so "FlowMatchingSpec" alone does not say which
        one failed."""
        return f"FlowMatchingSpec({self.func_name!r})"

    def estimator_spec(self) -> EstimatorSpec:
        """This spec's per-step call, as the plain declaration a bespoke sampler would write by hand --
        so both share one validation implementation and cannot drift apart."""
        return EstimatorSpec(topology=self.estimator, inputs=self.supplied_inputs)

    def validate_against_topology(self, topology: dict):
        self.estimator_spec().validate_against_topology(
            topology, label=f"FlowMatchingSpec({self.func_name!r})")


def render_sampler(spec: FlowMatchingSpec) -> str:
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
        "-- Generated from FlowMatchingSpec (tools/loom_mil_compiler/flow_matching_export.py):",
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
        f'        local v = loom.run_subgraph("{spec.estimator}", {{n_tokens = length, n_past = 0}}, args)',
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


def render_driver(driver_source: str, specs=(), topologies=None, estimators=(), checker=None) -> str:
    """Substitutes `SAMPLER_MARKER` in a hand-written driver with `specs`' generated sampler functions.

    When `topologies` is given (the exporter's own `topologies` dict), every spec is validated against
    its estimator's real declared inputs first -- so a mismatch is an export-time error naming the
    offending input, not a run-time engine error. `estimators` carries plain `EstimatorSpec`s for calls
    the driver still writes by hand (StyleTTS2's ADPM2 sampler): they are checked but generate nothing.

    With no `specs`, no marker is required -- a driver can opt into the validation alone.

    As of P4.0.5 the checking is `spec_protocol`'s (`EXPORT-PREPARATION.md` stage B.2). `checker` lets
    the caller pass the export-wide `LinkChecker` -- `MultiPhase.export` does, so a spec declared here
    and a spec declared anywhere else in the same export share one deferral ledger and one `finish()`.

    **Without one, a local checker is built and finished here -- and since P4.0.6/C.4 that is no longer
    the same guarantee.** `FlowMatchingSpec.func_name` is a `DriverSymbol`, answerable only against the
    built driver, which this function does not have: it produces the text, the builder produces the
    script. So the local checker is handed the substituted source as the driver, which is exactly what
    it is -- a Lua module whose top-level `local function` definitions are the symbols in question.
    """
    if topologies is not None:
        owned = checker is None
        checker = checker or LinkChecker()
        for spec in list(specs) + list(estimators):
            checker.check(spec)
        checker.provide(topologies=topologies)
        if owned:
            rendered = _substitute(driver_source, specs)
            checker.provide(driver=_TextDriver(rendered.split("\n")))
            checker.finish()
    return _substitute(driver_source, specs)


def _substitute(driver_source: str, specs) -> str:
    if not specs:
        return driver_source
    if SAMPLER_MARKER not in driver_source:
        raise ValueError(
            f"driver source has no {SAMPLER_MARKER!r} line for the generated sampler(s) to replace"
        )
    return driver_source.replace(SAMPLER_MARKER, "\n\n".join(render_sampler(s) for s in specs))


@dataclass
class _TextDriver:
    """A driver that is still text rather than a built `DriverScript`, for `DriverSymbol` to read.

    Every line is "prelude" because that is what the substituted source is from this function's point
    of view: a Lua module whose top-level definitions are visible and whose entry function's body is
    not this function's business."""

    prelude: list

    @property
    def entry(self):
        from .driver_ir import Function

        return Function("<text driver>", [], [])
