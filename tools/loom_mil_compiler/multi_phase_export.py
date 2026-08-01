"""`BaseMultiPhaseModelExportConfig` + `TTSFlowMatchingModelExportConfig` (BACKLOG.md P3.3): the shared
"trace N independently-traced topologies, merge their weights, optionally render a flow-matching
sampler, write one GGUF" driver that replaces the near-identical tail of every TTS `export_*_mil.py`
script (Kokoro/StyleTTS2/VITS's own `_build_topology` + weight-merge + `write_gguf` boilerplate, and
Matcha/Supertonic's additional `render_driver`/`FlowMatchingSpec` wiring on top of it).

This is domain-agnostic mechanics, not a TTS-specific concept -- any family whose export is "N
independently-traced topologies assembled into one GGUF" fits `BaseMultiPhaseModelExportConfig`, the
same way `optimum`'s own submodel decomposition isn't unique to any one task. Maps onto
`EXPORT-ROADMAP.md`'s R5 family table as: `LMCausalModelExportConfig` (`causal_lm_export.py`) ~ family
3's LM half and plain LMs; `ASRNemoEncoderExportConfig` (`nemo_asr_export.py`) ~ family 1;
`BaseMultiPhaseModelExportConfig`/`TTSFlowMatchingModelExportConfig` (here) ~ families 7/8/9. A future
family follows the same `{Domain}{Function}ExportConfig` naming convention before it's written, e.g.
`ASREncoderDecoderExportConfig` (family 2's AED cross-attention decoder loop, shared with family 6),
`ASRWhisperExportConfig`/`ASRGigaAMExportConfig` (concrete leaves under it), a future
`BERTTokenClassifierExportConfig` (family 12), or `TTSVocoderExportConfig` if a standalone codec/vocoder
family (11) ever needs its own abstract tier.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch.nn as nn

from .decomposition import Decomposition, MultiPhase
from .export_config import LoomExportConfig
from .flow_matching_export import EstimatorSpec, FlowMatchingSpec
from .spec_protocol import Axis, ConfigDerived, NestedSpec, Unchecked


@dataclass
class ExportPhase:
    """One traced topology within a multi-phase model: a wrapper module, its dummy inputs, and its MIL
    input declarations. Generalizes each script's own repeated `_build_topology` call (Kokoro's
    `build_albert_bert_encoder_topology`/`build_decoder_vocoder_topology`, VITS/Matcha's `_build_topology`
    helper, ...). `root_axis`/`declared_axes` are `LoomGGUFExporter`'s own kwargs of the same name --
    default to plain `"n_tokens"`/no overrides, matching every phase that doesn't need Kokoro's
    `decoder_vocoder`-style multi-axis declaration.

    The axis *declarations* are `spec_protocol` links as of P4.0.5 (`EXPORT-PREPARATION.md` stage B.5).
    `LoomGGUFExporter`'s own two raises stay exactly where they are -- they operate on the traced
    program, which no spec can see, and P4.0.2 wrote them for that reason. What moves here is the half
    that is answerable from the declaration alone, and was not being asked at all."""

    name: str
    wrapper: nn.Module
    dummy_inputs: tuple
    mil_inputs: List[ct.TensorType]
    root_axis: str = "n_tokens"
    declared_axes: Optional[dict] = None

    __links__ = {
        # A typo'd axis name is a perfectly good dict key: `_sub_symbol` substitutes it happily and the
        # phase emits shape expressions over a symbol nothing else in the model uses. Wrong, not
        # malformed, and no downstream gate looks at it.
        "root_axis": Axis(),
        "declared_axes": [
            Axis(form="declaration_table"),
            # Checked here rather than only in `_resolve_declared_axes`, which raises the same class of
            # error but only after the phase has been traced. The phase already knows its own declared
            # input names, so this costs nothing and fails before the expensive part.
            ConfigDerived(
                claim=lambda spec, ctx: True,
                measured=lambda spec, ctx: set(spec.declared_axes or ()).issubset(
                    _mil_input_names(spec)),
                detail=lambda spec, ctx: sorted(
                    set(spec.declared_axes or ()) - set(_mil_input_names(spec))),
                message=(
                    "ExportPhase '{spec.name}' declares axes for input(s) {detail}, which it does not "
                    "declare in mil_inputs. Its inputs are {spec.mil_input_names}."
                ),
                needs=(),
            ),
        ],
    }
    __unchecked__ = {
        "name": Unchecked(
            "the topology's name in the exported GGUF. It does not refer to anything -- it CREATES the "
            "reference every driver's run_subgraph call and every TopologyName link resolves against."
        ),
        "wrapper": Unchecked(
            "the nn.Module being traced. It is the model; there is no separate authority to check it "
            "against, which is what the trace itself and the per-model reference tests are for."
        ),
        "dummy_inputs": Unchecked(
            "the concrete tensors torch.jit.trace runs with. Their correspondence with mil_inputs is "
            "enforced by ct.convert, which raises on a count or dtype mismatch -- a link here would "
            "duplicate that while reading as if it checked something ct.convert does not."
        ),
        "mil_inputs": Unchecked("same: ct.convert is the authority on these, not this declaration."),
    }

    @property
    def mil_input_names(self) -> List[str]:
        """The names this phase declares to coremltools, for `declared_axes` to be checked against."""
        return _mil_input_names(self)


def _mil_input_names(phase) -> List[str]:
    return [getattr(t, "name", None) for t in (phase.mil_inputs or [])]


def merge_phase_weights(named_weights: List[Tuple[str, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
    """Content-aware merge across every phase's own weight dict: an identical name with an identical
    real value dedups silently (the common case for names that already carry a per-phase prefix, e.g.
    VITS's `f"{phase}.{weight}"` convention, where no two phases can ever collide in the first place);
    an identical name with a DIFFERENT value is a hard error naming the phase and the key. Exactly
    `export_kokoro_mil.py`'s original two-phase merge, generalized to N phases -- the same function is
    safe for VITS/Matcha's own already-fully-namespaced weights too, since a merge that can never
    actually find a collision behaves identically to a plain dict union."""
    merged: Dict[str, np.ndarray] = {}
    for name, weights in named_weights:
        for k, v in weights.items():
            if k in merged:
                if not np.array_equal(merged[k], v):
                    raise ValueError(
                        f"real weight name collision merging {name!r} weights: {k!r} has DIFFERENT values"
                    )
                continue
            merged[k] = v
    return merged


@dataclass(kw_only=True)
class BaseMultiPhaseModelExportConfig(LoomExportConfig):
    """Trace-each-phase / merge-weights-with-collision-check / optionally-render_driver / write_gguf.
    Subclasses implement `phases()`; `samplers()` (`FlowMatchingSpec`s -- codegen a sampler function)
    and `estimators()` (plain `EstimatorSpec`s -- validate a hand-written sampler call, generate
    nothing) both default to empty, so calling `render_driver` is always safe: with nothing to check or
    generate it returns the driver source unchanged, matching a family with no sampler at all (Kokoro,
    VITS) exactly as if `render_driver` were never called.

    The mechanics live in `decomposition.MultiPhase` (BACKLOG.md P4.0.3), which is `decomposition`'s
    default here rather than a caller choice: a phase split exists because the model genuinely cannot be
    traced as one graph, so unlike the causal-LM family there is no alternative to offer."""

    driver_script_path: Path
    decomposition: Decomposition = field(default_factory=MultiPhase)

    __unchecked__ = {
        "driver_script_path": Unchecked(
            "the hand-written Lua the export substitutes generated samplers into. Its *contents* are "
            "checked -- every declared sampler and estimator is cross-checked against the real traced "
            "topologies by render_driver -- but the path itself is only a file that must exist, which "
            "read_text() reports better than a link would. P4.0.6 turns the driver into IR, at which "
            "point this field stops being a path at all."
        ),
    }

    def phases(self) -> List[ExportPhase]:
        raise NotImplementedError

    def samplers(self) -> List[FlowMatchingSpec]:
        return []

    def estimators(self) -> List[EstimatorSpec]:
        return []

    def external_topologies(self) -> Dict[str, str]:
        """`{topology name: where it comes from}` for topologies this family's driver calls that this
        export deliberately does **not** produce.

        Discovered by P4.0.6/C.3's own gate, and this method is the finding. Kokoro and StyleTTS2 are
        *partial* MIL exports: their drivers run against a mix of the MIL-traced topologies in
        `*_mil.gguf` and the LSTM-bound ones still loaded from the pre-MIL `*.gguf` alongside it
        (`test_e2e_{kokoro,styletts2}_mil_lua_driver.cpp` registers both, from two `GgufModel`s). That
        is a real and load-bearing property of those two exports, and until now it was recorded only in
        a C++ test -- nothing on the export side said the emitted GGUF is not self-contained.

        Declaring it rather than inferring it is what keeps the check worth having. The alternative --
        skipping any `run_subgraph` call naming a topology this export did not produce -- makes a typo
        and a cross-GGUF dependency indistinguishable, which is the whole class of bug the check exists
        for. Both directions of the declaration are checked: a name here that this export *does*
        produce is stale, and one no call site references is dead."""
        return {}


@dataclass(kw_only=True)
class TTSFlowMatchingModelExportConfig(BaseMultiPhaseModelExportConfig):
    """Matcha + Supertonic: Euler integration of a learned vector field over a loop-carried tensor
    (`flow_matching_export.py`'s `FlowMatchingSpec` -- Matcha's own code calls this sampling style
    "CFM"), layered on top of the same multi-phase tracing every TTS family needs. StyleTTS2's ADPM2
    sampler is NOT this class -- `flow_matching_export.py`'s own docstring already documents why it
    can't be generalized (two network evals/step, Karras preconditioning, not flow matching) -- so
    StyleTTS2 stays a plain `BaseMultiPhaseModelExportConfig` with its sampler hand-written and merely
    `EstimatorSpec`-checked via `estimators()`."""
