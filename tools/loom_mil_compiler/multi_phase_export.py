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


@dataclass
class ExportPhase:
    """One traced topology within a multi-phase model: a wrapper module, its dummy inputs, and its MIL
    input declarations. Generalizes each script's own repeated `_build_topology` call (Kokoro's
    `build_albert_bert_encoder_topology`/`build_decoder_vocoder_topology`, VITS/Matcha's `_build_topology`
    helper, ...). `root_axis`/`declared_axes` are `LoomGGUFExporter`'s own kwargs of the same name --
    default to plain `"n_tokens"`/no overrides, matching every phase that doesn't need Kokoro's
    `decoder_vocoder`-style multi-axis declaration."""

    name: str
    wrapper: nn.Module
    dummy_inputs: tuple
    mil_inputs: List[ct.TensorType]
    root_axis: str = "n_tokens"
    declared_axes: Optional[dict] = None


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

    def phases(self) -> List[ExportPhase]:
        raise NotImplementedError

    def samplers(self) -> List[FlowMatchingSpec]:
        return []

    def estimators(self) -> List[EstimatorSpec]:
        return []


@dataclass(kw_only=True)
class TTSFlowMatchingModelExportConfig(BaseMultiPhaseModelExportConfig):
    """Matcha + Supertonic: Euler integration of a learned vector field over a loop-carried tensor
    (`flow_matching_export.py`'s `FlowMatchingSpec` -- Matcha's own code calls this sampling style
    "CFM"), layered on top of the same multi-phase tracing every TTS family needs. StyleTTS2's ADPM2
    sampler is NOT this class -- `flow_matching_export.py`'s own docstring already documents why it
    can't be generalized (two network evals/step, Karras preconditioning, not flow matching) -- so
    StyleTTS2 stays a plain `BaseMultiPhaseModelExportConfig` with its sampler hand-written and merely
    `EstimatorSpec`-checked via `estimators()`."""
