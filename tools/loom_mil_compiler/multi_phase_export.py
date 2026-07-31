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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from .export_config import LoomExportConfig
from .exporter import LoomGGUFExporter
from .flow_matching_export import EstimatorSpec, FlowMatchingSpec, render_driver


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


def _merge_phase_weights(named_weights: List[Tuple[str, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
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
    """Trace-each-phase / merge-weights-with-collision-check / optionally-render_driver / write_gguf
    driver. Subclasses implement `phases()`; `samplers()` (`FlowMatchingSpec`s -- codegen a sampler
    function) and `estimators()` (plain `EstimatorSpec`s -- validate a hand-written sampler call,
    generate nothing) both default to empty, so calling `render_driver` is always safe: with nothing to
    check or generate it returns the driver source unchanged, matching a family with no sampler at all
    (Kokoro, VITS) exactly as if `render_driver` were never called."""

    driver_script_path: Path

    def phases(self) -> List[ExportPhase]:
        raise NotImplementedError

    def samplers(self) -> List[FlowMatchingSpec]:
        return []

    def estimators(self) -> List[EstimatorSpec]:
        return []

    def export(self) -> str:
        phase_topologies = {}
        named_weights = []
        for phase in self.phases():
            traced = torch.jit.trace(phase.wrapper, phase.dummy_inputs)
            mil_prog = ct.convert(
                traced, inputs=phase.mil_inputs, convert_to="milinternal",
                compute_precision=ct.precision.FLOAT32,
            )
            exporter = LoomGGUFExporter(mil_prog, root_axis=phase.root_axis, declared_axes=phase.declared_axes)
            main_func = mil_prog.functions["main"]
            topo = exporter.generate_graph_topology(main_func, phase.name)
            print(f"  {phase.name}: {len(topo['nodes'])} nodes, {len(exporter.weights)} weights")
            phase_topologies[phase.name] = topo
            named_weights.append((phase.name, exporter.weights))

        merged_weights = _merge_phase_weights(named_weights)

        out_exporter = LoomGGUFExporter(None, output_path=self.output_path, architecture=self.architecture)
        out_exporter.topologies = phase_topologies
        out_exporter.weights = merged_weights

        driver_source = render_driver(
            self.driver_script_path.read_text(), self.samplers(),
            topologies=out_exporter.topologies, estimators=self.estimators(),
        )
        out_exporter.write_gguf(driver_source)
        print(f"wrote {self.output_path}")
        return self.output_path


@dataclass(kw_only=True)
class TTSFlowMatchingModelExportConfig(BaseMultiPhaseModelExportConfig):
    """Matcha + Supertonic: Euler integration of a learned vector field over a loop-carried tensor
    (`flow_matching_export.py`'s `FlowMatchingSpec` -- Matcha's own code calls this sampling style
    "CFM"), layered on top of the same multi-phase tracing every TTS family needs. StyleTTS2's ADPM2
    sampler is NOT this class -- `flow_matching_export.py`'s own docstring already documents why it
    can't be generalized (two network evals/step, Karras preconditioning, not flow matching) -- so
    StyleTTS2 stays a plain `BaseMultiPhaseModelExportConfig` with its sampler hand-written and merely
    `EstimatorSpec`-checked via `estimators()`."""
