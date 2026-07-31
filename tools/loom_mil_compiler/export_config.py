"""`LoomExportConfig` -- the root of every family's export-config hierarchy (EXPORT-ROADMAP.md R3,
BACKLOG.md P3.1/P4.0.3).

Mirrors `optimum-onnx`'s `OnnxConfig`, but deliberately shallow: it owns the three fields every family
needs regardless of its own mechanics (`architecture`/`output_path`/`decomposition`) and a single
`export()` contract. Everything else -- how a checkpoint is loaded, what its wrapper looks like, how
many phases it traces -- lives on the family-specific subclasses (`causal_lm_export.
LMCausalModelExportConfig`, `nemo_asr_export.ASRNemoEncoderExportConfig`,
`multi_phase_export.BaseMultiPhaseModelExportConfig`, ...). The `{Domain}{Function}ExportConfig` naming
convention every subclass follows is described in BACKLOG.md's P3.1 entry; `LoomExportConfig` itself is
the one name in that hierarchy with no domain prefix, since it sits above every domain.

`export()` is not overridden by any family any more: it delegates to `self.decomposition`, which owns
the trace-and-assemble mechanics, while the config owns the family knowledge the decomposition asks it
for. See `decomposition.py` for why that split, and for which families genuinely have a choice of
decomposition (only causal-LM) versus a structural one (everyone else).
"""
from dataclasses import dataclass

from .decomposition import Decomposition


@dataclass(kw_only=True)
class LoomExportConfig:
    """Base for every family template's top-level config object -- the thing a registry entry
    constructs and calls `.export()` on."""

    # GGUF `general.architecture` value.
    architecture: str
    # Output .gguf path.
    output_path: str
    # How this model's graph(s) get built: one flattened trace, a ModularExportSpec assembly, or N
    # merged phases. A family with only one possible answer defaults it; only the causal-LM family
    # currently accepts either (LFM2 exports both ways -- a caller decision, not a checkpoint property).
    decomposition: Decomposition

    def export(self) -> str:
        """Runs the whole export -- load, trace, compile, write GGUF -- and returns `output_path`."""
        return self.decomposition.export(self)

    # -- hooks the decompositions read; see decomposition.py for which one needs which ------------------

    def prepare_environment(self) -> None:
        """Import-order workarounds a family needs before its real third-party package is importable at
        all (`patcher.ModelPatcher`). A no-op for families that need none, so every decomposition can
        call it unconditionally."""

    def export_architecture(self) -> str:
        """The GGUF `general.architecture` value, after any per-family resolution. Overridden by the
        causal-LM family, which infers it from the checkpoint's own `model.config.model_type` when the
        caller did not name one."""
        return self.architecture

    def backend_kwargs(self) -> dict:
        """Extra keyword arguments for `LoomGGUFBackend.__call__` beyond `output_path`/`architecture`
        (tokenizer paths, quantization, `root_axis`, `profile`). Empty by default."""
        return {}
