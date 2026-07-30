"""`LoomExportConfig` -- the root of every family's export-config hierarchy (EXPORT-ROADMAP.md R3,
BACKLOG.md P3.1).

Mirrors `optimum-onnx`'s `OnnxConfig`, but named after `optimum`'s own vocabulary is deliberately
shallow here: it owns exactly the three fields every family needs regardless of its own mechanics
(`architecture`/`output_path`/`profile`) and a single `export()` contract. Everything else -- how many
topologies a family traces, what its dummy inputs look like, which axis is dynamic -- lives on the
family-specific subclasses (`causal_lm_export.CausalModelExportConfig`,
`nemo_asr_export.ASRNemoEncoderExportConfig`, `multi_phase_export.BaseMultiPhaseModelExportConfig`, ...).

See `BACKLOG.md`'s "Target class hierarchy and naming" section for the full family tree and the
`{Domain}{Function}ExportConfig` naming convention every subclass follows -- `LoomExportConfig` itself is
the one name in that hierarchy that stays exactly as BACKLOG.md's own P3.1 item names it, with no domain
prefix, since it sits above every domain.
"""
from dataclasses import dataclass


@dataclass(kw_only=True)
class LoomExportConfig:
    """Base for every family template's top-level config object -- the thing a registry entry
    constructs and calls `.export()` on."""

    # GGUF `general.architecture` value.
    architecture: str
    # Output .gguf path.
    output_path: str
    # "monolithic" (default, one flattened topology or one merged multi-phase GGUF) or "modular"
    # (EXPORT-ROADMAP.md R7 -- independently-traced submodules assembled per ModularExportSpec).
    profile: str = "monolithic"

    def export(self) -> str:
        """Runs the whole export -- load, trace, compile, write GGUF -- and returns `output_path`."""
        raise NotImplementedError
