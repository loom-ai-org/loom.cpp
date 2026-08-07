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
from .spec_protocol import NestedSpec, Unchecked


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

    # The three fields every family inherits, declared once here rather than restated by each of the
    # five (P4.0.5's standing rule; `spec_protocol.declared_raw` merges these along the MRO, which is
    # what makes declaring them once possible at all).
    __links__ = {
        "decomposition": NestedSpec(
            where="Decomposition.export(), which checks the decomposition's own specs -- Modular "
                  "carries a ModularExportSpec whose ModuleAttrPath links are checked against the "
                  "loaded model there, MultiPhase checks each ExportPhase's axes"
        ),
    }
    __unchecked__ = {
        "architecture": Unchecked(
            "the GGUF `general.architecture` string. Free-form by design -- it names the family to a "
            "GGUF reader, and the causal-LM family infers it from the checkpoint's own model_type, so "
            "there is no independent authority to check it against."
        ),
        "output_path": Unchecked(
            "where to write. Nothing real to check it against: the file does not exist yet, and "
            "whether its directory is writable is the filesystem's error to raise, not a spec claim."
        ),
    }

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
        (tokenizer paths, quantization, `root_axis`, `flat_namespace`). Empty by default.

        **An override must carry `hparams=self.hparams()` through.** Every override here builds its own
        dict rather than updating `super()`'s, which is fine for kwargs a family opts into and wrong
        for one every family has -- so the four that exist each pass it explicitly, and
        `test_export_hparams.py` walks the registry to check they still do. A hook honoured by one path
        out of four is worse than no hook: it reads as available and silently does nothing."""
        return {"hparams": self.hparams()}

    def hparams(self) -> dict:
        """`{key: number}` a **host** needs in order to call this model's driver at all -- written into
        the GGUF as `loom.<key>` KVs, the same namespace `loom::make_kv_cache` already reads its five
        geometry facts from.

        This is one half of a split, and which half a number belongs in is decided by who reads it:

        * a number the **driver** needs is an `ExportConstants` value (`driver_components.py`), bound as
          an IR local, because Lua cannot read GGUF metadata at all;
        * a number the **host** needs -- to size an input it must build, or to interpret an output --
          belongs here, because the driver cannot hand it over before being called.

        Kokoro's `style_dim` is the clearest case of the second: a caller cannot construct `ref_s`
        without knowing how long each of its two halves is, and until this existed the answer lived in
        `tests/tts_driver_inputs.h` -- a C++ test header, which is exactly the "self-contained GGUF"
        claim being false (P4.0.8's first follow-up; KV-CACHE.md 1.1/1.3 made the same argument for
        cache geometry).

        A number both sides need is declared once *here in the config* and rendered twice, which is not
        the "two spellings that can disagree" `kv_cache.h` warns about: both readings come from one
        attribute set in `phases()`, so there is no second authority. Supertonic's fixed text length is
        that case -- the host needs it to build `txt_ids`, the driver needs it to reject a wrong-length
        one.

        Empty by default; `int` values are written as u32 and `float` as f32."""
        return {}
