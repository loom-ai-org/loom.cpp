"""The causal-LM family (BACKLOG.md P3.1/P3.2, restructured in P4.0.3): ONE
`LMCausalModelExportConfig`, whose `decomposition` field says whether a checkpoint exports as one
flattened trace (`Flattened()` -- Qwen3, LFM2-monolithic) or as independently-traced submodules
assembled per a `ModularExportSpec` (`Modular(spec=...)` -- LFM2-modular, EXPORT-ROADMAP.md R7's
"modular" split).

P3.1 built these as two sibling classes, which is what BACKLOG.md P4.0.3 came back to fix: the two forms
differ in how the program is assembled, not in what a causal-LM config knows, and LFM2 exports BOTH ways
from the same checkpoint -- a caller decision, so it belongs in a field rather than in a type. What was
per-class now lives on the decomposition (`Modular.spec`/`dummy_seq_len`) or on the shared config
(`seq_len`, tokenizer paths, `quantize`), and the mechanics moved to `decomposition.py`; the traced
graphs and the GGUFs they produce are byte-identical either way.

Since P4.0.4 this family is reachable two ways: three *specific* recognizers (Qwen3, LFM2 ×2) and one
generic `hf-causal-lm` fallback that claims any HF directory declaring a `*ForCausalLM` architecture.
The family was already model-agnostic underneath -- one wrapper over plain `AutoModelForCausalLM`, the
architecture inferred from `config.model_type`, the tokenizer family and pretokenizer auto-detected in
the exporter -- so what the fallback removes is the requirement to hand-write an `_is_llama` +
`_build_llama` pair to reach code that never needed to know it was Llama.

`export_hf_causal_lm.py` keeps its own CLI as a thin wrapper around this class.
"""
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Bypass the transformers library hf-hub bounds check to import safely -- moved here verbatim from
# export_hf_causal_lm.py, which now imports this module instead of applying the stub itself.
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForCausalLM

from .decomposition import Flattened, Modular
from .export_config import LoomExportConfig
from .modular_export import ModularExportSpec


def _causal_mask(seq_len: int) -> torch.Tensor:
    # A real, already-prepared 4D additive mask short-circuits transformers' create_causal_mask /
    # _preprocess_mask_arguments entirely (`if isinstance(attention_mask, torch.Tensor) and
    # len(attention_mask.shape) == 4: return True, attention_mask, ...` -- "returned as-is"). That's
    # needed here for the SAME reason cache_position is passed explicitly: the internal mask-building
    # path (masking_utils.py) derives kv_length from a Python-level `input_embeds.shape[1]` query when no
    # cache is used, which torch.jit.trace bakes in as the tracing dummy's fixed length regardless of
    # ct.RangeDim declared afterward (confirmed empirically -- a RESHAPE feeding off `cache_position`
    # stayed hardcoded to the traced length even after cache_position itself became a genuine dynamic
    # input).
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


class _CausalLMWrapper(torch.nn.Module):
    """Reduces any HF causal-LM's forward() to the (tokens, cache_position, attention_mask) -> logits
    shape the traced/exported graph needs, regardless of the model's specific architecture."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, cache_position, attention_mask):
        outputs = self.model(tokens, cache_position=cache_position, attention_mask=attention_mask)
        return outputs.logits


@dataclass(kw_only=True)
class LMCausalModelExportConfig(LoomExportConfig):
    """Any plain `AutoModelForCausalLM`-shaped HF model -> Loom GGUF. ONE class for both export shapes
    (BACKLOG.md P4.0.3): which one you get is `decomposition`, not which class you instantiated.

    `Flattened()` traces the whole forward pass into one topology (Qwen3, LFM2-monolithic);
    `Modular(spec=...)` traces each submodule standalone and assembles them per a `ModularExportSpec`
    (LFM2-modular). LFM2 is the one model that genuinely exports either way, which is exactly why this
    is a field: "monolithic" vs "modular" is a caller decision about the same checkpoint, not a property
    of it -- see `_is_lfm2`, where both recognizers deliberately detect the same directory."""

    # `LoomExportConfig.architecture` is normally required, but this family alone infers it from the
    # checkpoint's own `model.config.model_type` when not given -- matching `export_causal_lm`'s
    # original `architecture: str = None` default exactly.
    architecture: Optional[str] = None
    model_dir: str
    tokenizer_dir: Optional[str] = None
    tokenizer_family: Optional[str] = None
    tokenizer_pre: Optional[str] = None
    quantize: Optional[str] = None
    # Concrete length torch.jit.trace runs at for `Flattened`; the dynamic range is declared separately
    # via ct.convert's own `inputs=` (EXPORT-BACKLOG.md item 3: a fixed traced shape bakes a literal
    # length into every exported slice, forcing the driver to pad every prompt to that fixed length).
    # `Modular` carries its own `dummy_seq_len` instead, for the collision reason documented there.
    seq_len: int = 128
    max_seq_len: int = 4096
    # Resolved from the checkpoint by `load_model()` when `architecture` was not given.
    _resolved_architecture: Optional[str] = None

    def load_model(self):
        print(f"Loading model from {self.model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32).eval()
        self._resolved_architecture = self.architecture or getattr(model.config, "model_type", None)
        if not self._resolved_architecture:
            raise ValueError(
                "architecture could not be inferred from model.config.model_type; pass it explicitly"
            )
        return model

    def export_architecture(self) -> str:
        return self._resolved_architecture or self.architecture

    def build_trace(self, model):
        """`Flattened`'s hook: the wrapper, its dummy inputs, and the MIL input declarations.

        `tokens`/`cache_position`/`attention_mask` share the SAME `ct.RangeDim` instance so coremltools
        ties them to one symbolic length (they must always be called with matching lengths at runtime)
        -- see `apply_monolithic_export`'s own auto-generation of "cache_position"-named inputs via
        `loom.range(...)` and "attention_mask"-named inputs via `loom.causal_mask(...)`, and P4.0.2's
        `_validate_input_axes`, which now enforces that one-symbol rule."""
        print(f"Tracing the complete PyTorch graph (dummy seq_len={self.seq_len})...")
        dummy_inputs = (
            torch.zeros((1, self.seq_len), dtype=torch.long),
            torch.arange(self.seq_len, dtype=torch.long),
            _causal_mask(self.seq_len),
        )
        seq_len_dim = ct.RangeDim(1, self.max_seq_len)
        mil_inputs = [
            ct.TensorType(name="tokens", shape=(1, seq_len_dim), dtype=np.int32),
            ct.TensorType(name="cache_position", shape=(seq_len_dim,), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, seq_len_dim, seq_len_dim), dtype=np.float32),
        ]
        return _CausalLMWrapper(model), dummy_inputs, mil_inputs

    def modular_dummy_inputs(self, dummy_seq_len: int) -> dict:
        """`Modular`'s hook: the same three tensors `build_trace` builds, in the kwarg form
        `export_modular` replays submodules with."""
        return dict(
            input_ids=torch.zeros((1, dummy_seq_len), dtype=torch.long),
            cache_position=torch.arange(dummy_seq_len, dtype=torch.long),
            attention_mask=_causal_mask(dummy_seq_len),
        )

    def backend_kwargs(self) -> dict:
        kwargs = dict(
            tokenizer_dir=self.tokenizer_dir or self.model_dir,
            tokenizer_pre=self.tokenizer_pre,
        )
        # Only the flattened form declares these: `flat_namespace` writes every weight under one flat
        # name, which is right for a single-topology export and wrong for the modular path, whose
        # per-submodule functions must keep their `{func_name}.` prefixes. `tokenizer_family` /
        # `quantize` likewise were never passed by the modular export.
        if isinstance(self.decomposition, Flattened):
            kwargs.update(
                flat_namespace=True,
                tokenizer_family=self.tokenizer_family,
                quantize=self.quantize,
            )
        return kwargs


def _hf_config(path: Path) -> Optional[dict]:
    """An HF-style directory's own `config.json`, parsed, or None if `path` isn't one. Never raises:
    `detect()` runs against unidentified paths by construction, so a malformed or unreadable
    `config.json` is a "no" rather than an error."""
    cfg_path = path / "config.json"
    if not path.is_dir() or not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _hf_model_type(path: Path) -> Optional[str]:
    """An HF-style directory's own `config.json`'s `model_type`, or None if `path` isn't one."""
    cfg = _hf_config(path)
    return cfg.get("model_type") if cfg is not None else None


def _is_qwen3(path: Path) -> bool:
    """Real structural check (BACKLOG.md P3.2): an HF-style directory whose own `config.json` declares
    `model_type == "qwen3"`."""
    return _hf_model_type(path) == "qwen3"


def _build_qwen3(path: Path, output_path: str) -> LoomExportConfig:
    return LMCausalModelExportConfig(
        architecture="qwen3", output_path=output_path, decomposition=Flattened(), model_dir=str(path),
    )


def _is_lfm2(path: Path) -> bool:
    """Real structural check: `model_type == "lfm2"`. Registered under BOTH `lfm2-monolithic` and
    `lfm2-modular` (BACKLOG.md's migration of LFM2 onto the registry, following P3.1/P3.2/P3.3) -- unlike
    Parakeet-TDT/-RNNT, which the checkpoint's own config genuinely distinguishes, "monolithic" vs
    "modular" is a caller CHOICE about how to export the same checkpoint, not a property `detect()` could
    ever read off it. So both recognizers legitimately match the same real LFM2 directory, and
    `TaskRegistry.detect()` correctly raises asking for `--model lfm2-monolithic`/`--model lfm2-modular`
    to disambiguate -- the same honest "can't guess, ask" behavior as any other genuine ambiguity, not a
    gap."""
    return _hf_model_type(path) == "lfm2"


def _build_lfm2_monolithic(path: Path, output_path: str) -> LoomExportConfig:
    return LMCausalModelExportConfig(
        architecture="lfm2", output_path=output_path, decomposition=Flattened(), model_dir=str(path),
        tokenizer_pre="llama3",
    )


def _build_lfm2_modular(path: Path, output_path: str) -> LoomExportConfig:
    return LMCausalModelExportConfig(
        architecture="lfm2", output_path=output_path, model_dir=str(path),
        decomposition=Modular(spec=ModularExportSpec(
            prefix_attr="model.embed_tokens",
            repeated_attr="model.layers",
            suffix_attrs=["model.embedding_norm", "lm_head"],
            aux_attr="model.pos_emb",
            aux_kwarg="position_embeddings",
        )),
        tokenizer_dir=str(path), tokenizer_pre="llama3",
    )


# -- the generic recognizer (BACKLOG.md P4.0.4) -------------------------------------------------------

def _is_hf_causal_lm(path: Path) -> bool:
    """Any HF-style directory that declares a `model_type` AND an `architectures` entry ending in
    `ForCausalLM`.

    **Both halves are load-bearing.** `model_type` alone is what `load_model` needs (it becomes
    `general.architecture`), but nearly every HF directory has one -- Whisper, Parakeet and GigaAM all sit
    beside the causal LMs on this machine, and a `detect()` that claimed them would break three other
    families' detection, since `TaskRegistry.detect` runs every recognizer against every path. The
    `architectures` entry is the checkpoint's own statement of which `AutoModelFor*` class it loads
    through, which is exactly the claim `_CausalLMWrapper` needs to be true: `WhisperForConditionalGeneration`
    and `ParakeetForCTC` are rejected by it, `Qwen3ForCausalLM` and `Lfm2ForCausalLM` accepted.

    Registered `fallback=True`, so it is consulted only when no specific recognizer matched -- Qwen3 still
    resolves to `qwen3` and LFM2 still raises its intended two-way ambiguity."""
    cfg = _hf_config(path)
    if cfg is None or not cfg.get("model_type"):
        return False
    architectures = cfg.get("architectures") or []
    if not isinstance(architectures, list):
        return False
    return any(isinstance(arch, str) and arch.endswith("ForCausalLM") for arch in architectures)


# Per-`model_type` exceptions to the generic path's defaults, as `LMCausalModelExportConfig` kwargs.
#
# **Empty, and that is the finding rather than an omission.** The two model types with a specific
# recognizer are the only evidence available on what a generic path would get wrong, and neither needs
# an entry: the exporter's own tokenizer auto-detection resolves LFM2 to `llama3` and Qwen3 to `qwen2`,
# which is exactly what `_build_lfm2_*`/`_build_qwen3` hardcode (asserted in `test_causal_lm_export.py`).
# Both stay specific recognizers anyway, for a reason no table could carry: LFM2's monolithic/modular
# split is a caller decision, not a checkpoint property. This table is where a real exception goes when
# a checkpoint turns one up -- a `tokenizer_pre` the hash cascade does not know, an `architecture` whose
# `model_type` disagrees with what the engine expects.
_MODEL_TYPE_OVERRIDES: dict[str, dict] = {}


def _build_hf_causal_lm(path: Path, output_path: str) -> LoomExportConfig:
    """The generic build: everything the family can infer, inferred. `architecture=None` is resolved from
    the checkpoint's own `model.config.model_type` in `load_model`, `tokenizer_pre=None` from the
    tokenizer's own hash in `_write_tokenizer`, and the decomposition is `Flattened()` -- the only one a
    checkpoint can be exported with without a caller naming submodule attribute paths."""
    overrides = _MODEL_TYPE_OVERRIDES.get(_hf_model_type(path) or "", {})
    return LMCausalModelExportConfig(
        architecture=None, output_path=output_path, decomposition=Flattened(), model_dir=str(path),
        **overrides,
    )


def register(registry) -> None:
    """Registers this family's `TaskRegistryEntry` (BACKLOG.md P3.2, extended with LFM2's two profiles,
    then with P4.0.4's generic recognizer).

    Qwen3 and both LFM2 profiles stay specific rather than folding into the generic recognizer: LFM2
    needs two entries for one directory, which is a choice `detect()` cannot read off a checkpoint at
    all, and Qwen3's stays as the worked example of what a specific recognizer looks like next to the
    generic one."""
    from .registry import ModelRecognizer, TaskRegistryEntry

    registry.register(TaskRegistryEntry(
        task="text-generation",
        config_class=LMCausalModelExportConfig,
        recognizers=[
            ModelRecognizer(name="qwen3", detect=_is_qwen3, build_config=_build_qwen3),
            ModelRecognizer(name="lfm2-monolithic", detect=_is_lfm2, build_config=_build_lfm2_monolithic),
            ModelRecognizer(name="lfm2-modular", detect=_is_lfm2, build_config=_build_lfm2_modular),
            ModelRecognizer(
                name="hf-causal-lm", detect=_is_hf_causal_lm, build_config=_build_hf_causal_lm,
                fallback=True,
            ),
        ],
    ))
