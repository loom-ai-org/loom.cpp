"""The causal-LM family (BACKLOG.md P3.1/P3.2): `LMCausalModelExportConfig` and its two concrete forms,
`LMMonolithicCausalModelExportConfig` (one flattened trace -- Qwen3's shape) and
`LMModularCausalModelExportConfig` (independently-traced submodules assembled per `ModularExportSpec`,
EXPORT-ROADMAP.md R7's "modular" profile -- LFM2's shape). Qwen3 only ever needs Monolithic; LFM2 is the
one model on the roadmap that exercises both.

This is a straight move of `export_hf_causal_lm.py`'s `export_causal_lm()` body (now
`LMMonolithicCausalModelExportConfig.export()`) and `export_lfm2_modular.py`'s `main()` body (now
`LMModularCausalModelExportConfig.export()`) -- no logic changes. `export_hf_causal_lm.py` keeps its
own CLI, now a thin wrapper around the class defined here; `export_lfm2_modular.py`/
`export_lfm2_monolithic.py` are NOT migrated to these classes this pass (BACKLOG.md's confirmed P3
scope) -- `test_causal_lm_export.py` instead reproduces LFM2's exports through these classes directly,
as the regression check that they genuinely generalize the shape.
"""
import sys
import types
from dataclasses import dataclass
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

from .export_config import LoomExportConfig
from .modular_export import ModularExportSpec, export_modular
from .register import LoomGGUFBackend


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
    """Shared base for the causal-LM family: one flattened trace (`LMMonolithicCausalModelExportConfig`)
    or independently-traced submodules assembled per `ModularExportSpec`
    (`LMModularCausalModelExportConfig`, EXPORT-ROADMAP.md R7's 'modular' profile). Qwen3 only ever needs
    Monolithic; LFM2 is the one model that exercises both."""


@dataclass(kw_only=True)
class LMMonolithicCausalModelExportConfig(LMCausalModelExportConfig):
    """Any plain `AutoModelForCausalLM`-shaped HF model -> Loom GGUF via one traced forward pass. See
    `export_hf_causal_lm.export_causal_lm`'s original module docstring (EXPORT-IMPROVEMENT-BACKLOG.md
    item 1) -- everything here is architecture-agnostic; a model needing bespoke submodule wiring wants
    `LMModularCausalModelExportConfig` instead."""

    # `LoomExportConfig.architecture` is normally required, but this family alone infers it from the
    # checkpoint's own `model.config.model_type` when not given -- matching `export_causal_lm`'s
    # original `architecture: str = None` default exactly.
    architecture: Optional[str] = None
    model_dir: str
    tokenizer_dir: Optional[str] = None
    tokenizer_family: Optional[str] = None
    tokenizer_pre: Optional[str] = None
    quantize: Optional[str] = None
    seq_len: int = 128
    max_seq_len: int = 4096

    def export(self) -> str:
        print(f"Loading model from {self.model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32).eval()
        architecture = self.architecture
        if architecture is None:
            architecture = getattr(model.config, "model_type", None)
            if not architecture:
                raise ValueError(
                    "architecture could not be inferred from model.config.model_type; pass it explicitly"
                )
        wrapper = _CausalLMWrapper(model)

        # torch.jit.trace always needs one concrete example shape -- the dynamic range is declared
        # separately below via ct.convert's own `inputs=` (EXPORT-BACKLOG.md item 3: a fixed traced shape
        # bakes a literal length into every exported slice, forcing the driver to pad every prompt to that
        # fixed length instead of using its real length).
        print(f"Tracing the complete PyTorch graph (dummy seq_len={self.seq_len})...")
        dummy_tokens = torch.zeros((1, self.seq_len), dtype=torch.long)
        dummy_cache_position = torch.arange(self.seq_len, dtype=torch.long)
        dummy_attention_mask = _causal_mask(self.seq_len)
        traced_model = torch.jit.trace(wrapper, (dummy_tokens, dummy_cache_position, dummy_attention_mask))

        # `tokens`/`cache_position`/`attention_mask` share the SAME ct.RangeDim instance so coremltools ties
        # them all to one symbolic length (they must always be called with matching lengths at runtime) --
        # see apply_monolithic_export's own auto-generation of "cache_position"-named inputs via
        # loom.range(...) and "attention_mask"-named inputs via loom.causal_mask(...).
        print(f"Compiling to GGUF ({self.profile} profile)...")
        seq_len_dim = ct.RangeDim(1, self.max_seq_len)
        mil_prog = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="tokens", shape=(1, seq_len_dim), dtype=np.int32),
                ct.TensorType(name="cache_position", shape=(seq_len_dim,), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, 1, seq_len_dim, seq_len_dim), dtype=np.float32),
            ],
            convert_to="milinternal",
            # ct.convert()'s default (compute_precision=None) FP16-casts every constant weight even for
            # convert_to="milinternal" (confirmed: coremltools' own `_need_fp16_cast_pass(None, "milinternal")`
            # returns True) -- root-caused as a real, meaningful precision bug via Conformer-CTC's own
            # multi-channel CONV_2D subsampling stage (see BACKLOG.md), but it silently applies to every model
            # this exporter has ever produced, weights included. Not specific to that one model/op.
            compute_precision=ct.precision.FLOAT32,
        )

        backend = LoomGGUFBackend()
        backend(
            mil_prog,
            output_path=self.output_path,
            architecture=architecture,
            profile=self.profile,
            tokenizer_dir=self.tokenizer_dir or self.model_dir,
            tokenizer_family=self.tokenizer_family,
            tokenizer_pre=self.tokenizer_pre,
            quantize=self.quantize,
        )
        print(f"SUCCESS! {self.profile.capitalize()} model exported cleanly to: {self.output_path}")
        return self.output_path


@dataclass(kw_only=True)
class LMModularCausalModelExportConfig(LMCausalModelExportConfig):
    """Independently-traced submodules (embedding, rotary-embedding table, each decoder layer, final
    norm, output head) assembled into one multi-Function Program per `ModularExportSpec`
    (EXPORT-IMPROVEMENT-BACKLOG.md item 2) -- LFM2's shape. See `modular_export.py`'s own module
    docstring for why this is the only split mechanism left after `profile="atomic"`'s retirement
    (EXPORT-ROADMAP.md R7)."""

    profile: str = "modular"
    model_dir: str
    modular_spec: ModularExportSpec
    tokenizer_dir: Optional[str] = None
    tokenizer_pre: Optional[str] = None
    # A dummy sequence length deliberately NOT equal to any of the model's own static dims (e.g. LFM2's
    # batch=1, hidden_size=1024, num_attention_heads=16, head_dim=64, vocab_size=65536) -- export_modular
    # marks an axis dynamic when its captured size equals this value, so a collision would wrongly mark a
    # static axis dynamic (or vice versa).
    dummy_seq_len: int = 37
    max_seq_len: int = 4096

    def export(self) -> str:
        print(f"Loading model from {self.model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.float32).eval()

        dummy_tokens = torch.zeros((1, self.dummy_seq_len), dtype=torch.long)
        dummy_cache_position = torch.arange(self.dummy_seq_len, dtype=torch.long)
        dummy_inputs = dict(
            input_ids=dummy_tokens,
            cache_position=dummy_cache_position,
            attention_mask=_causal_mask(self.dummy_seq_len),
        )

        print("Tracing each submodule standalone...")
        result = export_modular(
            model, self.modular_spec, dummy_inputs, seq_len=self.dummy_seq_len, max_seq_len=self.max_seq_len
        )

        print("Compiling to GGUF (modular-blueprint profile)...")
        backend = LoomGGUFBackend()
        # `profile` is deliberately NOT forwarded here, matching export_lfm2_modular.py's own original
        # call exactly: LoomGGUFExporter.export() dispatches on `kwargs.get("modular_layout") is not
        # None`, not on `self.profile` (exporter.py's own export() -- the modular Program has no "main"
        # function at all, so the `is_bespoke` branch never applies regardless of profile).
        backend(
            result.program,
            output_path=self.output_path,
            architecture=self.architecture,
            tokenizer_dir=self.tokenizer_dir or self.model_dir,
            tokenizer_pre=self.tokenizer_pre,
            modular_layout=result,
        )
        print(f"SUCCESS! Modular-blueprint model exported cleanly to: {self.output_path}")
        return self.output_path
