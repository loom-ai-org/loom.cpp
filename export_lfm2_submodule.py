#!/usr/bin/env python3
"""
Exports LFM2-350M via the submodule-export blueprint (EXPORT-IMPROVEMENT-BACKLOG.md item 2): each
submodule (embedding, rotary-embedding table, each decoder layer, final norm, output head) is traced
independently via its own real captured forward call, instead of scope-partitioning one flattened
trace the way export_lfm2_atomic.py does. Kept as its own script (separate from export_lfm2_atomic.py)
until this path is verified numerically, per the backlog's own "keep the existing apply_atomic_export
available... don't delete the fallback until parity is proven."

Usage:
  ~/.venvs/piper/bin/python3 export_lfm2_submodule.py
"""
import sys
import types
from pathlib import Path

mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # Registers the "loom" backend + applies torch-frontend patches
from loom_mil_compiler.submodule_export import SubmoduleExportSpec, export_submodules


def _causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def main():
    model_dir = "/home/flavio/Dev/models/lfm2-350m"
    out_path = "lfm2_350m_submodule.gguf"

    print(f"Loading LFM2-350M from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

    # A dummy sequence length deliberately NOT equal to any of the model's own static dims (batch=1,
    # hidden_size=1024, num_attention_heads=16, head_dim=64, vocab_size=65536) -- export_submodules
    # marks an axis dynamic when its captured size equals this value, so a collision would wrongly
    # mark a static axis dynamic (or vice versa).
    seq_len = 37
    dummy_tokens = torch.zeros((1, seq_len), dtype=torch.long)
    dummy_cache_position = torch.arange(seq_len, dtype=torch.long)
    dummy_inputs = dict(
        input_ids=dummy_tokens,
        cache_position=dummy_cache_position,
        attention_mask=_causal_mask(seq_len),
    )

    spec = SubmoduleExportSpec(
        prefix_attr="model.embed_tokens",
        repeated_attr="model.layers",
        suffix_attrs=["model.embedding_norm", "lm_head"],
        aux_attr="model.pos_emb",
        aux_kwarg="position_embeddings",
    )

    print("Tracing each submodule standalone...")
    result = export_submodules(model, spec, dummy_inputs, seq_len=seq_len, max_seq_len=4096)

    print("Compiling to GGUF (submodule-blueprint profile)...")
    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        result.program,
        output_path=out_path,
        architecture="lfm2",
        tokenizer_dir=model_dir,
        tokenizer_pre="llama3",
        submodule_layout=result,
    )
    print(f"SUCCESS! Submodule-blueprint model exported cleanly to: {out_path}")


if __name__ == "__main__":
    main()
