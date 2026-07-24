#!/usr/bin/env python3
"""
Generic export driver: any plain `AutoModelForCausalLM`-shaped HF model -> Loom GGUF.

Everything here is architecture-agnostic (load -> trace -> ct.convert -> LoomGGUFExporter). A model
needing bespoke submodule wiring (e.g. a hand-built multi-Function Program) still needs its own script,
but the ordinary "one traced forward pass, one attention-mask/cache_position dynamic seq-len axis" shape
covers most causal LMs including LFM2 -- see EXPORT-IMPROVEMENT-BACKLOG.md item 1.

The tokenizer family ("bpe"/"wordpiece"/"sentencepiece_proto") and, for "bpe", the pretokenizer regex
shape (`tokenizer.ggml.pre`) are auto-detected from the real HF tokenizer directory by default (see
tokenizer_detect.py) -- `--tokenizer-family`/`--tokenizer-pre` are optional overrides, only needed if
auto-detection raises (an unrecognized tokenizer hash, or a recognized-but-not-yet-implemented family).

Usage:
  ~/.venvs/piper/bin/python3 -m tools.loom_mil_compiler.export_hf_causal_lm \\
      /path/to/hf/model --profile atomic --architecture lfm2 --output model.gguf

  # Override auto-detection explicitly (e.g. if the tokenizer predates tokenizer_detect.py's hash table):
  ~/.venvs/piper/bin/python3 -m tools.loom_mil_compiler.export_hf_causal_lm \\
      /path/to/hf/model --profile atomic --tokenizer-pre llama3 --architecture lfm2 \\
      --output model.gguf
"""
import argparse
import sys
import types
from pathlib import Path

# Bypass the transformers library hf-hub bounds check to import safely
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loom_mil_compiler  # Registers the "loom" backend + applies torch-frontend patches


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


def export_causal_lm(
    model_dir: str,
    output_path: str,
    *,
    profile: str = "monolithic",
    architecture: str = None,
    tokenizer_dir: str = None,
    tokenizer_family: str = None,
    tokenizer_pre: str = None,
    quantize: str = None,
    seq_len: int = 128,
    max_seq_len: int = 4096,
) -> str:
    """Loads a plain HF causal-LM from `model_dir`, traces it, and exports it to Loom GGUF at
    `output_path` under the given profile ("monolithic" or "atomic"). Returns `output_path`."""
    print(f"Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    if architecture is None:
        architecture = getattr(model.config, "model_type", None)
        if not architecture:
            raise ValueError("architecture could not be inferred from model.config.model_type; pass it explicitly")
    wrapper = _CausalLMWrapper(model)

    # torch.jit.trace always needs one concrete example shape -- the dynamic range is declared
    # separately below via ct.convert's own `inputs=` (EXPORT-BACKLOG.md item 3: a fixed traced shape
    # bakes a literal length into every exported slice, forcing the driver to pad every prompt to that
    # fixed length instead of using its real length).
    print(f"Tracing the complete PyTorch graph (dummy seq_len={seq_len})...")
    dummy_tokens = torch.zeros((1, seq_len), dtype=torch.long)
    dummy_cache_position = torch.arange(seq_len, dtype=torch.long)
    dummy_attention_mask = _causal_mask(seq_len)
    traced_model = torch.jit.trace(wrapper, (dummy_tokens, dummy_cache_position, dummy_attention_mask))

    # `tokens`/`cache_position`/`attention_mask` share the SAME ct.RangeDim instance so coremltools ties
    # them all to one symbolic length (they must always be called with matching lengths at runtime) --
    # see apply_monolithic_export/apply_atomic_export's own auto-generation of "cache_position"-named
    # inputs via loom.range(...) and "attention_mask"-named inputs via loom.causal_mask(...).
    print(f"Compiling to GGUF ({profile} profile)...")
    seq_len_dim = ct.RangeDim(1, max_seq_len)
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

    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        mil_prog,
        output_path=output_path,
        architecture=architecture,
        profile=profile,
        tokenizer_dir=tokenizer_dir or model_dir,
        tokenizer_family=tokenizer_family,
        tokenizer_pre=tokenizer_pre,
        quantize=quantize,
    )
    print(f"SUCCESS! {profile.capitalize()} model exported cleanly to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Path to a local HF AutoModelForCausalLM checkpoint directory")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF path")
    parser.add_argument("--profile", choices=["monolithic", "atomic"], default="monolithic")
    parser.add_argument("--architecture", default=None, help="Defaults to model.config.model_type")
    parser.add_argument("--tokenizer-dir", default=None, help="Defaults to model_dir")
    parser.add_argument("--tokenizer-family", default=None,
                         choices=["bpe", "wordpiece", "sentencepiece_proto", "byte"],
                         help="Overrides auto-detection (see tokenizer_detect.detect_vocab_family)")
    parser.add_argument("--tokenizer-pre", default=None,
                         help="Overrides auto-detection for the 'bpe' family (see tokenizer_detect.detect_loom_pre_type)")
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--seq-len", type=int, default=128, help="Dummy trace sequence length")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Upper bound for the dynamic seq-len RangeDim")
    args = parser.parse_args()

    export_causal_lm(
        args.model_dir,
        args.output,
        profile=args.profile,
        architecture=args.architecture,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_family=args.tokenizer_family,
        tokenizer_pre=args.tokenizer_pre,
        quantize=args.quantize,
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
