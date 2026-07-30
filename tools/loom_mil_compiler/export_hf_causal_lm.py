#!/usr/bin/env python3
"""
Generic export driver: any plain `AutoModelForCausalLM`-shaped HF model -> Loom GGUF.

The actual mechanics (load -> trace -> ct.convert -> LoomGGUFExporter) live in
`causal_lm_export.LMMonolithicCausalModelExportConfig` now (BACKLOG.md P3.1) -- this module is a thin
CLI/function shim over that class, kept for its existing entry point:

  ~/.venvs/piper/bin/python3 -m tools.loom_mil_compiler.export_hf_causal_lm \\
      /path/to/hf/model --architecture lfm2 --output model.gguf

The tokenizer family ("bpe"/"wordpiece"/"sentencepiece_proto") and, for "bpe", the pretokenizer regex
shape (`tokenizer.ggml.pre`) are auto-detected from the real HF tokenizer directory by default (see
tokenizer_detect.py) -- `--tokenizer-family`/`--tokenizer-pre` are optional overrides, only needed if
auto-detection raises (an unrecognized tokenizer hash, or a recognized-but-not-yet-implemented family).
"""
import argparse

from .causal_lm_export import LMMonolithicCausalModelExportConfig


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
    `output_path` under the given profile ("monolithic" or "submodule"). Returns `output_path`."""
    return LMMonolithicCausalModelExportConfig(
        architecture=architecture,
        output_path=output_path,
        profile=profile,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer_family=tokenizer_family,
        tokenizer_pre=tokenizer_pre,
        quantize=quantize,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
    ).export()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Path to a local HF AutoModelForCausalLM checkpoint directory")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF path")
    parser.add_argument("--profile", choices=["monolithic"], default="monolithic")
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
