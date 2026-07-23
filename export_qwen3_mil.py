#!/usr/bin/env python3
"""
Exports Qwen3-0.6B-Base as a Monolithic GGUF model via the generic MIL exporter, mirroring
export_lfm2_monolithic.py's shape. Prototype for BACKLOG.md's "retrofit the bespoke tools/convert_*
scripts onto the MIL exporter" item -- Qwen3 was picked first since it's HF AutoModelForCausalLM-shaped,
same category as LFM2 (also GQA + tied embeddings), so it should need no bespoke wrapper code at all.

Usage:
  ~/.venvs/piper/bin/python3 export_qwen3_mil.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from loom_mil_compiler.export_hf_causal_lm import export_causal_lm


def main():
    export_causal_lm(
        model_dir="/home/flavio/Dev/models/qwen3-0.6b-base",
        output_path="qwen3_0.6b_mil_monolithic.gguf",
        profile="monolithic",
        architecture="qwen3",
    )


if __name__ == "__main__":
    main()
