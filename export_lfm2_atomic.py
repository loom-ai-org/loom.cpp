#!/usr/bin/env python3
"""
Exports LFM2-350M as an Automatically Partitioned "Atomic" GGUF model.

Usage:
  ~/.venvs/piper/bin/python3 export_lfm2_atomic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from loom_mil_compiler.export_hf_causal_lm import export_causal_lm


def main():
    export_causal_lm(
        model_dir="/home/flavio/Dev/models/lfm2-350m",
        output_path="lfm2_350m_atomic.gguf",
        profile="atomic",
        architecture="lfm2",
        tokenizer_pre="llama3",
    )


if __name__ == "__main__":
    main()
