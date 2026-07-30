#!/usr/bin/env python3
"""
Export the real NVIDIA NeMo Parakeet-RNNT-0.6B checkpoint's encoder (preprocessor + FastConformer
encoder, NOT the RNNT decoder+joint -- those stay the existing hand-derived small topologies, same
"host logic, not a graph primitive" precedent as export_parakeet_tdt_mil.py) through the generic MIL
exporter. Same FastConformer `dw_striding` subsampling family as Parakeet-TDT (depthwise +
1x1-pointwise stages only, no plain multi-channel kernel=3/stride=2 conv, so this checkpoint is NOT
expected to hit the CONV_2D bug found exporting Conformer-CTC-small's own "striding" (non-dw)
subsampling -- see BACKLOG.md), just biased/xscaled differently (xscale=32.0=sqrt(1024), unlike
Parakeet-TDT's xscale=False -- confirmed real via convert_parakeet_rnnt.py's own module docstring).

Everything shared with the other two NeMo ASR encoder exports lives in
tools/loom_mil_compiler/nemo_asr_export.py, the family template.

Usage:
  ~/.venvs/piper/bin/python3 export_parakeet_rnnt_mil.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from loom_mil_compiler.nemo_asr_export import (  # noqa: E402  (path setup must precede the import)
    EncoderOutput,
    NeMoASREncoderSpec,
    export_nemo_asr_encoder,
)

SPEC = NeMoASREncoderSpec(
    checkpoint="/home/flavio/Dev/models/parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo",
    output=EncoderOutput.ENCODER_BT_D,
    architecture="parakeet-rnnt-encoder",
    output_path="parakeet_rnnt_encoder_mil_monolithic.gguf",
)


if __name__ == "__main__":
    export_nemo_asr_encoder(SPEC)
