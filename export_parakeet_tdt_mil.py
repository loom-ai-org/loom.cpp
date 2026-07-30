#!/usr/bin/env python3
"""
Export the real NVIDIA NeMo Parakeet-TDT-0.6B-v3 checkpoint's encoder (preprocessor + FastConformer
encoder, NOT the RNNT/TDT decoder+joint -- those stay the existing hand-derived small topologies, same
"host logic, not a graph primitive" precedent as CTC greedy decode) through the generic MIL exporter,
tracing the REAL nemo.collections.asr.models.EncDecRNNTBPEModel directly via torch.jit.trace +
ct.convert, not a hand-reimplemented plain-PyTorch module.

Everything shared with the other two NeMo ASR encoder exports lives in
tools/loom_mil_compiler/nemo_asr_export.py, the family template -- including why `ENCODER_BT_D` stops
at the encoder (the TDT prediction network is an autoregressive LSTM + joint, driven host-side by the
C++ TdtDecoder) and why the output is transposed from NeMo's (B, D, T) to this project's (B, T, D)
convention. The claim is checked against this checkpoint's own cfg.encoder.d_model while tracing.

Usage:
  ~/.venvs/piper/bin/python3 export_parakeet_tdt_mil.py
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
    checkpoint="/home/flavio/Dev/models/parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo",
    output=EncoderOutput.ENCODER_BT_D,
    architecture="parakeet-tdt-encoder",
    output_path="parakeet_tdt_encoder_mil_monolithic.gguf",
)


if __name__ == "__main__":
    export_nemo_asr_encoder(SPEC)
