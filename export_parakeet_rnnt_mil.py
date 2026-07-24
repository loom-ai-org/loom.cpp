#!/usr/bin/env python3
"""
Export the real NVIDIA NeMo Parakeet-RNNT-0.6B checkpoint's encoder (preprocessor + FastConformer
encoder, NOT the RNNT decoder+joint -- those stay the existing hand-derived small topologies, same
"host logic, not a graph primitive" precedent as export_parakeet_tdt_mil.py) through the generic MIL
exporter. Near-identical to export_parakeet_tdt_mil.py -- same FastConformer `dw_striding` subsampling
family (depthwise + 1x1-pointwise stages only, no plain multi-channel kernel=3/stride=2 conv, so this
checkpoint is NOT expected to hit the CONV_2D bug found exporting Conformer-CTC-small's own "striding"
(non-dw) subsampling -- see BACKLOG.md), just biased/xscaled differently (xscale=32.0=sqrt(1024), unlike
Parakeet-TDT's xscale=False -- confirmed real via convert_parakeet_rnnt.py's own module docstring).

Usage:
  ~/.venvs/piper/bin/python3 export_parakeet_rnnt_mil.py
"""
import os
import sys
import types
from pathlib import Path

mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

os.environ.setdefault("TMPDIR", "/home/flavio/.claude/tmp/nemo_extract")
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # Registers the "loom" backend + applies torch-frontend patches


class _ParakeetEncoderWrapper(torch.nn.Module):
    """Same shape as export_parakeet_tdt_mil.py's own wrapper -- reduces EncDecRNNTBPEModel.forward's
    (encoded, encoded_len) pair to just `encoded`, transposed from NeMo's own (B, D, T) convention to
    (B, T, D)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, length):
        encoded, _encoded_len = self.model(input_signal=waveform, input_signal_length=length)
        return encoded.transpose(1, 2)


def main():
    import nemo.collections.asr as nemo_asr

    model_path = "/home/flavio/.claude/tmp/parakeet_rnnt_model/parakeet-rnnt-0.6b.nemo"
    print(f"Loading NeMo model from {model_path}...")
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location="cpu")
    model.eval()
    wrapper = _ParakeetEncoderWrapper(model)

    n_samples = 16000  # 1s @ 16kHz dummy trace length
    dummy_waveform = torch.randn(1, n_samples, dtype=torch.float32)
    dummy_length = torch.tensor([n_samples], dtype=torch.int64)

    print(f"Tracing the complete PyTorch graph (dummy n_samples={n_samples})...")
    traced = torch.jit.trace(wrapper, (dummy_waveform, dummy_length))

    print("Compiling to GGUF (monolithic profile)...")
    seq_dim = ct.RangeDim(1600, 16000 * 20)  # 0.1s .. 20s @ 16kHz, matching NeMo's own min/max_duration
    mil_prog = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="waveform", shape=(1, seq_dim), dtype=np.float32),
            ct.TensorType(name="length", shape=(1,), dtype=np.int32),
        ],
        convert_to="milinternal",
    )

    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        mil_prog,
        output_path="parakeet_rnnt_encoder_mil_monolithic.gguf",
        architecture="parakeet-rnnt-encoder",
        profile="monolithic",
    )
    print("SUCCESS! Monolithic encoder exported cleanly to: parakeet_rnnt_encoder_mil_monolithic.gguf")


if __name__ == "__main__":
    main()
