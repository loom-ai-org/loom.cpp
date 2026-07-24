#!/usr/bin/env python3
"""
Export the real NVIDIA NeMo Parakeet-TDT-0.6B-v3 checkpoint's encoder (preprocessor + FastConformer
encoder, NOT the RNNT/TDT decoder+joint -- those stay the existing hand-derived small topologies, same
"host logic, not a graph primitive" precedent as CTC greedy decode) through the generic MIL exporter,
mirroring export_conformer_ctc_mil.py's approach: trace the REAL
nemo.collections.asr.models.EncDecRNNTBPEModel directly via torch.jit.trace + ct.convert, not a
hand-reimplemented plain-PyTorch module.

Usage:
  ~/.venvs/piper/bin/python3 export_parakeet_tdt_mil.py
"""
import os
import sys
import types
from pathlib import Path

# Bypass the transformers library hf-hub bounds check (same as export_conformer_ctc_mil.py) -- needed
# because nemo.collections.asr eagerly imports transformers transitively via torchmetrics.
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

# NeMo's restore_from() extracts the (multi-GB) .nemo tarball into a tempfile.mkdtemp() dir -- default
# TMPDIR (/tmp) is too small on this machine (see env_disk_space_tmp memory), route it to /home instead.
os.environ.setdefault("TMPDIR", "/home/flavio/.claude/tmp/nemo_extract")
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # Registers the "loom" backend + applies torch-frontend patches


class _ParakeetEncoderWrapper(torch.nn.Module):
    """Reduces EncDecRNNTBPEModel.forward's (encoded, encoded_len) pair to just `encoded`, transposed
    from NeMo's own (B, D, T) convention to (B, T, D) -- matching every other model in this project's own
    ne[0]=feature/ne[1]=time GGUF convention. The RNNT/TDT decoder (2-layer LSTM prediction net) + joint
    network + greedy search loop are NOT traced here -- they stay the existing hand-derived small
    topologies (tools/convert_nemo/convert_parakeet_tdt.py's build_lstm_topology/build_joint_topology),
    driven autoregressively by the existing C++ TdtDecoder (include/loom/core/tdt_decoder.h), same
    "encoder graph vs. host-side autoregressive loop" split as every other ASR/LLM model in this project."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, length):
        encoded, _encoded_len = self.model(input_signal=waveform, input_signal_length=length)
        return encoded.transpose(1, 2)


def main():
    import nemo.collections.asr as nemo_asr

    model_path = "/home/flavio/.claude/tmp/parakeet_tdt_model/parakeet-tdt-0.6b-v3.nemo"
    print(f"Loading NeMo model from {model_path}...")
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location="cpu")
    model.eval()
    wrapper = _ParakeetEncoderWrapper(model)

    n_samples = 16000  # 1s @ 16kHz dummy trace length -- matches the existing bespoke fixture's default
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
        output_path="parakeet_tdt_encoder_mil_monolithic.gguf",
        architecture="parakeet-tdt-encoder",
        profile="monolithic",
    )
    print("SUCCESS! Monolithic encoder exported cleanly to: parakeet_tdt_encoder_mil_monolithic.gguf")


if __name__ == "__main__":
    main()
