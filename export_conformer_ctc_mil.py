#!/usr/bin/env python3
"""
Prototype: export the real NVIDIA NeMo Conformer-CTC-small checkpoint through the generic MIL exporter,
tracing the REAL nemo.collections.asr.models.EncDecCTCModelBPE (preprocessor + ConformerEncoder +
ConvASRDecoder) directly via torch.jit.trace + ct.convert -- not a hand-reimplemented plain-PyTorch
module (unlike tools/convert_generic/conformer_ctc_module.py's older aten_to_loom-oriented POC, which
needed a custom `loom::rel_pos_attention` op precisely because that converter requires the source
nn.Module to call a custom op; the MIL exporter has no such requirement -- it walks whatever real ops a
model decomposes into under tracing).

Usage:
  ~/.venvs/piper/bin/python3 export_conformer_ctc_mil.py
"""
import sys
import types
from pathlib import Path

# Bypass the transformers library hf-hub bounds check (same as export_hf_causal_lm.py) -- needed
# because nemo.collections.asr eagerly imports transformers transitively via torchmetrics.
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # Registers the "loom" backend + applies torch-frontend patches


class _ConformerCTCWrapper(torch.nn.Module):
    """Reduces EncDecCTCModelBPE.forward's (log_probs, encoded_len, greedy_predictions) 3-tuple to just
    log_probs -- CTC greedy decode + detokenization happen host-side (loom::ctc_greedy_decode +
    loom::Vocab), same "host logic, not a graph primitive" precedent as every other model here."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, length):
        log_probs, _encoded_len, _greedy = self.model(input_signal=waveform, input_signal_length=length)
        return log_probs


def main():
    import nemo.collections.asr as nemo_asr

    model_path = "/home/flavio/Dev/models/conformer-ctc-small/stt_en_conformer_ctc_small.nemo"
    print(f"Loading NeMo model from {model_path}...")
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path, map_location="cpu")
    model.eval()
    wrapper = _ConformerCTCWrapper(model)

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
        output_path="conformer_ctc_small_mil_monolithic.gguf",
        architecture="conformer-ctc",
        profile="monolithic",
    )
    print("SUCCESS! Monolithic model exported cleanly to: conformer_ctc_small_mil_monolithic.gguf")


if __name__ == "__main__":
    main()
