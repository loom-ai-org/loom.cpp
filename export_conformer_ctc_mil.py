#!/usr/bin/env python3
"""
Export the real NVIDIA NeMo Conformer-CTC-small checkpoint through the generic MIL exporter, tracing the
REAL nemo.collections.asr.models.EncDecCTCModelBPE (preprocessor + ConformerEncoder + ConvASRDecoder)
directly via torch.jit.trace + ct.convert -- not a hand-reimplemented plain-PyTorch module (unlike
tools/convert_generic/conformer_ctc_module.py's older aten_to_loom-oriented POC, which needed a custom
`loom::rel_pos_attention` op precisely because that converter requires the source nn.Module to call a
custom op; the MIL exporter has no such requirement -- it walks whatever real ops a model decomposes
into under tracing).

Everything this shares with export_parakeet_tdt_mil.py / export_parakeet_rnnt_mil.py -- the transformers
version-gate stub, the TMPDIR routing, the wrapper reducing NeMo's output tuple to one tensor, the
sample-rate-derived trace length and dynamic-axis bounds, and the load-bearing
compute_precision=FLOAT32 -- lives in tools/loom_mil_compiler/nemo_asr_export.py, the family template.
`output=CTC_LOG_PROBS` is checked against this checkpoint's own decoder.num_classes_with_blank while
tracing, so pointing this spec at a non-CTC checkpoint raises instead of exporting the wrong tensor.

Usage:
  ~/.venvs/piper/bin/python3 export_conformer_ctc_mil.py
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
    checkpoint="/home/flavio/Dev/models/conformer-ctc-small/stt_en_conformer_ctc_small.nemo",
    output=EncoderOutput.CTC_LOG_PROBS,
    architecture="conformer-ctc",
    output_path="conformer_ctc_small_mil_monolithic.gguf",
)


if __name__ == "__main__":
    export_nemo_asr_encoder(SPEC)
