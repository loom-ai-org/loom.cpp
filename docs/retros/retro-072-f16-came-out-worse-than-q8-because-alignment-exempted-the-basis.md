---
type: retro
date: 2026-10-03
domain: exporter
tags: [exporter, quantization, f16, dft, front-end, wakehubert, family-13, oracle]
---

# Retro-072: F16 Came Out Worse Than Q8_0, Because Alignment Exempted the Basis

## The Issue

WakeHuBERT tiny ([Epic-03](../epics/epic-03-model-coverage.md#family-13-small-audio-classifiers-and-embedders))
was exported at F32, F16, Q8_0 and Q4_0 and each was compared per 20 ms frame against upstream's
PyTorch model at f64, on jfk.wav. The F16 file was the second-worst of the four: worst-frame cosine
**0.976**, where Q8_0 -- eight bits on every packed weight, against F16's eleven -- held **0.9998**.

## Root Cause

**The two precisions did not pack the same tensors.** The model's log-mel front end is a DFT done as a
`CONV_1D` with a fixed `[402, 1, 400]` basis, and a conv kernel is eligible by op. Whether an eligible
weight is then packed depends on its fastest axis dividing the block size: 400 does not divide by
Q8_0's 32, so Q8_0 **declined the basis for shape** and left it F32; F16's block size is 1, so F16
packed it. Upstream's own int8 ONNX keeps the front end in float as well.

The basis is the one weight this model cannot spare precision in. The log in `log(mel @ |X|^2 + 1e-6)`
turns an absolute error in a near-empty high-frequency bin of a LOUD frame into a large relative one,
and a mantissa of 11 bits on the basis puts that error well above f32's. Exempting the basis alone
took F16 to a worst frame of **0.999998**. The quantization report had said nothing unusual: "18
tensor(s) quantized to F16", which was true.

## The Fix

`keep_float`: an export config names the weights it keeps at F32 at every `--quantize`, and the packer
honours it (and does not fold such a kernel). A name the export does not write is an error, so a
renamed tensor cannot fall out of its exemption silently. WakeHuBERT keeps its DFT basis and mel
matrix. Pinned by `tests/ci/test_quantize_export.py::TestAFamilyCanKeepAWeightAtF32` and a toy
WakeHuBERT export at F16 in `test_audio_classification_export.py`; turning the exemption off turns
three tests red.

The F32 file's own gap -- 3.4e-5 against torch's f32-vs-f64 spread of 7.3e-6 -- was chased to the
same front end and is NOT a defect: an f32 DFT accumulated in eight SIMD lanes, which is ggml's order,
lands at 3.05e-5 (median 7.31e-6 against loom's 7.26e-6). Summation order alone moves the log-mel's
worst bin between 1.2e-4 and 1.2e-3 in f32.

## Takeaway

**A lower precision is not a subset of a higher one's damage: which tensors get packed differs by
type.** Block alignment is an accidental exemption list, and F16 -- block size 1 -- has none, so it can
reach a sensitive tensor that every block type left alone. Measure each precision you ship against
the reference, not just the coarsest, and when one is worse than a coarser one, diff the two files'
tensor types before anything else. A fixed signal-processing constant (a DFT basis, a filterbank)
feeding a log or a division is the first suspect.
