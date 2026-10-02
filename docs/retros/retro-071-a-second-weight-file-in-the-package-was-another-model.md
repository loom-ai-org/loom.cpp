---
type: retro
date: 2026-10-02
domain: exporter
tags: [exporter, checkpoint, weights, silero, vad, family-13, oracle]
---

# Retro-071: A Second Weight File in the Package Was Another Model

## The Issue

The `silero-vad` 6.2.3 wheel ships the 16 kHz model three ways: `silero_vad.jit` (what
`load_silero_vad()` returns), several ONNX files, and `silero_vad_16k.safetensors` beside a 40-line
tinygrad definition. The safetensors and the plain-Python definition were the obvious export source --
no TorchScript to unpick. The first oracle asserted they equalled the JIT's weights, and failed.

## Root Cause

**Every tensor but the STFT basis differs** (conv4's weight by 18.2, the LSTM's by ~1). The safetensors
is a different training snapshot. The JIT and the default `silero_vad.onnx` carry identical numbers;
the README offers the safetensors "for simplicity" beside the tinygrad example, and nothing says it is
the same model. An export from it would have run, produced plausible speech probabilities, and been
compared against nothing that could tell.

## The Fix

The exporter loads `silero_vad.jit` and reads the 16 kHz weights out of its state dict (names checked);
the architecture is the tinygrad file's, rewritten for a whole clip, and the loader runs the JIT itself,
streamed frame by frame, against the rewrite before tracing. A torch port of the definition fed the
JIT's weights is bit-identical to the JIT (max |Δ| 0) on two real clips.

## Takeaway

**When a release carries the same model in several files, the one its loader returns is the reference,
and every other file is checked against it before it is used.** A convenience copy -- a safetensors
for a different framework, a quantized ONNX, a "simple" example's weights -- can be a different
snapshot, and nothing downstream of the export can notice: the model is plausible either way. Compare
weights, not outputs; it costs one assertion.
