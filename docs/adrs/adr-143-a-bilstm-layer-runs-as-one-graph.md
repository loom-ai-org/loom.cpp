---
type: adr
status: proposed
date: 2026-11-27
domain: edge-performance
tags: [performance, lstm, recurrent, rawi]
supersedes: []
---

# ADR-143: A BiLSTM Layer Runs as One Graph

Serves [Epic-05](../epics/epic-05-edge-performance.md). Follows
[ADR-137](adr-137-a-bilstms-input-half-is-one-product-before-the-sweep.md).

## Context

`run_bi_recurrent_and_retain` computes one cell graph per timestep and direction, with h and c on the
host. For a short sequence with a small hidden size, the dispatch of each step is a large part of the
time. The rawi Arabic diacritization ensemble has two networks, two BiLSTM layers each, hidden 256, and
30-60 characters per sentence (loom.cpp#143).

## Decision

1. **`loom.run_bi_lstm_and_retain`** takes the arguments of `run_bi_recurrent_and_retain` and leaves the
   same retained result in either layout. It builds one graph for the layer: `W_ih x + b` as one product
   per direction over the whole sequence, then every step unrolled, with h and c in backend memory.
2. The weights and the gate order come from the existing cell topology (`MUL_MAT` on `layer_input` and on
   `h_prev`, the bias `ADD`, four gate `VIEW`s). The export and the GGUF do not change; a driver changes
   only the function name. A cell of another shape is refused with a message that names the per-step
   binding.
3. A new name, as in ADR-137: a driver that needs it fails on an older engine instead of running slowly.

## Consequences

* rawi ensemble, 934 sentences, dev laptop, CPU, one binary, both drivers run twice in turn: median
  per sentence 14.3 -> 11.1 ms (F16) and 17.6 -> 10.7 ms (F32). onnxruntime on the same machine, one
  thread: 5.0 ms fp32, 1.4 ms int8. The argmax equals fp32 ONNX on all 24 477 characters; max |d|
  against the per-step path is 3.8e-6 (F32).
* The graph has about 16 nodes per step and direction and is rebuilt for each call. Very long sequences
  make a large graph; the per-step binding stays for them and for GRU cells.
* Tests: `test_bilstm_fused` checks two stacked layers (exporter gate order, then torch gate order, the
  second read by reference) against a float64 numpy BiLSTM in both layouts and against the per-step
  binding, and checks the refusal.
* loom-exporter still emits `run_bi_recurrent_and_retain`. Emitting the new call is a separate change.
