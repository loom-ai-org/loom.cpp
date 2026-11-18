---
type: adr
status: accepted
date: 2026-11-18
domain: edge-performance
tags: [performance, lstm, recurrent, exporter, kokoro, styletts2, paradee, raspberry-pi]
supersedes: []
---

# ADR-137: A BiLSTM's Input Half Is One Product Before the Sweep

Serves [Epic-05](../epics/epic-05-edge-performance.md) and [Epic-02](../epics/epic-02-mil-exporter-and-compiler.md).
Follows [ADR-136](adr-136-a-recurrent-sweep-runs-its-cells-on-one-thread.md), which made the sweep's
per-timestep graphs cheap enough for the work inside them to show.

## Context

An LSTM cell computes `W_ih x_t + W_hh h_(t-1) + b`. Only the second product depends on the previous step,
but the exported cell did both, one matrix-vector product each, at every timestep of every direction.
Measured per node on the Pi 4 after ADR-136, the input product (`W_ih x_t`) was 70 ms of a 755 ms Paradee
synthesis (input 192-224, 4H = 384, ~1 480 steps), and 0.98 s of Kokoro's 18.2 s (input 640, 4H = 1024).

## Decision

1. **`RecurrentPhase(project_input=True)`** emits `<name>_input`. It takes the whole sequence (`[I, T]`,
   `n_tokens`) and outputs `W_ih x + b` per direction (`[4H, T]`, forward then backward), one product each.
   The cells become *projected*: `layer_input` is the gate-block row and the cell adds only `W_hh h`. The
   sum becomes `(W_ih x + b) + W_hh h`: the same terms in another order, so float32 rounding moves.
2. **`run_bi_lstm`** runs the projection, then `loom.run_bi_recurrent_pair_and_retain`, a new engine
   binding taking one sequence per direction. It is a new name rather than an optional argument, so a
   driver that needs it fails on an older engine (`attempt to call a nil value`) instead of feeding the
   forward projection to both directions. Model cards state the 1.0.0 requirement
   (`_sweeps_projected_lstms`).
3. **Opt-in per phase**, on for the six prosody BiLSTMs Kokoro, StyleTTS2 and Paradee share
   (`kokoro_export.build_prosody_phases`). Off for the rest. A stack's later layers have no input before
   the sweep, so `project_input` is refused there. GRUs (AuroraTTS) and pyannote's and EnCodec's LSTMs
   use other driver components, and Silero's streaming LSTM runs one step per call, where a projection
   would only add a graph.
4. The library's helper declarations (`DrivenTopologies.more`) now allow one helper to drive topologies
   with different inputs (the cells and the projection), and the drives check compares each call's
   input table with its own group.

## Consequences

* Paradee on the Pi 4: F32 0.764 -> 0.709 s, Q8_0_PC 0.679 -> 0.667 s, so 1.05x and 1.22x onnxruntime
  ([BENCHMARKS](../BENCHMARKS.md#raspberry-pi-4), footnote 6). Kokoro on the Pi 4, the same export
  projected against unprojected: 18.24 -> 17.35 s (1.05x, 3 paired rounds). Dev box: Kokoro 7.5 -> 6.8 s,
  Paradee 283 -> 261 ms.
* Outputs move by rounding only. Kokoro's audio is 89.8 dB from the unprojected export (the 16-bit WAV's
  floor), with the same length. Paradee's gate is 63.6-67.2 dB from upstream, as before, and its WER is
  unchanged at both precisions. StyleTTS2's driver gate RMS moved by 2e-5 relative.
* Tests: `test_recurrent.py` checks a projected LSTM against `nn.LSTM` in both directions (a sabotaged
  bias fails it). `test_gru_sweep` checks the paired binding against a float64 reference. A sabotaged
  backward sequence fails it. Kokoro's and StyleTTS2's driver gates accept 0 or 6 projections.
* Published Kokoro, StyleTTS2 and Paradee files keep their old drivers and still run. A re-export
  needs 1.0.0.
