---
type: adr
status: accepted
date: 2026-11-27
domain: edge-performance
tags: [performance, lstm, recurrent, exporter, rawi, pyannote]
supersedes: []
---

# ADR-144: A BiLSTM Stack Swept a Layer at a Time Projects Every Layer

Serves [Epic-05](../epics/epic-05-edge-performance.md) and [Epic-02](../epics/epic-02-mil-exporter-and-compiler.md).
Extends [ADR-137](adr-137-a-bilstms-input-half-is-one-product-before-the-sweep.md) to `BiRecurrentCall`.
Found by the rawi Arabic diacritization ensemble (#143).

## Context

rawi is two character BiLSTM networks: embedding 128, then two layers of hidden 256 per direction. It
exports with `RecurrentPhase(number_layers=True)` and one `BiRecurrentCall` per layer, which emits
`loom.run_bi_recurrent_and_retain`. Each step of that sweep runs a cell that computes `W_ih x_t` and
`W_hh h`, each as a matrix-vector product. `W_ih` is 56% of rawi's multiply-adds: 131k of 393k per
step in layer 0, and 524k of 786k in layer 1, where the input is the previous layer's 512-wide rows.

ADR-137 already takes `W_ih x + b` out of the sweep as one product per direction. It was refused on a
stack, because "only layer 0's input exists before the sweep". That holds for a driver that steps the
whole stack per timestep (a transducer's prediction network). It does not hold for `BiRecurrentCall`.
There each layer is its own call, and its input, the previous layer's retained `[h_fwd | h_bwd]` rows,
exists before that layer's sweep starts.

#144 proposed `loom.run_bi_lstm_and_retain` instead: a binding that recognizes the exporter's LSTM cell
from its nodes and builds the whole layer as one graph, written by hand in C++.

## Decision

1. **`RecurrentPhase(project_input=True)` accepts a stack.** Each layer gets `<stem>_l<k>_input` at its
   own input width. A driver that never calls an `_input` topology fails the export at
   `check_subgraph_calls`, so a stepped stack cannot set it by mistake. GRUs are still refused.
2. **`BiRecurrentCall(input_topology=...)`** emits the layer's projection as a retained
   `run_subgraph_and_retain` over the layer's sequence. It then emits
   `loom.run_bi_recurrent_pair_and_retain` over the projection's two outputs, at width `4 * hidden_dim`.
   The projection is an ordinary `SubgraphCall`, so the driver IR checks its input and the sweep's
   references. The paired binding joins `RECURRENT_RETAIN_FNS`, so a reader of the forward store is
   checked as it is for the unpaired binding.
3. **Opt-in per export**, as in ADR-137. Without `input_topology` the component emits exactly what it
   did before. Of the two users in the zoo, pyannote opts in (below). AuroraTTS's BiGRU is a GRU, so it
   cannot opt in and its export is unchanged.
4. **No engine change.** The paired binding shipped in 1.0.0, so a projected export runs on the released
   engine. The card floor is already right: `_sweeps_projected_lstms` maps that binding to 1.0.0.

## Measurements

rawi ensemble, 934 gold sentences (24,477 characters). AMD Ryzen 3 3250U, CPU. One binary, the #144
branch, which also has both existing bindings. Three exports of the same weights:

* per-step: the plain export;
* projected: this ADR;
* fused: the plain export with its calls renamed to #144's binding. It is byte-identical to the
  published `rawi_ensemble.*.gguf`.

The runs were interleaved, three rounds each, at both thread counts this machine has cores for.
`LOOM_N_THREADS` sets the threads for the graphs outside the sweeps (embedding, projection, head). The
sweeps and the fused binding run on one thread either way (ADR-136). Every value is the median per
sentence, and the median of the three rounds.

| | per-step | projected | fused (#144) |
| :--- | ---: | ---: | ---: |
| F32, 1 thread | 22.2 ms | 17.9 ms (1.24x) | 18.9 ms (1.17x) |
| F32, 2 threads (the default here) | 23.0 ms | 19.7 ms (1.17x) | 20.5 ms (1.12x) |
| F16, 1 thread | 36.6 ms | 26.1 ms (1.40x) | 26.6 ms (1.38x) |
| F16, 2 threads | 38.9 ms | 27.9 ms (1.39x) | 27.7 ms (1.40x) |
| argmax ≠ fp32 ONNX, F32 / F16 | 0 / 0 | 0 / 2 | 0 / 2 |

At both precisions and both thread counts, the projected and fused outputs are **bit-identical**. Both
compute `(W_ih x + b) + W_hh h` with the same ggml ops. The two F16 differences are rounding from the F16
weights. The per-step path has none at F16 because it sums the terms in another order.

Per sentence and paired within a round, projected over fused is 0.96-0.97 at F32 and 0.97-0.99 at F16
on one thread. On two threads it is 0.88-1.00 at F32 and 0.99-1.02 at F16, which is noise. Two earlier
runs on two threads had fused 2-4% ahead (22.6 against 23.3 ms, then 21.5 against 21.9, at F32). Three
interleaved rounds did not reproduce that.

**Where the time goes.** A sentence is about 26 characters, so about 210 cell steps. With cycle-counter
timers in `cell_sweep`, one step of the projected sweep costs about 65 µs. About 3.3 µs of that is host
work around the compute: the graph lookup (1.0 µs), the row copy (1.5), setting `h` and `c` (0.3),
reading them back (0.17), and the sink (0.27). A prototype loop that builds the cell graph once per
sweep and moves one row view instead of making a view per step gave bit-identical output and no
measurable change. What does move every variant is the thread count. All three are 3-9% faster on one
thread: rawi's graphs outside the sweeps are small (26 rows), and two threads cost them more than they
save. A persistent CPU thread pool (a prototype, poll 0 and 50) did not recover any of it, at 0.99-1.02x
on two threads, so the cost is not creating the threads. It is also why projected leads only on one
thread: its projection is a graph of its own on `LOOM_N_THREADS` threads, where the fused binding does
the same products inside its one-thread graph.

## Options not taken

* **#144's fused binding.** It is not faster than the projected sweep: within noise on two threads,
  2-4% slower on one. The cost of it is an LSTM implemented by hand in the engine, the third path after
  `BiLstmStepper` and `cell_sweep`. It is recognized by matching nodes, so a cell with an extra node
  would still match and run without it. It bypasses the module's scheduler, so a GPU primary gets no CPU
  fallback and `LOOM_PROFILE` cannot see the graph. And it rebuilds and reallocates a graph of about
  40·T nodes on every call. The speedup it reported came from the projection.
* **Unrolling the cell topology into one graph** (generic: any cell, its declared carry per ADR-087).
  This is the right shape if the per-step dispatch ever matters. On rawi it does not: the host work
  around a step is about 5% of it, and a leaner loop did not change the time. It would also need
  `GraphBuilder` to build a topology into a caller's graph with rebound inputs, which is a change of its
  own.

## Consequences

* rawi needs two arguments in its export script: `project_input=True` on its `RecurrentPhase`, and
  `input_topology=f"{p}_lstm_l{layer}_input"` on each `BiRecurrentCall`. The file it writes runs on
  loom-py-rt 1.0.0, from PyPI: 0 argmax differences on the 934 sentences. The published fused file stops
  there with `attempt to call field 'run_bi_lstm_and_retain' (a nil value)`.
* pyannote segmentation-3.0 (four layers, hidden 128) takes the same two arguments. Measured on
  loom-py-rt 1.0.0, same machine, against the per-step export of the same weights (7 interleaved
  rounds, median):

  | | per-step | projected |
  | :--- | ---: | ---: |
  | `infer`, 5 s | 131 ms | 93 ms (1.41x) |
  | `infer`, 10 s | 280 ms | 187 ms (1.50x) |
  | `infer`, 21.5 s | 620 ms | 381 ms (1.63x) |
  | diart stream, jfk 11 s | 5.7x real time | 8.3x |
  | diart stream, two speakers 21.5 s | 4.2x real time | 6.6x |

  The output is within 5.1e-7 of the per-step export, at the same distance from pyannote's f64 forward
  (2.3e-5 to 6.4e-5), with 0 argmax flips. `test_e2e_diart_vad_stream` passes 3158/3158 over 14
  settings against diart's own code, with 0 rows off, and the card gate passes. A copy whose layer-2
  directions read each other's projection fails all three checks: 356 of 589 frames flip, the stream
  gate fails with 1,058 or more rows off, and the card gate reports speech in silence.
* Tests: `test_recurrent.py` runs a projected two-layer bidirectional stack a layer at a time against
  `nn.LSTM`, and fails if layer 1's directions swap projections.
  `test_a_projected_layer_runs_its_input_half_first` pins the emitted calls and checks the chain with
  `check_subgraph_calls`. A sweep without its projection is rejected.
