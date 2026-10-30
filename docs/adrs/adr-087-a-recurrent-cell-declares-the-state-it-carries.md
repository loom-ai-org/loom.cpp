---
type: adr
status: accepted
date: 2026-10-25
domain: model-coverage
tags: [recurrence, gru, lstm, lua-bridge, cell-sweep, auroratts, exporter]
---

# ADR-087: A Recurrent Cell Declares the State It Carries

Serves [Epic-03](../epics/epic-03-model-coverage.md) (AuroraTTS-100K, the first GRU in the zoo). Builds
on the recurrent sweep the LSTM families use (`loom.run_recurrent`, `loom.run_bi_recurrent_and_retain`,
[ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)'s boundary work).

## Context

ggml has no recurrent op, so a recurrence is a per-timestep **cell topology** that the engine steps in
C++ (`LoomLuaBridge::cell_sweep`), carrying the state between steps on the host. Every cell until now was
an LSTM's: `layer_input`, `h_prev`, `c_prev` in; `h_new`, `c_new` out. The sweep hardcoded that shape.

AuroraTTS-100K has a bidirectional `nn.GRU` (28 units per direction). A GRU's whole state is `h`. The
exporter could not reach it the LSTM way either: a traced `nn.GRU` arrives in MIL as an unrolled
`while_loop`, with no opaque op to read weights from (loom-exporter `recurrent.py`).

## Options

1. **No engine change: a GRU cell in the LSTM's shape**, with `c_new` a copy of `h_new` and `c_prev`
   unread. It would run on the released 1.0.0-rc19 and could be published at once, but every GRU cell
   would declare a state it does not have, and the sweep would copy it each step.
2. **The sweep accepts a second cell shape, told apart by the declaration**: `c_prev` in and two outputs
   is an LSTM; no `c_prev` and `h_new` alone is a GRU. Anything in between is refused.
3. **A GRU primitive in ggml.** A kernel for a structure that composes from MUL_MAT, ADD, SUB, MUL,
   SIGMOID and TANH already, against [ADR-003](adr-003-per-model-complexity-in-the-exporter.md).

## Decision

Option 2 (the user, 2026-10-25), accepting that AuroraTTS publishes only with the next release.

* `cell_sweep` looks for a `c_prev` input. With one, the cell must declare exactly two outputs and both
  halves are carried, as before. Without one, it must declare exactly one, and only `h` is carried. A
  cell that declares two outputs and no `c_prev` would silently drop a cell state, so it is refused with
  an error naming both shapes.
* The exporter builds a GRU's cells from the module, not the trace (`recurrent.build_gru_cell_topologies`,
  reached through `RecurrentPhase` when the module is an `nn.GRU`): torch's gate blocks `[r, z, n]`, the
  two biases kept apart because `b_hn` sits inside `r * (W_hn h + b_hn)`, and `h' = n + z * (h - n)`.
  The `_fwd`/`_bwd` names are the LSTM's, so `BiRecurrentCall` and the `rows`/`layout_a` retention apply
  unchanged.

## Consequences

* Measured: `tests/ci/test_gru_sweep.cpp` steps a float64 numpy GRU to 1.1e-7 in both directions and
  through the interleaving binding, and fails with the change reverted. AuroraTTS's BiGRU through the
  whole driver is inside upstream's own float32 spread (Epic-03).
* Every existing LSTM cell is untouched: the two-output path is the old code.
* A file with GRU cells fails on an older runtime with the old error ("a cell topology must declare
  both 'h_new' and 'c_new'"). Its card says which release it needs (loom-exporter
  `FEATURE_FIRST_RUNTIME`).
* Stacked GRUs (`num_layers > 1`) are refused at export until a model needs them; the LSTM path's
  per-layer naming is what they would follow.
