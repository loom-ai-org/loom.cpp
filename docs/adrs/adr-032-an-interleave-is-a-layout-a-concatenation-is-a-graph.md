---
type: adr
status: accepted
date: 2026-09-12
tags: [engine, exporter, drivers, tts, family-9]
---

# ADR-032: An Interleave Is a Layout; a Concatenation Is a Graph

## Context

[ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md) made every driver
edge whose consumer is a graph a retained reference, and closed with two exceptions it called
arithmetic:

* a BiLSTM's two directions, **interleaved per row** by the driver into `[h_fwd | h_bwd]`;
* Kokoro's and StyleTTS2's duration encoders, which **concatenate the style vector into every row**
  between stages.

It recorded both as "a re-trace per phase, not a driver fix". That was half right, and the half that
was wrong is the more interesting one.

## Decision

**The interleave was never arithmetic. It is a LAYOUT, and the only place that knows both halves is
the engine.** `loom.run_bi_recurrent_and_retain(fwd_module, bwd_module, sequence, seq_len, input_dim,
hidden_dim [, layout])` runs both directions and writes both halves of every row into ONE store slot.
Nothing could concatenate two stores by naming them, which is why the driver pulled both back into Lua
— but a single call owns both sweeps, and the `h` each timestep yields is already host-side because
the next step's `h_prev` needs it. The interleave costs an index.

`layout` picks which axis time runs down, because the consumers disagree and both are ordinary:
`"rows"` (`[2*hidden, seq_len]`, one contiguous row per timestep — what a rows_flat graph input and a
following recurrent sweep read) or `"layout_a"` (`[seq_len, 2*hidden]`, time on the fastest axis —
what the conv-family topologies declare). **Telling the PRODUCER which convention to write is what
retires the converters**: `to_row_major`, `from_row_major`, `to_layout_a` and `from_layout_a` are
deleted, because every edge they converted is now a reference and the one real layout disagreement is
answered at the source.

**The concatenation genuinely needed the re-trace, and it is one phase, not three.**
`duration_style_concat` is a traced graph — `torch.cat([x, s], axis=-1)`, which is what
`DurationEncoder.forward` itself does — called four times: once on the text encoder's output and once
after each AdaLayerNorm. Not folded into the AdaLayerNorms, for two reasons: the first concatenation
has no AdaLayerNorm to hang it on, and the three that do keep their existing interface and with it
their row in the bespoke-vs-MIL equivalence gate.

**Frame expansion follows the same rule as everything else.**
`loom.expand_by_duration_and_retain(module, durations [, layout [, index]])` repeats a retained
sequence's rows in place, in the producing module's own store. The COUNTS are genuinely host-side —
the host predicted them and sizes every later stage by their sum — but the sequence being repeated is
not, and it is the largest tensor either driver holds (`T_frames` rows of 640 and of 512).

Three phases were re-traced to make the chain hold end to end, and all three are the same move — a
transform that existed only because the consumer was Lua:

| phase | was | is |
|---|---|---|
| `text_encoder_cnn` | Layout A out | rows_flat out — a BiLSTM reads one row per timestep, and a `[T, C]` tensor's timestep is `T` apart |
| `bert_encoder` (StyleTTS2) | Layout A out | rows_flat out — same interface Kokoro's `albert_bert_encoder` already had, so one shared phase consumes both |
| `duration_proj` | one timestep | the whole sequence — the per-token Lua loop was free only while the rows were already Lua values |

## Consequences

* **Kokoro's and StyleTTS2's duration halves marshal nothing at all.** From the text encoder to
  `duration_proj` every edge is a name; the durations are the first value that crosses, and the host
  really does read them. The last two Lua tables in either driver are that one and the waveform.
* **A store can now be reshaped by something that is not a build, and that was a live hazard, not a
  latent one.** A retained graph ends in one `cpy` per declared output whose destination is the store's
  own tensor, and `GraphBuilder`'s cache key did not include those tensors — so a binding that reshaped
  a module's store between two builds had the cached graph copying into freed memory. `build()` now
  keys its cache on `OutputStore::layout_epoch()`, a counter the store bumps whenever `reshape` really
  reallocates, and rebuilds when it moves; an unchanged store costs one integer compare.
  **It is the epoch and not the slot POINTERS, and the difference was a heap corruption**: the first
  version of this guard compared addresses, the allocator handed the freed store's address straight
  back, and Kokoro's second synthesis in a process wrote through it — see
  [Retro-045](../retros/retro-045-the-allocator-handed-the-same-address-back.md), which is also why
  both TTS drivers' e2e tests now call `infer` twice.
* **Two phases are gone from the equivalence gate and named as such.** `bert_encoder`,
  `text_encoder_cnn` and `duration_proj` deliberately redefined their interface, which is exactly what
  that test compares, so they join `albert` and `diffusion` in its named skip list rather than the
  comparison being loosened for everyone.
* **Measured, and it is a WASH on this CPU — which is the expected shape, not a disappointment.**
  Kokoro, 32 phonemes, 2 threads on the 2-core dev box, interleaved A B B A A B, best of two runs per
  launch: `base` 3394 / 3436 / 3700 ms against `new` 3474 / 3369 / 3645 ms — mean of bests 3510 vs
  3496, a 0.4% difference inside a 10% per-launch spread. ADR-031 said the same thing about its own CPU
  numbers and measured −11.2% for Kokoro on an RTX 5090, where a crossing is a device round trip rather
  than a memcpy. What this removes is ~1.5 MB of Lua doubles per utterance against 3.4 s of graph
  compute; the machine that can see it is the GPU, and nobody has re-run the 5090 since.
  **The correctness result is the one that matters here:** the same text, the same seed and the same
  checkpoint produce a **bit-identical waveform** before and after, for both models — 90 000 and 78 600
  samples, zero differing.
* **The cell sweep is one implementation with three callers.** `run_recurrent`, its retaining form and
  the bidirectional one share `cell_sweep`, which walks one direction and hands each timestep's `h` to
  a sink. What the sink does — push a Lua double, write a store row, write half a row — is the only
  difference between the three bindings.
