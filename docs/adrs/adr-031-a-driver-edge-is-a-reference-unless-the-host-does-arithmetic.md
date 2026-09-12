---
type: adr
status: accepted
date: 2026-09-12
tags: [engine, exporter, drivers, performance, accelerators]
---

# ADR-031: A Driver Edge Is a Reference Unless the Host Does Arithmetic On It

## Context

`output_store.h` has stated the rule since P4.0.12: **marshal only when a value is genuinely
host-side** — a final result, a control decision, or host math the driver actually performs. Retained
outputs and `{from = 'module'}` references are the mechanism, and Whisper, Dia and T5 use them for
their large edges.

Nothing checked the rest. An audit of every driver in the zoo found four that pushed intermediates
through Lua for no host reader, and one binding — `loom.run_recurrent` — that predated the rule and
had no way to obey it.

**On CPU this is nearly free, which is why it went unnoticed.** Measured on EnCodec: the marshalling
round trip is ~0.6% of a decode, and an ABBA comparison of the two builds is inside the noise of a
two-core box. On an accelerator it is not free — see the measurement below.

## Decision

**Every driver edge whose consumer is a graph is a retained reference.** A value crosses the boundary
only when the host reads it: a returned result, a loop condition, or arithmetic the driver performs.

Three engine additions make that expressible where it was not:

* **`loom.run_recurrent_and_retain`**, and `sequence` may be a reference. A stacked LSTM is N calls
  with nothing crossing between them.
* **`{from = ..., row = t, rows = n}`** — a retained reference can name a row RANGE, not just a
  prefix. A transducer's joint consumes one encoder frame per call.
* **`loom.output_shape(module, index)`** — a retained output's shape without its data, so a loop can
  be driven by a tensor it never reads.
* **`loom.run_ode` / `loom.run_ode_and_retain`** — `dx/dt = f(x, t)` integrated with the loop, the
  state and the update on this side, and a choice of method (euler, midpoint, heun, rk4). A sampler's
  state is the model's whole spectrogram and its only reader is the next graph.

## What the Audit Found

| driver | crossed | now |
|---|---|---|
| EnCodec | `pre`→LSTM→LSTM→`post` sequences, 7·T·1024 doubles per call | codes in, waveform out |
| transducers (parakeet ×2, gigaam) | the whole encoder output — 1.1 MB for 11 s of audio — plus `embed`, per-layer h/c, `top_h` | one row named per frame; only the first pass's zeroed h/c cross |
| `run_bi_lstm` (5 in Kokoro, 5 in StyleTTS2) | `layer_input`, `h_prev`, `c_prev` in and `h_new`/`c_new` out, per timestep per direction — 4T calls | two C++ sweeps per BiLSTM |
| `run_resblk_stack` (2 per model) | each block's output rebuilt as a Lua table and written straight back — and the two conversions around it are exact inverses | blocks chain by reference; one conversion in |
| StyleTTS2 `albert` | a `T×768` table, re-passed to the diffusion estimator at every sampler step | retained; both readers name it |
| Supertonic `txt_emb` | same shape, every CFM step — and its topology is a text-length BUCKET, so the reference needed a computed name | retained; `OutputRef` gained `module_expr`/`variants` |
| Kokoro + StyleTTS2 F0/N branch | `resblk_stack` → `proj1x1` → vocoder, through Lua at every hop | one path, no tables; two layout helpers retired |
| Matcha + Supertonic CFM samplers | the whole state out and the velocity back, **per step** | one `run_ode_and_retain` call; the state never becomes a table |
| Matcha `denormalize` | `z * std + mean` host-side, which forced the state across anyway | folded into `VocoderWrapper` — a graph change, and the reason that model's last mel-sized crossing is gone |
| Whisper, Dia, T5 | — | already retained when written |

**What stays host-side stays for a reason, and the reasons differ.**

* A BiLSTM's two directions are *interleaved per row* by the driver, and Kokoro's and StyleTTS2's
  duration encoders *concatenate the style vector into every row* between stages. Those are
  arithmetic, so the values are genuinely host-side under the current topology split. Moving them
  means changing what the traced graphs accept — a re-trace per phase, not a driver fix.
* **StyleTTS2's ADPM2 sampler is midpoint-shaped and still stays in Lua.** Two denoiser calls around
  a midpoint in sigma space — but it is an ancestral SDE sampler, not an ODE integrator: its step
  ends at `sigma_down` rather than at the next schedule point, and it adds `noise * sigma_up`.
  Bending `run_ode` around that would make its contract mushy for no gain, because the state there is
  the STYLE vector — `2 * 128` floats per step, ~2.5k doubles per utterance, against the megabytes a
  CFM state moves.
* A duration predictor's output, and every returned waveform. Those are the rule, not exceptions to
  it: the host really does read them.

**22 plain `run_subgraph` calls remain across 24 models, and every one is in that list.** The audit is
finished in the sense that matters — what is left is not a crossing anyone can remove without moving
arithmetic into a graph.

## Consequences

* The rule is now enforceable where it was advisory: a driver that marshals an intermediate is visible
  as a `run_subgraph` whose output is bound to a local nothing but the next call reads.
* **Shape checking got stricter for free, and it caught a real bug immediately.** A retained copy
  asserts `ggml_are_same_shape`; marshalling only ever compared element counts. Chaining
  `run_resblk_stack` through the store failed on `[66,512]` vs `[512,512]` — same element count,
  different tensor — which the Lua path would have accepted silently.
* Nine models were re-exported and re-gated; the whole 24-model card gate passes, including the
  ASR-oracle rows that are the only real test for the TTS pair.
* **Those re-exports cannot be published before the engine is.** Each binding added here is a new
  requirement on the RUNTIME, and a driver that calls one is unloadable by every released
  `loom-py-rt`: `output_shape` (5 models), `run_ode_and_retain` (2), `run_recurrent_and_retain` and
  the `ELU` primitive (EnCodec), a retained ROW range (3). The model-card gate cannot see this — it
  runs against the local build, which by construction has them — so the ordering is a rule rather than
  a check: **engine merged → wheels published → GGUFs uploaded.** Worth a `loom.min_engine` declaration
  in the file if this recurs; it will recur every time a binding is added.
* **Measured on an RTX 5090** (GigaAM v3, the transducer whose encoder output stopped crossing; the
  same weights with the old driver and the new one, interleaved ABBA, 8 runs each):

  | clip | encoder output | old | new | |
  |---|---|---|---|---|
  | 11 s | 1.1 MB | 0.565 s | 0.550 s | **−2.7%** |
  | 121 s | 12.4 MB | 1.061 s | 1.013 s | **−4.5%** |

  The 121 s runs are bimodal (~1.01 and ~1.11 in both arms — the 285K's P-cluster/E-cluster split,
  not noise), and the gap inside each cluster is the same 47–50 ms.

  `scripts/bench_driver.cpp` then reached the families `loom_cli` cannot (anything whose answer is
  audio), same box, same ABBA:

  | model | crossing removed | before | after | | Lua-loop work removed |
  |---|---|---|---|---|---|
  | kokoro-82m | ~30 MB | 245.6 ms | 218.0 ms | **−11.2%** | 10 BiLSTM step loops, 2 layout round trips, per-row fan-out |
  | encodec-32khz, 8 s | 22.9 MB | 156.6 ms | 148.1 ms | **−5.4%** | none — bulk pushes only |
  | gigaam-v3, 121 s | 12.4 MB | 1061 ms | 1013 ms | **−4.5%** | a per-frame slice loop over 1512 frames |
  | matcha, 10 steps | 5.1 MB | 99.1 ms | 98.3 ms | −0.8% | one elementwise update per step |

  **The saving does not track megabytes; it tracks how the bytes were TOUCHED.** Per MB removed the
  four span 0.15 to 3.9 ms — a 25x range — and they sort by how much of the marshalling was an
  interpreted Lua loop rather than one bulk `push_number_array`. EnCodec moved the most bytes and
  saved little because its edges were always bulk; Kokoro moved fewer and saved the most because its
  driver walked them element by element. That is the useful rule for deciding what to fix next, and it
  is not the rule the byte counts suggested.

  On the two-core dev box the same three comparisons are inside the noise (best-of-3: −0.4%, −3.2%,
  −2.6%, against ±10% swings between repeats of one binary) — the effect is real there too and simply
  below what that machine can resolve.
