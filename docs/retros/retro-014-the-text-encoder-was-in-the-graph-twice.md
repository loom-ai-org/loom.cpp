---
type: retro
date: 2026-08-22
domain: performance
tags: [exporter, phase-split, node-census, duplicated-work, vits, matcha]
---

# Retro-014: The 24-Node Gap Was One Text Encoder, Run Twice

## The Issue

P4.16's per-shape table put one convolution group at **2.59x** onnxruntime — by far the worst row, and
the only one without a plausible kernel-level explanation. loom issued 153 `CONV_1D` nodes where
onnxruntime had 129 dense convolution nodes.

## Root Cause Analysis

Every one of loom's 153 dense convolutions had an onnxruntime counterpart of exactly the same shape,
and **36 of them were the same 36 convolutions run twice**: the VITS export split the text side into
two topologies, `stats` and `logw`, and each carried its own complete copy of the TextEncoder. loom
lowered nothing in more nodes than it needed and the ONNX exporter folded nothing away — the gap was
entirely a phase split in loom-exporter.

The census predicted it exactly: 1.91x the arithmetic at 1.38x lower throughput is 2.64x, against the
2.59x measured.

## Resolution & Lesson Learned

The text encoder runs once, in VITS and Matcha both. On a Pi 4 at 4 threads: **1.196 → 1.099 s, 98 ms
and 8.2%**, moving loom from 1.126x to **1.033x** of onnxruntime end to end.

* **Actionable takeaway 1 — when a ratio is inexplicable by the kernel, count the nodes before
  profiling them.** `scripts/conv_census.py` needs neither a run nor the target hardware.
* **Actionable takeaway 2 — rank by the machine's peak as well as by the competitor.** The row that
  eventually gave up the most time (`CONV_TRANSPOSE_1D`, 115 ms) ranked *last* in the table at 1.18x,
  because both engines were sitting on the same floor — 7.3 and 8.6 GFLOP/s where the machine does 25.
  A ratio against a competitor hides a row where both implementations are equally bad.
* **Actionable takeaway 3 — a phase split is a compute decision.** Splitting a model into topologies
  for export convenience duplicates whatever both phases need.

---

## Full record (verbatim from the ledger)

### P4.15d — the census


**The answer, in one sentence.** Every one of loom's 153 dense convolutions has an onnxruntime
counterpart of exactly the same shape, and **36 of them are the same 36 convolutions run twice**: the
VITS export splits the text side into two topologies, `stats` and `logw`, and each carries its own
complete copy of the TextEncoder. loom lowers nothing in more nodes than it needs and the ONNX exporter
folds nothing away — the gap is entirely a phase split in loom-exporter, and it is **~72 ms of the
1.205 s**, 78% of the excess in P4.16's worst row. (P4.15f removed it and measured **98 ms**; the
sentences below are the prediction, left as written.)

**The two node lists, and the tool that produces them.** `scripts/conv_census.py` walks the topology
JSON out of the GGUF, propagates shapes symbolically, and prints every convolution with its real
`(IL, IC, K, OC, OL, s, p, d)`; with `--onnx` it does the same statically for an ONNX graph and diffs
the two. The profiler cannot answer this on its own — it buckets by `(op, ne0, ne1)` and a 1-D
convolution's output is `[OL, 1, OC]`, so all 93 of the L~100 convolutions collapse into one row with
their weight shapes gone.

```sh
python3 scripts/conv_census.py ../hf-models/vits-piper-en-gb-miro/vits-piper-en-gb-miro.gguf \
    --syms n_tokens=100 --syms flow_vocoder:n_tokens=287 --quiet \
    --onnx ~/Dev/piper/pipertts_en-GB_miro/miro_en-GB.onnx
```

The two `--syms` are the driver script's own sequencing, not something the topology JSON reveals:
`stats`/`logw` run at the token count, `flow_vocoder` at the duration-expanded `y_length`. **The ONNX
file is already on the dev box** at `~/Dev/piper/pipertts_en-GB_miro/miro_en-GB.onnx` — sha256
`c2122147…`, byte-identical to the Pi's `~/pipertts-en-gb-miro/miro_en-GB.onnx` and to
`OpenVoiceOS/pipertts_en-GB_miro` on the Hub. Reading it needs `pip install onnx` (1.21 here) and no
runtime, no session and no Pi: **this whole item was done on the dev box with the Pi unreachable.**

Thirty-five distinct shapes, of which **exactly three differ**, and all three are text-encoder shapes:

| op | K | IC | OC | loom | onnx | diff |
|---|---:|---:|---:|---:|---:|---:|
| `CONV_1D` | 1 | 192 | 192 | 66 | 42 | **+24** — the six attention layers' q/k/v/o |
| `CONV_1D` | 3 | 192 | 768 | 12 | 6 | **+6** — the six FFN first convs |
| `CONV_1D` | 3 | 768 | 192 | 12 | 6 | **+6** — the six FFN second convs |
| everything else (32 shapes) | | | | 78 | 78 | 0 |
| **total** | | | | **168** | **132** | **+36** |

Both sides include the 12 depthwise and the 3 transposed convolutions, and both agree on them exactly.
**Both totals are also the ones P4.16's table arrived at independently, by profiling**: its onnx column
sums to 132 calls, which is exactly the ONNX graph's 132 convolution nodes; its loom column sums to 156,
which is the census's 168 minus the 12 depthwise that loom profiles under `IM2COL` rather than as
convolutions. Two engines, two methods, four numbers, no discrepancy — which is what makes the +36
believable.
The onnx side's own module split is the confirmation that the +36 is one encoder: `enc_p` 37, `dp` 32,
`flow` 40, `dec` 23 — and loom's `stats` is 37 (= `enc_p`), its `logw` is 68 (= `enc_p` minus its
projection, 36, **plus** `dp`'s 32), its `flow_vocoder` is 63 (= `flow` + `dec`).

**The duplicated prefix is provably the same computation, not merely the same shapes** — which is
`conv_census.py --shared-prefix`, a structural isomorphism over the two node lists with weights compared
by identity rather than by shape. `stats`'s first
**469 nodes and `logw`'s first 469 nodes are identical node for node** — same ops, same attrs, same
weight tensors once `loom.tensor_alias` is resolved — and they diverge at exactly the node where
`stats` applies `enc_proj` and `logw` applies `dp_pre`. That is 469 of the model's 1744 graph nodes
recomputed: 36 `CONV_1D`, 24 `MUL_MAT`, 12 `LAYER_NORM`, 18 `PAD_1D`, 18 `CONCAT` and the elementwise
around them. **The FILE already dedupes** — 137 of `logw`'s weights are `loom.tensor_alias` entries
pointing into `stats.`, which is why `logw` holds 2.2 MB of tensors against `stats`'s 44.2 MB. Only the
compute is duplicated.

**Three corrections to what the previous entries assumed.** Each of them was load-bearing:

1. **onnx has 129 `Conv` nodes in total, twelve of them grouped — so its DENSE count is 117, not 129.**
   The observation this item was written from ("129 dense convolution nodes plus the same 12
   depthwise") counted the depthwise twice. The real gap is 153 - 117 = **36**, which is one encoder
   exactly; there is nothing left over and nothing to over-explain.
2. **The 26 `MatMul`/`FusedMatMul` nodes are not convolutions in disguise.** None of the 26 has an
   initializer operand — 24 are the four attention products of `enc_p`'s six layers and 2 are the
   top-level path expansion. loom runs the same products as `MUL_MAT`, 48 of them, again 2x for the
   same reason. **This deletes the second half of P4.15c's premise**, which is corrected in place.
3. **The per-scale counts line up once the depthwise are put on the right side.** onnx's 69 at L~100 is
   `enc_p` 37 + `dp` 32, and `dp`'s 32 includes the 12 depthwise; its dense count there is 57. loom's
   93 dense is `stats` 37 + `logw` 56. 93 - 36 = 57. Nothing is unaccounted for at any scale.

**What it is worth, and the decomposition it hands P4.16's worst row.** From the census (arithmetic) and
P4.16's table (time), at L~100 on the Pi:

| | convs | MFLOP | ms | GFLOP/s |
|---|---:|---:|---:|---:|
| loom | 93 | 2598.7 | 151.1 | 17.2 |
| onnx | 57 | 1360.1 | 57.2 | 23.8 |

(Dense only on both sides, which is why onnx reads 57 and 57.2 ms here against the 69 and 58.3 ms in
P4.16's row: that row counts onnxruntime's 12 depthwise convolutions and their 1.1 ms, loom's does not —
loom's depthwise land in its `IM2COL` row instead. The FLOP columns are the census's, the ms columns are
P4.16's.)

**1.91x of the work at 1.38x lower throughput is 2.64x — and the row measures 2.59x.** So P4.16's "2.59x
cannot be a GEMM-throughput story" is right for a reason it did not consider: **most of that row is not
a kernel problem at all.** The duplicated encoder is 1238.6 MFLOP, and at loom's own measured throughput
for that group it is **72.0 ms** — 78% of the row's +92.8 ms excess. What remains after it is 1.38x,
which is the same ~1.3-1.4x the rest of the model already shows and is a throughput question like the
others.

That 72 ms is convolution only. The other 433 nodes of the duplicated prefix (24 `MUL_MAT`, 12
`LAYER_NORM`, the elementwise) are on top of it, out of the ~240 ms loom spends outside convolution.

**And VITS is not the only model that does it.** `--shared-prefix` over all seventeen local exports
finds one other: **Matcha-TTS's `encoder_mu` and `encoder_logw` share 642 nodes of 668/647**, including
40 convolutions, and its driver runs both. Everything else is clean. Details and the one false-positive
shape to know about are in P4.15f.

**The fix belongs in loom-exporter, and it is P4.15f below — DONE the same day.** The engine was doing
exactly what the GGUF told it to. Measured outcome: **1.196 -> 1.099 s on the Pi, 98 ms**, against the
~72 ms this entry predicted for the convolutions alone; the rest is the other 433 nodes of the
duplicated prefix, which is where the prediction said the remainder would be.


### P4.15f — the text encoder, once (VITS and Matcha)

**What it was worth.** Raspberry Pi 4, 4 threads, cool and idle, ABBA in both orders, medians of the
steady-state reps: **1.196 -> 1.099 s, 98 ms and 8.2%** on the reference utterance. Against
onnxruntime's 1.063 s for the same 73472 samples that is **1.126x -> 1.033x**. (A second session at a
warmer ambient measured 1.204 -> 1.115, 89 ms; the delta is stable, the absolute is not.) At ONE thread
the L~100 convolution group — the one P4.16 has at 2.59x — falls **376.7 -> 199.1 ms, 1.89x**, against
the 1.91x arithmetic reduction the census predicted. Matcha's half is real but small: see below.

**The change, in both exporters.** `stats` and `logw` were two topologies, each carrying its own copy
of the TextEncoder; they are now one two-output `text` topology
(`vits_export.py`'s `TextWrapper`, replacing `StatsWrapper` + `LogwWrapper`), and the driver's two
`run_subgraph` calls are one:

```lua
local stats, logw = loom.run_subgraph('text', {n_tokens = T, n_past = 0}, {tokens = ..., z_noise = ...})
```

`z_noise` moves ahead of the call — it is sized from the token count alone, so the RNG draw order does
not change, which is the whole reason the result can be required to be bit-identical. Matcha is the
same edit: `MuWrapper` + `LogwWrapper` -> `EncoderWrapper`, `encoder_mu` + `encoder_logw` -> `encoder`.

**Verification, in the order it is worth trusting.**

1. **The waveform is bit-identical**, `==` on the float lists, not a tolerance: VITS at seeds 0/7/42/1234
   and at two utterance lengths; Matcha at seeds 0/42 and n_steps 1/2/10. **And the comparison can
   fail** — changing the seed or one token id makes it differ, checked both ways.
2. **The tensor oracle.** `test_e2e_matcha_mil_text_encoder` now runs the merged topology and reads
   both outputs from one build, against `reference_forward_matcha_text_encoder.py`'s real-PyTorch
   fixture: **mu 1.24e-4, logw 6.4e-5**. Perturbing the reference by 0.01 makes it fail, so the bound
   is doing work.
3. **The census closes.** `conv_census.py --onnx` reads **132 against 132, every shape row zero** where
   it read 168 against 132; `--shared-prefix` reports no shared prefix for either model. loom now
   issues exactly onnxruntime's 117 dense convolutions, 12 depthwise and 3 transposed.
4. **The profiler agrees, per node.** VITS at one thread: `CONV_2D` **153 -> 117 calls per rep**, the
   L~100 bucket **93 -> 57**, `MUL_MAT` **60 -> 36** (the 24 duplicated attention products), and every
   other bucket unchanged to the call. Matcha: node executions 2972 -> 2083, its L=38 convolution
   bucket 84 -> 44 calls and 177.3 -> 88.3 ms.
5. **Structure.** The GGUF snapshot moves exactly where it should and nowhere else: three topologies to
   two, one fewer KV, `stats.*`/`logw.*` renamed to `text.*` — and **the multiset of tensor sha256s is
   unchanged**, so no weight moved, only the namespace.
6. **Suites.** loom-exporter `tests/ci` 572 passed; loom.cpp `-L ci` 68 passed and `-L gate` 82 passed
   with the new exports in place.

**Matcha: the same bug, and honestly not worth much.** 642 duplicated nodes of 668, 40 convolutions,
and the driver did run both — but its HiFi-GAN v1 vocoder dominates the synthesis so completely that
even at `n_steps=1` a 39-token utterance takes 5.1 s on the Pi, of which the duplicated encoder is
**~96 ms at one thread (0.7%)**. End to end at four threads it is **below the noise floor of a Pi that
heats from 53 to 83 C during the measurement** — eight interleaved rounds put before and after within
each other's spread while the drift moved both by 13%. Recorded as measured: the fix is right, the
speedup is not visible from outside on this model. **And that thermal drift is itself the lesson** — a
6-second-per-rep workload heats a Pi 4 faster than an 80 ms effect can be resolved; VITS at 1.2 s per
rep is measurable, Matcha at 6 s is not.

**What moved outside the two exporters.** Three engine gate tests hardcoded the old topology names and
now name the new ones: `test_e2e_vits_mil_lua_driver`, `test_e2e_matcha_mil_lua_driver`, and
`test_e2e_matcha_mil_text_encoder` (which also grew a `run_topology` that returns every declared
output). The two **bespoke**-conversion tests (`test_e2e_vits_lua_driver`, `test_e2e_matcha_lua_driver`)
still say `stats`/`logw` and `encoder_mu`/`encoder_logw`, correctly: the hand-built converters are
unchanged. Four hermetic exporter tests build the real Matcha config against stub topologies and needed
the stub to declare two outputs (`_topo(..., outputs=[...])`).

**Follow-ups this leaves open, none of them blocking:**

* **The published GGUFs are pre-P4.15f.** `loom-ai-org/vits-piper-en-gb-miro` and the Matcha repo both
  carry the three/four-topology export, and `../hf-models/` mirrors them. Re-exporting and pushing is a
  publishing decision, so it is not done here — but note that an OLD published GGUF still runs
  correctly on a new engine (nothing about the format changed), while a NEW GGUF needs no engine change
  either. The two are independent.
* **The local gate fixture set is refreshed, and one test that never ran now runs.**
  `loom-engine-artifacts/v5` has the new `vits_mil.gguf` and `matcha_mil.gguf`, plus
  `v5/vits_mil/vits_mil.gguf` — the DIRECTORY form `LOOM_VITS_MIL_DIR` actually resolves to, whose
  absence is why the VITS MIL gate had been skipping silently. `v5/matcha_text_encoder_ref/` is new
  too (regenerate with `reference_forward_matcha_text_encoder.py`, which needs
  `PYTHONPATH=~/Dev/Matcha-TTS` and the `matcha` venv), so the tensor oracle above runs by default.
  `LOOM_FIXTURES=~/Dev/loom-engine-artifacts/v5 ctest -L gate` is **82/82** on this machine.
* **`~/loom-p415/tts_main.cpp` on the Pi** is new: prof_main's sibling for a driver that takes raw
  token ids plus scalar knobs (`tts_main <gguf> <id,id,...> [reps] [name=value ...]`), which is what
  timing Matcha needed. Same disposable scratch tree, appended to its CMakeLists the same way.

