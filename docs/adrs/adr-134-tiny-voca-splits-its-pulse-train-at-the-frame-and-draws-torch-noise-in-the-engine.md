---
type: adr
status: accepted
date: 2026-11-17
tags: [model-coverage, tts, dsp, source-filter, precision, performance, lua-bridge, rng, tiny-voca]
supersedes: [adr-133 (its pulse-train decision)]
---

# ADR-134: tiny-voca Splits Its Pulse Train at the Frame and Draws Torch's Noise in the Engine

Serves [Epic-03](../epics/epic-03-model-coverage.md) and [Epic-05](../epics/epic-05-edge-performance.md).
Amends [ADR-133](adr-133-tiny-voca-renders-its-pulse-train-in-the-driver-and-speaks-a-sentence-at-a-time.md),
whose Option 2 (the driver renders the pulse train sample by sample in doubles) this replaces; the rest
of ADR-133 stands. The builtin follows the binding criterion in `lua_bridge.h` and
[ADR-003](adr-003-per-model-complexity-in-the-exporter.md).

## Context

tiny-voca was the first model slower on loom than under onnxruntime on the Pi 4 (0.77x at F32). A
per-call wall timer around the driver's `loom.*` calls (LuaJIT `clock_gettime`, no per-node profiler)
split the 664 ms run: the four graphs took 373 ms -- less than onnxruntime's whole 516 ms -- and two
per-sample Lua loops the rest: `aten_randn`, torch's noise stream in doubles, 202 ms, and the pulse
train, 85 ms, over 130k samples each. The F32 node profile had not shown it: it times graphs only.

The pulse train was in the driver for its precision (ADR-133): upstream sums the phase in float64. That
reason was taken on trust; measured, it holds only for one spelling of the float32 alternative.

## Options

Measured in upstream's own model against its float64 `pulse_train`, everything else identical, on a
5.4 s and a 13.7 s utterance, with ggml's cumsum emulated (it adds sequentially in float32; torch's CPU
cumsum accumulates float32 in double and makes every float32 spelling look better than it is):

| the pulse train | waveform max diff | SNR |
|---|---|---|
| 1. all float32, one running sum | 0.35 / 1.5 | 12.8 / 1.2 dB |
| 2. upstream's ONNX split (in-frame float32 sum, 24-bit integer phase steps across frames) | 1.2e-3 / 1.7e-3 | 65.2 / 67.0 dB |
| 3. per-frame starting phase from the driver in float64, theta in [0, 2 pi) | 2.2e-4 / 1.3e-2 | 82.5 / 61.0 dB |
| 4. **as 3, theta centred in [-pi, pi)** | **5.0e-5 / 1.1e-4** | **95.5 / 93.6 dB** |

5. Keep the Lua and make it faster: a bit-identical rewrite (integer MT19937, a float32 rounding by
   magic addition, locals for globals) took the run to 503 ms, 1.03x. The loops stay per sample and
   their 260k values still cross the boundary twice.
6. A C++ pulse-train primitive: per-model, which ADR-003 forbids.

Option 2's integer steps cannot be exact in a float32 graph (their running sum passes 2^24), and option
3 loses 30 dB where the closed form `sin((W + 1/2) theta) / (2 sin(theta / 2))` is evaluated at up to
~1400 rad right next to each pulse, where its denominator goes to zero. The form is 2 pi periodic, so
centring theta keeps every argument near a pulse small and float32's relative precision intact.

## Decision

Option **4**, and the noise drawn by the engine:

* **The driver sends one start phase per frame** (505 values for 5.4 s, not 130k samples), the float64
  sum done a frame at a time in closed form: inside a frame the frequency is a geometric series, so its
  cycles are `exp(a) expm1(b - a) / expm1((b - a) / HOP) / SR`, kept modulo 1. The synth graph does the
  in-frame float32 cumsum, the centred theta and every per-sample transcendental. The log-linear
  interpolation is a `[T, 2] @ [2, HOP]` matmul (ggml's elementwise ops broadcast only their second
  operand), and the two divisions are spelled from the log and with rsqrt (a constant over a tensor is
  MIL's `inverse`, which has no lowering and an epsilon of its own).
* **`loom.aten_randn(seed, n)`**: torch's CPU `randn` stream, ATen's scalar path. It is bit-identical
  to that path (torch 2.8 under `ATEN_CPU_CAPABILITY=default`: 20 seed/size cases, and 129,281 values
  through the engine, no value differs); ATen's x86 AVX2 path uses its own log/sincos and differs by
  up to 1.9e-6, as the Lua helper always did. It meets the binding criterion as per-task code: sanoTTS,
  vocetta2, AuroraTTS and tiny-voca seed their noise from torch. The shared Lua helper calls it when the
  engine has it and keeps its own loop as the fallback for an older engine.

## Consequences

* **Pi 4** (docs/BENCHMARKS.md, footnote 4; 5 paired rounds, cooled): 0.397 s at F32 against 0.515 s
  for upstream's ONNX file (1.30x; was 0.668 s, 0.77x), and 0.351 s at Q8_0_PC against 0.645 s for its
  `quantize_dynamic` int8 (1.81x per second of audio; was parity). Per phase at F32: encode 17, prosody
  66, acoustic 158, synth 135 (127 before, now with the pulse train), Lua 15 ms (was 295).
* **Pi Zero W** (docs/BENCHMARKS.md, Pi Zero footnote 1; 3 rounds, one inference per process): 13.20 s on
  the CPU at Q8_0_PC (RTF 2.44; was 16.15 s, 2.99) and 10.48 s on CPU+VC4 at F32 (RTF 1.95; was 13.85 s, 2.57), a
  1.25x speedup per second of audio. The same output hash in every round; the VC4 at 26.7 dB against the CPU.
* **Accuracy**: the new export is 5.9e-5 (93.6 dB) from the old at F32 and 7.7e-5 (94.0 dB) at Q8_0_PC.
  `test_e2e_tiny_voca` keeps its 4e-3 bound with the same maxima as before (0.62e-3, 1.12e-3, 2.56e-3:
  the STFT convolutions' float32 error dominates). Two new driver sabotages go red: the phase not carried
  across frames, and a 255-sample series. Whisper word error rates are unchanged (4.4% dizione, 20.0%
  the fold, 3.8% Q8_0_PC).
* Exporter CI `test_the_split_pulse_train_is_upstreams`: the wrapper fed by the driver's series against
  upstream's float64 formula on a 1200-frame low voice, with ggml's cumsum: centred 84.0 dB, and the
  [0, 2 pi) form must stay below 50 (it measures 31.4).
* **The other three families, re-exported** (2026-11-17, the card builder into a scratch tree): the
  engine's draw moves their output by at most 8.2e-7 (above 122 dB; sanoTTS heart and heart-nano,
  vocetta2 with `noise_scale`, AuroraTTS, F32 and Q8_0_PC), and it is the only change: each new file
  with its noise patched back to the Lua loop renders what the published rc20 file renders, sample for
  sample. vocetta2's default (no noise) is bit-identical. The model-card gate passes all five cards
  (whisper-small). On the Pi Zero (docs/BENCHMARKS.md, Pi Zero footnote 2) the engine's noise saves 17 to
  36 ms a run on heart and heart-nano; that re-measure found heart-nano's published file aborting on the
  VC4 ([Retro-109](../retros/retro-109-a-cropped-view-passed-the-vc4-fusion-matcher.md)), fixed in the same
  stack. Their published files run the Lua until they are re-published.
* 15 ms stays in Lua, the noise table's crossing included. A variant that leaves the draw in an output
  store, so it never becomes a table, was not built.
