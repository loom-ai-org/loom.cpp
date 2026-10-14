---
type: adr
status: accepted
date: 2026-10-14
domain: performance
tags: [armv6, pi-zero, vfp, runfast, flush-to-zero, numerics, ggml-patch]
---

# ADR-078: RunFast on ARMv6 Compute Threads, Behind a Gate Across the Zoo

Surfaced by [ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md)'s gate 1. Serves
[Epic-05](../epics/epic-05-edge-performance.md) on the tier
[ADR-026](adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) makes the engine's floor.
**Accepted 2026-10-14:** all four gates measured, below. The patch is `ggml-0031`.

## Context

The ARM1176's VFP11 has two modes. In its default IEEE mode it handles denormals and NaNs fully,
some of them by trapping to the kernel's VFP support code. In **RunFast** (FPSCR `FZ | DN`, with the
exception traps off, which is Linux's default) it flushes denormals to zero, returns the default NaN
for any NaN result, and does everything in hardware.

Two things measured on the Pi Zero W make that mode worth a decision:

* **`vmla.f32` runs 1.65x faster in RunFast**, on the same instruction stream: 0.41 GMAC/s against
  0.59-0.68 (`scripts/bench44.c`, eight independent accumulators). IEEE mode does not pipeline
  back-to-back `vmla`. On Citrinet's depthwise schedule a blocked F32 kernel went from 0.67 s to 0.50 s
  (`scripts/bench43.cpp`). This raises the ceiling of every F32 kernel on the board, and ggml's ARMv6
  F32 GEMM (`tinyBLAS_F32_ARMV6`) is the incumbent in most of the board's models.
* **Denormal and NaN arithmetic trap to software** in IEEE mode. A `-inf` in a branchless exp made
  softmax 18x slower (ggml-0029), and the fix was a predicated move added to every element. In RunFast,
  denormals never reach software. Whether the NaN case also stops trapping has to be measured.

**End to end, before any patch**, RunFast set from Python on the calling thread (ggml computes there
at one thread) takes jfk on Citrinet-256-ls's F32 card from 59.7-62.5 s to **20.0-22.5 s**, and its
Q8_0 card from 33.3-34.9 s to 29.3-30.9 s, with the same transcript ids as IEEE mode
([ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#found-while-measuring-gate-2-the-f32-card-was-paying-for-vfp-bounces)).
Most of the F32 gain is bounces. BatchNorm-folded dead channels leave weights at ~1e-33, a dead
channel's accumulator stays that small, and the VFP11 bounces every effective subtraction below 2^-96
in case it underflows (~1.1 us against 8 ns, `scripts/bench47.c`; gate 3 below). That checkpoint shape
is common, so other models on this board may pay the same.

The int8 SMLAD path does not use the VFP, so it gains nothing (~1% on ADR-077's GEMM).

The cost is numerics. A result below 1.18e-38 becomes 0, an input below it is read as 0, and a NaN
loses its payload (it is still a NaN). Every other result is bit-identical. FPSCR is per thread, and
Linux saves it on a context switch.

## Options

1. **Keep IEEE mode.** No numerics change, and the 1.65x stays out of reach.
2. **RunFast for the duration of a graph compute, on every compute thread**: set on entry to ggml's
   compute-thread function, and restored on exit. This is ARMv6 builds only, in a ggml patch. The caller's
   thread is thread 0, so the restore is what keeps the host (Python, numpy, the Lua driver) in IEEE mode.
3. **RunFast process-wide at engine init.** One line, but it changes the host's floating point too.
   Rejected for that reason.
4. **RunFast inside loom's ARMv6 kernels only**, set and restored per call. This is the tightest scope.
   But every op without a loom kernel (softmax, the norms, the unary ops) keeps trapping, and each call
   pays two FPSCR writes.

## Decision

**Option 2, with a runtime switch to turn it off** (an escape hatch for a model the gate does not
cover, not an instrument). Whether it is on by default was left to the gates; they say on (gate 4).

**Gates, in order. Each one can stop the work:**

1. **Does the emulator agree?** The ARMv6 numerics gate runs in qemu, which is bit-identical to the
   board in IEEE mode. Confirm that it honours `FZ | DN` bit for bit too, on a microbenchmark whose
   output depends on both (a denormal product, a NaN payload). If it does not, gate 2 runs on the board.
2. **Numerics across the zoo.** Every model with an ARMv6 oracle runs with and without the switch, at
   today's tolerances: VITS's hash, the ASR transcripts and their tensors, the NER labels. Any model that
   changes gets its difference recorded and explained. Each ASR model also needs the tensor oracle,
   not just the transcript, and each TTS model needs the ASR oracle on its audio.
3. **An audit of what depends on IEEE mode.** Loading Citrinet and running jfk leaves the
   invalid-operation, overflow and underflow flags set in FPSCR in either mode, so something in that
   window already makes NaNs and infinities; the audit names what. It covers `isnan`/`isinf` checks (still work), NaN
   payloads (none known), denormal-range constants such as an epsilon below 1.18e-38, and the
   predicated moves added to keep NaN and denormals out of the VFP (ggml-0029), which may become
   removable.
4. **Speed on the board, cooled and paired**, per model: the Pi Zero set (Citrinet-256-ls, Moonshine
   tiny, sanoTTS heart-nano, the piperlite voices) and the P7 set (VITS, conformer-ctc,
   distilbert-ner). An isolated op's 1.65x is a ceiling. The end-to-end number decides the default.

## Gate 1, measured 2026-10-14: qemu does not honour RunFast bit for bit, so gate 2 runs on the board

`scripts/bench45.c` runs 22 VFP instructions (every float and double arithmetic, convert and compare
instruction loom's ARMv6 build can emit) on 200,000 operands each, drawn to hit what RunFast changes:
denormals, products that underflow, NaNs with payloads, and the values around them. It runs both modes
in one process and hashes each mode's results and flags. Every instruction except `vabs` and `vneg`
changes results or flags between the modes in both environments, so the comparison can fail.
`scripts/bench45_diff.py` compares two dumps case by case. Run in the build container and on the Pi
Zero W:

* **In RunFast, the only differences are signs.** The VFP11 reads a denormal input as **+0** and
  flushes an underflowing result to **+0**, whatever its sign. qemu keeps the sign, which is the ARMv7
  behaviour. So a flushed negative product differs in the sign of its zero, and a division by a flushed
  denormal gives an infinity of the other sign. Separately, the default NaN after `vnmul`, `vnmla` or
  `vmls` is `0x7fc00000` on the board and `0xffc00000` in qemu. No finite non-zero result differs.
* **IEEE mode differs too, which the P7.1 calibration (one VITS hash) could not see.** The VFP11 bounces
  any operation near the bottom of the exponent range to the kernel's VFP support code, not only actual
  denormals. For `vmla`, `vmls` and `vnmla` that code rounds once, close to a fused multiply-add, where
  the hardware and qemu round the product and then the sum. 12 of 20,000 `vmla.f32` cases with
  all-normal operands differ by 1-3 ulp, plus the denormal and underflow cases. NaN signs differ through
  `vsub`, `vnmul` and the negating accumulates. A NaN accumulator plus an overflowing product gives +inf
  on the board and the NaN in qemu. `vcvt.f32.f64` of a signalling NaN comes back unquieted on the
  board. The underflow flag follows different rules. This is [Retro-086](../retros/retro-086-qemu-matches-the-board-only-where-the-vfp-never-bounces.md).

Gate 1 fails as written, so gate 2 runs on the board. Leaving the sign of zero aside, qemu agrees with
the board **more** often in RunFast than in IEEE mode, because nothing bounces in RunFast.

## Gate 2, measured 2026-10-14: outputs identical, two models differ below 1e-6 of the peak

On the Pi Zero W with the patched wheel, every model in both modes, one inference per launch,
`GGML_CPU_RUNFAST=0` against the default (on), two rounds in ABBA order. Both modes reproduce
themselves bit for bit across rounds.

| model | output compared | RunFast against IEEE |
|---|---|---|
| Citrinet-256-ls F32 / Q8_0 | CTC ids (jfk) | identical |
| Moonshine tiny F32 / Q8_0 | transcript (jfk) | identical |
| sanoTTS heart-nano, piperlite amy, de051 | waveform | **bit-identical** |
| VITS (Q4_0) | waveform | **bit-identical** |
| conformer-ctc, distilbert-ner (Q4_0) | transcript, labels | identical |
| sanoTTS heart | waveform | 52,172 of 76,288 floats differ, max 2.5e-7 (7e-7 of peak); 24 int16 samples differ by one LSB; whisper-small hears the same sentence |

**The tensor oracle:** each ASR and NER model's raw output, run alone through the phase probe (a
loom `Session` with an ad-hoc Lua function beside the driver), in both modes. For Citrinet, the argmax
of the probed log-probs decodes to the same CTC ids `infer` returned, so the probe ran the same graph.

| tensor | values | RunFast against IEEE |
|---|---|---|
| Citrinet-256-ls F32 log-probs | 35,466 | 29,727 differ, max 3.05e-5 (9.6e-7 of the peak) |
| Citrinet-256-ls Q8_0 log-probs | 35,466 | bit-identical |
| conformer-ctc logits | 77,900 | bit-identical |
| Moonshine tiny F32 / Q8_0 encoder states | 131,616 | bit-identical |
| distilbert-ner logits | 171 | bit-identical |

Citrinet F32 is the model with tiny values (its dead channels), and it is where the change shows, at
the ninth significant digit. de051 is a German voice given the English id sequence, so its transcript is noise in both modes. That
is fine here, because its output is bit-identical.

## Gate 3, measured 2026-10-14: what depends on IEEE mode

**The patch:** `cmake/patches/ggml-0031-armv6-runfast-compute-threads.patch`. It sets `FZ | DN` on
entry to `ggml_graph_compute_thread`, which every compute thread runs, the caller's included. On exit it
puts back only those two bits, so the cumulative flags survive. It is guarded on `__ARM_ARCH == 6` with
VFP, reads `GGML_CPU_RUNFAST` once at `ggml_cpu_init` (`0` turns it off), and is on by default. On the board, the Python thread's FPSCR reads `FZ = DN = 0` after a compute in either setting.

* **Nothing in ggml or the engine reads the FP environment.** The patched ggml tree has no `fenv` call,
  no other FPSCR access, no NaN payload or NaN bit pattern, and no constant in the denormal range. FP16
  conversion on ARMv6 is integer bit work (ggml-0017) whose float steps stay normal.
* **libm under FZ** (`scripts/bench46.c`, on the board's glibc, 100,000 operands per function, over every
  function the CPU ops call). `expf`, `sqrtf`, `powf`, `exp`, `roundf`, `erff`, `tanhf`, `sinf`, `cosf`,
  `log`, `floorf`, `ceilf`, `expm1f`, `erf` and `sqrt` all give exactly what the flush predicts,
  `f(flush(x))` flushed (`vcvt.f64.f32` flushes a float on its way into a double function). **One
  exception: `logf` of a denormal argument returns -103.97** where the flush predicts -inf (IEEE gives
  -87.3 to -103.3). glibc spots the subnormal by its bits and rescales it in floating point, which FZ
  zeroes. It can only matter for a `LOG` node whose input holds a denormal that no RunFast arithmetic
  produced, such as host-computed data.
* **What bounces, and what RunFast removes** (`scripts/bench47.c`, board). A hardware `vmla.f32` costs
  8 ns in either mode, normal products included, down to 2^-126. In IEEE mode each of these costs
  ~1.1 us instead: an effective subtraction between values at or below 2^-96 (2^-95 does not bounce),
  a NaN operand, and inf - inf. In RunFast all of them cost 8 ns. That corrects
  [Retro-085](../retros/retro-085-the-f32-card-was-slow-because-of-its-dead-channels.md)'s mechanism.
  The dead channels' accumulators cancel in the bounce range, and almost nothing actually underflows.
  It also means **ggml-0029's predicated move** (a masked `-inf` replaced before the exp, because
  inf - inf bounced) is not needed under RunFast. It is a select and costs little, and it keeps IEEE mode
  fast when the switch is off, so it stays.

**The flag census** (an `LD_PRELOAD` shim around `ggml_graph_compute`, with `LOOM_PROFILE` making
every node its own compute, so FPSCR's cumulative flags can be read per node; underflow, overflow,
invalid and divide-by-zero). It finds the same nodes in both modes, and none of them depend on IEEE
semantics:

* **Invalid operation:** one `DIV` in sanoTTS (heart, heart-nano), over the whole output. The output
  holds no NaN, so the NaNs are presumably in samples that get cut away (not traced).
* **Overflow:** `SILU` in heart and conformer-ctc. `exp(-x)` overflows for a large negative x, and
  x / inf is -0.
* **Underflow:** a handful of `SOFT_MAX`, `MUL_MAT`, `CONV_2D`, `CONV_2D_DW` and `SIGMOID` nodes
  (Citrinet F32: 5 of 3,869 nodes in IEEE mode; Moonshine: 56 of 15,336).
* **Nothing** in Citrinet Q8_0, VITS, amy, de051 or distilbert-ner.

So the invalid and overflow flags that loading Citrinet and running jfk left set came from host code,
not from its graph, which raises only underflow.

## Gate 4, measured 2026-10-14: faster on every model, 1.02-2.79x

Same runs as gate 2: the infer call only, model load excluded. Times are the two rounds' launches;
the speed-up is the ratio of their means. The bluetooth-km-switch resident was left running, as in
ADR-077's numbers, and it is the same for both arms.

| model | IEEE mode | RunFast | speed-up |
|---|---|---|---|
| Citrinet-256-ls F32 (jfk, 11 s) | 55.84-55.93 s | **20.06-20.07 s** | **2.79x** |
| Citrinet-256-ls Q8_0 | 31.52-31.57 s | 28.98-29.38 s | 1.08x |
| Moonshine tiny F32 (jfk) | 67.62-67.65 s | 61.11-61.78 s | 1.10x |
| Moonshine tiny Q8_0 | 68.71-68.81 s | 66.44-66.88 s | 1.03x |
| sanoTTS heart-nano | 3.12-3.18 s | 2.90-2.96 s | 1.08x |
| sanoTTS heart | 7.29-7.42 s | 5.49-5.60 s | 1.33x |
| piperlite amy | 45.23-46.03 s | 39.43-39.69 s | 1.15x |
| piperlite de051 | 9.49-9.64 s | 8.36-8.38 s | 1.14x |
| VITS Q4_0 | 66.81-66.95 s | 59.81-60.08 s | 1.12x |
| conformer-ctc Q4_0 (jfk3) | 11.77-11.91 s | 10.76-10.98 s | 1.09x |
| distilbert-ner Q4_0 | 5.53 s | 5.44-5.45 s | 1.02x |

**Every gate passed, so RunFast is on by default on ARMv6**, with `GGML_CPU_RUNFAST=0` as the escape
hatch. Citrinet-256-ls's F32 card is now the faster card on this board, at RTF ~1.8 (ADR-077's
baseline). The other models gain 2-33% from `vmla` pipelining and the occasional bounce.

## Consequences

* Every F32 op on ARMv6 gains without a kernel change. The F32 arms of ADR-077's gate 1 analysis are
  measured in RunFast from now on, which was already done for the depthwise conv.
* The engine's ARMv6 output differs from IEEE mode only where a value reaches the denormal range or a
  NaN is produced. Gate 2 found it in two models of eleven, both below 1e-6 of the peak and neither
in an output: sanoTTS heart (24 int16 samples by one LSB) and Citrinet F32's log-probs.
* The ARMv6 numerics gate in qemu now runs a mode qemu does not reproduce exactly: qemu keeps the sign
  of a flushed zero. A model whose oracle is a hash of raw floats could differ between qemu and the
  board by a zero's sign. The int16 VITS hash cannot, and the transcripts cannot.
* A ggml patch to carry, ARMv6 only, of a few lines (`ggml-0031`, UPSTREAM.md PR 28). It touches the
  compute-thread entry, which every other patch leaves alone.
* The tools stay in `scripts/`: `bench45.c` + `bench45_diff.py` (emulator against board, per
  instruction), `bench46.c` (libm under FZ), `bench47.c` (what bounces and what it costs), and the
  board harness `armv6_runfast_gate.sh` with `armv6_runfast_gate.py`, `armv6_runfast_compare.py` and
  the flag shim `armv6_fpflags.c`.
