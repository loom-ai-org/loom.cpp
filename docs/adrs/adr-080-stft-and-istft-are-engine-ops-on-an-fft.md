---
type: adr
status: accepted
date: 2026-10-15
domain: performance
tags: [fft, stft, istft, dft-basis, armv6, pi-zero, exporter, custom-op, vocoder]
---

# ADR-080: STFT and iSTFT Are Engine Ops on an FFT, Not DFT-Basis Convolutions

[ADR-079](adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md)'s P4.32g, which
the ADR left "an op in the topology, not a kernel, [that] needs its own design". It serves
[Epic-05](../epics/epic-05-edge-performance.md). Per-model work stays in the exporter
([ADR-003](adr-003-per-model-complexity-in-the-exporter.md)). The engine gains one generic transform,
which is per-task and not per-model. **Accepted 2026-10-15 (user)**, with all three of the
recommendations below.

## Context

The exporter writes every STFT and iSTFT in the zoo as a convolution over a windowed DFT basis.
coremltools lowers `torch.stft` that way, and `loom_exporter/istft.py` traces the inverse as three
`conv_transpose1d` (real and imaginary synthesis, and the overlap-added window square), then an ADD and
a DIV. That is N MACs per bin per frame, where an FFT needs O(log N). On the Pi Zero (one thread, wheel
v8, profiled 2026-10-15), these are the DFT-basis seconds:

| model | DFT basis | of the run | form |
|---|---|---|---|
| sanoTTS heart-nano | 2.29 s | 3.18 s | iSTFT, N 1024, hop 256 |
| sanoTTS heart | 2.26 s | 4.96 s | iSTFT, N 1024, hop 256 |
| Silero VAD | 0.98 s | 3.4 s | STFT N 256, reflect pad folded into a 576-tap basis by its exporter |
| Citrinet-256-ls F32 | 0.79 s | 14.6 s | STFT N 512, hop 160 (already on ggml-0034's tile, by phases) |
| WakeHuBERT tiny | 0.46 s | 3.0 s | STFT N 400 |
| conformer-ctc Q4_0 | 0.23 s | 10.3 s | STFT N 512, Q4_0 basis |

`scripts/bench54.cpp` measured the alternatives on those shapes (Pi Zero W, one thread, RunFast; every
arm within 9e-7 of an f64 DFT):

| arm | Citrinet STFT | heart iSTFT |
|---|---|---|
| today: two CONV_2D / three CONV_TRANSPOSE_1D | 786 ms | 2225 ms |
| one CONV_2D with re and im stacked | 737 | |
| the frames as an overlapping view, one MUL_MAT | 1590 | |
| two-stage Cooley-Tukey as MUL_MATs (16 x 32, window and twiddles folded into the matrices) | 656 | |
| a C real FFT in a `ggml_custom_4d` (naive radix 2) | **272** | **184** |

On x86 (one thread) the same C FFT took the STFT from 28 to 5.6 ms and the iSTFT from 44 to 3.1. The
matmul factorization was *slower* there than the direct conv, at 37 ms: five times fewer MACs did not
pay for its small batched GEMMs and the three CONTs between them.

## Options

1. **The exporter alone factors the DFT into MUL_MATs.** There is no engine change and every backend
   keeps the graph. Measured at 1.2x on the board and 0.76x on x86. Rejected.
2. **Engine STFT and ISTFT primitives on a C FFT**, run as a `ggml_custom_4d`. The exporter emits
   them through its existing custom-torch-op route (`loom::istft` -> MIL `loom_istft` -> `ISTFT`, as
   `group_norm_op.py` does for GROUP_NORM).
3. **A ggml op (`GGML_OP_FFT`) in the patch stack.** It is the same kernel as option 2 with a patch to
   carry, and it still runs on the CPU only.

## Decision

**Option 2.** The FFT is the engine's own: `include/loom/ops/fft.h`, a mixed-radix complex FFT
(radices 4, 2, 3, 5, plus a generic odd butterfly) of size N/2 with the real split. The zoo uses
N = 16, 20, 256, 400, 512, 1024 and 1920, which all factor into 2, 3 and 5. Vendoring pocketfft (BSD-3)
was the alternative. It is faster but about 3k lines, against roughly 300 here, and the ARMv6 path may
want its own pass later.

**The iSTFT comes first**, because it is most of the seconds and all six iSTFT families (sanoTTS nano,
F5, CosyVoice3, Chatterbox, Soprano, LFM2.5-Audio) trace one module. The STFT is the second step.
Silero is last, because its exporter folded the reflect pad into the basis, so it needs its own
rewrite.

**`ISTFT`** takes real and imag `[frames, n_fft/2 + 1, batch]`, a window `[n_fft]`, and the attrs
`n_fft`, `hop` and `center`. It returns `[(frames - 1) * hop + n_fft (- n_fft if center), batch]`, which
is exactly what `istft.py`'s module computed: numpy's irfft per frame, times the window, overlap-added,
divided by the overlap-added window square, cropped by `n_fft / 2` at each end when centered. The
division is unguarded, as the module's was, so a zero window sum gives 0/0 in both. **The threads split
the output, not the frames.** Each thread recomputes the few frames that straddle its edges, so the
overlap-add needs no atomics and is bit-identical at any thread count.

**`STFT`** takes a signal `[L, batch]`, already padded by the topology's own PAD as before, and a window
`[n_fft]`, with the attrs `n_fft` and `hop`. It returns real and imag `[(L - n_fft) / hop + 1,
n_fft/2 + 1, batch]`, the two convolutions' layout, so nothing downstream changes. One custom node
writes both halves, and the two outputs are views of it.

**The exporter** routes `ISTFT.forward` to `torch.ops.loom.istft` only under a trace. An eager call (the
fixtures, every reference check) still runs the convolutions, bit for bit. The STFT is caught one level
lower: `torch_patches.py` replaces coremltools' `_stft` helper, which builds the two convolutions, with
one that emits `loom_stft`. It does so for a real input, the one-sided half, no normalization and a
window known at export time, and it falls back to the convolutions for anything else.
`LOOM_EXPORT_NO_FFT=1` keeps the convolutions for both transforms, for an A/B or for an engine older
than these ops. The two STFTs that do not come from `torch.stft` stay convolutions: Silero (its exporter
folds the reflect pad into the basis) and WakeHuBERT's front end, until each gets its own rewrite.

## Consequences

* **A re-export is needed to gain anything**, and a re-exported file needs an engine that has `ISTFT`
  (an older one fails with an unknown op). Each iSTFT GGUF also drops its basis: 4.2 MB for N 1024.
* **On a GPU backend the node runs on the CPU fallback** (`ggml_custom_4d` is a function pointer), which
  costs a spectrogram download and a waveform upload at the end of a synthesis. If that ever measures
  badly, the remedy is a fallback composition built from the window, the P4.7e pattern that ATAN follows.
* The FFT is plain C++, with no short vectors, and in bench54 it won by 2.9x (STFT) and 12x (iSTFT)
  as a naive radix 2. A tuned pass for the VFP11 is the next lever, and only if a profile asks for it.

## Measured 2026-10-15: the board gate

Pi Zero W, one thread, wheel v9 (wheel v8's engine plus the two ops), GGUFs from the same exporter with
and without `LOOM_EXPORT_NO_FFT=1`, two ABBA rounds, one inference per launch:

| model | DFT-basis convs | FFT ops | | RTF | output |
|---|---|---|---|---|---|
| sanoTTS heart-nano (3.06 s of audio) | 3.08 s | **1.00 s** | 3.09x | 1.01 -> 0.33 | max 1.3e-7 of a 0.28 peak; 12 of 73,472 int16 samples move one LSB |
| sanoTTS heart (3.18 s) | 4.92 s | **2.75 s** | 1.79x | 1.55 -> 0.87 | max 1.5e-7 of 0.34; 8 of 76,288 samples one LSB |
| Citrinet-256-ls F32 (jfk, 11.0 s) | 13.03 s | **12.52 s** | 1.04x | 1.18 -> 1.14 | CTC ids identical |

Every arm gave one output hash across its four launches. heart now runs faster than real time on the
board. The waveform moves only where it sits within 1.5e-7 of a 16-bit rounding boundary, far below
anything the ASR oracle could hear. Citrinet saved 0.51 s, against 0.79 s for its two STFT
convolutions in the node-by-node profile and 0.51 s for bench54's conv-minus-FFT difference. The
profile's extra is not attributed.

On x86, the engine test (`tests/ci/test_fft_ops.cpp`) holds the FFT to 1.8e-7 of an f64 DFT at every
zoo size, and holds each op against the convolutions it replaces and against f64. It also checks 1 vs
4 threads bit for bit, and four sabotages (a butterfly sign, the radix-5 path, the ISTFT thread split,
the STFT frame offset) each turned it red. Re-exported on x86: heart-nano and heart match the
convolutions to 1.0e-7 and 8.9e-8, and Citrinet's CTC ids are identical. The exporter's CI suite passed
(1249) with both ops on by default.
