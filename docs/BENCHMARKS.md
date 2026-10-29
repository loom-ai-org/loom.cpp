---
type: reference
domain: performance
tags: [benchmarks, onnxruntime, raspberry-pi, pi-4, pi-zero, vc4, qpu, x86, methodology]
last_updated: 2026-10-29
---

# Benchmarks: loom against onnxruntime, and on the Raspberry Pis

The measured tables behind the README's summary, with the protocol, provenance (engine build, GGUF md5s,
onnxruntime build) and caveats of each. Moved here from the README on 2026-10-24, unchanged. How each number
was arrived at, and what changed it over time, is [Epic-05](epics/epic-05-edge-performance.md).

* [Raspberry Pi 4: F32 and Q8_0_PC against onnxruntime](#raspberry-pi-4) (2026-10-23/24, the newest)
* [Raspberry Pi Zero W: CPU against CPU+VC4](#raspberry-pi-zero-w-cpu-against-cpuvc4) (2026-10-30, P4.47)
* [Raspberry Pi Zero W: F32 against Q8_0_PC](#raspberry-pi-zero-w) (1.0.0-rc18)
* [Three tasks on three machines](#three-tasks-on-three-machines) (x86 and the Pi 4, 2026-09-02, an older engine)
* [What these numbers are not](#what-these-numbers-are-not): the methodology every table here follows

## Raspberry Pi 4

**Q8_0_PC is Q8_0 with one scale per output channel** (`--quantize Q8_0_PC`,
[ADR-076](adrs/adr-076-quantized-gemms-reach-xnnpacks-microkernels-through-a-ggml-buffer-type.md)). On the Pi 4's
Cortex-A72 (an aarch64 core without DOTPROD) and on ARMv6 it takes loom's integer GEMM paths; elsewhere it runs ggml's
Q8_0 kernels. Against it the Pi 4 runs onnxruntime's int8 models, the ones an onnxruntime user would pick. **Both halves of
the Pi 4 row were re-measured on the night of 2026-10-23 on one engine build not yet released**, main `159f4cb` plus
`ggml-0053`, which carries the A72 int8 GEMM (loom.cpp #90), the A72 F32 GEMM (#91), that GEMM for the convolutions too
(P4.39, `ggml-0049`) and the direct convolution on per-thread staged slices (P4.40, `ggml-0053`, #97). Same protocol as the
[three-machine table](#three-tasks-on-three-machines): 4 threads (the board's default), each engine launched once per arm, 5 paired rounds in ABBA order, cooled to
60 C before every arm, median per engine; onnxruntime is the 1.28.0 PyPI wheel. The ratio is onnx/loom, so **`>1.00x`
means loom is faster**, and **eleven of the twelve cells resolve**: each one's paired p10 and p90 sit on the same side
of 1.00x. whisper-small's F32 cell does not (p10 0.99, p90 1.01), so read it as parity.

**Each time is followed by its real-time factor (RTF) in brackets:** the time divided by the length of the audio, which
for ASR, VAD and wake word is the input clip and for TTS is the audio the model produced. Below 1.00 is faster than real
time. The LM row has no audio, so its time stands alone.

| Raspberry Pi 4B, 4 threads | loom F32 | onnxruntime FP32 | ratio | loom Q8_0_PC | onnxruntime int8 | ratio |
|---|---|---|---|---|---|---|
| VITS piper en-GB miro (TTS, 3.32 s of audio) | 0.802 s (0.24) | 1.072 s (0.32) | **1.34x** | 0.821 s (0.25) | 4.117 s (1.24) | **5.03x** |
| Citrinet-256-ls (ASR, jfk 11.0 s) | 0.454 s (0.04) | 0.294 s (0.03)¹ | 0.65x | 0.459 s (0.04) | 4.300 s (0.39)¹ | **9.31x** |
| Moonshine tiny (ASR, jfk 11.0 s) | 1.335 s (0.12) | 1.504 s (0.14) | **1.13x** | 0.848 s (0.08) | 1.485 s (0.13) | **1.75x** |
| whisper-small (ASR, jfk 11.0 s) | 20.73 s (1.88) | 20.63 s (1.88) | 1.00x | 12.52 s (1.14) | 14.74 s (1.34) | **1.17x** |
| Qwen3-0.6B-Base (LM, 24 tokens) | 16.59 s | 17.56 s | **1.06x** | 4.15 s | 14.28 s | **3.45x** |
| Pocket TTS (TTS, 4.80 s of audio)² | 9.107 s (1.90) | 18.77 s (3.91) | **2.06x** | 4.248 s (0.88) | 13.88 s (2.89) | **3.27x** |
| Pocket TTS 18m refl (TTS, 4.80 s of audio)³ | 2.412 s (0.50) | — | — | 1.439 s (0.30) | — | — |
| Pocket TTS 7m refl (TTS, 4.80 s of audio)³ | 1.275 s (0.27) | — | — | 1.067 s (0.22) | — | — |
| Pocket TTS 2m refl (TTS, 4.80 s of audio)³ | 0.619 s (0.13) | — | — | 0.593 s (0.12) | — | — |

¹ onnxruntime is timed from NeMo's mel features, computed outside the timer; loom's time includes its front end.
² Measured with the rest of each half, and each launch is the mean of 3
syntheses after an untimed one (`scripts/bench_driver.cpp` against `scripts/bench_pocket_onnx.py`). There is no official ONNX Pocket TTS, so the
onnxruntime side is the community export [`KevinAHM/pocket-tts-onnx`](https://huggingface.co/KevinAHM/pocket-tts-onnx)
at `58a6d00` (its fp32 and int8 files) through its own `generate()`. That export holds the `english_2026-04`
weights, an older release of the same architecture as loom's `english_2026-09`, and it keeps decoding on a second
thread and attends over a fixed 1000-position cache. Both sides run exactly 60 frames, with the EOS head disarmed, from
the same 28 text ids and a 126-row `alba` voice.
³ Kyutai's distilled `english_{18m,7m,2m}_refl_2026-10` (ADR-084), timed like the Pocket TTS row. There is no ONNX export
of them, so they have no onnxruntime cells. Their own F32/Q8_0_PC ratios are in the last bullet below.

* **Both engines did the same work in every pair**, checked by one untimed launch of each arm: the same 73 216 VITS
  samples, the same 24 tokens, and the same transcript at each precision except one word each of Citrinet (loom Q8_0_PC
  hears "asks", onnxruntime int8 "askd ... ask") and of Moonshine at Q8_0_PC (loom transcribes jfk exactly, onnxruntime
  int8 hears "as" for "ask", as loom's own Q8_0_PC did before the A72 int8 GEMM). **loom's Q8_0_PC Qwen3 decodes the
  same tokens as its F32**, where onnxruntime's int8 model decodes different ones from its own FP32 (`220, 16, 15, ...` against `12095, 13, 576, ...`).
* **onnxruntime's int8 is slower than its FP32 on both convolutional models** — 3.8x on VITS, 14.6x on Citrinet —
  which is what makes those two Q8_0_PC cells large. loom's Q8_0_PC is within 3% of its own F32 there, and is a win
  by itself on the transformer models: 1.57x on Moonshine, 1.66x on whisper and 4.00x on Qwen3. Before the A72 int8
  GEMM, whisper's was 1.13x and the cell read 0.64x: ggml's Q8_0 kernel needs DOTPROD, which the A72 lacks.
* **Where the int8 models come from:** Qwen3, whisper and the Moonshine decoder are onnx-community's published int8
  files (`model_int8`, `encoder_model_int8` + `decoder_model_merged_int8`); VITS and Citrinet are `quantize_dynamic`
  (QInt8) of the same FP32 models the left half runs. Moonshine's encoder is `quantize_dynamic` with MatMul only,
  because the published int8 encoder's quantized conv stem garbles jfk.
* The engine is `integration/rc18` at `d66c2ce`, whose code is 1.0.0-rc18's except for #83's fix to how the GPU wheels
  find the ggml patches, which a CPU build does not reach. The GGUFs are md5 `3f25d02d`/`6e095a95` (VITS), `0711bf17`/`37a8d3b6`
  (Citrinet), `4831ba56`/`9284eaa8` (Moonshine), `7b6af094`/`39077edf` (whisper) and `4882ce99`/`b93d7779` (Qwen3),
  F32/Q8_0_PC. Both sides run from `scripts/bench_{vits,asr,lm}_loom.cpp` and `scripts/bench_onnx_tasks.py`, whose
  `ctc` and `moonshine` tasks are new with this table. **Both halves** ran the same harness, binaries and GGUFs with
  only the engine libraries swapped for main `159f4cb` + `ggml-0053`: loom.cpp #89, #90 and #91 (`ggml-0045`-`0048`),
  `ggml-0049`, P4.41's `ggml-0050`-`0052` (two of them ARMv6-only) and `ggml-0053`. Every Q8_0_PC arm placed its
  per-channel weights in the A72's `CPU_NEON_I8` buffer. whisper's and Moonshine's F32 linears and whisper Q8_0_PC's
  attention products (which are F32) take the outer-product F32 GEMM. Since `ggml-0049` so do the im2col
  convolutions: VITS's text encoder and flow, and Citrinet's. Since `ggml-0053` the direct ones (VITS's vocoder, and
  some of Citrinet's and of Pocket TTS's Mimi decoder) run an 8 x 12 tile over per-thread staged slices of their
  input ([ADR-086](adrs/adr-086-the-direct-convolution-stages-its-activation-per-thread.md)), with VITS's audio
  byte-identical either way. Against rc18's F32 half that moves whisper 0.80x → 1.00x, Moonshine 1.03x → 1.13x,
  VITS 0.98x → 1.34x and Citrinet 0.57x → 0.65x; whisper Q8_0_PC goes 0.64x → 1.17x. `ggml-0053` alone took VITS
  from 1.03x to 1.34x, and from 3.90x to 5.03x at Q8_0_PC, whose convolution kernels run at F32.
* **Pocket TTS did the same work too:** 115 200 samples on every arm. The onnxruntime outputs at both precisions, and
  the Pi Zero's Q8_0_PC output below, transcribe under whisper-small as the sentence spoken, word for word (it spells
  "grey" as "gray"). Its GGUFs are
  md5 `26aa0137`/`6e71ec37`, F32/Q8_0_PC, the files `loom-ai-org/pocket-tts-loom` publishes.
* **The small Pocket TTS models are weighed against their own Q8_0_PC.** They ran in the same session, through the same
  `bench_driver` call as the Pocket TTS row, with the same 5 paired rounds, cooled to 60 C before every arm. F32/Q8_0_PC
  is **1.68x** on the 18m (p10-p90 1.63-1.73) and **1.21x** on the 7m (1.20-1.23). The 2m's 1.03x does not resolve
  (0.99-1.15). At F32 the 18m model is 3.8x faster than the full one and the 2m 14.7x. Every arm produced 115 200
  samples, and whisper-small hears all six outputs as the sentence spoken (the 7m Q8_0_PC writes "gray" for
  "grey"). The GGUFs are fresh exports on loom-exporter main `89750ac`, whose Mimi decoder attends over a window
  (P4.41), so they are not the Pi Zero rows' files and are not on the Hub yet: md5 `4564c993`/`740ec54f` (18m),
  `45aabdb0`/`f0b44c07` (7m) and `8a2ff5e2`/`d4d4e23c` (2m), F32/Q8_0_PC.

## Raspberry Pi Zero W: CPU against CPU+VC4

The Pi Zero W's GPU, a VideoCore IV with 12 QPUs, as an accelerator beside its one ARM11 core
([ADR-088](adrs/adr-088-the-pi-zero-qpus-are-an-accelerator-beside-the-cpu.md) to
[ADR-091](adrs/adr-091-the-qpus-take-the-transposed-convolutions-and-the-weights-leave-the-pool.md)): on a board booted
without `vc4-kms-v3d`, `device="auto"` runs the convolutions, transposed convolutions and matrix products ggml-vc4 takes
on the QPUs and every other node on the CPU. Measured on 2026-10-29/30 with the P4.47 build (loom.cpp #106 at
`6ffd561`, not released), the CPU arm (`LOOM_DEVICE=cpu`) on the same build: the [table below](#raspberry-pi-zero-w)'s
models and inputs, and the three Pocket TTS models of the [Pi 4 notes](#raspberry-pi-4) (the
P4.41 exports, md5 above: 60 frames with the end-of-speech check off, seed 1, the built-in voice, 4.80 s of audio), each
at F32 and Q8_0_PC on both devices, one process per arm timing its first call after loading, 3 rounds with the four
arms' order reversed every other round, a 3 s settle between launches, median per arm, nothing polling the board.
Seconds, RTF in brackets; each device's faster precision in bold, and the speedup is between those two.

| Raspberry Pi Zero W | CPU F32 | CPU Q8_0_PC | CPU+VC4 F32 | CPU+VC4 Q8_0_PC | speedup |
|---|---|---|---|---|---|
| sanoTTS amy en 1.46M (TTS, 5.90 s of audio) | **13.54 s (2.30)** | 13.62 s (2.31) | 4.10 s (0.70) | **4.08 s (0.69)** | 3.32x |
| sanoTTS de thorsten 0.51M (TTS, 4.40 s of audio) | 3.19 s (0.72) | **3.17 s (0.72)** | **1.53 s (0.35)** | 1.55 s (0.35) | 2.07x |
| sanoTTS heart (TTS, 3.18 s of audio) | 1.98 s (0.62) | **1.71 s (0.54)** | **0.92 s (0.29)** | 0.96 s (0.30) | 1.85x |
| sanoTTS heartnano (TTS, 3.06 s of audio) | 0.67 s (0.22) | **0.67 s (0.22)** | 0.57 s (0.19) | **0.57 s (0.19)** | 1.17x |
| VITS piper en-GB miro (TTS, 3.39 s of audio) | 19.78 s (5.83) | **19.77 s (5.83)** | **5.88 s (1.74)** | 6.20 s (1.83) | 3.36x |
| Pocket TTS 18m (TTS, 4.80 s of audio) | 31.10 s (6.48) | **23.27 s (4.85)** | 24.98 s (5.20) | **18.33 s (3.82)** | 1.27x |
| Pocket TTS 7m (TTS, 4.80 s of audio) | 20.70 s (4.31) | **18.91 s (3.94)** | 14.91 s (3.11) | **14.04 s (2.92)** | 1.35x |
| Pocket TTS 2m (TTS, 4.80 s of audio) | **8.85 s (1.84)** | 9.07 s (1.89) | **6.66 s (1.39)** | 7.08 s (1.48) | 1.33x |
| Citrinet-256-ls (ASR, jfk 11.0 s) | 9.81 s (0.89) | **7.14 s (0.65)** | **3.38 s (0.31)** | 3.49 s (0.32) | 2.11x |
| Moonshine tiny (ASR, jfk 11.0 s) | 25.50 s (2.32) | **19.77 s (1.80)** | 16.66 s (1.51) | **12.90 s (1.17)** | 1.53x |
| Conformer-CTC small (ASR, 3.0 s of jfk) | 5.88 s (1.96) | **5.70 s (1.90)** | 4.23 s (1.41) | **4.16 s (1.39)** | 1.37x |
| Silero VAD (jfk 11.0 s) | 2.36 s (0.21) | **1.99 s (0.18)** | 2.12 s (0.19) | **1.74 s (0.16)** | 1.15x |
| WakeHuBERT tiny (wake word, jfk 11.0 s) | 2.08 s (0.19) | **1.83 s (0.17)** | **1.36 s (0.12)** | 1.41 s (0.13) | 1.34x |

* **Every arm's output hashed the same in all three rounds**, and the CPU+VC4 arms' outputs against the CPU's are
  [ADR-091](adrs/adr-091-the-qpus-take-the-transposed-convolutions-and-the-weights-leave-the-pool.md)'s: transcripts
  identical, Citrinet's CTC ids with one near-tie, TTS within 50-98 dB SNR (the QPUs' float rounding is not IEEE).
  Pocket TTS samples its latents in a loop that turns any rounding difference into another waveform (the CPU's own
  Q8_0_PC output is at -1.4 dB against its F32 output), so its outputs are checked by ear instead: whisper base
  transcribes all twelve, every model at both precisions on both devices, as the sentence spoken.
* **The CPU column is faster than the rc18 table below** (amy 13.54 s against 14.95 s, Moonshine 19.77 s against
  24.32 s), on a build three weeks newer than that release's wheel; the rc18 table is kept as it was measured.
* **What the QPUs do not take stays at the CPU's speed**: Silero VAD's LSTM and Moonshine's decoder are most of their
  time, hence 1.15x and 1.53x; VITS and amy are almost all convolutions, hence 3.3-3.4x. Pocket TTS's latent loop
  takes one frame a step, a product of one column the QPUs decline, so they take its prompt and its Mimi decoder:
  1.27-1.35x. Its KV cache stays in RAM beside them, leaving the GPU's pool to the compute buffers
  ([ADR-091](adrs/adr-091-the-qpus-take-the-transposed-convolutions-and-the-weights-leave-the-pool.md) decisions 9
  and 10): held in the pool, 18m's 32 MB cache crowded its decoder's convolutions out, and CPU+VC4 ran no faster than the
  CPU (22.7 s, one run).
* **A first inference pays for packing the weights into the QPUs' layouts** (kept in the GPU's 64 MB pool while it has
  room): Conformer-CTC's second inference is 3.55 s (ADR-091's run) against this table's 4.16 s.

## Raspberry Pi Zero W

The Pi Zero W has no onnxruntime build, since onnxruntime ships no ARMv6 wheel. The Pi 4's new GEMMs and
`ggml-0053` are AArch64-only, so this table was not re-run; P4.41's ARMv6 patches are measured in Epic-05. It weighs loom's two precisions
against each other on the 1.0.0-rc18 ARMv6 wheel, with RTFs as above: its one core, each model launched once per arm through
`loom-py`, 3 rounds with the arms' order flipped every round and a 3 s settle between launches, median per arm, nothing
polling the board while it runs. The ratio is F32/Q8_0_PC, so **`>1.00x` means Q8_0_PC is faster**; a ratio in bold
resolves, its three paired rounds all on the same side of 1.00x.

| Raspberry Pi Zero W, 1 core | loom F32 | loom Q8_0_PC | ratio | paired rounds |
|---|---|---|---|---|
| Citrinet-256-ls (ASR, jfk 11.0 s) | 10.96 s (1.00) | 8.09 s (0.74) | **1.35x** | 1.35–1.37x |
| Moonshine tiny (ASR, jfk 11.0 s) | 30.41 s (2.76) | 24.32 s (2.21) | **1.25x** | 1.24–1.26x |
| Silero VAD (jfk 11.0 s) | 2.662 s (0.24) | 2.203 s (0.20) | **1.21x** | 1.14–1.22x |
| sanoTTS heart (TTS, 3.18 s of audio) | 2.308 s (0.73) | 2.030 s (0.64) | **1.14x** | 1.13–1.25x |
| WakeHuBERT tiny (wake word, jfk 11.0 s) | 2.383 s (0.22) | 2.102 s (0.19) | **1.13x** | 1.12–1.16x |
| Conformer-CTC small (ASR, 3.0 s of jfk) | 6.919 s (2.31) | 6.672 s (2.22) | **1.04x** | 1.02–1.07x |
| sanoTTS amy en 1.46M (TTS, 5.90 s of audio) | 15.49 s (2.63) | 14.95 s (2.53) | 1.04x | 0.96–1.05x |
| sanoTTS de thorsten 0.51M (TTS, 4.40 s of audio) | 3.490 s (0.79) | 3.553 s (0.81) | 0.98x | 0.96–1.01x |
| VITS piper en-GB miro (TTS, 3.39 s of audio) | 21.96 s (6.48) | 22.16 s (6.51) | 0.99x | 0.98–1.00x |
| sanoTTS heartnano (TTS, 3.06 s of audio), control | 0.847 s (0.28) | 0.862 s (0.28) | 0.98x | 0.96–1.01x |
| Pocket TTS (TTS, 4.80 s of audio) | does not fit | 115.0 s (23.96) | — | — |
| Pocket TTS 18m refl (TTS, 4.80 s of audio) | 67.02 s (13.96) | 58.85 s (12.26) | **1.14x** | 1.14–1.15x |
| Pocket TTS 7m refl (TTS, 4.80 s of audio) | 46.60 s (9.71) | 45.16 s (9.41) | 1.03x | 0.98–1.03x |
| Pocket TTS 2m refl (TTS, 4.80 s of audio) | 19.14 s (3.99) | 19.57 s (4.08) | **0.98x** | 0.98–0.99x |

* **sanoTTS heartnano is the noise floor.** None of its tensors is large enough to quantize, so its Q8_0_PC export is
  byte-identical to its F32 one (md5 `75579b65` for both). Its 0.96–1.01x is what two runs of the same file differ by on
  this board, and the three unresolved TTS rows sit inside it.
* **Q8_0_PC wins on every ASR, VAD and wake-word model, and on sanoTTS heart.** The other three TTS models sit inside
  the noise floor. VITS doesn't gain on the Pi 4's Q8_0_PC either.
* **The work is the same, checked on every launch:** the two ASR transcripts and Citrinet's 47 CTC ids hash
  identically at both precisions, and every arm's output hash is identical across its three rounds. Each TTS model's
  Q8_0_PC arm produces one 256-sample frame more than its F32 arm.
* **DistilBERT-NER is left out:** its F32 GGUF is 261 MB, and with Python beside it that is more than the board's
  427 MB. The first attempt drove the board into swap until sshd stopped answering.
* **Pocket TTS has no F32 arm for the same reason:** its F32 file is 405 MB. Its 130 MB Q8_0_PC file was timed alone,
  on 2026-10-21 with the 1.0.0-rc18 release wheel. The time is the median of 3 launches, each launch's first call, on the
  Pi 4 row's 60 frames. The output hashed the same in all three. 79 of its weights (97 MB) take the ARMv6 int8 path.
* **The small Pocket TTS models do fit at F32.** These are Kyutai's distilled `english_{18m,7m,2m}_refl_2026-10`
  (ADR-084): 77, 29 and 8.5 MB at F32, 31, 11 and 4.2 MB at Q8_0_PC. They were measured on 2026-10-23 with the same
  rc18 wheel and the same call as the Pocket TTS row (60 frames, EOS disarmed, seed 1, `alba`), timing each
  launch's first call. The token ids are fixed, so the text rules ADR-084 adds to the engine don't come into it.
  Every arm's output hashed the same in all three rounds, and whisper-small hears all six as the sentence
  spoken (the 7m Q8_0_PC writes "gray" for "grey"). Q8_0_PC pays off only on the 18m model. The 2m's 0.98x
  resolves by the rule above but sits inside heartnano's noise floor. The GGUFs are md5 `d0147aa6`/`2eb67407`
  (18m), `fa467304`/`98c9c51e` (7m) and `be656b10`/`89835bf9` (2m), F32/Q8_0_PC.
* The wheel is the `integration/rc18` ARMv6 build at engine `d66c2ce` (wheels CI run 39510002375). The GGUFs are
  md5 `0711bf17`/`37a8d3b6` (Citrinet), `4831ba56`/`9284eaa8` (Moonshine), `ae4a116f`/`a815c984` (Silero VAD),
  `b3df3baf`/`14df8278` (heart), `be98f1a8`/`6586f559` (WakeHuBERT), `9ae1d8a2`/`241b1bca` (Conformer-CTC),
  `1d2e4e7b`/`cc149c53` (amy), `715fc16d`/`1a7596ea` (thorsten) and `3f25d02d`/`6e095a95` (VITS), F32/Q8_0_PC.

## Three tasks on three machines

**An older table: the engine has moved since** (the Pi 4 row is superseded by the [Raspberry Pi 4](#raspberry-pi-4)
section above), kept for its x86 rows and for the measurements its notes record.

The reference question for an edge runtime: **the same checkpoint, on the same machine, at the same
thread count — how does loom compare to onnxruntime?** Three tasks, because no single one is
representative: an all-convolutional vocoder, an encoder-decoder ASR model, and an autoregressive LM.

**`>1.00x` means loom is faster.** **Every TTS and LM cell was re-sampled per launch on 2026-09-02**
and those are the numbers below, on all three machines; the ASR column is from 2026-08-29 and is not
re-sampled here. **Both engines run back to back on the machine in the row** — one second apart on x86,
and on the Pi cooled to a fixed 60 C before each arm, which is that board's standing protocol. See
[what these numbers are not](#what-these-numbers-are-not).

The **default** row for each machine is the one at that machine's physical core count, which is what
the engine now uses when `$LOOM_N_THREADS` is unset.

| machine | arch | threads | onnxruntime 1.28.0 build | TTS<br>VITS (piper en-GB) | LM<br>Qwen3-0.6B | ASR<br>whisper-small |
|---|---|---|---|---|---|---|
| Intel Core Ultra 9 285K | x86-64 | 4 | conda-forge | **1.05x** | 1.02x | 0.64x |
| Intel Core Ultra 9 285K | x86-64 | **24 (default)** | conda-forge | 1.20x | 0.99x | **1.37x** |
| AMD Ryzen 3 3250U | x86-64 | **2 (default)** | PyPI wheel | 0.98x | **1.01x** | 0.80x |
| AMD Ryzen 3 3250U | x86-64 | 4 (all) | PyPI wheel | 1.00x | **1.02x** | 0.81x |
| Raspberry Pi 4B | aarch64 | **4 (default)** | PyPI wheel | 0.96x | **1.03x** | 0.57x |

**Half of the ten TTS and LM cells resolve, and those are the bolded ones.** In those two columns a
cell is bolded when the paired p10 and p90 over its launches both sit on one side of 1.00x; the rest
straddle it, which means the measurement did not separate the two engines — not that they tied at
exactly the printed number. **Read an unbolded TTS or LM cell as parity.** The five that resolve are
the 285K's four-thread TTS (p10 1.03, p90 1.17), both Ryzen LM cells (p10 1.00, p90 1.02-1.03), and
both Pi cells — its LM at p10 1.03 / p90 1.04 and its TTS at p10 0.95 / p90 0.98, the one cell in the
table that resolves as a **loss**. In the ASR column bold keeps its older meaning of simply `>1.00x`.

**The Pi is the best-behaved machine in this table, not the worst.** Cooled to 60 C before every arm it
holds a 1.02-1.05x spread per engine across launches, where the 285K's hybrid placement gives 1.45x —
so the board that needs the most protocol produces the tightest numbers once it has it, and both its
cells resolve. Its LM arms run `nrun=2` on both sides rather than the x86 rows' 3, because a generation
there is 47 s; the estimator is matched within the row, which is what the ratio depends on.

**Every TTS and LM cell is now a MEAN over separate process LAUNCHES** — per engine, 31 launches for
TTS on x86 and 15 on the Pi; 21 for the LM on the 285K, 15 on the Ryzen and 7 on the Pi, where a single
generation is 47 s. A mean rather than a median because at 24 threads both engines' VITS time splits
into two per-launch modes about 1.4x apart at roughly even odds, and a median over that is a coin flip
between them that moves the cell by the whole 1.4x. **This retires the Pi's old "report the fastest
run" special case**: cooling to a fixed temperature before every arm is what makes its launches
comparable to each other, so it can use the same estimator as the rest of the table. The ASR column is
still a median over launches (9 on the 285K, 7 on the Ryzen, 5 cooled pairs on the Pi), taken
2026-08-29.

**The re-sample confirmed the table rather than overturning it.** No TTS or LM cell on any of the three
machines moved by more than 0.04, the largest being the Ryzen's two-thread LM at 1.05x → 1.01x. Three
faults were fixed on the way there and none of them was worth much: `bench_onnx_tasks.py`'s `vits` task
had no warm-up while loom's did, `bench_vits_loom.cpp` took `ts[ts.size()/2]` rather than a median, and
the LM's equal-work check was asserted rather than printed. What *did* move cells is the second of two
back-to-back launches paying for the first — see
[Retro-025](retros/retro-025-the-arm-that-ran-second-paid-for-the-first.md).

**All three machines were checked to be running byte-identical inputs on both sides**, and they are the
published artifacts rather than a working export: the VITS GGUF is `md5 949dd988…` and the
Qwen3-0.6B-Base GGUF is `md5 4882ce99…` (the files in `hf-models/`), with the onnx models matching
across all three boxes as well (`md5 395ed93d…` and `md5 44961f8a…`). The whisper GGUF was checked the
same way when the ASR column was taken (`md5 1deaac83…`). The Pi additionally reproduces the *same
audio digest* from the 62.8 MB post-P4.28 export and the 81.7 MB one before it — `fnv1a=aa320f8a…` from
both — which is what says the export that halved the file did not change the waveform.

**The Pi row was regressed by one of this repository's own ggml patches, and that is now fixed.**
`ggml-0011` — worth 2.15x at whisper's `A@V` shape on AVX2 — cost a Cortex-A72 **1.3-1.75x on every
f32 GEMM shape measured**, because it inlined a scalar tail loop into the register tile. NEON's
`KN = 4` divides every contraction these models have, so that tail never even ran; its presence alone
changed what GCC's allocator did with the tile. Dispatching it out of line, before the tile, restores
aarch64 without moving x86 at all:

| Raspberry Pi 4B, 4 threads | published 2026-08-24 | with the regression | **fixed** |
|---|---|---|---|
| TTS | 0.98x | 0.84x | **0.96x** |
| LM | 1.08x | 1.05x | **1.06x** |
| ASR | 0.58x | 0.45x | **0.57x** |

It was measured on x86 and shipped to every architecture, and **the Pi was not re-measured after it
landed** — which is the whole of why it survived four days.

**And it happened again with the next patch — now fixed, and the axis was not the ISA.** `ggml-0012` —
worth **2.75x** at whisper's `QK^T` on x86 — was checked on this board at that same shape and found
neutral, which it is. It was never checked at any other shape, and it cost the Pi **2.4%** on VITS,
most of it in the *convolution*, because this repository lowers convolution through the same `sgemm`.
**That is why the Pi's TTS cell moved 0.96x -> 0.93x between 2026-08-29 and 2026-08-30 with no change
to the harness.**

What the patch removes is a per-*output* cost and what it adds is per-*work*, so its benefit decays as
`1/k` while its cost does not: over a `k` sweep it is 2.04x at `k = 64` and 0.97x at `k = 2304` on the
285K, 1.01x and 0.90x on the Pi — monotone on both, crossing 1.0 at about the same place. **So the
predicate is `k`, not the ISA**, and `!defined(__aarch64__)` would have been wrong in both directions.
Gating the ragged prefix on `k <= 256` keeps every x86 win (whisper's `QK^T` improves further, 1.81x to
**1.94x**) and returns the Pi to parity with the unpatched tree. P4.26 in
[Epic-05](epics/epic-05-edge-performance.md) has the sweeps;
[Retro-022](retros/retro-022-a-benefit-and-a-cost-on-the-same-axis.md) is the lesson, which
generalises [Retro-019](retros/retro-019-a-patch-measured-on-one-isa.md)'s "measure every ISA a
patch is enabled for" to **every regime it is enabled for**.

**The Pi's LM and ASR cells did not carry it**, which was the open question when this was found: the
LM's decode step has `ne1 = 1` and never reaches the blocked GEMM, and whisper's own `QK^T` is `k = 64`
— the regime the branch is *for* — and measures neutral on that board at the op level and unresolved
end to end (ASR median 1.007 over four paired rounds, p10 0.989, p90 1.016; LM median 0.995, p10 0.983,
p90 1.003 over six).

**The Pi's TTS cell is re-measured, and it reads 0.95x** — it was 0.93x with the regression and 0.96x
before it. Four rounds, both engines back to back on the board, **cooled to a fixed 60 C before every
arm** and the arm order alternated round to round, onnxruntime normalised to loom's 73 216 samples:

| round | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| 1 (loom first) | 1.1154 s | 1.1172 s | 1.002 |
| 2 (onnx first) | 1.1118 s | 1.0567 s | 0.950 |
| 3 (loom first) | 1.1193 s | 1.0609 s | 0.948 |
| 4 (onnx first) | 1.1147 s | 1.0552 s | 0.947 |

**Round 1's onnxruntime arm is the session's first and pays the 63 MB model's cold page cache**; it is
kept in the table and in the median (0.949) rather than dropped, because dropping the round that
disagrees is how a number stops being reproducible. The other three span 0.5% and loom's four span
0.7%, which is what the cooling protocol is for.

**Re-sampled at 15 cooled rounds on 2026-09-02 the cell reads 0.96x**, which is the number in the
table: loom mean 1.1151 s over 15 launches (spread 1.05x), onnxruntime 1.0732 s (1.03x), paired p10
0.947 and p90 0.977, so it is one of the two cells in this table that resolve as a **loss** rather than
as parity. Four rounds had already put it within 0.01 of that, which is the useful part — on this board
the protocol, not the sample size, is what the number depends on.

**TTS and the LM are at parity on x86; ASR is still behind, and on the Pi loom wins the LM and loses
TTS.** Most x86 cells sit within a few percent of 1.00x, which on those machines is the noise floor
rather than a result — read them as parity, not as wins. The Pi is the exception in both directions:
its two cells are the tightest in the table and both resolve. The caveats below matter as much as the
numbers.

* **TTS is at parity everywhere**, 0.96x to 1.20x. That is what P4.14/P4.15 was for: a built-in F32
  GEMM micro-kernel, four convolution patches to the pinned `ggml`, and a duplicated text encoder
  removed from the export. Two cells resolve: the 285K's four-thread one at **1.05x**, and the Pi's at
  **0.96x**, which is the clearest *loss* in the table and the only place TTS is meaningfully behind.
  The 285K's 24-thread cell is the largest number in the column and also the one the launch lottery
  makes least certain (p10 0.94, p90 1.73 over 31 paired launches), so 1.20x is where the mean fell and
  not a win to quote. The Pi's cell read 0.84x for four days while `ggml-0011` was regressing it
  (above).
* **The LM is at parity**, a few percent either way, and it was the one task `ggml-0011`'s aarch64
  regression left alone — a decode step's `mul_mat` has `ne1 = 1` and never reaches the blocked GEMM,
  which is the control that made that diagnosis trustworthy. It only just became so — until 2026-08-23 the
  engine called every causal-LM driver's `infer` rather than its `infer_with_past`, so the host re-fed
  a growing prompt and each token recomputed the whole sequence. That was worth 2.83x. **Its one loss
  is the 24-thread cell, and that is the LM at a thread count that does not suit it**: a decode step's
  `mul_mat` has `ne1 = 1`, so this task peaks at 8 threads and plateaus after. It shows up as *spread* —
  loom ranges 22.0-28.6 tok/s over 21 launches there where onnxruntime holds 27.3-28.0. **Re-sampled
  per launch that cell is 0.99x rather than 0.96x**, which is parity and not a loss; what moved it was
  letting the machine settle between arms, worth 0.94x → 0.99x on its own. **Three LM cells resolve in
  loom's favour** — both Ryzen ones and the Pi's 1.03x — and they are the tightest measurements in this
  table (1.02-1.03x spread per engine). The LM is the one task where loom is ahead on the two machines
  a person would actually deploy on.
* **ASR is the one still behind — 1.24x on the Ryzen and 1.57x on the 285K at four threads, 2.2x on the
  Pi, against a 1.37x win at 24.** (The Pi is the outlier for a reason that is not the encoder: see
  `ggml-0011` above.) Whisper's
  exported driver used to hand the decoder the raw encoder output every step, so cross-attention K/V
  was re-projected over all 1500 encoder frames per token — `MUL_MAT 768x1500` ran 684 times for an
  11-second clip where the encoder itself needs 84, **57.7% of ASR runtime**, where onnxruntime
  computes those tensors once. Exporting the projected K/V as its own topology is worth **2.4-2.6x on
  every machine here** (on the Pi, 90.0 s to 37.4 s for the same clip).
* **Most of the remaining ASR gap is the ENCODER, and it is two kernels rather than a systemic
  deficit.** Split per shape at one thread with both engines' own profilers, interleaved round by round
  (2026-08-28, Ryzen; `scripts/whisper_encoder_split.py` re-derives it), loom's encoder is **1.20x**:

  | encoder piece | loom | onnxruntime | ratio | share of the gap |
  |---|---|---|---|---|
  | **`QK^T`** (12 calls) | 1957 ms | 1027 ms | **1.96x** | **51%** |
  | `Softmax` (12) | 780 | 491 | 1.60x | 16% |
  | `fc2` (12) | 2302 | 2016 | 1.14x | 16% |
  | `fc1` (12) | 2240 | 2034 | 1.09x | 11% |
  | Q/K/V/O (48) | 2138 | 1970 | 1.11x | 9% |
  | `A@V` (12) | 1101 | 1046 | **1.07x** | 3% |
  | GELU, LayerNorm + bias | 290 | 476 | **0.61x** | *loom ahead, −11%* |

  **The dense GEMMs are 1.09-1.14x each and 36% of the gap** — both true: per shape there is nothing in
  them to win, but they are 6.7 s of an 11.1 s encoder. `QK^T` alone is 6.0% of a whole transcription.
  *An earlier version of this table read the decoder as 1.50x FASTER and "nothing left there".* That was
  wrong: the 471 ms layout bucket under it was assigned to the encoder by eye, and 93% of it is the
  decode loop — which is why `$LOOM_PROFILE_NODES=1` now exists to attribute a bucket to a graph instead
  of to a guess. Three items on that list are now closed:
  * **GELU is done.** ggml's exact-erf GELU was a scalar `erff()` libm call per element with no SIMD
    path on any architecture; `cmake/patches/ggml-0010` replaces it with a rational approximation,
    **9.9x on the op in model** and 14.3-21.8x standalone. End to end on whisper that is **2.1% on the
    285K at four threads, 4.4% on the Ryzen at two, and 0.9% on the Pi** — a large kernel win that is a
    small share of a whole transcription, which is exactly what a 3.6%-of-runtime op can be.
  * **A GEMM cliff is done.** tinyBLAS rejected every matmul whose *contraction* was not a whole number
    of vectors (`k % KN`), handing the whole thing to ggml's generic kernel — and a contraction in
    attention is a sequence length or a head dimension, i.e. a number nothing rounds. `ggml-0011` splits
    it: **2.15x at whisper's `A@V` shape** — confirmed in the model, where `A@V` went 2.23x to **1.07x**
    against onnxruntime — and **1.19-1.24x end to end on Conformer-CTC**, whose head dimension
    (176/4 = 44) misses the vector width on every utterance. It also **regressed aarch64 by 1.3-1.75x
    for four days** by inlining its tail into the register tile; dispatching that out of line fixed it
    without moving x86 (see the note under the table).
  * **`SOFT_MAX` is measured out — but not for the reason first given.** Its five row passes really are
    three too many, and rewriting them is worth 1.08x on the machine that motivated it. The original
    reason, *"at 108 MB the op is DRAM-bandwidth bound"*, is false: against a `memcpy` of the same bytes
    ggml's row body is **3.6x on the Ryzen and 7.8x on the 285K**. It is closed on size instead —
    the exp is 7-26% of the op depending on the core, specialising `ggml_v_expf` to this domain is
    1.00-1.16x, and deleting the polynomial outright is only 1.8x
    ([Retro-012](retros/retro-012-optimizations-that-were-measured-out.md)).
  * **The decoder's loop-invariant V transpose is done** (2026-08-29), and it is an EXPORT change, so
    the table above does not contain it. Whisper's decoder re-materialised the transpose of its
    cross-attention V every token in every layer — 12 nodes x 4.6 MB, 47% of the decode loop — of a
    tensor the cross-KV fix had already made constant for the utterance. The `cross_kv` phase now emits
    it transposed and the traced chain is deleted from the decoder topology: **1.106x end to end**, and
    the decoder's logits against HF are bit-identical. **The published GGUFs do not have it yet** — the
    ASR column above is the rc6 artifact, and the re-export lands in rc7.
  * What is left is `QK^T` at `k = 64` — **51% of the remaining encoder gap**, and closing it entirely
    would take the encoder from 1.20x to 1.10x.
* **Read the 285K's four-thread ASR cell as unstable.** onnxruntime's own time at that thread count
  ranges **1.08 s to 1.64 s across launches, a 1.5x spread**, because that part is 8 P-cores plus 16
  E-cores and a process is placed once and then stays there; loom's spread over the same launches is
  1.09x. The cell has read 0.71x, 0.59x and now 0.64x across three samplings in which the engines did
  not change. **The two onnxruntime builds are indistinguishable here** — nine paired launches of the
  conda-forge and PyPI 1.28.0 wheels straddle 1.0 at four threads and are 1.01x at 24 — so the 1.86x
  build difference that VITS shows does **not** carry to whisper.
* **loom used to stop scaling at 8 threads and go backwards; that is fixed, and the 24-thread row is
  the fix.** VITS on the 285K was 0.080 s at 8 threads and **0.191 s at 24 — the same as at one
  thread**. The cause was not in this repository: `ggml` defaults to OpenMP, so `ggml_barrier` was
  `#pragma omp barrier` — one after every non-empty graph node, 2520 of them in a synthesis — and
  libgomp's default wait policy **slept every thread on a futex at each one** (334,609 voluntary
  context switches per 5 syntheses, against 160 now). Building `ggml` against its own threadpool, whose
  spin is bounded, made the curve monotonic again and is worth **4.8x at 24 threads**
  ([Retro-017](retros/retro-017-libgomp-slept-at-every-graph-node.md)).
* **The engine defaults to this machine's PHYSICAL CORE COUNT**, as of P4.30b. It used to default to
  `ggml`'s `GGML_DEFAULT_N_THREADS` — 4, whatever the machine has — which ran a 24-core workstation on
  four cores and said nothing about it; that is why two machines have two rows here. `$LOOM_N_THREADS`
  overrides it, and on Linux the count respects the process's CPU affinity, so `taskset` and a cgroup
  cpuset are honoured. Against the old default of 4 the 285K gains **1.98x on TTS, 2.41x on ASR and
  1.18x on the LM**, and the Ryzen gains **1.19x on TTS and 1.03x on the LM by going *down* to 2** —
  its two extra SMT siblings buy nothing on any task and cost on two, so "use every CPU" would have
  been right on one of these machines and a 1.19x TTS regression on the other. A Pi 4 does not move.
  **This does not move the published ratios much, because onnxruntime wants the same thing** — over the
  same 4 → 2 change on the Ryzen it gains 1.17x, 1.02x and 1.07x on the three tasks, so loom's ASR cell
  gets *worse* (0.80x → 0.67x at the time it was measured) purely because onnxruntime is the one that
  gains there. **The physical-core rule is a property of these CPUs, not of loom.**
  What the change costs is the case it was previously left at 4 for: every figure on this page is one
  inference at a time on an idle machine, and a host running several loom instances concurrently now
  has each of them claim every core. That host sets `$LOOM_N_THREADS`.

## What these numbers are not

**The onnxruntime build is named per row, because at the identical version it is worth up to 1.86x.**
The conda-forge build synthesises VITS in 0.065 s where the PyPI wheel takes 0.120 s — same machine,
same script — and against conda-forge the x86 TTS wins above become losses. The two 285K rows are
measured against conda-forge and the other three against the PyPI wheel, which is the channel `loom-py`
ships through. **That difference does not carry to whisper**: nine paired launches on the 285K put the
two builds within the thread-placement noise at four threads and at 1.01x at 24, so the ASR column is
comparable across rows even though its baselines are not the same package. The TTS column is not, and
the Pi and Ryzen TTS cells are the PyPI-wheel ones.

**Each pair is checked for equal work, not merely equal wall time.** TTS pins VITS's three scales so
both engines emit the same 73216 samples, and both harnesses print that count; ASR compares the
transcripts, which are identical; the LM runs the same prompt to the same token budget greedily on both
sides, and both emit the same tokens. **Model load is outside every timer on both sides.**

**Every cell is an average over separate process LAUNCHES, not over runs inside one.** The TTS and LM
cells are means — per engine, 31 TTS launches on x86 and 15 on the Pi; 21 LM launches on the 285K, 15
on the Ryzen and 7 on the Pi, where one generation is 47 s. The ASR cells are medians (9 per engine on
each 285K row, 7 on each Ryzen row, 5 cooled pairs on the Pi) and were taken 2026-08-29. **The Pi's TTS
and LM cells no longer report the fastest run.** That special case existed because thermal drift on
that board only ever makes a run slower; enforcing the cooldown before *every arm* rather than around
the session removes the drift instead of estimating around it, and the board then holds a 1.02-1.05x
spread per engine — tighter than either x86 machine.

**A median is used where the samples are unimodal and a mean where they are not, and the difference is
worth up to 1.4x.** At 24 threads on the 285K both engines' VITS time splits into two per-launch modes
about 1.4x apart at roughly even odds; a median over that lands on whichever mode drew the majority, so
the same eight minutes of benchmarking can report 1.19x or 1.66x with nothing changed. `paired_arms.py`
now prints both estimators and warns when an arm's launches split into two weighted clusters.

**Thread placement is chosen once per PROCESS on a hybrid part, it is worth up to 1.47x, and it is not
a lottery — it is WHICH CLUSTER.** The Core Ultra 9 285K is 8 P-cores (`/sys/devices/cpu_core/cpus` =
0-7) plus 16 E-cores (`cpu_atom` = 8-23), no SMT. A four-thread process lands on one cluster and then
stays there for its whole life, so every run inside it inherits that choice and a within-process median
cannot average it out. Pinned to a cluster, every number is reproducible to ~1%:

| VITS, 4 threads, 5 launches each | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| P-cluster (`taskset -c 0-3`) | 0.0624-0.0628 s | 0.0646-0.0661 s | **1.034-1.053** |
| E-cluster (`taskset -c 8-11`) | 0.0916-0.0927 s | 0.1033-0.1075 s | **1.114-1.165** |

So the two clusters give *different ratios*, and a cell that does not say which one the process got is
under-specified. Pinning is how that table was measured and it is the right way to *explain* the
spread, but it is not how the cells above are taken: it constrains onnxruntime more than loom (pinned
to P-cores its whisper time is 1.76 s, worse than its lucky unpinned launches), so a pinned table would
not be like-for-like.

**One second of idle between the two arms on x86, and a cooldown to 60 C on the Pi, because otherwise
the second arm pays for the first.** Run
immediately after loom, onnxruntime landed in its slow VITS mode in 29 of 31 launches; run alone on the
same box in the same session, 10 of 20. A `sleep 1` between arms restored it to 21 of 31 and moved the
285K's 24-thread TTS cell from 1.41x to 1.20x. Alternating the arm order does not cancel this, because
both orders penalise whoever is second, and it is not symmetric between the engines — ggml's threadpool
spins where onnxruntime re-decides placement at session creation. The settle is worth ~1% on the
homogeneous Ryzen and up to 1.18x on the hybrid 285K, which is what says it is a placement effect.
[Retro-025](retros/retro-025-the-arm-that-ran-second-paid-for-the-first.md) has the rest.

**Both sides use the same estimator**, which is not a detail: every harness here warms up and reports
a best-or-median over repeated runs in one process. `bench_lm_loom.cpp` used to time a single cold
generation instead, and comparing that against a warmed-up onnxruntime moved the LM column by 5-7% —
enough to flip its sign on all three machines. **`bench_asr_loom.cpp` had the same fault until
2026-08-29** and it was missed because the retro that fixed the LM harness asserted this one was
already warm rather than checking. Cold/warm on whisper is 1.25-1.7x at 24 threads on the 285K, 1.02x
at four, and *below* 1.0 on the Ryzen, where the box heats faster than the first run pays for itself —
a thread-count effect, which is why it landed on the one cell the table called an ASR win. The harness
now discards a warm-up run and prints it, and **`nrun` must be >= 3**: `times[size / 2]` on two samples
is the larger of the two, not a median.

**And the same two faults were still in the other half of the comparison until 2026-09-02.**
`bench_onnx_tasks.py`'s `vits` task had no warm-up while `bench_vits_loom.cpp` discards its first
synthesis, and `bench_vits_loom.cpp` carried the same `ts[size / 2]` index. Both are fixed. The audit
that fixed the ASR harness opened **loom's three harnesses** and asserted the onnxruntime side, which is
Retro-018's own lesson landing on its own blind spot; the check is one line and it has to name both
globs — `grep -n 'warm' scripts/bench_*_loom.cpp scripts/bench_onnx_tasks.py`. `bench_lm_loom.cpp` now
prints its first five tokens, so the LM's equal-work check is shown rather than asserted, the way VITS
prints a sample count and ASR prints a transcript.

**Reproducing it:** loom's side is `scripts/bench_{vits,lm,asr}_loom.cpp`, onnxruntime's is
`scripts/bench_onnx_tasks.py`. The latter drives `onnxruntime` directly rather than through `optimum`,
which added roughly 2x of its own overhead to whisper and mis-derives Qwen3's `head_dim`.

**The Pi throttles, and it will lie if allowed to.** It goes 55 -> 84 C during a single whisper run and
caps the ARM clock at 1580 MHz; two back-to-back measurements once came out 87.1 s and 115.8 s, 33%
apart. Its row is taken with a cooldown before every measurement and the two engines interleaved, so
both meet the same clock.
