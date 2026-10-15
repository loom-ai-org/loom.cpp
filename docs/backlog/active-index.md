---
type: index
category: backlog
last_updated: 2026-10-15
---

# Active Ledger — Open Work Across All Three Repos

Only **open** work lives here, one line each. Closed work is not tracked: its decisions are
[ADRs](../adrs/), its lessons are [retros](../retros/), its architecture is in the
[epics](../epics/), and its execution detail is in git history or [`archive/`](../archive/).

**Item IDs are the existing `P`-numbers.** Code comments reference them (`P4.3e`, `P4.15b`), so they
are not renumbered. New items continue the scheme.

---

## Now — the next things to pick up

| item | why now |
|---|---|
| **LiteRT thread (2026-10-11): int8 is where XNNPACK's case is. The queue, in the user's order** | Results: Engine — performance, the XNNPACK item. ~~**(1) Q4 arm**~~ **DONE 2026-10-11** (the XNNPACK item's "4-bit" block): Q4_0 is slower than Q8_0 in ggml except on ARMv6; ggml's own Q4_0 repack matches XNNPACK `qb4w` on the M1; `qb4w` wins 1.5-1.9x only where no repack applies (A72); XNNPACK's 4-bit is slower than its int8 everywhere; `qb4w` cannot take Q4_0 blocks as-is (positive BF16 scales only). ~~**(2) `CPU_REPACK` in the engine**~~ **DONE 2026-10-11, opt-in** ([ADR-075](../adrs/adr-075-quantized-weights-reach-ggmls-repack-by-placement-and-routing.md), `LOOM_CPU_REPACK=1|kleidiai`): on the M1, whisper-small 1.24-1.54x and Qwen3-0.6B decode 1.34-1.56x at one thread, 1.02-1.07x at 8 (memory-bound); KleidiAI best for Q8_0 prefill (1.53x) but 0.83x on Q8_0 decode at one thread. Whether it becomes the default is the open item under Engine — performance. ~~**(3) ADR: the XNNPACK microkernel bridge**~~ **WRITTEN 2026-10-11, proposed** ([ADR-076](../adrs/adr-076-quantized-gemms-reach-xnnpacks-microkernels-through-a-ggml-buffer-type.md)): a ggml patch adding a `CPU_XNNPACK` buffer type built like KleidiAI's, with only the selected per-ISA `qd8` kernels vendored and run on ggml's own threads; weights as row-uniform Q8_0 (`--quantize Q8_0_PC`), which every backend still reads as plain Q8_0 and the bridge packs losslessly; first cut `mul_mat` on Q8_0 only; main case the A72 (2.4-2.7x at the GEMM), then x86 Q8_0. ~~**(4) ADR: static int8 (qs8)**~~ **WRITTEN 2026-10-11, proposed** ([ADR-077](../adrs/adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md)): the RTF ~1.5 estimate was taken against the wrong baseline and is withdrawn. Against the F32 conv sweep Citrinet actually runs (0.22 GOP/s), XNNPACK `qs8` (0.54-0.62 on ARMv6) gives RTF ~1.8; even an infinitely fast GEMM leaves 0.82. So real time needs the non-GEMM 24% faster too, and an ARMv6 int8 GEMM ~2x XNNPACK's. Its four gates in order: split the 24% by op on the board; an ARMv6 int8 GEMM >= 1.2 GOP/s; WER; jfk on the board. ~~**(5) Port XNNPACK's softmax**~~ **PORTED 2026-10-13 as `ggml-0029`** (the `SOFT_MAX` item): one thread, dense/causal, x86 1.45x/2.05x, Pi 4 2.10x/2.14x, M1 1.54x/1.71x, Pi Zero 1.46x/1.15x; accuracy kept by a grouped f64 sum. CosyVoice3's free-running gate failed on it; a control showed it measures chaos, mel and ASR held on two more fixtures, and its bound is now 6e-2. ~~**(6) Fix ggml's F32 GEMM throughput at large K**~~ **PATCHED 2026-10-13 as `ggml-0030`, aarch64 only, bit-identical** (the large-K item): Pi 4 K=3072 1.35x at 1 thread, 1.41x at 4. Original: (whisper `fc2`, K=3072, runs at half the rate of K=768 on the Pi 4 and the Pi Zero, not on the M1). Find where the drop starts by sweeping K, then fix the K blocking. It also hits Qwen3-0.6B's `down_proj` (the large-K item). The gemma-3 "KV leak" was not one ([Retro-084](../retros/retro-084-a-sampler-was-logged-as-a-leaking-cache.md)). The LiteRT steps 1-2 (ADR-072, ADR-073), the T5 encoder bias (ADR-074) and the pad_crop retirement are on pushed branches `perf/delayed-shrink-zero-copy-inputs` / `chore/retire-pad-crop-relative-embeddings` (loom.cpp) and `feat/t5-bias-in-graph` (loom-exporter), no PRs yet. |
| **ADR-079: ARMv6 F32 kernels on the VFP11's short vectors, then dynamic int8 (user, 2026-10-14)** | ADR-077's cross-model analysis found F32 at equal effort beats today's ggml by more than either int8 option adds: projected 1.3-3.0x on 11 of 13 Pi Zero models from the F32 family alone. **P4.32a done 2026-10-14:** the `LEN = 4` harness (`scripts/armv6_shortvec.py`), and gate 1 passed on the board in RunFast and IEEE mode, with sabotage red ([ADR-079 gate 1](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#gate-1-measured-2026-10-14-passed)). **P4.32b done 2026-10-14** (`ggml-0032`): vocoder convs 2.67x at the op; amy/de051/VITS **1.73-1.74x** end to end (de051 RTF 1.91 -> 1.10), outputs identical to the ear (ASR oracle) ([ADR-079 P4.32b](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#p432b-measured-2026-10-14-the-direct-conv-tile-in-ggml-ggml-0032)). **P4.32b's wide/dilated shapes + P4.32c (GEMV, `ggml-0033`) done 2026-10-15**, all outputs bit-identical: VITS 1.37x more (RTF 7.91), amy 1.23x (3.16), Moonshine F32 1.32x (4.27), Silero VAD 1.47x ([ADR-079](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#p432bs-wide-and-dilated-shapes-and-p432c-the-gemv-measured-2026-10-15)). **P4.32d done 2026-10-15** (`ggml-0034`: GEMM n >= 24, 1x1 and strided convs on the same kernel): Citrinet Q8_0 1.72x (RTF 1.54), Moonshine F32 1.43x (RTF 2.98), conformer 1.18x, heart 1.16x; transcripts and CTC argmax unchanged ([ADR-079 P4.32d](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#p432d-measured-2026-10-15-the-gemm-1x1-and-strided-convolutions-ggml-0034)). **P4.32e done 2026-10-15** (`ggml-0035`, depthwise on a ring of VFP registers): 10.9x at the op; Citrinet F32 1.28x, Q8_0 1.27x end to end, CTC ids unchanged ([ADR-079 P4.32e](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#p432e-measured-2026-10-15-the-depthwise-kernel-ggml-0035)). **Next: P4.32f** (epilogue fusion). Static int8 is held. Items under Engine — performance, the ADR-079 item; [ADR-079](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md). |
| **P5 breadth: family 14 (music): the next leaf** | MusicGen Small is **published** (2026-10-10, `musicgen-small-loom`, [Epic-03 §2](../epics/epic-03-model-coverage.md#family-14-musicgen-two-families-composed)). The next leaves are unscoped. MusicGen medium/large are the same export at 1.5B/3.3B parameters; melody (chroma-conditioned) and stereo are different models. Music was last in [Epic-03 §3](../epics/epic-03-model-coverage.md)'s order; everything before it is complete or parked (fastpitch and bananamind-tts deferred 2026-10-01; kugelaudio, tada, dots-tts and irodori-tts dropped 2026-09-26) |
| **Nemotron 3.5 ASR: the English-only sibling** | Nemotron 3.5 ASR is **published** (2026-10-10, `nemotron-3.5-asr-streaming-0.6b-loom`, Hub `4a28fa5`, on rc16; [Epic-03 §2](../epics/epic-03-model-coverage.md#family-1s-fourth-transducer-nemotron-35-asr)). Next leaf: `nvidia/nemotron-speech-streaming-en-0.6b`, the same encoder with no language prompt (a `nemotron_asr_streaming` recognizer and an encoder wrapper without the prompt), NVIDIA Open Model License |

**State anchor, 2026-10-10 (evening) — `1.0.0-rc17` is released and the Hub equals the staging tree.**
Four packages are on PyPI at `1.0.0rc17` (17/2/2/1 files, as rc16). The signed tag `1.0.0-rc17` sits on
loom-py `adf130b`, the merge of loom-py #64, which pins loom.cpp `b4ef517`. `wheels.yml` run
38077776811 was green, including `publish-pypi`. The tag also triggered Build images, whose
`cuda, x86_64` job timed out logging in to ghcr.io; a re-run of that job passed, and the wheels never
depended on it (they pull the existing image).

rc17 carries the engine's `blank_after_bos` framing and packed GET_ROWS on a view (#73), WordPiece
decoding in `transcribe` (#74), and loom-py's phoneme-style fold (ADR-071). Published on it, each from a
fresh export at loom-exporter `92a2d27`, byte-identical to its verified stage, card-gated on the RELEASED
wheel, and loaded back from the Hub: **citrinet-256-ls** (new, F32 + Q8_0, Hub `950e723`),
**moonshine-tiny** (new, F32 + Q8_0, `57046eb`), **sanotts** (new, 28 voices, GPL-3.0, `084b9c0`) and
**vits-piper-en-gb-miro** (republished with the training framing and a card note, `53a2d7d`). 58 repos.
A dry run found the other 54 identical to the staging tree; conformer-ctc-small and citrinet-1024
re-export byte-identical after the NeMo tokenizer change.

**Previous anchor, 2026-10-10 — `1.0.0-rc16`.**
Four packages are on PyPI at `1.0.0rc16`: `loom-py-rt` (17 files), `-cuda` (2), `-vulkan` (2) and
`-metal` (1). The `linux_armv6l` wheel is a GitHub release asset. The signed tag `1.0.0-rc16` sits on
loom-py `8c96eff`, the merge of loom-py #58, which pins loom.cpp `7699ec1`. `wheels.yml` run
38031293831 was green, including `publish-pypi`.

rc16 carries three changes:
* ggml v0.26.0 (#65), which also brings llama.cpp's Vulkan misaligned-offset fix.
* The Vulkan conv_2d F32-precision fix, which stops F5-TTS clipping on NVIDIA
  ([Retro-077](../retros/retro-077-a-gpus-f32-convolution-was-half-precision-too.md)).
* Skipping a device that cannot hold the weights, for MOSS-TTS on an iGPU (#66,
  [Retro-076](../retros/retro-076-a-full-device-said-device-lost.md)).

`musicgen-small-loom` is **published on rc16** (Hub commit `3d9b41b`, 54 repos). It was a fresh export
at loom-exporter `afc7ac4`, byte-identical to the verified one. It was card-gated 2/2 on the RELEASED
wheel through the new music row (loom-py #59). The sha256 and README match, and the card runs from
the Hub. No other GGUF needed a re-export: exporter `main` changed only by adding MusicGen. Every
published topology from a phase that declares a KV cache has cached ATTENTION, so the new exporter
guard (Retro-078) rejects none of them. A dry run found the other 53 repos identical to the staging
tree.

**Previous anchor, 2026-10-08 — `1.0.0-rc15`.** The signed tag `1.0.0-rc15` sits on loom-py `9041c9e`,
which pins loom.cpp `90e3492`. rc15 carries:
* The frame-embedding door ([ADR-062](../adrs/adr-062-a-classifier-says-how-many-answers-it-gives.md)'s
  second amendment).
* P4.31's direct depthwise conv.
* Four GPU correctness fixes ([ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md)).

`wakehubert-tiny-loom` was republished on it.

What decides a publish is the standing rule, **WHEELS FIRST,
ALWAYS**: a GGUF or a card that needs something the released wheels have not got waits for the
release that carries it, because a file published before its wheel is a file nobody can run, and an
option an older engine does not know can be dropped silently rather than refused (rc10's `run_ode`
would have integrated F5-TTS **unguided**). And every publish is a fresh export first
([[feedback-release-gate-needs-a-fresh-export]]) → [Packaging & release](#packaging--release)

---

## Models

* [ ] **Moonshine base**: the same leaf as tiny (`moonshine_v1_export.py`), on disk at `~/Dev/models/moonshine-base`, not exported
  or measured. *Context: [Epic-03](../epics/epic-03-model-coverage.md#family-2s-third-leaf-moonshine-v1-tiny)*
* [ ] **MOSS-TTS + MOSS-Audio-Tokenizer: three open pieces.** Both are published
  (`moss-tts-local-transformer-v1.5-loom`, `moss-audio-tokenizer-v2-loom`), with voice cloning through voice files of
  reference codes; the card gate runs the cloning card when `LOOM_CARD_VOICES` supplies a voice. Left: (1) a Q8_0 build
  beside F32, at 16.8 GB; (2) the template's other lines (`Tokens` for duration, `Instruction`); (3) a clip-in door
  inside loom, which needs the codec encoder as phases. *Context:
  [ADR-049](../adrs/adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md)–[ADR-053](../adrs/adr-053-a-codec-lms-voice-is-its-references-codes-stamped-with-the-codec.md),
  [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **Voxtral-4B-TTS: a quantized build.** Published (`voxtral-4b-tts-2603-loom`, exported on the workstation,
  with 20 voices, all CC BY-NC 4.0). At F32 a frame reads the whole 13.7 GB LM, and the CPU runs 5.5x slower than real
  time on 24 cores, so a quantized build beside F32 is worth offering. CUDA is not measured. There is no cloning: the
  open checkpoint has no codec encoder. *Context: [Epic-03](../epics/epic-03-model-coverage.md),
  [ADR-054](../adrs/adr-054-a-tiktoken-vocabulary-is-merged-by-rank-in-the-shared-bpe.md),
  [Retro-062](../retros/retro-062-an-f32-wrapper-check-could-not-tell-a-spelling-from-a-defect.md)*
* [ ] **Chatterbox's other leaves**: voice cloning (voice encoder + S3 tokenizer + CAMPPlus), and the
  multilingual and Turbo checkpoints. The English model is PUBLISHED (2026-09-30,
  `loom-ai-org/chatterbox-loom`: exported on the workstation from loom-exporter `c61351e`, card gate 2
  passed there, x-linked-etag = local sha256 `eee92e72...`; no Perth watermark, per Epic-03 §2).
  *Context: [Epic-03](../epics/epic-03-model-coverage.md),
  [ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)*
* [ ] **Pocket-TTS: cloning a voice from a recording.** Published (`pocket-tts-loom`, with 26 voice files under
  Kyutai's use restrictions; two voices are non-commercial). Cloning needs the Mimi encoder, which is in the gated
  voice-cloning weights and zeroed in the other release, as one more phase feeding the text prefill's ordinary input.
  It is the same door F5-TTS's in-filling already has from the other side. *Context:
  [Epic-03 §2](../epics/epic-03-model-coverage.md),
  [ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md),
  [ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)*
* [ ] **VoxCPM2: two open pieces.** Published (`voxcpm2-loom`, Apache-2.0, 9.3 GB at F32).
  * [ ] **Voice cloning from a recording** needs the AudioVAE's encoder (16 kHz, x640) as one more
    phase feeding `feat_encode`'s prompt patches; the driver's prefill already takes patches and masks.
  * [ ] **The reference's f32-vs-f64 spread is not a floor for this model**: FSQ rounds `tanh(x) * 9`
    every step, so a boundary crossing flips a level, and the free-running gate arm could fail on another
    ISA with nothing wrong. Revisit if it ever does. *Context:
    [ADR-046](../adrs/adr-046-a-guidance-rule-the-integrator-cannot-express-stays-in-the-step-graph.md)*
* [ ] **Fun-CosyVoice3: two open pieces.** Published (`fun-cosyvoice3-0.5b-loom`, 2026-09-29, Apache-2.0). The card
  gate passes, and its cloning card runs when `LOOM_CARD_VOICES` supplies a voice.
  * [ ] **Cloning needs Python.** Voice files work (`loom_exporter.cosyvoice3_voices`, gate arm on a
    JFK clip), but making one runs the ONNX S3 tokenizer v3 and CAMPPlus in Python. Cloning inside
    loom means exporting both as phases, with the log-mel and fbank front ends in the graph, and the
    release ships neither as torch (check `s3tokenizer` on PyPI for a v3 port).
    *Context: [ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)*
  * [ ] **Normalisation is the rules path only.** The reference prefers `wetext` (WeTextProcessing
    FSTs, Apache-2.0) when installed, which reads dates, money and units the rules path spells digit by
    digit. Reproducing it needs the FSTs as data and an FST runtime in C++; check the FSTs' licence
    first. Deferred by the user 2026-09-25.
    *Context: [ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)*
* [ ] **Qwen3-ASR-0.6B variants beyond the exported leaf** — `qwen3-asr-0.6b-hf` is shipped; the 1.7B
  and the native-layout repo are not. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **P5 breadth: family 14 (music)**: MusicGen Small is published (2026-10-10); the next leaf is unscoped (see *Now*).
  Every earlier family in the coverage-per-effort order is complete or parked. Chatterbox showed that a leaf
  composes from the existing templates, so what the next one costs is its own front end and its own
  loop shape, not a template.
  *Context: [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) and
  [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md) for what family 12
  cost across three checkpoints, which is the estimate the rest of this list should be read against;
  [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md) and
  [Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md) for where families 4 and 5
  found the cost instead.*
* [ ] **Moonshine's encoder attends through a full `T x T` mask of which 20 diagonals are live.** 81.9 s
  of audio is 4096 rows: 16.7M scores per head per layer, 27 s for tiny on the 2-core box. A banded
  attention (keys gathered at the 19 live offsets) is the same softmax over a fraction of the work;
  measure it on the Pi before scoping, as P4.25-style ideas have measured out before.
* [ ] **`log_softmax` lowers as `log(softmax(x))`** (coremltools' torch frontend), which is -inf below
  about -103 in f32. Canary's head stops before it; the CTC heads still carry it. Harmless for an argmax,
  wrong as a tensor. A stable form needs a row max the engine has no primitive for.
* [ ] **Kyutai STT and LFM2.5-Audio are published at F32** (rc14, 2026-10-02): at F32 Kyutai runs
  about 5x slower than real time on the 2-core box and LFM2.5's two files are 5.2 and 5.5 GB. Measure
  Q8_0 against each gate (`test_e2e_kyutai_stt_lua_driver`, `test_e2e_lfm25_audio_{asr,tts}_lua_driver`,
  whose reference scripts are in `scripts/`) and republish if the transcripts and codes hold.
* [ ] **LFM2.5-Audio's interleaved speech-to-speech door** (`generate_interleaved`, text and audio in a
  fixed 6:12 pattern, multi-turn chat) -- not requested; the TTS door's phases are all it needs, plus
  an audio-in prompt segment (the ASR door's encoder) and the interleave schedule in the driver.
* [ ] **Requested for the zoo, unscoped** (added 2026-09-25 at the user's request, no order among
  them yet). None has been checked against its checkpoint; the template guesses are where scoping
  starts, not what it will find.
  * [ ] **Cohere ASR** — architecture and licence not yet looked at.
  * [ ] **Voxtral Mini realtime** (Mistral, streaming ASR) — check its size first against the machine
    floor that blocks Voxtral-Mini-3B (below).
  *Context: [Epic-03 §3](../epics/epic-03-model-coverage.md#3-roadmap)*
* [ ] **`flan-t5-small`'s vocabulary is 32,100 pieces against a 32,128-wide logit row.** T5 pads its
  embedding to a multiple of 128, so an argmax could in principle name an id with no piece — untrained
  rows, never observed in practice, and the model is shipped and verified without a bound on it. Worth
  one if a leaf in this family ever emits such an id. *The other limit recorded alongside this one is
  the driver marshalling `n_head * n_src²` doubles for the encoder bias — tens of thousands for a
  sentence, 1.5M at the 512-token ceiling;
  [ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md) records the in-graph
  alternative if it ever becomes measurable.*
* [ ] **Voxtral-Mini-3B waits on a machine, not on the exporter.** P5.0 is closed — all three changes
  are in ([ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md)): a phase releases its
  torch and MIL halves together, packs its weights as it converts, and under `--isolate-phases`
  converts in a process of its own. Its LM phase alone still needs ~14.4 GB of F32 weights beside
  ~14.4 GB of MIL constants, so the floor is ~29 GB against this box's 28 and no further exporter
  change moves it. The measurements a bigger machine would pick it up from are in
  [Epic-03 §2](../epics/epic-03-model-coverage.md). **The bigger machine exists now:** the workstation
  exported Voxtral-4B-TTS's 3.4B Ministral LM (the same backbone class) at a 32.2 GB peak in 49 s, with
  the checkpoint, the venv and the recipe under `~/loom-voxtral/` there.
  * [ ] *The one exporter change still worth making here, and only if load TIME starts to matter:*
    a per-family hook that loads one phase's submodule instead of the whole checkpoint. An N-phase
    model costs N+1 checkpoint loads under isolation today. It would not change the peak.

## Exporter / MIL compiler

* [ ] **A shape read as DATA aborts in the engine.** (Nemotron ASR met it too, 2026-10-10: `(n - 1) - arange(2n - 1)` put a 4-wide `n - 1` into a SUB; spelled `-arange(1 - n, n)` instead.) `length / waveform.shape[1]` (or `lengths *
  x.shape[-1]`, speechbrain's relative-length masks) traces to `SHAPE` -> `GET_ROWS` -> arithmetic. The
  engine's `SHAPE` is a four-element `ne` vector and `GET_ROWS` on a 1-D tensor returns it whole, so the
  arithmetic sees `[4]` where it expects `[1]` and `ggml_can_repeat` aborts (ECAPA, family 13: `DIV
  [1]/[4]`, then a RESHAPE of four elements into one). A shape consumed at BUILD time (FILL, RESHAPE
  targets) works; one consumed as a value does not. `arange(x.shape[i])` is unaffected (the walk folds
  it). Fix either end: the exporter folds a static `GET_ROWS` of a `SHAPE` into the dimension's own
  expression, or the engine's `GET_ROWS` indexes a 1-D tensor's elements. Until then ECAPA ships
  whole-clip only. *Context: [Epic-03 §2](../epics/epic-03-model-coverage.md#family-13-small-audio-classifiers-and-embedders)*
* [ ] **An export's op names depend on what its process converted earlier.**
  `coremltools`' `Builder.name_count` is a *class* attribute -- one counter for the whole process --
  and `_get_free_name` both reads and increments it, so an op a MIL pass builds without an explicit
  name is `transpose_0` in a fresh interpreter and `transpose_21` in one that has already converted
  twenty. That name reaches the emitted topology as the node's output, so **two exports of the same
  checkpoint from differently-warmed processes are not byte-identical.** Harmless in practice today --
  `loom-export` converts one model and exits, which is why the sweep has never seen it, and
  `whisper-small` came out byte-identical exported both with and without `--isolate-phases`
  (969,918,400 bytes, `cmp` clean) -- and it is *not* new with phase isolation, which is only what
  surfaced it, inside a pytest session where the in-process arm had a warmed counter and its workers
  did not. **The fix is one line** (clear `Builder.name_count` before each phase's `ct.convert`), and
  the reason it is filed rather than done is that it would rewrite those names in every artifact whose
  later phases contain an auto-named op, so it needs a sweep to say which models move and a decision
  about re-publishing them. Until then `tests/ci/test_phase_isolation.py` resets the counter per export
  and `test_the_mil_op_name_counter_is_process_global` pins the mechanism. *Context:
  [ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md).*

* [ ] **A dynamic slice with a NEGATIVE begin exports as twice the rows at a negative offset.**
  `_infer_dynamic_dim_expr` renders `x[:, -n:, :]` arithmetically instead of normalising the begin
  against the source length, so `begin = -n` stays `-n` and the size comes out `end - begin = 2n`. At
  the traced length the two readings coincide, which is why it converts, writes and loads and only
  fails when run. Found in F5-TTS, where x_transformers' `apply_rotary_pos_emb` opens with exactly
  that slice; fixed **in the family** by substituting a version without the (provably identity) trim,
  so nothing in the zoo emits one today and the walk is unchanged. Worth doing properly the first time
  a negative begin is not removable. *Context:
  [Retro-051](../retros/retro-051-a-negative-begin-doubled-the-slice.md), and
  [Retro-047](../retros/retro-047-an-inferred-dimension-outlives-the-reshape.md) for the same
  right-at-the-traced-length shape.*
* [ ] **F5-TTS's text front end ships the table and not the segmenter.** `convert_char_to_pinyin` is
  `rjieba` + `pypinyin`; `loom::F5Vocab` is the character half, measured at 2000/2000 against the real
  function for ordinary prose and diverging by one inserted space for multi-character punctuation runs
  and hyphen-joined digit groups. Closing it means a CJK segmenter in the engine, which is a decision
  about scope rather than a fix. *Context:
  [Epic-03 §2](../epics/epic-03-model-coverage.md), [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).*
* [ ] **Supertonic carries 2.3 MB of the same zero padding VITS just lost.** `ttl_text_512.emb_*`,
  99.1% zeros — 0.9% of that model, against VITS's 23.2%. **Not the same fix**: P4.28 made VITS's pad
  dynamic, and Supertonic's text axis is statically sized on purpose for two independent reasons its
  own export docstring gives, one of which is that `GraphBuilder` resolves only one dynamic-length
  symbol per topology. It belongs to that limitation, not to the pad.
  *Context: [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md),
  [Epic-05 §5](../epics/epic-05-edge-performance.md) P4.28.*
* [ ] **Modular-export generality is unproven** — the blueprint's whole point was structural rather than
  by-name discovery, and that claim still rests on LFM2 alone. Needs a second, structurally different HF
  model in the regression suite. *Context:
  [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md)*
* [ ] **Modular phase 2 — automatic prefix/suffix boundary discovery.** Deliberately not attempted;
  today's `ModularExportSpec` needs a ~3-line declarative boundary per model. Worth doing only if that
  starts feeling like real friction across 2–3 more models.
* [ ] **`LoomModelFor*` runtime entry points** — the *inference* half of the `optimum` analogy has no
  counterpart. Arguably should wait for a second consumer besides the tests. *Context:
  [ADR-005](../adrs/adr-005-export-config-and-task-registry.md)*
* [ ] **Driver templates as first-class artifacts** — `render_driver` is still marker-based string
  replacement into a hand-written `.lua`. The mechanism is `driver_ir`, which exists and is under-used.
  A family whose driver needs more than one substitution point will force it.
* [ ] **`.inputs` / `.generate_dummy_inputs()` / `.patch_model_for_export()` as named config members** —
  none exist under those names; axes are declared per phase and dummy inputs built inline.
* [ ] **Known gap: `matmul` composition only handles `transpose_x=False`.** Any other combination raises
  `NotImplementedError` by design rather than miscomputing. Not yet hit by any converted model; needs a
  real derivation and test case the first time one is.
* [ ] **StableHLO prototype on one solved model** — deliberately not started; filed as a validation
  exercise rather than a fix.
* [ ] **`_infer_dynamic_dim_expr`'s root-axis fallback should be observable.** Four separate producers
  stopped the walk during family 5 and each silently substituted the root axis, producing a graph that
  converts, loads, builds and declares a wrong length. Two of the four (`loom_scale`,
  `loom_broadcast_to`) are ops the exporter's OWN passes emit, so they had been reachable by every model
  since those passes were written, and one (`reduce_sum`) had a case that only handled half its inputs.
  The fallback cannot be removed — it is right for the common case, which is why
  [Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md) left it
  — but it is currently indistinguishable from a derived answer. `ValueFacts` already carries
  `scalar_expr_is_guess` / `range_scalar_is_guess` / `slice_axis_is_guess` for exactly this distinction
  on the scalar paths; `dim_expr` has no equivalent. Adding one, and having the REPEAT-emitting rules
  (`loom_broadcast_to`, `tile`, `fill`, `fill_like`) say so under an env var or in the export log, would
  have turned four five-minute export-and-read cycles into one. **Not a behaviour change** — a
  provenance flag and a diagnostic, nothing that alters an emitted graph.
  *Context: [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md),
  and [[feedback-instrument-the-walk-do-not-re-export]] for the 30-line repro that names the broken
  link without a re-export.*
* [ ] **P6 cleanup** — delete the `tools/convert_*` directories (~14,000 lines across 10), then the docs
  pass.

## Engine — correctness

* [ ] **`RANGE_1D` with a negative step returns -1 elements, silently.** `op_range_1d` forces `end > start` so `ggml_arange` (which asserts it and has no negative steps) cannot abort, so `arange(T - 1, -T, -1)` becomes `arange(start, start + 1, -1)`; the first error is a downstream RESHAPE. Lower a negative step as `start + step * arange(0, n)` with `n = ceil((end - start) / step)`, and throw on an empty range rather than inventing one element. Nemotron ASR's export avoids it so its file runs on released wheels. *Context: [Retro-079](../retros/retro-079-the-references-default-path-was-the-broken-one.md)*
* [ ] **Who still depends on the layout-"healing" heuristics?** The guards make size-guessing
  *unreachable for already-correct graphs*; they do not remove it, and `op_repeat`'s two branches are
  **unguarded**. Every one is a silent-wrong-answer generator with no error path.
  *Work, in order:* (1) instrument each branch with a counter naming op and shapes, run all models, and
  record which fire — that converts "some model probably needs these" into a list; (2) fix each real
  firing at the **exporter**; (3) delete the branch. A branch that fires for nothing can go immediately.
  *Context: [Retro-001](../retros/retro-001-layout-healing-heuristics.md)*
* [ ] **Known bug: `make_lfm2_gguf.py`'s per-layer zero-RoPE placeholder produces NaN.** Each of the 16
  decoder layers is traced independently with `position_embeddings` as `torch.zeros(1,1,64)` — the
  script's own comment calls them "placeholders that we will swap", and they never were. A NaN reaches
  SILU and trips `assert(!isnan(x))`, a hard `SIGABRT`. `test_e2e_lfm2_lua_driver` skips (77)
  unconditionally. **Not fixed**, and worth fixing only if that bespoke script's coverage still matters —
  the MIL path is otherwise a strict improvement over it.
* [ ] **A `REPEAT` whose source is a PERMUTE aborts the process.**
  `ggml_compute_forward_repeat_f32` asserts `nb00 == sizeof(float)`, and `_op_loom_broadcast_to` emits a
  bare `REPEAT` from whatever var the broadcast pass handed it. A permuted view has a non-unit innermost
  stride, so a graph that broadcasts a transposed tensor converts, exports, loads, builds — and then
  takes down the host process with a raw `GGML_ASSERT`, not a `loom::Error`. Family 5 hit it twice (an
  edge-pad repeat over a transposed mel, and the position table) and worked around it **in the family**,
  by keeping both tensors in a layout where the repeat source is contiguous. That is the right fix for
  one model and not a general one: nothing stops the next family from writing the natural torch
  spelling. Two candidate fixes, and they are not equivalent — the exporter could emit `CONT` before a
  `REPEAT` whose source it knows is permuted (it emits the `PERMUTE`, so it can know), or `op_repeat`
  could `ggml_cont` a non-contiguous input itself (always correct, costs a copy only when needed, but
  moves a graph decision into the engine and against the lean-runtime principle).
  *Context: `ggml_view_*`'s own `nb[0]` assumption and
  [Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md), which is the same
  shape of bug one op over.*

## Engine — performance

* [ ] **`SOFT_MAX`: PORTED 2026-10-13 as `ggml-0029`, measured on all four machines, CosyVoice3 gate re-bounded; open: commit + upstream.** (UPSTREAM.md PR 26.)
  XNNPACK's exponential kernels, a four-accumulator vector max, the copy/scale/mask fused into one pass
  (no pass at all for a plain softmax), and a sum that is NOT XNNPACK's. Its f32 accumulators raised
  |sum - 1| from 1.3e-7 to 7.1e-7, so each unrolled step adds in f32 (four per lane) and widens into f64
  vector accumulators. **What the Retro-012 attempt missed:** `ggml_v_expf` leaves its fast path whenever a
  vector holds |n| > 126, and a causal mask's -inf does exactly that. Masked rows cost 2x dense ones.
  Through the real op, one AVX2 thread, 12 x 1500 x 1500: dense 49.3 → 34.1 ms (1.45x; XNNPACK 29.0),
  F16 causal 101.9 → 49.7 ms (2.05x). Accuracy against f64 (plain, scaled, F32/F16 causal, ALiBi, 77-wide
  tails): AVX2 same-or-better on every metric; aarch64 under qemu within 5%; ARMv6 under qemu sum unchanged, max abs 3.6e-8 → 5.0e-8 (libm's 0.5-ulp expf → 1.96 ulp, the trade XNNPACK's Pi Zero speed rests on). Tools are in the session
  scratch (`sm/softmax_probe.cpp` row probe with every variant, `sm/smtest.cpp` op test with an XNNPACK arm).
  **On the boards** (1 thread, dense / causal): Pi 4 2.10x / 2.14x (XNNPACK's own margin 2.05x), M1 1.54x / 1.71x
  (XNNPACK 1.29x), Pi Zero 1.46x / 1.15x (XNNPACK 1.97x); Pi 4 at 4 threads ~1.0x / 1.08x (memory-bound). NEON uses
  `rr1_p5`, not XNNPACK's `lut64_p2`: faster on both chips (A72 2%, M1 20%). **ARMv6 trap, fixed:** XNNPACK's
  scalar kernel computes on -inf and zeroes at the end; the ARM1176's VFP traps the inf - inf NaN (and denormals)
  to software, and causal rows went 18x SLOWER (22.4 s). The kernel replaces sub-cutoff inputs by 0 first.
  **CosyVoice3 gate, re-bounded 2026-10-14 (user's call):** its free-running waveform rmse failed with this patch
  (cloned voice 0.0333 > 2e-2; unpatched 0.0069), and a control with ggml's own exp and only the sum reordered gave
  0.0343. On two more fixtures (new sentences, default voice and JFK clone) every build and both reference runs
  transcribe to the same words (whisper-small), and the log-mel distance stays inside the reference's own f32-vs-f64
  spread; the waveform rmse ranges 0.007-0.058 across builds, while the reference's own f32 run ranges 0.009-0.249. Bound
  raised to 6e-2, with the measurements in the test.
  *The original item:*
  **`SOFT_MAX`: XNNPACK shows 1.75-1.81x headroom over ggml on whisper's 1500-wide rows** (Ryzen 3
  3250U, one thread, `scripts/bench_xnnpack.cpp`, 2026-10-11). This tree's best attempt reached 1.16x
  on the same box (Retro-012: the 3-pass row and the `ggml_v_expf` specialisation). So the headroom is
  real and was not found. Read XNNPACK's `f32-raddstoreexpminusmax` and `f32-rmax` kernels (BSD-3)
  for what they do differently, prototype in `scripts/bench12.cpp`'s frame, and measure on x86 AND
  the Pi before writing a ggml patch. Softmax was 16% of whisper's encoder gap to onnxruntime.
  **Gate it on accuracy too (user, 2026-10-11):** compare the port's output with ggml's current `soft_max`
  (and an f64 reference) on whisper's rows and on masked/causal rows, not only the timing. It is step 5 of
  the LiteRT queue, after the Q4 arm, `CPU_REPACK` and the two ADRs.
  **ARM agrees (2026-10-11): XNNPACK wins softmax on every machine measured.** At one thread: M1 1.28-1.30x,
  Pi 4 2.05-2.06x, Pi Zero 1.96-1.97x (XNNPACK's scalar ARMv6 path). On the M1 at 8 threads it is
  2.27-2.30x. On the Pi 4 at 4 threads it is only 1.03-1.07x, because 108 MB in and out is memory-bound there.
  This is the one result that holds on every core.
* [ ] **Attention `QK^T` at k=64: XNNPACK 1.60-1.75x over ggml** on the same run. Retro-012's own
  outer-product tile already reached 1.85x on this box, and was closed on the 285K (1.38x against a
  1.5x gate) and on NEON (1.23x). XNNPACK's NEON number, from the Pi/M1 run, is what says whether a
  better ARM tile exists. `QK^T` was 51% of whisper's encoder gap to onnxruntime.
  **ARM (2026-10-11): headroom only on the A72, and smaller than x86.** Pi 4: 1.32x at one thread, 1.18x
  at four. M1: 1.02x and 1.05x. Pi Zero: 0.94-1.08x. A better NEON tile is worth ~1.3x on the Pi 4 and
  nothing on the M1.
* [ ] **ggml's F32 GEMM at large K: PATCHED for aarch64 as `ggml-0030` (2026-10-13, UPSTREAM.md PR 27); the Pi Zero
  is still open.** The sweep (Pi 4, 768 x 1500, tinyBLAS) is flat at 8.5 GFLOP/s to K=1536 on one core, then
  6.4 / 5.05 / 4.3 at K=2048 / 3072 / 4096. On four cores the knee is at K=1024. Two working sets: the 4-row A strip
  outgrows the 32 KB L1D at K=2048, and four 72-column B blocks outgrow the shared 1 MB L2 at K=1024. The fix chunks
  K inside each job (chunk = 3/8 of L1D, 768 there) and caps the B block at half of L2, both read from sysfs/sysctl.
  The vector accumulators carry across chunks, so outputs are BIT-IDENTICAL (hashes on AVX2/A72/M1; the CosyVoice3
  gate reproduces the unpatched numbers to every digit). Pi 4 K=3072: 5.05 → 6.84 (1 thread), 12.7 → 17.9 (4).
  About two thirds of the drop is recovered. M1: derived chunk 3072, untouched (flat anyway). x86: neutral to -6%,
  so compiled out. **Not covered:** the Pi Zero, whose ARMv6 GEMM is a different kernel class (ggml-0018).
  **The earlier "not on the M1" stands only with tinyBLAS:** without `GGML_LLAMAFILE` the M1 also steps down 20% at
  K=2048. *The original item:*
  **ggml's F32 GEMM halves its throughput at K=3072 on both ARM boards.** whisper `fc2`
  (3072x1500x768) runs at 4.3-4.4 GFLOP/s on one A72 core against 8.3 for `fc1` (768x1500x3072), the
  same FLOPs with K=768. At four threads it is 10.1 vs 24. The Pi Zero shows the same 2x, and the M1
  does not (91.8 vs 94.1). XNNPACK holds its rate on both, so the cause is ggml's K blocking, not the
  memory system alone. It matters beyond whisper: Qwen3-0.6B's MLP `down_proj` has K=3072. Source:
  `scripts/bench_xnnpack.cpp`, 2026-10-11. Check x86 next, then profile the tinyBLAS K loop on the Pi 4.
  Only K=768 and K=3072 were measured, so sweep K first to find where the drop begins. Step 6 of the
  LiteRT queue (user, 2026-10-11), right after the softmax port.
* [ ] **Turn `LOOM_CPU_REPACK` on by default?** Quantized weights now reach ggml's `CPU_REPACK` /
  `CPU_KLEIDIAI` when asked ([ADR-075](../adrs/adr-075-quantized-weights-reach-ggmls-repack-by-placement-and-routing.md)): M1
  whisper-small 1.24-1.54x, Qwen3-0.6B decode 1.34-1.56x at one thread; x86 Q4_0 1.18-1.28x. Three
  things decide it: (a) **the tied-embedding copy** (+165 MB on Qwen3 Q8_0, +26%): keep `GET_ROWS`-read
  weights out of placement, and measure what an unrepacked `lm_head` costs in decode; (b) **which
  engine per op on Arm**: KleidiAI wins Q8_0 prefill (1.53x vs 1.34x) and loses Q8_0 single-thread
  decode (0.83x), so it is a per-shape choice, not a build flag; (c) a gate run of the quantized
  model cards with it on, since repacked GEMMs move greedy tokens (Qwen3 Q4_0 parts at token 6).
  The A72 and ARMv6 have no repack for either type, so they gain nothing from it.
* [ ] **ggml Q8_0 `mul_mat` is slower than F32 on the A72 and ARMv6 for narrow GEMMs.** VITS's
  N=32/64/128 vocoder GEMMs: Pi 4 1.56x slower, Pi Zero 2.1x slower than ggml's own F32. Neither core has
  dotprod, so this is ggml's fallback `vec_dot_q8_0_q8_0`. XNNPACK's `qd8` NEON / ARMSIMD32 kernels on the same cores
  are 2.7x / 1.8x faster than it.
* [ ] **ggml's F16 conv path is slow on every CPU.** `ggml_conv_2d_direct` with an F16 kernel runs
  1.5x (Ryzen) to 2.9x (M1) behind XNNPACK's F16 conv, and 3.8x behind ggml's own F32 conv on the A72.
  Check which conv model exports ship F16 for CPU.
* [ ] **ggml scales badly on narrow-N GEMMs (N=32, M=73472).** One thread to 8 on the M1: ggml 4.5x,
  XNNPACK 7.3x. One to four on the Pi 4: ggml 2.7x, XNNPACK 4.1x. These are VITS's im2col-form vocoder
  GEMMs. Look at how `mul_mat` chunks the work when `ne01` is 32. Source: same run.
* [ ] **The Pi Zero, for the two models chosen for it.** Measured 2026-10-10 on the Pi Zero W (P4.31 engine): Moonshine
  tiny at ~7x slower than real time (jfk 11 s in 75 s, either precision), sanoTTS heart-nano at 1.1x, the 0.51M piperlite
  voices at 2.3x. heart-nano is the closest to real time: its iSTFT is a DFT-basis `conv_transpose1d` (~0.1 GMAC per second of
  audio, half its cost), the same lever as the FFT item above. No profile yet. **Citrinet-256-ls** (rc16 wheel): jfk in 37 s
  at Q8_0, 64 s at F32 (RTF 3.4 / 5.8). Profiled at Q8_0: 76% is its 1x1 `CONV_2D`s, ~6 GFLOP at ~0.22 GFLOP/s, which is
  the board's GEMM ceiling, so only fewer FLOPs or a faster ARMv6 GEMM moves it ([Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)
  closed the known GEMM levers). **ADR-077 gate 1 measured 2026-10-14, verdict HELD for a cross-model analysis**
  ([ADR-077](../adrs/adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#the-analysis-that-decides-gate-1)): per model
  on the board, per op class, the best F32 against the best int8 kernel, then seconds left under F32 fixes / int8 GEMM /
  full static int8. The other 24% is F32 inefficiency, not F32 arithmetic: the depthwise convs run 7.9x faster as a blocked VFP kernel in
  RunFast (3.96 -> 0.50 s, in isolation, `scripts/bench43.cpp`), where int8 manages only 0.74x of that. Their padding-mask
  `MUL`, bias `ADD` and `RELU` fuse in any precision. RunFast gives the VFP's `vmla` 1.65x (`scripts/bench44.c`), which
  also moves the "GEMM ceiling" above. **Gate 2 measured 2026-10-14 in `bench_xnnpack.cpp`** ([ADR-077](../adrs/adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#gate-2-measured-2026-10-14-117-120-gops-at-the-gemm-104-111-with-the-quantize)):
  an asm SMLAD 2x4 tile runs jfk's 1x1 convs in 5.73 s channels-first / 5.35 s channels-last (1.04 / 1.11 GOP/s; GEMM
  alone 1.17-1.20 against the gate's 1.2), vs XNNPACK `qs8` 9.3 s, ggml Q8_0 23.6 s, ggml F32 RunFast 12.0 s: **passed** (user, 2026-10-14).
  Channels-last is worth 0.38 s at the op, so it folds into gate 1's analysis rather than being built alone. **And the F32 card was paying for VFP bounces**
  (BN-folded dead channels, weights ~1e-33, tiny accumulators cancelling): RunFast alone takes jfk on the F32 card from ~61 s to 20-22 s, past the
  Q8_0 card's 29-35 s, same ids. Baseline now the F32 card with ggml-0031: jfk 20.1 s. **Gate 1's cross-model analysis
  MEASURED 2026-10-14** ([ADR-077](../adrs/adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#gate-1-the-cross-model-analysis-measured-2026-10-14)):
  13 models profiled on the board and joined to their shapes (`LOOM_PROFILE_NODES=src`); each class's best F32 against its
  best int8 at equal effort (asm both). int8 wins the GEMM classes by 1.46x/1.66x (Citrinet's 1x1, dynamic/static), 1.06x/1.31x
  (vocoder direct convs), 2.14x (GEMV), but **the VFP11's short vectors (`LEN = 4`), never tried here, put F32 at 0.72 / 1.11 GOP/s
  on those shapes against ggml's 0.50 / 0.25-0.65**, and a plain F32 GEMV streams 3.5x faster than ggml's. Projected: the F32
  items alone 1.3-3.0x on 11 of 13 models (Citrinet RTF 0.94, de051 0.74), option 1 +1.02-1.33x (distilbert-ner 3.25x vs Q4_0),
  option 2 a further 1.05-1.39x (vocoders most). **Decided (user, 2026-10-14): F32 kernels + option 1 go ahead as
  [ADR-079](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md) (items P4.32a-g, P4.33 below);
  option 2 (static int8) HELD, ADR-077 status `held`.** Still open here: why ggml's Q8_0 conv path is 1.65x slower than
  its F32 one (23.6 vs 14.4 s). *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **ARMv6 F32 kernels on the VFP11's short vectors (P4.32a-g), then dynamic int8 (P4.33)**
  ([ADR-079](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md), accepted; its gates in
  order). ~~P4.32a~~ the harness `scripts/armv6_shortvec.py` (enter/leave/lint) + gate 1 **PASSED on the board 2026-10-14** (ADR-079's gate-1 section; qemu-user fails it);
  ~~P4.32b~~ direct conv tile **SHIPPED as `ggml-0032` 2026-10-14** (`gen_armv6_conv_f32.py`, `bench50.cpp`; amy/de051/VITS 1.73-1.74x end to end;
  then input-channel chunks + TB 128 for the wide/dilated shapes, 2026-10-15: VITS 1.37x, amy 1.23x more, bit-identical);
  ~~P4.32c~~ F32 GEMV **SHIPPED as `ggml-0033` 2026-10-15**, 3.0x at the op bit-identical, then a float sum on the user's call (3.5x; not bit-identical, gated on the board) (Moonshine F32 1.32x, Silero 1.47x end to end, then ~3% more); was: **P4.32c** F32 GEMV (`scripts/bench48.c`; Moonshine 20.2 s, Silero 2.0); ~~P4.32d~~ GEMM **SHIPPED as `ggml-0034` 2026-10-15** (the k = 1 kernel: `mul_mat` n >= 24,
  1x1 and strided convs; Citrinet Q8_0 1.72x, Moonshine F32 1.43x); ~~P4.32e~~ depthwise **SHIPPED as `ggml-0035` 2026-10-15**
  (`gen_armv6_dw_f32.py`, `bench52.cpp`; 10.9x at the op, Citrinet F32 1.28x / Q8_0 1.27x end to end, not bit-identical: even + odd tap sums);
  **P4.32f** bias/mask/activation epilogue fusion (1.3-3.7 s in seven models); **P4.32g** FFT for the DFT-basis convs (an op,
  needs its own design; 1.2-2.1 s in five models). **P4.33** dynamic int8 on `Q8_0_PC` (ADR-076) for the GEMV (2.14x) and
  GEMMs (1.46x), NOT the vocoder convs (1.06x), only after the F32 family ships. Projection to check end to end against:
  `scripts/armv6_projection.py` over `armv6_opmix.py --json` tables. *Context: [Epic-05](../epics/epic-05-edge-performance.md)*
* [ ] **RunFast (VFP `FZ | DN`) on ARMv6 compute threads: all four gates passed 2026-10-14, to merge**
  ([ADR-078](../adrs/adr-078-runfast-on-armv6-compute-threads.md), accepted). `ggml-0031` on branch
  `perf/adr077-armv6-int8` (not pushed), on by default, `GGML_CPU_RUNFAST=0` off. Pi Zero, 11 models: faster on every
  one (Citrinet-256-ls F32 55.9 -> 20.1 s, 2.79x; the rest 1.02-1.33x), outputs identical (two differ below 1e-6 of
  the peak). qemu does not reproduce the board in this mode, or in IEEE mode where the VFP bounces ([Retro-086](../retros/retro-086-qemu-matches-the-board-only-where-the-vfp-never-bounces.md)).
  Left: merge, and a release so the ARMv6 wheel carries it. *Context: [Epic-05](../epics/epic-05-edge-performance.md)*
* [ ] **WakeHuBERT's remaining gap to ONNX int8** (11.2 ms against 5.67, dev box, F32, one thread, after
  P4.31). Its convs already run ggml's direct `CONV_2D` at 55-68% of the F32 peak, so a better F32 GEMM
  buys ~2 ms at most. Levers by size: an FFT/STFT primitive in place of the 80 MFLOP DFT matmul (~2 ms,
  and every DFT-basis front end in the zoo pays the same), int8 dot products for the 1x1 convs (~4.7 ms
  of work, unmeasured), and host overhead (~2 ms or more). *Context: [Epic-05 P4.31](../epics/epic-05-edge-performance.md#p431--depthwise-convolutions-a-direct-kernel-a-simd-interior-and-the-causal-block-fused--done-2026-10-03),
  [Retro-073](../retros/retro-073-the-profiler-cannot-see-a-fusion.md)*
* [ ] **LiteRT-class CPU speed: what it would actually take, and which three of its four pieces are
  runtime work.** The standing hope is that loom matches LiteRT on some models. LiteRT gets there with
  four things, and mapping them onto this tree ranks very unevenly — the important structural finding
  is that **three of the four are kernel/runtime work, not export-time metadata**, so this is a
  different bet from the GGUF memory-layout and prescribed-tiling thread and should not be expected to
  fall out of it.
  * [ ] **XNNPACK as the kernel set — measured on x86 (2026-10-11); the ARM run is what decides.**
    `scripts/bench_xnnpack.cpp` times XNNPACK's operator API (commit `4a5fe376`, static, 1.6 MB +
    3.8 MB of x86 microkernels) against this tree's patched ggml on the shapes loom runs: VITS's 11
    GEMMs and 20 dilated convs, whisper-small's encoder linears, `QK^T` and softmax. Both arms get the
    same inputs and are checked to agree (max relative difference at most 3e-6). Ryzen 3 3250U, one
    thread, ggml/XNNPACK over two runs:
    - **VITS:** GEMMs 1.07-1.12x. Convs **0.89-0.93x** (ggml FASTER), because XNNPACK loses the
      weight-heavy short-activation shapes 2x. With the channels-last transposes an integration would
      add (a naive upper bound), **0.60-0.64x**.
    - **whisper encoder:** **1.16-1.17x**, almost all from two ops: `QK^T` (k=64) **1.60-1.75x** and
      softmax **1.75-1.81x**. The linears are at parity (0.98-1.08x).
    - **At 2 threads:** the same picture, but the per-shape spread is 0.67-2.2x on this two-core box.

    So, on x86, a wholesale swap would buy nothing for VITS (and cost the layout), plus two ops for
    whisper. Those two are targeted items, not an integration. `QK^T`: Retro-012's own outer-product
    tile already reached 1.85x on this box, and was closed on the 285K (1.38x) and on NEON (1.23x).
    Softmax: our best was 1.16x here, and XNNPACK shows 1.8x is reachable. Read its `f32-raddstoreexpminusmax`
    kernels (BSD-3, so reading and porting is clean) before re-opening `SOFT_MAX`.
    **ARM, 2026-10-11** (same harness, XNNPACK `4a5fe376` built natively on each board; max relative
    difference at most 2.9e-6 everywhere; two runs each, which agree):

    | ggml/XNNPACK | VITS GEMMs | VITS convs | convs + layout tax | whisper encoder |
    |---|---|---|---|---|
    | M1 Pro, 1 thread | **0.81** | **0.82** | 0.72 | **0.84** |
    | M1 Pro, 8 threads | 1.20 | 1.04 (clean run) | 0.54-0.61 | 0.94-0.95 |
    | Pi 4 (A72), 1 thread | **1.14-1.16** | **1.46-1.47** | 0.94 | **1.45-1.46** |
    | Pi 4, 4 threads | 1.30-1.37 | 1.34-1.39 | 0.56-0.57 | 0.55 |
    | Pi Zero (ARM1176), 1 thread | **0.72-0.73** | **0.59-0.61** | 0.57-0.59 | 1.72-1.82 |

    - **The M1 favours ggml:** 92 GFLOP/s on one P-core against XNNPACK's 74, on every GEMM and conv.
      At 8 threads XNNPACK wins only the N=32 GEMMs (1.3-1.6x). That is ggml scaling 4.5x on those
      shapes while XNNPACK scales 7.3x (see the narrow-N item).
    - **The Pi 4 is the one place XNNPACK clears the ~15% bar.** ggml runs 6-8 GFLOP/s per A72 core,
      XNNPACK 10-12. The conv margin, though, does not survive the channels-last transposes even at one
      thread (0.94, a naive single-threaded upper bound on the tax). XNNPACK's whisper linears also fail to scale on the
      board: `qkv` takes 177 ms at four threads against 187 at one, `fc1` 924 against 785. Both runs
      reproduce this, and it is unexplained. That alone makes whisper 0.55x at four threads.
    - **The Pi Zero favours ggml's ARMv6 kernels** (`ggml-0018`/`0019`) over XNNPACK's scalar fallback. XNNPACK
      does build and run on ARMv6+VFPv2 (cross-compiled with clang 14 against the Raspbian sysroot).
      Its whisper win is all `fc1`/`fc2`, i.e. the large-K item below. It ran at `heads=2`, because 12
      heads do not fit in 427 MB.

    **So there is no wholesale case on any of the four machines.** The A72 gap is real, but it sits in
    ggml's A72 GEMM/conv throughput, and it disappears on the M1. A better A72 tile in ggml, as
    `ggml-0002` did for tinyBLAS, is the cheaper test of it. An `extra_buffer_type` integration (the
    KleidiAI route) would win on one core class and lose on the others.

    **Low precision, 2026-10-11** (`./bench_xnnpack <t> <reps> <heads> f16|q8`; GEMMs and convs only,
    because QK^T and softmax stay F32 in loom). Each engine runs what a loom export of that dtype runs.
    For ggml Q8_0 that means `mul_mat` doing int8 dot products, while a Q8_0 **conv dequantizes its
    kernel and runs the F32 sweep** (`ggml-0013`). XNNPACK runs `qd8` (per-channel int8 weights,
    activations quantized on the fly, F32 out, convert op inside the timing: the drop-in counterpart)
    and `qs8` (static int8 in and out: what a LiteRT full-int8 model runs, a ceiling that needs a
    calibrated model). Every int8 arm lands at 2e-3 to 1.7e-2 against F32. `qd8` is within ~2x of ggml's
    Q8_0 either way, lower on GEMMs and higher on some convs. `qs8` is about 2x worse. Two runs each, which agree:

    | ggml Q8_0 / XNNPACK `qd8` (`qs8`) | VITS GEMMs | whisper linears | VITS convs | convs + layout tax |
    |---|---|---|---|---|
    | Ryzen 3 3250U, 1 thread | 1.48-1.52 (1.94) | 1.57-1.69 (1.77-1.79) | 1.00-1.01 (1.07) | 0.66-0.67 |
    | M1 Pro, 1 thread | **2.12** (3.57) | **2.81** (2.91) | **3.85** (4.16) | **2.38-2.40** |
    | M1 Pro, 8 threads | 4.54-4.65 (9.3) | 2.46-2.82 (3.07) | 3.10-3.13 (4.5-4.6) | 0.84-0.86 |
    | Pi 4 (A72), 1 thread | **2.69** (4.13-4.16) | **2.41-2.42** (3.29) | **2.19** (2.74-2.75) | 1.17 |
    | Pi 4, 4 threads | 2.15-2.19 (5.42-5.45) | 2.49-2.50 (3.18-3.22) | 2.40-2.44 (3.02-3.10) | 0.70-0.71 |
    | Pi Zero, 1 thread | 1.77-1.80 (3.91-3.98) | 1.66-1.91 (2.94-3.19) | 1.09-1.10 (2.45-2.49) | 1.05-1.06 |

    - **XNNPACK's dynamic int8 beats ggml Q8_0 on every machine, at every thread count**: 1.5-2.8x on
      GEMMs at one thread, up to 4.7x at 8. The int8 conv gap is **compute**: loom's Q8_0 conv runs in F32 by design (so x86, where
      XNNPACK's AVX2 int8 conv is no faster than an F32 one, reads parity). Where `qd8` is fast (M1 dotprod,
      A72 NEON), it wins the convs 2.2-3.9x and keeps 1.17-2.4x even after the layout tax at one thread.
    - **ggml's own Q8_0 repack is already in the tree and loom does not use it.** The CPU backend's
      `CPU_REPACK` buffer type re-lays Q8_0 as `q8_0_4x4` (NEON+dotprod) or `4x8` (i8mm) at upload, and
      loom's engine never places weights there. On the M1 it gives 1.60x (VITS) / 1.78x (whisper) at
      one thread and 2.15-2.17x / 1.75-1.76x at 8, with error identical to plain Q8_0. `qd8` stays 1.3-1.6x
      ahead of it at one thread. The repack does not apply on x86 (no AVX2 Q8_0 repack), the A72 (no
      dotprod) or ARMv6, and those are exactly the places where ggml Q8_0 is slowest.
    - **ggml Q8_0 is SLOWER than ggml F32 on both ARM boards** on VITS's narrow GEMMs: Pi 4 1249 ms
      against 802 (one thread), Pi Zero 40.4 s against 19.0 s. On the Pi 4 the whisper linears are
      level, 37.1 s against 37.2. So on those boards a Q8_0 export of a GEMM-heavy model saves memory and
      costs time.
    - **F16:** ggml F16 against XNNPACK's native F16, at one thread: Ryzen VITS GEMMs 1.09-1.11, whisper 0.91, convs 1.54-1.56. M1 VITS GEMMs 1.45,
      whisper 0.81, convs **2.88** (3.11 at 8 threads). On the A72 XNNPACK declines F16 (no FP16
      arithmetic), and ggml's F16 convs are **3.8x slower than its own F32** there (10.2 s against
      2.7 s). An F16 conv model on CPU pays for it on every machine measured.

    **This reverses the F32 conclusion for quantized models.** The case is int8 GEMM and int8 conv
    compute, not F32 kernels, and it holds on every core class. Steps, cheapest first:
    (1) put Q8_0 weights in `CPU_REPACK` (an engine loader change, no ggml patch; helps dotprod/i8mm
    ARM only); (2) scope an `extra_buffer_type` that hands Q8_0 `mul_mat` (and a folded Q8_0 conv) to
    XNNPACK's `qd8` operators, as ggml's KleidiAI backend does for its own kernels. `GGML_CPU_KLEIDIAI`
    is OFF in loom's builds and was not measured. On accuracy, this harness only shows `qd8` comparable to Q8_0 on synthetic inputs.
    Per-channel scales are coarser than block-32, so a model-level gate (ASR oracle, WER) decides it.

    **4-bit, 2026-10-11** (`./bench_xnnpack <t> <reps> <heads> q4`; queue step 1). ggml runs what a
    Q4_0 export runs: `mul_mat` as Q4_0 x Q8_0, and a conv that dequantizes its folded Q4_0 kernel and
    runs F32. `rpk` is the same Q4_0 weight in `CPU_REPACK`, which re-lays Q4_0 for **AVX2 (8x8) as well
    as** dotprod (4x4) / i8mm (4x8). XNNPACK runs `qb4w` (qd8 x block-32 4-bit, Q4_0's arithmetic),
    `qc4w` (qd8 x per-channel 4-bit), `qs8_qc4w` (static int8 x per-channel 4-bit), and `qd8` (int8) as
    the in-process reference. **XNNPACK has no 4-bit convolution operator**, so on the conv rows only its
    int8 conv runs. Timing table, ms, the mean of the runs (all runs agree within 1% except where a
    range is given):

    | VITS GEMMs (one synthesis) | ggml Q4_0 | Q4_0 `rpk` | `qb4w` | `qc4w` | `qs8_qc4w` | `qd8` (int8) | *ggml Q8_0* | *Q8_0 `rpk`* |
    |---|---|---|---|---|---|---|---|---|
    | Ryzen 3 3250U, 1t (4 runs) | 323.6 | 252.3 | 279.4 | 227.3 | 146.4 | 193.0 | *274.5* | n/a |
    | M1 Pro, 1t | 67.3 | **34.5** | 36.2 | 31.0 | 19.2 | 27.6 | *58.3* | *36.4* |
    | M1 Pro, 8t | 25.8 | 11.6 | 6.6 | 5.6 | 3.0 | 5.1 | *24.9* | *11.6* |
    | Pi 4 (A72), 1t | 1445.7 | n/a | 771.9 | 658.9 | 2041.4 | 466.4 | *1250.4* | n/a |
    | Pi 4, 4t | 464.1 | n/a | 270.1 | 242.2 | 511.6 | 194.4 | *419.6* | n/a |
    | Pi Zero (ARM1176), 1t | 27331 | n/a | 25682 | 24221 | 21090 | 22339 | *40493* | n/a |

    | whisper linears (one encoder) | ggml Q4_0 | Q4_0 `rpk` | `qb4w` | `qc4w` | `qs8_qc4w` | `qd8` (int8) | *ggml Q8_0* | *Q8_0 `rpk`* |
    |---|---|---|---|---|---|---|---|---|
    | Ryzen, 1t | 12157-13786 | 7430-13054 | 9572 | 7332 | 5377 | 6124 | *9300* | n/a |
    | M1 Pro, 1t | 2063.8 | **889.7** | 972.9 | 747.0 | 697.2 | 608.5 | *1709.5* | *958.0* |
    | M1 Pro, 8t | 297.5 | 132.8 | 135.6 | 105.1 | 95.8 | 87.0 | *249.6* | *142.2* |
    | Pi 4, 1t | 40217.6 | n/a | 26109.2 | 21108.1 | 78107.0 | 15706.9 | *37131.1* | n/a |
    | Pi 4, 4t | 10228.7 | n/a | 6668.5 | 5579.4 | 19469.0 | 3749.4 | *9547.7* | n/a |
    | Pi Zero, 1t | 925432 | n/a | 868636 | 801938 | 855584 | 772490 | *1469275-1523519* | n/a |

    | VITS convs (weighted) | ggml Q4_0 | XNNPACK `qd8` (int8) | *ggml Q8_0* |
    |---|---|---|---|
    | Ryzen, 1t | 610.0 | 583.0 | *466.3 (another session)* |
    | M1 Pro, 1t / 8t | 211.8 / 38.8 | 54.7 / 12.6 | *210.8 / 39.6* |
    | Pi 4, 1t / 4t | 2721.2 / 1049.3 | 1215.0 / 430.6 | *2666.1 / 1053.7* |
    | Pi Zero, 1t | 94289 | 86617 | *95636* |

    *Italic columns are the q8 runs of the same morning, against the same ggml build on each machine. The M1, the
    Pi 4 and the Pi Zero reproduce across sessions (XNNPACK `qd8` reads the same in both). The Ryzen does not (the same
    `qd8` conv went 464 ms to 527-626 ms), so on x86 only the in-run columns compare.*

    - **A Q4_0 export is SLOWER than a Q8_0 one in ggml everywhere except the Pi Zero.** VITS GEMMs:
      M1 67.3 against 58.3, Pi 4 1446 against 1250, Ryzen 324 against 275. On the Pi 4 both are slower
      than ggml's own F32 (802 ms). The nibble unpack costs more than the halved weight bytes save on
      these compute-bound shapes. **On ARMv6 it is the other way round**: Q4_0 is 1.48x (VITS) / 1.62x
      (whisper) faster than Q8_0, through tinyBLAS's ARMv6 Q4_0 x Q8_0 kernel. Even so, ggml's F32
      (19.0 s) still beats it, and every other arm in the table. A Q4_0 conv costs the same as a Q8_0
      one, because both dequantize and run F32.
    - **ggml's own Q4_0 repack closes the XNNPACK gap where it applies.** On the M1 (4x4, dotprod)
      `rpk` is 1.95x (VITS) / 2.32x (whisper) over plain Q4_0, and **matches or beats** `qb4w` at one
      thread (34.5 vs 36.2 ms, 890 vs 973 ms). At 8 threads it ties on whisper (133 vs 136) and loses
      VITS's narrow-N GEMMs 1.76x (11.6 vs 6.6, the narrow-N scaling item). On x86 (8x8, AVX2) it is
      1.28x on VITS. This makes queue step 2 (`CPU_REPACK` in the engine) cover Q4_0 on x86 too, not
      only Q8_0 on dotprod ARM.
    - **Where the repack does not apply, `qb4w` beats ggml Q4_0**: Pi 4 1.87x (VITS) / 1.54x (whisper)
      at one thread, 1.72x / 1.53x at four. **On the Pi Zero the field is level**: ggml Q4_0 is within
      1.06x (VITS) / 1.07x (whisper) of `qb4w`, and within 1.22x / 1.20x of `qd8`.
    - **XNNPACK's dynamic 4-bit arms are slower than its own dynamic int8** (`qd8`) on every machine:
      `qb4w` is 1.1-1.8x behind it and `qc4w` 1.0-1.5x (the narrowest gaps are on the Pi Zero). So 4-bit buys memory and nothing else. If speed
      is the goal, the bridge's target is int8. Static `qs8_qc4w` is the fastest arm on the M1 and on
      x86, but on the A72 it is **2.6-5.2x slower than `qd8`** (2041 against 466 ms at one thread).
      XNNPACK has no fast A72 kernel for it.
    - **XNNPACK cannot take a Q4_0 block as it is.** Q4_0's scale is signed (`d = max/-8` puts the
      largest-magnitude element on nibble 0), and `qb4w` refuses any scale that is not positive
      (`create` returns `invalid_parameter`). With d < 0, nibble 0 has no image under q -> 16-q, so
      about half the blocks would lose their largest element. Its scale is also BF16, not F16. A bridge
      therefore requantizes at load from a lossy source, or the exporter emits a positive-scale form.
      Input for the bridge ADR (step 3).
    - **Accuracy** (max |y - ref| / max |ref| against F32, on GAUSSIAN weights): Q4_0 6.5e-2 to 1.2e-1,
      `qb4w` 6.9e-2 to 1.3e-1, per-channel `qc4w` 9.3e-2 to 1.9e-1 (`qs8_qc4w` 9.4e-2 to 2.0e-1),
      int8 6e-3 to 1.2e-2, over every shape and machine. The q4 mode
      fills its weights from N(0, sigma). The ramp the other modes use put both ends of every 32-block on
      the 4-bit clamp, and ranked per-channel ABOVE block-32 (weight RMS on Gaussian data: Q4_0 1.66e-2,
      positive-scale block-32 1.77e-2). Harness-level only, as for int8.
    - **Open on x86: ggml's 4-bit `fc1` (N=3072) is unstable rep to rep.** Within one run, Q4_0 took
      415-546 ms and `rpk` 226-509 ms, while the XNNPACK arms interleaved with them held within 5%
      (`BENCH_DUMP_REPS=1` prints every repetition). Q8_0 on the same shape held at 301 ms in both
      morning runs. Not chased: the dev box is the least representative machine.
  * [x] **Direct I/O buffers — measured, and the cost was not the Lua boundary.** Per-input timers
    (2026-10-11) put the Lua-built masks at 2.4 ms of a 10.4 s Qwen3 prefill and 1.2 ms of a VITS
    call. The real cost was retained outputs COPIED into a consumer's inputs: 501 ms of whisper's
    8.86 s jfk run (cross-attention K/V, every step). Those are now read in place
    ([ADR-073](../adrs/adr-073-a-referenced-input-is-read-in-place.md)): 13.5 ms, with identical
    output.
  * [ ] **Device builds still copy referenced inputs.** ADR-073 refuses an alias on a hybrid builder,
    because the scheduler plans its split copies from each tensor's buffer at allocation time. On a
    device the copy is device to device, so measure a GPU decode loop (whisper on the 5090) before
    deciding it is worth lifting. *(T5's Lua-built encoder bias, the other residue of this item, is in
    the graph now: [ADR-074](../adrs/adr-074-t5s-encoder-bias-is-a-gather-over-a-distance-table.md).)*
  * [ ] **Serializing a COMPILED accelerator program.** The "fewer dispatches" half of LiteRT's
    delegate story has already been probed here, and the honest reading is that **a split count does
    not predict it either way**: the Metal `PAD` prototype removed 27 of 56 splits and bought 1.8% on
    the model it was measured on
    ([Retro-026](../retros/retro-026-three-nodes-were-half-the-runtime.md) §5.4), and the same kernel
    later measured **11.0% on VITS** (97.9 → 88.7 ms) and shipped as `ggml-0016`
    ([Retro-028](../retros/retro-028-three-closing-arguments-that-were-never-measured.md)). So neither
    "dispatch reduction is worthless" nor "it is the gap" is supported — it is per model and has to be
    measured per model. What is NOT measured at all is the other half: Vulkan and Metal recompile
    their shaders at every startup, and caching a compiled program is a cold-start win nobody here has
    put a number on. That is the one piece of LiteRT's fourth item that fits the GGUF-metadata thread
    naturally.
  * [x] **Plan once, hold once — the dynamic-shape version of the arena.** Measured 2026-10-11 on
    VITS: a graph REBUILD is ~4-5 ms of host work, so a multi-slot graph cache would buy ~1%
    ([Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)). The real cost was the
    compute-buffer shrink refaulting 111 MB on every short/long switch, which is now fixed by a
    window of under-use, plus `hold`/`release` controls for the host
    ([ADR-072](../adrs/adr-072-the-compute-buffer-shrinks-after-a-window-of-under-use.md)).
  * [x] **FlatBuffers — investigated and declined, so it is not re-derived.** A flatbuffer replaces a
    load-time parse, not the per-call ggml graph build; ExecuTorch's own `GRAPH_REBUILD.md` rebuilds
    too. And the container is not where TFLite's memory win comes from — the **arena planner** is,
    which precomputes every tensor offset once and reuses it forever *because TFLite shapes are
    static*. Loom's are not (`n_tokens`/`n_kv` are dynamic, which is why P4.0.15 buckets and keys
    graph reuse on the bucketed length), so the equivalent is already partly held by `gallocr` plus
    graph reuse, and the rest is a trade this engine made deliberately for dynamic length rather than
    a gap to close.
  * *Context: [Epic-05](../epics/epic-05-edge-performance.md). Whoever picks this up measures against
    a competitor build that is NAMED —
    [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md) is the standing rule, and the
    reason is that conda-forge onnxruntime is 1.86x faster than the PyPI wheel at the same version. A
    LiteRT baseline has the same hazard and no one here has pinned one yet.*
* [ ] **Write down that loom-exporter's tests run under `~/.venvs/piper`.** `python3` on the dev box
  resolves to **`~/.venvs/ovos`** — transformers 5.14.1, **no `sentencepiece`** — which is the
  Qwen3-ASR-only env, and `tests/ci` under it is `4 failed, 568 passed`. **All four are the
  environment and nothing else: under `~/.venvs/piper` the same four are green** (38 passed, verified
  2026-08-29). `test_spec_protocol` fails on ovos because `spm_tokenizer_export` cannot import without
  `sentencepiece`; the three `test_causal_lm_export` registry tests fail with
  `LinkError: MonolithicCall … supplies input(s) it does not declare: ['cache_position']`, a
  transformers-version difference in what the trace sees. **`piper` (transformers 4.57.6,
  sentencepiece 0.2.1) is the env for everything except Qwen3-ASR** and must not be upgraded — NeMo
  pins `~=4.53`. It is written nowhere in the repo, so the next person meets four red tests with no
  way to tell. Put it in loom-exporter's README, and note that piper is ~3x slower to run them
  (4m12s against 1m21s for the same two files).
* [ ] **`FLASH_ATTENTION`.** Unbuilt. The blocker is **the gate suite's exact-fp32 comparisons, not the
  hardware** — `ggml_flash_attn_ext` forces an F16 K/V cast. A GPU exists now and the trade still has not
  been made; it is a decision about verification. *Context:
  [ADR-016](../adrs/adr-016-kv-cache-shape.md)*
* [ ] **KV-cache addressing policies beyond contiguous append.** The `ggml_set_rows` indirection exists;
  `KvCache::fill_cell_index` is the single place a second policy would go. What is missing is a policy
  that uses it.
* [ ] **Quantized KV cache.** Storage is always F32. Different mechanism and different pipeline point
  from weight quantization — check how the cache is allocated and typed before assuming a trivial
  extension.
* [ ] **General multi-scheme quantization tool.** `quantize_gguf_q8_0.py` is Q8_0-only and
  single-model-shaped. A model-agnostic tool with a per-tensor-role policy (skip norm weights and
  embeddings) is unbuilt. *Scope note: [ADR-017](../adrs/adr-017-no-k-quants.md)*

## Backends & accelerators

* [ ] **NPUs.** Open, and the shape of the problem is known: no NPU registers as a `ggml` device type
  the engine can resolve, so `"npu"` throws by design. CoreML (the Neural Engine, which Metal is not)
  and RKNPU2 are out of tree and cost more, licence check included. *Context:
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md), [Epic-04](../epics/epic-04-backends-and-accelerators.md)*
* [ ] **The device hierarchy ranks a GPU above the CPU on unified memory, where the proxy does not
  hold.** The rule stands for "has its own fast memory", which an Apple Silicon GPU does not — it has
  the CPU's. Measured at HEAD with `ggml-0016`: `device=""` picks `MTL0` and is **1.76x faster on
  whisper-small and 1.63x SLOWER on VITS** (2.75x at Q4_0). It is why Metal ships as an extra rather
  than in the base macOS wheel; if this changes, folding it in is worth revisiting. **P4.30a, P4.30d and
  P4.30c step 5 between them took the VITS ratio from 8.98x to 1.63x and did NOT rescue the rule** — a
  GPU that is 1.6x slower on one model and 1.8x faster on another still cannot be ranked above the
  CPU by a rule that reads neither. This stays open on its own terms, with a much smaller number.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md)

* [ ] **On Metal, a Q4_0 convolutional model is now SLOWER than the same model at f32** — 141.7 ms
  against 88.7 on VITS, an inversion `ggml-0015` created and did not exist before it (149.7 against
  278.7); `ggml-0016` helps both arms and so leaves the ratio slightly WIDER, at 1.60x. The mechanism is known and is not the quantization: ggml-metal declines loom's folded
  block-quantized convolution kernel on its type test ([Epic-04 §5.2](../epics/epic-04-backends-and-accelerators.md)),
  so a Q4_0 export lowers through `im2col` + `mul_mat` and never reaches the new kernel — whose fast
  path is F32/F16-only by the same test. The choice is between teaching that type test about the
  folded kernel (P4.13's format) and dequantizing into the new kernel's fast path the way `ggml-0013`
  does on the CPU — and **P4.30c step 3 has since done the CPU-side equivalent of the second option
  for the 2-D form** (`op_conv_2d` hands a folded kernel to `ggml_conv_2d_direct_packed`, which
  dequantizes and re-enters), so the shape of the fix is now demonstrated on one backend. **Neither is on any critical path** — Metal is an extra — but "quantizing costs
  1.5x on this backend" is a surprising thing to leave undocumented in the model cards.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-017](../adrs/adr-017-no-k-quants.md)
* [ ] **Run `ggml-0024`'s tensor-API branch on a Mac that has one.** On an M5-class GPU, Metal's
  `mul_mm` uses the tensor API, which has no float-tile variant, so `ggml-0024` sends a `GGML_PREC_F32`
  F32 product to the mat-vec kernel instead. That branch compiles but has never run: the M1 Pro does not
  have the tensor API. SenseVoice on `jfk.wav` is the check, and the cost of the mat-vec fallback is the
  number to take. (`ggml-0023`'s coopmat and coopmat2 branches HAVE run, on the RTX 5090.)
  → [ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md),
  [Epic-04 §6](../epics/epic-04-backends-and-accelerators.md#6-gpu-correctness-f32-matmul-precision-the-metal-norm-the-cuda-mat-vec-stride-2026-10-08)
* [ ] **`device_report()` still buckets every node as either device or CPU**, deliberately — it does not
  say *why* a node fell back.
* [ ] **Whisper's 400-wide reflect pad** is cheaper to fall back on than to compose. CUDA, Metal and SYCL
  run it natively regardless.

## Text front-ends

* [ ] **Kokoro: does it want the misaki fold?** Kokoro was trained on misaki IPA and declares no
  `tts.phoneme_style` ([ADR-071](../adrs/adr-071-a-phoneme-table-declares-its-conventions.md)). A/B the door
  with and without the fold (Whisper WER) before re-exporting. VITS measured WORSE folded and declares none.
* [ ] **Task #79 part 2 — the C++ `orthography2ipa` port.** `src/text/phonemize.cpp` +
  `include/loom/text/phonemize.h`, vendored as an Apache-2.0 submodule, verified against the Python
  door as its oracle. Part 1 is closed — every phoneme-input TTS GGUF carries its symbol table and both
  hosts read it ([Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md)).
  **Measure both risks first:** the fold-down into each checkpoint's fixed symbol→id table, and
  pinning the beam search's tie-break. **This is now the only thing `loom_cli` cannot do for a TTS
  model**: it synthesises from IPA phonemes, from a grapheme vocabulary (Supertonic, from plain
  English), and from codec codes, so what remains behind this port is exactly "type English at a
  phoneme-input family".
  *Context: [ADR-012](../adrs/adr-012-permissive-phonemizer.md)*
* [ ] **Generalize the grapheme front end out of C++** — when a real second grapheme TTS model exists.
  Qwen3-TTS is not one. *Context: [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md)*
* [ ] **Remaining BPE pretokenizer families** beyond the ~40 in `pre_spec_table()` (CJK-script splitters,
  case-transition shapes, `byte_encode=false` SPM-style families). Each raises a named error rather than
  mis-tokenizing — bounded; add one when a real model needs it. **Distinct from the added-token pre-pass** P4.23
  shipped (`<|im_start|>` and friends, which `encode` could not emit at all) rather than pretokenizer
  regexes — but both land in `bpe_vocab.cpp`, so read
  [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md) before opening this one.

## Host API

* [ ] **Expose the compute-buffer controls in loom-py.** `LoomLuaBridge::hold_compute_buffers`,
  `release_compute_buffers` and `compute_buffer_bytes` exist in the engine, but a Python host cannot
  reach them yet. The LM-server case needs `hold`, because each request otherwise refaults its prefill
  buffer. Do it with the next submodule bump.
  *Context: [ADR-072](../adrs/adr-072-the-compute-buffer-shrinks-after-a-window-of-under-use.md)*
* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

**Standing process, not an open item:** every engine change needs a `vendor/loom.cpp` bump in loom-py
before loom-py's card gate can run against it — `git -C vendor/loom.cpp fetch origin <branch>` →
`checkout <sha>` → rebuild → commit the pointer. A release pins that submodule at `main`'s tip, and
**that pin is what the wheels are built from**.

* [ ] **Republish flan-t5-small once `feat/t5-bias-in-graph` (loom-exporter) merges.** The encoder's
  bias moved into the graph (ADR-074), so the GGUF changes: a `positions` input replaces
  `position_bias`, a 257x6 `enc_rel_bias` weight is added, and the driver drops `ENC_REL_BIAS`. It
  needs a fresh export from the merged tree, then the card gate, then publish, in that order.
  Verified on the branch: the encoder matches HF
  at f64 to 2.5e-7, and generation is identical to rc17's GGUF and to HF greedy on both test prompts.
  The engine needs nothing new.
* [ ] **"Nothing needs a re-export" is checked on cards and Hub hashes, never on a fresh export.** The rc15 and rc16 checks compared regenerated cards and staged-vs-Hub sha256, which says nothing about whether exporter `main` still WRITES the published file. It does not, for `gigaam-v3-rnnt` (published 2026-09-30): a fresh export adds `tokenizer.ggml.unigram_scoring` (ADR-060, encode-only) and an encoder-output PERMUTE/VIEW/PERMUTE/CONT; both decode JFK 80/80 (gate 83/83), so it is not a defect, but a release sweep should re-export each card and diff before saying nothing moved.
* [ ] **A GGUF cannot say which engine features its driver needs.** An older engine ignores option keys
  it does not know, so a driver whose meaning depends on an OPTION runs plausibly and wrongly there:
  rc10's `run_ode` would have integrated F5-TTS unguided, and Chatterbox's `min_p` and CosyVoice3's
  `top_p_mass`/`banned`/`uniform` sampler keys are the same hazard. A missing FUNCTION fails loudly
  instead. WHEELS FIRST covers this by process today; a requirement the GGUF declares and the loader
  checks would cover it by construction.
* [ ] **`nlohmann/json` is fetched as a full ~290 MB clone** for a header-only library, and it failed
  twice over a slow link during the macOS work. `GIT_SHALLOW TRUE` on that `FetchContent_Declare`
  (it is pinned to a tag, so shallow works) would remove the largest download in a cold build.

## Standing scope limitations

Deliberate boundaries rather than tasks, each naming what would have to change. Kept in
[Epic-01 §4](../epics/epic-01-inference-engine-core.md#4-standing-scope-limitations): single-sequence KV
cache, F32 cache storage, one level of `repeat_for` nesting, no chunked/windowed inference for long
Conformer-CTC audio, only the small Conformer-CTC checkpoint verified, and the attention-variant
primitive set.

## Minor cleanups

* [ ] **`REL_SHIFT` / `REL_POS_ATTENTION` are unused too, and so is a tail of legacy-converter gate
  tests.** Found while retiring the Shaw family and `pad_crop_relative_embeddings` (2026-10-11):
  - **The ops:** a GGUF-reader pass over all 118 published and v5-fixture GGUFs (503 topologies, plus
    a raw search of their 20.4 MB of topology JSON) found no use of either op. The NeMo conformers trace
    their relative positions into ordinary ops. Both still have CI tests in `test_primitive_registry`.
  - **The gate tests:** `test_e2e_vits_smoke` and `test_e2e_vits_flow_vocoder_reference` need the
    deleted `tools/convert_piper_vits` output and always skip. Many kokoro/supertonic/matcha/styletts2
    gate tests also skip on v5, but whether each one's fixture is producible was NOT checked.
  - **Work:** confirm each skip, then retire what no tree can feed. Lean-runtime cleanup, not urgent.
  *Context: [Epic-01 §4](../epics/epic-01-inference-engine-core.md#4-standing-scope-limitations)*
* [ ] `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()`, which throws `std::out_of_range`
  rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr could in principle reach
  this uncaught-by-`catch (loom::Error&)` path — low risk today, since the index always comes from
  `repeat_for`'s own loop bound.
* [ ] `export_config.py`'s module docstring points at a ledger section that no longer exists.
* [ ] **`GgmlPatches.cmake` asks "already applied?" the wrong way, so every `cmake` re-run rebuilds
  ggml from scratch** (~30 min on the Pi). It reverse-applies **each patch individually against the
  final tree**, which only holds while no later patch rewrites lines an earlier one added —
  `ggml-0004`..`0007` all fail it today. **Not a context-width problem**: regenerating one at `-U3`,
  `-U1` and `-U0` all still fail, because the added lines themselves are gone. The fix is a stamp file
  holding the applied set's names and hashes, skipped when it matches. Build-time cost only; the
  reset-and-retry path it falls into is correct.

---

## Knowledge Hub

| | |
|---|---|
| **Domains** | [Epics](../epics/) — what each area is and how it works |
| **Decisions** | [ADRs](../adrs/) — why a technical choice was made |
| **Lessons** | [Retros](../retros/) — what broke, why, and the takeaway |
| **Specs** | [SPECIFICATION](../SPECIFICATION.md) · [KV-CACHE](../KV-CACHE.md) · [HIGH-LEVEL-API](../HIGH-LEVEL-API.md) · [PROCEDURAL-GENERALIZATION](../LOOM_PROCEDURAL_GENERALIZATION.md) |
| **Closed detail** | [archive/](../archive/) — unmaintained, kept for the reasoning trail |

**Before opening a performance item**, read
[Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md).

**Before trusting a green gate**, read
[ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md) and
[Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md).
