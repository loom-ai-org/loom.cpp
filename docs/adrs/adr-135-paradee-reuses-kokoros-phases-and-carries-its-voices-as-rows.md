---
type: adr
status: accepted
date: 2026-11-18
domain: model-coverage
tags: [model-coverage, tts, kokoro, distillation, source-filter, voices, phonemizer, streaming, paradee]
supersedes: []
---

# ADR-135: Paradee Reuses Kokoro's Phases, Carries Its Voices as Rows, and Phonemizes as Kokoro's Pipeline Does

Serves [Epic-03](../epics/epic-03-model-coverage.md), [Epic-06](../epics/epic-06-high-level-api-and-hosts.md)
and [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). Follows tiny-voca's split of a
source-filter head ([ADR-134](adr-134-tiny-voca-splits-its-pulse-train-at-the-frame-and-draws-torch-noise-in-the-engine.md))
and [ADR-071](adr-071-a-phoneme-table-declares-its-conventions.md)'s declared phoneme style.

## Context

Paradee (sahilmahendrakar, Apache-2.0) is Kokoro-82M distilled into 7.88M parameters. Its text side is
Kokoro's own `CustomAlbert`, `ProsodyPredictor` and `TextEncoder` at smaller widths, plus an MLP projecting
to the teacher's 512 channels. Its decoder is Kokoro's AdaIN trunk. From 1.1 on, Kokoro's upsampling
iSTFTNet generator is replaced by a head that never upsamples. At 80 frames a second, ConvNeXt blocks
predict gains on 80 mel-spaced points. A sine bank at the F0 and its 64 multiples is filtered by a
smooth gain and phase in an STFT of 600 taps, hop 150. White noise, and noise pulsing with the pitch,
are shaped beside it, with a second 3 ms-window noise path. Version 1.3 (2026-11-09) has eight of
Kokoro's voices in four languages: one learned row per voice in each half's style table. The user linked
1.1 (one voice) and chose 1.3, the same graph with the table.

## Decision

1. **The text side and the trunk are Kokoro's export.** `paradee_export.py` calls `kokoro_export`'s
   phase builders on the student's modules unchanged. That gives ALBERT as one graph, the six BiLSTMs as
   swept cells, and the AdaLayerNorms, duration head and F0/N stacks. Its driver makes Kokoro's calls,
   inlined into a per-sentence function. New: `asr_proj` before the frame expansion (applying it per
   token is the same as after the repeat), and one `synth` graph holding the trunk and the head.
2. **The sine bank's running phase is the driver's, per frame.** Upstream sums `f0 * 300 / 24000`
   cycles in float64 and wraps it, because harmonic 64 multiplies any error by 64. The driver reads the
   F0 curve back (`loom.get_output`) and sends each frame's wrapped start and its advance to the next
   centre. In the graph each sample's phase is `[start, advance] @ [1, ramp]`, cropped half a frame at
   each end, which is upstream's interpolation between frame centres with no slice of a dynamic axis.
   Outer products are K = 1 matmuls, since ggml broadcasts only the second operand. STFTs are
   convolutions with upstream's own constant bases. The noise is `aten_randn` (ADR-134), and its reverse
   feeds the 3 ms path. The harmonics are tiny-voca's closed form, `sin(K t/2) sin((K+1) t/2) / sin(t/2)`
   with theta centred, K counted per F0 frame by upstream's own `k * f0 < 11000` test. Against upstream's
   sum in f64 it is 1.5e-4 off where the 64-row f32 sum was 2.7e-4. It took a dev-box synthesis of 5.85 s
   from 470 to 362 ms (1.30x), and the gate's SNRs did not move.
3. **The voices are rows the file carries.** The two style tables go into the GGUF. The driver's `voice`
   input is the row, and `loom.tts.voices` names the rows. Each voice speaks one language, so
   `loom.tts.voice_languages` (new, engine contract and binding) declares it beside the voice.
   loom-py's `voice="ff_siwis"` becomes the row for a model that takes no voice files, and that voice's
   language becomes the G2P's unless the caller names one.
4. **A `kokoro` phoneme style.** Kokoro's pipeline writes English with misaki (one symbol per diphthong,
   punctuation kept) and every other language with espeak's IPA (ties compressed, length and aspiration
   kept). loom-py's door uses misaki when it is installed, and Paradee's own Apache-2.0 G2P
   (`paradee.g2p`) for Hindi, French and Spanish when that is installed. These are upstream's own
   runtime's front ends. Otherwise orthography2ipa runs between punctuation marks (it drops them, and
   the model reads a comma as a pause) and is folded per language.
5. **A sentence at a time.** Upstream cuts its text after `.`, `!`, `?` or `…` followed by whitespace. The
   driver cuts the ids at the same place, speaks each sentence with seed + i, and the stream yields one
   piece per sentence.

## Options not taken

* **Voice files per voice (ADR-045), as Kokoro's packs.** Eight 48-float rows do not need eight Hub files
  and a fingerprint. Kokoro's packs are length-dependent (one style per phoneme count); Paradee's rows
  are not.
* **Reusing `misaki` as the declared style.** Vocetta2 declares it and the fold is English-only: it turns
  `r` into `ɹ` and drops `ʰ` and `ː`, which are Spanish's trill and Hindi's aspiration and length.
* **Upstream's 64-row sine sum.** It was the first export, and it cost a [64, samples] SIN, outer product
  and reduction (~15% of a profiled synthesis). It was replaced by the closed form (decision 2).

## Consequences

* The engine against upstream's own training code (`torch.stft` head), on all eight voices: every
  duration and length upstream's, and 55.9 to 67.2 dB SNR. That is upstream's float32 noise: the synth
  graph run eagerly in torch is 63-69 dB from upstream, and at f64 the two agree to 111 dB.
  `test_e2e_paradee` holds five cases (four languages, two sentences, a speed change) at 50 dB, plus the
  seed and the stream. A one-line phase sabotage takes it to 4.8 dB.
* **Q8_0_PC keeps ALBERT in float** (19.5 MB, 60% of the weight bytes in int8). With every weight in
  int8 (17.3 MB) English read at 35.2% WER on ten sentences, against F32's 4.0%. Keeping ALBERT float
  recovers it, and so does keeping the five BiLSTMs float instead (2 MB more). ALBERT applies one shared
  layer six times, and the engine's int8 GEMMs round activations as well as weights. A phase is named
  float through the exporter's new `float_phases`, because its weights' traced names carry hashes.
  Quantized F0 moves the running phase, so the Q8 waveform is judged by ASR, not by SNR.
* A file needs loom-py 1.0.0 for the `kokoro` style and for voices by name. Its ops are all rc18's.
* **On the boards** ([BENCHMARKS](../BENCHMARKS.md#raspberry-pi-4), 2026-11-18, 5.85 s of speech): on the Pi 4,
  loom takes 0.76 s (F32) and 0.68 s (Q8_0_PC), and upstream's ONNX files in onnxruntime take 0.74 and 0.81 s, so
  0.97x and 1.20x. It was 1.51 and 1.29 s until the recurrent sweep ran its cells on one thread
  ([ADR-136](adr-136-a-recurrent-sweep-runs-its-cells-on-one-thread.md)); Paradee's six BiLSTMs are about 1 480
  per-timestep graphs. On the Pi Zero, CPU 15.73 s and CPU+VC4 11.83 s (Q8_0_PC both), 1.33x. The VC4's output drifts in phase from the CPU's (18.7 dB, spectra equal), and
  whisper hears every arm as the sentence.

Word error rate through the door, whisper-small, 2026-11-18: 20 held-out English sentences (10 for
bf_isabella), and 15 each in French, Spanish and Hindi (CER for Hindi, whose spellings vary). "Own" is
misaki for English and Paradee's G2P for the rest; the fold is orthography2ipa folded to `kokoro`, with
misaki's lexicon (`set_lexicon`) for English or without.

| voice | own, F32 | own, Q8_0_PC | fold + lexicon | fold alone |
|---|---|---|---|---|
| af_heart (en-US) | 3.7% | 3.7% | 8.9% | 28.2% |
| bf_isabella (en-GB) | 4.5% | 4.5% | 17.1% | 37.2% |
| ff_siwis (fr) | 2.8% | 2.8% | -- | 6.4% |
| em_alex (es) | 1.1% | 1.1% | -- | 1.1% |
| hf_alpha (hi) | 13.0% (CER 6.4%) | 15.3% (CER 8.7%) | -- | 20.6% (CER 8.1%) |
