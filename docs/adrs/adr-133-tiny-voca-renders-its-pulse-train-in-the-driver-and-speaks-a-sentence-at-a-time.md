---
type: adr
status: accepted
date: 2026-11-17
tags: [model-coverage, tts, dsp, source-filter, streaming, phonemizer, dizione, tiny-voca]
supersedes: []
---

# ADR-133: tiny-voca Renders Its Pulse Train in the Driver and Speaks a Sentence at a Time

Serves [Epic-03](../epics/epic-03-model-coverage.md), [Epic-06](../epics/epic-06-high-level-api-and-hosts.md)
and [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). Follows AuroraTTS's DSP split (its
phase carried in the driver, `auroratts_export.py`) and [ADR-071](adr-071-a-phoneme-table-declares-its-conventions.md)'s
declared phoneme style.

## Context

tiny-voca (n4ze3m, Apache-2.0, 4.5M parameters) runs a transformer encoder, a duration predictor, and
two frame transformers that predict pitch/voicing/energy and then the parameters of a source-filter head.
It has no neural vocoder. The head shapes a band-limited pulse train at the predicted F0, and white
noise, each with a time-varying filter. The filters are complex cepstra applied in the STFT domain. Four
things decide the port:

* The pulse train is computed in **float64** upstream ("Float64 keeps the phase exact"): a log-linear F0
  per sample, its running sum, and a closed-form sum of the harmonics. ggml has no float64.
* The STFTs use `torch.stft`/`istft`, which no graph lowering here takes.
* Upstream reads a text **one sentence at a time** (`speak_text`), each sentence its own utterance and
  noise seed, and joins the audio.
* Its phonemes come from **dizione**, the author's Apache-2.0 front end: 59 phones, stress as a separate
  input, `<sp>` between words.

## Options

1. Port the pulse train into the graph at float32 (upstream's own ONNX export does: phase steps as
   24-bit integers). Its error grows with the harmonic number.
2. **The driver computes the pulse train in double precision** (Lua numbers are doubles), and the graph
   gets float32 samples, the rounding upstream itself ends with.
3. A new engine primitive for the pulse train: per-model C++, which ADR-003 forbids.

## Decision

Option **2**, and:

* **Four graphs** (`encode`, `prosody`, `acoustic`, `synth`). Durations, frame owners and position
  features, the vowel and pitch-range adjustments, and the F0 run in the driver at float32, with
  round-half-to-even durations. The synth graph's STFTs are convolutions with the windowed DFT basis
  (upstream's own ONNX construction), with static crops at both ends. Its sub-frame lookups are gathers
  at indices the driver computes. The interleaved rotary rotation is spelled as a reshape to pairs.
* **Noise is ATen's own stream** (`aten_randn`), seeded `seed + sentence`, as upstream seeds it. A seed
  alone renders the reference's audio (6e-8 from the pinned draws). `noise` pins it for a gate.
* **The phoneme table** holds every symbol plus every phone with primary (`ˈ`) and secondary (`ˌ`)
  stress, at `VOCAB + k` and `2 VOCAB + k`. A dizione string encodes by longest match with the stress in
  the id, and the driver splits it back.
* **Utterances** are the input's `<bos> ... <eos>` runs. The text door cuts sentences the way
  `speak_text` does (`sentences(finished(text))`, in loom-py's `phonemizers`). The driver adds the final
  full stop and refuses an utterance over 512 tokens.
* **The text door (the user, 2026-11-17)**: the model declares `tts.phoneme_style = "dizione"`. loom-py
  uses dizione itself when installed. Otherwise orthography2ipa's IPA is folded to espeak's conventions
  and segmented into dizione's inventory.
* **Streaming**: one piece per sentence, the reference's own unit. Each piece is peak-limited on its
  own, where `infer` limits the joined audio once.

## Consequences

* Gate `test_e2e_tiny_voca` against `scripts/tiny_voca_reference.py`, with three cases (two voices,
  speed/pitch options, four sentences). Every duration is exact. The waveform is within 2.6e-3 of
  upstream's f32. Measured against upstream at f64, upstream's own f32 is 5.7e-4 away and the engine
  2.1e-3: the f32 convolutions against torch's FFTs; the same graph at f64 is 5.8e-8 from upstream.
  Sabotages (pitch reference, per-sentence seed) go red.
* Upstream's "a vowel is voiced" guard never fired on 106 ordinary sentences in either voice, so no
  gate case observes it; it is ported as written.
* **The text door, measured** (Whisper small, 30 LibriSpeech dev-clean sentences, through loom-py):
  4.4% word error rate through dizione, 20.0% through the orthography2ipa fold. Q8_0_PC (6.9 MB, 93% of
  the weights per-channel int8) reads 3.8% through dizione.
* **Raspberry Pi 4** (docs/BENCHMARKS.md, footnote 4): 5.39 s of audio in 0.668 s at F32, where upstream's
  own ONNX file takes 0.516 s under onnxruntime (0.77x), and 0.630 s at Q8_0_PC, against 0.643 s for a
  `quantize_dynamic` int8 of that file (1.01x per second of audio, parity). The F32 gap is open in the
  hub. One idea is measured out: reordering each head's q/k rows so the rotation is two halves, instead
  of a reshape to pairs, ran at 664 ms against 667 ms. A per-node profile had blamed the pair
  spelling's many small nodes for a third of the time, but that was the profiler's per-node overhead.
