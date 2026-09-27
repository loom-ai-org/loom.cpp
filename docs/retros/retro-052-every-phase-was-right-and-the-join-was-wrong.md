---
type: retro
date: 2026-09-22
domain: exporter
tags: [layout, verification, family-9, f5-tts, tts, asr-oracle]
---

# Retro-052: Every Phase Graded Clean and the Model Said Nothing

## The Issue

F5-TTS's export produced audio that an ASR oracle transcribed as **"(chimes ringing)"**. Peak 0.23
against the reference's 0.86.

Each of the four traced graphs had already been graded against torch, tensor for tensor, on the real
inputs:

| phase | max \|Δ\| | cosine |
|---|---|---|
| `mel` | 4.39e-02 | 0.999999881 |
| `text_embed` (conditional) | 1.13e-05 | 0.999999881 |
| `text_embed` (unconditional) | 8.11e-06 | 0.999999821 |
| `estimator` | 1.19e-05 | **1.000000000** |
| `vocoder` | 1.53e-05 | 0.999999881 |

Five clean comparisons, and the model was unintelligible.

## Root Cause

**The estimator retains FRAME-major mel and the vocoder declared CHANNEL-major, and the driver handed
one straight to the other.**

```
estimator output  ne = [100, n_tokens]      100 contiguous floats per frame
vocoder  input    ne = [n_enc_frames, 100]  n_enc_frames contiguous floats per channel
```

`Vocos.decode`'s own convention is `(B, C, L)`, so tracing its wrapper as written declared a
channel-major input. The estimator's is `(B, L, C)`, because that is what its own `x` and `cond` inputs
are. Nothing converts between them — [ADR-031](../adrs/adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)
and [ADR-032](../adrs/adr-032-an-interleave-is-a-layout-a-concatenation-is-a-graph.md) deleted the
host-side layout converters on purpose — so the vocoder read a transposed spectrogram.

**A frame-major array read as channel-major is still a plausible spectrogram.** Same element count,
same value range, no shape error, no assert. The vocoder does what it is asked and produces a sound.

Measured directly, by handing the fixed vocoder graph the transposed mel it used to get:

| arm | max \|Δ\| | cosine | peak |
|---|---|---|---|
| frame-major (correct) | 1.53e-05 | 0.999999881 | 0.8557 |
| **transposed (the bug)** | 8.49e-01 | **−0.007813632** | 0.2085 |

Cosine −0.008 is the signature: not "close but wrong", **uncorrelated**. And the 0.2085 peak is the
0.2312 the end-to-end run printed, which is how the two were tied together.

## What Found It, and What Could Not

The **ASR oracle** — transcribe the output and read it —
[Retro-006](retro-006-kokoro-shipped-noise.md)'s standing rule, earning itself again. The reference's
own output transcribes as the target sentence; the export's transcribed as a sound effect.

**Per-phase tensor comparison could not find it and never could have.** It grades each graph on inputs
*supplied in that graph's own layout*, which is exactly the assumption that was false. The vocoder
probe passed at 1.53e-05 *because* it was handed channel-major mel from a `.npy` — the one thing the
driver does not do.

The peak was also a signal and was read too late: 0.23 against 0.86 was printed by `loom_cli` on the
first successful run, and that print exists because Retro-006 added it. After the fix it reads 0.9768
against the reference's 0.8557, and the transcript is the target sentence exactly.

## Takeaways

* **A verified phase and a verified pipeline are different claims.** Every join between two graphs is
  an unchecked assumption until something exercises it, and the count of clean per-phase numbers says
  nothing about it. Five green comparisons were five statements about five graphs.
* **The consumer transposes, in its own graph.** The producer cannot: a bare `.transpose()` as a traced
  graph's declared output is a live GGML permute view that `ggml_backend_tensor_get` silently ignores
  (`vits_export.py`'s `TextWrapper` paid for that). Converting in Lua is what ADR-031 refuses — 86,800
  elements across the boundary for a reindexing no host decision depends on. So the fix is one
  `mel.transpose(1, 2)` at the top of `F5VocoderPhase.forward`, and the vocoder's declared input
  becomes frame-major.
* **Pin the join as a relationship, not as a fact about one side.**
  `test_the_vocoder_takes_the_layout_the_estimator_produces` compares the two phases' `symbolic_shape`s
  against each other. A test that asserted "the vocoder takes `(1, 100, m)`" would have passed on the
  broken export, because that was true and was the bug.
* **An oracle that cannot read the output is not an oracle for a generative model.** Cosine against
  PyTorch is necessary and is not sufficient; for anything whose answer is audio, listen to it or
  transcribe it.
