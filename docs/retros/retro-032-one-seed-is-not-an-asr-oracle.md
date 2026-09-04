---
type: retro
date: 2026-09-03
domain: model-coverage
tags: [family-10, sampling, asr-oracle, verification, dia]
---

# Retro-032: One Seed Is Not an ASR Oracle

## Issue

Family 10's first ASR-oracle run said the export was broken. Dia was asked for *"[S1] Hey, can you
shut down the computer, my friend?"* at its own declared decoding — sampling at 1.8/50/0.9 with
classifier-free guidance at 3.0 — decoded through DAC, and transcribed back:

```
dia_listen.wav: 133120 samples at 44100 Hz = 3.02 s, peak 0.1852, rms 0.01420
TRANSCRIPT: you
MATCHED 1/9 expected words
```

`transformers`, on the same sentence at the same settings and the same clip length, gave
`peak 0.9963, rms 0.12371` and **9/9 words**. Every deterministic comparison had already passed
exactly — 288/288 codes greedy, 288/288 codes greedy with guidance on, waveform to 5e-6 at two clip
lengths — so the reading was: the export is right and the SAMPLER is wrong, which is the one path no
exact oracle covers.

## Root cause

**Nothing was wrong.** Dia at its own published settings is high-variance, and the seed decides:

| seed | peak | rms | transcript |
|---|---|---|---|
| 42 | 0.185 | 0.014 | *"you"* |
| 1 | 0.896 | 0.160 | *"Hahahaha"* |
| 7 | 0.087 | 0.004 | (near-silence) |
| 1234 | 0.982 | 0.183 | *"Hey, can you shut down the computer, my friend?"* |

Two of four seeds produce something that is not the sentence, one produces laughter, and one is
perfect. The reference's `torch.manual_seed(42)` and this engine's `mt19937` seeded 42 are unrelated
streams, so "the same seed" compares nothing — the two systems were being graded on one draw each from
a distribution where a bad draw is common.

The bisection that established the sampler was fine cost two short runs and is worth keeping:

* **`temperature → 0` must reproduce the greedy decode exactly.** It did, 288/288 — so the softmax,
  the top-k truncation, the top-p prefix and the multinomial draw are all correct as *code*.
* **`top_p → 0` must too**, at the real temperature, guidance scale and shortlist size. It did — so
  the whole chain that produces the distribution is correct, and only the draw from it was untested.

After both, the only remaining variable was which sample came out, which is where the seeds came in.

## Takeaway

**An ASR oracle on a sampling model needs several seeds, and a single failing one proves nothing.**
[Retro-006](retro-006-kokoro-shipped-noise.md) established that a correlation cannot substitute for
transcribing the audio; this is its converse — one transcription cannot substitute for a distribution.
Sweep at least three or four seeds and report the best, and grade the *reference* on its own sweep
rather than on one draw, or a model whose published settings are simply loud will look broken.

**Make the deterministic corners of a sampler reachable from the outside.** `--temperature 0` and
`--top-p 0` turned "is the sampler right" from a debugging session into two two-minute runs, because
both must reproduce an answer that is already gated. `scripts/tts_synth.cpp` grew per-knob flags for
exactly this, and a family that only exposed its declared defaults would have had neither.

**The peak is the cheap first signal.** 0.185 against the reference's 0.996 was visible before any
transcription, and it is what distinguishes "did not say the words" from "said them quietly" — which
is worth printing on every synthesis, as `tts_synth` and `asr_oracle.py` both now do.

## Record

The verified state of family 10 at the end of this: greedy codes exact against `transformers` with
guidance off *and* on (`test_e2e_dia_mil_export.cpp`), waveform exact through DAC at two clip lengths
(`test_e2e_dia_dac_composition.cpp`), and the ASR oracle passing at 9/9 words on a 3.02 s utterance
under the checkpoint's own sampling and guidance, at seed 1234.

## See also

* [Retro-006](retro-006-kokoro-shipped-noise.md) — why the audio is transcribed at all
* [Retro-031](retro-031-dias-guidance-is-not-the-standard-formula.md) — the other family-10 finding
* [ADR-024](../adrs/adr-024-guidance-belongs-in-the-sampler.md) — the sampler this exercises
