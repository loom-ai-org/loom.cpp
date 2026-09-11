---
type: retro
date: 2026-09-12
domain: exporter
tags: [oracles, verification, family-11, audio, measurement]
---

# Retro-043: The Band Was 21 dB Down, and It Was Audible

## The Issue

SNAC's decoder adds `randn * linear(x)` at four points, so the reference model is stochastic. The
first export dropped the term and shipped the conditional mean — a decision taken on measurements,
recorded in [ADR-029](../adrs/adr-029-a-multi-rate-codec-keeps-one-row-per-coarsest-frame.md), and
reversed one listening session later.

Every number supported dropping it:

* removing the noise moved the waveform **2.4% in relative RMS**, against a **3.1%** spread between
  two seeds of the reference — i.e. the mean sits *inside* the model's own variation;
* the ASR oracle read **22/22 words** either way;
* a per-band spectral comparison against the source, on native 24 kHz speech and music, found every
  band where the two arms differ by more than 0.2 dB to be **more than 20 dB down**. The single real
  gap was 1.75 dB at 10–12 kHz on voiced frames, in a band carrying under 1% of the clip's energy.

Given the source and both decodes, a listener described the deterministic one, unprompted, as *"less
sharp, slightly more artificial"*.

## Root Cause Analysis

**The measurement was right and the inference from it was wrong.** The table said the difference lives
in the top octave on voiced frames; that is breath. The step that failed was converting "21 dB below
the clip's full-band energy" into "inaudible", which treats audibility as a share of total energy. It
is not one: breath is a *timbre* cue, and its contribution to broadband level (~0.02 dB) has almost
nothing to do with whether a listener hears its absence.

Two smaller things helped it along:

* **The oracle that could have caught it was aimed elsewhere.** The ASR oracle exists to answer *is
  this audio* (Retro-006, where Kokoro matched PyTorch at cosine 0.996 and shipped unintelligible).
  Sharpness is not intelligibility, and 22/22 was never evidence about it.
* **Cost pressure pointed the same way.** Restoring the noise meant a new driver component, four
  declared axes, and a stochastic codec every gate must seed. When the numbers are ambiguous and one
  answer is free, it takes deliberate effort not to read them as agreeing with it.

## What It Cost to Fix

Less than the analysis that avoided it: a `NOISE` binding kind in `DriverInputs`, one `noise_inputs`
kwarg, `declared_axes` entries the exporter already supported, and a wrapper that takes its noise as
arguments. Half a day, and the mechanism generalises to any future family with a stochastic leaf.

The reversal also *improved* verification rather than weakening it. Because the driver lets a caller
supply the arrays (`inputs.noise_0 or loom.gaussian_array(...)`), the oracle hands both sides the same
draw and stays exact — **max |Δ| 1.2e-06** — where a stochastic export with no such door could only
have been compared distributionally.

## Takeaways

* **For anything a person will listen to, a measurement scopes the question and a listener answers
  it.** The spectral table was worth computing: it predicted exactly where the difference was. It
  could not rank two waveforms for naturalness, and it was never able to.
* **"N dB down" is not a unit of audibility.** Energy share answers "how much of the signal is this",
  not "would someone notice it missing". For a high band, breath, or reverb tail, those are different
  questions with different answers.
* **Notice when the cheap answer is the one your evidence keeps supporting.** Both readings here were
  defensible; only one of them also happened to save a day of work, and that is the moment to get a
  second opinion rather than a third metric.
* **A stochastic export should let the caller inject its randomness.** It costs one `or` in the driver
  and it is the difference between an exact oracle and a distributional one.
