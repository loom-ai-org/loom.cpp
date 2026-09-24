---
type: retro
date: 2026-09-24
domain: testing
tags: [gate, oracle, autoregressive, continuous-latents, float-precision, family-9, pocket-tts]
---

# Retro-055: A Feedback Loop Cannot Be Gated Free-Running

## The Issue

Pocket-TTS's first engine run, from the reference's text and its pinned per-step draws, gave the
right number of frames and a correct transcript. The waveform, though, was **3.8e-03** from the
reference's. The reference's own float32-vs-float64 spread on the same clip is **5.3e-04**, so the
standing rule ([run the reference at f64](retro-042-a-converters-op-default-changed-the-function.md) before
calling a gap rounding) said this was a defect: seven times the rounding floor.

It was not. The error was **4.6e-05 or less over the first half of the clip** and grew only after
about step 42. The torch wrappers, before any engine was involved, did the same thing: **7e-06 per
step** fed the reference's history, 2.2e-03 free-running.

## Root Cause

Every latent is the next step's input, so the loop is a feedback system over continuous values, and
a feedback system amplifies whatever enters it. Two correct f32 implementations differ in rounding
from the first step, and the loop carries each difference forward and grows it at a rate the
trajectory decides. The f64 arm is ONE sample of that process, from one perturbation. It is not a
bound on the next implementation's distance, because a different perturbation in a different
direction can grow faster.

A token loop does not behave like this: an argmax quantises the state every step and absorbs
perturbations until one flips a token. Every earlier AR gate here was a token loop. Chatterbox, Dia
and Qwen3-TTS are gated greedy.

## The Fix

**Teacher forcing.** The driver takes `teacher_latents`: every step still computes, emits and
EOS-tests its own latent, but feeds the caller's back. The gate
(`tests/gate/test_e2e_pocket_tts_lua_driver.cpp`) runs two arms:

| arm | max \|Δ\| | rmse | checks |
|---|---|---|---|
| teacher-forced | 1.5e-04 | 1.8e-06 | tolerance 2e-3 / 2e-5; sabotage (temperature 0.7) gives 0.82 |
| free-running | 3.8e-03 | 9.9e-05 | same frame count (EOS margin 1.4), rmse < 1e-3 |

The free-running arm is what checks the LOOP itself: the EOS decision and the tail.

## Takeaways

* **For a loop over continuous values, compare each step on the reference's history.** A
  free-running comparison measures the trajectory's sensitivity, not the implementation.
* **Localise before you conclude.** A per-segment error profile (flat, then growing) tells
  amplification from a defect. A defect is wrong from the step it touches. Growth is how
  amplification looks.
* **An f64 arm bounds a feed-forward pipeline, not a feedback loop.**
