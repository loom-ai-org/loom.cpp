---
type: retro
date: 2026-09-24
domain: inference-engine
tags: [sampling, repetition-penalty, family-9, chatterbox, family-10, qwen3-tts]
---

# Retro-053: The Repetition Penalty Compounded Once per Occurrence

## The Issue

`loom.sample_row`'s repetition penalty applied once for every ENTRY in `penalized`. An id drawn k
times therefore sat `penalty^k` lower. `transformers`' `RepetitionPenaltyLogitsProcessor` penalises
each id exactly once, however often it recurs. Checked directly:
`P(1.2)([[0,0,0,1]], s) == P(1.2)([[0,1]], s)`.

Nothing had failed. Qwen3-TTS, the only shipped driver that passes a penalty, uses 1.05 over short
histories, and there the compounding never flipped an argmax. It surfaced while Chatterbox was being
scoped: that model uses 1.2 over hundreds of speech tokens, and a token model that repeats a unit
would have been pushed off it progressively harder.

## Root Cause

`gather`/`scatter` hides the semantics. The processor reads `scores.gather(1, input_ids)`, penalises
that copy, and writes it back with `scatter`. Every duplicate of an id writes the same once-penalised
value to the same slot. A loop over the history that modifies the score in place is the natural C++
reading of that code, and it is wrong exactly when the history repeats. The engine test had no
repeated id in any history it passed, so it could not tell the two readings apart.

## The Fix

Each id is penalised at most once, via a `done` bitmap over the window
(`src/core/lua_bridge.cpp`). `tests/ci/test_sample_row.cpp` §9 picks a penalty for which penalising
once keeps the best class on top and penalising three times would not. It then checks that the
history `{best, 0, best, best}` still returns the best class. With the old loop restored, that check
fails (50/51), so it can go red.

## What It Was NOT

The hub's open Qwen3-TTS item ("loom does not repeat frame 5; penalty 1.0 reproduces the reference
624/624 and 1.05 gives 96/624") looked like this bug, and it isn't. Re-run with the oracle's own inputs
(`xvec.txt`, not `xvec_fox.txt`), the result is 624/624 at 1.05 under **both** the old and the new
semantics, and 99/624 at 1.0. The A/B is what settled it, and the hub item now records that its
numbers do not reproduce. Had the fix been credited because it fits the story, the real question,
which build produced the recorded numbers, would have been closed without being answered.

## Takeaways

* **Port a gather/scatter processor as a set operation, not a loop.** Before transcribing any
  `scatter` or `index_put`, ask what happens to duplicate indices.
* **A test history with no repeats cannot test a repetition penalty.** The duplicate case is the
  whole behaviour.
* **Credit a fix to a symptom only after an A/B.** Run the old code on the symptom's repro.
