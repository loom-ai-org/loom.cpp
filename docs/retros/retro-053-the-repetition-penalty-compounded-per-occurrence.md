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

It surfaced while Chatterbox was being scoped: that model uses 1.2 over hundreds of speech tokens,
and a token model that repeats a unit would have been pushed off it progressively harder. **It had
already failed, in Qwen3-TTS**, the only shipped driver that passes a penalty. At only 1.05, it moved
a greedy decode off the reference at frame 6 of one sentence. It was filed as an open item on
2026-09-17, then written off as a harness mix-up on 2026-09-24 (below).

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

## It WAS the Qwen3-TTS divergence, and the first A/B said it wasn't

The hub's open Qwen3-TTS item said "the reference repeats a frame and loom does not; penalty 1.0
reproduces the reference 624/624 and the checkpoint's 1.05 gives 96/624". When this fix landed, an A/B
was run: 624/624 at 1.05 under **both** semantics. On that evidence the item was rewritten as "the
recorded numbers do not reproduce". **The A/B ran on the wrong sentence.** The 2026-09-17 numbers were
graded on `tokens.txt` ("Loom runs this model from one file.") against `reference_xvec40.txt`. The
re-run used the fixture directory's newer sentence, `tokens_fox.txt` against
`reference_xvec_fox.txt`, where the compounding happens never to flip an argmax.

Closed 2026-09-24 with one variable changed. The branch HEAD was rebuilt with only the `done` guard
removed, in a separate tree. All runs are `run_talker`, `GREEDY=1 MAXNEW=40 XVEC=xvec.txt`, graded over
39 frames:

| engine | sentence 1, ckpt 1.05 | sentence 1, 1.0 | fox, ckpt 1.05 | fox, 1.0 |
|---|---|---|---|---|
| per-occurrence (guard removed) | **96/624**, frame 6 | — | 624/624 | — |
| per-id (branch HEAD) | **624/624** | 624/624 | 624/624 | 99/624 |
| released `loom-py-rt==1.0.0rc10` | **96/624**, frame 6 | 624/624 | — | — |

The per-occurrence run is **bit-identical** to the 2026-09-17 `loom_xvec40.txt`, and so is the rc10
wheel's. Re-running the reference at an explicit penalty gives the same 39 frames at 1.05 and at 1.0
on sentence 1. That is why "1.0 reproduces the reference" was true: once per id, 1.05 never changes
the argmax on that sentence, and compounding does. The fox sentence shows the penalty is not
decorative (1.0 gives 99/624 there), and the talker's default of 1.05 stays.

**The shipped consequence:** on the released rc10 wheel, the published Qwen3-TTS talker's greedy
decode can diverge from `transformers`, and it does on sentence 1. rc11 carries the fix; the GGUF
needs no re-export.

## Takeaways

* **Port a gather/scatter processor as a set operation, not a loop.** Before transcribing any
  `scatter` or `index_put`, ask what happens to duplicate indices.
* **A test history with no repeats cannot test a repetition penalty.** The duplicate case is the
  whole behaviour.
* **Credit a fix to a symptom only after an A/B.** Run the old code on the symptom's repro.
* **An A/B that says "not this" must run on the input that recorded the symptom.** A fixture
  directory accretes inputs, and the newest is not the one the bug was seen on. Before ruling out a
  cause, reproduce the recorded number with the OLD code: matching the old output bit for bit (here,
  96/624) proves you are on the right input. A different input that agrees under both arms proves
  nothing about the symptom.
