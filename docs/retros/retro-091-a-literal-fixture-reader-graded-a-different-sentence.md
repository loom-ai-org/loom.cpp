---
type: retro
date: 2026-10-23
domain: testing
tags: [gate, fixture, json, unicode, pocket-tts, family-9, harness]
---

# Retro-091: A Literal Fixture Reader Graded a Different Sentence

## The Issue

Checking an ASR outlier on Pocket-TTS's 7m model, I pinned the reference's noise for "It costs 3.14
dollars — or maybe a little more." and ran the gate over eight seeds. Six crashed when the pinned
noise ran out. Two stopped at 19 frames where the reference ran 47. That looked like a broken EOS
head on the model. The same test had passed 61/61 on the default sentence a minute earlier.

## Root Cause Analysis

The gate reads the text from the reference's `meta.json` with a literal substring search. Python's
`json.dumps` escapes non-ASCII by default, so the em dash reached the vocabulary as the six characters
`—`. The test then fed the model a different sentence from the one the pinned noise and latents
belonged to. The first check, ids against the reference's, did fail (line 112), but it was one line
among thirty failures and a crash. The default sentence is ASCII, and every fixture before v4 was too,
so the reader had never met an escape. v4's `replace_characters` acts on exactly these characters.

## Resolution & Lesson Learned

`pocket_tts_reference.py` writes `meta.json` as UTF-8 (`ensure_ascii=False`). `meta_text` throws on a
backslash, so a stale or foreign fixture fails by name instead of grading another sentence. Regenerated,
the eight seeds reach the reference's EOS frame on every one.

**A fixture reader that cannot decode something must refuse it, not pass it through.** And when a run
fails many checks at once, read the FIRST failure: here it was the ids, which said the inputs were wrong
before any number about the model was printed. Related:
[Retro-090](retro-090-the-baseline-arm-was-not-the-engine.md) (another harness that tested the wrong
thing cleanly).
