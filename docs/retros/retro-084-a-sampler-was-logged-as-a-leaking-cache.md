---
type: retro
date: 2026-10-11
domain: inference-engine
tags: [sampling, kv-cache, benchmark-harness, gemma, misdiagnosis]
---

# Retro-084: A Sampler Was Logged as a Leaking Cache

## The Issue

While measuring ADR-072 on gemma-3-270m-it, `scripts/bench_lm_loom.cpp` aborted with "infer_with_past
returned different tokens on run 0 -- the KV cache is leaking state across calls". Four generations in
one `Session` gave four different sequences, differing even at the first token. Qwen3-0.6B passed the
same harness. The behaviour reproduced on `main`, so it was logged on the hub as an engine correctness
bug: "anyone generating twice in one process with this model gets wrong output". Gemma 3's interleaved
sliding-window layers were named as the likely cause.

There was no bug. gemma-3-270m-it's `generation_config` declares **sampling**: temperature 1.0, top_k
64, top_p 0.95. The export carries those as `loom.sampling.*`, and its driver passes them to
`loom.sample_row`. Successive runs draw from one RNG stream that keeps advancing, so they differ by
design. Qwen3-0.6B-Base declares greedy.

## Root Cause Analysis

Two things, one in the harness and one in the diagnosis.

* **The harness asserted determinism without asking for it.** It inherited each driver's defaults and
  read any difference between runs as a cache leak. That was right for every model it had been run on,
  all of them greedy, and wrong the first time a checkpoint declared a sampler.
* **The diagnosis believed the harness's message and went looking for a mechanism to fit it.** It never
  asked the cheaper question: is this model deterministic by declaration at all? The sampling keys
  were in the GGUF's metadata the whole time.

What settled it: greedy (`top_k = 1`), and sampling with the RNG reseeded before each run, both gave
four identical generations. That held on a chat prompt, on the original 5-token prompt, and on a
700-token prompt past the 512-token sliding window. A leaking cache would have shown in all three.

## Resolution & Lesson Learned

The harness now asks for greedy (`temperature = 0`, `sample_row`'s argmax shortcut), which is also
what the onnxruntime side decodes with. Its abort message now says "under greedy decoding". The hub
item was removed.

**Takeaway: before calling run-to-run differences a state bug, check whether the model is declared
deterministic, and rerun with the randomness pinned.** A greedy run and a reseeded run cost a minute
and separate "the sampler drew differently" from "the state is wrong". The same goes for any harness
that asserts reproducibility: it has to request the deterministic mode, not inherit it.
