---
type: adr
status: accepted
date: 2026-10-02
tags: [asr, kyutai-stt, reference, oracle, exporter, mimi]
supersedes: []
---

# ADR-068: Kyutai STT Follows Kyutai's `moshi`, Not the Transformers Port

## Context

Kyutai STT 1B en-fr ships twice: the moshi-format release (`kyutai/stt-1b-en_fr`) and a transformers
conversion (`kyutai/stt-1b-en_fr-trfs`, `KyutaiSpeechToTextForConditionalGeneration`). The zoo's habit is
the transformers layout as the reference (Qwen3-ASR's `-hf`). Same weights -- the embeddings are
byte-identical. Scoping found the two RUNS differ, in two places that are the port's:

1. **The first frame is encoded twice.** `prepare_inputs_for_generation` starts its window at `[0, 0]`
   and advances `start = end` before `end += 1`, so frames 0, 0, 1, 2, ... reach the Mimi encoder: its
   streaming state sees frame 0 twice and every later frame reaches the LM one step late. Proved by
   replay: encoding frame `max(k - 1, 0)` at call k reproduces the port's codes 155/155.
2. **The LM window is 375.** The conversion script hard-codes `max_position_embeddings=375` -- the
   2.6B model's `context`. This checkpoint's `config.json` says `context: 750`.

Kyutai's `run_inference` instead encodes each frame once and steps the FIRST frame's codes twice
(step 0 consumes the initial tokens whatever it is handed), with a 750-position ring. Both transcribe
jfk.wav correctly; they are different computations.

## Options

1. **The transformers port exactly**, its two divergences included and documented.
2. **The port, with the window fixed** -- a config-only deviation, keeping its frame handling.
3. **Kyutai's `moshi`**: the upstream implementation the checkpoint was trained and released with.

## Decision

Option 3 (the user, 2026-10-02). The export reads the moshi-format directory through moshi's own loader
(a checkout of `kyutai-labs/moshi`), and the oracle is `run_inference`'s STT path
(`scripts/kyutai_stt_reference.py`). Two details had to be MEASURED to reproduce it, because moshi's
streaming is what defines them:

* **The encoder transformer's window is 249 or 250.** Mimi streams two positions per frame into a ring of
  250, writing both before either attends, so the first of each pair has lost its oldest key. A one-call
  encode with that parity mask reproduces moshi's streamed codes 156/156, where a plain 250-window gets
  144 and a causal mask 127.
* **The downsample pads by replication** of the stream's first column (`pad_mode: replicate`), the rest
  of SEANet by zeros.

## Consequences

* loom matches moshi, not transformers. A user comparing against `generate` on the `-trfs` checkpoint
  sees a different (equally sensible) transcript at times.
* Needed an engine feature for the 750-ring: [ADR-066](adr-066-a-uniform-sliding-window-is-a-ring-kv-cache.md).
* The export needs a moshi checkout beside the exporter (as Pocket-TTS needs its own).

## Related

* `loom-exporter/loom_exporter/kyutai_stt_export.py`: the account in full
* [Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md): reproduce the reference --
  which here meant first deciding which one it is
