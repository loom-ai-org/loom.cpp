---
type: adr
status: accepted
date: 2026-10-01
tags: [asr, long-form, transcribe, contract, engine, canary]
supersedes: []
---

# ADR-063: A Long Clip Is Decoded the Way Its Reference Decodes It

## Context

Canary-1b-v2 ([Epic-03](../epics/epic-03-model-coverage.md)) takes audio of any length -- its export
is dynamic-length, like every NeMo ASR file -- but was trained on clips of at most 40 s
(`max_duration: 40.0`). `transcribe` handed a dynamic-length file the whole waveform in one call, and
past the ceiling one decode degrades fast. On LibriSpeech (one speaker, consecutive utterances), rc13:

| clip | one pass: WER, words returned | NeMo `transcribe()` |
|---|---|---|
| 79 s | 0.43, 116 of 181 | 0.055 |
| 161 s | 0.71, 109 of 361 | 0.091 |
| 304 s | 1.15, 939 of 703 -- looping on "of the world" until the token budget ran out | 0.036 |

NeMo's own `transcribe()` never does a single pass there: for one file it cuts overlapping windows and
stitches their token sequences (`PromptedAudioToTextLhotseDataset._chunk_waveform`,
`merge_parallel_chunks` over `lcs_alignment_merge_buffer`).

## Options

1. **Whisper's loop**: fixed windows, seek on the timestamps the model emits. Canary emits none (NeMo's
   timestamps come from a second, CTC model in the `.nemo`, not exported), so there is nothing to seek
   on.
2. **Something simple of our own** -- fixed 30 s windows, overlap dropped, text concatenated. Cheap, but
   a long file would transcribe differently from the reference, and every difference at a seam would
   need its own argument.
3. **A host-side loop** in loom-py. `loom_cli` would not get it, and the engine already owns
   transcription for exactly this reason ([transcribe.h](../../include/loom/core/transcribe.h)).
4. **Port NeMo's scheme into the engine, decision for decision, with its numbers declared by the
   file.**

## Decision

Option 4. A file declares its ceiling and its stitch as `loom.asr.window_max_samples`,
`window_min_samples`, `window_search_step_samples`, `window_overlap_samples`, `merge_search_tokens` and
`merge_head_tokens`; `transcribe`'s dynamic-length branch reads them (`AsrDecodeTable::long_form`) and,
for audio longer than the ceiling, decodes each window alone with the caller's arguments -- at its
planned length, real sample count in `length`, as NeMo batches them -- and merges the windows' tokens
(`asr_long_form.h`). Undeclared, nothing changes.

The ALGORITHM is per task and lives in the engine; the NUMBERS are per model and come from the export.
Canary's are read off NeMo's own `_find_optimal_chunk_size` signature (30-40 s, 1 s overlap) and
`merge_parallel_chunks` (search 24 tokens, align 7), not restated.

**Ported including the parts that read like accidents**: NeMo's "LCS" is a longest common substring
with two repair heuristics; the last longest run wins a tie, the first window size wins one; the
complete-merge branch drops its backtracked length; a one-token overlap is not trusted, so the seam
keeps a duplicate. Reproducing NeMo's transcript is the point, and each of those changes it.

## Consequences

* loom's long-form Canary transcript IS NeMo's: on all three clips every window's token ids and the
  stitched text came out identical (16 of 16 windows; WER 0.055, 0.091, 0.036, as NeMo's). The 304 s
  clip took 236 s against the one pass's 876 s, most of which had been the loop.
* `Transcription::windows` reports the window count; `segments` stays one span over the clip (no
  timestamps).
* Needs an engine with this (1.0.0-rc14); an older engine ignores the keys and decodes one pass, so a
  re-exported file is harmless on it, only no better.
* Pinned by `tests/ci/test_asr_long_form.cpp`: NeMo-generated plans, merges and window sequences, plus
  a fixture whose driver reads token ids off the audio. Six mutations of the port were run against it;
  five fail it, and the sixth (control ids kept through the merge) differs only where an end-of-sequence
  token takes one of the 24 searched slots.
* A future model trained with another ceiling declares its own numbers; one that stitches differently
  needs a new merge, chosen by a key it declares.
