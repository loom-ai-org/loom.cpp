---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, ctc, hubert, data2vec, wav2vec2, transformers]
supersedes: []
---

# ADR-126: A wav2vec 2.0-Style CTC Model Streams in the Transformers Pipeline's Strided Chunks

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for hubert-large-ls960-ft and
data2vec-audio-base-960h, the first two of the survey's "re-decode" rows. The NeMo CTC models' row is
[ADR-123](adr-123-a-nemo-ctc-model-streams-the-way-framebatchasr-does.md).

## Context

Neither checkpoint has a streaming mode, and neither does `transformers`. What `transformers` does have is a
documented way to run exactly these models over long audio: the ASR pipeline's chunking
(`pipeline("automatic-speech-recognition", chunk_length_s=C, stride_length_s=S)`):

1. `chunk_iter` cuts the raw waveform into chunks of C seconds, each starting `C - 2S` after the last;
2. each chunk is the model's whole input, with the feature extractor's normalisation over the chunk alone;
3. a chunk's first S seconds (none on the first chunk) and last S (none on the last) are strides:
   `rescale_stride` turns them into logit frames, and the frames between are kept;
4. one greedy CTC merge runs over all the kept argmax ids.

Every size is rounded to whole logit frames (`inputs_to_logits_ratio`, 320 samples here) with Python's
`round`, which rounds halves to even.

A chunk's ids depend on nothing after the chunk, except whether it is the last one, which only drops its
right stride. So a stream can answer a chunk once one sample past its end has arrived.

## Options

1. **Re-decode a growing window and commit a stable prefix** (whisper_streaming's local agreement). It has
   no reference, and a CTC model gains nothing from it: its frames are already local.
2. **A host loop in loom-py.** `loom_cli` would not have it, and every other row lives in the driver.
3. **The pipeline's chunking in the driver**, over the unchanged one-pass graph.

## Decision

Option **3**, chosen by the user over a VAD-cut or local-agreement scheme for these two models.

* **The export is unchanged except for the driver.** The graph, the weights and `infer` are byte-for-byte
  the same. `CtcGreedyBuilder` gained ADR-120's optional `stream` field, and `ASRCtcExportConfig` fills it
  with `ctc_chunk_stream.stream_components`: the top-level constants (`CTC_RATE`, `CTC_ALIGN` = the
  checkpoint's `inputs_to_logits_ratio`, `CTC_BLANK`, the graph's sample range) and
  `ctc_chunk_stream_driver/00_stream.lua`.
* **The graph already does the feature extractor's per-chunk normalisation**, because it normalises over
  whatever it is handed (family 4's wrapper). A chunk is the graph over the chunk's samples.
* **The driver is the pipeline's arithmetic**, in its order of double operations, with a `ctc_py_round`
  that rounds halves to even:
  * `chunk_len = round(C·rate/align)·align`, and the same for each stride;
  * `token_n = round(n·(1/align))`, then `round(stride/n·token_n)` for each stride's frames;
  * the slice `[left, token_n - right)`, clamped to the frames the graph produced.
  * The merge carries the previous id across chunks.
* **A chunk runs when a sample past its end arrives**, since only then is it known not to be the last.
  `stream_finish` runs the rest, the last chunk without a right stride. A chunk's ids are final, so the
  transcript only grows.
* `stream_open` takes `chunk_seconds` (default 4), `stride_seconds` (default a sixth of the chunk, the
  pipeline's rule), or `left_stride_seconds` / `right_stride_seconds`. It refuses:
  * a chunk no longer than its two strides (the pipeline refuses shorter, and a step of 0 never ends);
  * a chunk longer than the graph takes (30 s);
  * a left stride under the graph's 0.1 s minimum. Only the last chunk can be short, and it is always
    longer than the left stride (`chunk_iter` drops it otherwise), so this keeps every chunk in range.
* The default is 4 s, a choice of ours, since the pipeline has none (0 turns chunking off). That gives a
  stride of 0.66 s each side and a chunk every 2.68 s.

## Consequences

* **The stream is the pipeline.** The oracle is the pipeline itself, hooked rather than re-derived:
  `fixture_gen/reference_stream_ctc_chunked.py` records each chunk's logits from the model's forward and the
  concatenated ids from the tokenizer's `decode`, and checks that the recorded chunks rebuild those ids.
  `test_e2e_ctc_chunked_stream` covers jfk, f5_tts_ref, pocket_tts_ref and a 60 s joined clip, each at
  4/0.66, 10/1.67, 2/0.5 and 6/1:
  * **data2vec-audio-base-960h: 1414/1414.** Kept logit rows are within 5.7e-05 of the pipeline's
    (relative to their absmax).
  * **hubert-large-ls960-ft: 1414/1414**, the kept rows within 1.3e-04.
  * On the RTX 5090 (`LOOM_TEST_DEVICE=gpu`), both pass 1410/1410 (everything but the old-file section),
    rows within 4.3e-05 and 2.0e-04.
  * Every case ends on the pipeline's ids at 100 ms pieces and the whole clip; the first case also at
    10 ms, one chunk and 0.5 s.
  * After each chunk, the stream's next chunk start and its ids so far are the pipeline's.
  * Two interleaved streams and a one-pass call between them do not disturb each other, and the published
    file's one pass is unchanged while its live door is refused.
* **Chunked ids are not the one pass's, and the stream keeps that difference.** pocket_tts_ref at 2 s gives
  98 ids against the one pass's 95 ("ELLOW WORLD" against "HELLO WORLD" for HuBERT): the first chunk
  starts at the clip's start with no left context. At 10 s every short clip is one chunk, which is the one
  pass.
* **Sabotage** on data2vec: same-length byte patches of the driver in a GGUF copy, run on the four jfk cases
  and pocket at 2 s. Four of five are caught:
  * the right stride dropped: 265/304 (the default-sizes check, ids);
  * the left stride dropped: 271/304;
  * the merge's previous id reset per chunk: 300/304 (the ids);
  * a chunk run when its last sample, not one past it, has arrived: 305/307 (the per-chunk count).

  **Rounding halves up instead of to even passes (304/304).** No case lands on a tied stride. A tied
  `token_n` occurs only on the last chunk, where the slice is clamped to the frames the graph produced, so
  it cannot move a kept frame. That rule is covered by the CI test below, which mirrors the driver against
  transformers' own `chunk_iter` and `rescale_stride`.
* **Speed** (F32, 100 ms pushes, the 16 cases' 325 s):
  * dev box CPU: data2vec 2.17x real time (RTF 0.460); HuBERT-large 0.73x (RTF 1.369), measured while an
    oracle shared the CPU;
  * RTX 5090: data2vec 134x (RTF 0.007), HuBERT-large 113x (RTF 0.009).

  Every sample inside a stride is encoded twice, so a 2 s chunk with 0.5 s strides costs 2x its audio.
* `tests/ci/test_ctc_chunk_stream.py` checks the driver's cut, mirrored in Python, against `chunk_iter` and
  `rescale_stride` from transformers. It runs six size sets at alignments of 320 and 160, over clip lengths
  that end on, just before and just after a chunk's end, at three push patterns.
* The gated exports predate one comment fix in the fragment (a stale constant name); the driver's code is
  byte-for-byte the committed one.
* The exporter's gate snapshots of these two files move: the driver script only. Publishing needs a
  re-export on an engine with the door, 1.0.0 or later.

## Related

* `loom-exporter/loom_exporter/ctc_chunk_stream.py`, `ctc_chunk_stream_driver/00_stream.lua`,
  `ctc_asr_export.py`, `tests/ci/test_ctc_chunk_stream.py`, `fixture_gen/reference_stream_ctc_chunked.py`
* `tests/gate/test_e2e_ctc_chunked_stream.cpp`
