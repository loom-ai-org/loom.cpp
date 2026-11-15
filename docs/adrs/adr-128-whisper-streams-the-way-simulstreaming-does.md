---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, whisper, alignatt, simulstreaming]
supersedes: []
---

# ADR-128: Whisper Streams the Way SimulStreaming Does

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for whisper-small, the last of the survey's
"re-decode" rows. The others are HuBERT and data2vec
([ADR-126](adr-126-a-wav2vec2-style-ctc-model-streams-in-the-transformers-pipelines-strided-chunks.md)) and
the VAD-lines models ([ADR-127](adr-127-a-recogniser-without-a-streaming-mode-streams-in-moonshines-vad-lines.md)).

## Context

Neither openai/whisper nor transformers streams Whisper. ufal has two references:

* **whisper_streaming**: LocalAgreement-2 over a re-transcribed buffer, which needs word timestamps.
* **SimulStreaming**: its successor (whisper_streaming's README says it is being replaced), using the
  AlignAtt policy:
  * every `min_chunk_size` (1.2 s) of audio is a segment of a 30 s buffer;
  * each update re-reads the tokens committed so far and decodes greedily;
  * it stops at a token whose most-attended frame is within 25 frames (0.5 s) of the audio's end.
    That frame comes from the alignment heads' cross-attention, z-scored over the tokens,
    median-filtered and averaged.
  * Segments leaving the 30 s buffer move their tokens into a text context, which is read back
    after `<|startofprev|>`.

The user chose SimulStreaming. Three of its behaviours had to be read off the code rather than its docs:

* **With no CIF checkpoint, `load_cif` sets `always_fire`**, so an update's last word is never held back,
  although the help text says it is always trimmed.
* **On a rewind, the forced tokens are rebuilt without the context** (`torch.cat(self.tokens)`) and then
  sliced at the length that included it. The port keeps this.
* **The decoder's cache is cleared after every update**, so each update re-reads everything it forces.

## Options

1. **whisper_streaming**: needs word timestamps (DTW over the same heads) and is superseded upstream.
2. **A host loop in loom-py**: `loom_cli` would not have it.
3. **SimulStreaming in the driver**, with the alignment heads' attention as a decoder output.

## Decision

Option **3**.

* **The decoder's second output** is the alignment heads' cross-attention, `[1, 10, n_tokens, 1500]` for
  whisper-small, from `generation_config.alignment_heads` (OpenAI's table).
  * Each head is `softmax(q k^T / sqrt(head_dim))`, recomputed from that layer's query, which a forward
    hook keeps, and `xk_layer`.
  * Asking HF for `output_attentions` would need the eager attention path, which would change how every
    self-attention block traces. The hook leaves module and weight names untouched.
  * The one pass ignores the output, as Canary's does (ADR-124).
* **The driver** (`whisper_driver/10_stream.lua`) is `insert_audio` + `infer` + `process_iter` in
  SimulStreaming's order:
  * the suppressed set, with a bare space and the end of text also banned on the first step, through
    `loom.sample_row`'s `banned`;
  * the rewind rule, and the 4-frame threshold on the last update;
  * `trim_context` with `split_tokens_on_spaces`;
  * the context's text kept as its token ids, round-tripped through `loom.tokenize` / `loom.detokenize`.
* **The z-score uses running sums in double**, per head and frame, over every token row of the update.
  This replaces torch's f32 `std_mean`; the gate allows `kFrameSlack` = 2 frames.
* `stream_open` takes `chunk_seconds` (1.2), `frame_threshold` (25), `rewind_threshold` (200) and
  `language` (a token id, default English as SimulStreaming's CLI). It refuses a chunk of 0, or of 30 s or more. An English-only checkpoint
  takes no language.
* **The simulation's cadence is integer samples.** SimulStreaming's `--comp_unaware` loop slices with
  `int(end * 16000)` over a float `end += min_chunk`, which drifts by a sample now and then. The oracle
  cuts exact `int(chunk * 16000)` pieces, and so does the driver.

## Consequences

* **The stream is SimulStreaming.** Every update is checked by `test_e2e_whisper_simul_stream` against the
  reference at pushes of one chunk:
  * the tokens it started from (context + start of transcript + committed blocks);
  * the most-attended frame after every token decoded.

  The ids are also checked at 100 ms pushes and the whole clip (and 10 ms and 0.5 s on the first case),
  and the text is SimulStreaming's. Clips: jfk, f5_tts_ref, pocket_tts_ref, the 60 s joined clip and the
  48 s pause-free one (both past the 30 s buffer, so its trimming and the text context are exercised),
  at chunks of 0.5, 1.2 and 2 s.
  * **RTX 5090: 1921/1921**, including the published file's one pass unchanged and its door refused.
    Every update's starting tokens are SimulStreaming's, and every frame is identical but one, off by
    one in 257.
  * Dev-box CPU: jfk 68/68, every frame identical.
* **The rewind never fires at SimulStreaming's 200 frames**, on any clip, a repetitive one included
  (jfk's "ask" twelve times). So `stream_open` also takes `rewind_threshold`, as SimulStreaming's config
  does, and a second reference set runs at 20, where it fires on the repetitive clip: 495/495 and 137/137.
* **Sabotage**: same-length patches of the driver in a GGUF copy, on the RTX 5090, eight of eight caught:
  * the last update's threshold 9 for 4;
  * the median one rank low;
  * no first-step ban;
  * a trimmed block not moved into the context;
  * the z-score without its mean.
  * Three more caught only by the threshold-20 set: the rewind off, the trim's frame shift added
    instead of subtracted, and the special-previous-token rule flipped.
* **Speed** (100 ms pushes): 12.8x real time on the RTX 5090 over the 15 cases (RTF 0.078); 7-10x at
  0.5 s chunks, 19-27x at 2 s. On the dev-box CPU 0.13x real time (RTF 7.7): `loom_cli --stream 100` on jfk,
  alone, at the engine's 2 threads, 85.6 s for 11 updates, each about one one-pass transcription (8.5 s with
  the model load). The gate first measured 0.07x, at the test helper's old 4 threads and maybe beside
  other runs. Every update encodes the whole 30 s window, as
  SimulStreaming does, and re-reads everything it forces.
* `tests/ci/test_whisper_simul_stream.py` checks three things:
  * the second output equals HF's eager `cross_attentions` for the heads, with the logits unchanged;
  * the driver's frame arithmetic, mirrored in Python, equals SimulStreaming's `std_mean` +
    `median_filter` on 20 random cases;
  * every constant the fragment reads is bound.
* The exporter's gate snapshot of whisper-small moves: a second decoder output and the stream constants.
  Publishing needs a re-export on an engine with the door, rc21 or later.

## Related

* `loom-exporter/loom_exporter/whisper_export.py` (`_WhisperDecoderWrapper`, `stream_constants`),
  `whisper_driver/10_stream.lua`, `tests/ci/test_whisper_simul_stream.py`
* `scripts/whisper_simulstreaming_reference.py`, `tests/gate/test_e2e_whisper_simul_stream.cpp`
