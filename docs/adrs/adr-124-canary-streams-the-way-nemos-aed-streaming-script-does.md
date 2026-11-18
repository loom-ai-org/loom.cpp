---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, canary, aed, alignatt, wait-k, nemo]
supersedes: []
---

# ADR-124: Canary Streams the Way NeMo's AED Streaming Script Does, Wait-k and AlignAtt

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for canary-1b-v2. Its buffer is the one
[ADR-110](adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md) ported. The long-form merge
of [ADR-063](adr-063-a-long-clip-is-decoded-the-way-its-reference-decodes-it.md) stays the one pass's.

## Context

NeMo streams Canary with `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py`
(NVIDIA-NeMo/Speech main) over `GreedyBatchedStreamingAEDComputer`. Every step:

* the whole encoder runs over a buffer of up to `left + chunk + right` seconds (10 + 2 + 2,
  `StreamingBatchedAudioBuffer`);
* the decoder's cross-attention is masked to the left context and the chunk, plus the right context on the
  last chunk only;
* the decoder's cache carries from chunk to chunk.

Two policies decide what is kept:

* **Wait-k**, the default: nothing until two chunks are in, then one token per chunk; the last chunk decodes to
  the end.
* **AlignAtt**: per chunk, tokens while the head-mean of the decoder's layer -2 cross-attention peaks at least
  8 frames before the end of what the decoder may see; at most `10 * int(chunk + right)` tokens per chunk.

Both drop a token that repeats a pattern five, three or three times (NeMo's hallucination check). The user asked
for both policies.

Several things the code does differ from what the script appears to say. Each changes the transcript:

* **The prompt has 13 ids, not `transcribe()`'s 10.** The script encodes every default dialog turn, `user` and
  `user_partial`. The model then writes `<|emo:undefined|>` as its first token, and on cosyvoice3_ref the whole
  task header. NeMo's streamed text carries it, so ours does too.
* **The buffer keeps two bookkeepings.** The scalar context, which trims the buffer, is told when a chunk is
  the last. The batch context, which sets the encoder's length and the decoder's rows, never is: the script
  passes `is_last_chunk_batch=zeros`.
* **A token the policy rejects is not kept,** and neither is the decoder cache of its forward: NeMo saves the
  cache only after a token is kept. The next chunk re-runs the last kept token at its position against the new
  encoder output.
* **The two policies have different length caps:** wait-k the config's `max_generation_length` (512), AlignAtt
  the state's (256).

## Options

1. **A host loop**: `loom_cli` would not have it, and AlignAtt's attention would still have to leave the graph.
2. **Wait-k only**: no graph change, but the user asked for AlignAtt, the policy the script's docstring
   demonstrates and the one that emits text as it is heard.
3. **Both, in the driver, with the decoder giving AlignAtt its attention.**

## Decision

Option **3**. `infer` and its 10-id prompt are unchanged.

* **The decoder gets a second output**: layer `xatt_scores_layer`'s (-2) cross-attention probabilities,
  averaged over heads, `[1, n_tokens, n_enc]`. Cross-attention never fuses, so the softmax was in the graph
  already. The one-pass loop ignores the output.
* **The decoder sees only what NeMo's mask lets it.** `cross_kv` runs over the encoder's first `k` rows, the
  batch context's left context and chunk (all rows on the last chunk). NeMo's -10000 mask gives the hidden rows
  exactly zero weight, so leaving them out is the same attention.
* **Driver** (`canary_driver/10_stream.lua`): the script's loop at batch 1.
  * Both bookkeepings are kept.
  * The waitk-lagging gate and both policies are transcribed in NeMo's order of operations: rejection before
    end-of-text, hallucination before the length cap. Where wait-k and AlignAtt differ (which cap, whether the
    cap's token is kept), the code follows NeMo.
  * The decoder's KV cache is the stream's (ADR-099). A re-run position overwrites its row.
  * As in ADR-110, a step runs once a sample past its end has arrived, and the last step runs in
    `stream_finish`.
* **The numbers are NeMo's, read at export** (`stream_policy()`): the defaults of `AEDStreamingDecodingConfig`
  and `AEDStreamingState`. A NeMo whose AlignAtt default smooths the attention is refused. The streaming prompt
  is read off NeMo's default dialog, and its language slots are checked to be `infer`'s.
* `stream_open` takes `left_seconds`, `chunk_seconds`, `right_seconds`, `policy` (0 wait-k, 1 AlignAtt),
  `language`, `target_language` and `task`, with `infer`'s precedence for the last three. It refuses no right
  context, a chunk under one frame, a buffer over the encoder's 40 s, and any other policy.

## Consequences

* **The stream is NeMo's stream.** `test_e2e_canary_stream` covers 23 cases, 905/905 checks:
  * jfk, cosyvoice3_ref, voxtral_tts_ref and 60 s of joined clips, under both policies at 10/2/2 and at the
    docstring's 10/1/0.5;
  * jfk translated into German;
  * AlignAtt at 10/1.9/0.08 on jfk and long60. Its per-chunk cap is `10 * int(1.98)` = 10 tokens, and long60
    hits it at seven steps;
  * one jfk word repeated for 8.4 s under both policies and sizes, where NeMo's hallucination check stops the
    transcript after `S S S S` or `ask, ask, ask`.
  * The encoder rows the decoder sees are within 1.3e-05 of NeMo's.
  * The first decode's logits and AlignAtt's attention are within 6.4e-07 relative to their absmax, and its
    token is NeMo's.
  * Pushed one encoder frame at a time, the kept context after **every** step is as long as NeMo's, so each
    policy decision lands where NeMo's did.
  * Every case ends on NeMo's ids at 80 ms pieces and the whole clip, and on the first case at 10 ms, one chunk
    and 0.5 s too.
* **Six sabotages, all caught**, each by a same-length patch of the driver in a copy of the file:
  * the AlignAtt threshold at 1 frame for 8: the ids were still NeMo's on every case tried, and only the
    per-step context check caught it (3 of 11 steps);
  * no sink-frame exclusion;
  * wait-k lagging 1 for 2;
  * the decoder shown the right context;
  * no per-chunk cap;
  * no hallucination check.

  The last two passed until the cap and repetition cases were added: NeMo had triggered neither on the first
  17 cases.
* **NeMo's own quality is kept, quirks included.** Wait-k at the default writes jfk correctly but garbles the
  60 s clip ("So, my fellow citizens, I'm Kaylee"; 58 ids where AlignAtt writes 214), and the stream writes the
  same. AlignAtt at 10/1/0.5 emits from 6.1 s, a phrase per chunk.
* **Speed** (dev box CPU, F32): about 0.3 to 0.55x real time on jfk. Every step re-runs the 14 s encoder for
  1 to 2 s of audio. A GPU run has not been done.
* Publishing needs a re-export on an engine with the door, 1.0.0 or later. The published file gives the same
  text in one call, and the live door refuses it.

## Related

* `loom-exporter/loom_exporter/canary_export.py` (`stream_policy`, `_CanaryDecoderWrapper`, the streaming
  prompt), `canary_driver/10_stream.lua`, `tests/ci/test_canary_stream.py`, `fixture_gen/reference_stream_canary.py`
* `tests/gate/test_e2e_canary_stream.cpp`
