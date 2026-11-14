---
type: adr
status: accepted
date: 2026-11-13
tags: [engine, streaming, audio-in, asr, qwen3-asr, oruk-loop, speech-lm, re-decoding, kv-cache]
supersedes: []
---

# ADR-121: Qwen3-ASR Streams the Way Its Reference Does, Re-encoding Only What Changed

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md), with the hypothesis kept in the state as
[ADR-107](adr-107-a-re-decoding-asr-stream-keeps-its-hypothesis-in-the-state.md) keeps it.

## Context

Qwen3-ASR is a speech LM (family 3): an audio encoder, a projector, and a Qwen3 decoder that reads the audio
rows as a prompt. Its card advertises "streaming / offline unified inference with a single model", and the
`qwen-asr` package (0.0.6) has `streaming_transcribe`, on its vLLM backend only. That loop:

* every `chunk_size_sec` (2 s) of new audio, re-feeds **all** the audio so far to the whole model;
* from chunk `unfixed_chunk_num` (2) on, appends the previous raw answer (`language English<asr_text>...`)
  minus its last `unfixed_token_num` (5) tokens to the prompt, so the reply starts already written and the
  model only re-decides the tail. The cut is a text round trip (re-encode, drop, decode), backing off further
  while the decoded prefix holds U+FFFD;
* `finish_streaming_transcribe` re-feeds the tail the same way, without the back-off, keeping at least one token.

The survey called this row "driver/host only: a prefix input; cost grows with the stream". oruk-loop-153M, a
Qwen3-ASR derivative with its language forced in the prompt, has no streaming mode of its own; its teacher's is
the natural one.

Re-feeding everything is upstream's choice, not the model's need. The encoder runs a conv stem per 1 s chunk
(its position table restarts every chunk) and attends **block-diagonally, in fixed 8 s windows counted from the
clip's start** (`n_window_infer`, 800 mel frames, 104 rows). A window's rows depend on its own samples and on one
clip-wide number, the log-mel floor (`max - 8` over every frame). The decoder is causal, so its keys over the
prompt and a window's rows depend on nothing after them. The first version of this ADR re-fed everything and
claimed the encoder's attention spanned the clip; a question from the user ("is it an attention issue?") sent it
back to the code, and it does not.

## Options

1. **A loom-py loop over `transcribe` with a prefix input.** Only Python hosts get it, and the driver needs a
   re-export for the prefix input anyway.
2. **Upstream's loop in the driver, re-feeding everything.** Exact by construction, and every update re-runs
   the encoder over the whole stream and prefills all its rows again: the cost of update k grows with k.
3. **Upstream's loop in the driver, recomputing only what changed.** The same decodes; a finished window's
   rows and the decoder's keys over them are kept across updates, so an update re-encodes the window still
   filling and prefills its rows.

## Decision

Option **3**. `infer` is unchanged.

* `speech_lm_driver/01_stream.lua`: `stream_feed` collects samples and, for every whole update they complete
  (upstream's `while`), runs upstream's decode of all the audio so far; `stream_finish` decodes the tail.
  State: the whole updates' samples (`audio`, set once per update), the samples since (`pending`), the last
  answer's ids (`raw`), its transcript (`hypothesis`), and the windows whose keys are in the cache (`kept`,
  with each one's raw log-mel max and min and the floor it was encoded at).
* **Two phases on the encoder's weights.** `window_mel_range` returns one window's raw log-mel max and min;
  `encoder_window` encodes one window at a given floor. Each reads the window's samples with 200 either side
  (`center=True`'s reach), which the driver builds as `transcribe` frames the clip: zeros to a whole chunk, the
  frontend's reflection over their head, the STFT's reflection at both ends. The export checks the composition
  (20.5 s of noise with 1 s of silence, windows against the whole clip) and fails above 1e-4 of the largest
  row: 6.2e-07 on Qwen3-ASR, 4.0e-07 on oruk-loop, f32 rounding.
* **When a window is kept.** A window is finished once 200 samples past its end have arrived (its last frame
  reads 40 past it). Its keys are kept while every window before it is kept and the floor has not moved past
  its lowest frame. The floor can FALL as well as rise: the max counts the clip end's padding and reflection
  frames, which real audio then replaces. So a kept window is re-encoded when the floor differs from the one
  it was encoded at and its lowest frame is below either. The first window to go takes every later one with it,
  since the keys are a prefix of the cache.
* **The stream is bounded by the decoder's cache, not the encoder.** Upstream has no length limit (its
  feature extractor does not truncate). The stream holds the prompt, every row, the prefix and the budget, and
  is refused by name past `n_ctx`: Qwen3-ASR's 4096 positions are about four minutes; oruk-loop's 1024 (sized
  for one 30 s quiet-point piece) about 45 s. "Split the stream at a pause."
* `stream_open` takes upstream's three knobs under the names hosts use: `update_seconds` (2), `unfixed_chunks`
  (2), `unfixed_tokens` (5), and `max_new_tokens` (256, `infer`'s). They are upstream's `init_streaming_state`
  defaults, bound as literals, because the package that declares them is vLLM-only and not an export dependency.
* The stream's decode reads what `infer`'s reads, bound a second time top level (`S_`-prefixed, as Nemotron's
  stream does). The audio row count is the same IR expression `infer` inlines, rendered as a top-level function
  by a new exporter component, `ExprFunction`, rather than restated in the fragment.
* **The prefix is cut from text, as upstream cuts it.** The driver holds the answer as ids and does
  upstream's round trip through the file's own vocabulary: two new bridge bindings, `loom.tokenize(text)` and
  `loom.detokenize(ids)`, over the byte-level BPE of the model whose modules the bridge runs, loaded on first
  use. A tokenizer is per-task engine machinery, and the bindings read no model config. `detokenize` returns
  bytes, so the driver backs off while the text is incomplete UTF-8 or holds U+FFFD, which is upstream's
  condition on its replaced text.
* **Cutting ids was tried first and failed.** Dropping the last five of the model's own ids agreed with
  upstream on all 13 Qwen3-ASR clips, and broke on oruk-loop: it wrote " Voxtrol" as three ids where its
  tokenizer has two, so the prompt differed and the next word did ("Mistral" against upstream's "Mistrel").
  Any name a model spells non-canonically can do this.
* **Upstream's `finish` cuts characters; this one does not.** Its per-chunk path backs off on U+FFFD; its
  finish keeps `max(1, n - 5)` with no back-off. On a Korean clip the cut lands inside 편, the prompt string
  carries a literal U+FFFD, and upstream's transcript ends "훨씬 �연할 거야". The driver's finish backs off
  too and the transcript reads "훨씬 편할 거야". This is the one intended departure from upstream, and the
  oracle makes the same fix.
* oruk-loop inherits the fragment. Its prompt already holds `<asr_text>`, so the whole answer is the transcript,
  `infer`'s rule there. Its generated driver had `local TRANSCRIPT_AFTER = None` (a Python `None` rendered into
  Lua, nil by luck); `None` constants are no longer bound. Its file keeps 36,974 of the vocabulary's ids, so
  its reference is the teacher's loop through that kept tokenizer, for the cut AND the prompt: the full
  tokenizer spells " Qtai" with " Qt", an id the export dropped, and a prompt the file cannot hold read
  "catatized" where the file's own ids read "cutatai's".

## Consequences

* **The stream is upstream's stream.** The oracle (`scripts/qwen3_asr_stream_reference.py`) is upstream's loop
  line for line with transformers' greedy `generate` (f32, eager) in vLLM's place, re-feeding everything, finish
  fixed. It also runs the driver's own path (the round trip from ids) and refuses a clip where the two texts
  differ at any update. The gate (`test_e2e_qwen3_asr_stream`) holds 13 clips in six languages (English,
  Japanese, Korean, Cantonese, Mandarin) from 2 s to 60 s, and the same 0.5 s cadence twice. Every update's
  transcript is upstream's in 100 ms pushes; the end is the same at 10 ms, cadence-sized and whole-clip pushes;
  a cadence longer than the clip decodes once with no prefix and is `transcribe`'s text; two interleaved streams
  with a one-pass call between them end where they end alone; a stream past the cache is refused by name, and
  only on a clip past 30 s; the published file is refused by the live door.
  Qwen3-ASR: 684/684, every update of all 15 cases upstream's, the 60 s clip whole (31 updates, seven windows).
  oruk-loop-153M: 395/395 against its teacher's loop on 8 cases; its 60 s clip agrees for 23 updates and is
  refused at 46 s, where its 1024 positions run out.
* **Transcripts could not see a stale window; the cached keys can.** Each case was chosen by sabotaging the
  driver (a same-length patch to the file's script). Dropping 3 ids instead of 5 is caught only by the Mandarin
  clip at 0.5 s, never prefixing only by the Japanese one, a floor that ignores the kept windows only by the
  60 s clip. Two window sabotages passed every transcript check: keeping window 0 before its last frame's
  samples arrived (that frame read the reflection), and never re-encoding a kept window when the floor moved.
  Both change rows by too little to flip a greedy token. So the gate also compares, per clip, the decoder's K
  and V over the prompt and every audio row after `finish` with what a one-pass `transcribe` writes. Intact the
  worst is 8.3e-06 (60 s, seven windows; one window is exactly 0); the sabotages read 2.9e-04 and 1.8e-03. The
  floor rule fires on the 60 s clip alone, the one whose floor moves under a kept window.
* **It is not the one pass, and the card's mode costs accuracy.** On the Japanese clip the stream ends on
  弁当生で持って行けない where one pass reads 弁当制で持ってきてない; on the 13 s Mandarin clip it drops the last clause.
  Those are upstream's results too. The cadence changes the transcript.
* **The encoding stops growing with the stream; the prefix does not.** An update re-encodes the window still
  filling (two when one has just finished) and prefills its rows, then prefills the prefix: the transcript so
  far, which sits after rows that changed, so its keys cannot be kept, upstream's cost too. On the 60 s clip
  the floor moved enough to send 5 windows back (19 full-window encodes where finishing windows take 14), and
  the last prefixes are ~190 tokens. Dev-box CPU (2 cores), F32, 100 ms pushes, `loom_cli --stream 100`:

  | Qwen3-ASR | 11 s (jfk) | 30 s | 60 s |
  |---|---|---|---|
  | this stream | 0.45x real time (RTF 2.21) | 0.37x (RTF 2.67) | 0.26x (RTF 3.82) |
  | re-feeding everything (option 2) | 0.42x (RTF 2.36) | 0.22x (RTF 4.48) | refused past 30 s |
  | one pass, wall with load | 11.2 s | 31.1 s | 67.8 s |

  oruk-loop-153M streams at 0.20x (RTF 4.99) on jfk and 0.16x (RTF 6.38) on 30 s; its one pass takes 20.7 s on
  jfk, slower than Qwen3-ASR's on this box. Live needs a faster machine than this one for either.
* **The one-pass gate had gone stale.** `test_e2e_qwen3_asr_mil_export` compared the whole answer to fixtures
  holding only the transcript since 6cc57ea, and passed 52 of 80 ids. The generator now writes the transcript
  ids (`ref_asr_transcript.npy`) and the gate compares those: 81/81 on the new and the published files.
* Re-export required (rc21+); the published file keeps its one pass and has no live path.
