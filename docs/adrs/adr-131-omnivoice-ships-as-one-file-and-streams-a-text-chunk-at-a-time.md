---
type: adr
status: accepted
date: 2026-11-17
tags: [model-coverage, exporter, streaming, tts, masked-diffusion, omnivoice, codec, voice-cloning]
supersedes: []
---

# ADR-131: OmniVoice Ships as One File and Streams a Text Chunk at a Time

Serves [Epic-03](../epics/epic-03-model-coverage.md) and [Epic-06](../epics/epic-06-high-level-api-and-hosts.md)
(streaming). Departs from [ADR-022](adr-022-dia-and-its-codec-stay-two-files.md) for one model and follows
[ADR-038](adr-038-the-codecs-encoder-ships-inside-the-talker.md). The NaN that the first export produced is
[Retro-107](../retros/retro-107-log-softmax-lowered-as-log-of-softmax-made-a-nan-score.md).

## Context

OmniVoice (k2-fsa, 600+ languages) is a masked-diffusion TTS. A bidirectional Qwen3-0.6B fills eight
codebooks of HiggsAudio-v2 codec tokens over 32 unmasking steps. Each step runs the whole prompt
(conditional) and the target alone (unconditional), applies classifier-free guidance to the
log-probabilities, and unmasks the k most confident cells. The cells are scored by confidence minus a
per-codebook penalty, plus Gumbel noise. HiggsAudio-v2's decoder then makes 24 kHz audio.

Upstream defines its output at the waveform:

* A text whose estimated audio exceeds 30 s is generated in 15 s chunks. Each chunk is decoded on its own,
  and chunks are joined with a 100 ms fade-out, 100 ms of silence and a 100 ms fade-in.
* The joined audio then goes through pydub: an int16 round trip, silences over 500 ms shortened, the edges
  trimmed, the level set (peak 0.5 without a reference), and 100 ms fades and padding added.

No step emits audio before the last one, so the model cannot stream inside a chunk. No other model here
uses its codec. Cloning needs the codec's encoder, which is a HuBERT semantic branch plus a DAC acoustic
encoder.

## Options

1. **A talker and a codec, two files (ADR-022).** The pair would return codes. The chunked decode, the
   cross-fades and pydub's post-processing would have no home except the host.
2. **One file: the LM step, the codec's decoder and its encoder.** The driver ports the reference's text
   handling, chunking and post-processing.
3. **No stream door**: the model is not amenable within a chunk.

## Decision

Option **2**, with a stream door cut per chunk (the user, 2026-11-17). The user also chose the encoder
inside the file, and a stream that is approximate only in its post-processing.

* **Phases.**
  * `lm_step`: one unmasking step, with both guidance halves in ONE `L + T` sequence. A block-diagonal
    mask is built in-graph from a segment vector, and the unconditional positions restart at 0. The head
    runs on the `2T` target rows, gathered by index. Guidance, log-softmax, the banned mask id,
    argmax/confidence, the layer penalty and the Gumbel noise are all in the graph, so the driver reads
    `8 x T` ids and `8 x T` scores.
  * `codec_decode`: codes to the waveform.
  * `ref_encode`: a 24 kHz clip to codes. The resample is torchaudio's own kernel as a strided conv, HuBERT
    is traced eager, and the RVQ is an in-graph argmax chain.
* **Two exporter lowerings.**
  * `ExportPhase.dense_attention`: uncached, unwindowed fused ATTENTION over an in-graph mask. Unfused, the
    stack would carry `repeat_kv`'s reshape.
  * `reduce_max` over a static last axis: POOL_1D with k = s = the axis width.
  * The engine's `op_pool_1d` now makes a non-contiguous input contiguous: ggml's CPU pool ignores `nb`.
* **The driver ports the reference, and holds itself to it.** It ports `_combine_text`, the
  non-verbal-tag tokenization, `RuleDurationEstimator`, `chunk_text_punctuation`, `_resolve_instruct`, the
  float32 time schedule, top-k and pydub's silence code. The tables that need data ship as numbers: the
  unicodedata categories as 1375 weight runs, strings as UTF-8 byte arrays. Every draw is made at
  `stream_open`, in the reference's order, and can be pinned (`draws`).
* **The stream** cuts the one-pass loop per chunk, so its codes are the one pass's. Each piece gets its
  own silence trim and level; `infer` post-processes the joined audio once. With one chunk the two are
  the same samples. `chunk_threshold`/`chunk_duration` let a host cut a long text into more, earlier
  pieces.
* **Door inputs**: `tts.reference = "transcript"` (the clip, its transcript and the text as three inputs),
  `tts.instruct` (voice design), and the language as an index into the file's declared codes.

## Consequences

* Gate `test_e2e_omnivoice`, against `scripts/omnivoice_reference.py`, with every draw pinned. Three
  cases: own voice, clone, and four chunks with an instruct and a language. Every code is exact, and the
  waveform is within one int16 step of `generate`'s (the engine's f32 decode lands a sample on the
  neighbouring step now and then).
* CPU, 4 threads: the one pass is 94 s for 3.4 s of audio (0.036x real time, RTF 28), against upstream
  PyTorch's 144 s.
* Not covered: `class_temperature > 0` (top-k sampling of the ids, off by default), text normalization
  (opt-in upstream, needs WeTextProcessing), auto-transcription of a reference (upstream loads Whisper),
  and trimming of references over 20 s.
* The weights are CC-BY-NC (training data). The card says so.
