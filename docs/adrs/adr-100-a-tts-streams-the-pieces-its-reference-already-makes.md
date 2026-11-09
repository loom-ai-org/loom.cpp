---
type: adr
status: accepted
date: 2026-11-09
tags: [streaming, live-api, stream-state, tts, audio-out, pocket-tts, soprano]
supersedes: []
---

# ADR-100: A TTS Streams the Pieces Its Reference Already Makes

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The third live door, after
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) (audio in) and
[ADR-099](adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md) (tokens out), on the same `StreamState`.

## Context

Every TTS door took the whole text and returned the whole waveform from one `infer`. The survey in
Epic-06 splits audio out three ways: AR models whose reference already decodes sentence chunks from a fresh
state (Pocket-TTS, Soprano), AR models that could hand back audio per frame only with a decoder that
carries its state (Pocket-TTS's Mimi, VoxCPM2, MOSS) or a driver change (Voxtral, Qwen3-TTS), and twelve
non-AR models that make an utterance in one shot. The user chose sentence level first (2026-11-09).

## Options

1. **Per frame first.** Re-export Pocket-TTS's Mimi decoder as a step carrying its conv state and
   attention cache, so audio starts after the first 80 ms frame. The best latency, and the largest job:
   a stateful decoder export per family.
2. **Per sentence, cut by the driver**, where the reference itself decodes each chunk afresh. Exact by
   construction, and only a driver change: the chunk loop the driver already runs is cut at each chunk.
3. **Per sentence, cut by the host**, for every TTS: split the text, `infer` each sentence. Needs nothing
   from any file, and is not the one-pass audio at the joins.

## Decision

**2 where the reference makes the pieces, 3 everywhere else, and 1 later**, so every TTS streams now and
the exact ones say so.

* `loom::audio::SpeechStream` (`speech_stream.h`): `next()` returns the next piece's samples; `reset()`,
  `finished()`, `pieces_done()` / `pieces_total()`, `samples()`. The driver contract is ADR-098's three
  functions: `stream_open(inputs)` takes `infer`'s inputs whole and keeps what the pieces need as state
  values; `stream_feed({})` makes one piece; `stream_finish` returns any tail. The driver keeps `piece`,
  `n_pieces` and `done`. An empty piece is legal (a Soprano sentence too short to decode) and skipped, so
  the door's stall check is the cursor: a feed that makes nothing, moves `piece` nowhere and is not done
  throws rather than looping. The input is the whole text; text arriving while audio is made is another
  contract.
* **`infer` is the stream**, opened, fed to its end and finished in one call (Zipformer streaming's
  shape), so every one-pass gate also gates the live path.
* A seeded stream gets a generator of its own from `inputs.seed`, seeded by the door before `stream_open`;
  the driver's own `loom.seed_rng` then reseeds that generator, not the bridge's.
* Pocket-TTS (all four sizes) and Soprano export the functions. Neither keeps a cache row across pieces:
  each chunk re-seeds the voice (Pocket) or prefills from position 0 (Soprano) inside its one feed.
* loom-py's `text2speech.stream(...)` takes `infer`'s arguments and returns a `SpeechStream` of `Audio`
  pieces with `exact`: True through the driver, False when cut here (`split_sentences`, one `infer` per
  sentence; ids or phonemes are one piece). `loom_cli --out x.wav --stream 1` prints each piece as it is
  ready.

## Consequences

* Exact: the stream gate (`test_e2e_speech_stream`) has the pieces joined equal `infer`'s samples bit
  for bit, after a `reset` too, and two streams pulled in alternation with a one-pass `infer` between them
  each give their own, in both cache modes: Pocket-TTS 21/21, Soprano 21/21, Pocket-TTS 2m 19/19; each
  older published file is refused. The reference gates pass on the new files (Pocket 74/74, Soprano
  185/185), and the restructured drivers are bit-identical to the exporter before them on a three-sentence
  text, two seeds each. Sabotaged (the door not seeding the stream's generator), only the interleaved
  checks fail: 17/19 on Pocket, 14/16 in the hermetic `test_speech_stream`.
* First audio, dev box CPU, three sentences: Pocket-TTS 3.65 s against 11.0 s for the whole 17.5 s clip;
  Soprano 2.92 s against 9.69 s; Pocket-TTS 2m 0.44 s against 1.37 s. `loom_cli` on Pocket: first audio
  after 3.25 s, 15.36 s of audio in 8.69 s = 1.77x real time (RTF 0.566), the same speed as one pass.
* A piece is a whole chunk, so the first one waits for the first sentence's whole generation: the latency
  is a sentence's, not a frame's. Per-frame Pocket-TTS (option 1) is the next step.
* The host cut differs from one pass where a model carries prosody across a sentence. Pocket-TTS's older
  file cut that way gave 15.36 s against 13.92 s one pass. `exact` is how a caller tells.
* Pinned draws (`noise`) and teacher latents across several chunks are carried as state counters, as the
  driver's loop carried them before; no reference fixture has more than one chunk, so that path is graded
  only by the bit-identity check above.

## Related

* `include/loom/core/speech_stream.h`; `tests/ci/test_speech_stream.cpp`, `tests/gate/test_e2e_speech_stream.cpp`
* `loom-exporter/loom_exporter/pocket_tts_driver/00_stream.lua`, `soprano_driver/00_stream.lua`
* loom-py `loom/_interfaces.py` (`Text2Speech.stream`, `SpeechStream`, `split_sentences`)
