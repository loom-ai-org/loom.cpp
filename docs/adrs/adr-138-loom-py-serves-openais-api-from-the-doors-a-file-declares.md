---
type: adr
status: accepted
date: 2026-11-19
domain: host-api
tags: [api-design, host-bindings, loom-py, http, openai, server]
supersedes: []
---

# ADR-138: loom-py Serves OpenAI's API From the Doors a File Declares

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md). Asked for in
[loom-py#52](https://github.com/loom-ai-org/loom-py/issues/52): "a lot of things hook on to" OpenAI's API.

## Context

Tools, editors, chat front ends and agent frameworks talk to a model through OpenAI's HTTP API
(`/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`). loom had only in-process
hosts: the C++ API, `loom_cli` and loom-py. Every OpenAI endpoint that applies here already has a door
behind it ([ADR-013](adr-013-one-door-per-task.md)): chat and completions are `text2text` (the file's
chat template, [ADR-018](adr-018-chat-template-as-role-tags.md), and the engine's generation loop and
token stream, [ADR-099](adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md)). Speech is
`text2speech`, with voices ([ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md))
and pieces ([ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)). Transcription is
`speech2text`'s `transcribe`.

## Decision

1. **The server is loom-py's (`loom/server.py`, the `loom-serve` command), not the engine's.** It
   translates HTTP into door calls and holds no model logic. Each endpoint calls one door, picked by
   the contract's interface, never by architecture. A model answers only the endpoints its declared
   task allows (`/v1/models` lists each model's task and door).
2. **The standard library only** (`http.server`, `email` for multipart). The package stays
   dependency-free. ffmpeg on `PATH`, when present, is the one codec: it encodes mp3, opus, aac and
   flac, and decodes any upload or resamples a WAV. Without it the server reads WAV at the model's
   rate, writes WAV and PCM, refuses a named format it cannot write, and answers a speech request
   that names no format with WAV (OpenAI's default is mp3). As in `_interfaces._reference_samples`,
   the package chooses no resampler of its own.
3. **Which model answers: the request's name, else the one unambiguous model.** An OpenAI client often
   sends a fixed name (`"whisper-1"`, `"tts-1"`). An unknown name goes to the one served model with that
   endpoint's door, and the response carries its real name. With two or more, the request gets a 404
   listing them. This is the resolution order Epic-06 states: optional argument, then autodetect.
4. **A request the file cannot honour is refused, not ignored.** A refused option fails with OpenAI's
   error object (`{"error": {"message", "type", "param", "code"}}`) and a status: 400, 401 or 404.
   This covers: `speed` on a driver that does not read `inputs.speed`, `instructions` on a model that
   does not declare `tts.instruct`, and a voice the model lacks (the error lists the voices it has).
   It also covers `n > 1`, tool and image message parts, and `/v1/chat/completions` on a model with no
   chat template. A model with no voices ignores `voice`, which OpenAI requires on every request.
5. **Streaming is the engine's.** A file with stream functions (`can_stream_tokens`) streams token by
   token as server-sent events. One exported before them answers a streamed request in one piece, with
   the same SSE framing. Stop strings are cut in the host, and the stream holds back any text that
   could still begin one. `pcm` speech is chunked a piece at a time (`text2speech.stream`) at the model's
   rate, and the rate is in `X-Sample-Rate`. The other formats are encoded from the whole waveform.
6. **`finish_reason` comes from the engine's stop.** The ids `generate_ids` returns end where the
   driver stopped, without the stop token. Fewer than `max_tokens` is `"stop"` and exactly
   `max_tokens` is `"length"`. Translations try Whisper's `translate` task first, then
   `target_language="en"` (Canary). The engine refuses the one a file lacks before decoding anything.
7. **One request per model at a time.** A model holds one set of caches, so each model serialises its
   requests on a lock, and different models run concurrently (`ThreadingHTTPServer`). The server
   binds to 127.0.0.1 by default. `--api-key` (or `$LOOM_API_KEY`) requires `Authorization: Bearer`.

## Options not taken

* **A C++ server in the engine (llama-server's shape).** It would serve without Python, but it
  would duplicate the host logic loom-py already has. That logic is the voice-name resolution, the
  phonemiser front ends, sentence cutting and `Audio`. A web framework would also enter the edge build
  that the lean-runtime rule keeps small ([ADR-003](adr-003-per-model-complexity-in-the-exporter.md)). It can come later as a separate tool if a
  Python-free box needs it.
* **FastAPI / uvicorn.** That would mean three dependencies for six routes, in a package that declares
  none. The standard library server streams with chunked transfer encoding, which is all SSE needs.
* **Mapping OpenAI's voice names (`alloy`, ...) onto a model's voices.** Which loom voice is "alloy"
  is not something a file can say. The error lists the real names, and the caller picks one.
* **Embeddings, images, tools.** No `text2embeddings` door exists yet, and the LMs declare no tool
  format.

## Consequences

* Any OpenAI client reaches every chat LM, TTS and ASR file in the zoo with no per-model code:
  `OpenAI(base_url="http://127.0.0.1:8000/v1")`.
* `tests/ci/test_server.py` (36 tests) pins the server half with stand-in models: routing, request
  mapping, SSE framing, the stop scan, WAV/PCM bytes, subtitles and the error objects. A sabotaged
  `finish_reason`, `developer` role, WAV rate check or first PCM piece each fails it.
  `tests/gate/test_server.py` runs real GGUFs through the official `openai` client. With it, chat
  equals `model.chat` (SmolLM2-360M), LFM2-350M's stream joins to its whole reply, the transcription
  equals `transcribe` (Moonshine tiny) and Paradee's speech comes back as WAV, PCM and mp3.
* A newer door (a text embedder, say) becomes an endpoint by adding a route that calls it.
