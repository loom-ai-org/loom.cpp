---
type: epic
status: active
domain: host-api
last_updated: 2026-11-08
---

# Epic-06: The High-Level API and Its Hosts

## 1. Context and Scope

Two hosts consume the engine: `loom_cli` and `loom-py`. This epic covers the API they present, the
layer each piece of behaviour belongs in, and what a GGUF must declare for a host to dispatch on it
without knowing the architecture.

The authority is [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md); this epic is the orientation.

## 2. Architectural Overview

**One door per task, chosen by the modality pair the file declares** —
[ADR-013](../adrs/adr-013-one-door-per-task.md). X2Y interfaces come off `loom.task` plus the declared
input/output modalities, with canonical input names. A knob with no role in the declared contract stays
`infer`-only.

**The layering rule:** in the **FILE** when it is a property of the checkpoint; in the **ENGINE** when
it is a property of the task; in the **HOST** when it needs the host's ecosystem. Corollary, which
settles the hard cases: anything shipped inside a GGUF can only be fixed by re-exporting every model,
so an evolving policy must not be baked into files even when Lua could express it.

Net: **per-task code may live in any layer; per-architecture code only in the exporter.**

### What the engine owns

| | |
|---|---|
| `include/loom/core/model_contract.h` | the one place that knows the declared KV names. Every reader is absence-tolerant; `declared()` separates a file that states its contract from one a caller must know about |
| `include/loom/core/text_generate.h` | `loom::text::generate` — one LM loop, both driver shapes, the file's own EOS **set** (`eos_token_ids`, plural since P4.23) and its own decode rule (P4.24) |
| `include/loom/core/chat_template.h` | `loom::ChatTemplate` — a conversation to the prompt text a checkpoint was tuned on, from role tags the EXPORTER reduced its Jinja to ([ADR-018](../adrs/adr-018-chat-template-as-role-tags.md)) |
| `include/loom/core/session.h` | topologies registered and caches attached once, owned in an order that cannot dangle |
| `transcribe.cpp` | reads the declared ASR table; Whisper's spellings survive only as a flagged legacy fallback; long-form windows for a declared ceiling (`asr_long_form.h`) |

### Voices as files

A model whose voice is data rather than a reference clip can take **voice files**
([ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)): small
GGUFs whose tensors are driver inputs by name, stamped with a fingerprint of the weights they fit. The
ENGINE reads them and refuses a mismatch (`loom::load_voice`); the HOSTS resolve a name
(`loom_cli --voice <file>`; loom-py's `text2speech.infer(voice=...)`, `model.voices`, `model.voice()`,
looking beside the model file and then in its Hub repo's `voices/`). Pocket-TTS is the first model to
declare one.

### The language written (ASR)

A model whose output language is chosen separately from its input one declares
`loom.asr.target_language_names`/`_ids`, and `transcribe` resolves the `target_language` role against
it ([ADR-061](../adrs/adr-061-the-language-written-is-its-own-role.md)); loom-py
`transcribe(target_language=)`, `loom_cli --target-language`. Canary is the first, defaulting to
English. A declared language/task table also reaches a dynamic-length driver now -- before Canary,
only windowed (Whisper) drivers were handed `language`/`task`. `Model.contract` lists all three tables
as `asr_languages`, `asr_tasks`, `asr_target_languages`.

### Long audio past a training ceiling (ASR)

A dynamic-length ASR file that was trained on short clips declares its ceiling and how to stitch past it
(`loom.asr.window_*`, `loom.asr.merge_*`); `transcribe` then cuts longer audio into overlapping windows,
decodes each alone, and merges their tokens -- NeMo's own long-form scheme, ported decision for decision
([ADR-063](../adrs/adr-063-a-long-clip-is-decoded-the-way-its-reference-decodes-it.md),
`asr_long_form.h`). Canary is the first; its long-form transcripts are NeMo's, token for token.

### Reference clips (in-filling)

A model with no voice of its own that clones by IN-FILLING (F5-TTS) declares `loom.tts.reference =
"infill"` ([ADR-056](../adrs/adr-056-a-model-that-clones-by-infilling-says-so.md)). The host joins the
clip's transcript to the text and passes the clip, the joined ids and the join
(`waveform`/`text_ids`/`n_ref_text`): loom-py's `text2speech.infer(text, reference=, reference_text=)`
and `loom_cli --wav --ref-text` build the same three. A model that does not declare it is refused
rather than handed a clip it would ignore.

### Codec pairs in `loom_cli`

A `text2codes` file (Dia, Qwen3-TTS, MOSS-TTS) answers with codec tokens, and its codec is a second
GGUF ([ADR-022](../adrs/adr-022-dia-and-its-codec-stay-two-files.md)). `loom_cli --model <lm.gguf>
--prompt "<text>" --codec <codec.gguf> --out out.wav` runs both halves the way loom-py's
`text2codes.infer` → `codes2speech.infer` does. `--codes-out` keeps the frames as a codes GGUF that
states its own width, and a codec decodes it later with `--prompt @codes.gguf`
([ADR-055](../adrs/adr-055-a-codes-file-states-its-width.md)). It
dispatches on the declared interface, ahead of the vocabulary-tag branches — Dia's `byt5` tag was an
inspection-only branch and the Qwen2 `gpt2` tag the text generator. Three choices the loom-py doors do
not face:

* **The pairing is checked from the codec's header** (`GgufModel::load_metadata`) before any frame is
  generated: an LM no wider than the codec, and a narrower one only when the codec declares
  `codec.absent_code` ([ADR-050](../adrs/adr-050-a-codec-declares-its-absent-id-and-its-channels.md)).
* **The LM is released before the codec loads.** MOSS-TTS is 16 GB at F32 and its codec 4 GB.
* **A flag the driver would ignore is refused.** `--wav` (Qwen3-TTS's clip) and `--ref-text` (its ICL
  replay) are sent only to a driver that reads `inputs.waveform` / `inputs.ref_audio` +
  `inputs.ref_tokens`; `--language` is a 1-based index into `loom.text.languages`, as in loom-py.
  `--n-predict` caps FRAMES and is sent only when given.

### Streaming: the live door

**Built for ASR, first on Zipformer streaming** ([ADR-098](../adrs/adr-098-a-live-stream-keeps-its-state-in-the-engine.md)).
A driver that can stream exports `stream_open` / `stream_feed` / `stream_finish` beside `infer`, and keeps
nothing between calls: the host owns a `StreamState` per stream (`stream_state.h`) -- named device tensors
(`loom.keep_state(name, module)`, read back as `{state = name, index = k}`) and named host values
(`loom.state` / `loom.set_state`) -- and hands it to each call (`LoomLuaBridge::call(fn, args, state)`).
Streams on one loaded model are therefore independent, and a one-pass call between two pushes disturbs
none of them. A plain `call` gets a fresh state of its own, so a one-pass `infer` can be the three
functions in one call, and the one-pass gate then covers the live code.

| Host | Door |
|---|---|
| engine | `loom::audio::TranscriptionStream`: `push(samples)` -> ids, `text`, `delta`; `finish()`; `reset()` |
| `loom_cli` | `--wav x.wav --stream MS`: the clip pushed in MS pieces, each new piece of text printed at its audio time |
| loom-py | `model.speech2text.stream()` -> `TranscriptStream.push(samples)` / `.finish()` -> `TranscriptUpdate` |

A file without the three functions keeps its one-pass doors and is refused by the live one; streaming is a
second door, never a change to the first. Zipformer streaming chunks its front end in the driver: the fbank
phase runs per chunk over a window of samples starting one hop before the chunk's first frame, which under
kaldi's `snip_edges=False` gives the clip's own frames, so no topology changed. Streamed in 100 ms pieces
it runs at the one-pass speed (60 s: 7.67-7.82x real time, RTF 0.128-0.130, dev box) with 1.4 MB of stream
state, and peak RSS fell from 244 MB to 180 MB because no whole-clip feature matrix is kept.

### Streaming: which other models can, and how

Every other door takes a whole input and returns a whole output. A streaming API covers three things.
**Audio in:** the host pushes audio as it arrives and reads partial results. **Audio out:** a TTS hands back
audio while it is still generating. **Tokens out:** an LM hands back each token as it is drawn. The survey
below (2026-11-08, user: "all models amenable to a streaming API get it, even if they don't have stream in
the name") covers all 72 published model cards. It was read off the reference code, the checkpoints' READMEs and
the exports; no row has been run streamed yet.

The kinds of streaming, by how close a streamed result is to the one-pass one:

* **Exact:** a causal model whose state is carried between calls (or recomputed over a window at least
  as long as its receptive field) gives the one-pass numbers. Zipformer streaming shows the state can stay
  in the engine as retained outputs
  ([ADR-097](../adrs/adr-097-a-streaming-checkpoint-runs-its-chunk-loop-behind-the-one-pass-door.md)).
* **The reference's own streaming mode:** upstream defines one, and it computes different numbers from its
  offline mode (other masks, re-decoding). The oracle is that mode, not the one-pass export.
* **Generic:** no upstream mode. Approaches include re-decoding a growing or sliding window and committing
  a stable prefix (whisper_streaming's local agreement) for ASR, sentence-by-sentence synthesis for TTS,
  and overlapping windows for a non-causal codec. The output differs from one pass at the seams.

**Audio in (ASR, VAD, wake word, segmentation):**

| Model | Kind | What it needs |
|---|---|---|
| zipformer-small-streaming | exact | **built** (ADR-098); publishes with the rc21 republish |
| nemotron-3.5-asr-streaming-0.6b | reference's streaming mode (cache-aware: chunked attention, K/V and conv caches) | re-export: the encoder step with its caches; today the export is the offline mode, the whole clip at once |
| kyutai-stt-1b-en-fr | exact (frame-synchronous) | re-export: Mimi encoder with its conv state and ring KV carried; the LM loop is per frame already |
| moonshine-streaming-tiny / -small | exact encoder, decoder re-run per update | re-export: incremental encoder (windows (16, 4) and (16, 0), 3 frames lookahead on 4 layers) |
| silero-vad | exact (LSTMCell state) | re-export: the per-frame cell with its state as an input; today one sweep over the clip |
| wakehubert-tiny | exact (strictly causal, 2.5 s receptive field) | driver only: re-run over the last 2.5 s per chunk, or a state-carrying re-export |
| nemotron-labs-unified-asr-0.6b | reference's streaming mode (buffered: chunked masks, 5.6 s left context recomputed per chunk) | re-export: the streaming-mode encoder beside the offline one |
| qwen3-asr-0.6b | reference's streaming mode (every 2 s, all audio so far re-fed; previous transcript minus its last 5 tokens as prefix; first 2 chunks unprefixed) | driver/host only: a prefix input; cost grows with the stream |
| oruk-loop-153m | generic: Qwen3-ASR's scheme (its teacher) | as qwen3-asr; no reference streaming mode to grade against |
| parakeet-tdt-0.6b, parakeet-rnnt-0.6b, canary-1b-v2, conformer-ctc-small, citrinet-1024, citrinet-256-ls | generic: NeMo's buffered chunked inference (overlapping windows, merged tokens) | host: the window + merge loop; Canary's long-form merge ([ADR-063](../adrs/adr-063-a-long-clip-is-decoded-the-way-its-reference-decodes-it.md)) is the nearest code |
| whisper-small, moonshine-tiny, granite-speech-4.0-1b, lfm2.5-audio-1.5b-asr, sensevoice-small, paraformer-zh, gigaam-v3-rnnt, hubert-large-ls960-ft, data2vec-audio-base-960h | generic: re-decode a growing or sliding window | host only |
| marblenet-vad-v2 | generic: sliding window (finite receptive field, with lookahead) | driver only |
| pyannote-segmentation-3.0 | generic: rolling 10 s windows (diart-style) | host only |

Not streamable in a useful sense: titanet-large and ecapa-voxlingua107 (one embedding per utterance), and
the text classifiers distilbert-ner and punctuate-all (whole text in, labels out).

**New streaming leaves on disk, not in the zoo:**

| Candidate | Status |
|---|---|
| `funasr/paraformer-zh-streaming` | the paraformer leaf refuses it |
| `nemotron-speech-streaming-en-0.6b` | the English sibling |
| Zipformer's other two latencies | an export constant each |
| `kyutai-stt-2.6b-en` | on disk |
| `voxtral-mini-4b-realtime-2602` | on disk |

**Audio out (TTS and codecs):**

| Model | Kind | What it needs |
|---|---|---|
| pocket-tts, pocket-tts-18m / -7m / -2m | exact: AR latents, causal Mimi decoder (`generate_audio_stream`) | re-export: Mimi decoder step with its conv state; today one call over every latent |
| qwen3-tts-12hz-0.6b + qwen3-tts-tokenizer-12hz | exact: AR frames, causal codec | driver: emit each chunk of the existing chunked decode ([ADR-034](../adrs/adr-034-a-chunked-decode-is-the-drivers-loop-not-a-longer-call.md)) as it is made |
| audio8-tts-realistic-0.6b + audio8-tts-codec | exact by construction: AR frames; codec all causal (convs, transposed convs, window-128 transformer) | driver: decode with ≥ receptive-field left context, or a state-carrying codec re-export. Upstream's own code has no streaming entry point; the README's SGLang-Omni adapter does, unbenchmarked |
| voxcpm2 | exact: `generate_streaming`, the AudioVAE's `streaming_decode` carries causal-conv state per patch | re-export: the VAE decoder step with its state |
| moss-tts-local-transformer-v1.5 + moss-audio-tokenizer-v2 | exact: causal-transformer codec with streaming encode/decode (ring KV per layer) | re-export: codec step with its ring caches; a context-window chunk loop was 48% wrong here |
| voxtral-4b-tts-2603 | exact: chunked codec equals one call | driver: emit each `codec_chunk` as it is made |
| soprano-1.1-80m | exact: `infer_stream` re-runs the decoder over hidden states with a receptive-field margin ("lossless") | driver only: the decoder phase takes any length already |
| lfm2.5-audio-1.5b-tts | exact up to one ISTFT hop: detokenizer is LFM2 with a causal window of 30, then a 'same' ISTFT | driver: chunked detokenize with a hop of lookahead |
| fun-cosyvoice3-0.5b | reference's streaming mode ("bi-streaming": chunk-aware flow mask, HiFT cache) | re-export: the streaming flow and HiFT; the export rewrote only the cache-less call |
| chatterbox | generic: S3Gen's flow has a `finalize` lookahead hook, but the `S3GenStreamer` its docstring names is not shipped | driver: chunked flow + vocoder with overlap |
| dia-1.6b, musicgen-small | generic: AR codes, non-causal codec (DAC, EnCodec 32 kHz `use_causal_conv: false`) | driver: overlapping codec windows |
| speecht5-tts | generic: AR mel frames, HiFi-GAN over a window | driver: vocode overlapping mel windows |
| kokoro-82m, vits-piper-en-gb-miro, matcha-tts-ljspeech, styletts2-ljspeech, f5-tts-v1-base, supertonic-2, sanotts, vocetta2-276k, vocetta2-500k, vocetta2-1m, vaniq-edge, auroratts-100k | generic: non-AR, the whole utterance at once; sentence by sentence (Kokoro's own `KPipeline` yields per chunk) | host only: split the text, yield each sentence's audio |
| dac-44khz, encodec-32khz, snac-24khz (codecs alone) | generic: non-causal decoders (SNAC also injects noise) | host: overlapping windows |

**Tokens out:** every model with a token loop: qwen3-0.6b-base, lfm2-350m-monolithic, lfm2-350m-modular, smollm2-360m-instruct,
gemma-3-270m-it and flan-t5-small, plus the text side of every AED and speech-LM ASR. The drivers already
draw one token per step; what is missing is a way to hand each one back mid-run.

**Both directions:** LFM2.5-Audio's interleaved speech-to-speech mode is a full-duplex streamer upstream
(its demo decodes each audio frame as generated). The zoo exports its ASR and TTS halves separately, and
the interleaved mode is not exported.

What the rest of the tables need, now that the state and the hand-back exist (ADR-098):

1. **A driver with the three stream functions** for every "driver" and "re-export" row, which means a
   re-export even where no weight changes: the driver is in the GGUF.
2. **A door per direction.** `TranscriptionStream` is audio in; tokens out (LMs) and audio out (TTS,
   codecs) need their own, on the same `StreamState` and the same three-function shape.
3. **A chunked front end per family**, as Zipformer's windowed fbank: mel phases still run over the
   whole clip.

The "host only" rows (sentence-by-sentence TTS, re-decoded windows for ASR) need no re-export.

### Resolution order for anything a model can infer itself

**Optional argument → autodetect from the file → default.** Language is the worked example. Capability
is declared by the **export**, not by the host.

### Compatibility

Pre-contract GGUFs still work through the legacy fallback, and **the fallback must stay tested** — a
declared-only test would stay green while the path every GGUF on disk depends on rotted. The `v4`
fixture set is kept for exactly this reason; `v5` is the contract-declaring set.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Authority | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) |
| Decisions | [ADR-098](../adrs/adr-098-a-live-stream-keeps-its-state-in-the-engine.md) (the live door), [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md), [ADR-018](../adrs/adr-018-chat-template-as-role-tags.md), [ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md), [ADR-055](../adrs/adr-055-a-codes-file-states-its-width.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| Active tasks | [Backlog → Host API](../backlog/active-index.md#host-api) |

## 4. The Record

### P4.24 — the engine could not sample: `argmax_row` was the only decode rule — DONE (2026-08-29)

**One sentence.** There was no temperature, top-k, top-p or multinomial draw anywhere in `src/` or
`include/`; every causal LM decoded greedily, and a checkpoint whose own `generation_config.json` says
`"do_sample": true` was run in a mode its authors did not choose.

`loom.sample_row(module, row, {temperature =, top_k =, top_p =})` now sits beside `loom.argmax_row`,
reducing ON the tensor and drawing from the bridge's shared `rng_`.

**It has since grown two more options**, both for family 10 and both in the same table rather than in
new bindings: `lo`/`hi` restrict the draw to a half-open id window and return absolute ids, and
`guidance = {module =, scale =, top_k =}` is classifier-free guidance over a second module's retained
logits. [ADR-024](../adrs/adr-024-guidance-belongs-in-the-sampler.md) argues both, including why this
deliberately does not mirror the `argmax_row`/`argmax_row_range` split, and
[Retro-031](../retros/retro-031-dias-guidance-is-not-the-standard-formula.md) is why "guidance" needed
reading rather than assuming.

#### The shape, and the two decisions inside it

**Greedy is `temperature = 0`, not a separate flag**, and that is what makes the two bindings incapable
of disagreeing: at `temperature <= 0` or `top_k == 1`, `sample_tensor_row` returns
`argmax_tensor_row`'s own answer, by calling it. Two ways to get a token out of one forward pass that
can differ is the failure P4.0.14 already retired once.

**The knobs' path is `optional argument → autodetect from the file → default`** (Epic-06 §2). The
exporter writes the checkpoint's `generation_config.json` sampling defaults as `loom.sampling.*` KVs
and as the driver's own `inputs.temperature or <default>` fallbacks; `GenerateOptions`' three knobs are
`std::optional` and only a SET one is passed. Unset therefore means "what the file declared", and a
file declaring nothing means greedy — a host default filled in anywhere along that path would silently
overrule every checkpoint for every caller who named nothing.

The order is `transformers`' own processor order — temperature, then top-k, then top-p, then the draw —
because the reference this is defined against is `generate` under the checkpoint's own generation
config. Any other order gives a different distribution from the same three numbers.

#### Why it is a bridge builtin at all, against the criterion that seems to forbid it

`lua_bridge.h`'s binding criterion lists "a sampler" among the things that do NOT belong in C++. That
line meant an orchestration — a LOOP over steps, like the ADPM2 diffusion sampler, which is still Lua —
and it now says so. `loom.sample_row` is a **reduction over one row**, the same shape as `argmax_row`:
it reads no model config, every knob is a call argument, two unrelated families could use it unchanged,
and its input cannot cross the boundary at all — `run_subgraph` marshals every element into a Lua table
and LuaJIT's array part tops out near 2^27, so a 262144-wide vocab overflows at ~512 prompt tokens
([Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md)). Sampling in Lua would reinstate
that ceiling at exactly the vocabulary size that found it. The decode loop AROUND it is still driver
Lua.

Temperature + top-k + top-p covered the whole fixture set when this was written, and exactly one model
needed any of it (gemma-3-270m-it: `top_k 64, top_p 0.95`). Family 10 added the second — Dia declares
`do_sample` at 1.8/50/0.9 *and* a guidance scale, which is what the two extra options above are for.
Min-p, the repetition penalties, Mirostat and beam search are still not implemented and nothing asks
for them.

#### Gates

`tests/ci/test_sample_row.cpp`, over the 6-class toy fixture — which is what makes "every class is
eventually drawn" a statement a test can make:

* every greedy spelling equals the argmax — an empty options table, `temperature = 0`, `top_k = 1`, and
  a `top_p` below the best token's own probability (the "at least one candidate survives" clause);
* one seed gives the same 40 draws twice; a different seed differs;
* at a flattening temperature all six classes are reached, which no argmax ever is;
* `top_k = 2` returns exactly two distinct ids, and the argmax is among them;
* a negative `top_k` and a `top_p` outside `(0, 1]` raise by name.

Sabotaged to confirm it fails: forcing the greedy branch takes the test red.

**Greedy stayed the default and it was verified rather than asserted** — a re-exported greedy
checkpoint decodes the same ids, and every fixture in `v5` carries a driver that calls `argmax_row`, so
nothing already on disk changed path at all.

