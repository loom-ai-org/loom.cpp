---
type: epic
status: active
domain: host-api
last_updated: 2026-11-13
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
| engine | `loom::audio::TranscriptionStream`: `push(samples)` -> ids, `text`, `delta`, `revised`; `finish()`; `reset()`; `stream_open` inputs |
| `loom_cli` | `--wav x.wav --stream MS`: the clip pushed in MS pieces, each new piece of text printed at its audio time |
| loom-py | `model.speech2text.stream(**options)` -> `TranscriptStream.push(samples)` / `.finish()` -> `TranscriptUpdate` |
| engine (tokens out) | `loom::text::GenerationStream`: `next(n)` -> up to n more ids, `generate`'s ids exactly; `reset()` |
| `loom_cli` | `--prompt ... --stream N [--stream-caches auto\|park\|per-stream]`: N tokens per pull, text printed as drawn |
| loom-py | `model.text2text.stream(prompt)` / `chat_stream(messages)` -> `TextStream`, iterated for text pieces |
| engine (audio out) | `loom::audio::SpeechStream`: `next()` -> the next piece's samples, `infer`'s waveform exactly; `reset()` |
| `loom_cli` | `--prompt ... --out x.wav --stream 1 [--codec c.gguf]`: each piece reported as it is ready, the file written at the end |
| loom-py | `model.text2speech.stream(text, ...)` -> `SpeechStream` of `Audio` pieces, `exact` saying how it was cut |
| engine (codec pairs) | `loom::audio::CodecStream`: `push(codes)` -> their samples, given every frame before them; `finish()` |
| loom-py | `codec.codes2speech.stream(lm.text2codes.stream(text))` -> `Audio` per push, pushes growing 1 to 8 frames |
| engine (frame rows) | `loom::audio::FrameStream`: `push(samples)` -> the rows of the frames it completed, `first_row`; `finish()` |
| `loom_cli` | `--wav x.wav --stream MS` on a frame classifier: each change of the most probable label, printed when it is heard |
| loom-py | `model.speech2class.stream()` / `speech2embeddings.stream()` -> `FrameStream`; each push an `AudioClasses` / `FrameEmbeddings` at stream times |

A file without the three functions keeps its one-pass doors and is refused by the live one; streaming is a
second door, never a change to the first. Zipformer streaming chunks its front end in the driver: the fbank
phase runs per chunk over a window of samples starting one hop before the chunk's first frame, which under
kaldi's `snip_edges=False` gives the clip's own frames, so no topology changed. Streamed in 100 ms pieces
it runs at the one-pass speed (60 s: 7.67-7.82x real time, RTF 0.128-0.130, dev box) with 1.4 MB of stream
state, and peak RSS fell from 244 MB to 180 MB because no whole-clip feature matrix is kept.

**Tokens out, and the model's caches** ([ADR-099](../adrs/adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md)).
An LM's history is the model's KV cache (and LFM2's conv-state cache), which every stream shares. A stream
either leases it -- its written rows parked on the host while another stream runs, nothing copied while it
runs alone -- or owns a cache of its own, as `StreamCaches` says: `Auto` gives a discrete GPU a cache per
stream and parks on the CPU, an iGPU or an accelerator. A seeded stream draws from its own generator. The
LM driver's `stream_open` / `stream_feed` / `stream_finish` cut `infer_with_past`'s loop at each token and
set `done` when it would have stopped; a one-token driver streams with no re-export.

**Audio out, a sentence at a time** ([ADR-100](../adrs/adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)).
Where the reference already decodes each sentence chunk from a fresh state (Pocket-TTS, Soprano), the
driver's `stream_feed` makes one chunk, and the pieces joined are the one-pass waveform bit for bit. Every
other TTS is cut by loom-py instead -- the text split into sentences, one `infer` each -- and says so with
`exact=False`, because the joins are not one pass's. The first piece waits for a whole sentence: on the dev
box Pocket-TTS's first of three is ready after 3.65 s against 11.0 s for the clip.

**Pocket-TTS a frame at a time** ([ADR-101](../adrs/adr-101-pocket-tts-streams-a-frame-at-a-time-through-a-state-carrying-decoder-step.md)).
Its Mimi decoder runs as `mimi_step`, the next few latents with the decoder's state (conv tails,
transposed-conv overlaps, 250 keys and values per layer) carried in and out and kept on the device, as
upstream's `generate_audio_stream` decodes. A stream's pieces grow from one frame (80 ms) to eight; the
first audio is ready after 0.20 s instead of 3.65 s, at the one-pass speed (1.68x real time, RTF 0.595),
and the join is within 1.2e-06 of `infer`, which still decodes a chunk per call.

**VoxCPM2 a patch at a time** ([ADR-111](../adrs/adr-111-voxcpm2-streams-a-patch-at-a-time-through-a-state-carrying-vae-step.md)).
The same cut for a diffusion-AR loop: the AudioVAE runs as `vae_step`, with its 26 causal-conv tails and
transposed-conv overlaps carried, as upstream's `streaming_decode()` decodes each patch. `infer` and the
stream run one loop, and each step brings the LMs up to the patch it was handed before drawing, so no graph
output is read across a call. Pieces grow from one patch (160 ms) to eight. On the dev-box CPU the first
audio is ready after 3.4 s instead of 49 s, and the join is within 7.0e-07 of `infer`. Unlike `infer`, the
stream never retries a badcase, because upstream's stream does not.

**CosyVoice3 by re-running, as its reference does** ([ADR-112](../adrs/adr-112-cosyvoice3-streams-the-way-its-reference-does-by-re-running-the-flow-a-hop-at-a-time.md)).
Upstream's `tts(stream=True)` carries no flow state: each hop of tokens (25 plus a prompt pad, then 50, then
100) it runs the flow over the prompt and every token so far in the DiT's streaming mode (a block-causal mask
of 50 frames), keeps the new mel, and runs CausalHiFT over all of it, handing out what the next hop cannot
change. The export does the same. The estimator takes a chunk id per frame (all 0 is full attention), so
`infer` keeps the non-streaming mode by default and takes `streaming = 1`. A stream joins to that, not to the
default: the two modes' mels differ by ~5. Because the vocoder's sine phase turns 5e-3 of mel rounding into
O(1) sample differences, the gate compares a stream's mel with the one pass's (9.1e-4) and checks the vocoder
on its own (5.5e-06). On the dev-box CPU the first 1.36 s is ready after 46.5 s instead of 141 s.

**Soprano within a sentence** ([ADR-114](../adrs/adr-114-soprano-streams-within-a-sentence-by-decoding-windows-of-hidden-rows.md)).
Its decoder is local in time (a linear upsample, kernel-3 ConvNeXt blocks, a centred ISTFT), so a piece
of hidden rows is decoded from a window with the receptive field either side, 3 rows before and 4 after,
as upstream's `infer_stream` does with 4 and 4. No new graph: the driver cuts the LM loop per draw and keeps
only the rows a later window reaches. The pieces grow from one row (64 ms) to 16, join to `infer` bit for
bit, and the first is ready after 0.32 s on the dev box, against 2.4 s for a whole first sentence.

**LFM2.5-Audio's TTS a frame at a time** ([ADR-115](../adrs/adr-115-lfm25-audio-tts-streams-a-frame-at-a-time-by-detokenizing-windows.md)).
The same cut as Soprano's for a causal detokenizer: each piece's frames go through the existing
`detokenizer` phase from the backbone's receptive field back (97 rows: 5 convolutions x 2, 3 windowed
attentions x 29), and the last 480 samples wait for the next frame's ISTFT overlap. The join is within
1.6e-06 of `infer`; on the dev-box CPU at F32 the first 60 ms is ready after 2.1 s instead of 41 s.

**Voxtral-4B-TTS a frame at a time** ([ADR-116](../adrs/adr-116-voxtral-tts-streams-a-frame-at-a-time-through-its-chunked-causal-codec.md)).
Its codec is causal and the one pass already decodes it in chunks that re-read 32 frames before them, so
a piece is that decode started at the first frame not yet handed out, with nothing held back. The join is
within 4.7e-07 of `infer`; on the workstation's CPU the first 80 ms is ready after 4.4 s instead of 38 s.

**Chatterbox a hop at a time** ([ADR-117](../adrs/adr-117-chatterbox-streams-a-hop-at-a-time-by-re-running-its-full-attention-flow.md)).
S3Gen is CosyVoice2's flow and keeps its `finalize=False` look-ahead hook but no loop, so the stream is
CosyVoice2's: each hop re-runs the flow over every token so far and HiFT over all the mel, handing out what
its last 20 frames cannot change. The flow attends over everything, so no one pass is the join: the stream's
mel is a mean 0.085 from `infer`'s, where `infer` under another noise draw is 0.378 from it, and T3's tokens
are the same both ways. On an RTX 5090 the first piece is ready after 0.5 s and the 8 s sentence streams at
3.0x real time (6.2x in one pass), since both guidance pairs run as one batch of two
([ADR-118](../adrs/adr-118-chatterbox-runs-each-guidance-pair-as-one-batch-of-two.md)). Its gate found an engine defect: a graph built after a larger one took that one's
allocation plan, and the CPU conv fusion's layout check made the same input round two ways
([Retro-105](../retros/retro-105-a-graph-laid-out-by-the-graph-before-it.md)).

**SpeechT5 a few mel frames at a time** ([ADR-119](../adrs/adr-119-speecht5-streams-mel-frames-by-vocoding-windows.md)).
Soprano's cut on a spectrogram: the decoder loop is cut per step, and each piece of frames goes through the
existing `postnet` and `vocoder` phases over a window holding their receptive field, 34 frames either side,
derived from the layers. Pieces grow from 2 frames (32 ms) to 256. The join is within f32 rounding of
`infer` (9.7e-05 on a 21 s text: HiFi-GAN amplifies the CPU convolution's batch-dependent ulps); on the
dev-box CPU the first 32 ms is ready after 1.7 s, where the one pass takes 22.6 s.

**Codec pairs: two doors, the codec carrying its state** ([ADR-102](../adrs/adr-102-a-codec-pair-streams-as-two-doors-and-a-causal-codec-carries-its-state.md)).
A pair streams as its two files do one pass: the LM's `SpeechStream` hands back each frame of codes as it is
drawn, and the codec's `CodecStream` decodes pushes as they arrive, with the decoder's state carried by a
`codec_step` graph beside the unchanged one-pass decode. Qwen3-TTS and Audio8-TTS are built, and MOSS-TTS and Dia ([ADR-103](../adrs/adr-103-a-module-may-declare-a-ring-kv-cache-of-its-own.md)): MOSS's codec is six transformers whose
windows are kept in ring KV caches declared per module, and Dia's DAC, not causal, is decoded in windows wide
enough to cover its receptive field. MusicGen and its EnCodec, and SNAC alone, are built too
([ADR-104](../adrs/adr-104-a-recurrence-carries-across-pushes-and-noise-belongs-to-its-frame.md)): EnCodec's
convolutions are windowed and its LSTM's state is carried across pushes (`loom.run_recurrent`'s `carry`), and
SNAC's noise is drawn per frame, so a window decodes a frame with the noise it had before. For Qwen3-TTS the talker's stream is its one pass
bit for bit, the pair's audio is within 1.2e-06 of decoding the same codes in one pass, and the first audio
comes a frame after the talker draws it (1.58 s on the dev box, where the whole 30-frame generation took 28 s).

**Frame rows: a VAD a frame at a time** ([ADR-106](../adrs/adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md)).
A frame classifier or embedder answers through `FrameStream`, cut by its contract as `classify` cuts. Silero
VAD runs `infer`'s three graphs per push: `frontend` over a window that starts one whole frame back, so the
64-sample context of each frame is read from inside it, and the LSTM's h and c carried by `run_recurrent`.
On jfk the rows are 1.9e-06 from one call at any piece size (3.3e-06 from upstream's own frame loop), at 46x
real time in 32 ms pieces.

**A re-decoding ASR: Moonshine streaming** ([ADR-107](../adrs/adr-107-a-re-decoding-asr-stream-keeps-its-hypothesis-in-the-state.md)).
Its decoder cross-attends to every encoder row, so no prefix of the transcript is final. The driver keeps
its whole hypothesis in the state value `hypothesis`, the door shows it as the transcript, and an update
that changes earlier text is `revised`. The encoder runs over windows of the one-pass graph: a new
`row_offset` input places the position table, and the margins (93 rows back and 12 ahead for tiny) come
from the bands and the convolutions. Rows are encoded when a decode needs them, every `update_seconds`
(0.5 s), and a mid-stream decode teacher-forces the last hypothesis. The stream ends on one call's ids at
every cut tried. Tiny runs at 2.84x real time on jfk at 100 ms pushes, small at 0.81x (dev-box CPU). The
cost grows with the stream, since every update re-decodes from its start, so a host opens one stream per
utterance.

**A frame-synchronous ASR: Kyutai STT** ([ADR-108](../adrs/adr-108-kyutai-stt-streams-its-encoder-with-moshis-state-carried.md)). moshi streams it already, and the
export now does what moshi does. `mimi_encode_step` carries the encoder's state on the device: each causal conv's
last inputs, and the transformer's last 250 keys and values, masked as moshi's ring fills two positions at a time.
`rvq_encode` draws all 32 codes in one graph. The LM takes one step per frame over its ring KV cache. Each step's
id is final, so the transcript only grows, 0.5 s behind the audio as the model was trained. The stream ends on
one call's ids. The LM step sets the speed: at Q8_0 the stream is 0.68x real time at 80 ms pushes against 0.70x
for the one pass (dev-box CPU).

**A cache-aware transducer: Nemotron 3.5 ASR** ([ADR-109](../adrs/adr-109-nemotron-asr-streams-its-cache-aware-encoder-a-chunk-at-a-time.md)).
The encoder was trained to stream: chunked attention (4 frames, 14 chunks back) and causal convolutions.
`encoder_step` takes whole 320 ms chunks with its 75 states on the device: each subsampling stage's last column,
each layer's last 56 keys and values and its conv's last inputs. It is the one-pass encoder frame for frame,
including the clip's last partial chunk, which `transformers`' own streaming recipe drops. The greedy RNN-T
decode carries its LSTM state across calls, and its emissions are final, so the stream ends on one call's ids.
A stream's `language` is the prompt id, which `TranscriptionStream::language_input` resolves from a name
(`stream(language="de-DE")` in loom-py).

**A buffered transducer: Nemotron-Labs-Unified-ASR** ([ADR-110](../adrs/adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md)).
Its streaming mode is NeMo's buffered loop, not a cache: every step runs the whole model on up to
`left + chunk + right` seconds (5.6 + 0.56 + 0.56 by default) under a chunked attention mask, and decodes the
chunk. `encoder_step` is the encoder with that mask, its three sizes inputs, so `stream_open`'s `left_seconds`,
`chunk_seconds` and `right_seconds` pick the latency (0.16 to 2.08 s). The driver is the script's buffer
arithmetic. The stream ends on NeMo's streamed ids, which are not the one pass's. Every step re-encodes the
buffer, so it costs the buffer over the chunk: 0.33x real time at the default on the dev-box CPU.

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
| nemotron-3.5-asr-streaming-0.6b | reference's streaming mode (cache-aware: chunked attention, K/V and conv caches) | **built** (ADR-109): the encoder a few 320 ms chunks per call with its caches carried, the RNN-T decode's state carried; the stream ends on one call's ids, the last partial chunk included; publishes on rc21+ |
| kyutai-stt-1b-en-fr | exact (frame-synchronous) | **built** (ADR-108): the Mimi encoder a few frames per call with moshi's streaming state carried, the RVQ in one graph, an LM step per frame; the stream ends on one call's ids; publishes on rc21+ |
| moonshine-streaming-tiny / -small | exact encoder, decoder re-run per update | **built** (ADR-107): encoder windows with its receptive field either side, the hypothesis re-decoded every `update_seconds`; the stream ends on one call's ids; publishes on rc21+ |
| silero-vad | exact (LSTMCell state) | **built** (ADR-106): `infer`'s graphs per push behind one frame of lead, the cell's h and c carried; publishes on rc21+ |
| wakehubert-tiny | exact (strictly causal, 2.5 s receptive field) | driver only: re-run over the last 2.5 s per chunk, or a state-carrying re-export |
| nemotron-labs-unified-asr-0.6b | reference's streaming mode (buffered: chunked masks, 5.6 s left context recomputed per chunk) | **built** (ADR-110): the encoder over each buffer under the chunked mask, NeMo's buffer arithmetic in the driver; the stream ends on NeMo's streamed ids; publishes on rc21+ |
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
| pocket-tts, pocket-tts-18m / -7m / -2m | exact: AR latents, causal Mimi decoder (`generate_audio_stream`) | **per frame: built** (ADR-101; a sentence chunk per piece with `piece_frames = 0`, ADR-100), after a re-export |
| qwen3-tts-12hz-0.6b + qwen3-tts-tokenizer-12hz | exact: AR frames, causal codec | **built** (ADR-102): the talker hands back a frame per pull, the codec decodes pushes with its state carried; after a re-export of both |
| audio8-tts-realistic-0.6b + audio8-tts-codec | exact by construction: AR frames; codec all causal (convs, transposed convs, window-128 transformer) | **built** (ADR-102): the talker hands back a frame per pull, the codec decodes pushes with its state carried; after a re-export of both. Upstream's own code has no streaming entry point |
| voxcpm2 | exact: `generate_streaming`, the AudioVAE's `streaming_decode` carries causal-conv state per patch | **per patch: built** (ADR-111): `vae_step` with its 26 conv states carried, pieces of 1, 2, 4, then 8 patches; 7.0e-07 from one pass; never retries a badcase, as upstream's stream does not; publishes on rc21+ |
| moss-tts-local-transformer-v1.5 + moss-audio-tokenizer-v2 | exact: causal-transformer codec with streaming encode/decode (ring KV per layer) | **built** (ADR-103): the codec's six transformer stages each over a ring KV cache of its own, the talker a frame per pull; after a re-export of both |
| voxtral-4b-tts-2603 | exact: chunked codec equals one call | **per frame: built** (ADR-116): the one pass's chunked causal decode started at the first new frame, 32 frames of context; pieces of 1, 2, 4, then 8 frames; 4.7e-07 from `infer`; publishes on rc21+ |
| soprano-1.1-80m | exact: `infer_stream` re-runs the decoder over hidden states with a receptive-field margin ("lossless") | **within a sentence: built** (ADR-114): the decoder over windows of rows with its receptive field either side (3 rows before, 4 after), pieces of 1, 2, 4, 8, then 16 rows; bit-identical to `infer`; first 64 ms after 0.32 s on the dev box; publishes on rc21+ (a sentence per piece: ADR-100) |
| lfm2.5-audio-1.5b-tts | exact up to one ISTFT hop: detokenizer is LFM2 with a causal window of 30, then a 'same' ISTFT | **per frame: built** (ADR-115): the detokenizer over windows from its 97-row receptive field back, 480 samples held for the next frame's overlap; pieces of 1, 2, 4, then 8 frames; 1.6e-06 from `infer`; publishes on rc21+ |
| fun-cosyvoice3-0.5b | reference's streaming mode: the flow re-run per hop under a chunk mask, HiFT over all the mel with the last 8 frames withheld | **per hop: built** (ADR-112): the estimator takes chunk ids, the stream joins to `infer` `streaming = 1` (mel 9.1e-4), not to the full-attention default (~5 apart); publishes on rc21+ |
| chatterbox | generic: S3Gen keeps CosyVoice2's `finalize` look-ahead hook (broken as shipped: the mask is not trimmed), but no loop; the `S3GenStreamer` its docstring names is not shipped; flow attends over everything | **per hop: built** (ADR-117): CosyVoice2's re-run loop, hops of 25, 50, then 100 tokens with 3 of look-ahead; HiFT re-run over all the mel, its last 20 frames held; the mel 1.3e-05 from the same loop in PyTorch and a mean 0.085 from `infer` (another noise draw: 0.378); publishes on rc21+ |
| dia-1.6b, musicgen-small | generic: AR codes, non-causal codec (DAC, EnCodec 32 kHz `use_causal_conv: false`) | **built** (dia ADR-103, musicgen-small ADR-104): the talker hands back a frame once every delayed channel has reached it; DAC decodes windows with 16 frames of context each side, which its receptive field makes one call's result to f32 rounding; EnCodec windows its convolutions (3 frames before the LSTM, 1 after) and carries the LSTM's h and c |
| speecht5-tts | generic: AR mel frames, HiFi-GAN over a window | **per few frames: built** (ADR-119): the postnet + HiFi-GAN over windows holding their 34-frame receptive field either side; pieces of 2, 4, ... then 256 frames; 9.7e-05 from `infer` on a 21 s text; publishes on rc21+ |
| kokoro-82m, vits-piper-en-gb-miro, matcha-tts-ljspeech, styletts2-ljspeech, f5-tts-v1-base, supertonic-2, sanotts, vocetta2-276k, vocetta2-500k, vocetta2-1m, vaniq-edge, auroratts-100k | generic: non-AR, the whole utterance at once; sentence by sentence (Kokoro's own `KPipeline` yields per chunk) | host only: **built** in loom-py (`text2speech.stream`, `exact=False`): the text split, each sentence's audio yielded |
| dac-44khz, encodec-32khz, snac-24khz (codecs alone) | generic: non-causal decoders (SNAC also injects noise) | **built**: dac-44khz (ADR-103, windows of its one-pass graph), encodec-32khz and snac-24khz (ADR-104: windows round EnCodec's carried LSTM; SNAC's windows with its noise drawn per frame, 3 coarse frames each side) |

**Tokens out:** **built** for the causal LMs (ADR-099): qwen3-0.6b-base, smollm2-360m-instruct,
gemma-3-270m-it and lfm2-350m-monolithic stream after a re-export; lfm2-350m-modular streams as it is,
through the door's own loop. Left: flan-t5-small and the text side of every AED and speech-LM ASR, whose
loops carry per-call context (an encoder's output, a built prompt) that the stream state has to hold
first.

**Both directions:** LFM2.5-Audio's interleaved speech-to-speech mode is a full-duplex streamer upstream
(its demo decodes each audio frame as generated). The zoo exports its ASR and TTS halves separately, and
the interleaved mode is not exported.

What the rest of the tables need, now that the state, the hand-back and the caches exist (ADR-098,
ADR-099):

1. **A driver with the three stream functions** for every "driver" and "re-export" row, which means a
   re-export even where no weight changes: the driver is in the GGUF.
2. **A door per direction:** built. `TranscriptionStream` is audio in, `GenerationStream` tokens out and
   `SpeechStream` audio out; a codec pair is a text-to-codes `SpeechStream` composed with a `CodecStream`
   by the host (ADR-102); a frame classifier or embedder answers through `FrameStream` (ADR-106); a
   re-decoding recognizer keeps its `hypothesis` in the state (ADR-107); a frame-synchronous one appends a final
   id per frame (Kyutai STT, ADR-108), as a cache-aware transducer does per chunk (Nemotron 3.5 ASR, ADR-109) and a buffered one per buffer
   (the unified ASR, ADR-110).
3. **Per-call context in the stream state** for the decode loops that have it (Whisper's and T5's
   encoder output, a speech LM's built prompt), so their `PrefillDecodeLoop` can be cut per token too.
4. **A chunked front end per family**, as Zipformer's windowed fbank: mel phases still run over the
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
| Decisions | [ADR-098](../adrs/adr-098-a-live-stream-keeps-its-state-in-the-engine.md) (the live door), [ADR-099](../adrs/adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md) (the caches, tokens out), [ADR-100](../adrs/adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md) (audio out), [ADR-101](../adrs/adr-101-pocket-tts-streams-a-frame-at-a-time-through-a-state-carrying-decoder-step.md) (Pocket-TTS per frame), [ADR-102](../adrs/adr-102-a-codec-pair-streams-as-two-doors-and-a-causal-codec-carries-its-state.md) (codec pairs), [ADR-103](../adrs/adr-103-a-module-may-declare-a-ring-kv-cache-of-its-own.md) (per-module ring caches; MOSS-TTS, Dia), [ADR-104](../adrs/adr-104-a-recurrence-carries-across-pushes-and-noise-belongs-to-its-frame.md) (a carried LSTM, per-frame noise; MusicGen, EnCodec, SNAC), [ADR-106](../adrs/adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md) (frame rows; Silero VAD), [ADR-107](../adrs/adr-107-a-re-decoding-asr-stream-keeps-its-hypothesis-in-the-state.md) (a re-decoding ASR; Moonshine streaming), [ADR-108](../adrs/adr-108-kyutai-stt-streams-its-encoder-with-moshis-state-carried.md) (Kyutai STT's encoder state carried), [ADR-109](../adrs/adr-109-nemotron-asr-streams-its-cache-aware-encoder-a-chunk-at-a-time.md) (Nemotron 3.5 ASR's cache-aware encoder), [ADR-110](../adrs/adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md) (the unified ASR's buffered streaming), [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md), [ADR-018](../adrs/adr-018-chat-template-as-role-tags.md), [ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md), [ADR-055](../adrs/adr-055-a-codes-file-states-its-width.md), [ADR-111](../adrs/adr-111-voxcpm2-streams-a-patch-at-a-time-through-a-state-carrying-vae-step.md) (VoxCPM2 a patch at a time), [ADR-112](../adrs/adr-112-cosyvoice3-streams-the-way-its-reference-does-by-re-running-the-flow-a-hop-at-a-time.md) (CosyVoice3 a hop at a time), [ADR-114](../adrs/adr-114-soprano-streams-within-a-sentence-by-decoding-windows-of-hidden-rows.md) (Soprano within a sentence), [ADR-115](../adrs/adr-115-lfm25-audio-tts-streams-a-frame-at-a-time-by-detokenizing-windows.md) (LFM2.5-Audio a frame at a time), [ADR-116](../adrs/adr-116-voxtral-tts-streams-a-frame-at-a-time-through-its-chunked-causal-codec.md) (Voxtral-4B-TTS a frame at a time), [ADR-117](../adrs/adr-117-chatterbox-streams-a-hop-at-a-time-by-re-running-its-full-attention-flow.md) (Chatterbox a hop at a time), [ADR-118](../adrs/adr-118-chatterbox-runs-each-guidance-pair-as-one-batch-of-two.md) (its guidance as one batch), [ADR-119](../adrs/adr-119-speecht5-streams-mel-frames-by-vocoding-windows.md) (SpeechT5 a few mel frames at a time) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md), [Retro-104](../retros/retro-104-a-state-value-that-dropped-its-named-keys.md) (a stream's state value dropped its named keys), [Retro-105](../retros/retro-105-a-graph-laid-out-by-the-graph-before-it.md) (a graph laid out by the graph before it) |
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

