---
type: epic
status: active
domain: model-coverage
last_updated: 2026-09-24
---

# Epic-03: Model Coverage

## 1. Context and Scope

The measure of the data-driven design is how cheaply a new architecture arrives. This epic covers which
models ship, which family template each belongs to, and the roadmap for the rest — with the standing
acceptance criterion that **a new family should need no engine work**.

**Thirty models are published** at
[huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org) as of 2026-09-17, every one of them
re-exported and card-gated for `1.0.0-rc10`; nothing exported and verified is waiting to be published.
**Family 11 is closed**: DAC, SNAC and EnCodec cover the uniform, multi-rate and recurrent shapes a
codec decoder comes in, and Qwen3-TTS's tokenizer adds the chunked-attention one. **Family 4 is closed
on three checkpoints**, covering the three architectures HF's `AutoModelForCTC` resolves. **Family 5 is
closed on two** — SenseVoice-Small and Paraformer-zh. Each is a single GGUF carrying its own
topologies, driver and — where the architecture has one — its vocabulary, with the two deliberate
exceptions where a codec is its own file ([ADR-022](../adrs/adr-022-dia-and-its-codec-stay-two-files.md)).

Two checkpoints are verified and deliberately **not** published, for different reasons.
`omniASR-CTC-300M-v2` is family 4's third structural witness: its own documented processor path
transcribes garbage and loom reproduces that exactly, so the export is right and the checkpoint's
declared front end disagrees with its weights. `bert-base-NER` is family 12's first checkpoint and
stays in the export sweep as a **structural witness only** — DistilBERT-NER was chosen as the
family's published English NER representative, and shipping both would put two cards on the Hub for
one task and one vocabulary. A family proves itself on several checkpoints and publishes one per
task; that is the rule these two are instances of, not an omission in either case.

## 2. Architectural Overview

### Shipped families

| domain | models | template |
|---|---|---|
| **Language** | Qwen3-0.6B-Base, LFM2-350M (monolithic *and* modular), SmolLM2-360M-Instruct, Gemma-3-270M-it | `causal_lm_export.py` |
| **ASR — NeMo encoders** | Conformer-CTC-small, Parakeet-TDT-0.6B, Parakeet-RNNT-0.6B, GigaAM v3 | `nemo_asr_export.py` |
| **ASR — CNN + transformer + CTC** | any HF `*ForCTC` (HuBERT, data2vec-audio, wav2vec 2.0) | `ctc_asr_export.py` |
| **ASR — SANM / FunASR** | SenseVoice-Small | `sanm_asr_export.py` |
| **ASR — SANM + CIF** | Paraformer-zh | `paraformer_export.py` |
| **ASR — encoder-decoder** | Whisper-small | `multi_phase_export.py` |
| **ASR — composition** | Qwen3-ASR-0.6B, Granite-Speech-4.0-1B | `speech_lm_export.py` |
| **TTS — flow matching** | Matcha-TTS, SupertonicTTS, F5-TTS | `flow_matching_export.py`, `f5_tts_export.py` |
| **TTS — other** | Kokoro-82M, StyleTTS2, VITS (piper) | `multi_phase_export.py` |
| **Token classification** | any HF `*ForTokenClassification` (BERT-NER, DistilBERT-NER) | `token_classification_export.py` |
| **Audio codec (decode)** | DAC-44kHz, SNAC-24kHz | `audio_codec_export.py` |
| **Audio codec + recurrence** | EnCodec-32kHz | `encodec_export.py` |
| **Audio codec + chunked attention** | Qwen3-TTS-Tokenizer-12Hz | `qwen3_tts_export.py` (companion) |
| **Text → codec tokens** | Dia-1.6B, Qwen3-TTS-12Hz-0.6B-Base | `dia_export.py`, `qwen3_tts_export.py` |
| **Text encoder-decoder** | flan-t5-small (and every `model_type: t5`) | `t5_export.py` |

The two LFM2 entries are the *same checkpoint exported two ways*, which is how the engine's two
decomposition paths stay honest about producing the same model.

Each ASR model takes a **raw waveform**: the mel front end is inside the graph, not in front of it.

### The composition template, and why it was cheap

The audio-encoder + projector + causal-LM family is the largest group on the roadmap — ~19 converters,
~36 models. The finding that made it cheap: **the prompt needs no concatenation anywhere.**

"Inject audio embeddings into the prompt" reads as though something must build one `inputs_embeds` out
of text and audio embeddings, which would need a backend-side concatenation of two retained tensors —
an engine op that does not exist. It is not needed. Attention is causal and the decoder is KV-cached,
so a call at `n_past = k` over `n` rows writes cells `[k, k+n)` and attends over `[0, k+n)`: **feeding
a prompt as N successive cached calls is the same arithmetic as feeding it concatenated.** Measured
against HF before any component was written. `PromptSegments` is that walk, and it stops one segment
short so the final text segment is the decode loop's own first iteration.

### The family-3 template contract

A leaf owns the encoder; segmented prefill needs no concat; chunk arithmetic is per-model
(Qwen3-ASR 1 s / 13 frames, Granite 12 s / 120).

### Family 12, and what a family is supposed to cost

Family 12 — BERT-family token classifiers, `text` in and one class per token out — is the roadmap's
smallest template and the first non-audio task, and it is here as the **measurement of the acceptance
criterion** rather than for its coverage. It needed no new engine primitive: `WordPieceVocab` already
read its vocabulary, and `loom.argmax_rows` already performed its reduction (built for Conformer-CTC's
frame-wise head in P4.0.17). Its whole orchestration is one component — `TokenLabelsEpilogue`, which is
`CtcGreedyEpilogue` with the collapse removed, because for a token classifier the alignment between row
*i* and token *i* IS the answer. The synthesized driver is five lines.

It also proves the registry off audio. `loom.task = "token-classification"` with
`loom.output.kind = "class"` is the first export whose contract's non-audio half is exercised end to
end, and it turned `Text2Class` from a `_Planned` interface into an answered one
([ADR-013](../adrs/adr-013-one-door-per-task.md)).

What it did cost is one door in the engine, `loom::text::classify` — per-TASK by the §2 rule of
[HIGH-LEVEL-API](../HIGH-LEVEL-API.md), because whether the framing tokens an encode adds come back
labelled is a policy two hosts would otherwise decide independently. And one tracing rule, recorded as
[ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md): **every tensor the model needs is
derived from an input, never computed from `.shape[1]`** — the mask, `token_type_ids` and
`position_ids` alike, because `transformers` computes all three from the Python-level sequence length
and a traced graph bakes each of them.

**It is the first family here proved on a second, structurally different checkpoint**, which is what
the "modular-export generality is unproven" item asks of a template and what this one now has.
DistilBERT has no token-type embeddings, no `position_ids` argument and `.transformer` where BERT has
`.encoder`; the template absorbs all three by reading `base.forward`'s signature rather than its name,
and the two exports produce identical graph inputs and identical drivers. The first version of ADR-019
generalised from BERT alone and was falsified within a day, which is the argument for the second
checkpoint stated as cheaply as it can be.

Verified against `transformers` on the tensor rather than on the argmax (the *tensor oracle, not token
oracle* standing rule), over 138 tokens: max |Δ| 1.24e-05 (BERT) and 5.72e-06 (DistilBERT), worst
sentence cosine 0.99999988 for both, 138/138 argmax, and the engine's own WordPiece encode identical to
HF's ids. The sabotage arm — the same graph against a different sentence's reference — reads 11.94 and
9.54.

#### The third checkpoint, where the TOKENIZER was the new thing (2026-09-04)

BERT and DistilBERT are structurally different **encoders** and the same WordPiece vocabulary with the
same CoNLL-03 head, so what the first two checkpoints left untested was the other half of the door.
`oliverguhr/fullstop-punctuation-multilang-large` — XLM-R, SentencePiece Unigram, 250,002 pieces, a
punctuation-restoration head — is the third, and it moved nothing in the template and two things
either side of it.

**The tokenizer half cost one reader.** A fairseq-derived checkpoint's ids are not its protobuf's piece
order: `transformers`' converter re-heads the vocabulary and shifts every piece by one, so writing the
proto verbatim yields a file that loads, decodes to readable text, and is off by one against the
embedding table everywhere. The fix is to read the ids off the `tokenizer.json` the checkpoint already
ships and keep the piece TYPES and the normalizer from the protobuf —
[ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md), whose seam
(`tokenizer.json`'s presence) is why the six NeMo SentencePiece models in the sweep are byte-identical.
Engine encode now matches `AutoTokenizer` on 8/8 multilingual sentences including the `<s> … </s>`
framing. **No engine change**: `loom::Vocab`'s UGM Viterbi already did all of it, and had simply never
been handed a vocabulary this family produced.

**The template half cost three lines and a retro.** XLM-R numbers positions from `padding_idx + 1`, so
the family's 0-based `loom.range(0, n_tokens)` read two rows of the position table that were trained as
padding — max |Δ| 7.040 in the logits with **24 of 27 argmaxes still agreeing**
([Retro-039](../retros/retro-039-position-zero-was-not-row-zero.md)). `_position_offset` reads it off
the table's own `padding_idx`, the wrapper adds it, and the artifact's contract is unchanged for every
member. The same offset shortens the family's cap: XLM-R's 514-row table serves 512 tokens.

**The engine half cost one KV in one list.** `loom::text::classify` strips the framing an encode added,
and read BOS/SEP/PAD only — so a SentencePiece checkpoint's trailing `</s>` came back labelled. The
framing is per-TOKENIZER, not per-task.

Verified the same way as the first two, over 70 tokens: max |Δ| **4.196e-05**, cosine 0.999999973,
70/70 argmax, sabotage arm 17.49. The larger delta than BERT's is the model — 24 layers at width 1024
against 12 at 768.

### Family 11, and the bug that had no symptom

Family 11 — neural audio codec decoders, `audio_codes` in and a waveform out — is family 10's
connector: an AR codec-token LM (~20 models whose LM half already exports) emits integers and is
silent without it. DAC is the first leaf.

Like family 12 it needed **no new engine primitive** — the whole decode path lowers to ops the
dialect already had, because the HiFi-GAN/iSTFT half was exported inside families 7/8/9 and Snake
decomposes into Kokoro's SnakeBeta ops. And like family 12, the real work was tracing rather than
architecture. Two findings worth carrying to the next codec:

* **The RVQ loop is a graph fact and unrolls**, correctly: `from_codes` is a Python loop over N
  codebook lookups, and N is a property of the checkpoint, not of the input. No hparam the driver
  reads, no Lua loop.
* **The dynamic axis broke on a rank-reducing slice.** `audio_codes[:, i, :]` drops the codebook axis,
  and the exporter's shape walk bailed on any squeezing slice — so the length came back as a literal
  `1`, and every transposed convolution downstream was cropped to `(1-1)*stride + kernel - pad`.
  **Nothing raised.** The export succeeded, the audio was correct, and the model returned one frame's
  worth of samples for every input. It is the sharpest instance yet of the standing lesson: an export
  that runs is not an export that works, and the only thing that catches this class is asserting on
  the emitted shapes rather than on the call.

Verified against `transformers` on the waveform, on real speech at two clip lengths: max |Δ| 1.85e-04
(2 s) and 2.22e-04 (5 s), cosine ~1.0. Sabotage arm — the same graph against a different clip's
reference — 1.14.

#### EnCodec: both blockers closed, and a third that was never named

**Shipped 2026-09-12.** The scoping below was written before the work and is kept because it was
half right in an instructive way: the two blockers it named were real, and neither was where the
difficulty actually lay.

* **Blocker 1 (dynamic padding) was one line, and is now PROVED rather than argued.** All 10
  `EncodecConv1d` on the decode path have stride 1 and their extra padding is 0 at every length, so
  the patch is a constant — re-derived from the real modules by
  `tests/ci/test_encodec_export.py` every run rather than trusted from a comment.
* **Blocker 2 (the LSTM) needed no new machinery at all.** `RecurrentPhase` has traced a stacked
  `nn.LSTM` into per-timestep cell topologies since Parakeet's prediction network, and
  `loom.run_recurrent` runs one in C++. What was missing was the DRIVER-side call, now `RecurrentCall`
  — one Lua call per layer, where `run_bi_lstm` loops timesteps in Lua.
* **The blocker nobody named cost the most.** EnCodec crops with `x[..., left : shape[-1] - right]`,
  and MIL retires its symbolic algebra through a shape-derived slice — `8*is0 + 8` in, a fresh opaque
  `is118` out. The exporter's rule for an unknown symbol is to substitute the root axis, so a crop that
  should read `8*n_codes + 2` was emitted as `n_codes + 2` and the model decoded 4 seconds of audio
  into 200 samples. No error anywhere. [Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)
  has the two-part fix and the general lesson.

**One engine change: `ELU`** — EnCodec's SEANet decoder activates with it where every vocoder in
families 7-9 uses LeakyReLU. `ggml_elu` already existed unexposed, so it is one registration, one
topology rule (guarding `alpha != 1`, which ggml's is fixed at) and one line in the shape walk.

**The contract is unchanged and the export shape is not**, which is
[ADR-030](../adrs/adr-030-a-task-is-a-contract-not-an-export-shape.md): the task fixes
`audio_codes -> audio`, and whether that is one traced graph or three phases around a C++ loop is
the family's own business. `audio-codec` now declares the root base class for the same reason
`automatic-speech-recognition` does.

Verified against `transformers` on the waveform, at 32 kHz: max |Δ| **5.07e-07**, cosine 1.000000, the
exact sample count (`n_frames * 640`), sabotage arm 0.55, and the ASR oracle reading the clip back.
4 s of audio decodes in 1.7 s on the two-core dev box, the LSTM loop included.

*The original scoping, kept for the record:*

* **coremltools refuses its convolution padding** once the frame axis is dynamic — `Dynamic padding
  for n-dimensional tensors is not supported`, because `EncodecConv1d` pads by a length-derived
  amount. It is the same limitation that keeps Supertonic's text axis static
  ([Retro-005](../retros/retro-005-supertonic-fixed-text-length.md)). Tractable: every decode-path
  convolution is stride 1, where the extra padding works out to exactly 0, so patching it to a
  constant should be sound — but a wrong pad is a silent output shift rather than an error, so it has
  to be proved per stage.
* **Its decoder contains a 2-layer LSTM over the time axis.** DAC's is purely convolutional. A
  flattened trace unrolls the LSTM at the traced length and bakes it, so this is a
  `ScriptedLoop`/`run_recurrent` export rather than the four-line `Flattened` one. The machinery
  exists — StyleTTS2's BiLSTM already goes through it — and what is missing is wiring this family to
  it.

The half that is already done: `CodecFamily.decode` knows EnCodec's chunked
`(audio_codes, audio_scales)` signature and `[chunks, batch, n_q, frames]` layout, and `geometry`
knows its config spellings, which differ from DAC's in every field but `codebook_size`. Both are
pinned by tests so they cannot rot while the blockers stay open, and the recognizer **detects** an
EnCodec directory and raises naming both reasons — detection is what makes the failure sayable at all.

**This is why the composition target changed.** MusicGen was picked for its small LM and would have
dragged in this codec; Dia decodes through DAC, which is done, so it costs the LM half only.

#### The second leaf was SNAC, and the layout claim held

SNAC 24 kHz shipped 2026-09-11 and is the leaf Epic-03 had named for the purpose: `vq_strides =
[4, 2, 1]` puts its three codebooks at three different frame rates, which is what tests whether "codes
in, frame-major" survives a multi-rate codec. **It survives**, with the row read as the coarsest
codebook's frame — `sum(coarse // stride)` ids wide, 7 here, level-major — which is `n_codebooks`
exactly when every stride is 1, so DAC is the same formula rather than a second branch.
[ADR-029](../adrs/adr-029-a-multi-rate-codec-keeps-one-row-per-coarsest-frame.md) records that and the
two consequences a caller sees: `codec.n_codebooks` is code streams per frame (7 for 3 codebooks), and
`codec.frame_rate` is the rate of the rows (11.72 Hz, not the codec's own 46.875).

**No new engine primitive, again** — the third family in a row. The decode path lowers to convolutions
(depthwise and transposed), `SIN`/`SQR` and the `REPEAT` that carries `from_codes`' `repeat_interleave`
back up to the finest rate. What it cost outside the family was one export dependency (`snac` is its
own MIT package, imported lazily), two class-level patches — `Snake1d`, whose `@torch.jit.script` body
reshapes through `x.shape[i]` and does not convert, and `NoiseBlock` — and **the family's first
stochastic leaf**.

That last one is the engine-adjacent part, and it needed no engine change either: a topology is a pure
graph and cannot draw, but the Lua bridge has had `loom.seed_rng`/`loom.gaussian_array` since VITS, so
the noise became four graph INPUTS the synthesized driver draws at `multiple * n_codes`. `DriverInputs`
gains a `NOISE` binding kind beside `POSITION` and `MASK` — named by the export rather than by input
name, because the length ratio is not recoverable from a name — and any future family with a
stochastic leaf gets it free. The driver also accepts the arrays from the caller, which is what keeps
the oracle exact on a stochastic model.

**The noise was dropped first, on measurements, and put back after a listening test** — the export
shipped the conditional mean until a listener called it *"less sharp, slightly more artificial"*.
ADR-029 carries both halves and [Retro-043](../retros/retro-043-the-band-was-20db-down-and-audible.md)
the lesson: a band 21 dB down is not an inaudible band.

It also paid for itself outside family 11 entirely. The first export matched at cosine 0.999998 and
was wrong: coremltools lowers `reciprocal` to an op whose epsilon defaults to 1e-4, and Snake's
`1/alpha` had folded to `1/(alpha + 1e-4)`. That is fixed in `torch_patches.py` for every model this
pipeline will ever trace — [Retro-042](../retros/retro-042-a-converters-op-default-changed-the-function.md),
which also records the f32-vs-f64 arm that told a defect from float noise.

Verified against the package's own decode on real speech, both sides given the same noise: max |Δ|
**1.20e-06**, cosine 1.000000, the exact sample count (`n_rows * 4 * 512`), and the ASR oracle reading
22/22 words. Sabotage arm — the same graph against a reversed-code reference — 1.07. Three further
checks the stochastic half needs: a different draw moves the waveform (8.4% relative RMS, so the noise
is load-bearing), two runs at the default seed are bit-identical, and a named seed differs from the
default.

### Family 4, and the attribute nobody was reading

Family 4 — a convolutional feature encoder over the RAW waveform, a transformer, and one linear CTC
head — is wav2vec 2.0, HuBERT, data2vec-audio and everything fine-tuned from them. The roadmap's
estimate was "family-1-shaped once the encoder template generalizes past NeMo, and it needs no new
head at all" (`EXPORT-ROADMAP.md` ordering note 6). **The head half was exactly right and the
generalization half did not happen**, which is worth stating because it is the second time a family's
cost was misplaced by the same kind of reasoning.

The head is free. `CtcGreedyBuilder` is reused verbatim, `loom.argmax_rows` already existed, and the
synthesized driver is family 1's five statements with a different blank id. The *encoder template* is
not shared at all: family 1's `build_trace` is a NeMo-shaped `(input_signal, input_signal_length)`
pair around a mel front end, and this family has neither a mel front end nor a length argument. So
`ctc_asr_export.py` is a sibling template rather than a leaf of `nemo_asr_export.py`, and what the two
genuinely share is the epilogue — which is the honest reading of "family-1-shaped".

Four things are this family's own.

* **The waveform normalization is part of the model and is NOT in the checkpoint's `forward`.** Every
  member ships `do_normalize: true` in its `preprocessor_config.json`, and `Wav2Vec2FeatureExtractor`
  applies `(x - x.mean()) / sqrt(x.var() + 1e-7)` before the model sees a sample. Family 1's standing
  rule is that the front end is INSIDE the graph — it is what lets a host hand the engine a waveform
  and nothing else — so the wrapper does it, with the same population variance and the same epsilon.
  Omitting it neither raises nor changes a shape: it feeds a correctly-shaped graph audio at the wrong
  scale, and a checkpoint trained on normalized input transcribes plausible nonsense from it.
* **The blank is the tokenizer's `pad_token` and its id cannot be derived from the class count.**
  NeMo's convention is "last class"; HF's is row 0. They disagree at both ends, and `pad_token` is not
  always spelled `<pad>` — see [ADR-033](../adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md).
* **The attention mask is omitted, and that is what keeps the length dynamic.** This is
  [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) one modality over: a family whose
  door hands the model exactly the samples the caller recorded has no padding to describe, and
  `_get_feature_vector_attention_mask` builds its frame count from a Python-level `.shape[1]` that a
  trace bakes. `attn_implementation="eager"` for family 12's reason as well.
* **One engine reader, `CtcVocab`** — the CTC character table, decode-only, `tokenizer.ggml.model ==
  "ctc"`. ADR-033 is why it is a tag of its own rather than a `"t5"` file with the delimiter rewritten,
  and why a vocabulary reader is not the kind of engine work the acceptance criterion is about.

**And the cost that was in none of the scoping: the exporter had been mislabelling grouped
convolutions as depthwise since the first export.** `groups > 1` is not "depthwise" — depthwise is
one input channel per output channel — and the two coincide at both ends of the range, so eight
families' worth of dense and genuinely-depthwise convolutions never separated them. A
`Wav2Vec2PositionalConvEmbedding` is `groups=16` over 768 channels, the first model in the zoo that
sits in the middle, and it aborted the engine inside `ggml_im2col` naming neither the op nor the
model. `CONV_1D` honours `groups` now (G slices re-entering the same op, one `ggml_concat`), `CONV_1D_DW`
rejects a kernel that is not depthwise, and the exporter reads the kernel's own shape. The second half
of that fix is the sharper lesson —
[Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md).

Verified against `transformers` on the LOGITS rather than the transcript, over 11 s of real speech
(549 frames), on **three structurally different checkpoints**:

| checkpoint | class | max \|Δ\| | cosine | argmax |
|---|---|---|---|---|
| `data2vec-audio-base-960h` | `Data2VecAudioForCTC` | 1.87e-03 | 0.999999821 | 549/549 |
| `hubert-large-ls960-ft` | `HubertForCTC` | 1.73e-03 | 0.999999881 | 549/549 |
| `omniASR-CTC-300M-v2` | `Wav2Vec2ForCTC` | 5.45e-04 | 1.000000000 | 549/549 |

Sabotage arm — one checkpoint's engine output against another's reference, same shape — 32.7. The
deltas are larger than family 12's 1e-05 and the f64 arm is what says why rather than assuming: torch's
own f32 is **7.0e-04** from the same model at f64 and loom is **1.2e-03**, the same order, over a graph
that is seven strided convolutions and 12-24 transformer layers deep across 176,000 samples.

The three differ in the three places a branch could have been needed and was not: HuBERT's
`feature_projection` returns a bare tensor where the other two return a pair, data2vec's positional
convolution is five stacked kernel-19 layers where the others have one kernel-128 layer with an
odd-padding trim, and omniASR carries a 10,288-piece multilingual vocabulary with a literal space as
its word delimiter against the other two's 32 characters and `|`.

**`omniASR-CTC-300M-v2` is verified and NOT shippable, and the reason is the checkpoint.** It is
scale-sensitive: its own documented `AutoProcessor` path (which normalizes) transcribes `م` for
English speech, while the same audio un-normalized transcribes correctly. loom reproduces `transformers`
exactly on it — including on the broken arm, which is what the 5.45e-04 above measures — so the export
is right and the checkpoint's declared front end disagrees with its weights. Kept as the family's third
structural witness; not published.

### Family 5, where the estimate was wrong the same way twice and the cost was somewhere else

Family 5 — a kaldi-fbank front end, a low-frame-rate stack, an SANM encoder (self-attention with a
depthwise FSMN memory block beside it) and one linear CTC head — is SenseVoice, Paraformer and the
FunASR checkpoints around them. **Its first leaf, `SenseVoiceSmall`, shipped 2026-09-15**:
`sanm_asr_export.py`, one recognizer claiming a FunASR directory whose `config.yaml` declares
`model: SenseVoiceSmall`.

The roadmap scoped this family with the same sentence it scoped family 4 with, and the correction
family 4 wrote down applies again unchanged: **the CTC epilogue is free and the encoder template is not
shared**. `CtcGreedyBuilder`, `loom.argmax_rows` and the five-statement driver are family 1's, reused;
`blank_id = 0` costs nothing because `ctc_blank_id` has been a parameter since family 4. Three siblings
now share one epilogue and no trace.

**What was NOT in any estimate is that the front end had to be rebuilt.** FunASR's `WavFrontend` calls
`torchaudio.compliance.kaldi.fbank` and then `apply_lfr`, and neither traces: the first frames with
`as_strided` over strides computed from `.shape[0]`, the second pads the frame sequence at the end by an
amount that depends on the frame count modulo the LFR stride. Both are rebuilt in ops whose shapes are
derivable, and the rebuild is exact rather than approximate because **kaldi's per-frame work is all
linear** — DC removal, pre-emphasis, windowing and zero-padding compose into one constant matrix, so
the framing becomes a single strided convolution whose kernel that matrix is. The real DFT is two
matmuls (`|X|² = (Cx)² + (Sx)²`), which is how the complex intermediate that blocks `torch.stft` one
family over never has to exist. The LFR stacking is `lfr_m` depthwise convolutions and a concatenation.
Verified against `torchaudio` at 218 lengths covering every residue of the frame count, worst relative
error 6e-6, zero shape mismatches — and at f64 the two agree to 3.7e-07, which is what says the f32 gap
is accumulation.

**The reference front end is stochastic by default, and that had to be found before anything could be
graded.** `WavFrontend`'s `dither` defaults to kaldi's `1.0`, so FunASR's own pipeline adds Gaussian
noise to the waveform before every fbank and does not transcribe a file the same way twice at the bit
level. The exported model is the `dither=0` model; the oracle forces it to 0 on the reference side.
This is [Retro-032](../retros/retro-032-one-seed-is-not-an-asr-oracle.md)'s rule arriving one family
later and one stage earlier — the randomness is in the FEATURES, not the sampler.

**Two of the four prompt rows are knobs, so the prompt is a graph input.** `SenseVoiceSmall.inference`
prepends four rows of a 16-entry embedding table to the features: a language id, two fixed
event/emotion queries, and a text-normalization id. Text normalization is not cosmetic — `withitn`
returns *"And so my fellow Americans ask not what your country can do for you, ask what you can do for
your country."* where `woitn` returns the same words lowercase and unpunctuated — so baking it would
ship a model that can never punctuate. `prompt_ids` is therefore the graph's second input, after the
waveform so the root-axis expression still reads the waveform's own length, and the driver DEFAULTS it,
which is the family's one new driver component: a `DEFAULTED` binding emitting
`inputs.prompt_ids or {0, 1, 2, 15}`. A caller who wants `transcribe(audio)` never learns the input is
there.

**The languages are published, but not under Whisper's keys, and resisting that was a decision.**
`language` is a recurring ASR role, so `loom.asr.language_names`/`loom.asr.language_ids` are the obvious
place — except those are read into `AsrDecodeTable`, whose ids are decoder prompt TOKENS pushed into a
cross-attention prompt, and this family's ids index an embedding table prepended to the features. Same
concept for a caller, different object for the engine. The mechanism-free half goes in
`loom.text.languages` (which `ModelContract` already reads and the engine already uses to refuse a
language a file cannot serve); the name→row tables go under a `sanm.` prefix. Today the misuse would
have been inert — that path is gated on a declared clip length and this file declares none — which is
exactly the kind of accident that stops being inert later.

**The vocabulary is free and was not expected to be.** The CTC head's 25,055 rows are exactly the
25,055 pieces of the checkpoint's own SentencePiece BPE protobuf in id order, with no `tokenizer.json`
beside it — so [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md)'s seam
sends it down the same `sentencepiece_proto` path family 1's NeMo checkpoints take, byte for byte.

**What it cost the EXPORTER is four entries in the shape walk, and that is the finding worth carrying.**
Every one of them was the same failure — the walk met a producer it did not know, fell back to the root
axis, and the topology declared one row per audio SAMPLE where it should have had one per encoder frame
— and two of the four were the exporter's OWN dialect ops, introduced by its own lowering passes and
reachable by every model since those passes were written.
[Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md) is the account;
[Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md) is the
same lesson three days earlier, and the part it got wrong is that the risk grows with the zoo.

Verified against FunASR on the LOGITS at three lengths — 187 frames of `samples/jfk.wav` (max |Δ|
3.0e-04, cosine 1.000000000, 187/187 argmax) and both 96-frame halves of it (8.6e-05 and 2.3e-05,
96/96 each) — with a sabotage arm at max |Δ| 30.7, cosine 0.954 and **68 of 96 argmaxes still
agreeing**, which is this zoo's standing reason to grade the tensor rather than the tokens. The f64 arm
inverts the naive reading of the headline number: FunASR's own f32 path is **3.4e-04** from its f64
self, while loom is **5.7e-05** from it — the export is six times closer to the truth than the
reference it is being compared against, and the 3.0e-04 is almost entirely the reference's own error.

#### The second leaf was Paraformer, and the length came from the VALUES (2026-09-16)

`paraformer-zh` shares the SANM encoder and inherits the front end and the position encoding
unchanged, which is the part of "family 5" that is genuinely a family. What it adds is the first model
in this zoo whose **output length depends on the values rather than on any shape**: a continuous
integrate-and-fire predictor emits a token each time a running sum of per-frame `alpha`s crosses an
integer.

**The split is the design.** The host decides the boundary — it is the only party that can, for the
reason below — and the graph does arithmetic with no threshold in it at all. Concretely the host hands
over the whole `(n_tokens, n_frames)` **linear resampling matrix**, because every token is a weighted
sum of encoder frames whose weights depend on `alphas` alone, and the graph is then one matmul. That
form was forced as well as preferred: FunASR differences two cumulative sums, which would need a
cumulative sum along the graph's slow axis (`ggml_cumsum` only sums over `ne[0]`) and a row gather from
the permuted result (`ggml_get_rows` needs a contiguous source — the constraint the first leaf hit in
`ggml_repeat`). It is also *more accurate* than the reference, since differencing two cumulative sums
is catastrophic cancellation by construction: against an f64 evaluation of the same algebra FunASR's
own f32 result is 7.18e-07 away and loom's is **4.43e-08**.

**Why the host and not the graph**, and it is not performance.
[Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md) is the account. FunASR
accumulates at float64, **casts to float32**, then floors — and computes the remainder as
`(indicator + prefix_sum) - floor(prefix_sum)` in float32, left to right. Both halves decide frames: the
crossings sit within 1.9e-06 of an integer, so a pure-float64 host is *more accurate and wrong*, and at
`prefix_sum ≈ 32` the float32 spacing is 3.8e-06, so `1 + 31.999998` rounds to exactly 33 and a
remainder collapses from ~1 to 0. A graph has no float64 and its own f32 cumsum lands elsewhere. The
driver reproduces the recipe in Lua doubles with an explicit `to_f32`, and **the threshold is crossed in
exactly one place** — the first design had the graph re-derive the remainders and it disagreed with the
host's indices on precisely the frames that matter.

It needed **no new engine binding**: `OutputRef`, `loom.get_output` and `loom.output_shape` already
existed, the last of them for a transducer's decode loop asking the same question. It reuses family
12's `TokenLabelsEpilogue` — a non-autoregressive decoder emits one token per row and must *not*
collapse duplicates, which Chinese produces legitimately — and adds one driver component,
`cif_boundary`.

Verified against FunASR on the encoder TENSOR (max |Δ| 4.48e-05 / 4.81e-06 on 13 s of Chinese and 11 s
of English, cosine ≈ 1, sabotage arm 3.81e-01 at cosine 0.168) and on the transcript, which is
**character-for-character identical on both clips**.

**Its detokenization is a vocabulary scheme of its own, and that was the second reading rather than the
first.** `tokens.json` is 8,404 flat decode-only pieces — the same shape family 4's table has — but
concatenating them is not text:

    ['and','so','my','f@@','el@@','low']  ->  "and so my fellow"   @@ continues into the NEXT piece
    ['hello','你','好']                    ->  "hello你好"           the space is REMOVED before CJK
    ['b','b','c','news']                  ->  "BBC news"           letter runs collapse AND UPPERCASE
    ['<s>','and','</s>']                  ->  "and"                control pieces drop

This was first written up as a per-TASK postprocess for the `transcribe` door, on the grounds that the
CJK rule needs lookahead and the abbreviation rule is a transform. That was wrong, and
[ADR-036](../adrs/adr-036-composition-is-the-scheme-not-the-table.md) records why: `decode` has never
been a lookup for any family here — SentencePiece's rewrites U+2581, WordPiece's strips `##` — and
`model.detokenize(ids)` has to answer something, where `"andsomyf@@el@@low"` is not an answer. So it is
`loom::FunasrVocab` under `tokenizer.ggml.model == "funasr"`, which cost the engine one class and one
KV.

It could not have been folded into an existing tag: `@@` is a SUFFIX meaning "I continue" where
U+2581 and `##` are PREFIXES meaning "a word starts here", they are duals, and the same piece string
occurs in both roles. And the per-piece SCRIPT is computed by the EXPORTER, because the reference's
tests are per-character against Python's Unicode `isalpha()` and this vocabulary holds one character
(U+2B5AF) that is alphabetic and outside the CJK block a C++ range check would use — ADR-027's
principle one family over.

**Verified differentially, which is what found the defects**: 20,000 random id sequences from the real
vocabulary, engine against `sentence_postprocess`, **0 mismatches**. Three things it caught that no
example would have — `f@@` is neither CJK nor Latin and is *still* a continuation (the marker test is
orthogonal to the script, and a first version nested it inside the Latin branch); `9@@` IS CJK by the
reference's own test, since digits and `@` count, so it is *not* a continuation; and a `<blank>` the
exporter was dropping where the reference prints it, which no realistic input could expose because a
non-autoregressive decoder cannot emit one.

### Family 6, and the primitive the engine already had



Family 6 — text in, text out, through an encoder read once and a KV-cached decoder cross-attending to
it — is structurally family 2's and family 10's shape, so the `encoder`/`cross_kv`/`decoder` split
came for free. Dia is the closer precedent of the two: its encoder length is genuinely dynamic, so its
decoder carries a second symbol, and T5's does for the same reason.

**What made it look expensive was one paragraph of scoping, and it was wrong.** T5 has no positional
embedding at all: every attention score is offset by a learned value chosen by the *bucketed* distance
between the query and the key. The backlog entry read that mechanism accurately, concluded correctly
that the engine has no primitive for it, and named three ways out — a new primitive, a Lua
computation, or a per-length constant fold.

None was needed. By the time the bias reaches attention, `transformers` has already summed it with the
causal mask into ONE `[1, n_head, q, k]` additive tensor, in exactly the place `fuse_loom_attention`
anchors on — and `ggml_soft_max_ext` has always accepted a mask with a head axis, since it asks only
that `a->ne[2] % mask->ne[2] == 0`. Nothing here had used that degree of freedom because every mask
this tree ever built was one head deep. **T5's relative bias is a mask this engine could already
take**, and the only open question was who computes it: the driver does, from a 192-float table per
stack that the export writes in as constants
([ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md)).

So this family, like 11 and 12, needed **no new engine primitive** — but unlike them it was *scoped*
as needing one, which is the lesson
([Retro-040](../retros/retro-040-the-blocker-was-scoped-from-the-mechanism.md)): a capability gap is a
claim about an interface, and it was costed entirely from the producing side.

Two things did cost real work, and neither was the bias:

* **The fusion matched zero blocks.** `merge_consecutive_transposes` merges T5's head-split and score
  transposes into a permutation MIL can no longer fold into `transpose_y`, because — unlike Qwen3 and
  Whisper — T5 has no RoPE between them. An unfused block still runs and still answers, just without a
  cache, so nothing failed; a `topology_rewrite` that counts is what said so
  ([Retro-041](../retros/retro-041-two-transposes-merged-and-the-fusion-went-quiet.md)).
* **`inner_dim` is not `d_model`.** T5 sizes its heads (`num_heads * d_kv` = 384) independently of its
  residual stream (512), and this is the first model here where the two differ. The engine's own
  retained-output-vs-input shape check caught it at the first decode step.

Verified as a tensor rather than as a token sequence: greedy generation is id-for-id identical to
`transformers` on four prompts of 11–43 source tokens, and the encoder matches at 3.1e-7 max absolute
error on a tensor whose max magnitude is 0.481. The second is the check that covers the bias; the
first would pass with a slightly wrong one.

### Family 10, and the axis that did not have to be padded

**It ships as two files, not one** — the LM and the codec, chained by the host, with a frame-major
array of integers between them. One codec serves ~20 autoregressive LMs and the codes are worth
having on their own, so merging would ship the same 217 MB inside every model and make the useful
intermediate unreachable.
[ADR-022](../adrs/adr-022-dia-and-its-codec-stay-two-files.md) has the decision; what it costs is a
test of the JOIN, which neither file's own suite covers —
`tests/gate/test_e2e_dia_dac_composition.cpp` drives both real GGUFs and compares codes and waveform
against `transformers` running the identical pipeline, at two clip lengths.

Family 10 — an AR LM that emits **codec tokens** rather than text — is the other half of family 11,
and the pair is the whole point: text → Dia → nine delayed code streams → realign → DAC → waveform.
Dia is the leaf because its own `audio_tokenizer_config.json` names `descript/dac_44khz`, the codec
already exported and verified, so it costs the LM half only.

Structurally it is family 2's shape — encoder once, then a KV-cached decoder cross-attending to its
output — and `whisper_export`'s three-phase split (`encoder`, `cross_kv`, `decoder`) transfers
unchanged. **Three things do not, and each is the reason for code that has no Whisper counterpart.**

* **Two dynamic axes, not one.** Whisper's encoder always emits 1500 frames; Dia's emits one per input
  BYTE, so the cross-attention K/V carry a second independent symbol. Received wisdom said a topology
  could resolve only one — [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md)
  says so about Supertonic — and that turned out to be a statement about *one model's trace*, not
  about the machinery: `declared_axes` and the engine's axis map both already supported it. So Dia's
  text axis is fully dynamic, with no padding, no buckets and no mask input.
  [ADR-021](../adrs/adr-021-dias-decoder-resolves-two-dynamic-axes.md) has the argument and the
  alternatives.
* **Nine output heads, and still no engine primitive.** Every decode loop in this tree reduces one row
  to one token; this one emits nine per step. The wrapper slices `hidden[:, -1:, :]` before the head,
  so the graph's output is `[vocab, 9]` on a prefill and on a decode step alike — which is exactly the
  `[n_classes, n_rows]` tensor `loom.argmax_rows` was built for in P4.0.17 and family 12 reused. The
  driver takes it a step further and restricts the draw per channel, because
  `DiaEOSChannelFilterLogitsProcessor` bans the control ids per channel rather than globally — the same
  restricted reduction `whisper_driver` detects a language with. **Three families running, and none has
  needed engine C++ for its graph**, which is the acceptance criterion the roadmap states — though its
  *sampler* did, and §"What its sampler cost" below is that bill.
* **The delay pattern lives in the driver**, by [ADR-020](../adrs/adr-020-audio-codes-is-its-own-modality.md)'s
  reasoning and [ADR-013](../adrs/adr-013-one-door-per-task.md) §2's: it is declared in `config.json`,
  so it is read rather than derived, and undoing it is index arithmetic over a nine-element array. The
  engine never learns what a delay pattern is. Two halves of it are in Lua — a scaffold on the way in
  (channel k is forced to BOS until step `delay[k]` has passed) and a gather on the way out (audio
  frame t's channel k was emitted at row `t + delay[k]`).

The tracing lesson is [Retro-030](../retros/retro-030-a-guard-that-could-not-fire.md), and it is worth
reading before the next family: under `torch.jit.trace` *every* shape read is a 0-d Tensor, the static
ones included, so a guard testing `isinstance(dim, int)` detects tracing rather than staticness.
`rotate_half` is fixed with `torch.chunk`, which asks for a count instead of an index and therefore
needs no arithmetic over the axis at all.

#### What its sampler cost

The graph needed no engine C++. **The sampler did**, and that is the honest form of the acceptance
criterion: the checkpoint declares `do_sample: true` at `temperature 1.8 / top_k 50 / top_p 0.9` with
`guidance_scale 3.0`, so a greedy, guidance-free export is a file whose default output is not the
model anyone published. Two things were added, both per-*task* rather than per-model by
[ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md):

* **`loom.sample_row` gained `lo`/`hi` and `guidance`** — a restricted draw, and classifier-free
  guidance over two modules' retained logits — as entries in its options table rather than as new
  bindings. [ADR-024](../adrs/adr-024-guidance-belongs-in-the-sampler.md) says why that differs from
  the `argmax_row`/`argmax_row_range` pair, and why the greedy-equals-argmax invariant came out of it
  stronger.
* **A topology can declare `kv_cache_scope: "private"`** and `ExportPhase.extra_streams` emits it.
  Guidance runs the decoder twice per step over two histories, and this engine's cache is
  single-sequence, so the second run is a second module rather than a second batch row.
  [ADR-023](../adrs/adr-023-a-second-stream-is-declared-not-derived.md).

**Dia's guidance is not the standard formula**, in two ways, and neither is visible from the name of
the technique — [Retro-031](../retros/retro-031-dias-guidance-is-not-the-standard-formula.md) is the
finding and the rule it produced. The engine implements the general form; the model's centring is one
`+ 1` in its own driver.

**What the gate can and cannot compare.** Exact integer equality needs a deterministic algorithm, so
the oracles are greedy — but *guidance* is deterministic too, so there are two of them: greedy with
guidance off (32 frames) and greedy with guidance on at the checkpoint's own 3.0. The second is what
grades the two streams, the shortlist and the channel filter. Sampling itself has no exact oracle in
either direction and is not claimed to reproduce `transformers`' draw.

**And it has been listened to.** Under the checkpoint's own sampling and guidance, a 3.02 s utterance
transcribes back at 9/9 words through `whisper-small` — but only at some seeds: at four tried, one was
perfect, one was laughter and two were near-silence, and `transformers` behaves the same way. That is
the model, not the export, and establishing it took two deterministic bisections rather than a guess —
[Retro-032](../retros/retro-032-one-seed-is-not-an-asr-oracle.md), which is
[Retro-006](../retros/retro-006-kokoro-shipped-noise.md)'s converse and the rule for grading any
sampling family. The model card has to name a seed.

**It publishes at F32, like every other model in the catalogue, and that was decided twice.** At 6.4 GB
it is by far the largest artifact here, and `--quantize Q8_0` packs 253 of its 344 tensors — 99% of the
float weight bytes — down to 1.8 GB, verified working: the ASR oracle passes on the quantized file at
the card's own sentence and seed. It shipped that way briefly and was reverted, because **consistency
across the collection is worth more than one model's download size**. These artifacts are reference
exports as much as downloads; the sweep snapshots them byte-for-byte, and one lossy member among
seventeen faithful ones is a difference the sweep is not comparing and nothing in the file announces.
The card names the `--quantize` invocation instead, which puts the trade-off where it belongs — with
whoever is short of disk.

**The quantizer bug that surfaced underneath it stands regardless.** The eligible weights are derived
from the topologies rather than from tensor names — and the standalone
`tools/quantize/quantize_gguf_q8_0.py` read only `model.graph_topology`, singular, so it silently
quantized *nothing* on every multi-phase family (Whisper's three topologies, Dia's five) while printing
a success line. It unions them now, and the two paths agree byte-for-byte on Dia, which is the
cross-check that they implement one rule.

See [the backlog](../backlog/active-index.md#models) for what is left.

#### The second leaf was Qwen3-TTS, and an op had to be REMOVED rather than converted (2026-09-13)

`Qwen3-TTS-12Hz-0.6B-Base` is family 10's second leaf and ships as **two files** for ADR-022's reason,
exactly as Dia does — a 914 M talker plus a 114 M codec at 12.5 Hz, where the codec is family 11's
fourth shape (the first with ATTENTION over the frame axis, hence the first CHUNKED decode:
[ADR-034](../adrs/adr-034-a-chunked-decode-is-the-drivers-loop-not-a-longer-call.md), which
`encodec_export` had predicted in as many words — "a chunked one is a different driver, not a longer
call"). `loom-export` on the checkpoint root emits both, through the new `LoomExportConfig.companions()`
hook; the CLI opts in and `main_export()` does not, because `build_model_cards.py` calls the API once
per Hub repo.

**One audio frame is sixteen transformer forwards, not one.** A 28-layer Qwen3-shaped talker emits
codebook 0 and a 5-layer **code predictor** emits the other 15 from the talker's hidden state, its KV
cache reset per frame. The input embedding is a SUM of 16 codebook embeddings plus a text hidden,
never a token lookup. The `mrope` in its config is **decorative** — `get_rope_index` always expands one
row to three identical ones, so `apply_interleaved_rope` collapses to plain RoPE at θ=1e6, and the
scariest-looking thing in the config costs nothing.

Three things cost real work, and the order they were found in is the lesson.

* **`repeat_kv` does not survive conversion, and the fix was to delete the op.** coremltools folds its
  expand (a broadcast) before `passes.fuse_gqa_repeat_kv` can match it, and rewriting it as
  `repeat_interleave` only moves the failure into the merge reshape, which comes back as
  `[128, n_tokens, n_tokens, 8]` — [Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)'s
  substitution one family later. So `materialise_gqa` duplicates `k_proj`/`v_proj` interleaved until
  K/V heads equal query heads: **+69.2 M parameters, 277 MB at F32**, checked against the written
  file's own growth, and a doubled cache. The KV-geometry error this first presented as (28 blocks
  reporting 16 K/V heads against the predictor's 5 reporting 8) was a symptom; the census the error
  now prints is what made it legible as one.
* **A greedy decode without a repetition penalty never terminates.** `transformers` applies the penalty
  as a **processor** rather than a warper, so it moves a greedy argmax too: without it the decode ran
  200 frames against the reference's 42. `loom.sample_row` gained `repetition_penalty` + `penalized`,
  and with them the decomposition reproduces the reference **bit-identically at 672 codes**.
* **[Retro-047](../retros/retro-047-an-inferred-dimension-outlives-the-reshape.md) is the genuinely new
  failure**: an inferred `-1` reaches the topology as a literal and the next op derives `floor(1/n)`
  from it. Export, write and load all pass; only running fails.

The codec half verifies at max abs **4.167e-06** at 42 frames and **1.699e-05** at 700 against the
reference's own `chunked_decode`, exact sample count at both, with the ASR oracle reading the decode
back verbatim. It cost one driver component (`ChunkedCodecCall`), no engine change and no new binding.

#### ICL landed 2026-09-17, and the bill it was scoped with had already been paid

`spk_id` is empty in this checkpoint, so voice cloning is the only mode and its two arms are nested
rather than alternative. `x_vector_only_mode` shipped first; **ICL — the reference clip replayed, its
transcript on the text stream and its own codec frames on the codec stream — shipped 2026-09-17**, and
what it cost is not what the roadmap said it would.

**The sampler's half of the estimate was already paid, by the work that opened the item.** ICL was
filed as needing a non-contiguous allowed set, because `suppress_tokens` bans `[2048, 3072)` except
`codec_eos = 2150` and `lo`/`hi` cannot express a hole. But `_TalkerWrapper` had already **trimmed the
head** to 2048 real codes plus one EOS row — for `min_new_tokens`, on the x-vector path — so the hole
does not exist in the drawable space at all, and the driver has carried temperature, top-k, top-p and
the repetition penalty since the talker landed. **No engine change of any kind**: the file runs on the
rc10 wheels.

**The prompt's branch is the host's.** The reference picks the shorter of the text and codec streams
and slices the other by its length; a graph can carry neither the branch nor the slice. Both lengths
are known to the driver before the call — one is a token count, the other a frame count — so it hands
the graph two already-sized id arrays, and `bos_mask` selects `codec_bos` over a zero row rather than
concatenating one row under a second symbol. Verified against `generate_icl_prompt` on **both arms** of
its branch: max |Δ| **1.19e-07** on a tensor of magnitude 7.5.

**The encoder is the real cost, and it lives here rather than in the codec's file** —
[ADR-038](../adrs/adr-038-the-codecs-encoder-ships-inside-the-talker.md): the encode direction has no
task, so a separate GGUF would need a new door in the high-level API for a model whose only caller is
three phases away. It is four phases and 190 MB (4.8%): the SEANet stack, an eight-layer transformer,
the downsampling convolution, and one `rvq_step` topology called sixteen times. Three things made it
tractable and each is reusable:

* **The RVQ needs no `argmin`.** `argmin_j ||x - e_j||²` is `argmax_j (2 x·e_j - ||e_j||²)`, which is a
  matmul and a constant row — and `argmax` over rows is a driver binding, so the loop is the driver's:
  scores out, ids in, one call per stage. All sixteen codebooks are written once and a stage gathers
  its own with a range, the device that makes the code predictor one topology instead of fifteen.
* **The convolution padding is constant on frame boundaries.** Mimi pads by a length-derived amount
  that coremltools refuses; trimmed to a multiple of 1920 samples, that amount is **zero at all
  fifteen convolutions** — proved from the real modules, with a partial frame as the arm where it is
  non-zero at three of them. The driver trims, and the cost is at most 79 ms off a reference clip.
* **The mask is causal and the config says otherwise**
  ([Retro-050](../retros/retro-050-the-config-declared-a-window-the-reference-never-applied.md)).

Verified end to end against the reference's own greedy ICL generation, from **raw audio** — the driver
drawing the reference codes itself — on two sentences and two clip lengths: **624 of 624 ids identical
over 39 frames each**, and the same run given pre-computed codes agrees with it exactly. The x-vector
path is unchanged and byte-identical to the published GGUF (640/640).

### Family 9's third leaf, and the primitive that had to be added

Family 9 is the **flow-matching acoustic stage** — Matcha-TTS and SupertonicTTS since the MIL thread,
both integrating a learned vector field with plain uniform Euler. F5-TTS is its third leaf (P5,
2026-09-18) and the first one whose *sampler* is not that: it runs the estimator **twice per step**
under classifier-free guidance, on a **non-uniform** schedule.

**It is also the first P5 family in seven that needed an engine change**, which is worth saying plainly
because the acceptance criterion had held six times running. The change is small and it is the right
shape: `loom.run_ode` learned `guidance = {inputs = ..., scale = ...}` — one module, one graph, one
cache, evaluated a second time with a different fixed-input table and combined where `k[stage]` is
filled, so every integrator in the table keeps working unchanged. Guidance at scale 0 is bit-identical
to the unguided call, which is what protects the two models that already shipped through this binding.
[ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md) has the decision
and why guidance differs in `inputs` here where `loom.generate` differs in `module`.

On the export side it is **two declarations on the existing template rather than a bespoke sampler**:
`FlowMatchingSpec.guidance` and `FlowMatchingSpec.schedule` (`"caller"` — F5-TTS integrates
`t + coef*(cos(pi/2 t) - 1 + t)` over a linspace, not `k/n_steps`). The loop is unchanged, which is the
test of whether a template still fits.

**F5-TTS has no duration model and no phonemiser: it in-fills.** The reference clip's mel occupies the
first frames of one spectrogram, the rest is noise, the text is the reference transcript followed by
what to say, and the decoder integrates the whole grid at once. So there is exactly one dynamic axis
across the two text-and-estimator phases — the total frame count — and the driver slices the prompt's
frames off the answer before the vocoder sees them. Four phases: the mel front end (a `power=1`
spectrogram, the first un-squared complex magnitude in the zoo, which is how `reduce_l2_norm` was found
to have no ggml mapping — Whisper writes `abs()**2` and the square cancels the root; later MIL passes
decompose it again on this particular graph, so the mapping is verified directly rather than through
the model), the text embedding, the estimator, and Vocos.

Three things cost more than the scoping predicted, and none of them was the sampler:

* **The conditioning length is the mel's own frame count, not `ref_audio_len`.** `sample()` takes it
  from `cond.shape[1]` (`n_samples//hop + 1`) and only the output slice uses `n_samples//hop`. One
  frame — and 32 Euler steps turn it into max |Δ| 1.77 on the mel, cosine 0.9998, which reads exactly
  like accumulated float noise and is not.
* **A defensive re-slice in library code.** `apply_rotary_pos_emb`'s `freqs[:, -seq_len:, :]` is the
  identity once the table has been cut to length, and a negative begin over a dynamic axis exported as
  twice the rows at a negative offset, 44 times.
  [Retro-051](../retros/retro-051-a-negative-begin-doubled-the-slice.md).
* **The JOIN between two graphs, which every per-phase check passed over.** The estimator retains
  frame-major mel and `Vocos.decode`'s convention is channel-major, so the driver handed the vocoder a
  transposed spectrogram — which is still a plausible spectrogram, so nothing raised and the audio came
  out as a sound effect. Five clean tensor comparisons said nothing about it; the ASR oracle found it in
  one listen, and the sabotage arm measures it at cosine **−0.008**.
  [Retro-052](../retros/retro-052-every-phase-was-right-and-the-join-was-wrong.md).

**Verified end to end.** Per phase against torch on the real inputs: `mel` 4.39e-02 / cosine
0.999999881, `text_embed` 1.13e-05 (conditional) and 8.11e-06 (unconditional), `estimator` 1.19e-05 at
**cosine 1.000000000**, `vocoder` 1.53e-05. The torch decomposition of the whole sampler reproduces the
reference's integrated mel at **1.42e-05 / cosine 0.99999994** over the generated frames. And the
exported GGUF, driven by `loom_cli` from a reference clip, synthesises audio at peak 0.9768 (reference
0.8557) that the Whisper ASR oracle transcribes as the target sentence exactly — the check that found
the layout defect, and the one that closes it. The gate compares the WAVEFORM against the reference's
own at **max |Δ| 4.14e-03, rmse 1.94e-04** over 92,416 samples.

**That gate is only a comparison because the NOISE is pinned, and finding that out cost a red run.**
Flow matching starts from a Gaussian draw; torch's RNG and the engine's are different algorithms, so
handing both sides the same *seed* hands them different *noise*, and a different draw is a different
valid sample — 1.25 max |Δ| on audio that was intelligible and correctly voiced. So
`FlowMatchingSpec.caller_noise` makes the initial state a driver input that the engine falls back to
drawing, `scripts/f5_tts_reference.py` writes the draw it used, and the gate hands it over. This is
`DriverInputs`'s own NOISE argument one layer up, and the reason it is worth restating is that the
alternative — loosening the bound until 1.25 fits — would have produced a gate that measures nothing.
Sabotaged the other way, against a reference generated with guidance off, it reads 1.278: 300× the
bound, so the gate can fail.

**Its weights are `cc-by-nc-4.0` and that is decided rather than pending.** The F5-TTS *code* is MIT;
the released checkpoint is non-commercial because Emilia is, which the repository states and the Hub
card confirms. Same shape as EnCodec's licence, and the artifact declares its own — nothing about it
changes this project's own MIT terms.

**Its text front end is the family's real boundary, and it was measured rather than assumed.**
`convert_char_to_pinyin` is `rjieba` segmentation plus `pypinyin` before a single id is looked up. What
ships is the character table (`tokenizer.ggml.model == "f5"`, `loom::F5Vocab`), and against the real
function over generated English prose: **2000/2000 identical for ordinary space-separated prose**,
1100/2000 with multi-character punctuation runs and 1758/2000 with hyphen-joined digit groups — the two
divergent classes differing by exactly one inserted space, always in the same direction. CJK is refused
by name rather than mapped character by character. Same boundary the phoneme-input families draw around
g2p.

### Family 9's fourth leaf: the first AR-plus-flow composition in one file

Chatterbox (`ResembleAI/chatterbox`, English, MIT) is the shape EXPORT-ROADMAP's correction 4 names:
**an AR token LM and a flow-matching decoder as two stages of one pipeline.** T3 is a Llama-520M that
turns text into 25 Hz S3 speech tokens under classifier-free guidance. S3Gen embeds those tokens behind
the voice's own prompt tokens, runs a conformer, and in-fills a mel after the prompt's frames with a
guided 10-step ODE on a cosine schedule. HiFT (an NSF sine source plus iSTFTNet) vocodes the result.
Six phases, one GGUF, one driver.

**It needed no new template, and that was the test.** T3 is family 10's shape: a KV-cached decoder
whose unconditional twin is an `extra_streams` alias with a private cache
([ADR-023](../adrs/adr-023-a-second-stream-is-declared-not-derived.md)), driven by a hand-written loop
like Dia's and Qwen3-TTS's. S3Gen is F5-TTS's `FlowMatchingSpec` (guidance, caller schedule, caller
noise) with a Matcha-style estimator. What was new is only that both halves share one driver.

**What it cost, which was again not where the scoping looked:**

* **Two sampler changes.** `loom.sample_row` gained `min_p`. The repetition penalty turned out to be
  applied once per *occurrence*, where `transformers` applies it once per id. Even at Qwen3-TTS's
  1.05 this was the talker's open greedy divergence (96/624 ids on one sentence, 624/624 once fixed),
  and at Chatterbox's 1.2 it would have been worse
  ([Retro-053](../retros/retro-053-the-repetition-penalty-compounded-per-occurrence.md)).
* **A text front end whose rules are data** ([ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)):
  a character-level BPE rather than a byte-level one, plus the reference's `punc_norm`, as
  `tokenizer.ggml.model == "chatterbox"`. **3000/3000** ids identical to the reference across six input
  classes.
* **The vocoder's two random draws are inputs.** The NSF source draws a phase per harmonic and a
  Gaussian per harmonic per output sample. Both are pinned the way F5's ODE noise is. Kokoro had
  already found the `% 1` trap (`remainder(x, 1)` lowers to `sub(x, x)`), and `f0 > 10` became an exact
  clamp because the exporter maps no `greater`.
* **A fixture that was silently transposed**: a Fortran-ordered `.npy` read in C order
  ([Retro-054](../retros/retro-054-a-transposed-view-saved-fortran-ordered.md)).

**Verified end to end.** Every wrapper against the module it took over, in torch: the prefill is
bit-identical, the LM's first-step logits are 7.6e-06, the mel after the wrapper-driven ODE is 5.7e-06,
and the vocoder is 1.2e-06. The GGUF on the engine, from text, with guided greedy decoding and the
reference's draws, gives a waveform at **max |Δ| 2.5e-05, rmse 1.5e-06** over 40,320 samples. The
reference's own float32-vs-float64 spread is 2.2e-05, so that is the rounding floor. The Whisper oracle
transcribes it exactly, and so does the default sampled mode at two seeds. The sabotage arm (the flow's
guidance off) gives 0.897. The gate is `tests/gate/test_e2e_chatterbox_lua_driver.cpp`. It runs in
~50 s at 3.7 GB, and T3 and the vocoder together take 1.7 s of audio in ~49 s on the 2-core box.

**Shipped without Resemble's Perth watermark, deliberately** (2026-09-23). Perth is a separate neural
model applied after synthesis, and this build targets local inference and dev kits, where a smaller,
faster model is the point. The model card says so as its first limitation. One voice is built in
(`conds.pt`, as `driver_weights`); cloning needs the voice encoder, the S3 tokenizer and CAMPPlus.

### Family 9's fifth leaf: a loop that carries continuous latents

Pocket-TTS (`kyutai/pocket-tts`, English `english_2026-09`, CC-BY-4.0, ~110M parameters) is the loop
shape the hub had costed as the one nothing shipped: **an AR model whose step emits a continuous 32-d
latent rather than a token.** A 6-layer transformer runs over the previous latent. Its last row feeds
an EOS head and a one-step flow head (Lagrangian Self Distillation: `x0 + v(c, s=0, t=1, x0)` from one
Gaussian draw scaled by `sqrt(temperature)`), and the result is both the frame and the next input. Mimi
decodes the latents: a depthwise ×16 upsample, a 2-layer windowed transformer, and a SEANet. Five
phases (`text_embed`, `lm`, `step_embed`, `flow_head`, `mimi_decoder`), one GGUF, 405 MB at F32.

**The loop needed no primitive.** It is family 10's KV-cached decoder driven by a hand-written Lua
loop, with `loom.sample_row` replaced by one `flow_head` call and 32 floats read back. The two
products the reference does in f32 (`z * std`, `x + v / n`) are inside that graph, because the
driver's LuaJIT has only doubles.

**What it cost, and again it was not the loop:**

* **The voice is a KV cache**, precomputed by the reference and not recoverable as inputs. It needed
  the one new binding, `loom.seed_kv`
  ([ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md)). The built-in voice
  (`alba`, 126 rows) ships as a 6.2 MB driver weight, and a caller may seed any saved state.
* **The text front end chunks.** The reference splits text into sentence chunks of at most 50 tokens
  and generates each from a fresh voice. The vocabulary does that and returns the chunks separated by
  `</s>` ([ADR-044](../adrs/adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)). It
  also made `loom::Vocab` implement SentencePiece's byte fallback, which the writer had been dropping
  silently. **7000/7000** texts are identical to the reference's whole text path.
* **Three re-spellings for the trace**, each checked against the module it replaces: interleaved-pair
  RoPE in four dimensions (the reference uses a 5-D view), `in_proj` split into Q/K/V, and Mimi's
  window mask built from a `positions` input as two outer products, because ggml's SUB broadcasts only
  its second operand. The streaming Mimi is ONE call: its convolutions are causal and its attention
  windowed, and the reference script checks one-shot against streamed at 3.4e-06.
* **The gate had to change shape** ([Retro-055](../retros/retro-055-a-feedback-loop-cannot-be-gated-free-running.md)).
  A latent loop amplifies rounding, so free-running comparison measures the trajectory, not the
  implementation.

**Verified.** Teacher-forced against the reference with its pinned draws: **max |Δ| 1.5e-04, rmse
1.8e-06** over 155,520 samples (6.5 s); the sabotage arm (temperature 0.7 for the checkpoint's 0.3)
gives 0.82. Free-running from the same draws, it reaches the same EOS frame at rmse 9.9e-05. The
Whisper oracle transcribes both the reference and loom identically, and a three-chunk paragraph
correctly. The engine synthesises 17.4 s of audio in 10.9 s on the 2-core dev box. The gate is
`tests/gate/test_e2e_pocket_tts_lua_driver.cpp`.

**Not in this export:** cloning a voice from audio (the Mimi encoder, which the released
voice-cloning weights carry and the other release zeroes), and the other 25 predefined voices, which
are loadable as `voice_kv` but not shipped.

### Text input

**Supertonic, F5-TTS, Chatterbox and Pocket-TTS take text.** Each encodes graphemes itself and each GGUF carries its own
table (Chatterbox's is a character-level BPE, Pocket-TTS's a SentencePiece Unigram). The other four TTS models consume *phoneme* ids produced outside the engine — a real
limitation of those checkpoints, addressed by
[Epic-07](epic-07-text-frontends-and-tokenizers.md) and
[ADR-012](../adrs/adr-012-permissive-phonemizer.md).

## 3. Roadmap

Ordered by coverage-per-effort. Live items are tracked in
[the backlog](../backlog/active-index.md#models); the ordering and its reasoning are here.

**Next families:** the remaining TTS families → small classifiers → music. **Six are done** —
token classifiers (12), codec decoders (11, all four shapes), the AR codec-token LM (10), text
encoder-decoders (6), CNN + transformer + CTC (4) and, as of 2026-09-16, SANM / FunASR (5, on **both**
leaves: SenseVoice-Small and Paraformer-zh) — and family 9 is at **five of its twelve** leaves since
Pocket-TTS landed 2026-09-24, right after Chatterbox (F5-TTS, on 2026-09-18, is where the "no engine primitive" run ended:
[ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md)).
§2 says what each cost, which is the number the rest of this list should be estimated against.
Family 10 landing means the `text2codes` → `codes2speech` composition has both halves in the tree;
family 6 landing means the zoo has an encoder-decoder text model and a SentencePiece Unigram LM for
the first time.

**Family 4 was the correction to a standing estimate, and family 5 confirmed it.** "Family-1-shaped
once the encoder template generalizes past NeMo" was half right for both: the CTC *head* is free, and
the *encoder template* is not shared at all. Three sibling templates now share one epilogue and no
trace, which is the honest statement of what "family-1-shaped" buys. Family 5 also moved the cost
somewhere neither estimate looked — it needed no engine primitive and no new head, and what it did need
was a rebuilt kaldi front end and **four entries in the exporter's own shape walk**, two of them for
ops the exporter's own passes emit. §2 has the detail.

Family 12 is now proved on **three** checkpoints and both tokenizer halves — two WordPiece encoders
and one SentencePiece Unigram one — which is what closes the "a family-12 checkpoint that is not
WordPiece" item.

**Named but unstarted:** the Qwen3-ASR-0.6B variants beyond the exported leaf (the 1.7B and the
native-layout repo). Qwen3-TTS was the other name on this line and is no longer unstarted — it shipped
2026-09-13 in `x_vector_only_mode`, and what remains of it is ICL mode, which the hub carries as an
open item because its bill is a second family-11-scale export plus a sampler that can express a
non-contiguous allowed set.
F5-TTS is no longer deferred: it shipped 2026-09-18 as family 9's third leaf. The prediction it was
deferred under — "likely sharing primitives with Matcha" — held for the ODE and not for the sampler
around it; see §2.

**The constraint that decides what is exportable at all** is not the template — it is peak memory
during conversion. `MultiPhase.export` made peak memory a *sum* where it should be a *max*, and P5.0
is closed as of 2026-09-17 with all three changes in
([ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md)): dropping the traced module,
the wrapper and the converted MIL program **together** took Granite-Speech from 30.4 GB to 22.9 GB
peak RSS; each phase now packs its own weights as it converts, so what is carried between phases is
the on-disk payload rather than an F32 array; and `--isolate-phases` converts each phase in a child
process that spills its packed weights for the parent to memory-map.

**Of those, the one that moves the number is isolation.** Measured on Granite-Speech at Q8_0, packing
per phase left the peak where it was (20.99 → 21.11 GB) because the phase that sets the peak is the
40-layer decoder, which converts *last*, with almost nothing carried into it. Isolation took the
largest single process from **20.99 to 14.56 GB** and the whole process tree to **15.85 GB**. The ADR has the table, including the two results
that went the wrong way: isolation makes a small model's peak *worse* (kokoro 2.67 → 4.34 GB, the
framework floor paid twice), which is why it is off by default.

Voxtral-Mini-3B is still not exportable here and that is now a statement about the machine: its LM
phase needs ~14.4 GB of F32 weights beside ~14.4 GB of MIL constants whatever the export does around
it, so the floor is ~29 GB against 28. The measurements below are what a bigger machine picks it up
from.

## 4. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md), [ADR-005](../adrs/adr-005-export-config-and-task-registry.md), [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md), [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md), [ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md), [ADR-033](../adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md), [ADR-035](../adrs/adr-035-a-shared-role-is-not-a-shared-table.md), [ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md), [ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md), [ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md), [ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md), [ADR-044](../adrs/adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md), [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md), [Retro-039](../retros/retro-039-position-zero-was-not-row-zero.md), [Retro-040](../retros/retro-040-the-blocker-was-scoped-from-the-mechanism.md), [Retro-041](../retros/retro-041-two-transposes-merged-and-the-fusion-went-quiet.md), [Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md), [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md), [Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md), [Retro-051](../retros/retro-051-a-negative-begin-doubled-the-slice.md), [Retro-052](../retros/retro-052-every-phase-was-right-and-the-join-was-wrong.md), [Retro-053](../retros/retro-053-the-repetition-penalty-compounded-per-occurrence.md), [Retro-054](../retros/retro-054-a-transposed-view-saved-fortran-ordered.md), [Retro-055](../retros/retro-055-a-feedback-loop-cannot-be-gated-free-running.md) |
| Archive | [Flagship coverage, Aug 2026](../archive/ledger-2026-08-model-coverage.md) |
| Active tasks | [Backlog → Models](../backlog/active-index.md#models) |
