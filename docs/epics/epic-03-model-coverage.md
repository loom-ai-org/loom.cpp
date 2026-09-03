---
type: epic
status: active
domain: model-coverage
last_updated: 2026-09-03
---

# Epic-03: Model Coverage

## 1. Context and Scope

The measure of the data-driven design is how cheaply a new architecture arrives. This epic covers which
models ship, which family template each belongs to, and the roadmap for the rest — with the standing
acceptance criterion that **a new family should need no engine work**.

Seventeen models are published at
[huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org). Each is a single GGUF carrying its own
topologies, driver and — where the architecture has one — its vocabulary.

## 2. Architectural Overview

### Shipped families

| domain | models | template |
|---|---|---|
| **Language** | Qwen3-0.6B-Base, LFM2-350M (monolithic *and* modular), SmolLM2-360M-Instruct, Gemma-3-270M-it | `causal_lm_export.py` |
| **ASR — NeMo encoders** | Conformer-CTC-small, Parakeet-TDT-0.6B, Parakeet-RNNT-0.6B, GigaAM v3 | `nemo_asr_export.py` |
| **ASR — encoder-decoder** | Whisper-small | `multi_phase_export.py` |
| **ASR — composition** | Qwen3-ASR-0.6B, Granite-Speech-4.0-1B | `speech_lm_export.py` |
| **TTS — flow matching** | Matcha-TTS, SupertonicTTS | `flow_matching_export.py` |
| **TTS — other** | Kokoro-82M, StyleTTS2, VITS (piper) | `multi_phase_export.py` |
| **Token classification** | any HF `*ForTokenClassification` (BERT-NER, DistilBERT-NER) | `token_classification_export.py` |
| **Audio codec (decode)** | DAC-44kHz | `audio_codec_export.py` |
| **Text → codec tokens** | Dia-1.6B | `dia_export.py` |

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

#### What the second codec costs, measured before starting it

EnCodec 32 kHz (MusicGen's codec) was probed and is **not** a repeat of DAC. Two blockers, both
confirmed on the real checkpoint and neither a gap in this exporter:

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

### Family 10, and the axis that did not have to be padded

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
  driver takes it a step further and uses `loom.argmax_row_range` per channel, because
  `DiaEOSChannelFilterLogitsProcessor` bans the control ids per channel rather than globally — the same
  restricted argmax `whisper_driver` detects a language with. **Three families running, and none has
  needed engine C++ for its graph**, which is the acceptance criterion the roadmap states.
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

**Not yet graded on what it sounds like.** The driver is greedy and runs without classifier-free
guidance, while the checkpoint declares `temperature 1.8 / top_k 50 / top_p 0.9` and
`guidance_scale 3.0`. Correctness against `transformers` under the *same* algorithm is what the gate
asserts; whether the audio is intelligible is a separate question and
[Retro-006](../retros/retro-006-kokoro-shipped-noise.md) is the standing warning about answering it
with a correlation. See [the backlog](../backlog/active-index.md#models) for what is left.

### Text input

**Only Supertonic takes text.** It encodes graphemes itself and its GGUF carries the codepoint table.
The other four TTS models consume *phoneme* ids produced outside the engine — a real limitation of
those checkpoints, addressed by [Epic-07](epic-07-text-frontends-and-tokenizers.md) and
[ADR-012](../adrs/adr-012-permissive-phonemizer.md).

## 3. Roadmap

Ordered by coverage-per-effort. Live items are tracked in
[the backlog](../backlog/active-index.md#models); the ordering and its reasoning are here.

**Next families:** the second codec decoder (EnCodec or SNAC) → CNN+CTC and SANM encoders (both
family-1-shaped once the encoder template generalizes past NeMo) → the remaining TTS families → text
encoder-decoders → small classifiers → music. Three are **done** as of 2026-09-03 — token
classifiers (12), codec decoders' first leaf (11) and the AR codec-token LM (10) — and §2 says what
each cost, which is the number the rest of this list should be estimated against. Family 10 landing
means the `text2codes` → `codes2speech` composition now has both halves in the tree.

**Named but unstarted:** Qwen3-ASR-0.6B and Qwen3-TTS-0.6B variants — Qwen3-TTS is expected to be the
most architecturally novel item in that family and needs its own source-level read before scoping.
F5-TTS is deferred by explicit direction (flow-matching, `OdeStepper`-adjacent, likely sharing
primitives with Matcha).

**The constraint that decides what is exportable at all** is not the template — it is peak memory
during conversion. `MultiPhase.export` made peak memory a *sum* where it should be a *max*; dropping
the traced module, the wrapper and the converted MIL program **together** took Granite-Speech from
30.4 GB to 22.9 GB peak RSS, the difference between OOM and a clean export. See
[the backlog](../backlog/active-index.md#models) for the two remaining changes.

## 4. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md), [ADR-005](../adrs/adr-005-export-config-and-task-registry.md), [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md), [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md) |
| Archive | [Flagship coverage, Aug 2026](../archive/ledger-2026-08-model-coverage.md) |
| Active tasks | [Backlog → Models](../backlog/active-index.md#models) |
