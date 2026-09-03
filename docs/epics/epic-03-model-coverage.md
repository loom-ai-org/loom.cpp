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
| **Token classification** | any HF `*ForTokenClassification` (verified on bert-base-NER) | `token_classification_export.py` |

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
labelled is a policy two hosts would otherwise decide independently. And one tracing decision, recorded
as [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md): a single unpadded sequence exports
with **no attention mask at all**, because every route `transformers` takes to build one bakes the
traced length into the graph.

Verified against `transformers` on the tensor rather than on the argmax
(the *tensor oracle, not token oracle* standing rule): max |Δ| 1.24e-05 over every logit of 138 tokens, worst
sentence cosine 0.99999988, 138/138 argmax agreement, and the engine's own WordPiece encode identical
to HF's ids. The sabotage arm — the same graph against a different sentence's reference — reads 11.16.

### Text input

**Only Supertonic takes text.** It encodes graphemes itself and its GGUF carries the codepoint table.
The other four TTS models consume *phoneme* ids produced outside the engine — a real limitation of
those checkpoints, addressed by [Epic-07](epic-07-text-frontends-and-tokenizers.md) and
[ADR-012](../adrs/adr-012-permissive-phonemizer.md).

## 3. Roadmap

Ordered by coverage-per-effort. Live items are tracked in
[the backlog](../backlog/active-index.md#models); the ordering and its reasoning are here.

**Next families:** codec decoders (unlocks the back half of the remaining TTS group) → CNN+CTC and
SANM encoders (both family-1-shaped once the encoder template generalizes past NeMo) → the remaining
TTS families → text encoder-decoders → small classifiers → music. BERT token classifiers came first and
are **done** (2026-09-03) — see §2 for what they cost, which is the number the rest of this list should
be estimated against.

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
