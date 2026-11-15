---
type: adr
status: accepted
date: 2026-11-15
tags: [engine, exporter, streaming, tokens-out, t5, encoder-decoder]
supersedes: []
---

# ADR-130: An Encoder-Decoder Token Stream Keeps Its Per-Call Context in the State

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Extends the tokens-out door of
[ADR-099](adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md) from causal LMs to flan-t5-small.

## Context

ADR-099 cut the generated decode loop (`PrefillDecodeLoop`) per token for causal LMs. Its state is the
cursor, the count, the last token and the stop rule, plus the model's KV cache, leased or owned.
`stream_supported()` refused any loop with per-call context. T5 has three pieces of it:

* `bound`: the decoder's cross-attention K/V, `OutputRef`s to the `cross_kv` phase's retained output.
* `extra_axes`: `n_enc_frames`, the source length.
* `prompt`: the decoder start token, not the caller's tokens (those are the source).

A retained output belongs to the model: the next call of that module overwrites it, so a stream cannot
read another stream's `cross_kv`. T5's `infer` runs the whole loop, and the file has no
`infer_with_past`. So `GenerationStream::supported(bridge)` took it for a one-token causal LM. Its loop
then fed the answer back as the source: the published flan-t5 file streamed one token and stopped.

The user chose flan-t5 alone for this row. The AED and speech-LM ASRs already stream audio in
(ADR-107, ADR-121, ADR-124, ADR-128); a token-by-token transcript of their one pass adds little.

## Options

1. **A hand-written `stream_*` fragment for T5**, as SpeechT5 has (ADR-119): a second copy of a loop the
   exporter already generates.
2. **The generated loop declares its context** and the stream entries keep it in the stream state.
3. **A host loop**: the encoder output would have to cross into the host.

## Decision

Option **2**.

* **`PrefillDecodeLoop.stream_context`**: the components that build the context (T5's `encoder` and
  `cross_kv` calls), the same objects `infer` runs. When set:
  * `stream_supported()` accepts `bound`, `extra_axes` and a constant `prompt`;
  * `stream_open` runs those components on the caller's tokens, then `loom.keep_state(m, m)` for every
    module `bound` reads, then `loom.set_state('axis_<name>', expr)` for every extra axis;
  * the step reads `{state = m, index = i}` (new IR node `StateRef`) and `loom.state('axis_<name>')`.

  `infer` is untouched: the published file's `infer` text is byte-identical.
* **`MultiPhaseDriverBuilder.extra_entries`** emits the stream entries for such a loop. A loop without
  `stream_context` gets none: Whisper, Canary, Moonshine and the speech LMs keep their own fragments.
* **Module-scope names**: T5 binds its constants a second time at top level, since `stream_open` builds
  the step's bias outside `infer`.
  * `DriverComponent.module_names()`: top-level `ExportConstants` keys and a top-level fragment's
    `defines`.
  * `driver_ir.validate` takes those names as defined in every function. Without this, a generated
    function beside `infer` could read no constant.
* **The one-token path needs a causal LM.** `GenerationStream::supported(bridge, model)` refuses a
  `text2text-generation` file without `stream_open`; the constructor and loom-py's `can_stream_tokens`
  use it. The bridge-only form stays, for a caller without the model.

## Consequences

* `test_e2e_t5_token_stream` checks 5 prompts against transformers' greedy `generate`
  (`scripts/t5_stream_reference.py`), stopping on `</s>` or at the 48-token ceiling:
  * `generate` and the stream pulled 1, 4 and 48 at a time return those ids, the stop included;
  * `reset` repeats them;
  * two streams on different sources, interleaved with a one-pass `generate` on a third, match under
    both cache modes;
  * the published file's one pass is unchanged, and the door now refuses that file (72 checks).
* Cross-attention K/V kept per stream: 2 x 8 layers x 512 x source length floats. 0.3 MB for 10 tokens.
* Publishes after a re-export. The four all-zero biases ADR-113 drops leave the file too (196 tensors).
* Any later encoder-decoder (a translation T5, mT5 once its leaf exists) streams by setting
  `stream_context`.
