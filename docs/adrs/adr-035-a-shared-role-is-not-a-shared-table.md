---
type: adr
status: accepted
date: 2026-09-15
domain: export
supersedes: []
---

# ADR-035: A Knob That Changes the Output Is an Input, and a Shared Role Is Not a Shared Table

## Context

`SenseVoiceSmall` (family 5) prepends four rows of a 16-entry embedding table to its acoustic features
before the encoder sees them. Two of the four are fixed. The other two are choices:

* **a language id** — `auto` (detect), or one of six named languages;
* **a text-normalization id** — `woitn` or `withitn`.

Neither is cosmetic. On the same 11 seconds of speech:

    woitn    and so my fellow americans ask not what your country can do for you ask what you can ...
    withitn  And so my fellow Americans ask not what your country can do for you, ask what you can ...

Casing, punctuation and digits appear or do not. The reference pipeline takes both as ordinary
run-time arguments.

Two questions follow, and they were answered in opposite directions.

## Decision 1: the prompt is a graph INPUT, not a baked constant

Every other export in this zoo bakes what the caller did not ask about. Baking here would ship a model
that **can never punctuate** — a capability loss invisible in the file, recoverable only by exporting a
second GGUF per combination (fourteen of them, differing in four integers).

So `prompt_ids` is the graph's second input, declared after the waveform so
`apply_monolithic_export`'s root-axis expression still reads the waveform's own length.

The cost is one driver binding kind. A plain `CALLER` binding would mean `transcribe(audio)` no longer
works, because an input nobody knows about is still a required input — so `DriverInputs` gained
`DEFAULTED`, which emits the same `inputs.<name> or <fallback>` shape `NOISE` already uses, with a
literal array in place of a draw:

    local prompt_ids = (inputs.prompt_ids or {0, 1, 2, 15})

That is the whole of the family's orchestration difference from family 1's CTC driver. The fallback is
the vector `SenseVoiceSmall.inference` itself builds from its own default arguments, read off the
checkpoint's `lid_dict`/`textnorm_dict` rather than written down.

**The general rule this states:** a knob the reference exposes at run time, whose value changes what
the model outputs, belongs in the graph with a driver-supplied default — not in the exporter's
arguments. An export variant is the right answer only when the knob changes the model's SHAPE or its
weights.

## Decision 2: the language table is NOT published under `loom.asr.language_ids`

`language` is a recurring ASR role in [HIGH-LEVEL-API](../HIGH-LEVEL-API.md)'s canonical-name table,
and Whisper already publishes `loom.asr.language_names` / `loom.asr.language_ids`. Reusing them is the
obvious move and it is wrong.

Those keys are read into `AsrDecodeTable` (`model_contract.cpp`), whose ids `transcribe.cpp` pushes
into a **cross-attention decoder prompt** as token ids. Family 5's ids index a **16-row embedding table
prepended to the features**. The two are the same concept for a caller and different objects for the
engine, and a table read for the wrong mechanism is a defect waiting for an unrelated change to
uncover it. Today the misuse would be inert — that path is gated on a declared clip length, and this
file declares none — which is the shape of an accident, not an argument.

So the statement is split by what is mechanism-free and what is not:

| what | key | read by |
|---|---|---|
| which languages the model speaks | `loom.text.languages` | `ModelContract`, and the engine's existing refusal of a language a file cannot serve |
| name → prompt row | `loom.sanm.language_names` / `loom.sanm.language_ids` | this family, and the model card |
| name → text-normalization row | `loom.sanm.textnorm_names` / `loom.sanm.textnorm_ids` | this family, and the model card |

`auto` and `nospeech` are excluded from `loom.text.languages`: one asks the model to detect a language
and the other is a verdict it can return, and neither is a language a caller can ask it to speak.

**The general rule this states:** a canonical ROLE name is a promise about the caller's vocabulary, not
a licence to write into another family's table. Two families that share a role and not a mechanism
share the mechanism-free statement and nothing else.

## Consequences

* A host can offer `language=` and an ITN switch without per-model code, through the `sanm.` tables and
  `infer`; the high-level `transcribe(audio)` door is unchanged and unaware.
* `DEFAULTED` is available to any later family with an optional graph input. Families 1 and 4 pass an
  empty `defaults` map and their drivers are byte-identical to what they were.
* The engine gains nothing and needs to change nothing for either decision. Family 5 needed no new
  primitive, reader or binding — the first family since 12 of which that is true.
* Whisper's `asr.*` table keeps its meaning. If a second family ever genuinely has decoder prompt
  tokens, it can use those keys and mean them.
