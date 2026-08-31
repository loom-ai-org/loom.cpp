---
type: adr
status: accepted
date: 2026-08-29
tags: [tokenizer, text-frontend, chat, exporter, gguf]
---

# ADR-018: A Chat Template Ships as Role Tags, Reduced by the Exporter

## Context

An instruction-tuned checkpoint carries its prompt format as a **Jinja program** in
`tokenizer_config.json`. Without applying it the model is asked to continue malformed text: Gemma 3
270M IT ran to `max_new_tokens` emitting turn after turn, and SmolLM2 360M Instruct returned the empty
string (P4.23, [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md)).

Applying it needs the format to reach the engine somehow. Every option had to satisfy two constraints
already decided elsewhere: per-model complexity belongs in the exporter
([ADR-003](adr-003-per-model-complexity-in-the-exporter.md)), and `loom_cli` is a first-class host, so
a door only `loom-py` can open is not a door.

## Options Considered

1. **Carry the Jinja source as a KV; the HOST renders it.** No Jinja in the engine and no per-model
   C++ — but `loom_cli` has no Jinja evaluator and could never get one for the cost, and `loom-py`
   would need a Python templating dependency it does not have. The C++ half of the fleet would have no
   chat door at all.
2. **A Jinja subset in the engine.** Reaches every host, and is per-model machinery in the one place
   this project keeps removing it from. [ADR-014](adr-014-patch-ggml-rather-than-write-kernels.md)'s
   reasoning applies unchanged: a subset is a promise about programs nobody has read.
3. **Reduce the template in the EXPORTER to the role tags it emits.** The exporter already has a Jinja
   evaluator (`transformers`) and the real template; the engine gets two parallel arrays and three
   strings, and concatenates them.

## Decision

**Option 3.** `loom-exporter/loom_exporter/chat_template_export.py` renders the checkpoint's real
template with sentinel content, recovers the parts by DIFFERENCING those renders, and writes:

```
tokenizer.chat_template.roles              [str]   what this checkpoint can express
tokenizer.chat_template.prefixes           [str]   parallel
tokenizer.chat_template.suffixes           [str]   parallel
tokenizer.chat_template.prologue           str     before the first message
tokenizer.chat_template.system_prologue    str     ... when the conversation opens with a system turn
tokenizer.chat_template.generation_prefix  str     the opening of the reply being asked for
tokenizer.chat_template.trim_content       bool    the template's own `| trim`
```

`loom::ChatTemplate` (`src/core/chat_template.cpp`) assembles them. Nothing parses Jinja on either
side of the boundary, so a template written in any style is handled the same way — **what decides
whether a checkpoint is supported is whether its own output DECOMPOSES**, not whether its source is
recognised.

**Every decomposition is verified against `apply_chat_template` before it is written**, on real
multi-turn conversations, at the string level and again at the ID level. The id check is not
redundant: a template opening with `<bos>` while the tokenizer also prepends one produces text that is
character-identical and ids that are `[2, 2, 105, …]` against `[2, 105, …]`.

## Consequences

* **A template that does not decompose is not exported**, and the model then has no chat door rather
  than a wrong one. Qwen3-0.6B-Base is the live case: its template rewrites earlier assistant turns
  (it strips `<think>` blocks), so a turn's rendering depends on its position and the differencing
  rejects it — correctly, since it is a base model.
* **A role may be missing from a model that has a template.** Gemma 3 folds a system message into the
  text of the first user turn instead of emitting a block for it, so it exports `[user, assistant]`
  and `ChatTemplate::apply` raises by name on a system message. An error rather than a dropped
  argument, because a silently ignored system prompt is a wrong answer with no signal.
* **This depends on the tokenizer fix and is worthless without it.** A rendered template whose markers
  encode as seven literal ids apiece is not a chat turn. The added-token pre-pass
  (`tokenizer.ggml.token_type`, `BpeVocab::encode`) and this land together.
* **Content is not escaped**, and cannot be: a user string containing `<|im_end|>` becomes that token,
  exactly as it does in `transformers`. Not a new exposure, and worth knowing.
* **A template the exporter's differencing cannot handle is a bug in that file**, not a checkpoint
  problem — the verification distinguishes the two, raising for the first and declining for the second.

## Serves

[Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md) (text front-ends and tokenizers),
[Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (the `text2text.chat` door).
Related: [ADR-003](adr-003-per-model-complexity-in-the-exporter.md),
[ADR-013](adr-013-one-door-per-task.md), [ADR-006](adr-006-model-constants-belong-to-the-export.md).
