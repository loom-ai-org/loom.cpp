---
type: adr
status: accepted
date: 2026-09-25
tags: [exporter, text-front-end, contract, host-api, family-10, moss]
supersedes: []
---

# ADR-052: A Prompt Template's Language Line Ships Pre-Encoded, One Per Declared Language

## Context

MOSS-TTS wraps the text in a `<user_inst>` template whose lines include `- Language:` followed by a
language NAME (`French`) or `None`. The README recommends setting it. Two facts constrain how loom
builds that prompt:

* **The processor encodes each template piece SEPARATELY** and concatenates the ids. That is not the
  same as encoding the rendered string, because BPE merges across the joins differ. So the prompt's
  ids are a function of the segmentation, not only of the text.
* **The driver's Lua cannot tokenise.** Only the host (loom-py, loom_cli) holds the vocabulary.

## Options

1. **An engine-side vocabulary wrapper** that renders the template and encodes it piece by piece,
   the way CosyVoice3's front end ships as data ([ADR-048](adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)).
   That is C++ for what is, here, a fixed string with one variable line.
2. **The host renders and encodes the whole prompt.** That puts the template in every host, which
   is per-model code in loom-py.
3. **The export pre-encodes every segment** with the checkpoint's own tokenizer: the head, one
   "after the reference" segment per supported language plus the no-language one, and the tail. The
   driver concatenates them around the caller's text ids.

## Decision

**Option 3.** There are 31 languages from the README's table, and 32 short id arrays are a few KB in
the driver. The contract declares the language CODES as `loom.text.languages`. loom-py's
`text2codes.infer(language="fr")` passes the code's 1-based position as the driver input `language`,
where 0 means no language line. An undeclared code is refused with the list of declared ones.

## Consequences

* The prompt is exact by construction: the same tokenizer on the same segments. loom's own BPE
  matches transformers' on the text itself (checked on English and French).
* The template's other lines (`Instruction`, `Tokens` for duration, `Quality`, and so on) stay
  `None`. Each could be added the same way where the values are a closed set. `Tokens` is a
  number, so it would need the host to encode it, and it waits for a caller who asks.
* The `language` driver input is a position in the contract's list, which is the rule loom-py
  applies to any `text2codes` model that declares languages. Qwen3-TTS declares none and keeps its
  own `language_id`.

## Related

[Epic-03](../epics/epic-03-model-coverage.md) family 10,
[ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md),
[ADR-051](adr-051-a-remote-code-stack-of-a-known-architecture-is-rehosted-and-asserted.md).
