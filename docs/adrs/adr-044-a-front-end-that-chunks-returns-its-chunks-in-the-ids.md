---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, tokenizer, text-front-end, drivers, family-9, pocket-tts]
supersedes: []
---

# ADR-044: A Front End That Chunks Returns Its Chunks in the Ids

## Context

Pocket-TTS never generates more than one sentence-sized chunk at a time. `generate_audio_stream`
first runs `split_into_best_sentences`: prepare the whole text, tokenize it, cut after runs of
sentence-ending ids, and group the pieces into chunks of at most 50 tokens. A period between two
digits is not a cut. A sentence over the budget is cut again on clause marks. Then each chunk is
prepared AGAIN, tokenized again, and generated from a fresh copy of the voice state. Its audio comes
from a fresh Mimi state too. A text of more than one sentence is the normal case, not an edge.

The engine's text door is `vocab->encode(text) -> ids`, and a driver receives only ids. Some part of
the stack has to do the chunking.

## Options

* **The driver cuts the ids.** Rejected: it cannot be exact. The decimal rule is judged on DECODED
  text (`prefix[-2:] == digit + "."`, `suffix[0].isdigit()`). The per-chunk re-prepare capitalises
  each chunk's first letter and replaces a clause-split chunk's trailing comma with a full stop.
  Both need text, and the driver has none.
* **The host chunks and calls the driver once per chunk.** Rejected. Every host (loom-py,
  `loom_cli`, the C API) would reimplement it, and `Text2Speech.infer(text)` would stop being a
  single call.
* **The vocabulary chunks and returns every chunk's ids, each marked in the stream.** Chosen.

## Decision

`loom::PocketTtsVocab::encode` runs the reference's whole text path and returns each chunk's
prepared ids, each chunk OPENED by a header id: `<s>` when `prepare_text_prompt`'s tail guess for it
is the short one (at most four words: 3 frames after EOS), `</s>` otherwise (1 frame). Neither is ever
produced by an encode, since control pieces are not matchable. The driver splits on the headers and
runs the voice seed, prefill, latent loop and Mimi per chunk, with the tail its header names. Then it
concatenates the audio.

*Amended the same day.* The first version put a single `</s>` BETWEEN chunks and left the tail to the
driver, counted off the ids. That missed chunks whose words are separated by tabs or NBSP (47 of 9452
generated chunks), because the guess is `len(text.split())` of TEXT. The header carries it instead.
`chunks(text)` and `prepare(text)` are exposed so a host can show what each generation will say.

Every constant the path reads ships as data under `tokenizer.ggml.pocket_tts.*`
([ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md)'s rule): the replacement table, the
terminal, weak and closer sets, the case table, Python's `isdigit` set, the sentence-end and
clause-end ids, and the budget.

## Consequences

* **Exact against the reference, tail included.** `split_into_best_sentences` → `prepare_text_prompt`
  → tokenize, per chunk, each opened by the header of the reference's own tail guess, over 8000
  generated texts in eight classes (two seeds). Every one is identical. A reference sabotaged to skip
  the per-chunk prepare differs on 1894 of 3000. Getting there
  also meant teaching `loom::Vocab` SentencePiece's byte fallback, which the writer had been dropping
  silently.
* **A host that tokenizes elsewhere still works.** Plain SentencePiece ids carry no header, and the
  driver runs them as one chunk, counting its words off the ids (a piece opening with `▁`). That
  fallback alone is approximate: exact for ASCII spaces, and it misses words separated by a tab or an
  NBSP, at a cost of two frames (0.16 s) of tail.
* The pattern for the next front end that segments: the segmentation belongs where the text is.

## Related

* [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md): rules as data
* [ADR-033](adr-033-a-decode-only-table-is-still-a-vocabulary-family.md): a tag names a scheme
* [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md): text front ends
* `loom-exporter/loom_exporter/pocket_tts_tokenizer_export.py`: the writer
