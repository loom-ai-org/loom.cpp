---
type: adr
status: accepted
date: 2026-09-25
tags: [engine, tokenizer, text-front-end, drivers, family-9, cosyvoice3]
supersedes: []
---

# ADR-048: CosyVoice3 Normalises Text by the Reference's Rules Path, and Each Chunk Is a Generation

## Context

CosyVoice3 shipped with a plain `"gpt2"` text door: the Qwen2 BPE and nothing else. The reference runs
`CosyVoiceFrontEnd.text_normalize(text, split=True)` before any id reaches the model. That function
spells digits out, applies Chinese punctuation rules, and splits a paragraph into pieces of about 80
tokens. It then synthesises **each piece separately** (its own LM decode, flow and vocoder, from the
same voice) and joins the audio. Without it, "1998" is read however the LM guesses, and a long paragraph
is one decode that runs past what the model was trained on.

`text_normalize` has three levels, and it picks one by what is INSTALLED: `ttsfrd` (Alibaba's, a wheel
plus a resource archive, not open), then `wetext` (Apache-2.0: WeTextProcessing's tagger and verbalizer
FSTs, downloaded from ModelScope at run time and run through `kaldifst`), then neither. With neither
installed it runs a set of rules: inflect's `number_to_words` on every digit run, the Chinese replacement
table, `split_paragraph`, and a punctuation-only filter.

## Options

* **wetext (or ttsfrd) too.** Deferred to its own item (user's decision, 2026-09-25). It means shipping
  the FSTs as data AND an FST runtime in C++, at OpenFst scale, and checking the FSTs' licence. The
  engine is for edge devices, and one model's normaliser does not justify an FST engine yet.
* **The host normalises and chunks.** Rejected on
  [ADR-044](adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)'s reasoning. Every host
  would reimplement it, and `Text2Speech.infer(text)` would stop being one call.
* **The rules path, in the vocabulary, shipped as data.** Chosen.

## Decision

`tokenizer.ggml.model == "cosyvoice3"`, implemented by `loom::CosyVoice3Vocab`. It wraps an ordinary
`BpeVocab`, which now has `load_bpe`, a tag-agnostic loader. `encode` is `text_normalize` with no
frontend installed, then the BPE per piece. Following
[ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md), every constant is a
`tokenizer.ggml.cosyvoice3.*` key written from the reference, inflect and `regex`:

* the replacement table, the trailing-comma rule, the ender, closer and terminal sets, and the budgets;
* Python's `isdigit` set (the runs) and the `\d` digits with their values (what inflect keeps). "2²" is
  "two", and "m²" is "mzero", as in the reference;
* inflect's word tables and default words, and the `[\p{P}\p{S}]` ranges.

**Chunks follow ADR-044's shape.** `encode` opens every chunk with `<|endoftext|>`. That token can open
chunks because a text holding `<|` and `|>` skips the whole front end and is one chunk, and no other
text can spell it. The one text that could, a marked-up one that types `<|endoftext|>` itself, is
refused by name.

**The driver loops by calling itself.** Its first fragment splits the ids on the header and runs the
global `infer` once per chunk with the chunk's ids and every other input unchanged. It then concatenates
the waveforms. The seed is applied once, so the engine's stream carries on from one chunk to the next,
as the reference's RNG does. Inputs that pin ONE generation (`speech_tokens`, `draws`, `noise`,
`nsf_noise`, `return_tokens`, `return_mel`) are refused when there is more than one chunk. Ids with no
header (a host that tokenized elsewhere) are one chunk, as before.

**Where the reference crashes, `encode` throws `Error`.** That covers empty or whitespace-only text,
37+ digits (inflect's `NumOutOfRangeError`), and a closing quote after an ender that ends no sentence
(an `IndexError`). A text of nothing but punctuation yields no audio in the reference and throws here.

## Consequences

* **Measured exact.** Against the reference's `text_normalize` + tokenizer, the ids match per chunk,
  headers included, on **9000/9000** generated texts. The texts come from nine classes (prose with
  numbers, long English, Chinese, long Chinese, mixed, markup, tags, exotic digits and spaces,
  punctuation soup), over two seeds, with ~2000 multi-chunk texts and ~400 reference crashes compared as
  refusals. Three sabotaged references each go red: numbers not spelled (547/1800 differ), a budget one
  token wider (5/1800), and `replace_blank` skipped (201/1800).
* Getting there found an engine bug shared by every byte-level BPE: `\s` was ASCII-only
  ([Retro-059](../retros/retro-059-a-shared-tokenizers-whitespace-was-ascii.md)).
* **Voice files need nothing new.** A CosyVoice3 voice is four driver inputs (`prompt_text`,
  `prompt_speech_tokens`, `prompt_feat`, `embedding`). `loom_exporter.cosyvoice3_voices` computes them
  with the reference's own front end, running the two ONNX models once in Python, and writes an
  [ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md) voice file. The
  model declares `voice.compat` (a sha256 over `llm.pt` and `flow.pt`; HiFT never reads a voice) and
  names its built-in voice `zero_shot_prompt`.
* **An rc10 engine does not know the tag.** loom-py's dispatch leaves such a model with no tokenizer,
  so the text door fails loudly rather than running un-normalised. This leaf already needs rc11.

## Related

* [ADR-044](adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md): chunks in the ids
* [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md): rules as data
* [ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md): voice files
* [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md): text front ends
* `loom-exporter/loom_exporter/cosyvoice3_tokenizer_export.py`: the writer
