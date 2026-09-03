---
type: epic
status: active
domain: text-frontend
last_updated: 2026-09-03
---

# Epic-07: Text Front-Ends and Tokenizers

## 1. Context and Scope

Everything between a string and the ids a model consumes: BPE and SentencePiece tokenizers, grapheme
vectorizers, and the phonemizer that four TTS models are waiting on.

This is one of the few areas where C++ in the engine is correct, because a tokenizer is per-**task**,
not per-model ([ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md)) — with one
deliberate, temporary exception noted below.

## 2. Architectural Overview

### Tokenizers

`bpe_vocab.cpp`'s `pre_spec_table()` carries ~40 pretokenizer families. `BpeShape::kSpmByteFallback`
covers SentencePiece-style byte-fallback BPE, which has four structural differences from every other
shape: no regex pretokenization (one chunk), no GPT-2 byte-level mapping (initial symbols are
characters), and two more — each measured against the real tokenizer rather than inferred.

**A family the table does not cover raises a named error rather than mis-tokenizing.** That is what
turns "add a tokenizer" into a bounded job instead of a mystery, and it is the pattern to keep.

**ADDED tokens are handled before any of that** (P4.23). `encode` splits the raw input on the
vocabulary's added set — every CONTROL and USER_DEFINED entry of `tokenizer.ggml.token_type` — longest
match first, and only the runs between them reach the normalizer and the pretokenizer. That is HF's own
order, and without it a marker merges into ordinary pieces and a chat template is unrepresentable. A
file carrying no `token_type` skips the pre-pass entirely, which is what keeps every GGUF exported
before P4.23 tokenizing exactly as it did.

**And the checkpoint's chat template is DATA in the file**, not a program: the exporter reduces its
Jinja to role tags and `loom::ChatTemplate` concatenates them — [ADR-018](../adrs/adr-018-chat-template-as-role-tags.md).

### Grapheme front ends

`src/core/supertonic_text_vectorizer.cpp` is per-**model** C++ in an engine whose rule forbids it. It
is kept because it is written and verified — and **a second grapheme TTS model must not add a second
class.**

The split that generalizes, when a real second data point exists:

* the **codepoint table is data** and already lives in the GGUF;
* the **normalization pipeline** (emoji ranges, the replacement table, the `<lang>` wrap) is per-model
  and belongs in exporter-emitted data or driver Lua.

Qwen3-TTS is **not** that second data point: it has a real HF tokenizer and needs only `tokenizer_dir`
set in its export config — no new C++ at all.

### Phonemization

Four TTS models (VITS, Kokoro, StyleTTS2, Matcha) take phoneme ids. The licence blocker is resolved: a
C++ port of the Apache-2.0 `orthography2ipa`, vendored as a submodule, with rules and data in the engine
as one copy — [ADR-012](../adrs/adr-012-permissive-phonemizer.md).

**The work splits in two, and only the second half needs the port:**

1. **Export the phoneme symbol table** as a vocabulary family — **DONE**. The data was already in each
   checkpoint and simply not exported; `tokenizer.ggml.model = "phonemes"` plus an id-indexed
   `tokenizer.ggml.tokens` and the four `tokenizer.ggml.phoneme.*` assembly KVs now travel in all four
   published GGUFs, and `loom::PhonemeVocab` (`src/core/phoneme_vocab.cpp`) reads them back for both
   hosts. No licence question arose at any point. §4 records what shipping it took a second pass to
   finish — [Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md).
2. **The C++ port**, verified against the Python door as its oracle. Still open.

Target shape: `src/text/phonemize.cpp` + `include/loom/text/phonemize.h`. The Python half already ships
behind `loom-py-rt[phonemes]`.

**Two risks, both measurable before any C++ is written:** the fold-down into each checkpoint's fixed
symbol→id table (a symbol outside it has *no id*, which is a lookup with no answer rather than a
degradation), and pinning the beam search's tie-break so the CLI and `loom-py` cannot drift.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-012](../adrs/adr-012-permissive-phonemizer.md), [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md), [ADR-018](../adrs/adr-018-chat-template-as-role-tags.md) |
| Design | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) §5 |
| Retros | [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md), [Retro-021](../retros/retro-021-nine-oracle-cases-and-none-was-a-marker.md), [Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md) |
| Active tasks | [Backlog → Text front-ends](../backlog/active-index.md#text-front-ends) |

## 4. The Record

### SentencePiece-style byte-fallback BPE — DONE


`BpeShape::kSpmByteFallback`, added so Gemma 3 tokenizes correctly. `pre_spec_table()`'s own comment had
already scoped it ("needs a different symbol-initialization step in `BpeVocab::encode()` itself") and
`tokenizer_detect.py` raised a named `NotImplementedError` rather than mis-tokenizing — which is what
made this a bounded job instead of a mystery.

Four structural differences from every other shape, each measured against the real tokenizer: no regex
pretokenization (one chunk); no GPT-2 byte-level mapping, so initial symbols are characters and the
vocabulary holds literal UTF-8; a space→U+2581 normalizer with no dummy prefix (`"Hello world"` →
`['Hello', '▁world']`), and no NFC, because the HF normalizer is that substitution and nothing else; and
`<0xNN>` byte fallback for characters with no entry. `decode` mirrors all of it.

Gated by `test_e2e_spm_byte_fallback_tokenizer` — nine cases, every expectation `AutoTokenizer.encode`
verbatim, all encoding exactly and round-tripping. Gemma now exports with no `--tokenizer-pre` override.
The remaining unimplemented families in `_LLAMA_PRE_TO_LOOM_PRE_TYPE` (CJK-script splitters,
case-transition shapes, cascading-whitespace shapes) are still `None` and still raise by name.


### P4.23 — an instruction-tuned causal LM could not be prompted correctly — DONE (2026-08-29)

**One sentence.** `detokenize` knew a checkpoint's special tokens and `tokenize` could not produce
them, so a chat template was not merely un-applied but **unrepresentable** — and every
instruction-tuned LM in the fixture set was being run outside the distribution it was trained on.

Three independent defects, one symptom, and **fixing any one alone was not enough**. All three are
closed. The reference the item was defined against — the same prompt through the checkpoint's own
`apply_chat_template` in `transformers`, greedy — now matches **exactly**:

| | loom | `transformers` |
|---|---|---|
| gemma-3-270m-it | `The discovery of Brazil was made by **Hernán Cortés**.\n` | identical |
| smollm2-360m-it | `Brazil was discovered by Portuguese explorers in the 15th century. …` | identical |

#### What was wrong, and what each fix was

**1. `BpeVocab::encode` had no added-token pre-pass.** HF's `AddedVocabulary` splits the raw input on
the added set *before* the normalizer and the pretokenizer; loom ran the input straight through BPE, so
`tokenize("<|im_start|>")` was seven literal ids and `tokenize("<start_of_turn>")` eight. The decode
side worked only because a marker's *spelling* is in the vocabulary like any other entry.

Fixed at both ends, because **the file did not even carry the information**: the exporter now writes
`tokenizer.ggml.token_type` (llama.cpp's own KV, parallel to the token list), and `encode` splits on
every CONTROL and USER_DEFINED entry, longest match first, on the raw bytes.

* **All added tokens, not just the special ones.** Gemma 3 declares 6408 non-special ones — the
  whitespace runs — and HF splits on those too, so a pre-pass that only knew about markers would have
  disagreed with the reference tokenizer on ordinary prose rather than only on chat.
* **BYTE entries are excluded.** `<0xNN>` exists to be reached by fallback; treating it as splittable
  would turn someone's literal text `"<0x41>"` into byte 0x41.
* **A file with no `token_type` behaves exactly as it did**, which is what keeps every pre-P4.23 GGUF
  tokenizing identically. Guessing (every `<…>`-looking piece, say) would have made `encode` disagree
  with the reference on ordinary text that happens to look like a marker.
* `decode` now returns an added token's piece verbatim rather than through the byte decoder. For
  `<|im_end|>` the two agree; for one containing a newline the byte map has no key at all and every
  such character was silently dropped.

**2. The export read the tokenizer config's single `eos_token`.** gemma-3-270m-it's
`generation_config.json` says `eos_token_id: [1, 106]` — `<eos>` and `<end_of_turn>` — and a loop
knowing only the first runs to `max_new_tokens` on every utterance. `bpe_tokenizer_export.py` now
reads the generation config (as `granite_speech_export.py:437` and `qwen3_asr_export.py:363` already
did), writes the whole set as `tokenizer.ggml.eos_token_ids` beside the unchanged scalar, and passes
the remainder to `PrefillDecodeLoop.extra_eos_tokens`. `loom::text::eos_token_ids` is where a host
reads it back, and `strip_eos` now strips ANY of them — stripping only the scalar left
`<end_of_turn>`'s literal spelling on the end of every chat answer.

**3. There was no chat template anywhere.** Decided and recorded as
[ADR-018](../adrs/adr-018-chat-template-as-role-tags.md): the EXPORTER reduces the checkpoint's Jinja
to role tags by differencing real renders, verifies the reduction against `apply_chat_template` at the
string AND id level, and writes seven KVs; `loom::ChatTemplate` concatenates them. No Jinja on either
side of the boundary, and no per-model C++.

Reached from `loom_cli --chat`/`--system`, `model.chat(...)` and `model.text2text.chat(...)`.

#### What this does NOT fix, and did not need to

**The reported "prefill echo" was never a bug.** The model *re-generates* the prompt's text: an
instruction-tuned model given an un-templated prompt continues in the prompt's own format. Verified on
rc6 over five doors — the ids do not start with the prompt, they re-emit it from position 3. Closed by
measurement, not by code.

There WAS a real hole beside it, and it is now closed: `text_generate.h` promised both driver shapes
were normalised to generated-ids-only and the list branch did not check. **The check is on the COUNT,
deliberately** — a conforming driver breaks at `#_gen >= max_new_tokens` and can never return more,
whatever it generated. "Does the return start with the prompt" was rejected as the test: it cannot tell
an echo from a model repeating its own prompt back, which is exactly what an un-templated instruction
model does, so it would have thrown on correct output.

#### Gates

* `tests/ci/test_chat_template.cpp` — hermetic, over a 263-token fixture with three added tokens and a
  ChatML decomposition. **Two files from one generator**, with and without `token_type`, because the
  backward-compatibility claim is only a claim until both are compared.
* `tests/gate/test_e2e_spm_byte_fallback_tokenizer.cpp` — nine more cases, every expectation
  `AutoTokenizer.encode` verbatim, including markers, a marker inside prose, a non-special added token,
  `<bos>` and `<0x41>`. **A pre-P4.23 GGUF fails it**, which is correct: the fix is a re-export. Why
  the nine that were already there could all pass while `encode` was broken is
  [Retro-021](../retros/retro-021-nine-oracle-cases-and-none-was-a-marker.md).
* `tests/gate/test_e2e_chat_generation.cpp` — generic over any GGUF carrying a template. The central
  assertion is that **no generated id is a CONTROL token**: a model given a turn produces text, one
  that was not produces structure. Both sabotage checks bite (the pre-pass off, and the eos set
  narrowed to the scalar).
* `loom-py/tests/ci/test_api.py` — that this layer hands the engine a conversation and renders none
  itself.

#### The one thing left over

Every causal-LM model card shipped the failing snippet (`build_model_cards.py`). The five IT cards now
show `text2text.chat(...)`; the base-model cards keep `infer(...)`, which is the right call for them.
**The re-exported GGUFs are not on the Hub** — that is the rc7 push, tracked in the hub with the three
already-stale models.

### Task #79 part 1 — the phoneme symbol table — DONE (2026-08-15, finished 2026-09-03)

**What the file carries.** `tokenizer.ggml.model = "phonemes"`, an id-indexed `tokenizer.ggml.tokens`
(159 symbols for piper's VITS, 178 each for Kokoro, StyleTTS2 and Matcha), and four
`tokenizer.ggml.phoneme.*` KVs for the assembly. The table was always in the checkpoint; nothing was
computed, only exported. `ExportConfig.phoneme_table()` is the per-family door and
`exporter.py` writes the KVs.

**The assembly is declared, not reimplemented.** Piper builds `[BOS, p1, blank, ..., pn, blank, EOS]`
with no blank after BOS; StyleTTS2 wraps with neither an EOS nor blanks. So `bos_id`/`eos_id`/`blank_id`
and `interleave_blank` travel with the table and `loom::PhonemeVocab::encode` applies them — the same
arrangement `SupertonicTextVectorizer` has for its `<lang>` wrap, one modality over. **A `-1` is the
sentinel for "this model has none"**, not an id: three of the four declare at least one, and appending
it literally reaches the engine as an out-of-range `GET_ROWS`.

**Two properties the lookup needs.** Longest match, because IPA symbols are not one codepoint each
(`t͡ʃ`, `aɪ`), so a per-codepoint walk splits every diphthong into pieces no model has seen. And an
unknown symbol is **skipped, not raised on**, with a count returned — a rule-based G2P emits a superset
of any one checkpoint's inventory, so refusing an utterance over one diacritic would fail every long
sentence.

**What the first pass left.** `loom_cli` never learned the tag, so the C++ host reported no tokenizer
for any of the four and an IPA `--prompt` died in the LM path's `parse_token_ids`; and `PhonemeVocab`
had no test in this tree. Both closed 2026-09-03, and both hosts now agree id for id on the same string.
The lesson is [Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md).

**What is still outside every GGUF, deliberately:** grapheme→phoneme, which is a property of the
language rather than of any checkpoint. That is part 2.

### Grapheme text front-ends: the shape to generalize to

`src/core/supertonic_text_vectorizer.cpp` is per-MODEL C++ in an engine whose rule is that per-model
complexity belongs in the exporter. It is kept because it is written and verified, but a SECOND grapheme
TTS model must not add a second class. The split that generalizes: the codepoint table is data and already
lives in the GGUF; the normalization pipeline (emoji ranges, the replacement table, the `<lang>` wrap) is
per-model and belongs in exporter-emitted data or driver Lua. Worth doing when there is a real second data
point to generalize against — Qwen3-TTS is not one, since it has a real HF tokenizer and needs only
`tokenizer_dir` set in its export config, no new C++ at all.

---

