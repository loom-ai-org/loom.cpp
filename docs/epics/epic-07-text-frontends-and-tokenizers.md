---
type: epic
status: active
domain: text-frontend
last_updated: 2026-08-22
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

Four TTS models (VITS, Kokoro, StyleTTS2, Matcha) take phoneme ids and have no text door. The licence
blocker is resolved: a C++ port of the Apache-2.0 `orthography2ipa`, vendored as a submodule, with
rules and data in the engine as one copy —
[ADR-012](../adrs/adr-012-permissive-phonemizer.md).

**The work splits in two, and only the second half needs the port:**

1. **Export the phoneme symbol table** as a vocabulary family. The data is already in each checkpoint
   and simply not exported. Gives four models a real `model.tokenizer` and a working
   `synthesize(phonemes=...)`, with no licence question at any point.
2. **The C++ port**, verified against the Python door as its oracle.

Target shape: `src/text/phonemize.cpp` + `include/loom/text/phonemize.h`. The Python half already ships
behind `loom-py-rt[phonemes]`.

**Two risks, both measurable before any C++ is written:** the fold-down into each checkpoint's fixed
symbol→id table (a symbol outside it has *no id*, which is a lookup with no answer rather than a
degradation), and pinning the beam search's tie-break so the CLI and `loom-py` cannot drift.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-012](../adrs/adr-012-permissive-phonemizer.md), [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| Design | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) §5 |
| Retros | [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md) |
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


### P4.23 — an instruction-tuned causal LM cannot be prompted correctly — SCOPED, NOT STARTED

**One sentence.** `detokenize` knows a checkpoint's special tokens and `tokenize` cannot produce them,
so a chat template is not merely un-applied — it is **unrepresentable**, and every instruction-tuned
LM in the fixture set is being run outside the distribution it was trained on.

Reported as three symptoms; two reproduce, and they share one root cause. All measurements below are
on the shipped GGUFs in `hf-models/`, through `loom-py` and `tools/loom_cli`, 2026-08-29.

#### The asymmetry, which is the whole item

```
smollm2-360m-instruct   tokenize("<|im_start|>") -> [44,108,306,79,3738,108,46]   should be [1]
                        tokenize("<|im_end|>")   -> [44,108,306,79,486,108,46]    should be [2]
                        detokenize([1, 2])       -> "<|im_start|><|im_end|>"      correct
gemma-3-270m-it         tokenize("<start_of_turn>") -> [2,236820,3041,236779,1340,236779,887,236813]
                                                                                  should be [105]
```

`BpeVocab::encode` runs the input straight through BPE. HF's tokenizers **split on added tokens first**
and emit their ids atomically; there is no such pre-pass in `bpe_vocab.cpp` (no `added`/`special`
handling anywhere in the file). The decode side works only because a special token's *spelling* is in
the vocabulary like any other entry.

**And the file does not even carry the information.** Neither GGUF has
`tokenizer.ggml.token_type` — the standard KV marking which ids are `CONTROL` / `USER_DEFINED`. The
exporter never writes it, so the engine could not do the pre-pass today even if `encode` wanted to.

This is a gap in a tokenizer that is otherwise verified: `test_e2e_spm_byte_fallback_tokenizer` checks
nine cases against `AutoTokenizer.encode` verbatim and all of them round-trip. **None of the nine
contains a special token**, which is exactly how a correct-looking tokenizer ships unable to encode a
chat turn.

#### What it does to real models — reproduced, not inferred

**Gemma 3 270M IT**, hand-written template through `loom_cli`, `--n-predict 80`:

```
<start_of_turn>user\nWho discovered Brazil?<end_of_turn>\n<start_of_turn>model\n
  -> "<end_of_turn>model\n<start_of_turn>artist\n<end_of_turn>artist\n<start_of_turn>artist..."
```

runs to the ceiling emitting turn after turn. **Two independent faults produce that**, and fixing
either alone is not enough:

1. the template's markers went in as ~7 literal tokens each, so the model never saw a turn boundary;
2. the file declares `tokenizer.ggml.eos_token_id = 1` (`<eos>`) while an IT checkpoint ends a turn on
   **`<end_of_turn>` = 106**, so the loop would not have stopped even if the model had emitted it.

`driver_components.py` **already has the field for (2)** — `GenerationLoop.extra_eos_tokens`, whose
own comment says "a chat-formatted checkpoint has two … and a loop knowing only the first runs to
`max_new_tokens` on every utterance". It is set by `speech_lm_export.py:770` and **never by
`causal_lm_export.py`**. That half is a one-line export change, not a design question.

**SmolLM2 360M Instruct** shows the same cause with the opposite symptom: asked a bare question it
emits its own `<|im_end|>` (id 2, its declared eos) as the *first* token, `strip_eos` drops it, and
`generate` returns **the empty string**. With `eos_token=-1` it continues
`"<|im_end|>\n<|im_start|>assistant\nThe discovery of Brazil"` — it is trying to open the assistant
turn itself, because nothing opened one for it. **An empty completion is the same bug as a runaway
one.**

**"Answers differ from HF"** follows from the above and needs no separate mechanism: the model is
being asked to continue malformed text. Gemma 3 on a bare `"Who discovered Brazil?"` produces a
repeating bulleted list (`"The Portuguese / The indigenous people of Brazil / …"`).

#### The reported symptom that did NOT reproduce, and the latent path it names

**"Inference outputs the prefill as well as the generated tokens" did not reproduce**, and the three
things that would have explained it away are each ruled out by measurement rather than by argument:

* **Not a door.** Five of them, all returning generated-only: `Model.generate`, `generate_ids`,
  `text2text.infer`, `Model.infer` (one token, as designed), and `Model.call("infer_with_past", …)`
  straight at the driver. Plus `loom_cli --prompt`. Four causal LMs — gemma-3-270m-it,
  smollm2-360m-instruct, lfm2-350m-monolithic, qwen3-0.6b-base.
* **Not "already fixed since the reported version".** The report is against **loom-py-rt 1.0.0rc6**,
  which pins loom.cpp `646c91c`. `git diff 646c91c..HEAD` over `text_generate.{h,cpp}`,
  `bpe_vocab.cpp` and `lua_bridge.cpp` is **empty** — the only engine change since rc6 is P4.19's
  profiler — and loom-py's Python layer is unchanged since its `1.0.0-rc6` tag.
* **Not a stale model.** The Hub artifact (`loom-ai-org/gemma-3-270m-it-loom`) and the local export are
  **byte-identical**, `md5 d6591bcf…`.
* **And rc6 itself was run**, installed from PyPI into a clean venv, against that same file: no echo,
  and output identical to the working tree's.

**The likeliest benign explanation, which should be checked before any code is touched:** an
instruction-tuned model given an un-templated prompt behaves like a base model and **continues in the
prompt's own format**. `"Question: Who discovered Brazil?\nAnswer:"` generates
`" The Portuguese\n\nQuestion: What is the capital of Brazil?\nAnswer: Brasília\nQuestion: …"` — the
prompt's shape reappears in the output as *generated* text, which reads exactly like a prefill echo and
is not one. That is the same root cause as the rest of this item, and it would be fixed by the same
work.

**Do not start here without a repro from the reporter** — model, exact call, and the observed output.
If it is the above, this bullet closes with the rest of the item.

There is nonetheless a real hole to close while the item is open. `text_generate.h` promises "returns
the GENERATED ids only, never the prompt — **both driver shapes are normalised to that here**", and
`text_generate.cpp` does not normalise: the list branch copies the driver's return verbatim with no
check that it excludes the prompt. The two shapes are told apart by *what the driver returns* (list vs
number), never by *whether it contains the prompt*, so a driver that returned prompt+generated would
be echoed and nothing would catch it. A generated `infer_with_past` starts `_gen` empty
(`driver_components.py:652`) and is fine today; a hand-written or future driver is unconstrained.
**Cheapest fix is an assertion plus a CI case**, independent of whether the reported symptom is ever
reproduced.

#### Where the work divides

* **Engine — `BpeVocab`**: an added-token pre-pass in `encode`, driven by a `tokenizer.ggml.token_type`
  the exporter must start writing. Longest-match-first over the added set, applied before regex
  pretokenization, which is what HF does.
* **Exporter**: write `tokenizer.ggml.token_type`; set `extra_eos_tokens` for `causal_lm_export.py`
  the way `speech_lm_export.py` already does.
* **Chat template — the open design question, and it is the only one.** The checkpoint's template is a
  Jinja string in `tokenizer_config.json`. Options, in ascending cost: (a) carry it as a KV and let the
  HOST render it, which needs no Jinja in the engine and no per-model C++; (b) ship the rendered
  role-tag strings as structured KV and have `text2text` assemble them; (c) a Jinja subset in the
  engine, which ADR-014's reasoning argues against on the same grounds as per-model kernels.
  **Decide this before writing anything**, and note that (a) and (b) both need the tokenizer fix first
  — a rendered template is worthless if it cannot be encoded.

#### Acceptance

* `tokenize("<|im_start|>") == [1]` on SmolLM2 and `tokenize("<start_of_turn>") == [105]` on Gemma 3,
  and `test_e2e_spm_byte_fallback_tokenizer` grows special-token cases against `AutoTokenizer.encode`
  verbatim — the nine that exist are the reason this shipped, and they are the shape to copy.
* Gemma 3 IT, correctly templated, **stops at `<end_of_turn>`** instead of running to the ceiling.
* SmolLM2 IT, correctly templated, returns a non-empty answer.
* Both models' answers compared against `transformers` on the same prompt through the checkpoint's own
  `apply_chat_template` — the reference this item is defined against.
* Round-trip preserved: `detokenize(tokenize(s)) == s` for text with and without special tokens.

#### What NOT to do

* **Do not add a `--chat-template` flag to `loom_cli` and call it done.** The tokenizer cannot encode
  the result; the flag would produce exactly the output above.
* **Do not special-case Gemma 3.** Every instruction-tuned checkpoint in the set has this, with
  different markers — SmolLM2's are `<|im_start|>`/`<|im_end|>`, Gemma's are
  `<start_of_turn>`/`<end_of_turn>`. It is one mechanism, not two models.
* **Do not touch the decode loop for the runaway turns.** `max_new_tokens` is doing its job; the loop
  is stopping where it was told to stop.

### Grapheme text front-ends: the shape to generalize to

`src/core/supertonic_text_vectorizer.cpp` is per-MODEL C++ in an engine whose rule is that per-model
complexity belongs in the exporter. It is kept because it is written and verified, but a SECOND grapheme
TTS model must not add a second class. The split that generalizes: the codepoint table is data and already
lives in the GGUF; the normalization pipeline (emoji ranges, the replacement table, the `<lang>` wrap) is
per-model and belongs in exporter-emitted data or driver Lua. Worth doing when there is a real second data
point to generalize against — Qwen3-TTS is not one, since it has a real HF tokenizer and needs only
`tokenizer_dir` set in its export config, no new C++ at all.

---

