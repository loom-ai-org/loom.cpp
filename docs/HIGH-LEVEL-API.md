# The high-level layer: one door per task, declared by the file

Status: proposal. Prompted by loom-py #6 / loom.cpp #5, which added `transcribe` and had to argue from
first principles where it belonged. That argument was won on the merits and should not have to be won
again for the third door and the fourth. This is the rule it implies, written down, plus the metadata
gap that stops the rule from being applicable today.

---

## 1. The diagnosis

`generate` is not the low-level API's high-level counterpart. It is *the causal-LM task's door*, added
when that was the only task with one, and named as if it were universal. `transcribe` is the second
door, and TTS will need a third — the phoneme families need a G2P step, the flow-matching ones need
sampler defaults and a voice, and every one of them needs a waveform back with a sample rate attached.
So the layer is real and it is not a Whisper accident. What is missing is a rule for where each door's
work goes, and a way for a host to know which door a file answers.

Three symptoms, all present in the tree today:

**The same per-task loop exists twice and has already drifted.** The causal-LM decode loop for a driver
that returns one token at a time is written in `tools/loom_cli/main.cpp` (~line 604) and again in
loom-py's `generate_ids`. They disagree: the CLI runs exactly `n_predict` steps with **no EOS stop at
all**, takes `vec[0]` when the driver returns a list, and clamps ids to `< 65536`; loom-py stops on the
file's own `eos_token_id`, takes `vec[-1]`, and strips the EOS before returning. Same model, same
driver, two transcripts. This is exactly the asymmetry #6 removed for audio, one task over, and it is
the one that decided that PR — so it should decide this one.

**Hosts guess what a model IS from its tokenizer tag.** `loom_cli` branches on
`tokenizer.ggml.model == "bert" | "byt5" | "supertonic"` and dead-ends each as "inspection-only",
because a vocabulary family is the closest thing in the file to a statement of what the model does.
A Supertonic GGUF — a complete TTS model — reaches a branch whose comment explains that falling through
to the generation path would have parsed its text as literal token ids. That branch is per-model code
in a host, written because the file does not say what it is.

**The engine's ASR loop is per-task in shape and per-family in its constants.** `transcribe.cpp`
resolves timestamps with `piece_to_id("<|0.00|>")`, languages with `"<|" + name + "|>"`, and drops
`<|notimestamps|>` by spelling. Those are Whisper's conventions, not ASR's. Canary, Qwen3-ASR and
Granite-Speech spell all three differently, so the second timestamped family costs C++ in an engine
whose stated rule is that a family costs Python in the exporter.

The common cause is one sentence: **nothing in a GGUF says what contract it implements.** The file
declares `loom.architecture` — a per-model name — and a scattering of per-family hparams. Any host-side
dispatch on that is per-architecture code, which loom-py's CLAUDE.md forbids outright and the engine's
own headers argue against. So today a high-level layer is either impossible or illegal, and `transcribe`
got through by being hand-placed.

---

## 2. The rule

The engine already states half of it, in `loom.h`, over the vocab/CTC headers:

> Task-level helpers a host uses around a driver, not inside one [...] These are per-TASK, not
> per-model — one CTC decoder covers every CTC model.

That is the whole doctrine, and it needs one addition to be decidable, because "per-task" does not by
itself say whether a per-task thing goes in the file, the engine or the host. The addition:

> **In the file** when it is a property of the CHECKPOINT.
> **In the engine** when it is a property of the TASK.
> **In the host** when it needs the host's ecosystem.

with a corollary that settles every hard case, including the one #6 argued:

> **Anything shipped inside a GGUF can only be fixed by re-exporting every model.** So a policy that
> will evolve — a seek strategy, a stop condition, a sampler default — must not be baked into files,
> even when the file could technically express it.

That is why Whisper's window/seek loop is correctly in the engine rather than emitted as Lua, although
Lua could do it: the seek policy is one policy for every timestamped ASR model, and improving it must
not mean re-exporting the fleet. And it is why prompt construction stays in Whisper's driver: which
tokens a prompt needs *is* a property of that checkpoint.

Restated for each layer, with its admission test:

| Layer | Owner | Admission test |
|---|---|---|
| **0. GGUF metadata** | exporter | A host or the engine must branch on it, and only the checkpoint knows it. |
| **1. driver Lua** | exporter | Orchestration over *this* model's graphs and constants. |
| **2. engine C++** (`loom::<task>::`) | engine | Identical for every family under the task; branches only on **declared** facts, never on a spelling, a name, or an architecture; two or more hosts would otherwise implement it — and, per §1, implement it differently. |
| **3. host** (loom-py, CLI) | host | Typed results, I/O, and anything needing the host's ecosystem (numpy, wav files, a G2P package). Per-**task** code is allowed here; per-**architecture** code is not. |

The forbidden thing is unchanged and is worth restating in the new vocabulary: **per-task code may live
in any layer; per-architecture code may live only in the exporter.**

---

## 3. Tier 0: what the file must declare

This is the linchpin and the only part with no workaround. `loom_exporter/tasks.py` currently argues
the opposite — *"A task name is a CLI argument, not something stored in an exported GGUF"* — and that
was right when no host offered a task-shaped door. It stops being right the moment one does: a door
that a caller can knock on must be answerable from the file.

Proposed keys. Every reader is `has_kv`-guarded with today's behaviour as the fallback, so old GGUFs
keep working and the sweep can be incremental.

**Keys that already exist are NOT renamed.** `loom.n_samples`, `loom.sample_rate`, `loom.txt_len`,
`loom.n_audio_ctx` and `loom.n_text_ctx` are read under their own names, because renaming a declared key
costs a re-export of every model to buy nothing. `loom.sample_rate` needs no input/output qualifier for
the same reason it never did: the modality pair says which side the audio is on. Only a model with audio
on *both* sides would need two, and none exists yet.

```
loom.task                    str    canonical name from tasks.py
loom.entry_points            str[]  Lua functions the driver defines ("infer", ...)
loom.input.kind              str    "text" | "token_ids" | "phoneme_ids" | "audio" | "image"
loom.output.kind             str    "text" | "token_ids" | "audio" | "class" | "embeddings"
```

ASR decode table — what turns `transcribe` from Whisper-flavoured into per-task:

```
loom.asr.timestamp_first_id  i32    absent = this model emits no timestamps
loom.asr.timestamp_step_sec  f32
loom.asr.control_ids         i32[]  drop before detokenizing (<|notimestamps|>, ...)
loom.asr.language_names      str[]  parallel arrays rather than a map, because that is what GGUF stores
loom.asr.language_ids        i32[]
loom.asr.task_names          str[]
loom.asr.task_ids            i32[]
loom.asr.prev_context        u32    prev_tokens cap        (falls back to loom.n_text_ctx)
loom.asr.chunk.*                    segmented-prefill arithmetic (Qwen3-ASR 1 s/13, Granite 12 s/120)
```

Text front end and TTS synthesis:

```
loom.text.frontend           str    "vocab" | "phonemes"
loom.text.phoneme_alphabet   str    "ipa" | "arpabet" | ...
loom.text.languages          str[]
loom.phonemizer.ruleset      str    the rule-set version this export was validated against (§5)
loom.tts.default_steps       u32    sampler steps when the caller names none
loom.tts.voices              str[]  named styles the file carries
                                    (loom.default_style.{ttl,dp} is the one-voice case of this)
```

`include/loom/core/model_contract.h` is the one place that knows these names, and every reader is
absence-tolerant: a file that declares nothing gets what the engine already inferred, and `declared()`
is how a host tells the two apart. The fallback is a migration measure with an end — when the fleet is
re-exported it becomes dead weight, and removing it should be a deliberate commit.

Two things this buys immediately, beyond dispatch: the CLI's three tokenizer-tag dead-ends become one
`loom.task` switch, and a second timestamped ASR family costs zero C++.

---

## 4. Tier 2: what the engine owns, per task

```
loom::text::generate(bridge, model, prompt_ids, opts) -> ids        NEW — one LM loop, both driver shapes
loom::audio::transcribe(bridge, model, waveform, opts) -> Transcription   EXISTS (#5)
loom::speech::synthesize(bridge, model, ids, opts) -> Waveform      NEW — thin: pad the text axis,
                                                                    apply declared sampler/voice defaults
```

`loom::text::generate` is the smallest and most overdue: it is the loop that exists twice and disagrees
with itself. It absorbs both driver shapes (returns-a-sequence vs returns-one-token, told apart by the
return type, as both hosts already do), the file's own EOS, and the `prompt.append(token)` regrow. The
CLI and loom-py then both delete a loop.

It is also the **reuse point for speech-LMs**, which is the cleaner answer to "ASR models with causal
backbones": the decode loop is a component the ASR path *calls*, not a second public door bolted onto
an ASR model. A Qwen3-ASR model's public door stays `transcribe`; `generate` on it would be a method
whose prompt requires audio it has no parameter for. Reuse belongs inside the tier, not in the API.

`loom::speech::synthesize` stays deliberately thin, because the samplers are already Lua and must stay
there (the ADPM2 diffusion sampler and the CFM Euler loop are per-checkpoint orchestration by the §2
rule, and `lua_bridge.h` already argues that if ADPM2 did not need C++, no orchestration shape does).
What the engine adds is only what every host would otherwise redo: padding to the declared text axis,
filling in declared defaults for steps/seed/voice, and returning a waveform with its declared rate.

---

## 5. Tier 3: the Python surface

The low-level API is unchanged and stays raw: `infer`, `call`, `tokenize`, `detokenize`. What is added:

```python
model.task            # "automatic-speech-recognition" — declared, not guessed
model.capabilities    # ("transcribe",) — which high-level doors this file answers

model.generate(prompt, *, max_new_tokens=64, ...) -> str
model.transcribe(audio, *, language=None, task=None, timestamps=False) -> Transcription
model.synthesize(text=None, *, phonemes=None, tokens=None,
                 voice=None, steps=None, seed=None, language=None) -> Audio
```

**One `Model` class with task-named methods, not task subclasses.** `from_pretrained` keeps one return
type; a model that legitimately sits under two tasks is not forced into a false hierarchy; and calling
the wrong door raises an error naming the file's actual task, which is a better failure than an
`AttributeError` on a method that was never generated.

Two symmetry rules, so the fourth door does not need this document:

* **Every door accepts the natural type and the intermediate one.** `synthesize(text=...)` for a model
  with a text front-end, `synthesize(phonemes=...)` when the caller has their own G2P,
  `synthesize(tokens=...)` for ids. Same ladder as `generate`/`generate_ids`.
* **Every door returns a result object, never a bare value.** `Transcription` keeps the times the model
  computed (#6's argument: they cannot be re-derived afterwards). `Audio` carries `sample_rate` with
  the samples, plus `.save(path)` — a bare float list whose rate the caller has to remember is the same
  defect one modality over.

Python-only, correctly: `loom.audio` (wav read/write, resample-to-declared-rate), numpy interop, and
the G2P plugin below.

### Canonical input names

A high-level door passes **canonical names only**, and the canonical name follows from the kind or the
role rather than from which model it is. Three tiers, and the third is the boundary between the two
APIs rather than a gap in the first:

| | name | fixed by |
|---|---|---|
| primary input | `waveform` / `tokens` | `loom.input.kind` |
| recurring roles | `language`, `task`, `timestamps`, `prev_tokens`, `max_new_tokens`, `eos_token`, `seed`, `steps` | the role |
| model-specific knobs | `noise_scale_w`, `ref_s`, `style_ttl`, … | nothing — they stay per-model |

**A knob with no canonical role is not part of the high-level API**, and `infer` is what it is for. That
also answers "what does a new family cost the API": nothing, unless it introduces a new *role*.

Every driver accepts the canonical name for any role it has. `caller_input()` does that for a
synthesized driver at each read site; for a driver adopted from hand-written Lua the builder emits one
alias at the top of `infer` from the family's declared `driver_primary_input()`. The private name is
never written into the GGUF — the canonical one is the only public name, and a file that published its
own spelling would invite hosts to use it.

ASR already agreed on all six of its roles before any of this. TTS did not: `n_steps` (Matcha,
Supertonic) and `diffusion_steps` (StyleTTS2) are one concept spelled twice, and normalising them is
the remaining work in this section.

### The phonemizer, and how this splits Task #79

Task #79 is currently one blocked item: VITS, Kokoro, StyleTTS2 and Matcha take phoneme ids, real
phonemization is GPL-3, and vendoring it was rejected. It is actually two items, and only one of them
is blocked:

1. **The phoneme symbol table is data in the checkpoint and is simply not exported.** Export it as a
   vocabulary family (`tokenizer.ggml.model = "phonemes"`, `loom.text.frontend = "phonemes"`) and
   `model.tokenizer` stops being `None` for four families, `tokenize("ˈhɛloʊ")` works, and
   `synthesize(phonemes=...)` becomes a real door. No licence question is involved at any point.
2. **Grapheme→phoneme is the only part that needs an external engine**, and it can be an *optional
   Python dependency* resolved through a registry:

   ```python
   loom.phonemizers.register("ipa", fn)     # phoonnx, or whatever the user has
   ```

   With `loom-py[phonemes]` installed, `synthesize(text=...)` works. Without it, it raises with an
   actionable message while `synthesize(phonemes=...)` and `infer` keep working. Nothing GPL is
   vendored or linked — the user's own environment supplies it — so the repo stays MIT either way.

Supertonic is unaffected either way — it is grapheme-native and needs neither.

### The native phonemizer: scoped, and no longer licence-blocked

**Decided (2026-08-15): Python-only for now, with the CLI's text door scoped as the target.** The
intended mechanism is a C++ port of [`orthography2ipa`](https://github.com/TigreGotico/orthography2ipa),
vendored as a submodule — **Apache-2.0**, which is permissive and compatible with this repo's MIT, so
the constraint that killed espeak-ng and piper-phonemize does not apply. Task #79 is unblocked; what
remains is work, not a licence.

It also arrives shaped the way this document argues everything should be. Its own description —
*"every language is a JSON file describing which graphemes map to which IPA phonemes [...] a shared,
language-agnostic engine (tokenizer, beam search, allophone rules, stress, sandhi) turns that data into
transcriptions"* — is the interpreter/data split of §2, already made by someone else: rule-based
transduction over ~900 language JSON specs, no weights.

**Where the rule data lives — DECIDED (2026-08-15, user direction): in the engine, one copy.** The
reason is a single source of truth that is improvable across the board: bumping the `orthography2ipa`
submodule improves phonemization for every model at once, for every host, with no re-export and no
per-model action. This is §2's corollary stated positively — the rules for English are a property of
*English*, not of the Kokoro checkpoint, and a phonemization bug is precisely the kind of thing that
gets fixed repeatedly.

* **Rules ship with the engine**: submodule + JSON, loaded on demand by language code, subsettable at
  build time for an edge target that wants two languages rather than nine hundred.
* **The file declares only what it needs**: `loom.text.phoneme_alphabet`, `loom.text.languages`, and
  its own symbol→id table (step 4).

Rejected, and worth recording because it is genuinely attractive: embedding each model's declared
languages at export, which would preserve loom's signature "the model is one file" property for a cost
of one to three languages' JSON per GGUF. It buys self-containment and pays with the re-export
corollary — every rule fix becoming a fleet re-export — which is the wrong side of that trade for data
that is expected to keep improving.

**The consequence to design for, since it is the flip side of the same property:** if the rules can
improve under a model, then upgrading loom can change the audio a model produces for the same text.
That is desirable and it must not be *silent*. So the file should also declare the rule-set version its
export was validated against, and a mismatch should **warn, not fail** — enough to make an output
change attributable rather than mysterious. The same version has to be pinned in whatever test asserts
synthesis output, or the suite fails on every upstream bump with no signal as to whether the change was
an improvement.

**Two risks, both measurable before any C++ is written.**

*Symbol coverage — a mapping problem, not a quality one.* The four phoneme families were trained
against **espeak-ng** IPA, and `orthography2ipa` is a **superset** of the union of espeak-ng, Epitran
and others, harmonized across references and validated against the academic literature. The *quality*
risk of feeding a model a richer, better-grounded transcription than its training front-end produced is
low, and the empirical reason is strong: these models degrade gracefully, to the point that Piper
performs decently in some languages with graphemes substituted for phonemes outright.

Being a superset is what leaves work to do, though, and it is not the same work. A symbol outside the
checkpoint's fixed id table has **no id at all** — that is a lookup with no answer, not a
mis-transcription the model rides out. Graceful degradation covers wrong-but-present symbols; it cannot
cover absent ones. So what is required is a defined **fold-down** into each checkpoint's own inventory
(diacritic stripping, nearest-phoneme collapse) and a stated policy for whatever remains unmapped —
fold, drop, or raise. Cheap, mechanical, and measurable the moment the symbol tables are exported.

*Determinism.* The engine returns ranked lattices from a beam search. Synthesis needs one string, and
the CLI and loom-py must pick the *same* one — an undefined tie-break is how the two hosts drift, which
is the failure this whole document exists to stop. The C++ port must pin the selection rule, and the
Python path must use the same one.

*What follows from both:* **build it in Python first, then port.** The Python `orthography2ipa` package
exists today and the symbol→id tables land in step 4 regardless, so the fold-down can be defined and
its residual measured — phonemize a corpus, count what falls outside each checkpoint's table — with no
engine work at all. `piper_phonemize` remains available as an A/B reference for spot-checking a
language, which is a sanity check rather than a gate.

The Python door is therefore not a stopgap: it is the oracle the C++ port is verified against, the same
relationship `fixture_gen/`'s reference forwards have to the exported graphs. Port against a fold-down
that is already pinned down and a corpus whose expected output is already recorded, and "did the port
work" is a diff rather than a listening test.

---

## 6. The special cases, worked through

**Whisper's timestamp chunking is not special.** It is the fixed-clip branch of one loop, selected by
two declared facts: `input.clip_samples` present (window) and `asr.timestamp.first_id` present (seek on
the model's own boundaries rather than a fixed stride). Both branches already exist in `transcribe.cpp`;
the change is that they are chosen from declared data instead of from a token spelling. Segmented-prefill
ASR (Qwen3-ASR at 1 s/13 frames, Granite at 12 s/120) is the same loop again with the chunk arithmetic
declared instead of derived — which is the case that proves the metadata is load-bearing, since no
amount of cleverness recovers those numbers from the file as it stands.

**Speech-LMs**: §4 — the LM loop is a component, not a second public door.

**Supertonic and grapheme front-ends**: with `loom.text.frontend = "vocab"` it goes through
`synthesize(text=...)` with no special case at all. The backlog's standing warning — that a second
grapheme TTS model must not add a second `supertonic_text_vectorizer.cpp` — is served by the same
declaration: the codepoint table is already data in the file, and the normalization pipeline becomes
exporter-emitted data read by one front-end rather than a second C++ class.

**CTC collapse and transducer decode** stay in Lua. They are per-checkpoint orchestration over that
model's own graphs, they are already there, and they work.

---

## 7. Sequence

Each step ships alone and leaves the tree working; no step requires a second re-export of the same
models.

1. **Exporter — declare the contract.** `loom.task`, the input/output kinds, the ASR decode table.
   Sweep re-exports; the snapshot diff should show *added KVs only*, every topology, driver and tensor
   unchanged — the same shape the Supertonic tokenizer KVs landed in.
2. **Engine — `loom::text::generate`.** Both hosts call it, both delete their loop. This is the fix for
   the divergence in §1 and is worth doing first among the engine work because it is the one with a
   demonstrable wrong answer today.
3. **Engine — `transcribe` reads the declared table** instead of spelling `<|0.00|>` / `<|en|>` /
   `<|notimestamps|>`. Whisper's numbers land in the file; no behaviour changes; the next ASR family
   costs no C++.
4. **Exporter — phoneme symbol tables as a vocabulary family.** Four TTS models gain a real tokenizer.
5. **Engine + loom-py — `synthesize` and `Audio`**, plus `loom.audio` wav I/O.
6. **loom-py — the G2P registry** and the `[phonemes]` extra, backed by the Python `orthography2ipa`.
7. **Define the fold-down, before porting anything.** Map o2ipa's superset into each checkpoint's own
   inventory, state the policy for what stays unmapped, and record a corpus's expected output as the
   port's oracle. No C++ needed, and it is what makes step 8 a diff rather than a listening test.
8. **Engine — `orthography2ipa` as a submodule**, ported: the language-agnostic interpreter in C++, the
   JSON specs as engine-side data, the beam-search tie-break pinned so both hosts agree.
9. **CLI — a synthesis flag**, so the TTS door exists in both hosts and cannot drift the way the LM
   loop did.

Steps 1–3 are worth doing regardless of whether TTS lands, because they pay for themselves on ASR
alone. Steps 4–7 are the Python-only TTS door and are independent of 8–9.

## 8. What this does not change

The engine does not grow per-model code; it grows three per-task entry points, which is the category
`loom.h` already carries CTC decoding and tokenization under. loom-py does not gain per-architecture
code; it gains per-task methods keyed on a declared string. The exporter keeps every per-model fact,
and gains the job of *writing down what it already knows* — which is the cheapest of the three.

## 9. Decided, and still open

**Decided (2026-08-15, user direction):**

* **`loom.task` goes into the GGUF.** This reverses `tasks.py`'s written position — *"a task name is a
  CLI argument, not something stored in an exported GGUF"* — which was right while no host offered a
  task-shaped door and stops being right the moment one does. Everything else here follows from it.
* **Python-only text door first, the CLI's scoped as the target**, via a C++ port of the Apache-2.0
  `orthography2ipa` as a loom.cpp submodule (§5). The licence blocker on Task #79 is gone.
* **Phonemization rules and data live in the engine, one copy** — single source of truth, improvable
  for every model at once by bumping the submodule. Per-GGUF embedding rejected (§5).

* **A rule-set version change is allowed to change a model's output**, and warns rather than fails
  (§5). `orthography2ipa` being a literature-validated superset of the engines these checkpoints were
  trained against is why the direction of such a change is expected to be an improvement.

**Still open:**

* One `Model` with task-named methods (recommended) vs. task subclasses. The last one.
