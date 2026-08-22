---
type: adr
status: accepted
date: 2026-08-15
tags: [api-design, host-bindings, model-contract, layering]
---

# ADR-013: One Door Per Task, Declared by the File

## Context

`generate` was the only high-level entry point, and it had never been *placed* by a rule — it was the
causal-LM task's door, added when that was the only task with one, and named as if it were universal.
Adding `transcribe` forced the question from first principles, and TTS would have forced it again.

What was actually broken underneath:

* **The causal-LM decode loop existed twice and had drifted.** The CLI ran the full `--n-predict` with
  no EOS stop, took `vec[0]` where the new token is last, and silently rewrote any id `>= 65536` to
  `0`; `loom-py` stopped on the file's own `eos_token_id`, took `vec[-1]`, and stripped the stop token.
  Same model, same driver, **two different transcripts depending on which host you asked**. None of the
  three differences was a decision.
* **Hosts inferred what a model IS from its tokenizer tag**, so a complete Supertonic TTS model hit a
  branch that existed to stop it being parsed as token ids.
* **`transcribe.cpp` was per-task in shape and per-family in its constants** — Whisper's spellings, not
  ASR's — so the second timestamped family would have cost engine code.
* **The bridge/cache setup existed three times**, and one copy attached a `KvCache` and no
  `ConvStateCache`.

The common cause is one sentence: **nothing in a GGUF said what contract it implements.**

## Options Considered

1. **Per-architecture branching in each host.** What existed; it does not survive the second family and
   it drifts between hosts.
2. **A door per model family.** Multiplies the API surface by the model count.
3. **A door per *task*, dispatched off a contract the file declares.**

## Decision

**One door per task, chosen by the modality pair the file declares** — X2Y interfaces off
`loom.task` plus the declared input/output modalities, with canonical input names. A knob with no role
in the declared contract stays `infer`-only.

**The layering rule** (`docs/HIGH-LEVEL-API.md` §2): in the **FILE** when it is a property of the
checkpoint; in the **ENGINE** when it is a property of the task; in the **HOST** when it needs the
host's ecosystem. With the corollary that settles the hard cases: anything shipped inside a GGUF can
only be fixed by re-exporting every model, so an evolving policy must not be baked into files even when
Lua could express it.

Net: **per-task code may live in any layer; per-architecture code only in the exporter** — the same
rule as [ADR-003](adr-003-per-model-complexity-in-the-exporter.md), one level up.

Every reader is absence-tolerant, and `declared()` separates a file that states its contract from one a
caller must know about, so pre-contract GGUFs keep working through a flagged legacy fallback.

## Consequences

* **Positive:** one decode loop, one session setup, one transcribe path. Three classes of host drift
  became impossible rather than fixed.
* **Positive:** a new family reaching an existing task needs no host code at all.
* **Negative:** the contract is now part of every export's obligations, and a wrong declaration is not
  caught by a numeric gate — [Retro-006](../retros/retro-006-kokoro-shipped-noise.md).
* **Negative:** the legacy fallback must stay tested. A declared-only test would stay green while the
  fallback every GGUF on disk depends on rotted.

## Related

* Authority: [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md)
* Epic: [Epic-06: High-Level API and Hosts](../epics/epic-06-high-level-api-and-hosts.md)
* Ledger record, verbatim:


Design: **`docs/HIGH-LEVEL-API.md`**, which is the authority; this entry is the ledger stub and the
record of what shipped.

**What prompted it.** loom-py #6 added `transcribe` and had to argue from first principles where a
high-level door belongs, because `generate` — the only one that existed — was never placed by a rule.
It was the causal-LM task's door, added when that was the only task with one, and named as if it were
universal. TTS needs a third door and the modality list does not stop there, so the placement question
was going to recur until it was answered once.

**What was actually broken, found while writing it up:**

* **The causal-LM decode loop existed twice and had drifted.** `tools/loom_cli/main.cpp` ran the full
  `--n-predict` with **no EOS stop at all**, took `vec[0]` of a list return where the new token is the
  last, and silently rewrote any id `>= 65536` to `0`. loom-py's `generate_ids` stopped on the file's
  own `eos_token_id`, took `vec[-1]`, and stripped the stop token. Same model, same driver, two
  different transcripts depending on which host you asked. None of the three differences was a decision.
* **Hosts inferred what a model IS from its tokenizer tag.** `loom_cli` branched on
  `tokenizer.ggml.model == "bert" | "byt5" | "supertonic"` and dead-ended each as inspection-only — a
  Supertonic GGUF is a complete TTS model reaching a branch that exists to stop it being parsed as
  token ids.
* **`transcribe.cpp` was per-task in shape and per-family in its constants** (`<|0.00|>`, `<|en|>`,
  `<|notimestamps|>` — Whisper's spellings, not ASR's), so the second timestamped family would have
  cost engine code.
* **The bridge/cache setup existed three times**, and `run_asr`'s copy attached a `KvCache` and no
  `ConvStateCache` — a speech model with ShortConv blocks would have thrown on its first SHORT_CONV
  node, the same failure P4.0.10 fixed for the umbrella header.

The common cause is one sentence: **nothing in a GGUF said what contract it implements**, so every
host-side high-level door was either impossible or per-architecture code.

**The rule adopted** (docs/HIGH-LEVEL-API.md §2): in the FILE when it is a property of the checkpoint;
in the ENGINE when it is a property of the task; in the HOST when it needs the host's ecosystem. With
the corollary that settles the hard cases: anything shipped inside a GGUF can only be fixed by
re-exporting every model, so an evolving policy must not be baked into files even when Lua could
express it. Net: **per-task code may live in any layer; per-architecture code only in the exporter.**

**Shipped in the engine (this branch):**

| | |
|---|---|
| `include/loom/core/model_contract.h` | the one place that knows the declared KV names; every reader absence-tolerant, `declared()` separates a file that states its contract from one a caller must know about |
| `include/loom/core/text_generate.h` | `loom::text::generate` — one LM loop, both driver shapes, the file's own EOS. CLI switched to it; the three CLI-only behaviours above are gone |
| `include/loom/core/session.h` | topologies registered and caches attached once, owned in an order that cannot dangle. Kills three copies including the one missing `ConvStateCache` |
| `transcribe.cpp` | reads the declared ASR table; the Whisper spellings survive only as a flagged legacy fallback for files that predate it |
| `audio_window.h` | header comment said the timestamp seek "stays in the CLI", which its own sibling in the same PR contradicted |

**Verified.** ci 60/60 (the new `test_model_contract` covers the declared file AND the legacy fallback,
because a declared-only test would stay green while the fallback every GGUF on disk depends on rotted).
gate 82/82 against `loom-engine-artifacts/v4`. Behaviourally on real files: `jfk.wav` through
whisper_mil transcribes identically with `--language en` (the legacy spelled-lookup path) and produces
one closed segment at `00:00:00.000 --> 00:00:11.000` with `--timestamps`; LFM2 generates coherent text
through the unified loop.

**A gate that could not fail, found on the way.** `LOOM_CHECK` only counts failures —
`LOOM_TEST_REPORT_AND_RETURN()` is what turns the count into an exit code. `test_e2e_qwen3_asr_mil_export`
and `test_e2e_granite_speech_mil_export` both ended with `return 0`, so every check in the two newest
ASR family gates was decorative: they printed "OK" and exited 0 no matter how many had fired. Both
fixed here. Caught only because the new test was deliberately sabotaged to confirm it could go red, and
did not — which is the argument for that habit rather than a coincidence.

**The seek truncated a sample index, and re-exporting the fixtures is what showed it.** A segment end
is a float number of seconds, so the sample index it maps to is almost never exact. With the timestamp
step read from the file as f32 (`loom.asr.timestamp_step_sec`), 550 steps of 0.02 s come to 10.99999975
rather than 11, and 11 s at 16 kHz truncated to 175999 -- one sample short of the end. The loop then ran
a second window over four samples of real audio plus 30 s of zero padding, and Whisper transcribed the
silence as `[BLANK_AUDIO]`, appended to the transcript. Rounded now, which is what a measurement with
error either side of it deserves; truncation was arbitrary regardless of where the float came from.

Invisible until the fixtures carried the declared table: the old path derived the step from three
hparams in double and absorbed the error by luck, so v4 gave one window and v5 gave two on identical
audio. Pinned by `tests/ci/test_transcribe_seek.cpp` now, against a fixture that emits timestamp tokens and
carries a vocabulary to detokenize them -- built for this, since the only ASR artifact that emitted
timestamps at all was a 970 MB Whisper export. The step is stored as **f32 on purpose**: storing it as
f64 would sidestep the precision loss and let the test pass against the truncating code it exists to
catch. Verified by re-introducing the truncation, which fails it on `windows == 1`.

### P5.0 status (2026-08-15, end of the thread)

**Done and verified end to end on real models:**

| | |
|---|---|
| the declared contract | every export writes `loom.task` + the modality pair; 17/17 models swept, added-KVs-only |
| `loom::text::generate` | one LM loop; both hosts call it; the CLI's three divergent behaviours gone |
| `loom::audio::transcribe` | reads the declared ASR table; Whisper declares timestamp ids, languages and tasks |
| `loom::Session` | one bridge/cache setup, replacing three including one missing `ConvStateCache` |
| `loom::PhonemeVocab` | the phoneme symbol tables all four families were carrying, now exported and read back |
| loom-py's X2Y layer | `text2text` / `speech2text` / `text2speech`, selected by the declared pair |
| gate fixtures | re-exported as `loom-engine-artifacts/v5`, 13 models, all declaring their contracts |

`text2speech.infer("hello world")` works on all five TTS families. Kokoro and Supertonic ship a
default voice; the other three take one through `infer`.

**Open, in the order it matters:**

1. **Four TTS families declare no sample rate** (VITS, Matcha, Kokoro, StyleTTS2). loom-py warns and
   falls back to a caller-supplied 16 kHz, which is wrong for all four -- they run at 22.05, 24 and
   24 kHz. Each needs it read from its own checkpoint. Silent when wrong, which is why it warns.
2. **`orthography2ipa` is not ported.** The Python path behind `loom-py-rt[phonemes]` is what runs
   today; Task #79 above holds the plan and the two risks.
3. **`loom.tts.voices` is empty everywhere, and cannot mean one thing across TTS models.** The five
   families obtain a voice three different ways, which is a property of what was traced rather than a
   gap:

   | | where the voice comes from | what selects it |
   |---|---|---|
   | StyleTTS2 | the traced `diffusion` topology samples it from noise; no `ref_s` input exists | `seed` |
   | Kokoro | passed in as `2*style_dim` floats; its diffusion is NOT traced | the vector (a default now ships in-file) |
   | Supertonic | two style tensors; default `F1` in-file, nine more as repo files | the vectors |
   | VITS, Matcha | single-speaker — the voice IS the weights | nothing |

   So `voice=` would select a vector for two families and a seed for one, and mean nothing for two.
   Worth settling before the key is filled in rather than after. Reference-audio cloning is out of
   scope for all of them for the same reason: it needs the style-encoder chain none of these exports
   traces.

4. **Gate fixtures live at `loom-engine-artifacts/v5`** — 13 models, all declaring their contracts,
   `ctest -L gate` green against them. v4 is the pre-contract set and is superseded; it is what the
   engine's legacy fallbacks are still tested against, so do not delete it without reading the
   `model_contract.h` fallback note first.

5. **Any recorded export-sweep baseline from before 2026-08-15 is stale.** Every model gained the
   contract KVs, so a sweep against an older baseline reports 17 diffs that are all expected. Re-record
   before using one to judge a change.
6. **The legacy fallbacks in `model_contract.h` are now dead weight** for any re-exported model, and
   deleting them should be its own commit once the published fleet is re-exported.

**Not done here, and the sequence for it** is docs/HIGH-LEVEL-API.md §7. Next: the exporter writes the
contract (§3), then loom-py's X2Y interface layer consumes it.

