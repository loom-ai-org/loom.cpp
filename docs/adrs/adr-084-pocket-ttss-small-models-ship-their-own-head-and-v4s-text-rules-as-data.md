---
type: adr
status: accepted
date: 2026-10-23
domain: model-coverage
tags: [pocket-tts, family-9, flow-head, gqa, text-frontend, sentencepiece, driver, gate]
---

# ADR-084: Pocket-TTS's Small Models Ship Their Own Head, and v4's Text Rules as Data

Serves [Epic-03](../epics/epic-03-model-coverage.md) (family 9's fifth leaf) and
[Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md). Builds on
[ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md),
[ADR-043](adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md),
[ADR-044](adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md) and
[ADR-045](adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md).

## Context

Kyutai released three distilled English Pocket-TTS models (`english_{18m,7m,2m}_refl_2026-10`, the
reward fine-tuned ones; `kyutai/pocket-tts@db04f0e`), with v4.0.1 of the reference. Against the
published `english_2026-09` they change seven things: a one-step flow head with **no time condition**
(`drifting`: the sample is `head(noise)`; `pointwise`: `noise + head(noise)`), grouped-query attention
with queries narrower or wider than the model (`attn_dim`, `num_kv_heads`), a low-rank text table
(`rank` -> bias-free `proj`), a bias-free `z_proj` that the flow and EOS heads read, a separate,
smaller Mimi decoder, an EOS threshold of -1, and a SentencePiece `.model` tokenizer. v4 also changes
the text path for every model: a config's `replace_characters` (rewrite characters, collapse whitespace,
drop a `,;:` right after `.!?…`), a decimal-period rule that sees the space in "3. 14", and a 5 ms
fade-in at the start of every chunk.

## Options

1. **One driver, branching on the head type at run time.** Rejected by the export itself: every
   `run_subgraph` call site is checked against its topology's inputs both ways, so the LSD call
   (`s`, `t`, `n_steps`) fails to link against a graph that takes `c`, `x`, `x_scale`, even inside a
   branch that never runs. Giving the one-step graph inputs it ignores would make the check lie.
2. **One driver, the head call in a per-export helper fragment.** Chosen.
3. **v4's text rules restated in C++ and switched on by a version number.** Rejected by ADR-041: the
   rules are data. The character table is the config's; the two mark classes and the decimal rule are
   probed off the reference in use.

## Decision

* **The head call is `pocket_flow_head(x, std, n_steps)`**, defined in one of two top-level fragments,
  `00_flow_head_lsd.lua` or `00_flow_head_one_step.lua`. The export ships only the one matching the
  config's `flow.type`, so the driver's single `flow_head` call matches its graph. The one-step graph
  (`OneStepFlowHeadPhase`) takes `(c, x, x_scale)` and does the residual or not in the graph.
* **The wrappers read v4's knobs off the modules**, defaulting to v3's shapes: `_attn_dims` and
  `_split_in_proj` for `attn_dim`/`num_kv_heads`, HF's `repeat_kv` spelling so `fuse_gqa_repeat_kv`
  strips it (the cache holds the KV heads, the rows a voice state has), `proj` and `z_proj` when present.
* **v4's text rules ship as optional keys** (`tokenizer.ggml.pocket_tts.char_replace_from/_to`,
  `sentence_marks`, `stray_clause_marks`, `decimal_period_rule`). The mark classes come from running
  `prepare_text_prompt` over every punctuation character. The decimal rule comes from asking
  `_is_decimal_period_boundary` about "3. 14". A file without the keys keeps v3's behaviour, and an
  export against v3.2.0 writes rule 1 and no table.
* **The fade-in ships only when the reference applies it**, read off `_decode_audio_worker`'s source: a
  120-sample ramp as a driver weight, multiplied in Lua (exact: an f32 product in a double, rounded once).
* **The reference is chosen by `LOOM_POCKET_TTS_REPO`**, defaulting to the v3.2.0 clone.
* **The gate's free-running bound is 3x the fixture's own f32-vs-f64 rmse**, never tighter than 1e-3
  ([Retro-055](../retros/retro-055-a-feedback-loop-cannot-be-gated-free-running.md)). On one sentence
  and eight pinned seeds, the 7m model's reference drifts 4.6e-04 to 2.5e-02 between f32 and f64, and
  loom drifts 0.24-0.98x of that. The sample count stays exact, and teacher forcing stays the tight arm.

## Consequences

* **Measured** (x86, pinned draws, the reference's default text): teacher-forced max |d| 6.3e-05
  (18m), 2.2e-05 (7m), 2.5e-05 (2m), with the same EOS frame free-running; voice files 1.4e-05 and
  5.2e-06. Tokenizer differential against v4: 3000/3000 (18m, 7m), 9000/9000 (2m). Against v3.2.0 on
  the published file: 6000/6000. Sabotage: the dropped `pointwise` residual fails teacher-forced at
  1.19; swapped KV heads never reach EOS; a GGUF without the new keys diverges from v4 on the
  differential; the engine with both rules disabled fails 9 of 51 checks of `test_pocket_tts_vocab`.
* **Size.** The export drops Mimi's encoder, so the GGUFs are 77 / 29 / 8.5 MB at F32: the first
  Pocket-TTS that fits a Pi Zero at F32.
* **Re-exporting the published `english_2026-09` against v3.2.0** reproduces its five topologies and
  200 tensors byte for byte. Only the driver text (the head is now a fragment) and the one new key
  (`decimal_period_rule = 1`, v3's behaviour) differ.
* **Publishing these needs an engine release.** The text rules are new C++ in `PocketTtsVocab`, and an
  older engine ignores the keys (it reads neither), so it would tokenize v4 text the v3 way.
