# Handover: family 10 (Dia) — what is left

**Temporary. Delete it when the three items below close** — the same way P4.18's handover was deleted
when that work closed. Nothing here is the source of truth: it is a *sequence*, plus pointers to the
tiers that are. If this document and the hub disagree, the hub is right.

**Most of what this file used to hold has moved**, because it is no longer open work:

| | |
|---|---|
| open work | [`backlog/active-index.md`](backlog/active-index.md) — the three remaining items, each with its own reasoning |
| what family 10 is and what it cost | [Epic-03 §2](epics/epic-03-model-coverage.md) — "Family 10, and the axis that did not have to be padded" |
| why the text axis is not padded | [ADR-021](adrs/adr-021-dias-decoder-resolves-two-dynamic-axes.md) |
| the tracing lesson | [Retro-030](retros/retro-030-a-guard-that-could-not-fire.md) |
| why codec tokens are their own modality | [ADR-020](adrs/adr-020-audio-codes-is-its-own-modality.md) |
| one door per task | [ADR-013](adrs/adr-013-one-door-per-task.md) |

---

## 1. State

**Dia exports and drives end to end.** `loom-export ~/Dev/models/dia-1.6b -o dia_mil.gguf` produces a
6.1 GB F32 GGUF that declares `loom.task = "text-to-codes"`, resolves to the `text2codes` interface,
carries its own byte vocabulary, and whose driver runs the whole loop: byte encoder once, 36
cross-attention K/V projected once, then a nine-channel KV-cached decode with the delay scaffold and
the realignment back to audio frames.

| | |
|---|---|
| exporter | `loom_exporter/dia_export.py`, `loom_exporter/dia_driver/*.lua` |
| exporter tests | `tests/ci/test_dia_export.py` — 14 cases, hermetic (a tiny randomly-initialised Dia) |
| engine test | `loom.cpp/tests/gate/test_e2e_dia_mil_export.cpp` — the whole family in one call, against a `transformers` reference |
| engine change | `ByteVocab` generalised — `tokenizer.ggml.byte_offset` / `add_eos_token` / added tokens, all defaulting to ByT5's behaviour |

**Where the artifacts are.** Nothing below is in the repo, and re-deriving both costs about fifteen
minutes:

* `dia_mil.gguf` — 6.4 GB, F32, unquantized, at `~/.claude/tmp/dia-export/dia_mil.gguf`. Rebuild with
  `loom-export ~/Dev/models/dia-1.6b -o <path>` (~10 min). The gate test finds it through
  `LOOM_DIA_MIL_GGUF`, or through `$LOOM_FIXTURES/dia_mil.gguf` by the derived rule.
* the `transformers` reference the gate compares against — regenerate with
  `~/.venvs/piper/bin/python scripts/dia_reference_codes.py --model ~/Dev/models/dia-1.6b --frames 32`,
  which prints the declarations to paste into the test. **Verified to reproduce the committed numbers
  byte-for-byte.** That script is the durable form of the capture; it exists because family 2's
  equivalent does not, and `test_e2e_lfm2_mil_export.cpp` says "see the conversation/PR description
  for the capture script", which is a reference nobody can re-derive.

The one engine change was the tokenizer, and it was the answer to the question the previous handover
asked first: **`byte_vocab` did not cover Dia.** ByT5 puts byte 0 at id 3 and always appends eos; Dia
puts byte 0 at id 0, appends nothing, and has `[S1]`/`[S2]` at ids 1 and 2 — *inside* its own byte
range. The generalisation carries the constants as KVs and reuses `BpeVocab`'s added-token scan
verbatim; every pre-existing ByT5 file loads and tokenizes identically.

## 2. Environment — the things that bite in the first ten minutes

* **Use `~/.venvs/piper`.** `python3` resolves to `~/.venvs/ovos`, which is the Qwen3-ASR-only
  environment. Never upgrade piper: NeMo pins `transformers~=4.53` and it currently holds 4.57.6,
  torch 2.8.0, coremltools 9.0.
* **Always `TMPDIR=/home/flavio/.claude/tmp`.** `/tmp` is small and the suite writes tens of GB.
* **`Dev/models` is a symlink to an external drive** at 100%. Write exported GGUFs to `/home`.
* **A standalone `ct.convert` probe is not this exporter's conversion.** `import loom_exporter`
  installs the coremltools frontend patches, and one of them (`_cast` folding a 1-element array) is
  what makes Dia's decoder convert at all. A probe without it reports blockers that do not exist —
  see [Retro-030](retros/retro-030-a-guard-that-could-not-fire.md)'s closing note.
* `parler_tts` **cannot** be installed into piper (pins `transformers==4.46.1`; `--no-deps` pulls in
  `protobuf<3.20` through `audiotools`, which breaks coremltools). Verified and fully reverted.

## 3. The sequence

Reordered from the obvious order, and the reason is in step 3: sampling turns out to carry engine work,
while the composition does not. Each step names the check that closes it. **A step is not done because
it ran.**

**1. The composition with DAC.** Independent of everything else and the actual point of the family, so
it goes first. Decide deliberately: one GGUF with DAC merged (~6.6 GB, loom's "the model is one file"
property) or two files chained by the host. Either way:

> text → Dia → 9 delayed code streams → realign → DAC → waveform,
> against `transformers` running the identical pipeline.

It does **not** wait on step 3: run both sides greedy and CFG-free and the comparison is still exact,
which is what the current gate already demonstrates on the codes. `codec.n_codebooks` is written on
both sides under the same key, so a host can match them.
*Closes when:* the waveform matches its reference at two clip lengths, the way family 11's did.

**2. `Text2Codes` in loom-py.** Small and independent. The interface already resolves from the contract
— `Model.contract()` on the exported file returns `interface: 'text2codes'` — and nothing implements
it. Follow `Codes2Speech`, which family 11 added for the other half of this pair.
*Closes when:* an arm in loom-py's `tests/gate` drives the real file through the door.

**3. Sampling and classifier-free guidance — and this one has ENGINE work in it.** The previous
version of this document said `loom.sample_row` "already exists and takes the knobs", implying a
driver-only change. That is wrong, in two places:

* **`loom.sample_row(module, row, opts)` samples the WHOLE row.** There is no range-restricted form,
  while `argmax_row_range` — which the driver uses today — has one. Dia needs sampling restricted to
  `[0, EOS)` for channels 1–8 and `[0, EOS]` for channel 0, or channel 8 can emit PAD or BOS. So this
  needs a `sample_row_range` binding, symmetric with the argmax pair.
* **CFG combines LOGITS**, `uncond + scale * (cond - uncond)`, and those live in the engine's retained
  output buffer. No existing binding composes two retained outputs, so it is either a second primitive
  or 9252 floats × 2 crossing the Lua boundary every step — which is exactly the cost every retained
  reduction in this tree exists to avoid. CFG also means a second encoder pass over empty text and a
  second decoder call per step.

Both are per-*task* reductions rather than per-model logic, so they belong in the engine by
[ADR-003](adrs/adr-003-per-model-complexity-in-the-exporter.md) — but note what it costs: "family 10
needed no engine C++ for its graph" stays true, and stops being true of its sampler. Once sampling is
live, the two clauses of `DiaEOSChannelFilterLogitsProcessor` that are no-ops under an argmax (force
EOS when it is already top-1; suppress it when it is not) become real and must be implemented.

`scripts/dia_reference_codes.py` must learn the same algorithm **on the same commit**, or the gate
quietly starts measuring two samplers against each other instead of the export against `transformers`.
*Closes when:* the ASR oracle transcribes the output and the words match — not when a cosine looks good
([Retro-006](retros/retro-006-kokoro-shipped-noise.md)). **Nothing has been listened to yet.**

**4. Quantize, catalogue, card.** Last, and genuinely blocked on step 3: a model card shipping greedy,
CFG-free output would be [Retro-006](retros/retro-006-kokoro-shipped-noise.md) repeating itself. 6.1 GB
F32 is the unquantized export. Then a row in the export sweep, an entry in `build_model_cards.py` (the
`text-to-codes` task is claimed now, not reserved), and an arm in loom-py's model-card gate.

## 4. Traps this thread has already paid for

* **Verify a tracing fix by tracing it.** The previously recorded `rotate_half` patch was verified
  eagerly and could not run under a trace at all. Count `aten::Int` in
  `traced.inlined_graph` — [Retro-030](retros/retro-030-a-guard-that-could-not-fire.md).
* **`nil` is not an error in Lua, it is a shorter array.** The realignment read one row past the end
  when the loop stopped on its row bound rather than on EOS, and `_codes[#_codes + 1] = nil` appends
  nothing — 186 codes came back where 189 were due, with no frame boundary to notice it at. Found by
  running it on a synthetic checkpoint that never emits EOS. The driver now raises there.
* **An export that runs is not an export that works.** DAC's first working version returned one
  frame's worth of audio for every input and nothing raised. Assert on the emitted SHAPES, and export
  at two trace lengths requiring an identical topology — `test_the_traced_lengths_do_not_reach_the_graph`
  varies *both* of Dia's axes, because one baked axis is enough and either could be the one.
* **A gate that cannot fail proves nothing.** A test helper that derived `num_channels` from
  `delay_pattern` made the delay-pattern check unable to fail; it was caught only because the assertion
  then never fired. Sabotage every check and confirm it goes red.
* **Tensor oracle, not token oracle.** A wrong encoder still decodes a plausible transcript.
* **`build_model_cards.py` snippets are Python**, so they may contain braces — `render_snippet`
  substitutes named placeholders rather than `str.format`.
