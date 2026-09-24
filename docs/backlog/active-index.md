---
type: index
category: backlog
last_updated: 2026-09-24
---

# Active Ledger — Open Work Across All Three Repos

Only **open** work lives here, one line each. Closed work is not tracked: its decisions are
[ADRs](../adrs/), its lessons are [retros](../retros/), its architecture is in the
[epics](../epics/), and its execution detail is in git history or [`archive/`](../archive/).

**Item IDs are the existing `P`-numbers.** Code comments reference them (`P4.3e`, `P4.15b`), so they
are not renumbered. New items continue the scheme.

---

## Now — the next things to pick up

| item | why now |
|---|---|
| **Land the review stack, in order: P5.0, then family 9** | Nothing below should start on a branch that stacks on unmerged work. **P5.0** is three open PRs — loom-exporter **#23** (`feat/p5-0-pack-weights-per-phase` → `main`), **#24** (`feat/p5-0-phase-process-isolation` → #23's branch) and loom.cpp **#30** (`feat/p5-0-phase-process-isolation` → `main`). **Family 9** is pushed on `feat/p5-family-9-f5-tts` in all three repos and **has no PRs yet** (its fourth leaf, Chatterbox, is pushed with no PRs on `feat/p5-family-9-chatterbox` on top of it: loom.cpp `271243a`, loom-exporter `36980f4`, loom-py `ced4be1` pinning `vendor/loom.cpp` at `271243a`; its fifth, Pocket-TTS, is committed on `feat/p5-family-9-pocket-tts` on top of THAT: loom.cpp `c528244` (+ docs-only commits after it), loom-exporter `4db8c4f`, loom-py `6f3397e` pinning `vendor/loom.cpp` at `c528244`): the loom.cpp and loom-exporter branches stack on the P5.0 branches above (one commit each on top), and loom-py's is one commit on `main` pinning `vendor/loom.cpp` at `d914b25`. Two things to do before merging it: run loom-py's **model-card gate** against the family-9 build (never run — it proves the 30 shipped models did not regress through the `run_ode` change), and **if loom.cpp's PR is squashed, re-bump loom-py** to the squashed sha before loom-py's PR merges, or its pin names a commit that no longer exists → [Packaging & release](#packaging--release) |
| **P5 breadth is the work now — remaining TTS, then small classifiers, then music** | [Epic-03 §3](../epics/epic-03-model-coverage.md)'s coverage-per-effort order is **9/10 (remaining TTS) → 13 (small classifiers) → 14 (music)**. Families 4, 5, 6, 10, 11 and 12 are complete and **family 9 is at five of twelve leaves** (Matcha, Supertonic, F5-TTS, and Chatterbox and Pocket-TTS as of 2026-09-24). F5-TTS ended the run of families that needed no engine primitive: `loom.run_ode` had to learn classifier-free guidance ([ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md)). Chatterbox was the first AR-LM + flow composition, and it needed **no new template**: family 10's guided decode plus F5's sampler, in one driver. Its cost was sampler options and a text front end ([ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)). Pocket-TTS was the first loop over CONTINUOUS latents, and that loop needed **no primitive** either (a Lua loop around one flow-head call); its cost was a voice shipped as a KV cache ([ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md)) and a front end that chunks ([ADR-044](../adrs/adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)). **What to pick next, costed:** the other seven family-9 leaves are mostly the same composition (cosyvoice3 shares Chatterbox's HiFT vocoder and flow lineage but its speech tokenizer and CAMPPlus are ONNX-only; voxcpm2 predicts continuous latents per AR step like Pocket-TTS, with a LocDiT where Pocket has a one-step MLP head); family 9b's SpeechT5 is local too but decodes MEL FRAMES autoregressively, a loop shape nothing ships yet; family 13 is one forward pass and an argmax per leaf but needs every checkpoint downloaded and new contract output kinds (speaker embeddings, per-frame VAD probabilities). Estimate against [Epic-03 §2](../epics/epic-03-model-coverage.md): the bill lands where the scoping did not look, and for F5-TTS it was a layout JOIN between two verified graphs ([Retro-052](../retros/retro-052-every-phase-was-right-and-the-join-was-wrong.md)) |

**State anchor, 2026-09-17 — `1.0.0-rc10` is fully released and nothing in the release pipeline is
open.** Four packages on PyPI at `1.0.0rc10`, both macOS architectures included; the `linux_armv6l`
wheel rides as a **GitHub release asset** rather than a PyPI file, because PyPI rejects that tag at
upload (`wheels.yml` says so at the job). **Thirty** models are on
[huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org), etags and cards verified in both
directions, of which seven were first-time publishes: `sensevoice-small`, `paraformer-zh`,
`data2vec-audio-base-960h`, `hubert-large-ls960-ft`, the `qwen3-tts` pair and `encodec-32khz`.
EnCodec ships `cc-by-nc-4.0` and that is **settled, not pending** — the weights are MusicGen's output
by `facebook/encodec_32khz`'s own card, whatever the MIT code says. → [[loom-release-state]],
[Epic-08](../epics/epic-08-packaging-and-release.md)

**NOTHING STAGED IS PUBLISHED UNTIL 1.0.0-rc11 IS OUT.** Author's decision, 2026-09-17: staged model
updates wait for the next release even when they need nothing from it. So a freshly staged GGUF is
**not** a publishable one, whatever the coupling below says — the Hub and the staging tree are allowed
to diverge until rc11 ships, and today they do: `hf-models/qwen3-tts-12hz-0.6b` is the ICL build and
the Hub's copy is rc10's. That divergence is expected, and it does not survive the release either way,
because [[feedback-release-gate-needs-a-fresh-export]] requires a fresh export at rc11 time regardless
of what is sitting in the tree.

*The technical coupling, which still decides what rc11 must contain:* rc10 carried every binding the
staged drivers needed (`output_shape`, `run_ode_and_retain`, the retrace's retained-reference
bindings, a retained ROW range, `ELU`, `repetition_penalty`, the `ctc` and `funasr` vocabulary
readers, grouped `CONV_1D`), and Qwen3-TTS's ICL half needs nothing beyond them — verified by running
its GGUF on the released `loom-py-rt==1.0.0rc10` wheel from PyPI. The moment a model needs something
the released wheels have not got, **WHEELS FIRST, ALWAYS** is the harder constraint on top of the
rule above: a GGUF published before its wheel is a file nobody can run.

**F5-TTS is the first model that needs rc11 technically, not only by that rule** — and it is worse than
"nobody can run it". Its driver passes `guidance = {...}` in `loom.run_ode`'s options table, and the
rc10 engine reads that table field by field with no check for keys it does not know: an F5-TTS GGUF on
the rc10 wheels would integrate **unguided**, silently, and produce a different model's audio. The text
door fails loudly there (rc10's loom-py has no `"f5"` tokenizer), but ids passed straight to `infer` do
not. So rc11 must carry family 9's engine half before that file goes anywhere → [Packaging &
release](#packaging--release)

---

## Models

* [ ] **Chatterbox (family 9's fourth leaf) is built, verified and pushed (no PRs), and not yet
  published or gated on the Hub.** Branch `feat/p5-family-9-chatterbox` in all three repos, stacked on family 9's
  F5-TTS branches. Verified: the gate is 2.5e-05 from the reference waveform, the tokenizer is 3000/3000
  ids against the reference, and the Whisper oracle is exact at guided greedy and at two sampled seeds.
  Card entry `chatterbox` in `build_model_cards.py`, whose first limitation says why there is **no
  Perth watermark** (decided 2026-09-23; see Epic-03 §2). To publish: rc11 (below), a fresh export, and
  loom-py's model-card gate against it. The card's snippet is the plain text door, since the built-in
  voice needs no reference clip, so unlike Qwen3-TTS and F5 it IS executed by the gate. Voice cloning
  (voice encoder + S3 tokenizer + CAMPPlus) and the multilingual and Turbo checkpoints are separate
  leaves. *Context: [Epic-03](../epics/epic-03-model-coverage.md),
  [ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)*
* [ ] **Pocket-TTS (family 9's fifth leaf) is built, verified and pushed (no PRs), and not yet
  published or gated on the Hub.** Branch `feat/p5-family-9-pocket-tts` in all three repos, stacked on
  Chatterbox's. Verified: the gate is rmse 1.8e-06 teacher-forced and reaches the same EOS frame
  free-running ([Retro-055](../retros/retro-055-a-feedback-loop-cannot-be-gated-free-running.md)), and
  a voice FILE (`marius`) is 8.0e-06 teacher-forced; the text path is 8000/8000 ids against the
  reference, EOS-tail headers included; the Whisper oracle is exact on the reference's default text, a
  three-chunk paragraph and two voice files. Weights CC-BY-4.0 under Kyutai's use restrictions; each
  voice carries its recording's licence (two NON-COMMERCIAL). Card entry `pocket-tts` in
  `build_model_cards.py` (the plain text door, executed by the card gate, plus a voice table), and the
  repo is STAGED at `hf-models/pocket-tts/` with all 26 `voices/*.gguf`
  ([ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)). To
  publish: rc11 (below), a fresh export, and loom-py's model-card gate against it. The voice files
  stay valid across re-exports of the same checkpoint (the fingerprint is of its weights). *Context:
  [Epic-03 §2](../epics/epic-03-model-coverage.md),
  [ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md),
  [ADR-044](../adrs/adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)*
  * [ ] **Cloning a voice from a recording** needs the Mimi encoder (in the gated voice-cloning
    weights, zeroed in the other release) as one more phase feeding the text prefill's ordinary
    input. The same voice-cloning door F5-TTS needs.
* [ ] **The Qwen3-TTS talker's card is never EXECUTED by the model-card gate**, and it is the only
  voice-cloning row so this has no second example to be measured against. `test_the_card_runs` runs
  every `python` block of a published card in one namespace, seeding `audio` because "a card cannot
  ship a recording"; this card instead tells the reader to bring `reference.wav`, which is a
  legitimate precondition the harness skips on by design. The consequence is that `-k qwen3-tts`
  reports **2 passed, 14 skipped** and none of the passes ran the snippet — the ASR oracle included,
  since it grades the audio the card itself produced. Pre-existing (the x-vector snippet has the same
  shape) and surfaced by ICL's own verification, which had to be done outside the gate entirely.
  **What would close it:** either the harness seeds a 24 kHz clip under a name the card can use
  without lying to a reader, or the card's first block loads `audio` with a comment saying it stands
  for the reader's own recording. Worth deciding once, because every future voice-cloning leaf
  inherits it. *Context: [ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md),
  [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md)*
* [ ] **F5-TTS has no working high-level door, and its catalogued card documents one.** The export,
  the driver and `loom_cli` all work (gate: max |Δ| 4.14e-03 against the reference waveform; ASR
  oracle exact), but `Text2Speech.infer(text)` in loom-py sends only the text's ids as `tokens` and
  has no way to say "here is a reference clip and what it says". F5-TTS in-fills, so it needs the
  clip (`waveform`), the transcript's ids concatenated with the text's (`text_ids`) and where the
  join is (`n_ref_text`, or an explicit `duration`). Worse than a missing door: the driver binds
  `inputs.waveform or inputs.tokens` — the generic `caller_input` fallback — so a bare
  `infer("hello world")` hands the mel front end a handful of ids *as audio samples* before failing.
  The catalogue entry `f5-tts-v1-base` in loom-exporter's `tools/build_model_cards.py` gets the
  generic `text-to-speech-with-vocab` snippet (`model.text2speech.infer("hello world", ...)`), which
  therefore fails — the card gate would catch it, but it has never been staged or run.
  **What closes it:** a voice-cloning TTS door in loom-py (kwargs `reference=`/`reference_text=`,
  encoding both and computing `n_ref_text`, the shape `loom_cli --wav/--ref-text` already has), a
  matching `text-to-speech-voice-clone` snippet keyed the way `text-to-codes-voice-clone` is, dropping
  the `tokens` fallback from this driver's two CALLER bindings, then staging
  (`build_model_cards.py --only f5-tts-v1-base`) and the card gate — which inherits the item above,
  because this card also needs the reader's own `reference.wav`. Publish only after rc11. The weights
  are `cc-by-nc-4.0` (Emilia), recorded in [Epic-03 §2](../epics/epic-03-model-coverage.md).
  *Context: [ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md),
  [Retro-052](../retros/retro-052-every-phase-was-right-and-the-join-was-wrong.md)*
* [ ] **Qwen3-ASR-0.6B variants beyond the exported leaf** — `qwen3-asr-0.6b-hf` is shipped; the 1.7B
  and the native-layout repo are not. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **P5 breadth**, in coverage-per-effort order. **Families 4, 5, 6, 10, 11 and 12 are COMPLETE** —
  4 is HuBERT/data2vec-audio/wav2vec 2.0, 5 is SenseVoice-Small and Paraformer-zh, 11 is DAC/SNAC/
  EnCodec — **family 9 is at five of twelve leaves** (Matcha, Supertonic, F5-TTS 2026-09-18,
  Chatterbox and Pocket-TTS 2026-09-24), and the remainder is **9/10 (remaining TTS) → 13 (small
  classifiers) → 14 (music)**. The seven leaves left in 9 are mostly compositions whose AR half is
  family 10's (cosyvoice3, voxcpm2, …). Chatterbox showed that one composes from the existing
  templates, so what the next one costs is its own front end and its own loop shape, not a template.
  *Context: [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) and
  [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md) for what family 12
  cost across three checkpoints, which is the estimate the rest of this list should be read against;
  [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md) and
  [Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md) for where families 4 and 5
  found the cost instead. Family 6 (translation encoder-decoders) inherits ADR-027's fairseq id
  handling for free.*
* [ ] **`flan-t5-small`'s vocabulary is 32,100 pieces against a 32,128-wide logit row.** T5 pads its
  embedding to a multiple of 128, so an argmax could in principle name an id with no piece — untrained
  rows, never observed in practice, and the model is shipped and verified without a bound on it. Worth
  one if a leaf in this family ever emits such an id. *The other limit recorded alongside this one is
  the driver marshalling `n_head * n_src²` doubles for the encoder bias — tens of thousands for a
  sentence, 1.5M at the 512-token ceiling;
  [ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md) records the in-graph
  alternative if it ever becomes measurable.*
* [ ] **Voxtral-Mini-3B waits on a machine, not on the exporter.** P5.0 is closed — all three changes
  are in ([ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md)): a phase releases its
  torch and MIL halves together, packs its weights as it converts, and under `--isolate-phases`
  converts in a process of its own. Its LM phase alone still needs ~14.4 GB of F32 weights beside
  ~14.4 GB of MIL constants, so the floor is ~29 GB against this box's 28 and no further exporter
  change moves it. The measurements a bigger machine would pick it up from are in
  [Epic-03 §2](../epics/epic-03-model-coverage.md).
  * [ ] *The one exporter change still worth making here, and only if load TIME starts to matter:*
    a per-family hook that loads one phase's submodule instead of the whole checkpoint. An N-phase
    model costs N+1 checkpoint loads under isolation today. It would not change the peak.

## Exporter / MIL compiler

* [ ] **An export's op names depend on what its process converted earlier.**
  `coremltools`' `Builder.name_count` is a *class* attribute -- one counter for the whole process --
  and `_get_free_name` both reads and increments it, so an op a MIL pass builds without an explicit
  name is `transpose_0` in a fresh interpreter and `transpose_21` in one that has already converted
  twenty. That name reaches the emitted topology as the node's output, so **two exports of the same
  checkpoint from differently-warmed processes are not byte-identical.** Harmless in practice today --
  `loom-export` converts one model and exits, which is why the sweep has never seen it, and
  `whisper-small` came out byte-identical exported both with and without `--isolate-phases`
  (969,918,400 bytes, `cmp` clean) -- and it is *not* new with phase isolation, which is only what
  surfaced it, inside a pytest session where the in-process arm had a warmed counter and its workers
  did not. **The fix is one line** (clear `Builder.name_count` before each phase's `ct.convert`), and
  the reason it is filed rather than done is that it would rewrite those names in every artifact whose
  later phases contain an auto-named op, so it needs a sweep to say which models move and a decision
  about re-publishing them. Until then `tests/ci/test_phase_isolation.py` resets the counter per export
  and `test_the_mil_op_name_counter_is_process_global` pins the mechanism. *Context:
  [ADR-039](../adrs/adr-039-a-phase-boundary-is-a-process-boundary.md).*

* [ ] **A dynamic slice with a NEGATIVE begin exports as twice the rows at a negative offset.**
  `_infer_dynamic_dim_expr` renders `x[:, -n:, :]` arithmetically instead of normalising the begin
  against the source length, so `begin = -n` stays `-n` and the size comes out `end - begin = 2n`. At
  the traced length the two readings coincide, which is why it converts, writes and loads and only
  fails when run. Found in F5-TTS, where x_transformers' `apply_rotary_pos_emb` opens with exactly
  that slice; fixed **in the family** by substituting a version without the (provably identity) trim,
  so nothing in the zoo emits one today and the walk is unchanged. Worth doing properly the first time
  a negative begin is not removable. *Context:
  [Retro-051](../retros/retro-051-a-negative-begin-doubled-the-slice.md), and
  [Retro-047](../retros/retro-047-an-inferred-dimension-outlives-the-reshape.md) for the same
  right-at-the-traced-length shape.*
* [ ] **F5-TTS's text front end ships the table and not the segmenter.** `convert_char_to_pinyin` is
  `rjieba` + `pypinyin`; `loom::F5Vocab` is the character half, measured at 2000/2000 against the real
  function for ordinary prose and diverging by one inserted space for multi-character punctuation runs
  and hyphen-joined digit groups. Closing it means a CJK segmenter in the engine, which is a decision
  about scope rather than a fix. *Context:
  [Epic-03 §2](../epics/epic-03-model-coverage.md), [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).*
* [ ] **Supertonic carries 2.3 MB of the same zero padding VITS just lost.** `ttl_text_512.emb_*`,
  99.1% zeros — 0.9% of that model, against VITS's 23.2%. **Not the same fix**: P4.28 made VITS's pad
  dynamic, and Supertonic's text axis is statically sized on purpose for two independent reasons its
  own export docstring gives, one of which is that `GraphBuilder` resolves only one dynamic-length
  symbol per topology. It belongs to that limitation, not to the pad.
  *Context: [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md),
  [Epic-05 §5](../epics/epic-05-edge-performance.md) P4.28.*
* [ ] **Modular-export generality is unproven** — the blueprint's whole point was structural rather than
  by-name discovery, and that claim still rests on LFM2 alone. Needs a second, structurally different HF
  model in the regression suite. *Context:
  [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md)*
* [ ] **Modular phase 2 — automatic prefix/suffix boundary discovery.** Deliberately not attempted;
  today's `ModularExportSpec` needs a ~3-line declarative boundary per model. Worth doing only if that
  starts feeling like real friction across 2–3 more models.
* [ ] **`LoomModelFor*` runtime entry points** — the *inference* half of the `optimum` analogy has no
  counterpart. Arguably should wait for a second consumer besides the tests. *Context:
  [ADR-005](../adrs/adr-005-export-config-and-task-registry.md)*
* [ ] **Driver templates as first-class artifacts** — `render_driver` is still marker-based string
  replacement into a hand-written `.lua`. The mechanism is `driver_ir`, which exists and is under-used.
  A family whose driver needs more than one substitution point will force it.
* [ ] **`.inputs` / `.generate_dummy_inputs()` / `.patch_model_for_export()` as named config members** —
  none exist under those names; axes are declared per phase and dummy inputs built inline.
* [ ] **Known gap: `matmul` composition only handles `transpose_x=False`.** Any other combination raises
  `NotImplementedError` by design rather than miscomputing. Not yet hit by any converted model; needs a
  real derivation and test case the first time one is.
* [ ] **StableHLO prototype on one solved model** — deliberately not started; filed as a validation
  exercise rather than a fix.
* [ ] **`_infer_dynamic_dim_expr`'s root-axis fallback should be observable.** Four separate producers
  stopped the walk during family 5 and each silently substituted the root axis, producing a graph that
  converts, loads, builds and declares a wrong length. Two of the four (`loom_scale`,
  `loom_broadcast_to`) are ops the exporter's OWN passes emit, so they had been reachable by every model
  since those passes were written, and one (`reduce_sum`) had a case that only handled half its inputs.
  The fallback cannot be removed — it is right for the common case, which is why
  [Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md) left it
  — but it is currently indistinguishable from a derived answer. `ValueFacts` already carries
  `scalar_expr_is_guess` / `range_scalar_is_guess` / `slice_axis_is_guess` for exactly this distinction
  on the scalar paths; `dim_expr` has no equivalent. Adding one, and having the REPEAT-emitting rules
  (`loom_broadcast_to`, `tile`, `fill`, `fill_like`) say so under an env var or in the export log, would
  have turned four five-minute export-and-read cycles into one. **Not a behaviour change** — a
  provenance flag and a diagnostic, nothing that alters an emitted graph.
  *Context: [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md),
  and [[feedback-instrument-the-walk-do-not-re-export]] for the 30-line repro that names the broken
  link without a re-export.*
* [ ] **P6 cleanup** — delete the `tools/convert_*` directories (~14,000 lines across 10), then the docs
  pass.

## Engine — correctness

* [ ] **Who still depends on the layout-"healing" heuristics?** The guards make size-guessing
  *unreachable for already-correct graphs*; they do not remove it, and `op_repeat`'s two branches are
  **unguarded**. Every one is a silent-wrong-answer generator with no error path.
  *Work, in order:* (1) instrument each branch with a counter naming op and shapes, run all models, and
  record which fire — that converts "some model probably needs these" into a list; (2) fix each real
  firing at the **exporter**; (3) delete the branch. A branch that fires for nothing can go immediately.
  *Context: [Retro-001](../retros/retro-001-layout-healing-heuristics.md)*
* [ ] **Known bug: `make_lfm2_gguf.py`'s per-layer zero-RoPE placeholder produces NaN.** Each of the 16
  decoder layers is traced independently with `position_embeddings` as `torch.zeros(1,1,64)` — the
  script's own comment calls them "placeholders that we will swap", and they never were. A NaN reaches
  SILU and trips `assert(!isnan(x))`, a hard `SIGABRT`. `test_e2e_lfm2_lua_driver` skips (77)
  unconditionally. **Not fixed**, and worth fixing only if that bespoke script's coverage still matters —
  the MIL path is otherwise a strict improvement over it.
* [ ] **A `REPEAT` whose source is a PERMUTE aborts the process.**
  `ggml_compute_forward_repeat_f32` asserts `nb00 == sizeof(float)`, and `_op_loom_broadcast_to` emits a
  bare `REPEAT` from whatever var the broadcast pass handed it. A permuted view has a non-unit innermost
  stride, so a graph that broadcasts a transposed tensor converts, exports, loads, builds — and then
  takes down the host process with a raw `GGML_ASSERT`, not a `loom::Error`. Family 5 hit it twice (an
  edge-pad repeat over a transposed mel, and the position table) and worked around it **in the family**,
  by keeping both tensors in a layout where the repeat source is contiguous. That is the right fix for
  one model and not a general one: nothing stops the next family from writing the natural torch
  spelling. Two candidate fixes, and they are not equivalent — the exporter could emit `CONT` before a
  `REPEAT` whose source it knows is permuted (it emits the `PERMUTE`, so it can know), or `op_repeat`
  could `ggml_cont` a non-contiguous input itself (always correct, costs a copy only when needed, but
  moves a graph decision into the engine and against the lean-runtime principle).
  *Context: `ggml_view_*`'s own `nb[0]` assumption and
  [Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md), which is the same
  shape of bug one op over.*

## Engine — performance

* [ ] **LiteRT-class CPU speed: what it would actually take, and which three of its four pieces are
  runtime work.** The standing hope is that loom matches LiteRT on some models. LiteRT gets there with
  four things, and mapping them onto this tree ranks very unevenly — the important structural finding
  is that **three of the four are kernel/runtime work, not export-time metadata**, so this is a
  different bet from the GGUF memory-layout and prescribed-tiling thread and should not be expected to
  fall out of it.
  * [ ] **XNNPACK-class microkernels — the actual gap, and the one worth the most.** Per-ISA
    hand-written microkernels, packed weights, **indirection buffers** so a convolution never
    materialises an im2col matrix, and conv+bias+activation fused into one pass over the accumulator
    tile. That last is the same idea as "evaluate activations in accumulators to avoid the round trip
    through cache", and XNNPACK is the existence proof that it pays. **This tree is already walking
    the same road**: the F32 GEMM microkernel was measured at **71% of the whole onnxruntime gap** and
    shipped as two tinyBLAS patches; the ARMv6 run-at-a-time im2col patch gather is an indirection
    buffer in miniature; P4.29's dequantize-at-the-top-and-re-enter is a reusable piece. What is
    missing is doing it deliberately and across ops rather than one measured hotspot at a time.
    *Two standing cautions apply and both are ours: a node-by-node profile swung a 78 ms gap by 230 ms
    depending on how its own overhead was apportioned, and 3.92x on an isolated op became 0.5% on the
    model. Fusion wins are routinely smaller than a node table implies — measure what fusion is worth
    separately.*
  * [ ] **Direct I/O buffers — which for loom means the LUA MARSHALLING BOUNDARY, not the host API.**
    LiteRT hands a caller a pointer into the arena so an input costs no copy. The analogous cost here
    is `loom.causal_mask` and its siblings building a Lua table of doubles that is then converted to
    float and copied into a backend tensor: three passes plus a table allocation. This class has bitten
    once already as the prefill ceiling (P4.0.14,
    [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md)), and family 6 made it larger —
    `t5_position_bias` marshals `n_head * n_src²` doubles per encoder call. The fix is a "fill the
    tensor in place" primitive: the driver names the builder, the engine writes straight into the
    declared input. Cheap, bounded, and the measurement is easy.
  * [ ] **Serializing a COMPILED accelerator program.** The "fewer dispatches" half of LiteRT's
    delegate story has already been probed here, and the honest reading is that **a split count does
    not predict it either way**: the Metal `PAD` prototype removed 27 of 56 splits and bought 1.8% on
    the model it was measured on
    ([Retro-026](../retros/retro-026-three-nodes-were-half-the-runtime.md) §5.4), and the same kernel
    later measured **11.0% on VITS** (97.9 → 88.7 ms) and shipped as `ggml-0016`
    ([Retro-028](../retros/retro-028-three-closing-arguments-that-were-never-measured.md)). So neither
    "dispatch reduction is worthless" nor "it is the gap" is supported — it is per model and has to be
    measured per model. What is NOT measured at all is the other half: Vulkan and Metal recompile
    their shaders at every startup, and caching a compiled program is a cold-start win nobody here has
    put a number on. That is the one piece of LiteRT's fourth item that fits the GGUF-metadata thread
    naturally.
  * [x] **FlatBuffers — investigated and declined, so it is not re-derived.** A flatbuffer replaces a
    load-time parse, not the per-call ggml graph build; ExecuTorch's own `GRAPH_REBUILD.md` rebuilds
    too. And the container is not where TFLite's memory win comes from — the **arena planner** is,
    which precomputes every tensor offset once and reuses it forever *because TFLite shapes are
    static*. Loom's are not (`n_tokens`/`n_kv` are dynamic, which is why P4.0.15 buckets and keys
    graph reuse on the bucketed length), so the equivalent is already partly held by `gallocr` plus
    graph reuse, and the rest is a trade this engine made deliberately for dynamic length rather than
    a gap to close.
  * *Context: [Epic-05](../epics/epic-05-edge-performance.md). Whoever picks this up measures against
    a competitor build that is NAMED —
    [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md) is the standing rule, and the
    reason is that conda-forge onnxruntime is 1.86x faster than the PyPI wheel at the same version. A
    LiteRT baseline has the same hazard and no one here has pinned one yet.*
* [ ] **Write down that loom-exporter's tests run under `~/.venvs/piper`.** `python3` on the dev box
  resolves to **`~/.venvs/ovos`** — transformers 5.14.1, **no `sentencepiece`** — which is the
  Qwen3-ASR-only env, and `tests/ci` under it is `4 failed, 568 passed`. **All four are the
  environment and nothing else: under `~/.venvs/piper` the same four are green** (38 passed, verified
  2026-08-29). `test_spec_protocol` fails on ovos because `spm_tokenizer_export` cannot import without
  `sentencepiece`; the three `test_causal_lm_export` registry tests fail with
  `LinkError: MonolithicCall … supplies input(s) it does not declare: ['cache_position']`, a
  transformers-version difference in what the trace sees. **`piper` (transformers 4.57.6,
  sentencepiece 0.2.1) is the env for everything except Qwen3-ASR** and must not be upgraded — NeMo
  pins `~=4.53`. It is written nowhere in the repo, so the next person meets four red tests with no
  way to tell. Put it in loom-exporter's README, and note that piper is ~3x slower to run them
  (4m12s against 1m21s for the same two files).
* [ ] **`FLASH_ATTENTION`.** Unbuilt. The blocker is **the gate suite's exact-fp32 comparisons, not the
  hardware** — `ggml_flash_attn_ext` forces an F16 K/V cast. A GPU exists now and the trade still has not
  been made; it is a decision about verification. *Context:
  [ADR-016](../adrs/adr-016-kv-cache-shape.md)*
* [ ] **KV-cache addressing policies beyond contiguous append.** The `ggml_set_rows` indirection exists;
  `KvCache::fill_cell_index` is the single place a second policy would go. What is missing is a policy
  that uses it.
* [ ] **Quantized KV cache.** Storage is always F32. Different mechanism and different pipeline point
  from weight quantization — check how the cache is allocated and typed before assuming a trivial
  extension.
* [ ] **General multi-scheme quantization tool.** `quantize_gguf_q8_0.py` is Q8_0-only and
  single-model-shaped. A model-agnostic tool with a per-tensor-role policy (skip norm weights and
  embeddings) is unbuilt. *Scope note: [ADR-017](../adrs/adr-017-no-k-quants.md)*

## Backends & accelerators

* [ ] **NPUs.** Open, and the shape of the problem is known: no NPU registers as a `ggml` device type
  the engine can resolve, so `"npu"` throws by design. CoreML (the Neural Engine, which Metal is not)
  and RKNPU2 are out of tree and cost more, licence check included. *Context:
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md), [Epic-04](../epics/epic-04-backends-and-accelerators.md)*
* [ ] **The device hierarchy ranks a GPU above the CPU on unified memory, where the proxy does not
  hold.** The rule stands for "has its own fast memory", which an Apple Silicon GPU does not — it has
  the CPU's. Measured at HEAD with `ggml-0016`: `device=""` picks `MTL0` and is **1.76x faster on
  whisper-small and 1.63x SLOWER on VITS** (2.75x at Q4_0). It is why Metal ships as an extra rather
  than in the base macOS wheel; if this changes, folding it in is worth revisiting. **P4.30a, P4.30d and
  P4.30c step 5 between them took the VITS ratio from 8.98x to 1.63x and did NOT rescue the rule** — a
  GPU that is 1.6x slower on one model and 1.8x faster on another still cannot be ranked above the
  CPU by a rule that reads neither. This stays open on its own terms, with a much smaller number.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md)

* [ ] **On Metal, a Q4_0 convolutional model is now SLOWER than the same model at f32** — 141.7 ms
  against 88.7 on VITS, an inversion `ggml-0015` created and did not exist before it (149.7 against
  278.7); `ggml-0016` helps both arms and so leaves the ratio slightly WIDER, at 1.60x. The mechanism is known and is not the quantization: ggml-metal declines loom's folded
  block-quantized convolution kernel on its type test ([Epic-04 §5.2](../epics/epic-04-backends-and-accelerators.md)),
  so a Q4_0 export lowers through `im2col` + `mul_mat` and never reaches the new kernel — whose fast
  path is F32/F16-only by the same test. The choice is between teaching that type test about the
  folded kernel (P4.13's format) and dequantizing into the new kernel's fast path the way `ggml-0013`
  does on the CPU — and **P4.30c step 3 has since done the CPU-side equivalent of the second option
  for the 2-D form** (`op_conv_2d` hands a folded kernel to `ggml_conv_2d_direct_packed`, which
  dequantizes and re-enters), so the shape of the fix is now demonstrated on one backend. **Neither is on any critical path** — Metal is an extra — but "quantizing costs
  1.5x on this backend" is a surprising thing to leave undocumented in the model cards.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-017](../adrs/adr-017-no-k-quants.md)
* [ ] **`device_report()` still buckets every node as either device or CPU**, deliberately — it does not
  say *why* a node fell back.
* [ ] **Whisper's 400-wide reflect pad** is cheaper to fall back on than to compose. CUDA, Metal and SYCL
  run it natively regardless.

## Text front-ends

* [ ] **Task #79 part 2 — the C++ `orthography2ipa` port.** `src/text/phonemize.cpp` +
  `include/loom/text/phonemize.h`, vendored as an Apache-2.0 submodule, verified against the Python
  door as its oracle. Part 1 is closed — every phoneme-input TTS GGUF carries its symbol table and both
  hosts read it ([Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md)).
  **Measure both risks first:** the fold-down into each checkpoint's fixed symbol→id table, and
  pinning the beam search's tie-break. **This is now the only thing `loom_cli` cannot do for a TTS
  model**: it synthesises from IPA phonemes, from a grapheme vocabulary (Supertonic, from plain
  English), and from codec codes, so what remains behind this port is exactly "type English at a
  phoneme-input family".
  *Context: [ADR-012](../adrs/adr-012-permissive-phonemizer.md)*
* [ ] **Generalize the grapheme front end out of C++** — when a real second grapheme TTS model exists.
  Qwen3-TTS is not one. *Context: [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md)*
* [ ] **Remaining BPE pretokenizer families** beyond the ~40 in `pre_spec_table()` (CJK-script splitters,
  case-transition shapes, `byte_encode=false` SPM-style families). Each raises a named error rather than
  mis-tokenizing — bounded; add one when a real model needs it. **Distinct from the added-token pre-pass** P4.23
  shipped (`<|im_start|>` and friends, which `encode` could not emit at all) rather than pretokenizer
  regexes — but both land in `bpe_vocab.cpp`, so read
  [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md) before opening this one.

## Host API

* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

**Standing process, not an open item:** every engine change needs a `vendor/loom.cpp` bump in loom-py
before loom-py's card gate can run against it — `git -C vendor/loom.cpp fetch origin <branch>` →
`checkout <sha>` → rebuild → commit the pointer. A release pins that submodule at `main`'s tip, and
**that pin is what the wheels are built from**.

* [ ] **rc11 must carry family 9's engine half, and an older engine runs an F5-TTS file WRONGLY
  rather than refusing it.** Needed in the wheels before F5-TTS is published: `loom.run_ode`'s
  `guidance` option, `loom::F5Vocab` under `tokenizer.ggml.model == "f5"`, and loom-py's `"f5"`
  tokenizer branch (all on `feat/p5-family-9-f5-tts`). The hazard is the silent half: rc10's
  `run_ode` ignores option keys it does not know, so the file integrates unguided on rc10 and
  produces plausible, wrong audio. Worth deciding whether a GGUF should be able to declare the engine
  features it requires — this is the first driver whose meaning depends on an OPTION an older binding
  would drop, rather than on a function an older binding would lack and fail on. Verify on the
  released rc11 wheel the way ICL was verified on rc10's (`loom.Model.from_file(...)` plus one
  synthesis through the ASR oracle), after a fresh export.
  * [ ] **Chatterbox adds three more** (on `feat/p5-family-9-chatterbox`): `loom.sample_row`'s
    `min_p` option (rc10 ignores it silently, the SAME hazard as `guidance`: the file would sample
    wider than the model's own setting), the per-id repetition penalty
    ([Retro-053](../retros/retro-053-the-repetition-penalty-compounded-per-occurrence.md), which
    changes an rc10 decode wherever an id recurs, and is why the PUBLISHED Qwen3-TTS talker's greedy
    decode diverges from `transformers` on rc10: 96/624 ids on `tokens.txt` in
    `/home/flavio/.claude/tmp/qwen3_icl/`, 624/624 per id. The GGUF needs no re-export; re-grade
    that sentence on the released rc11 wheel), and `loom::ChatterboxVocab` under
    `tokenizer.ggml.model == "chatterbox"` plus loom-py's branch
    ([ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)).
  * [ ] **Pocket-TTS adds two more** (on `feat/p5-family-9-pocket-tts`): `loom.seed_kv`
    ([ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md); an rc10 engine
    FAILS loudly on it, since the function does not exist) and `loom::PocketTtsVocab` under
    `tokenizer.ggml.model == "pocket_tts"`, with `loom::Vocab`'s SentencePiece byte fallback and
    loom-py's branch ([ADR-044](../adrs/adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)),
    and `loom::load_voice` with loom-py's `voice=` door and its root-only `download()`
    ([ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)). An rc10
    loom-py has no `voice=`, so the card's voice snippet fails loudly there.
* [ ] **`nlohmann/json` is fetched as a full ~290 MB clone** for a header-only library, and it failed
  twice over a slow link during the macOS work. `GIT_SHALLOW TRUE` on that `FetchContent_Declare`
  (it is pinned to a tag, so shallow works) would remove the largest download in a cold build.

## Standing scope limitations

Deliberate boundaries rather than tasks, each naming what would have to change. Kept in
[Epic-01 §4](../epics/epic-01-inference-engine-core.md#4-standing-scope-limitations): single-sequence KV
cache, F32 cache storage, one level of `repeat_for` nesting, no chunked/windowed inference for long
Conformer-CTC audio, only the small Conformer-CTC checkpoint verified, and the attention-variant
primitive set.

## Minor cleanups

* [ ] `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()`, which throws `std::out_of_range`
  rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr could in principle reach
  this uncaught-by-`catch (loom::Error&)` path — low risk today, since the index always comes from
  `repeat_for`'s own loop bound.
* [ ] `export_config.py`'s module docstring points at a ledger section that no longer exists.
* [ ] **`GgmlPatches.cmake` asks "already applied?" the wrong way, so every `cmake` re-run rebuilds
  ggml from scratch** (~30 min on the Pi). It reverse-applies **each patch individually against the
  final tree**, which only holds while no later patch rewrites lines an earlier one added —
  `ggml-0004`..`0007` all fail it today. **Not a context-width problem**: regenerating one at `-U3`,
  `-U1` and `-U0` all still fail, because the added lines themselves are gone. The fix is a stamp file
  holding the applied set's names and hashes, skipped when it matches. Build-time cost only; the
  reset-and-retry path it falls into is correct.

---

## Knowledge Hub

| | |
|---|---|
| **Domains** | [Epics](../epics/) — what each area is and how it works |
| **Decisions** | [ADRs](../adrs/) — why a technical choice was made |
| **Lessons** | [Retros](../retros/) — what broke, why, and the takeaway |
| **Specs** | [SPECIFICATION](../SPECIFICATION.md) · [KV-CACHE](../KV-CACHE.md) · [HIGH-LEVEL-API](../HIGH-LEVEL-API.md) · [PROCEDURAL-GENERALIZATION](../LOOM_PROCEDURAL_GENERALIZATION.md) |
| **Closed detail** | [archive/](../archive/) — unmaintained, kept for the reasoning trail |

**Before opening a performance item**, read
[Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md).

**Before trusting a green gate**, read
[ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md) and
[Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md).
