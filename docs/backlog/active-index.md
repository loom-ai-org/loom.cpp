---
type: index
category: backlog
last_updated: 2026-10-09
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
| **P5 breadth: family 14 (music), MusicGen first** | Picked by the user 2026-10-09, ahead of rc16. Music was last in [Epic-03 §3](../epics/epic-03-model-coverage.md)'s coverage-per-effort order, and everything before it is done or parked: families 4, 5, 6, 9 (eight leaves), 10, 11, 12 and 13 are complete; family 9b has SpeechT5, with fastpitch and bananamind-tts deferred by the user 2026-10-01; kugelaudio, tada, dots-tts and irodori-tts were dropped 2026-09-26. MusicGen was set aside while its codec, EnCodec, was blocked ([Epic-03 §2](../epics/epic-03-model-coverage.md)). EnCodec 32 kHz is published now, so the expected bill is the LM half, as Dia's was. Read that estimate against Epic-03 §2: the bill lands where the scoping did not look ([Retro-052](../retros/retro-052-every-phase-was-right-and-the-join-was-wrong.md)) |
| **rc16, after MusicGen** | `main` carries what rc15 does not: ggml v0.26.0 and the Vulkan conv_2d F32-precision fix that stops F5-TTS clipping on coopmat GPUs ([Retro-077](../retros/retro-077-a-gpus-f32-convolution-was-half-precision-too.md)). No GGUF needs a re-export for either. The backend wheels' exact `loom-py-rt == 1.0.0rc15` pin moves with the release → [Packaging & release](#packaging--release) |

**State anchor, 2026-10-08 — `1.0.0-rc15` is released and the Hub equals the staging tree.**
Four packages on PyPI at `1.0.0rc15` (`loom-py-rt` 17 files, `-cuda` 2, `-vulkan` 2, `-metal` 1); the
`linux_armv6l` wheel rides as a **GitHub release asset**, because PyPI rejects that tag at upload
(`wheels.yml` says so at the job). The signed tag `1.0.0-rc15` sits on loom-py `9041c9e` (the merge of
loom-py #54), which pins loom.cpp `90e3492` (the merge of #63); `wheels.yml` run 37806963314 was green
including `publish-pypi`. rc15 carries the frame-embedding door (`loom.output.embedding_dim`,
`FrameEmbeddings`; [ADR-062](../adrs/adr-062-a-classifier-says-how-many-answers-it-gives.md)'s second
amendment), P4.31's direct depthwise conv + fusion, and four GPU correctness fixes
([ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md)). `wakehubert-tiny-loom` was
**republished on rc15** the same day (Hub commit `aa39908`, exported at loom-exporter `29f0168`, which
adds the width key; card-gated 2/2 on the RELEASED rc15 wheel, TitaNet's per-clip rows too; sha256 +
README verified, each of the four files run from the Hub through `speech2embeddings`). Its card now
calls the door and needs rc15. A dry-run comparison found the other 52 repos identical to the staging
tree, and every card regenerated from the catalogue byte-identical, so nothing else needed a re-export.

**Previous anchor, 2026-10-02 — `1.0.0-rc14`.** The signed tag `1.0.0-rc14` sits on loom-py `1ca8a42`
(the merge of loom-py #51), which pins loom.cpp `70195f1`. rc14 carries Canary's long-form decode
([ADR-063](../adrs/adr-063-a-long-clip-is-decoded-the-way-its-reference-decodes-it.md)), the
SentencePiece dummy prefix ([ADR-065](../adrs/adr-065-a-converted-sentencepiece-bpe-declares-its-dummy-prefix.md)),
`loom::PyRegex` + Soprano's vocabulary ([ADR-067](../adrs/adr-067-a-regex-rule-table-ships-its-reference-patterns.md))
and the ring KV cache ([ADR-066](../adrs/adr-066-a-uniform-sliding-window-is-a-ring-kv-cache.md)).
**Fifty-three** models are on [huggingface.co/loom-ai-org](https://huggingface.co/loom-ai-org) since
2026-10-03, when `wakehubert-tiny-loom` was published on rc14 (infer-only; four precisions in one repo,
`wakehubert-tiny-{f32,f16,q8_0,q4_1}.gguf`, the first multi-file repo; Hub commit `776f68b`, exported at
loom-exporter `aadac8a`, card-gated 2/2 on the RELEASED rc14 wheel, sha256 + README verified, each file
loaded from the Hub by name). Before it, fifty-two. On rc14 as
released, each a fresh export, card-gated on the RELEASED wheel (17 rows passed, 0 failed), sha256- (or
git-blob-, for the voice files) and README-verified: `canary-1b-v2-loom` republished with its six
long-form keys, and seven new -- `silero-vad-loom`, `moonshine-streaming-tiny-loom`,
`moonshine-streaming-small-loom`, `soprano-1.1-80m-loom`, `kyutai-stt-1b-en-fr-loom`,
`lfm2.5-audio-1.5b-asr-loom` and `lfm2.5-audio-1.5b-tts-loom` (with three voice files). Moonshine tiny,
Silero and Soprano were also run from the Hub through `from_pretrained`. EnCodec and F5-TTS ship `cc-by-nc-4.0` and MarbleNet the NVIDIA Open Model License,
and all three are **settled, not pending**. → [[loom-release-state]], [Epic-08](../epics/epic-08-packaging-and-release.md)

What decides a publish is the standing rule, **WHEELS FIRST,
ALWAYS**: a GGUF or a card that needs something the released wheels have not got waits for the
release that carries it, because a file published before its wheel is a file nobody can run, and an
option an older engine does not know can be dropped silently rather than refused (rc10's `run_ode`
would have integrated F5-TTS **unguided**). And every publish is a fresh export first
([[feedback-release-gate-needs-a-fresh-export]]) → [Packaging & release](#packaging--release)

---

## Models

* [ ] **MOSS-TTS + MOSS-Audio-Tokenizer: three open pieces.** Both are published
  (`moss-tts-local-transformer-v1.5-loom`, `moss-audio-tokenizer-v2-loom`), with voice cloning through voice files of
  reference codes; the card gate runs the cloning card when `LOOM_CARD_VOICES` supplies a voice. Left: (1) a Q8_0 build
  beside F32, at 16.8 GB; (2) the template's other lines (`Tokens` for duration, `Instruction`); (3) a clip-in door
  inside loom, which needs the codec encoder as phases. *Context:
  [ADR-049](../adrs/adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md)–[ADR-053](../adrs/adr-053-a-codec-lms-voice-is-its-references-codes-stamped-with-the-codec.md),
  [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **Voxtral-4B-TTS: a quantized build.** Published (`voxtral-4b-tts-2603-loom`, exported on the workstation,
  with 20 voices, all CC BY-NC 4.0). At F32 a frame reads the whole 13.7 GB LM, and the CPU runs 5.5x slower than real
  time on 24 cores, so a quantized build beside F32 is worth offering. CUDA is not measured. There is no cloning: the
  open checkpoint has no codec encoder. *Context: [Epic-03](../epics/epic-03-model-coverage.md),
  [ADR-054](../adrs/adr-054-a-tiktoken-vocabulary-is-merged-by-rank-in-the-shared-bpe.md),
  [Retro-062](../retros/retro-062-an-f32-wrapper-check-could-not-tell-a-spelling-from-a-defect.md)*
* [ ] **Chatterbox's other leaves**: voice cloning (voice encoder + S3 tokenizer + CAMPPlus), and the
  multilingual and Turbo checkpoints. The English model is PUBLISHED (2026-09-30,
  `loom-ai-org/chatterbox-loom`: exported on the workstation from loom-exporter `c61351e`, card gate 2
  passed there, x-linked-etag = local sha256 `eee92e72...`; no Perth watermark, per Epic-03 §2).
  *Context: [Epic-03](../epics/epic-03-model-coverage.md),
  [ADR-041](../adrs/adr-041-a-text-front-ends-rules-ship-as-data.md)*
* [ ] **Pocket-TTS: cloning a voice from a recording.** Published (`pocket-tts-loom`, with 26 voice files under
  Kyutai's use restrictions; two voices are non-commercial). Cloning needs the Mimi encoder, which is in the gated
  voice-cloning weights and zeroed in the other release, as one more phase feeding the text prefill's ordinary input.
  It is the same door F5-TTS's in-filling already has from the other side. *Context:
  [Epic-03 §2](../epics/epic-03-model-coverage.md),
  [ADR-043](../adrs/adr-043-a-voice-that-is-attention-state-is-seeded-not-run.md),
  [ADR-045](../adrs/adr-045-a-voice-is-a-file-of-driver-inputs-stamped-with-its-weights.md)*
* [ ] **VoxCPM2: two open pieces.** Published (`voxcpm2-loom`, Apache-2.0, 9.3 GB at F32).
  * [ ] **Voice cloning from a recording** needs the AudioVAE's encoder (16 kHz, x640) as one more
    phase feeding `feat_encode`'s prompt patches; the driver's prefill already takes patches and masks.
  * [ ] **The reference's f32-vs-f64 spread is not a floor for this model**: FSQ rounds `tanh(x) * 9`
    every step, so a boundary crossing flips a level, and the free-running gate arm could fail on another
    ISA with nothing wrong. Revisit if it ever does. *Context:
    [ADR-046](../adrs/adr-046-a-guidance-rule-the-integrator-cannot-express-stays-in-the-step-graph.md)*
* [ ] **Fun-CosyVoice3: two open pieces.** Published (`fun-cosyvoice3-0.5b-loom`, 2026-09-29, Apache-2.0). The card
  gate passes, and its cloning card runs when `LOOM_CARD_VOICES` supplies a voice.
  * [ ] **Cloning needs Python.** Voice files work (`loom_exporter.cosyvoice3_voices`, gate arm on a
    JFK clip), but making one runs the ONNX S3 tokenizer v3 and CAMPPlus in Python. Cloning inside
    loom means exporting both as phases, with the log-mel and fbank front ends in the graph, and the
    release ships neither as torch (check `s3tokenizer` on PyPI for a v3 port).
    *Context: [ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)*
  * [ ] **Normalisation is the rules path only.** The reference prefers `wetext` (WeTextProcessing
    FSTs, Apache-2.0) when installed, which reads dates, money and units the rules path spells digit by
    digit. Reproducing it needs the FSTs as data and an FST runtime in C++; check the FSTs' licence
    first. Deferred by the user 2026-09-25.
    *Context: [ADR-048](../adrs/adr-048-cosyvoice3-normalises-by-the-references-rules-path.md)*
* [ ] **Qwen3-ASR-0.6B variants beyond the exported leaf** — `qwen3-asr-0.6b-hf` is shipped; the 1.7B
  and the native-layout repo are not. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **P5 breadth: family 14 (music) is next, MusicGen first** (user, 2026-10-09). Every earlier family
  in the coverage-per-effort order is complete or parked (see *Now*). Chatterbox showed that a leaf
  composes from the existing templates, so what the next one costs is its own front end and its own
  loop shape, not a template.
  *Context: [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) and
  [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md) for what family 12
  cost across three checkpoints, which is the estimate the rest of this list should be read against;
  [Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md) and
  [Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md) for where families 4 and 5
  found the cost instead.*
* [ ] **Moonshine's encoder attends through a full `T x T` mask of which 20 diagonals are live.** 81.9 s
  of audio is 4096 rows: 16.7M scores per head per layer, 27 s for tiny on the 2-core box. A banded
  attention (keys gathered at the 19 live offsets) is the same softmax over a fraction of the work;
  measure it on the Pi before scoping, as P4.25-style ideas have measured out before.
* [ ] **`log_softmax` lowers as `log(softmax(x))`** (coremltools' torch frontend), which is -inf below
  about -103 in f32. Canary's head stops before it; the CTC heads still carry it. Harmless for an argmax,
  wrong as a tensor. A stable form needs a row max the engine has no primitive for.
* [ ] **Kyutai STT and LFM2.5-Audio are published at F32** (rc14, 2026-10-02): at F32 Kyutai runs
  about 5x slower than real time on the 2-core box and LFM2.5's two files are 5.2 and 5.5 GB. Measure
  Q8_0 against each gate (`test_e2e_kyutai_stt_lua_driver`, `test_e2e_lfm25_audio_{asr,tts}_lua_driver`,
  whose reference scripts are in `scripts/`) and republish if the transcripts and codes hold.
* [ ] **LFM2.5-Audio's interleaved speech-to-speech door** (`generate_interleaved`, text and audio in a
  fixed 6:12 pattern, multi-turn chat) -- not requested; the TTS door's phases are all it needs, plus
  an audio-in prompt segment (the ASR door's encoder) and the interleave schedule in the driver.
* [ ] **Requested for the zoo, unscoped** (added 2026-09-25 at the user's request, no order among
  them yet). None has been checked against its checkpoint; the template guesses are where scoping
  starts, not what it will find.
  * [ ] **Cohere ASR** — architecture and licence not yet looked at.
  * [ ] **Nemotron ASR** (NVIDIA) — expected NeMo-shaped; which checkpoint and head is to be decided.
  * [ ] **Voxtral Mini realtime** (Mistral, streaming ASR) — check its size first against the machine
    floor that blocks Voxtral-Mini-3B (below).
  *Context: [Epic-03 §3](../epics/epic-03-model-coverage.md#3-roadmap)*
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
  [Epic-03 §2](../epics/epic-03-model-coverage.md). **The bigger machine exists now:** the workstation
  exported Voxtral-4B-TTS's 3.4B Ministral LM (the same backbone class) at a 32.2 GB peak in 49 s, with
  the checkpoint, the venv and the recipe under `~/loom-voxtral/` there.
  * [ ] *The one exporter change still worth making here, and only if load TIME starts to matter:*
    a per-family hook that loads one phase's submodule instead of the whole checkpoint. An N-phase
    model costs N+1 checkpoint loads under isolation today. It would not change the peak.

## Exporter / MIL compiler

* [ ] **A shape read as DATA aborts in the engine.** `length / waveform.shape[1]` (or `lengths *
  x.shape[-1]`, speechbrain's relative-length masks) traces to `SHAPE` -> `GET_ROWS` -> arithmetic. The
  engine's `SHAPE` is a four-element `ne` vector and `GET_ROWS` on a 1-D tensor returns it whole, so the
  arithmetic sees `[4]` where it expects `[1]` and `ggml_can_repeat` aborts (ECAPA, family 13: `DIV
  [1]/[4]`, then a RESHAPE of four elements into one). A shape consumed at BUILD time (FILL, RESHAPE
  targets) works; one consumed as a value does not. `arange(x.shape[i])` is unaffected (the walk folds
  it). Fix either end: the exporter folds a static `GET_ROWS` of a `SHAPE` into the dimension's own
  expression, or the engine's `GET_ROWS` indexes a 1-D tensor's elements. Until then ECAPA ships
  whole-clip only. *Context: [Epic-03 §2](../epics/epic-03-model-coverage.md#family-13-small-audio-classifiers-and-embedders)*
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

* [ ] **WakeHuBERT's remaining gap to ONNX int8** (11.2 ms against 5.67, dev box, F32, one thread, after
  P4.31). Its convs already run ggml's direct `CONV_2D` at 55-68% of the F32 peak, so a better F32 GEMM
  buys ~2 ms at most. Levers by size: an FFT/STFT primitive in place of the 80 MFLOP DFT matmul (~2 ms,
  and every DFT-basis front end in the zoo pays the same), int8 dot products for the 1x1 convs (~4.7 ms
  of work, unmeasured), and host overhead (~2 ms or more). *Context: [Epic-05 P4.31](../epics/epic-05-edge-performance.md#p431--depthwise-convolutions-a-direct-kernel-a-simd-interior-and-the-causal-block-fused--done-2026-10-03),
  [Retro-073](../retros/retro-073-the-profiler-cannot-see-a-fusion.md)*
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
* [ ] **Run `ggml-0024`'s tensor-API branch on a Mac that has one.** On an M5-class GPU, Metal's
  `mul_mm` uses the tensor API, which has no float-tile variant, so `ggml-0024` sends a `GGML_PREC_F32`
  F32 product to the mat-vec kernel instead. That branch compiles but has never run: the M1 Pro does not
  have the tensor API. SenseVoice on `jfk.wav` is the check, and the cost of the mat-vec fallback is the
  number to take. (`ggml-0023`'s coopmat and coopmat2 branches HAVE run, on the RTX 5090.)
  → [ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md),
  [Epic-04 §6](../epics/epic-04-backends-and-accelerators.md#6-gpu-correctness-f32-matmul-precision-the-metal-norm-the-cuda-mat-vec-stride-2026-10-08)
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

* [ ] **A GGUF cannot say which engine features its driver needs.** An older engine ignores option keys
  it does not know, so a driver whose meaning depends on an OPTION runs plausibly and wrongly there:
  rc10's `run_ode` would have integrated F5-TTS unguided, and Chatterbox's `min_p` and CosyVoice3's
  `top_p_mass`/`banned`/`uniform` sampler keys are the same hazard. A missing FUNCTION fails loudly
  instead. WHEELS FIRST covers this by process today; a requirement the GGUF declares and the loader
  checks would cover it by construction.
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
