---
type: index
category: backlog
last_updated: 2026-09-12
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
| **TWELVE staged models are newer than the Hub — and MUST NOT be uploaded before the engine ships** | Their drivers call bindings that exist only on this branch, so publishing one now would put a GGUF on the Hub that no released `loom-py-rt` can run. Which feature each needs: `encodec-32khz` the ELU primitive + `run_recurrent_and_retain`; `gigaam-v3-rnnt`, `parakeet-rnnt-0.6b`, `parakeet-tdt-0.6b` `output_shape` + a retained ROW range; `kokoro-82m`, `styletts2-ljspeech` `output_shape` plus the retrace's own two, `run_bi_recurrent_and_retain` + `expand_by_duration_and_retain` ([ADR-032](../adrs/adr-032-an-interleave-is-a-layout-a-concatenation-is-a-graph.md)); `matcha-tts-ljspeech`, `supertonic-2` `run_ode_and_retain`. (`dac-44khz`'s file also differs but needs nothing new — a re-export, not a new capability.) Family 5's `sensevoice-small` joins them too (2026-09-15): its driver binds a DEFAULTED `prompt_ids` input, and a released wheel would also abort inside `ggml_im2col` on the LFR stacking, which is a GROUPED convolution and so needs family 4's `CONV_1D` fix. Family 4's two leaves join them: `hubert-large-ls960-ft` and `data2vec-audio-base-960h` need the `ctc` vocabulary reader AND the grouped-convolution lowering, so a released wheel would decline the file's tokenizer and abort inside `ggml_im2col` on its positional convolution. Qwen3-TTS's two join them (2026-09-13): `qwen3-tts-12hz-0.6b` needs `loom.sample_row`'s new `repetition_penalty`/`penalized`, without which it never emits EOS and runs to the token cap, and `qwen3-tts-tokenizer-12hz` needs nothing new but is half of a pair that does. **These go out WITH rc10, after the engine is merged and its wheels are on PyPI, never before** → [[loom-release-state]] |
| **EnCodec's Hub upload — BLOCKED ON A LICENCE DECISION, and on rc10** | Exported and verified (max \|Δ\| 5.07e-07, exact sample count, card-gated in `hf-models/encodec-32khz`); it also needs the ELU primitive, so it ships with the engine like the eight above. `facebook/encodec_32khz` declares NO `license:` tag: the EnCodec CODE is MIT, but this checkpoint was trained as part of MusicGen, whose weights are CC-BY-NC-4.0. The card takes the stricter reading; whether to re-upload non-commercial weights to the org is not a call this work should make alone → [Epic-03 §2](../epics/epic-03-model-coverage.md) |
| **P5 family 5 — Paraformer's DETOKENIZATION, the one piece left (and it is not a vocabulary)** | The exporter half SHIPPED 2026-09-16 and the GGUF's ids are correct: run through FunASR's own postprocess they give a character-for-character identical transcript on Chinese and English. **An earlier version of this row proposed a per-piece `space_after` flag and that is WRONG** -- measured against `sentence_postprocess`, the rule is stateful and includes a transform no table can hold: `['hello','你','好']` -> `hello你好` (the space after a Latin word is REMOVED when CJK follows, so it needs lookahead), and `['b','b','c','news']` -> **`BBC news`** (`abbr_dispose` collapses single-letter runs AND UPPER-CASES them). The four rules are: drop `<s>/</s>/<unk>/<OOV>`; merge `@@`-suffixed continuations; space between Latin words except before CJK; collapse+uppercase letter runs. **The TABLE needs nothing new** -- `CtcVocab` ([ADR-033](../adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)) is already one array indexed by id and would load these 8404 pieces with no word delimiter. What is missing is a per-TASK text postprocess, which by [HIGH-LEVEL-API §2](../HIGH-LEVEL-API.md) belongs in the engine's `transcribe` door beside Whisper's control-token stripping, NOT in a vocabulary class → [Epic-03 §2](../epics/epic-03-model-coverage.md) |

***Branch state: everything below is on `feat/p5-families-11-4-5`, pushed in all three repos, no PRs
opened and nothing merged.*** Working trees are clean on all three.

**loom-py needs a `vendor/loom.cpp` bump before its gate can run against this branch** — the engine
gained `loom.sample_row`'s `repetition_penalty`/`penalized` for Qwen3-TTS, and loom-py's pin predates
it. [[loom-p5-family12-shipped]] has the exact steps.

***Family 5's SECOND LEAF, Paraformer, SHIPPED its exporter half 2026-09-16 (P5)*** — SANM encoder +
**CIF predictor** + NAR decoder, `paraformer_export.py`, two phases. The first model here whose output
LENGTH depends on the VALUES. The host decides the boundary and hands the graph a linear resampling
matrix, so the graph has no cumsum, no gather and no threshold; **no new engine binding was needed**,
and it reuses family 12's `TokenLabelsEpilogue`. The cost was numerical —
[Retro-049](../retros/retro-049-being-more-precise-than-the-reference.md): reproducing a reference
means reproducing its ROUNDING and its arithmetic ORDER, not improving on either. Verified on the
encoder tensor (4.48e-05 / 4.81e-06, sabotage 3.81e-01) and on the transcript, identical to FunASR on
Chinese and English. Its vocabulary is the open item in the Now table above.

***Family 5's FIRST LEAF SHIPPED 2026-09-15 (P5)*** — SANM / FunASR, `sanm_asr_export.py`,
`SenseVoiceSmall`. No engine primitive and no new head: the CTC epilogue is family 1's, reused, and
the vocabulary goes down the existing `sentencepiece_proto` path. The front end is rebuilt (FunASR's
`WavFrontend` does not trace) and verified against `torchaudio` at 218 lengths; the four-row prompt is
a graph INPUT with one new driver binding kind (`DEFAULTED`) because inverse text normalization is a
real capability. Verified on the LOGITS at three lengths, 187/187 + 96/96 + 96/96 argmax, sabotage arm
30.7. See [Epic-03 §2](../epics/epic-03-model-coverage.md) and
[Retro-048](../retros/retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md).

**SECOND NEW ITEM, same session — a `REPEAT` whose source is a PERMUTE aborts the process.**
`ggml_compute_forward_repeat_f32` asserts `nb00 == sizeof(float)`, and `_op_loom_broadcast_to` emits a
bare `REPEAT` from whatever var the broadcast pass handed it. A permuted view has a non-unit innermost
stride, so a graph that broadcasts a transposed tensor converts, exports, loads, builds — and then
takes down the host process with a raw `GGML_ASSERT`, not a `loom::Error`. Family 5 hit it twice (an
edge-pad repeat over a transposed mel, and the position table) and worked around it in the FAMILY, by
keeping both tensors in a layout where the repeat source is contiguous. That is the right fix for one
model and not a general one: nothing stops the next family from writing the natural torch spelling.
Two candidate fixes, and they are not equivalent — the exporter could emit `CONT` before a `REPEAT`
whose source it knows is permuted (it emits the `PERMUTE`, so it can know), or `op_repeat` could
`ggml_cont` a non-contiguous input itself (always correct, costs a copy only when needed, but moves a
graph decision into the engine and against [the lean-runtime principle](../epics/epic-03-model-coverage.md)).
Related: `ggml_view_*`'s own `nb[0]` assumption, [Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md).

**NEW ITEM, from that work — `_infer_dynamic_dim_expr`'s root-axis fallback should be observable.**
Four separate producers stopped the walk during family 5 and each silently substituted the root axis,
producing a graph that converts, loads, builds and declares a wrong length. Two of the four
(`loom_scale`, `loom_broadcast_to`) are ops the exporter's OWN passes emit, so they had been reachable
by every model since those passes were written, and one (`reduce_sum`) had a case that only handled
half its inputs. The fallback cannot be removed — it is right for the common case, which is why
[Retro-044](../retros/retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md) left it
— but it is currently indistinguishable from a derived answer. `ValueFacts` already carries
`scalar_expr_is_guess` / `range_scalar_is_guess` / `slice_axis_is_guess` for exactly this distinction
on the scalar paths; `dim_expr` has no equivalent. Adding one, and having the REPEAT-emitting rules
(`loom_broadcast_to`, `tile`, `fill`, `fill_like`) say so under an env var or in the export log, would
have turned four five-minute export-and-read cycles into one. **Not a behaviour change** — a
provenance flag and a diagnostic, nothing that alters an emitted graph.

***Family 4 SHIPPED 2026-09-12 (P5)*** — CNN + transformer + CTC, `ctc_asr_export.py`, one generic
recognizer claiming any HF `*ForCTC` directory. Verified on three structurally different checkpoints
against `transformers` on the LOGITS (549 frames of real speech, 549/549 argmax each, sabotage arm
32.7). It cost one engine READER (`CtcVocab`, [ADR-033](../adrs/adr-033-a-decode-only-table-is-still-a-vocabulary-family.md))
and one engine FIX that was nobody's estimate: `groups > 1` had been read as "depthwise" since the
first export, which is right at both ends of the range and wrong in the middle
([Retro-046](../retros/retro-046-groups-greater-than-one-was-read-as-depthwise.md)). `CONV_1D` honours
`groups` now. The third checkpoint, `omniASR-CTC-300M-v2`, is verified and deliberately unshipped — it
is scale-sensitive and its own documented processor path transcribes garbage, which loom reproduces
exactly.

***1.0.0-rc9 is DONE: tagged, its models published, and all four packages on PyPI*** — `loom-py-rt`
and the `-cuda`/`-vulkan`/`-metal` accelerators, verified at `1.0.0rc9`. It ships **ARMv6 as a supported target** (P7/P7.1 — a `linux_armv6l` wheel for the Pi Zero
and Pi 1, VITS from 57x to 18.9x real time), **family 12's SentencePiece reading** (XLM-R, where a
fairseq checkpoint's ids are not its protobuf's piece order — plus a `tokenizer.json`-only path and a
correctness fix, since `framing_ids` had been returning a SentencePiece encode's trailing `</s>`
*labelled*), and **family 6, `flan-t5-small`** — the first text encoder-decoder and first Unigram LM
in the zoo. The org now lists **twenty-three** models, every one re-exported and card-gated against this
tree, and `1.0.0-rc9` is tagged on loom-py at `330b10a`.

***The next release is 1.0.0-rc10, and it is what unblocks the Hub.*** Everything on
`feat/p5-family-11-snac` goes out in it — the four new engine bindings, `ELU`, the ODE integrator,
family 11's third leaf, and `loom_cli --out`. Its shape is fixed by the coupling above: the wheels
carry the bindings, so **the nine staged GGUFs can only be uploaded once rc10's packages are on PyPI**,
in that order. A version bump is TEN strings in FOUR files — see [[loom-release-state]] for the list
and for how to verify a Hub push afterwards.

**SNAC-24kHz was published 2026-09-12** (`loom-ai-org/snac-24khz-loom`, family 11's second leaf) off
`feat/p5-family-11-snac`, which is pushed in all three repos and not yet merged. It is the zoo's
first stochastic graph: the driver draws its noise, seeded, through the host RNG — see
[ADR-029](../adrs/adr-029-a-multi-rate-codec-keeps-one-row-per-coarsest-frame.md).

---

## Models

* [ ] **Qwen3-TTS-12Hz-0.6B-Base — the CODEC half is DONE, the talker half is not.** The source-level
  architecture read this entry used to ask for has been done (2026-09-12), against the reference
  implementation running end to end on CPU rather than against the source alone. It ships as **two
  GGUFs** by [ADR-022](../adrs/adr-022-dia-and-its-codec-stay-two-files.md)'s argument — one codec
  serves every size and variant of the talker:

  * **`qwen3-tts-tokenizer-12hz` (family 11) — EXPORTED AND VERIFIED.** 16 codebooks at 12.5 Hz in,
    24 kHz waveform out; 114 M parameters, 456 MB at F32. Max abs difference **4.167e-06** at 42
    frames and **1.699e-05** at 700, against the reference's own `chunked_decode`, on the engine's
    floats; exact sample count at both; the ASR oracle reads the decode back verbatim. It is the
    family's first leaf with ATTENTION over the frame axis and therefore its first CHUNKED one —
    [ADR-034](../adrs/adr-034-a-chunked-decode-is-the-drivers-loop-not-a-longer-call.md), which is
    the decision `encodec_export` predicted ("a chunked one is a different driver, not a longer
    call"). It cost one driver component (`ChunkedCodecCall`), no engine change and no new binding.
    Three conversion blockers, all of them another family's known failure: coremltools' dynamic-pad
    refusal (EnCodec's — and it DISSOLVES here, the pad is provably zero at stride 1), Dia's
    `rotate_half`, and transformers' `create_causal_mask` `vmap` path. The one genuinely new failure
    is [Retro-047](../retros/retro-047-an-inferred-dimension-outlives-the-reshape.md).
  * **`qwen3-tts-12hz-0.6b` (family 10) — EXPORTS AND RUNS; NOT YET VERIFIED, NO CARD.**
    Text plus a reference voice in, 800 codes = 50 frames out, through seven topologies and a nested
    loop. **Two things are open and they are probably one bug**: greedy decoding does not terminate
    where the reference stops at 42 frames, and `max_new_tokens` does not reach the driver — a cap of
    3 still produced 50 — which points at how a SCALAR input is marshalled into the driver's `inputs`
    table. Until that is fixed neither the codes nor the repetition penalty's effect can be graded,
    and no card should be written: a card here is gate-tested against a real GGUF.

    The KV-geometry blocker recorded below is CLOSED, and it was a symptom rather than a cause. The
    cause is that **`repeat_kv` does not survive conversion**: coremltools folds its expand (a
    broadcast) before `passes.fuse_gqa_repeat_kv` can match it, and rewriting it as
    `repeat_interleave` only moves the failure into the merge reshape, which comes back as
    `[128, n_tokens, n_tokens, 8]` — [Retro-044]'s substitution. The export now removes the op instead
    of converting it: `materialise_gqa` duplicates `k_proj`/`v_proj` interleaved so K/V heads equal
    query heads — **+69.2 M parameters, 277 MB at F32**, checked against the written file's own 275 MB
    growth, and a doubled cache. The code predictor is uncached besides, which is worth keeping on its
    own terms.

    (Superseded, kept for the record:)
    `qwen3_tts_export.py` traces and converts all **seven** phases and writes a 3.67 GB GGUF; the
    decomposition is verified against the reference **bit-identically in PyTorch** (all 42 frames ×
    16 codes, driven through the export's own wrappers) before anything was traced. What fails is
    `_kv_cache_geometry`: the talker's 28 fused ATTENTION blocks report **16** K/V heads and the code
    predictor's 5 report **8**, and one KvCache has one per-layer width. The talker's `repeat_kv`
    survives into the topology where the predictor's is absorbed, and the two phases are otherwise
    the same GQA 16/8 geometry. **Ruled out:** the mrope patch (the talker's rotary now returns rank
    3, identical in form to the predictor's), the talker's trace length colliding with its 8 K/V
    heads, and the predictor's colliding with its `n_rep` of 2. The error now prints the census
    (`{(16,128,128): 28, (8,128,128): 5}`), which is what makes it legible as a per-phase split.
    **The cheapest way out is probably to stop caching the code predictor at all**: it never sees
    more than 16 positions, so re-running its prefix costs 136 token-forwards of a 5-layer model per
    frame, and with one cached phase the geometry question disappears. That means folding
    `predictor_prompt` and `predictor_step` into two uncached phases that rebuild the prefix in-graph
    from ids, so the driver still passes only integers.

    **No model card yet, deliberately**: a card here is gate-tested against a real GGUF
    (`LOOM_MODEL_CARDS=... pytest tests/gate/test_model_cards.py`), and writing one for a file that
    does not run would be asserting a verification that has not happened.

    **The engine gained what this model needs and nothing else does yet**: `loom.sample_row` now takes
    `repetition_penalty` + `penalized`. That is not a sampling nicety — `transformers` applies the
    penalty as a PROCESSOR rather than a warper, so it moves a greedy argmax too, and a greedy decode
    without it never emits EOS: 200 frames against the reference's 42. With it, the decomposition
    reproduces the reference exactly. 914 M parameters. A 28-layer Qwen3-shaped
    talker (hidden 1024, GQA 16/8, head_dim 128) emitting codebook 0, plus a 5-layer **code
    predictor** that emits the other 15 from the talker's hidden state — so one audio frame is 16
    transformer forwards, not one, with the predictor's KV cache reset per frame. Its input embedding
    is a SUM of 16 codebook embeddings plus a text hidden, never a token lookup. **The `mrope` in its
    config is decorative**: `get_rope_index` always expands one row to three identical ones, so
    `apply_interleaved_rope` collapses to plain RoPE at θ=1e6 — verified in the source, and the
    scariest-looking thing in the config turns out to cost nothing.

  **Which generation mode, and why it is not a free choice.** `spk_id` is empty in this checkpoint, so
  there are no built-in speakers and voice cloning is the only mode. It has two arms and they are
  nested, not alternatives:

  * `x_vector_only_mode=True` needs a 128-mel front end at 24 kHz and an 8.9 M ECAPA speaker encoder,
    and nothing else. Verified working greedily against the reference — the ASR oracle reads back
    *"The quick brown fox jumps over the lazy dog."* exactly.
  * ICL mode (`ref_text` + `ref_code`) additionally needs the tokenizer's **encoder**, which is a
    `transformers` `MimiModel` — a whole second family-11-scale export, and one family 11 deliberately
    does not do ("the DECODE half only"). It also **degenerates under greedy decoding**: 200 frames to
    the cap, 15.9 s of audio transcribing as *"country can do for you."* So it needs the sampler as
    well as the encoder.

  So the order is forced: x-vector-only first, ICL after. **What the sampler will need that
  `loom.sample_row` has not got**: `repetition_penalty` (1.05 over the generated codes) and a
  non-contiguous allowed set — `suppress_tokens` bans `[2048, 3072)` *except* `codec_eos = 2150`,
  which `lo`/`hi` cannot express. That is the shape of ADR-024's bill for this family, and it is the
  one place engine C++ is currently expected.

  *Context: [Epic-03](../epics/epic-03-model-coverage.md). The reference is Alibaba's Apache-2.0
  `qwen-tts` package, imported lazily like SNAC's; `pip install --no-deps qwen-tts` into the **piper**
  venv, whose transformers 4.57.6 matches the package's `==4.57.3` pin. Ovos is the wrong venv for it:
  transformers 5.x has moved the internals it imports.*

* [ ] **Qwen3-ASR-0.6B variants beyond the exported leaf** — `qwen3-asr-0.6b-hf` is shipped; the 1.7B
  and the native-layout repo are not. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **F5-TTS** — deferred by explicit direction. Flow-matching, `OdeStepper`-adjacent, likely shares
  primitives with Matcha-TTS. Last of the original 7-model TTS list still untouched.
* [ ] **P5 breadth**, in coverage-per-effort order. Families 10, 11 and 12 are DONE — the remainder:
  5 (SANM) → 9/10 (remaining TTS) → 13 (small classifiers) → 14 (music). **Families 4, 6, 10, 11 and
  12 are complete**: family 11 is DAC, SNAC and EnCodec; family 4 is HuBERT, data2vec-audio and
  wav2vec 2.0. *Context:
  [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) and
  [ADR-027](../adrs/adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md) for what family 12
  cost across three checkpoints, which is the estimate the rest of this list should be read against.
  Family 6 (translation encoder-decoders) inherits ADR-027's fairseq id handling for free.*
* [ ] **`flan-t5-small`'s vocabulary is 32,100 pieces against a 32,128-wide logit row.** T5 pads its
  embedding to a multiple of 128, so an argmax could in principle name an id with no piece — untrained
  rows, never observed in practice, and the model is shipped and verified without a bound on it. Worth
  one if a leaf in this family ever emits such an id. *The other limit recorded alongside this one is
  the driver marshalling `n_head * n_src²` doubles for the encoder bias — tens of thousands for a
  sentence, 1.5M at the 512-token ceiling;
  [ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md) records the in-graph
  alternative if it ever becomes measurable.*
* [ ] **P5.0 — per-phase process isolation for conversion.** Decides which models are exportable at all
  on a given machine. Change 1 done (30.4 → 22.9 GB peak on Granite-Speech). Two remain:
  * [ ] quantize/`astype` each phase's weights as it converts, rather than at write time
  * [ ] convert each phase in its own process, loading only that phase's submodule — needs partial
    checkpoint loading and a merge that reads children back off disk
  * *Note: even all three leave Voxtral at ~29 GB against 28. Not a fix for that model.*

## Exporter / MIL compiler

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
