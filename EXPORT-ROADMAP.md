# Export Roadmap — towards an `optimum`-shaped exporter

Where the MIL exporter goes next, in seven work items (R1–R7) taken from a review of the current state.

Relationship to the other documents:

* [`EXPORT-IMPROVEMENT.md`](EXPORT-IMPROVEMENT.md) — the *previous* proposal thread (rule table, value
  resolution, control flow, family templates). Closed; [`BACKEND.md`](BACKEND.md) is its working log and
  records what each item actually cost and what it found.
* [`BACKLOG.md`](BACKLOG.md) — everything else that is open, including the two remaining follow-ups from
  that thread (multi-output topologies, the StableHLO validation exercise).
* **This document** — the next thread. It assumes BACKEND.md's conclusion (per-family templates, not
  universal orchestration inference) and asks a different question: *what does the exporter look like to
  someone who just wants to convert a model?*

The north star is **`optimum-onnx`**. Not as a slogan: the concrete claim is that its object model —
a task/architecture registry, a per-family export config declaring named dynamic axes and dummy inputs,
a model patcher that does the wrapping, and `ModelFor<Task>` entry points — is the right shape for this
exporter too, and that most of what makes the current export scripts long is work that object model
already has a home for.

| `optimum-onnx` concept | what it does | Loom equivalent today |
|---|---|---|
| `optimum-cli export onnx --model <id>` / `main_export()` | one entry point, HF repo id in, artifacts out | none — 12 hand-written `export_*.py` scripts |
| `TasksManager` | maps `(model_type, task)` → export config, infers task from the checkpoint | none — the user picks the script |
| `OnnxConfig.inputs` = `{"input_features": {0: "batch_size", 2: "sequence_length"}}` | **named** dynamic axes, per input, per family | one global symbol, `n_tokens` (R1) |
| `DummyInputGenerator` | builds trace inputs from the config, no hand-written shapes | hand-written `torch.randn(...)` per script |
| `ModelPatcher.patch_model_for_export()` | the wrapping, declared once per family | hand-written `nn.Module` wrappers per script |
| `NormalizedConfig` | pulls `num_layers`/`hidden_size`/… from heterogeneous HF configs | ad hoc attribute reads |
| submodel decomposition (`encoder_model.onnx`, `decoder_model_merged.onnx`) | family-level decision, not per-model | `ModularExportSpec` (one family), profiles (R7) |
| `ORTModelForCausalLM` / `ForSpeechSeq2Seq` / `ForCTC` | runtime entry points per task | Lua drivers + C++ backends, no unified surface |

Three of the seven items (R3, R4, R5) are that programme. R1 and R2 are the two pieces of exporter
internals that block it. R6 and R7 are cleanup decisions that should be made before, not after.

---

## R1 — `n_tokens` is a lie: name dynamic axes properly

**What is wrong.** The exporter has exactly one name for every dynamic dimension in every model:
`n_tokens`. `get_var_info` substitutes it for any symbolic MIL dim it cannot derive; `value_facts`'
producer-less fallback returns it for any scalar it cannot resolve; `shape_expr.N_TOKENS` is a
module-level singleton. In the models already exported it stands for at least five different quantities:

| model | what `n_tokens` actually is | what other axes get derived from it |
|---|---|---|
| Conformer-CTC / Parakeet | **raw audio samples** at 16 kHz | STFT frames `floor(n_tokens/160) + 1`, then subsampled encoder frames |
| Qwen3 / LFM2 | **subword tokens** | `n_past`, `n_kv` (bound separately by `GraphBuilder`) |
| VITS / Matcha | **phoneme tokens** before duration expansion | mel frames after it, which the driver computes host-side |
| Kokoro `decoder_vocoder` | ASR frames | `2*n_tokens`, `600*n_tokens`, `600*n_tokens+20` — declared by hand via `symbol_overrides` |
| StyleTTS2 diffusion | style-vector length | — |

Two concrete costs, both already paid:

* **`symbol_overrides` exists only because there is no axis vocabulary.** Kokoro's export has to reach
  into the traced program, read four raw `isN` symbol names, and map them to string expressions
  (`"600*n_tokens+20"`), because four independently-traced leaf inputs each mint their own symbol and
  the exporter has no way to say "this input's axis 1 is 600× that input's axis 1".
* **The single fallback name is load-bearing in a heuristic.** `slice_by_index` had to distinguish "the
  answer is the sequence length" from "the walk gave up and guessed the sequence length". Until this
  week it did so by *spelling*; normalizing expressions broke that and silently corrupted two models
  (BACKEND.md). It is now fixed with an explicit `is_guess` flag — but the flag exists because the
  guessed value and the derived value are the same symbol. With named axes, "I don't know" would not be
  spelled the same as any real answer.

**The engine side is already fine.** `SymbolEnv` is a plain `name → double` map seeded from the model's
hparams (`src/core/graph_builder.cpp:122`); `n_tokens`/`n_past`/`n_kv` are just three `env.set` calls.
Nothing in the C++ evaluator is single-axis. The only API-level constraint is
`GraphBuilder::build(uint32_t n_tokens, uint32_t n_past)`, which needs an overload taking a map (or a
small `DynamicAxes` struct) so the driver can bind whatever the topology declares.

**Design sketch.**

1. An **axis vocabulary** with real names — `n_samples`, `n_frames`, `n_enc_frames`, `n_tokens`,
   `n_latent`, `n_codes`, `batch` — defined in one place, with a docstring per name saying what it
   counts and at what rate.
2. The export config (R3) declares, per model input, which axis each dynamic dimension *is*: exactly
   optimum's `inputs = {"waveform": {1: "n_samples"}}`. That declaration replaces `symbol_overrides`:
   Kokoro would say `{"noise_in": {1: "600*n_frames"}}` and the relation is a first-class fact rather
   than a patch keyed on a coremltools-internal symbol name.
3. The derivation walk keeps doing what it does — it already derives `floor(n_samples/160) + 1` from
   the conv formula — but seeds from the *declared* axis of the function input it bottoms out at,
   instead of the hardcoded fallback.
4. `n_tokens` stays a legal name (it is the right name for LLM and text-encoder inputs) and every
   existing model keeps a correct export; the change is that it stops being the *only* name.
5. Emission: unchanged (`render()` already prints whatever symbols the expression carries). Runtime:
   the driver binds each declared axis before `run_subgraph`.

**Verification.** The snapshot gate plus `compare_snapshots.py` — but note that this change *renames*
symbols, so equivalence has to be checked under a substitution (`n_samples := n_tokens`) rather than
numerically at fixed probes. Extending `compare_snapshots.py` with an alias map is part of the item.

**Risk.** Low-to-medium. The failure mode is a topology declaring an axis the driver never binds, which
`SymbolEnv::get` already turns into a loud `unbound symbol` error rather than a wrong number.

---

## R2 — lower into a `loom.*` MIL dialect with passes, not at emission time

**The question.** Can most of the exporter's remaining ambiguity be removed by rewriting the MIL graph
in place — custom ops plus real `PassPipeline` stages — so that the JSON emitter becomes mechanical?

**Short answer: yes, and the codebase is already half-way there, but it is a relocation of complexity
and must be judged on what it *removes*, not on how it looks.** `EXPORT-IMPROVEMENT.md` rejected
StableHLO for exactly this reason and that reasoning still applies. The difference is that rewriting
*inside* MIL keeps every op at the abstraction level it already has (no decompose-then-re-fuse), and a
pass is testable against graph structure instead of against emitted JSON.

**What already exists.** `dialect.py` registers three custom ops (`loom_fused_attention`, `loom_spline`,
`loom_group_norm`); `passes.py` runs one real MIL→MIL pass (`fuse_gqa_repeat_kv`) and its docstring
already states the principle: *coremltools' own backend never mixes graph rewriting with serialization*.
The rest of the lowering happens in `topology_ops.py` at emission time, where composites are synthesized
node-by-node and guards re-derive facts about the graph each time they run.

**One of those three ops is registered but never produced, and it is load-bearing** (measured
2026-08-01). `loom_fused_attention` has a MIL op class, a lowering to the engine's `ATTENTION` primitive
(`exporter.py:131`) and a pass to create it, `FuseLoomAttention` — whose `_fuse_blocks` body is `pass`,
a documented placeholder (`dialect.py:259`). So no MIL export has ever emitted an `ATTENTION` node:
Qwen3's exported topology is `ADD 450, MUL 450, MUL_MAT 254, RESHAPE 228, PERMUTE 142, VIEW 140, …` —
attention fully expanded — while the bespoke converter hand-writes the node with its layer index
(`convert_qwen3.py:93`).

That is not cosmetic, because **`ATTENTION` is the only door to the engine's KV cache**
(`src/ops/primitives_attention.cpp:29-80`): a topology without that node cannot use one. So this
missing pass is the prerequisite for KV-cached generation in every MIL-exported causal LM, and
therefore for R5's family 2/6 cross-attention decoder loop, which the table below lists as needing "a
cross-attention decoder loop with KV cache". **Implementing `FuseLoomAttention` is the highest-value
item in R2 by a distance** — every other pass here removes an emitter guard; this one unlocks a
capability. See `EXPORT-PREPARATION.md` §4 for the full four-step breakdown and BACKLOG.md P4.4.

**What it would remove.** Every one of these is a place the emitter currently has to *decide* something:

| today, at emission | as a pass |
|---|---|
| `matmul` rejects `transpose_x=True` (a known gap, BACKLOG) | `normalize_matmul` rewrites it into an explicit `transpose` + canonical matmul; the emitter sees one form |
| mutual-broadcast detection inserting `REPEAT` nodes by comparing rendered shape strings | `insert_explicit_broadcasts` makes every broadcast a real op; the emitter never compares shapes |
| `less` bypass decided by *numerically probing* two derived expressions at 8 sequence lengths | `fold_length_masks` proves the mask is all-true once, on the graph, and deletes it |
| `reduce_mean` split three ways by whether the reduced axis is static and whether it is `ne[0]` | `lower_reduce_mean` rewrites into `reduce_sum` + `scale` or into a `loom.mean` op, per the same rule, once |
| `conv_transpose` depthwise "stuffing" composed inline from PAD/RESHAPE/VIEW/CONT | `loom.conv_transpose_dw` op, expanded by a pass |
| `pad(mode="replicate")` composed from VIEW/REPEAT/CONCAT with a hand-built dynamic offset | `loom.replicate_pad` op |
| `stack` composed from RESHAPE+CONCAT | `loom.stack` lowering pass |
| attention emitted as ~20 raw nodes per layer, with no way to reach the engine's `KvCache` | `FuseLoomAttention` (registered, body is a `pass`) emits `loom_fused_attention` → the `ATTENTION` primitive, which *is* the cache's only entry point — see the note above |
| shape expressions re-derived by a backward walk at every consumer | `annotate_dynamic_shapes` pass writes the resolved expression onto each Var once (the machinery exists in `value_facts.py`; today it is a cache, not an annotation) |

There is a second, larger prize downstream: `BACKLOG.md` tracks the C++ "dynamically heal
transposed/permuted layouts" heuristics in `primitives_basic.cpp` (`op_mul`, `op_repeat`) as unverified
and probably obsolete. They exist because the exporter used to emit ambiguous layouts. A canonicalizing
pass is what lets those be deleted with an argument rather than a hope.

**Sequencing.** This is the one item that can be done incrementally with a hard gate at every step: each
pass must be output-preserving under `snapshot_gguf.py` + `compare_snapshots.py`, and each one removes
a guard from `topology_ops.py`'s rule table (which prints itself — `python3 -m
loom_mil_compiler.topology_ops` — so the shrinkage is directly observable). Suggested order: the two
canonicalizers first (`normalize_matmul`, `insert_explicit_broadcasts`), because they are pure
rewrites with no new ops; then the composite ops; then `annotate_dynamic_shapes`, which is the one that
interacts with R1.

**Explicit non-goal.** Rewriting in place makes the MIL program stop being a faithful record of the
traced model, which is exactly what makes `Program`-level debugging pleasant today. Every pass should be
individually switchable, and `apply_loom_mil_passes` should keep printing what it changed.

---

## R3 — a task/architecture registry and `LoomModelFor*` entry points

**What.** The `optimum` object model, with Loom's own vocabulary:

```python
# what a user should have to write, in full:
from loom.exporters import export
export("nvidia/parakeet-tdt-0.6b-v3", "parakeet.gguf")            # task inferred
export("hexgrad/Kokoro-82M", "kokoro.gguf", task="text-to-speech")

# or from the shell
loom-export nvidia/parakeet-tdt-0.6b-v3 -o parakeet.gguf
```

**Pieces**, each mapping onto something that already exists in some form:

| piece | mirrors | built from |
|---|---|---|
| `LoomExportConfig` | `OnnxConfig` | the three existing family templates' spec dataclasses |
| ~~`.inputs` / `.outputs` (named axes)~~ — **not on the config; see below** | `OnnxConfig.inputs` | R1 |
| `.generate_dummy_inputs()` | `DummyInputGenerator` | the `torch.randn` blocks in the current scripts |
| `.patch_model_for_export()` | `ModelPatcher` | the wrapper classes in the current scripts (R4) |
| `.decomposition` (which submodels, which driver) — **built, see `decomposition.py`** | `OnnxSeq2SeqConfigWithPast` | `ModularExportSpec`, `FlowMatchingSpec`, `NeMoASREncoderSpec` |
| `TaskRegistry` | `TasksManager` | new; keyed on HF `config.json` `model_type`/`architectures`, and on the `target` class inside a `.nemo` archive |
| `LoomModelForCTC` / `ForSpeechSeq2Seq` / `ForCausalLM` / `ForTextToSpeech` | `ORTModelFor*` | the Lua drivers + C++ backends behind one Python/C++ surface |

**Correction: `.inputs` belongs to a phase, not to a config** (decided in BACKLOG.md P4.0.2, after R1
and P3 had both landed). `OnnxConfig.inputs` works as a config-level property because an `OnnxConfig`
describes exactly one graph. A `LoomExportConfig` frequently does not: 5 of the 11 models exported today
are multi-phase (Kokoro 2, VITS 3, Matcha 4, Supertonic 4, StyleTTS2 3), each phase with its own input
signature, its own dynamic axes, and — for Kokoro — its own `root_axis`. A config-level `.inputs` for
those is necessarily `{phase: {input: {axis: name}}}`, which is `ExportPhase` with an extra level of
nesting and no extra information. So the axis declaration stays where R1 put it: on the phase
(`multi_phase_export.ExportPhase.root_axis`/`declared_axes`), or on the single-graph family's own
`export()` for the families that trace exactly one topology.

What P4.0.2 did build instead is the check that makes a per-phase declaration safe: `LoomGGUFExporter`
now validates that every dynamic input axis is accounted for. That closes the actual hole a schema
would have closed — see BACKLOG.md P4.0.2 for the two silent failure modes.

**The one design decision that is not a port.** In `optimum`, the exported artifact is a graph and the
*orchestration* lives in the Python runtime class. Here the orchestration is exported too, as the
embedded Lua driver. So `LoomExportConfig` owns something `OnnxConfig` does not: the driver template.
That is a feature — it is why a Loom GGUF is self-contained — but it means the family templates are
doing strictly more work than an `OnnxConfig`, and the registry has to carry driver templates as
first-class artifacts (they are already `.lua` files with a `--@loom:samplers` marker; generalize that
into a proper template mechanism rather than string replacement).

**Acceptance criterion, stated up front:** for every flagship family, the export script is deleted and
replaced by a registry entry plus a CLI invocation. Not "shortened" — deleted.

---

## R4 — transparent export: no user-written wrapping

**What is wrong.** `export_kokoro_mil.py` is 503 lines; `export_matcha_mil.py` 448; `export_vits_mil.py`
382; `export_styletts2_mil.py` 350. BACKEND.md's item-4 measurement already established what is in
them: **not orchestration — tracing setup.** Wrapper `nn.Module`s, dummy inputs, `RangeDim`
declarations, per-phase topology assembly, and import stubs for version-pinned third-party packages.

**Is the wrapping strictly necessary?** Mostly yes, and it is worth being precise about why, because
that determines what can be automated:

| why a wrapper exists | necessary? | can it be declared instead of written? |
|---|---|---|
| reduce a multi-value `forward` to one tensor (the engine's one-output-per-topology rule) | yes | **yes** — `NeMoASREncoderSpec`'s `EncoderOutput` already does exactly this, as a validated claim |
| call a submodule that is not the model's own `forward` (StyleTTS2 phases, Matcha's estimator) | yes | **yes** — an attribute path + a call signature, as `ModularExportSpec` already declares |
| neutralize trace-hostile source code (`sequence_mask` with a dynamic `torch.full`, `.item()` calls, Python-side sampling loops) | yes | **partly** — these are genuine per-model monkeypatches; they belong in a `ModelPatcher` subclass, versioned with the family, not inline in a script |
| import-order stubs (`transformers.dependency_versions_check`, `diffusers` version gates) | yes | **yes** — environment preparation is family-wide; `nemo_asr_export.prepare_nemo_environment()` is the pattern |
| kwarg → positional adaptation for `torch.jit.trace` | yes | **yes** — mechanical, from the config's declared input names |
| naming: binding a traced value to a Python local *renames the topology's declared output* | — | must be **avoided**, and the rule needs to live in the framework: this was found this week, when adding one `selected = ...` local silently renamed both Parakeet outputs |

So: wrapping stays, but as `patch_model_for_export()` per family, exactly like optimum. The user-visible
surface for a flagship family becomes a repo id and a task.

**Where the long tail goes.** Kokoro-style pipelines will not collapse to zero lines, and pretending
otherwise is how a template becomes a worse thing to read than the code it replaced (item 4's own
lesson). The realistic target: flagship families are fully transparent; the long tail writes a
`LoomExportConfig` subclass — a declaration with validation — instead of a script with a `main()`.

---

## R5 — cover everything CrispASR covers: the grouping study

`/home/flavio/Dev/crispasr/models/` holds **120 `convert-*.py` scripts**. That is not 120 families, and —
as P0.3 found — it is not 120 models either, in *both* directions at once.

**Status: confirmed (P0.3), with corrections.** The table below replaces the original hypothesis, which
was read off the README's architecture column alone. Evidence: the module docstring of all 120
converters (every one of them states its architecture explicitly, several line-by-line against the
checkpoint) plus CrispASR's own README model tables, which list the *checkpoints* each converter covers.
Not read: the 120 converter bodies. That is enough to group by architecture and to count models; it is
not enough to cost a template, which is P4's job per family.

**Four corrections that change the plan:**

1. **The file count overstates and understates at the same time.** 12 of the 120 are not model
   converters at all — 5 voice/reference bakers (`kokoro-voice`, `cosyvoice3-voices`, `kugelaudio-voice`,
   `vibevoice-voice`, `tada-ref`), 3 non-GGUF format utilities (`whisper-to-coreml`,
   `whisper-to-openvino`, `h5-to-coreml`), 3 alternate write paths for one model (`vibevoice-large`,
   `vibevoice-stream-gguf`, `vibevoice-stream-q4k`) and 1 duplicate of another converter's model
   (`chatterbox-gianni`). Meanwhile one converter routinely covers 2–6 checkpoints (`convert-parakeet`
   alone backs 6 README rows; `convert-piper` backs every Piper voice). **~108 real converters,
   ~165 README-listed models.**
2. **Family 2 is not "Whisper-family".** The shared shape is *audio encoder → AR cross-attention
   decoder*, and in more than half the members the encoder is a **Conformer**, not Whisper's conv+
   transformer stem: canary, firered-asr, firered-lid and cohere-asr are all Conformer-encoder AED.
   That is good news — the encoder half is family 1, already exported. What family 2 actually needs is
   the *cross-attention decoder loop*, which is also what family 6 needs. They should be one template.
3. **Family 3 is bigger than estimated: ~19 converters / ~36 models, not ~20.** It is the single
   largest group by a wide margin and the conclusion below only gets stronger.
4. **The 9/10 split is not a partition — it is two stages of the same pipeline.** chatterbox,
   cosyvoice3, tada, voxtral-tts, pocket-tts, kugelaudio and voxcpm2 each have an AR token LM *and* a
   flow-matching/diffusion acoustic stage, so the hypothesis counted them twice. The axis that actually
   predicts exporter work is the **acoustic decoder**: flow-matching ODE (9) vs. RVQ/FSQ codec decoder
   (11) vs. HiFi-GAN/iSTFT vocoder (7/8). The AR-LM half of all of them is `ModularExportSpec`, done.
   A fourth acoustic decoder appeared that the hypothesis filed under "one-offs": **mel-spectrogram TTS
   + HiFi-GAN** (fastpitch, speecht5, bananamind-tts), which shares its vocoder with 7/8.

| # | family | representative members | conv. | models | the connector it needs | Loom status |
|---|---|---|---|---|---|---|
| 1 | **Conformer/FastConformer encoders** + CTC / TDT / RNNT heads | parakeet ×6, reazonspeech, `stt_*_fastconformer_ctc_*` ×4, canary-ctc, nemotron-streaming, **GigaAM v3 ×4** | 4 | ~13 (+4) | CTC head (done) / TDT / RNNT decode loop (host-side) | **done for the four checkpoints here** — the CTC leaf is `ASRNemoEncoderExportConfig`, the transducer leaves are `BaseTransducerExportConfig`; GigaAM v3's second loader landed in P4.2 and is what split loader from template |
| 2 | **Audio encoder + AR cross-attention decoder** (AED) | whisper (all sizes), distil-whisper, tiron, canary, cohere-asr ×2, firered-asr, firered-lid, moonshine ×3, moonshine-streaming, whisper-vad | 8 | ~12 | **cross-attention decoder loop with KV cache** — shared with family 6 | **whisper done** (P4.1): the connector is `PrefillDecodeLoop.bound`, one field, and family 6 needs no more than it |
| 3 | **Speech-LLM adapters** (audio encoder → projector → causal LM) | voxtral, voxtral4b, qwen3-asr ×4, glm-asr, granite ×7, gemma4 ×2, higgs-stt, mimo-asr, canary-qwen, omniasr-llm ×2, moss ×3, vibevoice ×2, ark-asr, lfm2-audio ×2, mini-omni2, kyutai-stt ×2, funasr ×2 | 19 | **~36** | **a projector** (linear / 4-frame stack / Q-Former / VQAdaptor / GatedMLP) + embedding-injection driver | **template done, two leaves** (P4.3, P4.3c): `speech_lm_export.BaseSpeechLMExportConfig`, accepted on Qwen3-ASR-0.6B and Granite-Speech-4.0.1b. The connector is `PromptSegments` — the prompt fed to the KV cache as one cached call per text/audio segment, which needs no tensor concatenation at all — plus three fields on `PrefillDecodeLoop`. The second leaf shares neither the encoder (window attention vs. conformer) nor the projector (linear vs. **Q-Former**) with the first, and it needed **no new driver component**: what the two share is the log-mel frontend and the `(samples_per_chunk, frames_per_chunk)` contract. Since P4.3d/P4.3e the chunk padding costs nothing either: the encoder takes the caller's real sample count and masks the padding out of its attention, convolutions and projector windows, and both leaves now reproduce HF's rows on a partially filled chunk (4.8e-07 and bit-exact, in torch) |
| 4 | **CNN + transformer + CTC** | wav2vec2, data2vec, hubert, omniasr-ctc ×2, tada-aligner | 3 | ~6 | CTC head (already done for family 1) | not started; family-1-shaped |
| 5 | **SANM / FunASR encoders** (+ CIF, + CTC) | funasr ×2, paraformer, sensevoice | 3 | ~4 | CIF predictor + NAR decoder (paraformer only) | not started |
| 6 | **Encoder-decoder text** (translation) | m2m100, wmt21 ×2, madlad (T5) | 2 | ~4 | same decoder loop as family 2 | not started |
| 7 | **VITS / VITS2** | piper (many voices), melotts (+bert-base cond.), openvoice2 (TCC), rvc | 5 | 4 + voices | — (self-contained) | **done** (piper) |
| 8 | **StyleTTS2 / iSTFTNet** | kokoro (+ per-voice packs), styletts2 | 2 | 2 + voices | — (self-contained) | **done** |
| 9 | **Flow-matching / diffusion acoustic stage** | matcha, supertonic, f5-tts, cosyvoice3 (DiT-CFM), voxcpm2 (LocDiT), kugelaudio (DiT), chatterbox (S3Gen), tada, voxtral-tts, pocket-tts (LSD), dots-tts, irodori-tts | 12 | ~12 | the ODE loop (done) + per-model preconditioning | **2 done** (`FlowMatchingSpec`) |
| 9b | **Mel-spectrogram TTS + HiFi-GAN** *(new — was "one-offs")* | fastpitch (non-AR), speecht5 (AR), bananamind-tts (Tacotron-lite) | 3 | 3 | duration/pitch predictor or AR mel loop; vocoder already exported inside 7/8 | not started |
| 10 | **AR LM + neural codec TTS** | orpheus+SNAC, outetts+WavTokenizer, qwen3-tts ×4, moss-tts ×2, miotts, omnivoice, csm, dia, bark, zonos+dac, indextts, parler, vibevoice-tts ×2, lfm2-audio-tts | 16 | ~20 | **the codec decoder (family 11)** + a delay/RQ token-emission driver | LM half done; codec decoders are the missing piece |
| 11 | **Neural audio codecs / vocoders** (decode side) | DAC, dacvae, SNAC, WavTokenizer, MioCodec, mimo-tokenizer, omnivoice-tokenizer, cosyvoice3-s3tok, qwen3-tts-tokenizer, tada-codec, tada-encoder | 11 | ~11 | — (it *is* the connector for 10) | HiFi-GAN/iSTFT done inside 7/8/9; RVQ/FSQ not started |
| 12 | **BERT-family token classifiers** | fireredpunc, fullstop-punc, punctuate-all, pcs (4 heads), bert-base | 4 | 5 | — (one linear head) | not started — smallest possible template |
| 13 | **Small classifiers / embedders / VAD / LID** | titanet, ecapa-tdnn-lid, cosyvoice3-campplus, silero-vad, silero-lid, marblenet-vad, firered-vad (DFSMN), pyannote-seg, whisper-vad, crepe, cld3, glotlid, lstm-truecaser | 13 | ~13 | — (single forward, argmax) | not started — bigger than estimated |
| 14 | **Music / audio analysis CNN-RNNs** | beat-this, btc, tabcnn, piano-transcription, mel-band-roformer, htdemucs | 6 | ~6 | — | not started |
| — | **genuine one-offs** | audioseal (watermark), beatrice (VC), sidon (restoration) | 3 | 3 | — | — |

**The shape of the plan this implies — unchanged, and now with a bigger margin.** Coverage is dominated
by families 1, 3, 10, and all three are *compositions* of pieces Loom already exports (a
conformer/whisper encoder, a causal LM, a codec decoder) plus one connector each (a CTC/TDT head, a
projector, a codec). The highest-value next template is not another architecture: it is a **composition
mechanism** — declare "this model is encoder family X + adapter Y + LM family Z" and get the driver
generated. R3's registry is the natural place for it, and family 3 (**~36 models**) is the acceptance
test.

Ordering proposal, by coverage-per-effort (revised by the corrections above):

1. **Whisper** (family 2) — the one flagship with no MIL export, and a prerequisite for ~10 models in
   family 3. Also the project's own reference model. Deliver the **cross-attention decoder loop** as the
   reusable half: it is also all family 6 needs, and it is the half families 1 and 3 do *not* provide.
2. **GigaAM v3** (family 1) — cheapest flagship by graph work, most informative by plumbing; see below.
3. **Composition/adapter template** (family 3), on top of Whisper + the existing `ModularExportSpec`.
   ~36 models, the single largest lever in the roadmap. **DONE (P4.3)** — and it needed neither Whisper's
   encoder nor `ModularExportSpec`: the LM half is an ordinary `Flattened`-style trace taking
   `inputs_embeds`, and what the composition actually turns on is that a KV-cached decoder makes a
   segmented prefill identical to a concatenated one.
4. **BERT token classifiers** (family 12) — trivially small, and it proves the registry works for a
   non-audio task.
5. **Codec decoders** (family 11) — unlocks family 10's back half (~20 models), and 11 is itself ~11
   models, so it pays twice.
6. **CNN+CTC** (family 4) and **SANM** (family 5) — both are family-1-shaped once the encoder template
   is generalized past NeMo. Family 4 needs no new head at all.
7. The long tail, as `LoomExportConfig` subclasses.

Two ordering notes P0.3 produced that were not in the hypothesis: family 2's decoder loop should be
built as a *shared* artifact with family 6 rather than a Whisper-specific one, and family 9b
(mel+HiFi-GAN TTS) is cheap enough to fold in wherever convenient — its vocoder is already exported as
part of families 7 and 8.

### GigaAM v3 — a flagship Loom should cover even though CrispASR does not

Tracked here at the author's direction. It is a coverage gap on *both* sides: CrispASR has no backend for
it (`crispasr/docs/learnings.md` §10 lists it, alongside MedASR, as existing only in `transcribe.cpp`), and
Loom has never exported it. That makes it the one flagship where Loom would be *ahead* of the fork rather
than catching up.

What it is: a Conformer-based Russian(+EN) foundation model, 220–240M parameters, released as five
variants — `ssl` (HuBERT-CTC pretrained encoder), `ctc`, `rnnt`, `e2e_ctc` and `e2e_rnnt`, the two `e2e`
ones adding punctuation and text normalization. CrispASR's "4 variants" counts the four ASR heads and
excludes `ssl`.

Why it is cheap: **the graph is family 1.** A Conformer encoder plus a CTC or RNN-T head is precisely
what `NeMoASREncoderSpec` already exports and what the Parakeet TDT/RNNT drivers already decode
host-side. `e2e_ctc`'s punctuation is carried in its 256-piece SentencePiece vocabulary rather than by a
separate model, so it stays an ordinary CTC decode — no new graph work and no second network.

Why it is worth doing early anyway: **it breaks the loader assumption, which is exactly what R3 needs
proving against.** The three current NeMo models are all `ASRModel.restore_from(<file>.nemo)`; GigaAM v3
is loaded either through its own `gigaam` package (`gigaam.load_model(...)`) or through
`AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision="<variant>", trust_remote_code=True)`. Today
`nemo_asr_export.py` hardcodes the NeMo loader, so adding GigaAM forces the split the registry wants:
the template's real contract is *"give me an `nn.Module` and tell me what its forward returns"*, and
"which library restored it" belongs in a per-model loader entry. Doing this with two loaders and one
family template is a much better test of that boundary than doing it with one.

**Done, 2026-08-09 — BACKLOG.md P4.2, `gigaam_export.py`.** Both open questions answered, one each way:

* **The heads match exactly.** `modeling_gigaam.RNNTJoint` and `RNNTDecoder` are the same three modules
  under the same three names as NeMo's, so the joint wrapper, the embedding phase, the LSTM cell phase
  and the whole decode loop are shared *verbatim* — `transducer_export.BaseTransducerExportConfig`, with
  the two leaves supplying only their loader and where the checkpoint keeps those three modules. The one
  thing that did move is the depth: GigaAM's prediction network is one layer where Parakeet's is two, so
  `RecurrentPhase` gained `number_layers`.
* **The remote code does need patching, but not for the predicted reason.** No `.item()`, no Python
  control flow. Two other things: a lazily-built `persistent=False` rotary buffer that must be
  materialized outside `inference_mode` before tracing, and torchaudio's `MelSpectrogram`, which cannot
  be converted at all — its batch pack/unpack reads `.shape` on a *complex* tensor, emitting a
  `complex_shape` op coremltools cannot lower. Both handled in `load_model`, the second by a rewritten
  frontend that every export checks against the real one.

And the prediction that the split is what this exercise buys held: the loader is now four lines and two
argument names, and `nemo_asr_export.py`'s `EncoderOutput`/`build_trace`/`ASREncoderWrapper` are family-1
machinery that GigaAM imports without a parameter. Two exporter bugs fell out of it, both older than this
model and both silent (see P4.2) — the more interesting of the two is that a wrong encoder still decoded
71 of the first 80 tokens correctly, which is the argument for tensor-level oracles over token-level ones.

---

## R6 — retiring the bespoke converters

**Current state:** `tools/convert_*` is 10 directories, ~14,000 lines of Python
(`convert_kokoro` 3,764; `convert_supertonic` 2,470; `convert_nemo` 2,287; …). They are *not* dead code
today: several C++ reference tests consume fixtures they generate, and BACKEND.md's verification chain
leans on them as numerical oracles.

**Policy for this thread.** Per model, in order:

1. the MIL export exists and passes the same numeric reference test the bespoke path backs;
2. the test is re-pointed at the MIL-exported GGUF (several already are — `test_e2e_*_mil_*`);
3. any fixture-generation the test still needs moves to `tools/fixture_gen/`;
4. the bespoke converter is deleted in the same commit that re-points the last test consuming it.

Nothing is deleted while it is the only description of a model's numerics. The reference-oracle role is
real and has caught real bugs — but it is a role for `tests/`, not for a parallel converter.

The same policy applies to the long `export_*_mil.py` scripts (R4): they are deleted when the family
config that replaces them passes the same tests, not before.

**Documentation.** The advanced/bespoke MIL path (`LOOM_MIL_CONVERSION.md`) stays documented — hand-built
`Program`s and hand-written drivers remain supported for extreme cases, and `test_compiler.py` covers
that entry point. What goes away is the expectation that a *user* touches it for a mainstream model.

---

## R7 — profiles: my take on dropping `atomic`

**Measured first**, because the stated reason to drop atomic (file size) turns out not to be about
atomic. Same model, same weights, three profiles:

| profile | tensors | stored | unique payloads | redundant | topologies |
|---|---|---|---|---|---|
| `monolithic` | 257 | 1353 MiB | 1353 MiB | **0** | 1 |
| `atomic` | 313 | 1609 MiB | 1353 MiB | **256 MiB** | 21 |
| `modular` | 307 | 1609 MiB | 1353 MiB | **256 MiB** | 20 |

Atomic and modular are the same size to within 12 KB. And the entire 256 MiB is **one tensor**:
LFM2's tied embedding matrix, `[1024, 65536]`, stored twice — once as `prefix.module_weight` and once as
`suffix_1.module_weight`. Every other duplicate in the file is a bias vector; they total 0.5 MiB.

So: **the size argument does not distinguish the profiles, and it is a writer bug, not a profile
property.** Content-addressing weight payloads in the GGUF writer (hash → emit once → alias the name)
removes 256 MiB from *both* split profiles and is a small, isolated change. It should be done first, and
it should be done regardless of what happens to atomic.

**With size off the table, the real difference is how the split is obtained:**

* `atomic` **infers** the boundary, by partitioning one flat trace on coremltools'
  `ScopeSource.TORCHSCRIPT_MODULE_NAME` metadata, with a regex fallback for hand-built programs — and if
  partitioning fails for any reason it **downgrades to monolithic with a warning** and produces a
  working-but-different model.
* `modular` **declares** the boundary (`ModularExportSpec`), traces each piece independently, and
  re-derives the repeated blocks structurally (`find_repeated_blocks`) so a wrong claim raises
  immediately.

That is precisely the distinction BACKEND.md's closing section arrived at from the other direction:
*declare only what varies, re-derive the rest structurally, fail loudly when the claim and the model
disagree*. Atomic is the one part of the exporter still on the other side of that line, and its silent
downgrade is a footgun — it is the same "produces something plausible instead of failing" shape as the
two shape bugs found this week.

**Decision (approved): retire `profile="atomic"`.** Two profiles remain — `monolithic` (default) and
`modular` — and `modular` is the only split mechanism. Steps:

1. dedup the writer (above), so the comparison is honest;
2. show that `lfm2_350m_modular` and `lfm2_350m_atomic` have the same *runtime* peak-activation
   profile, not just the same topology count — this is the capability atomic was built for (defer
   intermediate-state memory to the driver instead of pre-allocating), and it is the only thing that
   would justify keeping two mechanisms;
3. delete the `profile="atomic"` branch and keep scope-based partitioning *only* as an optional,
   opt-in discovery aid for `ModularExportSpec` (BACKLOG already tracks "Phase 2: fully automatic
   prefix/suffix boundary discovery" — that is the useful half of atomic, and it belongs there, where a
   wrong guess is checked against a declaration instead of silently exported).

That leaves two profiles with a clear rule: **monolithic** when the whole graph fits one topology and
speed matters; **modular** when the driver should own activation memory or the model has real
structural boundaries. Both are declared, neither silently degrades.

Step 2 is a *nice-to-have measurement, not a blocker*: it is worth knowing whether the split profiles
deliver the memory win they were built for, but the decision to remove atomic does not depend on the
answer, since modular provides the same per-layer topologies either way.

---

## Flagship families

The four families that must export with **zero user-written Python** — a repo id and a task, nothing
else — and against which R3/R4 are judged:

| flagship | family | state |
|---|---|---|
| **Whisper** | 2 | MIL export done (BACKLOG.md P4.1) — `whisper_export.py`, two phases + a generated decode loop; the bespoke converter still backs the whisper-tiny tests (R6 blocker recorded there) |
| **NeMo ASR** (Conformer-CTC, Parakeet TDT/RNNT) | 1 | MIL export done, still a script per model |
| **GigaAM v3** | 1 | MIL export done (BACKLOG.md P4.2) — `gigaam_export.py`, the `e2e_rnnt` variant through the shared transducer template; the CTC variants are unclaimed for want of a checkpoint |
| **Qwen3** | causal LM / speech-LLM | `ModularExportSpec` done for the base LM; ASR/TTS variants not started |

"Covered" for a flagship means: the export script is deleted, the registry entry exists, and the
numerical reference test runs against a GGUF produced by `loom-export <repo-id>`.
