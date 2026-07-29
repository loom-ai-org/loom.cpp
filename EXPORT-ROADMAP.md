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
| submodel decomposition (`encoder_model.onnx`, `decoder_model_merged.onnx`) | family-level decision, not per-model | `SubmoduleExportSpec` (one family), profiles (R7) |
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
| `.inputs` / `.outputs` (named axes) | `OnnxConfig.inputs` | R1 |
| `.generate_dummy_inputs()` | `DummyInputGenerator` | the `torch.randn` blocks in the current scripts |
| `.patch_model_for_export()` | `ModelPatcher` | the wrapper classes in the current scripts (R4) |
| `.decomposition` (which submodels, which driver) | `OnnxSeq2SeqConfigWithPast` | `SubmoduleExportSpec`, `IterativeRefinementSpec`, `NeMoASREncoderSpec` |
| `TaskRegistry` | `TasksManager` | new; keyed on HF `config.json` `model_type`/`architectures`, and on the `target` class inside a `.nemo` archive |
| `LoomModelForCTC` / `ForSpeechSeq2Seq` / `ForCausalLM` / `ForTextToSpeech` | `ORTModelFor*` | the Lua drivers + C++ backends behind one Python/C++ surface |

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
| call a submodule that is not the model's own `forward` (StyleTTS2 phases, Matcha's estimator) | yes | **yes** — an attribute path + a call signature, as `SubmoduleExportSpec` already declares |
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

`/home/flavio/Dev/crispasr/models/` holds **120 `convert-*.py` scripts**. That is not 120 families. A
first-pass grouping from the README's own architecture column (this is the *hypothesis*; the first
deliverable of this item is to confirm it against the converters themselves):

| # | family | representative members | count | Loom status |
|---|---|---|---|---|
| 1 | **Conformer/FastConformer encoders** + CTC / TDT / RNNT heads | parakeet ×6, reazonspeech, `stt_en_fastconformer_ctc_*`, parakeet-ctc ×3, **GigaAM v3 ×4** | ~16 | **mostly done** — `NeMoASREncoderSpec` (encoder); TDT/RNNT decoder still host-side; GigaAM needs a second loader (below) |
| 2 | **Whisper-family encoder-decoder** | whisper (all sizes), distil-whisper, moss-diarize | ~6 | bespoke only — **no MIL export at all** |
| 3 | **Speech-LLM adapters** (audio encoder → projector → causal LM) | voxtral, voxtral4b, qwen3-asr ×3, glm-asr, granite ×4, gemma4 ×2, higgs-stt, mimo-asr, canary-qwen, omniasr-llm, moss-audio, vibevoice ×2, ark-asr, lfm2-audio | ~20 | LM half done (`SubmoduleExportSpec`); encoder+projector half is family 2/1 plus a projector |
| 4 | **CNN + transformer + CTC** | wav2vec2, data2vec, hubert, omniasr-ctc ×2 | ~5 | not started; closest existing shape is family 1 |
| 5 | **SANM / FunASR encoders** (+ CIF, + CTC) | funasr ×2, paraformer, sensevoice | ~4 | not started |
| 6 | **Encoder-decoder text** (translation) | m2m100, wmt21, madlad (T5) | ~3 | not started; shares its decoder loop with family 2 |
| 7 | **VITS / VITS2** | piper (many voices), melotts | ~2 + voices | **done** (piper) |
| 8 | **StyleTTS2 / iSTFTNet** | kokoro (+ per-voice packs), styletts2 | ~2 + voices | **done** |
| 9 | **Flow-matching / diffusion TTS** | matcha, supertonic, f5-tts, cosyvoice3 (DiT), voxcpm2, kugelaudio, chatterbox (S3Gen), tada | ~8 | **2 done** (`IterativeRefinementSpec`); the rest are the same loop with different preconditioning |
| 10 | **AR LM + neural codec TTS** | orpheus+SNAC, outetts+WavTokenizer, qwen3-tts, moss-tts ×2, miotts, omnivoice, csm, dia, bark, zonos+dac, indextts, parler, pocket-tts, voxtral-tts, vibevoice-tts | ~16 | LM half done; **codec decoders are the missing piece** |
| 11 | **Neural audio codecs / vocoders** (decode side) | DAC, SNAC, WavTokenizer, MioCodec, dacvae, s3tok, miocodec, AudioVAE | ~8 | HiFi-GAN/iSTFT vocoders done inside 7/8/9; RVQ/FSQ codecs not started |
| 12 | **BERT-family token classifiers** | fireredpunc, fullstop-punc, punctuate-all, pcs, bert-base | ~5 | not started — smallest possible template, high model count |
| 13 | **Small classifiers / embedders** | titanet, ecapa-tdnn-lid, silero-vad, marblenet-vad, pyannote-seg, cld3, glotlid, crepe, whisper-vad | ~9 | not started |
| 14 | **Music / audio analysis CNN-RNNs** | beat-this, btc, tabcnn, piano-transcription, mel-band-roformer, htdemucs | ~6 | not started |
| — | **genuine one-offs** | audioseal, beatrice, rvc, sidon, fastpitch, speecht5, bananamind, m2m-adjacent oddities | ~15 | — |

**The shape of the plan this implies.** Coverage is dominated by families 1, 3, 10 — and all three are
*compositions* of pieces Loom already exports (a conformer/whisper encoder, a causal LM, a codec
decoder) plus one connector each (a CTC/TDT head, a projector, a codec). That suggests the highest-value
next template is not another architecture: it is a **composition mechanism** — the ability to declare
"this model is encoder family X + adapter Y + LM family Z" and get the driver generated. R3's registry
is the natural place for it, and family 3 (~20 models) is the acceptance test.

Ordering proposal, by coverage-per-effort:

1. **Whisper** (family 2) — the one flagship with no MIL export, and a prerequisite for ~10 models in
   family 3. Also the project's own reference model.
2. **GigaAM v3** (family 1) — cheapest flagship by graph work, most informative by plumbing; see below.
3. **Composition/adapter template** (family 3), on top of Whisper + the existing `SubmoduleExportSpec`.
4. **BERT token classifiers** (family 12) — trivially small, and it proves the registry works for a
   non-audio task.
5. **Codec decoders** (family 11) — unlocks family 10's back half.
6. **CNN+CTC** (family 4) and **SANM** (family 5) — both are family-1-shaped once the encoder template
   is generalized past NeMo.
7. The long tail, as `LoomExportConfig` subclasses.

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

Two things to check when the work starts, neither yet verified:

* whether the `trust_remote_code` modeling code traces cleanly under `torch.jit.trace`, or needs a
  `ModelPatcher` (R4) — custom remote code is exactly the category that tends to contain `.item()`
  calls and Python-side control flow;
* whether the `rnnt`/`e2e_rnnt` heads match the RNN-T decoder+joint layout the Parakeet path already
  drives host-side, or differ enough to need their own decoder topology.

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
| `submodule` | 307 | 1609 MiB | 1353 MiB | **256 MiB** | 20 |

Atomic and submodule are the same size to within 12 KB. And the entire 256 MiB is **one tensor**:
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
* `submodule` **declares** the boundary (`SubmoduleExportSpec`), traces each piece independently, and
  re-derives the repeated blocks structurally (`find_repeated_blocks`) so a wrong claim raises
  immediately.

That is precisely the distinction BACKEND.md's closing section arrived at from the other direction:
*declare only what varies, re-derive the rest structurally, fail loudly when the claim and the model
disagree*. Atomic is the one part of the exporter still on the other side of that line, and its silent
downgrade is a footgun — it is the same "produces something plausible instead of failing" shape as the
two shape bugs found this week.

**Recommendation: retire `profile="atomic"` as a user-facing profile**, in three steps:

1. dedup the writer (above), so the comparison is honest;
2. show that `lfm2_350m_submodule` and `lfm2_350m_atomic` have the same *runtime* peak-activation
   profile, not just the same topology count — this is the capability atomic was built for (defer
   intermediate-state memory to the driver instead of pre-allocating), and it is the only thing that
   would justify keeping two mechanisms;
3. delete the `profile="atomic"` branch and keep scope-based partitioning *only* as an optional,
   opt-in discovery aid for `SubmoduleExportSpec` (BACKLOG already tracks "Phase 2: fully automatic
   prefix/suffix boundary discovery" — that is the useful half of atomic, and it belongs there, where a
   wrong guess is checked against a declaration instead of silently exported).

That leaves two profiles with a clear rule: **monolithic** when the whole graph fits one topology and
speed matters; **submodule** when the driver should own activation memory or the model has real
structural boundaries. Both are declared, neither silently degrades.

---

## Flagship families

The four families that must export with **zero user-written Python** — a repo id and a task, nothing
else — and against which R3/R4 are judged:

| flagship | family | state |
|---|---|---|
| **Whisper** | 2 | bespoke converter only; no MIL export |
| **NeMo ASR** (Conformer-CTC, Parakeet TDT/RNNT) | 1 | MIL export done, still a script per model |
| **GigaAM v3** | 1 | never exported; gap in CrispASR too (see R5) |
| **Qwen3** | causal LM / speech-LLM | `SubmoduleExportSpec` done for the base LM; ASR/TTS variants not started |

"Covered" for a flagship means: the export script is deleted, the registry entry exists, and the
numerical reference test runs against a GGUF produced by `loom-export <repo-id>`.
