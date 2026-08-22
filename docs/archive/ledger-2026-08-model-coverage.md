---
type: archive
status: closed
domain: model-coverage
covers: 2026-08-08 – 2026-08-12
last_updated: 2026-08-22
---

# Archive: Flagship Model Coverage, August 2026

> **This file is not maintained.** Verbatim record of the P4 flagship-coverage work — Whisper, GigaAM
> v3, the composition template, and Supertonic's text door. Architecture lives in
> [Epic-03](../epics/epic-03-model-coverage.md); lessons in the [retros](../retros/). Nothing here is
> open work. Do not add to it; do not cite it as current.


- **P4.1 — Whisper — DONE (2026-08-08).** The only flagship with no MIL export, the project's own
  reference model, and a prerequisite for ~10 models in family 3. First family born inside the registry.

  `tools/loom_mil_compiler/whisper_export.py` + `whisper_driver/00_header.lua`. Two phases —
  `encoder` (457 nodes) and `decoder` (740 nodes, 12 cached `ATTENTION` blocks) — one GGUF, and a
  generated driver that is the encoder once followed by `PrefillDecodeLoop`. Registered as
  `automatic-speech-recognition/whisper` against a real `model_type == "whisper"` check, so
  `loom-export <dir> -o whisper_mil.gguf` needs no `--task`/`--model`.

  **The reusable half, which is what the roadmap actually asked for** (`EXPORT-ROADMAP.md`: "deliver the
  cross-attention decoder loop as the reusable half: it is also all family 6 needs"). It is
  `PrefillDecodeLoop.bound` — `{input name: IR expression}` for a step input that is neither
  host-computed from `n_tokens`/`n_past` nor the step's own tokens. That single field is the whole
  difference between a causal-LM decode loop and a cross-attention one, because nothing else about the
  loop changes: a Whisper step is the same cached call at `n_tokens = 1`, with `xa` bound to the
  encoder's single run. A family-6 text encoder-decoder needs the same field and no new component.

  **Finding — decision 2's fourth `Decomposition` is not needed, and building it is what showed why.**
  `EXPORT-PREPARATION.md` §5 decision 2 reserved a `Decomposition` of its own for this shape, reasoning
  that a new orchestration needs a driver builder the family cannot supply. The orchestration turned out
  to be the one `MultiPhase` already has: N independently traced phases, a component list, and
  `MultiPhaseDriverBuilder`. What genuinely differs is **two facts**, and each is now a field on the
  piece that owns it — `ExportPhase.fuse_attention` (per PHASE, because this decoder must be cached and
  this encoder, from the same checkpoint in the same GGUF, must not) and `PrefillDecodeLoop.bound`. A
  `Decomposition` subclass would have restated `MultiPhase.export` verbatim around those two, which is
  the "declare only what genuinely varies" rule the other three templates are built on.

  **Finding — cross-attention is deliberately NOT fused, and that is correct rather than a miss.**
  `fuse_loom_attention` anchors on the `add(scores, mask)` only a masked block has, and Whisper's
  cross-attention has no mask at all. So the pass fuses exactly the 12 self-attention blocks and leaves
  12 `SOFTMAX`-shaped cross-attention blocks expanded. Fusing them would be wrong twice over: the K/V
  would be the encoder's, identical at every step, and `layer` indices are assigned in dense occurrence
  order, so cached cross-attention blocks would consume the slots the self-attention blocks address.
  What it costs is that cross-attention K/V are recomputed per step — the same thing
  `convert_whisper_decoder.py` did with `kv_cache=false`, so this is parity and not a regression.
  A `cross_kv_cache` that projects once from `xa` is the real optimization, and it needs an engine-side
  cache kind that is filled once rather than appended per step; filed under Engine, not done here.

  **Four holes this item found and closed, each one a place a multi-phase export could not say
  something a single-graph export could:**

  1. **A multi-phase GGUF wrote no KV-cache geometry at all.** `_kv_cache_geometry` walks
     `self.program`, and `MultiPhase`'s output exporter has none — each phase was converted by its own
     exporter and only the finished topologies were handed over. The artifact was therefore unloadable:
     `ATTENTION` nodes with `kv_cache=true` and no `loom.n_layer`/`n_head_kv`/`kv_cache_size` for
     `make_kv_cache` to read. `LoomGGUFExporter.phase_programs` + `_fused_ops()` is one enumeration both
     geometries (KV and conv-state) now read from, and the capacity travels from the phase that declared
     it rather than from a second `backend_kwargs()` entry that could disagree.
  2. **`MultiPhaseDriverBuilder.called_topologies` was keyed on `isinstance`**, so a `PrefillDecodeLoop`
     — which declares `topology` as a `TopologyName` exactly like `SubgraphCallComponent` — was invisible
     to it, and the export log reported Whisper's decoder as "a topology no call site names". It now
     reads the components' own declared links, so a new component is counted the day it declares one.
  3. **The encoder output would have crossed the Lua boundary.** 1.15M floats for whisper-small, read
     once per generated token. `SubgraphCallComponent.retain` + an `OutputRef` binding makes it a
     backend-side tensor copy, which is exactly what P4.0.12 built `OutputStore` for; the driver now
     marshals nothing but token ids.
  4. **Two MIL ops had no ggml mapping**, both from the mel frontend: `clip` (MIL's spelling of
     `torch.clamp`, whose bounds are `alpha`/`beta` INPUTS, so the generic path would have dropped them
     — `OP_MAP`'s existing `"clamp"` entry names an op the torch frontend never emits) and `reduce_max`.
     The latter is composed as `RESHAPE` to one row + `POOL_1D(max)` spanning it, the same reduction
     `convert_whisper_encoder.py` hand-wrote, and only for a *global* maximum over statically-sized axes
     — a per-axis maximum is rejected by name rather than approximated.

  **The mel frontend is in the exported graph.** HF's `WhisperEncoder` starts at `input_features`;
  `WhisperMelFrontend` reimplements `WhisperFeatureExtractor._torch_extract_fbank_features` as a
  traceable module (bit-identical to the extractor — `max abs diff 0.0` — with the checkpoint's own
  `mel_filters`), so the exported model takes a **waveform**. Two things a change here must not lose:
  the input is `(1, n_samples)` because a 1-D one makes `torch.stft` trace an `aten::size`/`aten::Int`
  chain, and the magnitude is taken **before** the final-frame slice, because coremltools' complex
  dialect has no `slice_by_index` over a complex tensor.

  **Finding — this task's declared base config class was already false, for Parakeet, and nothing could
  tell.** `tasks.py` declared `automatic-speech-recognition` as building
  `ASRNemoEncoderExportConfig`, but `_build_parakeet_tdt`/`_build_parakeet_rnnt` have returned
  `ASRParakeetExportConfig` (a `BaseMultiPhaseModelExportConfig`) since P4.0.17 step 2. `register()`
  checks the *entry's* `config_class` and nothing checks what a recognizer's `build_config` actually
  returns, so the declaration was never contradicted. Whisper is a third shape again, so the task's base
  is now `LoomExportConfig` — the honest answer for an I/O-contract task whose families genuinely do not
  share an export shape. **Making the check bite again means moving `config_class` onto the
  `ModelRecognizer`**, where the build happens; not done here, because it touches every family and is
  not Whisper's job.

  **Verification.** `tests/test_e2e_whisper_mil_export.cpp` against HF's own
  `WhisperForConditionalGeneration` (`tools/fixture_gen/reference_forward_whisper_mil.py`), on 30 s of
  deterministic noise — harder than speech, which collapses greedy decoding onto a few high-confidence
  tokens where a wrong logit rarely moves the argmax:

  | check | result |
  |---|---|
  | `encoder` topology (mel frontend included) vs HF's encoder | `max_abs_diff = 0.0132` against a reference whose own absmax is 29.86 — 4.4e-4 relative; `mean_abs_diff = 9.3e-06` |
  | the whole driver — encoder, prefill, cached decode loop, greedy argmax — vs HF | **exact**: `522 8002 45981 8 50257`, including stopping on eos |
  | nothing in the test names a per-model C++ struct | `uses_kv_cache()`, `make_kv_cache(*model)` and `loom.n_samples` are all read from the artifact |
  | qwen3-0.6b-base, lfm2-350m-monolithic re-exported after the `fuse_loom_attention` fix | byte-identical by snapshot diff |
  | the Python suite | 481 passed |

  **Follow-up, done 2026-08-08: the driver owns its prompt, and `loom-cli --wav` runs Whisper.** The two
  open host decisions were put to the author, who answered with a rule rather than a case: *an explicit
  argument wins; absent it, autodetect if the model can; otherwise a documented default* — and that this
  should hold for anything a model can infer about its own input, not just Whisper's language. That
  answer decides **where** the logic goes, which is the more important half: whether detection is
  possible at all is a checkpoint fact, so it belongs in the driver, not the host.

  * **`whisper_export.decoder_prompt_constants`** reads five ids off the checkpoint's own generation
    config (`SOT`, the `LANG_LO`/`LANG_HI` window, `TRANSCRIBE`/`TRANSLATE`, `NO_TIMESTAMPS`) and binds
    them as `ExportConstants`. Which of them *exist* is the capability statement: an English-only
    checkpoint has no language or task tokens, so it gets `0`s and the driver correctly builds the short
    prefix and never tries to detect. Read off the config rather than `forced_decoder_ids`, which leaves
    its language slot `None` on a multilingual checkpoint because HF fills it in from detection.
  * **`whisper_driver/01_prompt.lua`** builds the prompt: given language wins, else one decoder step from
    `SOT` alone with the argmax restricted to the language block, else nothing. It costs no extra encoder
    pass (`xa` is already retained) and writes only KV cell 0, which the prefill then overwrites with the
    identical K/V. So `PrefillDecodeLoop` gained `prompt`, an expression for the loop to start from
    instead of `inputs.tokens` — the third and last field this family added to it.
  * **`loom.argmax_row_range(module, row, lo, hi)`** is the new engine binding it needs. The 99 language
    tokens sit *inside* the 51865-wide transcript vocabulary, so an unrestricted argmax answers with a
    word. It is the same `argmax_tensor_row` with an id window (so the two cannot disagree about how a
    maximum is found) and reads only the window off the backend. A separate binding rather than an
    overload, which is the opposite of `argmax_row`'s own documented choice, for a mechanical reason:
    `argmax_row`'s module form already ends in an optional `generation`, so `(module, row, lo, hi)` and
    `(module, row, generation)` cannot be told apart.
  * **`loom-cli`** grew `--language` (resolved to the model's own `<|xx|>` id by TEXT via the new
    `BpeVocab::piece_to_id`, so the CLI carries no per-checkpoint number) and 30 s chunking. The author
    chose sequential windows over rejecting long audio; the seam is real and documented at the loop —
    each window is decoded independently, where Whisper's own long-form algorithm conditions each on the
    previous window's text and cuts at emitted timestamps.

  Three real bugs surfaced doing it: `Vocab::load` **throws** on a `gpt2` schema while `BpeVocab::load`
  politely returns nullptr, so the SentencePiece loader has to be asked second or it kills the run; the
  driver returns the eos token it stopped on, which decodes literally as `<|endoftext|>` in a
  transcript; and — in the test, not the engine — binding `const auto&` into
  `std::get<...>(bridge.call(...))` reads the vector after the returned variant is destroyed. That last
  one is worth remembering because of how it *presents*: the first one or two elements come back as
  freed-heap garbage and the rest are correct, so it reads exactly like a driver computing a wrong
  prompt. It cost a full bisect to find, and the thing that found it was making the driver print each
  token it computed — which were all correct.

  The driver already accepts `task` and `timestamps` on the same terms; only `--language` is exposed on
  the CLI, since that is the one the author decided.

  **R6 retirement — DONE (2026-08-08), once the blocker was removed.** The four `test_e2e_whisper_*`
  tests were fixtured on `whisper-tiny` in **OpenAI `.pt`** form, which this family cannot load; the
  author downloaded the HF conversion (the `.pt` moved to `openai-whisper-tiny/`), and the retirement
  followed. `src/core/whisper_driver.cpp`, its header, `tools/convert_whisper/` (8 files) and the four
  tests are gone, and so is `include/loom/loom_legacy.h` — all six per-model C++ drivers are now retired,
  so the header that carried the policy has nothing left to carry.

  **The coverage was moved, not dropped**, which is the whole content of the R6 rule. The retired tests
  checked four things; `test_e2e_whisper_mil_export` now checks all four, and against a better oracle —
  `test_e2e_whisper_driver` and `test_e2e_whisper_lua_driver` compared two of *this engine's* own
  implementations with each other, while every check here compares against HuggingFace:

  | retired test | what carries it now |
  |---|---|
  | `test_e2e_whisper_encoder_reference` | check 1: the `encoder` topology vs HF's encoder, mel frontend included |
  | `test_e2e_whisper_decoder_reference` | check 1b, added for this: the `decoder` teacher-forced over the whole prompt at `n_past=0`, logits **and** per-row argmax |
  | `test_e2e_whisper_driver` | check 2: the full loop, vs HF's greedy token sequence |
  | `test_e2e_whisper_lua_driver` | check 2, which *is* a Lua driver run — now compared against HF rather than against the C++ driver it was ported from |

  Check 1b runs through a hand-written Lua script rather than a bare `GraphBuilder`, deliberately: the
  mask and positions then come from `loom.causal_mask`/`loom.range`, the same host math a driver uses,
  instead of a second implementation of Whisper's mask living in a test.

  **`whisper-tiny` also made the gate cheap.** 4 layers at `d_model=384` against small's 12 at 768:
  the same test runs in **38 s** instead of ~7 minutes, so the flagship's end-to-end check is now
  something you can run while working rather than once at the end.

  **Measured, since leanness is the stated goal and P4.0.8's own gate asks for it:** `libloom_engine.so`
  12,846,600 → 12,529,328 bytes, −309 KB, with `whisper_driver.cpp` the only translation unit removed.
  8 Python files, 4 C++ tests and 2 headers went with it.

  **Timestamps: interpreted, and driving the chunking (2026-08-08).** `--timestamps` now prints
  `[hh:mm:ss.mmm --> hh:mm:ss.mmm] text` segments, and the long-form loop advances by the model's own
  segment boundaries instead of a fixed 30 s stride — the seam the naive version documented is closed.

  **The finding that made it work at all: omitting `<|notimestamps|>` from the prompt does not get you
  timestamps.** Left to itself the model *emits* that token as its first output — it outscores every
  alternative on most audio — and then produces none. Confirmed against HF directly, which is also where
  the fix came from: Whisper's rule is that the token after the task **must** be a timestamp, enforced in
  HF by a logits processor. The driver does the same thing with machinery it already had — one prefill,
  then `loom.argmax_row_range` over the timestamp block, which cannot answer with `<|notimestamps|>` or
  with a word. On the same clip that produced no timestamps at all, it then produces
  `<|0.00|> [Motor] <|10.00|>`, and the model has correctly noticed the real audio ends at 10 s.

  That token is fed back in *and* returned, which is why `PrefillDecodeLoop` grew a third field,
  `generated_prefix`: the model chose it, so it belongs in the loop's output even though it is also part
  of the prompt. The host needs no per-model constants for any of this — `<|0.00|>`'s id comes from
  `piece_to_id`, and seconds-per-timestamp is `(n_samples / sample_rate) / n_audio_ctx` off the file's
  own KVs, which is what `sample_rate` was added to `hparams()` for.

  **Two bugs worth recording, both of the same shape — a check that could not fail.**

  1. `TS_HI` shipped as **0**, because `vocab_size` lives on the model config and I read it from the
     *generation* config. The export-time cross-check that should have caught it was written as
     `if ts_lo and ts_hi and ...`, so a zero bound skipped its own verification and the driver silently
     stopped forcing timestamps. It is unconditional now: once a checkpoint has a `<|notimestamps|>` at
     all, the block's width is verified against the encoder's frame count or the export fails.
  2. The chunking loop special-cased the final window — advance a full stride once `avail < clip`,
     "because the rest is padding". But `avail < clip` only means the window is not *full*; the audio in
     it is real. A model that closes at 23 s of a 20..45 s window has transcribed three seconds and
     stopped, and the guard threw the other twenty-two away. Removing it turns one window into three
     (`20 → 23 → 26 → 45`) and the transcript covers the whole file contiguously. What replaces it is a
     one-second floor on the advance, which is a progress guarantee rather than a tuning knob.

  **Context conditioning across window cuts — DONE (2026-08-08), completing the long-form algorithm.**
  Timestamps fixed *where* a window is cut; this is what gives the model context *across* the cut. The
  previous window's text goes in front of `<|startoftranscript|>` behind a `<|startofprev|>` marker, so
  the model reads it as context and then starts its real prompt — at most `MAX_PREV` tokens of it
  (`n_text_ctx // 2 - 1`, half the context minus the marker, Whisper's own budget), keeping the most
  recent, since the words just before a window are the ones that predict its first.

  Two constants (`PREV_SOT`, `MAX_PREV`) and one driver branch; the host accumulates what each window
  generated and hands it over **raw**. Which of those ids count as *text* is the driver's question, not
  the host's, because the driver is what has the constants — it drops timestamps, `<|notimestamps|>` and
  eos, all of which are the model's decisions about the previous window rather than words it said.
  `--no-condition-on-previous` turns it off, on by default as in Whisper's own CLI and switchable for
  the same reason it is there: carried context is what makes a sentence survive a boundary, and it is
  also what lets a repetition loop persist across one, with no temperature fallback here to break out.

  **The check nearly could not fail, and fixing that is the interesting part.** The obvious fixture is
  "condition on this run's own output" — which is what the CLI actually does window to window, so it
  looked right. It changes nothing: the greedy path is already consistent with its own output, so HF's
  conditioned and unconditioned runs were **token for token identical**, and a driver that ignored
  `prev_tokens` entirely would have passed. The fixture now uses an ordinary sentence instead, which
  moves the output completely (` (sad music)` → ` The weather is cold and rainy in the north.`), and the
  test asserts up front that the two references differ. The context handed to the driver also has a
  timestamp, a `<|notimestamps|>` and an eos appended that HF's oracle never saw, so forgetting to
  filter fails too. Both halves verified: 52/52.

  **The CLI's remaining two flags landed here too**, on the same terms as `--language`:
  `--task transcribe|translate` (resolved by text, so a bogus one names the two Whisper has and says an
  English-only checkpoint has neither) and `--timestamps`, which omits `<|notimestamps|>` from the
  prompt so the model may emit timestamp markers. That flag found one more leak: with the token no
  longer forced, the model *chooses* it and emits `<|notimestamps|>` as its first output, which decoded
  literally into the transcript. Control tokens are now dropped before detokenizing — eos and
  `<|notimestamps|>`, both resolved by text — while timestamp markers are kept, since asking for them
  is the point of the flag. `--task translate` is visibly real: the same German audio transcribes as
  `(Lockere Musik)` and translates as `[Water sizzling]`.
- **P4.2 — GigaAM v3 — DONE (2026-08-09).** Graph is family 1 (already exported); the point was the
  *second loader* (`AutoModel.from_pretrained(..., trust_remote_code=True)` instead of
  `ASRModel.restore_from`), which is what proves P3.2's loader/template split.

  `tools/loom_mil_compiler/gigaam_export.py` (~200 lines, over half of it prose) + the transducer
  template it reuses. `v3_e2e_rnnt` exports as four phases — `encoder` (1455 nodes), `embed`,
  `pred_lstm_l0_fwd`, `joint` — in one 851 MiB GGUF, registered as
  `automatic-speech-recognition/gigaam-rnnt` against a real check on the checkpoint's own
  `cfg.model.cfg.model_class`, so `loom-export <dir> -o gigaam.gguf` needs no `--task`/`--model`.
  `test_e2e_gigaam_lua_driver.cpp` decodes `samples/jfk.wav` to the same **80 tokens** the checkpoint's
  own `RNNTGreedyDecoding` emits, and the exported encoder matches PyTorch's to 3.2e-4 over 275 frames.

  **The split the roadmap asked for, drawn where the evidence put it.** `parakeet_export.py` was the
  only transducer, so everything in it read as Parakeet's. Peeling GigaAM out of it found the boundary
  is not between the two models at all — it is between *how a checkpoint is loaded and where it keeps
  its modules* and *everything else*:

  * `transducer_export.py` (new) holds `BaseTransducerExportConfig`: the four phases, their shapes, the
    blank-id derivation, the joint/duration cross-check and the whole component list. Both leaves.
  * `parakeet_export.py` shrank to ~70 lines: `ASRModel.restore_from`, `decoder.prediction`/`joint`, and
    two recognizers.
  * `gigaam_export.py`: `AutoModel.from_pretrained(...).model`, `head.decoder`/`head.joint`, one
    recognizer — plus two workarounds the remote code needs, below.
  * `nemo_asr_export.py`'s `EncoderOutput`, `build_trace` and (renamed) `ASREncoderWrapper` turned out
    to be family-1's encoder template with no NeMo in them at all. GigaAM imports all three unchanged.
    **The entire remaining difference is two strings**: `GigaAM.forward` names its inputs
    `features`/`feature_lengths` where NeMo names them `input_signal`/`input_signal_length`, which is
    now `ASREncoderWrapper.input_names`. The module keeps its name because NeMo is still the only
    loader with a config of its own in it; a third loader is when the encoder half earns its own module.

  `transducer_driver/` (was `parakeet_driver/`) is shared verbatim, and Parakeet's own driver text
  changes by exactly the five header-comment lines that named it — the one intended difference in the
  re-export gate below. `RecurrentPhase` gained `number_layers`, because GigaAM's prediction network is
  **one** layer where Parakeet's is two, and the driver loop addresses the cells by index: the phase
  says "number the layers whatever the depth" rather than the Lua restating `topologies()`' rule.

  **The remote code needed two workarounds, as EXPORT-ROADMAP.md R4 predicted, and neither is the one
  it predicted.** No `.item()`, no Python control flow:

  1. The rotary positional table is a `persistent=False` buffer built lazily on the first forward.
     Built during the trace it becomes graph ops; built under `torch.inference_mode()` it becomes an
     *inference tensor* and the trace dies in autograd. `load_model` builds it eagerly, outside
     inference mode.
  2. **torchaudio's `MelSpectrogram` cannot be converted, for a reason that is not about mel.**
     `torchaudio.functional.spectrogram` reshapes the STFT result back to the input's batch shape, and
     reading `.shape` on a *complex* tensor emits a `complex_shape` op coremltools' own
     `lower_complex_dialect_ops` pass cannot lower. `_TraceableMelSpectrogram` is the same arithmetic
     without that reshape — the same shape of rewrite as `WhisperMelFrontend`, for the neighbouring
     reason. It is not asserted equivalent: every export runs both on a chirp and compares, and they
     agree **bit for bit**.

  **Two real exporter bugs, both found by this model and both older than it.** Neither is GigaAM-shaped;
  GigaAM is just the first checkpoint whose frontend and attention are spelled the way that reaches them.

  * **`square` and `clip` were missing from the shape walk's unary-passthrough set**
    (`exporter._UNARY_PASSTHROUGH_OPS`). `spec.abs()` on a complex tensor lowers to
    `sqrt(square(re) + square(im))`, so an `.abs()`-based magnitude puts a `square` immediately
    downstream of the STFT conv whose frame-count formula that walk exists to derive; and
    `torch.clamp` converts to MIL's `clip`, so the `clamp` entry never matched anything. NeMo spells the
    same magnitude `view_as_real(...).pow(2).sum(-1)` and calls neither, which is why this survived four
    ASR exports. The failure was **not an error**: the walk fell back to the bare root axis, the encoder
    frame count came out as the subsampling formula with the STFT's own `/160` simply missing, and the
    export succeeded. It failed at run time as a rotary-table VIEW asking for 44000 rows of a 10000-row
    constant.
  * **`gather_shape_value` assumed axis 0 is the batch axis.** `x.shape[0]` on a rank≥2 activation
    short-circuited to a hardcoded `1` — a claim about the model's *layout*, and GigaAM's
    `RotaryPositionMultiHeadAttention` is the counterexample: it transposes to `(T, B, H, D)` *before*
    applying the embedding, so `q.shape[0]` there is the sequence length. Read as a batch size it made
    the rotary cos/sin crop `pe[0:1]` — one position wide, which ggml then broadcasts over every frame,
    so **every position got position zero's rotation**. The rule is now about the derivation's
    provenance, the same distinction `scalar_expr_is_guess` draws: trust a real answer, fall back to the
    batch reading only when the walk had nothing better than the bare root axis. A genuine batch axis
    traces as a literal `1` and never reaches this code at all.

  **The second bug is the one worth remembering, because nothing structural could have caught it.**
  The graph built, ran, produced the right output shape, and decoded 71 of the first 80 tokens
  correctly — a plausible transcript that was simply wrong. Only dumping the exported encoder's own
  output and comparing it against PyTorch's found it (max abs diff 2.35 on a scale of 2.29), and only
  bisecting the encoder into prefixes — static vs. dynamic axis, mel / subsampling / attention /
  convolution — localized it to one slice in one op. **Token-level agreement is not a substitute for a
  tensor-level oracle**; a transducer decode is robust enough to absorb a badly wrong encoder and still
  look about right.

  **Not claimed:** GigaAM v3 ships five variants and only `e2e_rnnt` is on this machine. The recognizer
  requires `model_class == "rnnt"` rather than claiming every `model_type == "gigaam"` directory, so a
  `ctc`/`e2e_ctc` checkpoint fails detection with the candidate list instead of being exported by a path
  nothing here has run. Those are family 1's `Flattened` shape (`ASRNemoEncoderExportConfig` with
  `CTC_LOG_PROBS`) plus this loader — a small addition, and an untested one until a checkpoint exists.
- **P4.3 — composition template — DONE (2026-08-09).** The audio-encoder + projector + causal-LM family
  (`EXPORT-ROADMAP.md` R5's family 3), the largest group on the roadmap: ~19 converters, ~36 models.

  `tools/loom_mil_compiler/speech_lm_export.py` (the template) + `qwen3_asr_export.py` (~180 lines, the
  loader) + `speech_lm_driver/00_header.lua`. Four phases — `encoder` (793 nodes: mel frontend, chunked
  conv stem, window attention, projector), `embed`, `decoder` (2286 nodes, 28 cached `ATTENTION`
  blocks) and `lm_head` — in one 3.1 GiB GGUF, registered as `automatic-speech-recognition/qwen3-asr`,
  so `loom-export <dir> -o qwen3_asr.gguf` needs no `--task`/`--model`.

  **Acceptance: `test_e2e_qwen3_asr_mil_export.cpp` against HF's own `Qwen3ASRForConditionalGeneration`
  on `samples/jfk.wav`.**

  | check | result |
  |---|---|
  | `encoder` topology (mel frontend + projector included) vs HF's `get_audio_features` | `max_abs_diff = 2.2e-05` against a reference whose absmax is 0.128 — 1.7e-4 relative; `mean_abs_diff = 3.7e-07` |
  | the whole driver — encoder, segmented prompt, cached decode loop — vs HF's greedy sequence | **exact**: all 30 tokens, including stopping on `<|im_end|>` |
  | nothing in the test names a per-model C++ struct | `uses_kv_cache()`, `make_kv_cache(*model)`, `loom.samples_per_chunk`/`frames_per_chunk` all read from the artifact |
  | whisper-tiny, conformer-ctc-small, qwen3-0.6b-base re-exported after the shared-exporter fixes | byte-identical by sha256 |
  | the Python suite | 503 passed |

  **The finding that made this cheap: the prompt needs no concatenation anywhere.** "Inject audio
  embeddings into the prompt" reads as though something must build one `inputs_embeds` out of text
  embeddings and audio embeddings, which would need a backend-side concatenation of two retained
  tensors — an engine op that does not exist and one `OutputStore` has no shape for. It is not needed.
  Attention is causal and the decoder is KV-cached, so a call at `n_past = k` over `n` rows writes cells
  `[k, k+n)` and attends over `[0, k+n)`: feeding a prompt as N successive cached calls is *the same
  arithmetic* as feeding it concatenated. Measured against HF before any component was written —
  segmented and concatenated prefill agree to 2.3e-04 on hidden states whose absmax is 95.7, and pick
  the same first token. `PromptSegments` is that walk, and it is the whole of the "embedding-injection
  driver" the roadmap asked for. It stops one segment short and leaves the running `n_past` for
  `PrefillDecodeLoop.initial_n_past`, so the final text segment is the loop's own first iteration —
  exactly as a plain causal LM's prefill is.

  **`PrefillDecodeLoop` grew three fields and no new component**, the same outcome P4.1 reached:
  `embed_topology` (the step's tokens reach an `inputs_embeds`-traced decoder through the embedding
  graph, bound by `OutputRef`, so a token id is still all that crosses the Lua boundary),
  `head_topology`, and `initial_n_past`. All default to None/0, so every earlier family emits
  byte-identical driver text — which the re-export gate above checks rather than assumes.

  **The head is split off for a cost reason.** A family-3 prompt is dominated by audio rows — 143 of
  this one's 158 — and a head inside the decoder graph would project every one through a 151936-wide
  vocabulary that nothing reads (22 GFLOP). `lm_head` is its own phase, run only where a token is
  needed.

  **The encoder had to be rewritten, and the rewrite is bit-identical.** HF's `Qwen3ASREncoder.forward`
  is untraceable twice over: it packs valid frames with `valid_mask.flatten().nonzero()` (a
  data-dependent output shape) and runs attention per window via `torch.split(q, lengths.tolist())`
  (Python-level lengths that `torch.jit.trace` bakes in). Both dissolve into one observation —
  `get_audio_cu_seqlens` cuts full `block`-sized windows plus a shorter final one, and attention runs
  independently inside each, which *is* a block-diagonal additive mask. With every frame valid the
  packing step is the identity. Verified in torch before anything was exported: `max abs diff 0.000e+00`
  against HF's eager path (6.3e-05 against sdpa, which is that fused kernel's accumulation order).

  **The contract this imposes**: the waveform is a whole number of chunks, all valid — one chunk is
  `hop_length * n_window * 2` = 16000 samples, one second — so a host pads up to the next second. That
  is what makes "every frame valid" true, and it also makes the checkpoint's own feature extractor an
  exact oracle, since its mel-axis right-pad becomes a no-op on such a waveform. The cost is that up to
  one second of trailing silence becomes real audio embeddings the LM reads, where HF would have masked
  them out. Trimming them needs a way to feed a *prefix* of a retained tensor: **P4.3d** below, done,
  and shared with the second leaf. **P4.3e went further and removed the cost entirely** — the encoder
  takes the real sample count and masks the padding out of its own attention and features, so the rows
  it emits for a partial chunk are bit-identical to HF's.

  **Five bugs in shared exporter machinery, four of which fail silently.** None is Qwen3-ASR-shaped;
  it is the first checkpoint whose encoder is spelled the way that reaches them.

  1. **A conv's batch axis was hardcoded to `1`** in the shape walk — the same correction P4.2 made to
     `gather_shape_value`, one layer down. This encoder folds the CHUNK COUNT into the batch axis so the
     conv stem sees a fixed 100-frame window per chunk, and reading that as 1 did not fail: it made the
     post-stem sequence length 13 instead of 13 *per chunk*, surfacing hundreds of ops later as a mask
     whose two sides had different lengths. Conv preserves batch, so it now recurses; a genuine
     batch-of-1 still resolves to a literal 1 through the recursion, which is why all three re-exports
     are byte-identical.
  2. **`slice_by_index` was missing from the shape walk.** Whisper's frontend drops the final STFT frame
     the same way (`(stft.abs() ** 2)[..., :-1]`) and never needed it, because its clip is always 30 s
     and every dim downstream is a literal. Without it the walk returned `-1` — not an error anywhere,
     just a wrong number that reached a `POOL_1D` span as `-128`.
  3. **`reduce_max` rejected a dynamic global maximum.** `POOL_1D`'s `k0`/`s0` are already read through
     `resolve_attr_int(..., pc.symbols)` on the engine side, so a span known only once the axes are
     bound is as good as a literal one; this was exporter-side only. Whisper's clip is static and folded
     to a constant, and a variable-length mel frontend is what needs the symbolic form.
  4. **`MultiPhaseDriverBuilder.called_topologies` could not see through `WhenSet`.** It reads
     `TopologyName` off the declarations precisely so a new component is counted the day it declares its
     link — but an OPTIONAL topology field is `WhenSet(TopologyName())`, whose wrapper is not a
     `TopologyName`, so the field was checked and yet invisible. It reported `embed` and `lm_head` as
     "topologies no call site names" while the loop was calling both every step. Same failure P4.1 fixed
     for `PrefillDecodeLoop`, one wrapper deeper.
  5. **`driver_ir.Len` only accepted a local's name**, so `#inputs.waveform` had no spelling. It now
     takes either and delegates `reads()`, which is what keeps `validate()` honest: the string form
     reports the local, the expression form reports what the expression reads, rather than a dotted path
     that is not a symbol at all.

  **The two that cost the most were both silent successes, and both are worth remembering.**

  * **Forcing `attn_implementation="eager"` in the loader disabled the KV cache.** It looks harmless —
    `WindowedAudioEncoder` reimplements the tower's attention and never calls the tower's own — but it
    also reaches the LANGUAGE model, and `fuse_loom_attention` matches the sdpa shape. Under eager the
    decoder converted with **zero** fused `ATTENTION` nodes: no cache, no `n_kv` mask retyping, and all
    four phases still exported "successfully". It surfaced only at run time, as a mask sized 143×143
    where the cache wanted 143×152. Eager belongs on the *reference*, which is where it now is.
  * **`cache_position` was pruned from the decoder graph.** Handed `inputs_embeds` and an already-built
    4D mask, this model consumes it nowhere — it exists to derive position ids and to build a mask, and
    both jobs were done — so the trace dropped it, folding the rotary embedding to the eight positions
    the trace ran at. Every call at a different `n_past` would have rotated by the wrong angle. Caught
    by the exporter's own "supplies an input it does not declare" link; the fix is to pass the positions
    the model actually indexes with (`position_ids`).

  **Two ggml-shaped constraints the graph had to be written around**, both in the window mask:

  * ggml's elementwise ops repeat `b` into `a` and cannot do a **two-way broadcast**, so the natural
    `window.unsqueeze(1) == window.unsqueeze(0)` aborts in `ggml_sub` inside `equal`. `.expand(T, T)` on
    both operands is the obvious repair and does **not** work — MIL's `equal` broadcasts natively, so
    coremltools folds the expands away and re-emits the same broadcast. An **outer product against a
    vector of ones** survives, because a `matmul` is not a broadcast and nothing rewrites it into one.
  * That ones vector must be built by arithmetic (`window * 0 + 1`), not `torch.ones_like`, which
    converts to a MIL `fill` whose length the exporter resolves through the fill's own shape input — and
    it resolved to a *different expression for the same quantity*, so the two sides of the comparison
    disagreed about T.

  **The registry now skips a family whose optional dependency is absent**, loudly and only on
  `ImportError`. Families carry mutually incompatible optional dependencies — `nemo_toolkit` pins
  `transformers~=4.53`, and `qwen3_asr` first ships in **5.13** — so there is no single environment that
  imports all of them, and an eager import of every module made the registry only as usable as its least
  installable member: exporting Qwen3-ASR failed on `No module named 'kokoro'`, from a family the caller
  had not asked for. A failed *detection* now names the families that were not loaded, so "unrecognized"
  and "unloadable" stay distinguishable.

  **Not claimed:** only `Qwen3-ASR-0.6B` is on this machine and only the **`-hf`** repo works. Qwen ships
  the same weights twice, and the native `qwen-asr` layout (`thinker.*` weights, sub-configs under
  `thinker_config`) declares the identical `model_type` while transformers reads class defaults off it
  *without raising* — a 1024-wide/24-layer encoder instead of this checkpoint's 896/18. `detect()`
  requires `audio_config` and `text_config` at the top level for that reason, so the native layout fails
  detection with the candidate list rather than being exported as a plausible wrong model.

  **P4.3b — Voxtral-Mini-3B as a second leaf — DEFERRED, and the reason is this machine rather than the
  template.** It was the obvious second leaf (a Whisper encoder, a 4-frame-stack projector, a Llama LM)
  and it is the one member that does not fit here. Measured, so a machine with more memory can pick it
  up without re-deriving any of it:

  | measurement | value |
  |---|---|
  | parameters | 4.68B (`model.safetensors.index.json`'s own `total_size / 2`) |
  | F32 artifact it would write | **18.7 GB** |
  | fp32 checkpoint, resident | **18.7 GB** (peak RSS 20.6 GB; the excess is mmap'd checkpoint pages) |
  | fp16 checkpoint, resident | **9.36 GB** (peak RSS 18.8 GB — RSS double-counts the 9.35 GB of file pages, which are evictable) |
  | `torch.jit.trace` of the 3.6B LM phase | **free** — 18.8 GB before and after, at fp16 |
  | `ct.convert` on the LM phase | ~14.4 GB inferred from the parameter count, **not measured** — every probe died at or before this call |
  | this machine | 28 GB available, **no swap** |

  `MultiPhase.export` holds the torch model, every phase's converted MIL program *and* the merged
  weights simultaneously, so the peak is a sum rather than a max: ~35 GB against 28.

  **Two routes were measured and closed.**

  * **Half precision does not work, and not for a memory reason.** Tracing a Llama-3B at fp16 on CPU
    succeeds — the risk that half kernels would be missing did not materialise — but `ct.convert`
    refuses the resulting graph. With fp16 *inputs* it wants a `minimum_deployment_target >= iOS16`;
    supplying one, and separately declaring fp32 inputs over fp16 weights, both fail with an internal
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`. coremltools 9.0 with
    torch 2.8 will not take a half-precision traced graph through its torch frontend.
  * **The `quantize=` argument cannot help**, though the mechanism is real and proven
    (`test_e2e_qwen3_q8_0`, `test_e2e_lfm2_q8_0`). It is a *serialization*-time transform: the loop in
    `write_gguf` iterates `self.weights`, which by then holds every phase's f32 array, and it
    `astype(np.float32)`s each one before deciding whether to quantize. It shrinks the artifact —
    Voxtral would write ~9.6 GB at F16 or ~5 GB at Q8_0 — and changes peak memory not at all. Disk was
    never the binding constraint.

  **What would actually move it** is P5's per-phase process isolation below. Even then the floor is
  ~29 GB (the LM phase alone needs its 14.4 GB of fp32 weights and their 14.4 GB of MIL constants to
  coexist), so a bigger machine is the honest answer, not exporter surgery.

  **Granite Speech 4.0.1b is the second leaf instead**, and it was a better test of the template than
  Voxtral would have been: a conformer encoder over 160-bin features and a **Q-Former** projector
  (`num_queries = window_size // downsample_rate`), where Voxtral is a Whisper encoder and a linear
  stack — so it varies both halves the template claims to abstract, not one. Done, in P4.3c below.

  **P4.3c — Granite Speech 4.0.1b, family 3's second leaf — DONE (2026-08-09).** A 16-layer conformer
  over 160-bin features (Shaw relative attention in blocks of 200 frames), a **BLIP-2 Q-Former** that
  turns each 15-frame window into three query rows, and a 40-layer Granite causal LM (2048-wide, GQA
  16/4) — 2.31B parameters in one 8.75 GB artifact, registered as
  `automatic-speech-recognition/granite-speech`.

  `tools/loom_mil_compiler/granite_speech_export.py` (~330 lines, the loader plus the one rewritten
  encoder) + `tests/test_e2e_granite_speech_mil_export.cpp` +
  `tools/fixture_gen/reference_forward_granite_speech_mil.py`. The template gained **one hook with a
  default** (`mel_frontend`) and **one shared helper** (`split_prompt_on_audio`), and nothing else.

  **Acceptance: `test_e2e_granite_speech_mil_export.cpp` against HF's own
  `GraniteSpeechForConditionalGeneration` on `samples/jfk.wav`.**

  | check | result |
  |---|---|
  | the 160-bin features `LogMelFrontend` + the pair-stack produce, vs the checkpoint's own extractor | **`max_abs_diff = 0.000e+00`** — bit-identical, in torch, before anything was exported |
  | `ConformerQFormerEncoder` vs HF's `get_audio_features`, in torch | `max abs diff 5.8e-06` against a reference whose absmax is 0.999 |
  | the exported `encoder` topology vs the same tensor, through ggml | `max_abs_diff = 7.3e-04` against a reference whose absmax is 0.999 — 7.3e-4 relative; `mean_abs_diff = 3.7e-06` |
  | the whole driver — encoder, segmented prompt, cached decode loop — vs HF's greedy sequence | **exact**: all 24 tokens, stopping on `<\|end_of_text\|>`; `"and so my fellow americans ask not what your country can do for you ask what you can do for your country"` |
  | the driver, the phase list, the component list | **unchanged from Qwen3-ASR's** — no new component, no new field |
  | kokoro, whisper-small, lfm2-monolithic re-exported against a `git archive HEAD` baseline | byte-identical, every snapshot file |
  | qwen3-asr re-exported | differs in `model.driver_script` **and nothing else** — every topology and every tensor identical. It is the model that makes this gate able to fail, and the difference is exactly the comment header rewritten now that `speech_lm_driver/00_header.lua` serves two leaves |
  | the Python suite | 503 passed |

  **The claim this leaf exists to test, and it held.** The two leaves share *no* encoder and *no*
  projector: a Qwen3-Omni window-attention stack over one-second chunks of 128-bin mel and a two-layer
  linear projector against a conformer over twelve-second chunks of 160-bin features and a Q-Former.
  What they share is the log-mel frontend and `(samples_per_chunk, frames_per_chunk)` — the contract
  P4.3 wrote down before there was a second member to check it against — and that turned out to be
  enough for `PromptSegments` and `PrefillDecodeLoop` to serve both unmodified.

  **`audio_geometry()` returns `(192000, 120)`, and both numbers are forced rather than chosen.**
  Encoder frames must be a multiple of `context_size` (200, the conformer's blocks) **and** of
  `window_size` (15, the Q-Former's), so the chunk is `lcm(200, 15) = 600` encoder frames = 1200 mel
  frames = **192000 samples, 12 s**; the rows are `600/15 × 3 = 120`. Twelve seconds is coarse — a host
  pads up to it — and it is a property of this checkpoint's two block sizes, not of the family, which
  is exactly why the template publishes the pair rather than a duration. `phases()`' existing
  cross-check verified it against the traced encoder rather than trusting the arithmetic.

  **The mel frontend needed two constructor arguments and no new class.** Granite's extractor is a
  torchaudio `MelSpectrogram` (n_fft 512, **win_length 400**, hop 160, 80 mels) followed by
  `clip_(1e-10).log10_()`, a global `amax`, `maximum(x, mx - 8)` and `.div_(4).add_(1)` — and
  `x/4 + 1` *is* Whisper's `(x + 4)/4`. So `LogMelFrontend` needed `win_length` decoupled from `n_fft`
  and a filterbank read off a `MelScale` instead of an extractor attribute; `mel_frontend()` is that
  three-line override, and it has a default because every other family-3 extractor spells it Whisper's
  way. **The final-frame drop turned out to be Granite's too**: HF computes `L // hop + 1` frames and
  drops the last only when the count is odd, and under the chunk contract that count is `1200k + 1`,
  always odd — so Whisper's unconditional `[..., :-1]` removes the identical frame. That is why the
  features are bit-identical rather than merely close, and it is what makes the checkpoint's own
  extractor an exact oracle.

  **The encoder rewrite is of two `math.ceil`s, and two ggml shapes.** `num_blocks =
  math.ceil(num_features / context_size)` with a `remainder`-driven right-pad
  (`GraniteSpeechConformerAttention`) and `nblocks = math.ceil(seq_len / window_size)`
  (`GraniteSpeechEncoderProjector`) are Python-level and bake into the trace — the same wall Qwen3-ASR's
  encoder hit, and the same fix: require the chunk contract so every remainder is zero, and spell the
  block count `reshape(-1, block, ...)`. Two more things had to be written around the backend:

  * **Shaw's relative-position term is a 5-D einsum, and ggml tensors are 4-D.**
    `einsum("b m h c d, c r d -> b m h c r")` becomes a batched matmul over the *query position* axis,
    which contracts the same index with every operand 3-D or 4-D.
  * **The Q-Former's learnable query is `(1, num_queries, hidden)` broadcast against N windows** — a
    two-way broadcast ggml's elementwise ops cannot do. An **outer product against a ones column**
    materializes one copy per window, the same trick and the same reason as `WindowedAudioEncoder`'s
    window mask. The Q-Former's two attention masks are dropped entirely rather than built: HF makes
    them with `torch.ones(...)` over a dynamic batch and then `(1 - mask) * -10000`, which is
    identically zero, so building them would put a MIL `fill` over a dynamic extent in the graph for no
    arithmetic at all.

  **The logits are Granite's, not `lm_head`'s.** `GraniteForCausalLM.forward` ends with
  `logits / logits_scaling` (8.0) *outside* the head, so exporting `lm_head` alone drops it — and the
  driver would never notice, because it takes an argmax and a positive divisor cannot move one. A host
  that sampled would. `_ScaledLMHead` puts it back; the phase is 3 nodes instead of 2.

  **The export did not fit, and fixing that is P5.0's first item** — recorded there in full. Short
  version: peak RSS **30.4 GB**, OOM-killed in the `lm_head` phase, on a 33 GB machine; now **22.9 GB**.
  The finding was that the torch module and the converted MIL program are the *same* arrays, so
  releasing either alone frees nothing (0.2 GB, measured) and `MultiPhase.export` has to release the
  traced module, the wrapper and the program together.

  **Not claimed:** only `granite-speech-4.0.1b`. `detect()` requires `model_type == "granite_speech"`
  **and** `has_lora_adapter == false`, because Granite Speech ships variants whose language model is
  only correct with a LoRA adapter merged in for audio — this exporter traces base weights, so on such
  a checkpoint it would produce a model that runs and transcribes badly, and a candidate list is a
  better answer than that. The two other Granite Speech checkpoints on this machine declare
  `granite_speech_plus` and `granite_speech_nar`, neither of which transformers 4.57.6 has a module
  for at all; `-plus` has the identical block geometry and is the obvious third leaf the day its
  module ships.

  **P4.3d — a prompt segment can be a PREFIX of a retained tensor — DONE (2026-08-09).** Both encoders
  require the waveform to be a whole number of chunks with every frame valid — that is what makes HF's
  `nonzero()`-packing and its `ceil`-padding collapse into identities, which is what makes them
  traceable at all. So a host zero-pads up to the chunk boundary, and the encoder emits **real**
  embedding rows for that padding, which the LM read as speech. HF masks them out
  (`input_features_mask`); the driver had no way to, because the encoder's output deliberately never
  crosses into Lua (P4.0.12) and there was no spelling for "these rows of it".

  Three pieces, one per layer:

  1. **`{from = 'm', rows = N}`** in `set_tensor_from_output_ref` — `copy_row_prefix` builds a
     `ggml_view_2d` over the retained tensor and hands it to `ggml_backend_tensor_copy`, so the prefix
     reuses ggml's own copy strategy rather than restating its three branches. A view needs
     `ggml_backend_view_init` before it has a buffer at all; without that the copy dereferences null.
  2. **`OutputRef.rows`**, an *expression* rather than an int, with `reads()` delegating to it — the
     `Len` lesson from P4.3: a reference this class did not report is a read `validate()` never
     resolved.
  3. **`PromptSegments.audio_rows`** replaces the `(samples_per_chunk, audio_rows_per_chunk)` pair, and
     the component feeds the SAME local as the segment's `n_tokens` axis and as `rows`. That is what
     makes the two impossible to disagree about: a mismatch is a shape error the engine raises, where a
     segment that merely asked for the wrong length would be a silently short prompt.

  **The host says how long its real audio was, or says nothing.** `inputs.audio_samples` falls back to
  `#inputs.waveform`, so a caller that omits it gets exactly the behaviour this family had before — and
  `rows` then equals the whole retained tensor, which check 5b of
  `test_lua_bridge_retained_outputs.cpp` pins as bit-identical to the untrimmed copy.

  | check | result |
  |---|---|
  | a prefix of a retained output vs the same rows sliced out of the marshalled table | identical, at two row counts; and *not* equal to the same number of rows taken from the tail |
  | `rows` = everything retained vs no `rows` at all | identical |
  | the four ways to get it wrong (past the end, zero rows, disagreeing with the consumer's axis, non-2-D) | all raise, naming the real problem — 10 error cases now, was 7 |
  | Granite's rendered Lua row formula, evaluated in **luajit**, vs `_get_num_audio_features` | identical at all sixteen lengths |
  | the driver with `audio_samples` = the whole 192000-sample chunk | the untrimmed sequence, token for token |
  | the driver with `audio_samples` = 32000 (2 s of the 12 s chunk) | `"and so my fellow americans ask"` + eos — the first two seconds and nothing else, from 21 of the retained 120 rows |
  | kokoro, whisper-small, lfm2-monolithic re-exported | byte-identical, every snapshot file |
  | qwen3-asr re-exported | two lines of driver text, both in the audio segment: `inputs.audio_samples or #inputs.waveform`, and `rows = _seg_tokens` on the `{from = 'encoder'}` reference. Every topology and every tensor identical, and the row count is the same number it always computed — this leaf keeps the padded default |
  | the Python suite | 503 passed |

  **The row count is a per-checkpoint formula, and the template's default is the padded one.**
  Granite Speech's is `ceil((L // hop + 1) // 2 / window) * num_queries`, i.e.
  `_get_num_audio_features` line for line, **checked against the extractor at sixteen lengths**
  including both sides of a chunk boundary and lengths shorter than one hop. Qwen3-ASR's is
  deliberately **not** overridden: its count comes from three stride-2 convs over a final partial chunk
  whose valid mel-frame count the extractor derives from a mask with its own padding rules, and a
  closed form over the sample count disagreed with it at 5 of 12 probe lengths. Its chunk is one second
  (≤13 rows), so it keeps the padded default rather than a formula that is wrong at the edges. **P4.3e
  superseded both halves of that paragraph**: the padded default is gone (a leaf must now state its row
  count) and Qwen3-ASR's formula exists, because the mel-frame count turned out to be `floor(L / hop)`
  in both branches of the extractor's mask rescaling rather than the `ceil` it looks like.

  **What this does NOT fix, measured rather than assumed.** Trimming makes the LM stop reading the
  silence rows; it does not make the retained rows equal HF's. On a 13 s clip padded to 24 s, the first
  132 rows of our padded encoder differ from HF's own 132 rows by **max 3.9e-01** against a reference
  whose absmax is 0.992 — because HF's conformer crops back to the real frame count after every
  attention block and masks its final partial block, where ours runs the whole padded sequence
  unmasked. Greedy decoding was unmoved (all three of HF-unpadded, HF-padded and ours-trimmed produced
  the identical transcript), which is why this was worth doing as a prompt-level fix and separately
  from the encoder rewrite. That rewrite is **P4.3e**, and it is done — the same comparison is now
  4.8e-07.

  **P4.3e — the encoder's own padding handling — DONE (2026-08-10).** The other half of P4.3d, and the
  larger one numerically. The encoder phase now takes the caller's real sample count as a second input
  and keeps the padding out of every place it could reach an output row. Both leaves reproduce HF's own
  rows for a partially filled chunk, in torch:

  | | before | after | reference absmax |
  |---|---|---|---|
  | Granite Speech, 11 s in a 12 s chunk | 1.7e-01 | **4.8e-07** | 0.971 |
  | Qwen3-ASR, 10.625 s in eleven 1 s chunks | 1.0e-01 | **0.0e+00** (bit-identical) | 0.128 |

  **The input is the SAMPLE count, not any frame count, and that is what keeps the template one
  template.** `valid_samples` is a `(1,)` f32 the driver fills from the same local `audio_rows` reads;
  every leaf's encoder turns it into its own checkpoint's frames in graph. Declaring frames instead
  would have put Granite's `(L // hop + 1) // 2` and Qwen3-ASR's `floor(L / hop)` into the template,
  which is exactly the per-checkpoint arithmetic the leaf boundary exists to hold.

  **Three statements per encoder, and they are three different statements rather than one mask.** This
  is the finding: "mask the padding" is not a single operation, because a padded frame reaches a real
  row by three unrelated routes and each has its own correct place to be stopped.

  1. **Attention keys.** Masked with HF's own `-finfo.max` rather than `-inf` — HF's reason, plus one
     more: under the chunk contract the padding can span *whole* blocks, and a wholly padded block
     would softmax a row of `-inf` into NaN. A NaN cannot be masked back out downstream; a uniform
     finite row can, and is never read.
  2. **The convolution.** Granite's conformer conv module is zeroed **after the GLU**, not on the
     block's input: `up_conv` carries a bias, so a zeroed frame is not a zeroed activation. Post-GLU is
     the exact point where HF's sequence ends and `depth_conv`'s own `F.pad` supplies zeros, so this is
     not an approximation of HF's crop — it *is* the crop.
  3. **The features.** Qwen3-ASR's extractor right-pads `input_features` with literal **0.0**, and
     log-mel of silence is emphatically not zero. Its conv stem is 2-D over a whole chunk, so feeding
     it a zero-padded *waveform* changes the valid rows of the final chunk and not only the padded
     ones. Zeroing the mel frames past the real audio is what makes that chunk the extractor's.

  Nothing masks the query rows and nothing needs to: a padded row's output reaches no real row, and
  `audio_rows` stops the prompt before it.

  **The fourth statement is not in the encoder at all, and it is the difference between close and
  exact.** With 1–3 in place Granite was still 7.7e-04 out, in *one* mel frame — the one whose STFT
  window straddles the caller's real end. `torch.stft(center=True)` reflects the signal by `n_fft / 2`
  at each end, so the extractor computed that frame against a mirror of the audio where a host's zero
  padding puts silence. The new `WaveformValidLength` driver component writes `n_fft / 2` mirrored
  samples over the head of the padding — `w[valid + i] = w[valid - i]`, torch's reflect in 1-based Lua,
  bounded by what the caller actually padded — and the same comparison becomes 4.8e-07. It costs one
  bounded loop over a table the bridge built for this call, and it also binds the `_audio_samples`
  local the encoder input and the row count both read, so those two cannot disagree.

  **`audio_rows` is now required of a leaf, and Qwen3-ASR's exists.** P4.3d left it defaulting to the
  padded count on the reasoning that a leaf without a formula still read a little silence. That
  reasoning ended here: the rows past `valid_samples` are no longer a reading of silence, they are rows
  whose keys were all masked, so a leaf that cannot state its row count cannot use this template. The
  formula P4.3d could not write is four lines of `Qwen3ASRProcessor._get_audio_token_length` over
  `floor(L / hop)` valid mel frames — and `floor` is the whole of what that attempt got wrong: the
  extractor's mask is `attention_mask[:, ::hop]` with its last entry dropped when the sample count is
  not a multiple of the hop, which is `floor(L / hop)` in *both* branches rather than the `ceil` the
  rescaling looks like. It agrees with the processor at every probe length, where the earlier closed
  form disagreed at 5 of 12.

  **Two checks now run on every family-3 export**, both new, and the first of them found a real bug in
  the second's own implementation:

  * **padding invariance** — the same audio at the same declared length, once in a waveform that
    exactly fits and once padded by two further chunks, must produce identical rows. Noise, not
    silence, because on silence every frame is the same frame and a leak leaks nothing. This is what
    caught the missing mirror: Qwen3-ASR failed it at 9.0e-05 with masking that was otherwise complete.
  * **`audio_rows` against `audio_geometry`** — at a whole number of chunks there is no padding for the
    two to disagree about, so `audio_rows(k * samples_per_chunk)` must be `k * frames_per_chunk`. This
    needed `driver_ir.evaluate`, which evaluates the arithmetic subset of the IR in Python — so what is
    checked is the *same node* the driver renders, rather than a restatement of it. P4.3d checked
    Granite's formula once, by hand, in luajit; this runs on every export of every leaf.

  **The fixtures changed, and that is the substantive part of the test change.** Both generators used
  to hand HF the *padded* waveform, so both sides read the same trailing silence and neither fixture
  could see the padding question at all. They now run HF on the real audio — which is what HF's own
  pipeline does — and write the host's padded waveform, the real length, and the driver's own
  reflected copy of it separately. Qwen3-ASR's default input is trimmed to land mid-chunk, derived from
  the checkpoint's chunk rather than written down, because `samples/jfk.wav` is exactly 11.0 s and its
  chunk is one second; both tests now assert that the fixture does not fill its last chunk, so a
  generator that silently stopped exercising this would fail rather than pass quietly.

  | check | result |
  |---|---|
  | Granite `encoder` topology vs HF on 11 s of a 12 s chunk | max 6.1e-05, mean 2.1e-06, ref absmax 0.971 (was gated at 7.3e-04 against a *padded* oracle) |
  | Qwen3-ASR `encoder` topology vs HF on 10.5 s of eleven 1 s chunks | max 1.4e-05, mean 3.1e-07, ref absmax 0.128 |
  | both drivers vs HF's greedy tokens, `audio_samples` supplied | identical, 24 and 30 tokens |
  | `audio_samples` omitted vs `= #waveform` | identical, both leaves |
  | `audio_samples` = 2 s | a different, shorter transcript, both leaves |
  | `audio_rows` vs the checkpoint's own extractor | identical at every probe length, both leaves |
  | kokoro, whisper-small, lfm2-monolithic re-exported | byte-identical, every snapshot file |
  | qwen3-asr re-exported against the baseline | 2104 changed snapshot lines — the gate can fail |
  | the Python suite | 511 passed (8 new, for `IndexAssign` and `evaluate`) |

  **What is still not HF's, stated as a residual rather than left implicit.** The log-mel's `max - 8`
  clamp reads a global maximum over the whole padded clip, where the extractor's reads over the real
  one. Silence cannot raise a maximum and the mirrored samples are a copy of audio already in the clip,
  so on both leaves the two maxima agree exactly — measured, not argued. A clip whose loudest frame
  only exists across the mirror boundary would shift every bin sitting at the floor, which is the one
  place this family's exactness is empirical rather than structural.
- **P4.4 — KV cache in MIL-exported causal LMs — DONE, as P4.0.9.** Kept as a stub because
  `EXPORT-ROADMAP.md:129` points here. This row's full text — the measurement that `FuseLoomAttention`
  was the blocker, and a four-step plan — is superseded by [`KV-CACHE.md`](../KV-CACHE.md) and P4.0.9's
  entry, which record what was actually built and where the plan was wrong (step 2, `use_past` tracing,
  was **not needed**: once the SDPA subgraph is an `ATTENTION` node the engine supplies the past
  itself). Two remainders are live items of their own, P4.0.10 and P4.0.11.

  **What survives that P4.0.9 does not say.** `EXPORT-PREPARATION.md` decision 2 routes this gap to the
  cross-attention AR decode `Decomposition` in P4.1, and that is still the right home — but the
  prerequisite reading has inverted. It was "that decomposition cannot start here, the fusion pass comes
  first"; the pass now exists, `infer_with_past` exists, and `whisper_driver.lua` has been doing exactly
  this orchestration by hand since the Lua port (`KV-CACHE.md` §1.2). P4.1's decomposition is now the
  *reuse* of a solved shape, not a blocked one.
- **P4.6 — Supertonic takes real text: a padded text axis and a real `txt_msk` — DONE (2026-08-12, two
  commits).** Scheduled *before P5* by explicit user direction. Scoped 2026-08-11.

  **Why.** The export carries its own grapheme vocabulary (see the Models section) and
  `model.tokenize("hello world")` returns the same ids the real Python `TextVectorizer` does. It could
  not drive synthesis: `T_TEXT_FIXED = 10`, and `<en>` + the pipeline's inserted final period +
  `</en>` is *exactly 10 ids for the empty string*. `"hello world"` is 21. So the vocabulary in the
  file was good for encoding and inspection, and synthesis effectively still took ids directly.

  **`T_TEXT_FIXED` is 256 now, the driver pads to it, and `txt_msk` is a real input.** `"hello world"`
  synthesizes; so does a 155-character sentence (161 ids). The axis is still static — both reasons in
  `supertonic_export.py`'s docstring stand — so text past 256 still needs chunking, which this item did
  not open.

  ### What the plan got right, and the one thing it got wrong

  The scoping was right that **raising the constant alone is a trap**: this export faked the mask
  (`_ones_mask_from_ids`, all-ones regardless of content) while the real modules genuinely read it, so
  padding without threading the mask would attend to padding as text and recover a text length of `N`.
  Step 1 un-faked it and step 2 raised the constant, as planned.

  It was wrong about **why padding might not be inert**, and the difference is the whole of the work.
  The plan named three candidates from a read of the source and picked the DP text encoder's
  attention-weighted `sentence_token` pooling as "the most likely place for this to break". It does not
  break: `DPTextEncoder` builds a `full_mask` and forms `attn_mask = full_mask^T * full_mask` from it,
  so those scores *are* masked, and so is `VFTextCrossAttention` (`masked_fill(-inf)` plus
  `txt_len = txt_msk.sum()`), which is why `vfe`'s gate is the one that stays green even in the
  falsification below.

  The plan's second candidate was dismissed as the mechanism that ought to make padding *work* — "a
  conv at the last real position reads zeros either way — this is the mechanism that ought to make
  padding inert, and the reason the whole approach is plausible". **That is exactly backwards, and it is
  the bug.** `ConvNextBlock` pads with `mode="replicate"`, not with zeros. On a masked tensor the edge
  it replicates *is* a zero column; on an unpadded run it replicates the last REAL column. The two
  differ, and the reference implementation is the unpadded one — `TextVectorizer.tokenize` pads only to
  the longest string in its batch, and synthesis is a batch of one.

  **Measured in PyTorch before a line of export code was written** (ten ids, N=256): `txt_emb` moved by
  1.77 max-abs against a tensor whose own max is 1.82 — 97% wrong, not a near-miss — and the predicted
  duration by 0.17%. So the plan's step-2 gate ("require the same waveform to 1e-3") would have failed,
  and its stated fallback (regenerate the reference from the padded Python) would have *hidden* the
  problem rather than found it: it would have shipped a model that disagrees with the reference
  implementation for every text, and made that disagreement the new definition of correct.

  ### `_edge_fill`: the fix, and why it needs no new primitive

  Fill the padded tail with a copy of the last real column, before each block's replicate pad. Then
  every real position's conv window is byte-for-byte what the unpadded run sees, and everything else in
  the block is position-local. The last real index is data-dependent, so the obvious implementation is a
  gather — but it does not need one:

      edge = msk - shift_left(msk)      # one-hot at the last real column
      last = (x * edge).sum(dim=2)      # (B, C, 1)
      x    = x * msk + last * (1 - msk)

  A multiply and a reduction, both of which the exporter already lowers — **no new binding, no new
  primitive, no engine change at all**, which is what the scoping predicted for the mask and turns out
  to hold for the fix too. At an all-ones mask `1 - msk` is zero, so it is *exactly* the identity, which
  is why it costs the `T_TEXT = 10` references nothing (see step 1's gate below).

  It is applied by patching `ConvNextBlock.forward` **per instance** on the two text encoders' blocks
  only, via `types.MethodType`, with the mask reaching it through a holder object. Per-instance rather
  than on the class because the VectorFieldEstimator's own blocks take a real `msk` over the latent
  axis, which is dynamic and never padded; and a patched `forward` rather than a wrapper module because
  a wrapper changes state-dict paths, and those paths are the exported tensor names. A holder rather
  than an argument because `DPTextEncoder`/`TTLTextPreEncoder` call their blocks as `block(x)` and mask
  outside — there is no argument to thread, and the alternative was copying both encoders' `forward`
  into this repo, where every future divergence would be silent.

  ### Where `n` comes from: neither of the two options the plan listed

  The plan resolved this per [[feedback_optional_arg_then_autodetect]] to "an explicit input, or
  inferred from a pad sentinel", with id 162 named as the sentinel. Both assume the *host* pads. It is
  simpler for the **driver** to pad: it already receives `txt_ids` as a Lua table, so `#inputs.txt_ids`
  *is* `n`, with nothing to infer and nothing to declare. The host API did not change and got strictly
  more permissive — any count up to `txt_len` instead of exactly `txt_len`.

  The sentinel survives as `PAD_ID = 162` anyway, but only as documentation: **which id pads was
  measured not to matter at all.** `x = x * txt_msk` zeroes every padded embedding before anything reads
  it, and ids 0, 1 and 162 give bit-identical `txt_emb` and duration. 162 is used because it is the
  vocabulary's one unused row (`n_tokens()` is 162 against an `nn.Embedding(163)`), so a dump of the
  padded ids reads unambiguously as padding.

  ### Step 1 — `txt_msk` as a real traced input, `T_TEXT_FIXED` still 10

  The three wrappers stop synthesizing it and take it as a forward argument, which the real modules
  already accept, so this un-fakes an input rather than inventing one. `lat_msk` stays synthesized.
  `vfe` needed it for a second reason: its mask was derived from `txt_emb`, whose padded columns are
  zero, so it would have read all-ones no matter how much padding it was handed.

  **Gate: the numbers, against every reference that already existed.** All five comparisons returned the
  exact values they returned before — duration 1.19209e-07, `txt_emb` 2.14577e-06, `v` 5.13345e-06,
  decoder 1.02073e-06, and 3.45102e-06 against the frozen `supertonic_driver_waveform_F1.npy`. A step
  that changed the interface and provably not the numbers. Adding `_edge_fill` while still at `T = 10`
  reproduced all five again, unchanged to the last digit — which isolated "does the fill lower
  correctly" from "does padding work" before either was in question.

  ### Step 2 — the axis at 256, and what the gates say

  | gate | T=10 (before) | T=256, 10 real ids | T=256, no `_edge_fill` |
  |---|---|---|---|
  | `dp` duration | 1.19209e-07 | **0** | 0.0291 (1.8% short) ❌ |
  | `ttl_text` `txt_emb` | 2.14577e-06 | 2.0843e-06 | 1.00379 ❌ |
  | `vfe` velocity | 5.13345e-06 | 5.66989e-06 | 5.66989e-06 ✓ |
  | frozen F1 waveform | 3.45102e-06 | 4.70784e-06 | **0.379652** ❌ |

  The frozen fixture became the test of the new capability rather than a casualty of it, exactly as
  scoped: the same ten ids padded to 256 reproduce the waveform the retired C++ driver left behind, to
  4.7e-06 against a 1e-3 target. **And the gate can fail** — the last column is a real export with the
  fill removed, and it is worth reading twice: a 0.38 max-abs error on the waveform is audibly wrong
  audio, which is precisely the "longer, more usable-looking input and quietly wrong audio" the scoping
  warned about. `vfe` passing there is not a weakness of the falsification; it is the prediction that
  the VFE's masking was already correct, confirmed.

  ### A new gate, because ten ids is the empty string

  `test_e2e_supertonic_mil_real_text.cpp` runs a real 161-id sentence — the only shape where the
  real/padding boundary sits in the *middle* of the axis rather than at its very end — against the real
  Python pipeline's own unpadded answer for that sentence (`reference_forward_supertonic_mil_extra.py`
  grew a third case). Four checks, in the order a failure is worth reading:

  1. **The ids**, from `loom::SupertonicTextVectorizer` reading the GGUF's own vocabulary against what
     the real `TextVectorizer` produced. First because every number after it is meaningless if the two
     tokenized different text — and it is a real cross-check of two independent implementations at a
     string neither was tuned on. **Identical.**
  2. **The duration**, relative rather than absolute, because this sentence is ~10.6 s where the ten-id
     fixtures are ~1.6 s. **Exact to 0** (10.581952 s).
  3. **`txt_emb`** over the real columns, plus an exact zero over the padded ones. **4.55976e-06**, tail
     exactly 0.
  4. **The driver**, end to end. Checks 1–3 build the graphs directly and pad by hand, which
     re-implements `01_text_inputs.lua` rather than testing it, and every other end-to-end test hands
     `infer` exactly ten ids — so nothing else exercises the branch where the driver has real padding to
     do. There is no waveform oracle (the CFM noise is the driver's own), so what is checked is the
     sample COUNT, which the reference duration fixes exactly through the driver's own `get_latent_mask`
     arithmetic, plus a peak-amplitude floor so that silence cannot pass. **466944 samples (10.59 s of
     real audio from real text), peak 0.175, in 4.3 s wall.**
  5. **The ceiling is a ceiling.** `txt_len + 1` ids must be REFUSED — by name, from the driver, not by
     a shape mismatch deep in the engine and above all not by silently dropping the tail, which would
     produce perfectly plausible audio of the wrong words. `LoomLuaBridge::call: error in 'infer':
     ...:74: supertonic: 257 txt_ids exceeds this export's T_TEXT of 256`. The branch was written in
     step 2 and unexercised until this check; it is exactly the kind of guard that is worth nothing
     until something proves it is reachable.

  Without the fill, three of the four fail: duration 1% short, `txt_emb` off by 0.285, and the driver
  emits 463872 samples — 151 latent frames where the reference implies 152. The tokenizer check and the
  peak floor stay green, correctly: neither is about padding.

  The fill was also checked in PyTorch across four real texts (12, 21, 53 and 161 ids at N=256): every
  duration matches to ≤4.8e-07 and every `txt_emb` to ≤5.9e-05, which is fp32 reduction-order noise from
  the wider matmuls rather than a residual mechanism.

  ### Why 256 and not 512 — measured, not chosen

  The axis is static, so its cost is paid on **every** synthesis regardless of how long the real text
  is. Full 1.6 s synthesis, this machine: **T=10 → 1.65 s / 275 MB, T=256 → 2.14 s / 291 MB, T=512 →
  2.72 s / 330 MB.** 512 is 27% more wall clock and 39 MB for capacity the overwhelming majority of
  calls would not touch, on an engine whose target is edge devices. Accuracy does not enter into it —
  the `ttl_text` gate reports 2.0843e-06 at both. 256 ids is roughly 245 characters after the wrap:
  `"Hi."` is 12 ids, `"hello world"` 21, a 44-character sentence 53, a 155-character one 161.

  ### Blast radius, as executed

  *loom-exporter:* `supertonic_export.py` (three wrappers, three `mil_inputs` lists, the constant,
  `PAD_ID`, `_edge_fill`/`_patch_text_convnext`, `driver_components`, `hparams`),
  `supertonic_driver/01_text_inputs.lua` (new) and `00_header.lua`,
  `reference_forward_supertonic_mil_extra.py` (the real-text case), `tools/build_model_cards.py` (the
  "Known limitations" section is now a ceiling rather than a refusal, and the snippet no longer warns
  about length), one `tests/ci` topology declaration. *loom.cpp:* the four
  `test_e2e_supertonic_mil_*.cpp` gates, the new `real_text` one and its `CMakeLists.txt` entry,
  `tests/support/tts_driver_inputs.h`'s comment (its `txt_len_fixed = 10` still describes the *bespoke*
  conversion and is deliberately not a second copy of the MIL number), and `tools/loom_cli`'s hint,
  which told users to "pad or shorten" anything that was not exactly `txt_len`. **No engine source
  change, as predicted** — no new binding, no new primitive. *loom-py:* none.

  Nothing shared changed, so no other model in the sweep can have moved; the diff is confined to
  `supertonic_export.py`, its driver fragments, its fixture generator, the model-card catalogue and
  supertonic's own tests.

  ### Two follow-ups, settled 2026-08-12 by explicit user direction

  **(a) Chunking long text is OUT OF SCOPE for the engine and the driver — closed, not deferred.**
  Unlike ASR's *output* chunking (family 3's segmented prefill, P4.3/P4.3d/P4.3e), which is the model's
  own contract and therefore has to live in the driver, splitting an over-long utterance is
  preprocessing: it is a decision about where sentences may be broken, which is a text-domain question
  the engine has no business answering, and the pieces are then just ordinary calls. It does not belong
  behind `infer`. The ceiling is reported as `loom.txt_len` and enforced by a named error (check 5
  above), which is the whole of what the engine owes a caller here.

  **(b) A BUCKETED text axis — DONE (2026-08-12, P4.6a), built before P5 by explicit user direction.**
  The ceiling and the per-call cost used to be the same number, which is the only reason 256 was a
  compromise at all: a static axis is paid for on every call, so "long enough to be useful" and "cheap
  enough to always pay" had to be traded against each other. Each text-touching topology is now traced
  at every width in `TEXT_BUCKETS = (32, 64, 128, 256, 512)`, and the driver runs the smallest that
  fits `#inputs.txt_ids`.

  **The result is the trade removed rather than re-struck: the ceiling DOUBLED to 512 and an ordinary
  call got faster than it was at 256.** Full 1.6 s synthesis on this machine, same test, same host:

  | export | ceiling | 10-id synthesis | peak RSS |
  |---|---|---|---|
  | fixed 10 (pre-P4.6) | 10 — unusable | 1.65 s | 275 MB |
  | fixed 256 (P4.6) | 256 | 2.14 s | 291 MB |
  | fixed 512 | 512 | 2.72 s | 330 MB |
  | **bucketed** | **512** | **1.96 s** | **289 MB** |

  The bucketed row includes loading a file with sixteen topologies instead of four, so it is a fair
  comparison and not a favourable one. Against the fixed-512 export it is the same ceiling for 28% less
  wall clock; against the shipped fixed-256 one it is twice the ceiling and still faster.

  **What it cost, against what was predicted when this was scoped:**
  * **Weights: nothing, as predicted.** The GGUF writer dedups by content hash (dtype + shape + bytes),
    so every bucket's identical weights alias despite differing namespaced names — 2333 aliased tensors.
    The file went 263.0 → 268.2 MB, **+5.2 MB for five buckets**, all of it the genuinely T-dependent
    constant-folded `(1, 2T-1, D)` relative-position tables.
  * **Metadata: +0.94 MB** (263 KB → 1204 KB), against a predicted ~+0.7 MB for four buckets. Five.
  * **Export time: 36 s → 1 m 52 s.** Fifteen traces instead of three. This is the real cost and it is
    paid once, by whoever exports.
  * **No engine source change**, again. The host side needed one change and it is a simplification:
    register what `topology_names()` reports instead of naming four topologies by hand, which is what a
    host should have been doing anyway.

  **Two shared-machinery generalizations, and neither is a Supertonic special case.** The scoping said
  "no new machinery", which was right about the ENGINE and wrong about the exporter — `ComputedCall`
  covers a computed name in *hand-written* Lua, and both of Supertonic's computed call sites are
  synthesized IR, where dropping to text would have cost the output-arity and define-before-read checks
  that are the reason the driver is IR at all. So:
  * `SubgraphCallComponent` gained `topology_expr` + `variants`, and `driver_ir.SubgraphCall` gained
    `module_expr`. `topology` stays a real name — the canonical member — so every check the component
    already had keeps running against it, and `sub_specs()` extends the declared-input check to the
    rest through the same `EstimatorSpec` a hand-written call site uses.
  * `FlowMatchingSpec` gained `estimator_variants`, so the generated sampler takes its estimator as an
    argument. `vfe` had to be bucketed — it runs once per CFM step, so leaving it at the ceiling would
    have given back most of what the other two save — and that is the template's business, not a
    Supertonic patch. Matcha declares no variants and its generated Lua is byte-identical (verified by
    rendering it: `local function sample_decoder(length, n_elems, n_steps, step_inputs)` and
    `loom.run_subgraph("decoder", ...)`, unchanged).

  `called_topologies` needed widening too, and it reported the problem itself: twelve of the sixteen
  topologies came out as "no call site names" because it recognized `RunSubgraphCall` by class rather
  than reading the topology field off whatever a sub-spec declares. The same lesson `_names_a_topology`
  records one wrapper deeper, so the fix matches it — read the declaration, not the class.

  **Why this ladder.** `<lang>` plus the inserted final period costs 10 ids flat, so 32 is ~22 real
  characters ("Hi." is 12 ids, "hello world" 21), 128 a sentence (a 44-character one is 53), 512 a
  short paragraph (~490 characters). Doubling holds worst-case waste just under half the axis while
  keeping the ladder short enough that export time stays in minutes.

  **Gates.** The existing ones, and they got sharper for free rather than being rewritten: the ten-id
  fixtures now land in **bucket 32** and the real-text sentence in **bucket 256**, so two different
  exported widths are exercised by tests that already existed. `tests/support/supertonic_buckets.h`
  DISCOVERS the widths from `topology_names()` rather than listing them, which is what keeps those
  tests checking the export instead of agreeing with themselves. Results: duration exact, `txt_emb`
  2.74e-06 at bucket 32 and 4.56e-06 at 256, padded tails exactly zero, and the frozen F1 waveform —
  ground truth from a retired C++ driver that predates padding *and* buckets — still holds at
  5.065e-06 against a 1e-3 target.

  Two failures found by the gates while building this, both worth recording because both were mine:
  the ceiling check in `real_text` used the sentence's own bucket rather than the largest one, so it
  asked for 257 ids and got audio (correctly — 257 fits in 512); and `test_driver_components` was
  updated to build its fake topologies from `TEXT_BUCKETS` after the first attempt spelled the names
  out by hand, which would have made it agree with itself.

  **The damper stands.** Text length and `t_lat` correlate, because the duration is predicted from the
  text, so the worst padding waste is still on short utterances where total latency is already lowest.
  Bucketing is why that waste is now bounded by the *next rung down* rather than by the ceiling.

  ### P4.6b — the voice style is optional, and one travels in the file — DONE (2026-08-12)

  **The report was "we can't use different styles"; the finding was that the ceiling was lower than
  that.** `style_ttl`/`style_dp` have always been `infer` inputs, the driver has always passed them
  through, and loom-py forwards any named array — so a *different* style always worked, and the gate
  test has been loading `F1.json` and using it since the file existed. What did not work was using
  **any** style: a published GGUF carried none, so every caller needed the upstream checkpoint repo to
  get one, and the model card's own snippet said `style_ttl=style_ttl` referencing a variable it never
  defined. Worth recording because the fix that the literal request implies — "expose them as input
  arguments" — was already done, and doing it again would have changed nothing.

  So: **both style inputs are OPTIONAL, and the checkpoint's own F1 embeddings ship in the GGUF as the
  default.** Resolution order is `[[feedback_optional_arg_then_autodetect]]` with the middle rung
  absent — an explicit argument, else the default; there is nothing to autodetect about a voice.

  **F1 specifically, and not as a coin flip:** it is the style
  `legacy_driver_reference/supertonic_driver_waveform_F1.npy` was recorded with, so "call `infer` with
  no style at all" is gated by ground truth that already existed rather than by a fixture recording the
  decision itself.

  **A third kind of thing a GGUF can carry, and the first time anything needed it.** The two tensors
  are read by the DRIVER (`loom.get_weight`), not by any topology node — so `_prune_dead_weights`
  deletes them by construction, since its rule is "no node names it as an input". `write_gguf` gained
  `driver_weights`, merged in *after* pruning rather than exempted from it: the pruner's own job is
  catching MIL's incidental attribute constants, and weakening its rule would have cost more than
  ordering around it. They cannot be `ExportConstants` — 12928 float literals would swamp a driver
  script that is 6 KB and meant to be read out of the GGUF by a person. Cost: 51.7 KB.

  `loom.get_weight`'s first argument is any *registered module*, and every module shares one
  `GgufModel`, so the driver reads through `"decoder"` — the one topology whose name carries no text
  bucket and is therefore spelled the same at every text length.

  **Gates, in the driver test, all three needed:**
  1. **Omitting the style reproduces passing F1 bit-for-bit** — `max_abs_diff` exactly **0**, not
     "close": the driver either reached the same numbers or it did not, and there is no arithmetic in
     between to blur it.
  2. **The frozen F1 waveform still holds**, 5.065e-06.
  3. **A different voice produces different audio.** Without this the first two are *both* satisfied by
     a driver that ignores `inputs.style_*` and always uses the default — passing F1 would match the F1
     fixture for the wrong reason. M1 gives **76800 samples against F1's 70656** (a different voice
     predicts a different duration) and 0.268 max-abs where they overlap.

  Skipped rather than failed for a non-F1 style or a GGUF with no default, so the test's meaning does
  not depend on which fixture the runner points at.

  **Still out of scope, and a genuinely bigger thing: deriving a style from your own audio.**
  `SpeechGenerator.encode_voice_style` runs mel → `SpeechEncoder` → `lat_compressor` → the two style
  encoders, i.e. three more real modules than this export traces, plus a second entry point. Selecting
  among existing voices — which is what was actually asked for — needs none of it. The model card now
  says which of the two it can do.

