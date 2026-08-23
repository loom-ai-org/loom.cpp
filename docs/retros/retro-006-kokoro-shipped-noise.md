---
type: retro
date: 2026-08-16
domain: model-coverage
tags: [tts, verification, oracle, declarations, released-defect]
---

# Retro-006: Kokoro Shipped Speaking Noise, and Four Declarations Were Wrong at Once

## The Issue

Reported against the published 1.0.0-rc4 as "Kokoro generates non-intelligible phrase". The reporter had
already tried `sample_rate=` at 16000/22050/24000/44100/48000 — the rate *was* a real defect, and it was
the **last** of four, which is why trying rates could not help.

## Root Cause Analysis

**The engine was never wrong.** Against upstream `KModel.forward_with_tokens` on the same tokens and
the same `ref_s`, the driver returned the same samples at cosine **0.996**. Every defect was in what
the file *declared* about itself, and four of them stacked.

## Resolution & Lesson Learned

* **Actionable takeaway 1 — cosine similarity against PyTorch is not a shipping gate.** 0.996 against
  the reference and unintelligible output are compatible states. Transcribe the audio through an ASR
  model and check the peak. This is the standing "ASR oracle for TTS" rule.
* **Actionable takeaway 2 — declarations are as load-bearing as weights, and are not covered by a
  numeric gate.** A byte-identity sweep proves the tensors did not move; it says nothing about whether
  the metadata describing them is right.
* **Actionable takeaway 3 — when several things are wrong at once, a user's bisection over one
  parameter cannot converge.** Diff the declaration against sibling families in the same export.

---

## Full record (verbatim from the ledger)


Reported against the published 1.0.0-rc4 as "Kokoro generates non-intelligible phrase", with the
reporter having already tried `sample_rate=` at 16000/22050/24000/44100/48000. The rate was a real
defect and it was the *last* of four, which is why trying rates could not help: nothing about the audio
was a playback-speed problem.

**The engine was never wrong.** Against upstream `KModel.forward_with_tokens` on the same tokens and the
same `ref_s`, the driver returns the same 35400 samples at cosine **0.996** and peak 0.3301 vs PyTorch's
0.3298 — the residual is SineGen's own noise draw. Every defect was in what the export DECLARED. That is
the load-bearing part: four independent facts about one model, each a constant, none of them reachable
by any test that ran.

1. **The default voice was random noise.** `TTSKokoroExportConfig._default_voice()` built `ref_s` from
   the checkpoint's `ref/ref_decoder_core_style.npy` + `ref_duration_style.npy`. Those are written by
   `reference_forward_kokoro_decoder_core.py` and `..._duration_predictor.py`, both of which build style
   as `rng.normal(scale=0.3, size=(style_dim,))`. Measurable rather than arguable: a real `af_heart` row
   has decoder-half std 0.14–0.20, the baked vector's is 0.301, and its L2 distance to the *nearest* of
   the pack's 510 rows is 3.6 against a typical row-to-row distance of 0.21. Its waveform leaves [-1, 1]
   entirely — peak ~300 where a real voice gives ~0.33 — and whisper-small transcribes it `[MUSIC
   PLAYING]`. The `backend_kwargs` docstring asserted this was "real style data (verified against the
   synthetic pattern the gate uses, which it is not)"; the check was aimed at the *gate's* synthetic
   pattern and the tensors came from a *different* one.
2. **`text.frontend` fell through to `"vocab"`.** So `Text2Speech._resolve_ids` skipped G2P entirely and
   encoded the caller's English SPELLING against the phoneme table — every letter that is also an IPA
   symbol got an id and the model read the spelling aloud. Nearly invisible on "hello world", whose
   letters and phonemes almost coincide; total on anything else ("the quick brown fox jumps over the
   lazy dog" → *"Take whichg brompho ymps avertelazidoh"*). VITS, Matcha and StyleTTS2 each declare
   `"phonemes"`. **Kokoro was the fourth phoneme model and the only one that never did.**
3. **Nobody applied the `[0, *ids, 0]` wrap.** `phoneme_table()` declared `bos: -1, eos: -1`, reasoning
   that "Kokoro's driver wraps the ids itself (its own header says so)". The header says the OPPOSITE —
   "caller wraps with leading/trailing 0". Each side documented the wrap as the other side's job and
   neither performed it. Costs the final phoneme its duration: *"Hello, worth"*.
4. **No `sample_rate`.** Kokoro is 24 kHz; undeclared, `loom-py` warns and guesses 16000.

**Fixes.** The default voice is now an upstream pack, `voices/af_heart.pt` beside the checkpoint, baked
WHOLE (510 × 256 f32, 522 KB) rather than as one row — upstream's selection is `pack[len(ps)-1]`, so the
driver reproduces that indexing instead of the export guessing a phoneme count. `contract()` declares
`text.frontend = "phonemes"`, `text.phoneme_alphabet = "ipa"` and `sample_rate = 24000`. The phoneme
table declares `bos: 0, eos: 0`, putting the wrap in the vocabulary — `tokenize` is where phonemes
become this model's ids, and a caller who assembles ids himself already has the driver header telling
him to wrap, so doing it in the driver would double-wrap exactly him. Both headers now say the same
thing. Separately in `loom-py`, `phonemes=` accepts the STRING every G2P returns and not only ids; it
was `[int(p) for p in phonemes]`, byte for byte the `tokens=` branch, so the string form died on
`invalid literal for int(): 'h'` and the two parameters were one parameter under two names.

**Verified with an ASR oracle rather than by ear**, which is the only reason "how bad is it" had an
answer at each step. whisper-small on the re-export: text door → *"The quick brown fox jumps over the
lazy dog."*, phoneme door → same, ids door → *"Hello, world."*, every waveform inside [-1, 1] and no
rate warning. Residual errors on rarer words ("Loom Rentsiguff" for "loom runs a gguf") are the bundled
`orthography2ipa` G2P, not the model: misaki's phonemes for the same sentence come back clean. That is
Task #79's territory — o2ipa emits no stress marks at all and standard IPA diphthongs (`aʊ`, `oʊ`, `dʒ`)
where Kokoro was trained on misaki's compressed symbols (`W`, `O`, `ʤ`), so **the C++ port cannot be a
straight transduction port if Kokoro is to sound its best; it needs misaki's symbol convention too.**

**Why nothing caught it, and what does now.** The export sweep diffs each GGUF against its own recorded
baseline, so a value wrong since the first snapshot is exactly what it certifies as unchanged. Nothing
tested a default voice at all. Supertonic got this right only by accident of ordering: its default is a
real `assets/voice_styles/F1.json` gated by a frozen end-to-end waveform recorded with it, so "call
`infer` with no style" cannot silently become noise there. `loom-exporter/tests/ci/test_tts_text_door.py`
is the standing rule, driven by the registry so a new TTS family cannot skip it, and it was made to fail
on purpose before being kept.

**No release is needed to fix this, and that was verified rather than assumed.** Every defect above is
DATA IN THE FILE, and rc4's engine already read every KV involved — the phoneme vocabulary's
`bos_id`/`eos_id`, the contract's `text.frontend`/`sample_rate`, and `loom.get_weight` for a driver
weight (which Supertonic's default voice already used). Against the published wheel installed fresh
from PyPI into a clean venv, with no local source on the path:

```
rc4 wheel + PUBLISHED gguf : peak 7697.901 -> ' [BLANK_AUDIO]'
rc4 wheel + RE-EXPORT gguf : peak    0.326 -> ' The quick brown fox jumps over the lazy dog.'
```

So the fix ships as a re-export and an HF re-upload. The one thing that does need a version bump is
loom-py's `phonemes=` string acceptance, which is ergonomics rather than correctness — `phonemes=[ids]`
and `tokens=[ids]` both work on rc4 — so it waits for the next release rather than forcing one. **This
is what the "engine hardcodes no model" design is for, and it is the first time it has paid out on a
real defect:** a model this badly broken was fixed without touching the runtime.

**The model cards carry two things they did not.** `tools/build_model_cards.py` now records a
`sample_rate` per TTS entry and renders it into the usage snippet as an argument, with the note that
the checkpoint does not carry it and the value has to come from the model's documentation — the five
values are Kokoro 24000, Matcha 22050, Supertonic 44100, VITS 22050 (the Piper voice JSON's
`audio.sample_rate`, per-voice rather than per-architecture) and StyleTTS2 24000 (`preprocess_params.sr`
in its `config.yml`, NOT the `slm.sr: 16000` beside it, which is the training discriminator's). Kokoro
also gains a "Choosing a voice" section like Supertonic's: the 54 upstream packs already ship in the HF
repo under `voices/`, and the snippet — run verbatim against the live repo before being written down —
shows the one thing that is not guessable, that a Kokoro voice is a PACK indexed by phoneme count
(`pack[len(ps)-1]`) rather than a single vector.

**Open, and deliberately not folded in:** Matcha, StyleTTS2 and VITS still declare no `sample_rate` —
the same defect 4, on three models nobody has reported yet. They are named in that test's
`NO_SAMPLE_RATE_YET`, whose companion test fails if a family gains a rate and is not removed, so the
exemption cannot outlive the gap. Each needs its rate off its own checkpoint (VITS's is genuinely
per-voice, in the Piper voice JSON's `audio.sample_rate`, and the model cards already record all three
— what is missing is the export DECLARING them). **The published `loom-ai-org/kokoro-82m-loom` still
carries the broken file** and needs re-uploading with the re-export.

**And the gate that would actually have caught defect 1 is still not written.** `test_tts_text_door.py`
pins declarations; `_default_voice` checks the pack's shape. Neither can tell a voice from noise — the
only thing that can is Supertonic's shape, an end-to-end waveform recorded with the default and
compared against. Worth stating plainly because the first draft of this fix claimed in a docstring that
the sweep "now range-checks the baked pack", which was never written: the same species of unbacked
claim as the "verified against the synthetic pattern the gate uses" that shipped the noise in the first
place, caught this time only by grepping the diff for its own promises before committing.

