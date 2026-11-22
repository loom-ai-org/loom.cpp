---
type: adr
status: accepted
date: 2026-11-22
domain: model-coverage
tags: [model-coverage, tts, kokoro, paradee, modernbert, ettin, text-front-end, tokenizer, javascript, regex, voices]
supersedes: []
---

# ADR-142: kale-tts-1-tiny Runs Its JavaScript Front End in the Engine, and Exports From the Teacher Its Release Was Built On

Serves [Epic-03](../epics/epic-03-model-coverage.md) and [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).
Follows [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md) (a front end's rules ship as data, its shape is
code) and [ADR-135](adr-135-paradee-reuses-kokoros-phases-and-carries-its-voices-as-rows.md) (Paradee: Kokoro's
phases, voices as rows).

## Context

kale-tts-1-tiny (KaleComputer, Apache-2.0; 2026-11-19) is a 45.7M-parameter English TTS with ten voices designed
with Maya-1, 24 kHz. Its release graph is ONNX. Its trained parts are also published as PyTorch weights
(`pytorch/tiny/`), with the training code (`recipe/`), so it qualifies for the zoo.

* **Text side.** It is distilled from kale-tts-1-base, a Kokoro-82M fine-tune. ALBERT is replaced by:
  * Ettin (`jhu-clsp/ettin-encoder-17m`, a 7-layer ModernBERT, MIT), which reads the sentence with the previous
    one as context; the mean of its last two layers is averaged over each word's sub-tokens;
  * K20Text (3.0M parameters): a phoneme embedding, conv blocks, the phoneme's word row and two transformer
    blocks, giving the 512-wide input the teacher's predictor read.
* **Predictor and text encoder.** These are the teacher's (Kokoro's classes), with fine-tuned duration and
  pitch heads.
* **Decoder.** This is Paradee's `SourceFilterHead2` behind Kokoro's AdaIN trunk at width 256. Kale vendors
  Paradee's `vhead.py` unchanged.
* **Voices.** Each voice is a Kokoro voice pack plus a learned 128-number pitch offset for the predictor, a
  16-number row for the decoder, and four end-clamp numbers.
* **Front end.** The front end is JavaScript (`frontend/`), and upstream's Python runtime runs it under QuickJS.
  * Kale's terms table and dizione's text rules (numbers, money, dates, times, units, codes, addresses) come
    first.
  * Then sentences, then a CMUdict word list with 3,800 hand-checked overrides.
  * Homographs go through dizione's picker and Kale's own rules.
  * Unknown words go to a 0.9M-weight GRU guesser.
  * The British voices (Amy, Jack) read British files and rules.

  Each sentence's model inputs are its phoneme ids (Kokoro's symbols), each phoneme's word index, and the Ettin
  sub-tokens with each one's word.

Two findings shaped the work:

1. **The tiny repo's `pytorch/teacher/kale-tts-1-teacher.pth` is not the teacher its release graph was built
   from.** Against the fp32 ONNX:
   * K20Text, the pitch heads, the duration projection and the decoder are those weights, tensor for tensor;
   * the DurationEncoder and the TextEncoder differ (by up to 0.11);
   * durations were 1 to 7 frames off on two sentences of three.

   `KaleComputer/kale-tts-1-base`'s `kale-tts-1-base.pth` matches the graph exactly. Its voice packs, plus the
   pitch heads' `pv`, are the release's `voices/<id>.bin` to 2e-8.
2. **No loom front end produced a word encoder's input.** The phoneme door can give the phonemes. It cannot give
   the words beside them, the sub-tokens of a second vocabulary, or the previous sentence.

## Options

* **A phoneme door plus a second vocabulary in the GGUF.** loom-py would phonemize with the `kokoro` style (misaki
  folded), and the engine would carry Ettin's BPE for the words. Rejected at the user's choice ("do it like you
  did for Chatterbox").
  * It is still not upstream's front end: misaki is not Kale's lexicon, and has no dizione rules, no homographs
    and no guesser.
  * It would also put a new door shape in loom-py for one model.
* **Upstream's whole text path as a tokenizer kind** (Chatterbox, CosyVoice3, Soprano). Chosen.

## Decision

### The export: from the PyTorch release, kale-tts-1-base as the teacher

`kale_tts_export.py` loads:
* the tiny repo's four checkpoints;
* kale-tts-1-base's predictor and text encoder, with `kline.load_kmodel`'s weight-norm rename, every tensor
  required;
* Ettin at the commit Kale pinned.

It finds the teacher and Ettin as sibling checkouts of the release, or on the Hub. K20Text is built from
upstream's own source: `textside.py`'s `_mods`, compiled out of a file that imports `modal` at the top.

The phases:
* **`ettin`:** ModernBERT's eager forward, batch-free. HF views the fused QKV as 5-D, one axis more than ggml has.
  The rotary tables come per layer from each layer's own `inv_freq`. The sliding window (`finfo.min` beyond 64
  either side) is built from an outer difference spelled as two K = 1 matmuls. Exact against HF eagerly, including
  300-token inputs.
* **`text_side`:** K20Text, with upstream's `gather(wavg @ h, widx)` reordered as `wavg[widx] @ h`. The driver
  builds `wavg[widx]`, so the phase has two axes, not three.
* **Kokoro's prosody phases** and **Paradee's synth**, unchanged.

The driver carries the voices as rows of three tables:
* the prosody packs: each voice's 510 rows for the phoneme count, `pack[min(n, 510) - 1]`;
* the decoder rows;
* the clamp numbers.

It applies the release graph's per-voice end clamp and unvoicing floor to the F0 it reads back.
* The clamp: in the last `tail` frames up to the last voiced one, voiced F0 is held at the voice's floor.
* The floor: frames under `unvoice_hz` are set to 0.

`infer` follows every sentence with 0.12 s of silence, as upstream's `speak` does; the stream yields sentences
alone.

**Exporter finding.** Ettin's positions read their length off the input's shape. Read off the LayerNorm's output,
the exporter's shape walk resolved it to the trace's constant. `torch.arange(S)` became a static one-row
`RANGE_1D`, so every position was 0 and the encoder 18.6 off. A hub item tracks the shape walk.

### The front end: `tokenizer.ggml.model == "kale_tts"`

`loom::KaleTtsVocab` is the shape of upstream's path, one class per JS module. The file supplies all data:
* every rule's pattern and flags, built in `kale_tts_tokenizer_export.py` as the JS builds them;
* dizione's tables and homograph classifier;
* both accents' lexicons, with the overrides read over them;
* both GRU guessers' fp16 weights;
* Ettin's BPE (the engine's `BpeVocab` under this tag);
* NFKC and case tables taken from Python's `unicodedata`.

`encode(text, lang)` returns the record count, then per sentence:

`n, ids[n], widx[n], n_words, n_sub, sub[n_sub], sub_word[n_sub]`

`lang = "en-GB"` reads the British front end. loom-py passes the voice's language (`tts.voice_languages`).

**`loom::JsRegex`: an ECMAScript dialect beside `PyRegex`.** The rules use:
* lookbehinds;
* named groups;
* `\p{L}`, `\p{N}`, `\p{Ll}`;
* ASCII `\w` and `\b`;
* ECMAScript's `\s` and `.`;
* `$1` and `$<name>` templates;
* `split` with captures spliced in.

Python's semantics differ on enough of these that they get their own class. A lookbehind is bounded by its
body's width, and a search skips positions its first character cannot start at. Without both, a corpus took over
10 minutes; with them, 82 s for 5,678 texts.

**Ettin's sub-tokens are tokenized a word at a time.** GPT-2's pre-tokenizer joins a single space to the word
after it, so each word's sub-tokens are its own ` word`'s. One exception: a sub-token that is only the leading
space (`Ġ` before `)))'`) spans no character of the word, and upstream's overlap test gives it to no word.

## Measured (2026-11-22)

* **Front end.** Identical to upstream's JavaScript (packed records, sentences and IPA) on **8,678 / 8,678 texts**,
  both accents:
  * Kale's three front-end test sets, its 192-line bench script and its whisper-check items;
  * 2,500 lines of loom's docs and READMEs (numbers, versions, paths, URLs, code);
  * 26 hand-written edge cases;
  * 3,000 randomized strings built to stress the rules.
* **Model.** Against upstream's modules with the noise pinned (`scripts/kale_tts_reference.py`), every duration
  is upstream's:
  * 56-74 dB on four sentences × three voices;
  * the gate's four cases (Kenzie, Amy, Magnus, Toby): 57.7-64.3 dB;
  * the seed alone is the pinned run to 1.2e-7;
  * the stream gives one piece per sentence.
* **Whisper small through `text2speech.infer(text)`** (loom-py, the voice's own accent; Paradee's 20 held-out
  English sentences, numbers and names included):

  | file | over the ten voices | per voice |
  |---|---|---|
  | F32 | 3.5% | 2.0-5.8% (Amy 5.8%, Jack 4.3%) |
  | Q8_0_PC | -- | Kenzie 3.7%, Amy 4.3%, Magnus 3.5% (F32: 3.7%, 5.8%, 3.2%) |

  Kale's own bench lines (12 per voice, no digits) are 0/1210 words wrong.

## Consequences

* **`phonemes=` is not a door for this model.** Its vocabulary is the front end's. `tokens=` takes the packed
  records.
* **Upstream quirks are not reproduced:**
  * a JS object lookup makes the words `constructor`, `toString` and their kin crash or misread upstream;
  * NFKC and `\p{...}` follow Python's `unicodedata` (Unicode 14) where V8 uses its ICU's.
* **The guesser's arithmetic matches upstream's order.** It is evaluated in double precision, one term at a time
  (the engine builds with `-ffp-contract=off`). glibc's `exp` and `tanh` can still differ from V8's in the last
  place, so a near-tie in the argmax could flip; none did over the corpora above.
* **The model needs the engine that has the kind.** A GGUF of this model needs a `loom-py-rt` after 1.0.0. Its
  card waits for that release.
