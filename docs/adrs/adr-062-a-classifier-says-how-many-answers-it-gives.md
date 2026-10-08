---
type: adr
status: accepted
date: 2026-10-01
tags: [high-level-api, contract, classification, embeddings, vad, family-13, engine, loom-py]
supersedes: []
---

# ADR-062: A Classifier Says How Many Answers It Gives

## Context

Family 13 ([Epic-03](../epics/epic-03-model-coverage.md#family-13-small-audio-classifiers-and-embedders))
brought three new readings of an audio model's output: a speaker EMBEDDING (one vector per clip,
TitaNet), a CLASS distribution per CLIP (language id, ECAPA-TDNN on VoxLingua107), and a class
distribution per encoder FRAME (voice activity, MarbleNet; speaker segmentation, pyannote).

The contract already had `loom.output.kind = "class"`, and it meant one class **per input token**,
because family 12's token classifiers were the only classifiers. `model_contract.h` named an
`embeddings` kind that no export declared and no door answered. A host could not tell a VAD's per-frame
answer from a language id's per-clip one except by the answer's shape, and a one-frame VAD answer is
one row too. A per-frame answer is also useless without the time each row covers, which only the
checkpoint knows (MarbleNet's frames are 20 ms from 0; pyannote's are 16.875 ms from 22.5 ms, because
its SincNet front end pads nothing).

The user decided the names once for the family, before any leaf (2026-10-01).

## Options

1. **A kind per granularity** (`class` for argmax labels, a new `frame_scores` for per-frame
   probabilities). Gives a VAD its own door, but makes the language id's top-k a third kind, and splits
   one contract (a distribution over declared labels) three ways.
2. **One `class` kind plus a granularity key**, `embeddings` for vectors.
3. **Infer the granularity from the answer's shape.** Free, and wrong for a one-row frame answer.

## Decision

Option 2. Three keys, read by `ModelContract`:

| key | type | meaning |
|---|---|---|
| `loom.output.granularity` | str | `token`, `frame` or `clip`. **Absent on a `class` output means `token`**, so every family-12 file reads as before. |
| `loom.output.frame_rate` | f32 | frames per second of a `frame` output |
| `loom.output.frame_offset` | f32 | where frame 0 starts, in seconds. Written only when not 0 |

Frame `i` covers `[offset + i / rate, offset + (i + 1) / rate)`.

**A `frame` or `clip` answer is the model's own distribution, not a decision.** The driver returns
softmax probabilities, row-major `[n_rows, n_labels]`. Every VAD consumer thresholds and smooths in its
own way, a language id is read top-k, and pyannote's powerset classes are converted to speaker activity
by a rule the caller owns. A token classifier keeps returning argmax ids, because its door answers
"which label" and it predates this.

Two engine doors, `loom::audio::classify` and `loom::audio::embed` (`audio_classify.h`), run the
driver once over the waveform and cut the flat answer by the contract. They refuse an answer that is not
a whole number of label rows, and a clip answer with more than one row. The cut lives in the engine for
`text_classify.h`'s reason: two hosts reshaping it independently is how they would come to disagree
about which frame is which. The interfaces are `speech2class` and `speech2embeddings`. loom-py's
two planned doors became real ones, and `loom_cli --wav` runs both.

## Consequences

* Two new tasks, `audio-classification` and `audio-embedding`. The first covers both granularities:
  how many answers there are is the contract's, not a second task.
* `speech2class` returns `AudioClasses` (rows, labels, granularity, row start times, `top`, `best`,
  `probability(label)`). `speech2embeddings` returns the vector unnormalised, a bare list like
  `text2text`'s string and `text2codes`'s frames: there is nothing to attach to it. Comparing two
  embeddings, and the threshold, are the application's.
* The doors take one whole clip at the file's own `loom.sample_rate`. There is no windowing, because
  none of family 13's leaves wants it. pyannote is TRAINED on 10 s windows, so a host running it over a
  long recording slides its own window, as pyannote's pipeline does.
* A future frame-level EMBEDDING (a per-frame encoder) is `embeddings` at `frame`, and needs only a
  door: the keys already say it. **Amendment 2026-10-03:** the first one exists -- WakeHuBERT tiny
  (Epic-03), 128 features per 20 ms frame -- and ships with the keys and NO door, by the user's call:
  `loom::audio::embed` refuses it with an error naming the granularity, and its card calls `infer` and
  cuts the rows itself. One wart remains until the door exists: the engine still reports the interface
  as `speech2embeddings` (`ModelContract::interface_name()` reads only the modality pair), so
  `model.capabilities` lists a door that refuses this file. The hub tracks the door.
* **Amendment 2026-10-08: the door answers frames** (released in 1.0.0-rc15, the user's call that it
  lands before that release). A fourth key, read by `ModelContract`:

  | key | type | meaning |
  |---|---|---|
  | `loom.output.embedding_dim` | i32 | the width of one row of an `embeddings` output -- the role `loom.labels` plays for classes |

  The exporter reads it off the traced output's last axis and writes it on every `embeddings` output;
  it is REQUIRED at `frame`, because a flat frame answer cannot be cut without it (the frame count is
  not the host's to derive: an encoder may pad or offset). `loom::audio::embed` now returns an
  `Embeddings` (`granularity`, `n_rows`, `dim`, `frame_rate`, `frame_offset`, `values`, `row(i)`,
  `row_start(i)`) shaped like `ClassProbabilities`, for both granularities. It refuses a `frame` file
  with no declared width (every frame file exported before the key: re-export it), an answer that is
  not a whole number of rows, and a `clip` answer of more than one row. A `clip` file with no width --
  TitaNet's and ECAPA's published files -- is one row of the whole answer, so nothing published moves.
  `interface_name()` needed no change: the door it names now answers the file, which closes the wart
  above.

  **The return type splits by granularity in loom-py**, and that is the one place the Consequences above
  change. `speech2embeddings.infer` keeps returning the bare vector for a `clip` file (TitaNet's
  published card depends on it), and returns a `FrameEmbeddings` (`rows`, `times`, `dim`, `frame_rate`,
  `frame_offset`, `len`) for a `frame` file. The bare list was right for a clip because there was
  "nothing to attach"; a frame answer has its times, so it gets the object `AudioClasses` already is.
  Rejected: one object for both, which would have broken a published card for no gain to its reader.
* Pinned by `tests/ci/test_audio_classify.cpp` (the cut, the frame times, both refusals, and the
  `token` default; since 2026-10-08 also the frame-embedding cut, its times, the missing-width and
  skewed-row refusals and the clip width check), whose sabotage arms turn it red, and in loom-py by
  `test_api.py`.
