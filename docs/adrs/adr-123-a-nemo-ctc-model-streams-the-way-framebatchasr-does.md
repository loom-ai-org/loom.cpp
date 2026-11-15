---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, ctc, conformer, citrinet, nemo]
supersedes: []
---

# ADR-123: A NeMo CTC Model Streams the Way FrameBatchASR Does

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for the NeMo CTC models: Conformer-CTC small,
Citrinet-1024 and Citrinet-256-ls. The transducers' row is
[ADR-122](adr-122-parakeet-streams-the-way-nemos-buffered-script-does.md).

## Context

NeMo has two references for streaming a CTC model offline-trained on whole utterances:

* **`FrameBatchASR`**, driven by `examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py`
  (through r2.6; NeMo 3.0 removed the script and `get_buffered_pred_feat`, but `FrameBatchASR` is still in
  `streaming_utils`). It works on features, not audio:
  1. the clip, padded with `delay` encoder frames' worth of zeros, gets one raw log-mel (normalisation off);
  2. 1.6 s chunks of it slide into a 4 s mel buffer;
  3. each buffer is normalised per feature over itself and run through the model;
  4. of each buffer's argmax ids, the 1.6 s ending `delay` frames before its end are kept;
  5. one greedy CTC merge runs over everything kept.
* **`BufferedCTCPipeline`** (NeMo 3.0's `nemo.collections.asr.inference`, where the README now sends CTC).
  It clips around the buffer's middle too, but its transcript also depends on overlap detection and
  end-of-utterance heuristics, with thresholds of their own.

The user chose `FrameBatchASR`: deterministic, and gateable exactly.

Two facts of the reference had to be read off the running code, because the script says otherwise:

* From r2.4 the script reads `asr_model.subsampling_factor`, which no CTC model has (an `AttributeError`). Up to
  r2.3 `model_stride` was a parameter the docstring sets to 4 for Conformer and 8 for Citrinet. Those are the
  encoders' own `subsampling_factor`s, which is what the port reads.
* `FeatureFrameBufferer` fills the buffer with -16.635 (log-zero) only if `asr_model.preprocessor.log` is true.
  The attribute lives on the featurizer, not the preprocessor, so the fill is **0.0** for every NeMo CTC model.
  The port uses NeMo's actual value, read by NeMo's own test at export.

## Options

1. **Windows over the one-pass graph**: it computes and normalises its mel over its own waveform, while
   `FrameBatchASR` normalises over a mel buffer that starts with fill frames and ends with zero-filled ones.
   Not the reference.
2. **A host loop in loom-py**: `loom_cli` would not have it, and every other row is in the driver.
3. **FrameBatchASR in the driver**, over two graphs the one pass is cut into.

## Decision

Option **3**. The NeMo CTC leaf is now multi-phase (`ASRNemoCtcExportConfig`; it was a `Flattened` export,
which writes one graph). Every phase reads the same weights, so the file grows by 0.3 MB.

* **`ctc`** is the one pass, the same trace, decoded by the same greedy CTC components. `infer` gives the
  published file's text. It is not called `main_topology` any more: a phase of that name writes its weights
  unprefixed (the `Flattened` convention), and its anonymous constants then collided with the other phases'
  value names.
* **`raw_mel(waveform, length) -> [1, T, n_mels]`** is the preprocessor with normalisation shadowed off while
  it traces, as `FrameBatchASR.raw_preprocessor` runs it.
* **`ctc_buffer(features, length) -> log-probs`** is `normalize_frame_buffers` (per-feature mean and population
  std over the buffer, `+ 1e-5`, written as sums over the `length` input), then the model from its processed
  signal on.
* **The driver** (`nemo_ctc_driver/01_stream.lua`) is `FrameBatchASR`'s loop, in its own order of double
  operations.
  * The whole-clip mel is computed a chunk at a time, over windows reaching 4 frames (640 samples) past the
    chunk on each side. A frame reads n_fft/2 = 256 samples either side of its centre, plus one more for the
    pre-emphasis.
  * In PyTorch the kept frames equal the whole clip's exactly.
  * A chunk runs once its window's last sample has arrived. `stream_finish` appends the zero pad and runs the
    rest, ending on `AudioFeatureIterator`'s zero-filled last chunk, even an empty one.
* `stream_open` takes `chunk_seconds` and `total_seconds` (1.6 and 4, the script's). It refuses a chunk under
  one mel frame, a buffer shorter than its chunk, and a buffer longer than the step was traced for (20 s).

## Consequences

* **The stream is NeMo's FrameBatchASR.** `test_e2e_nemo_ctc_buffered_stream` runs jfk, cosyvoice3_ref,
  voxtral_tts_ref and a 60 s joined clip, each at 1.6/4, 0.8/3.2 and 4/10. All three models pass 1767/1767:
  Conformer-CTC small, Citrinet-1024 and Citrinet-256-ls.
  * Every case ends on NeMo's ids at 100 ms pieces and the whole clip, and on the first case at 10 ms, one
    chunk and 0.5 s too.
  * `raw_mel`'s kept frames, and the driver's own buffer after every step, are within 2.7e-04 of NeMo's mel.
  * `ctc_buffer`'s kept log-probs are within 6.1e-05 of NeMo's, relative to their absmax.
* **The buffered transcript is not the one pass's, and NeMo's own artefacts are kept.** jfk at 1.6/4 reads
  "and so am my fellow americans" (29 ids, the one pass 28). At 4/10 the first buffer's kept frames reach 3 s
  before the clip and decode nothing for its first second, so "and so" is lost. The stream reproduces both.
* **The driver's own mel is checked.** Four byte-patched drivers were run against jfk:
  * the log-zero fill NeMo intended (-16 for 0.0): "and so am i fellow americans";
  * no zero pad at finish: "your country" is lost;
  * the kept slice one frame later: "and" is lost;
  * a margin of 1 frame for 4: **passed**, because 1a probes `raw_mel` with its own windows, and jfk's ids
    survived the perturbed edge frames.
  
  So the gate now pushes one hop at a time and compares the driver's buffer (state `buffer`) after every step
  with the buffer NeMo builds from NeMo's mel. The short-margin driver fails it by 6.5; the real one is within
  2.5e-04.
* **Speed** (dev box CPU, F32, 100 ms pushes, the twelve cases' 256 s): Conformer-CTC small 7.7x real time
  (RTF 0.129), Citrinet-256-ls 12.1x (RTF 0.083), Citrinet-1024 1.75x (RTF 0.572). Each buffer is 2.5x its
  chunk at the default, so each second of audio is encoded 2.5 times.
* The exporter's gate snapshots of these three files move: new graphs, and the one pass renamed. Publishing
  needs a re-export on an engine with the door, rc21 or later. The published files give the same text in one
  call, and the live door refuses them.

## Related

* `loom-exporter/loom_exporter/nemo_ctc_export.py`, `nemo_ctc_driver/01_stream.lua`,
  `tests/ci/test_nemo_ctc_stream.py`, `fixture_gen/reference_stream_nemo_ctc_buffered.py`
* `tests/gate/test_e2e_nemo_ctc_buffered_stream.cpp`
