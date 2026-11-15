---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, parakeet, transducer, tdt, nemo]
supersedes: []
---

# ADR-122: Parakeet Streams the Way NeMo's Buffered Script Does

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for Parakeet TDT and RNN-T. The loop is
[ADR-110](adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md)'s, now shared.

## Context

The survey listed six NeMo checkpoints under "NeMo's buffered chunked inference" as one host-side row, with
Canary's long-form merge ([ADR-063](adr-063-a-long-clip-is-decoded-the-way-its-reference-decodes-it.md)) as
the nearest code. Reading NeMo showed three different references, not one:

* **Transducers** (Parakeet TDT and RNN-T): `speech_to_text_streaming_infer_rnnt.py`. It is the script ADR-110
  ported for the unified model. Every step runs the model's own forward over up to `left + chunk + right`
  seconds (10 + 2 + 2 by default) and decodes the chunk's rows with the label-looping greedy decoder, its
  state carried. The script sets a chunked attention mask only for a `chunked_limited_with_rc` encoder;
  Parakeet's attention is full, and stays full over the buffer.
* **CTC** (Conformer-CTC, Citrinet): the CTC script left NeMo's examples, and its README sends CTC to
  NeMo 3.0's `BufferedCTCPipeline`. That pipeline is a different design, with overlap and end-of-utterance
  heuristics.
* **Canary** (AED): `speech_to_text_aed_streaming_infer.py`, AlignAtt or wait-k over the decoder's
  cross-attention.

This ADR is the first: no host loop, no merge.

## Options

1. **A host loop with ADR-063's merge** (the survey's guess): an approximation of nothing NeMo runs for these
   models.
2. **Windows over the one pass** ([ADR-107](adr-107-a-re-decoding-asr-stream-keeps-its-hypothesis-in-the-state.md)'s
   way): full attention means a window's rows move as audio arrives; there is no finite reach to cut at.
3. **NeMo's script, in the driver**, as ADR-110 does for the unified model.

## Decision

Option **3**, with **no new graph**. Parakeet's one-pass `encoder` phase is already the step: it takes
`(waveform, length)`, runs NeMo's preprocessor (the mel normalised over its input, here the buffer), and emits
exactly `encoded_len` rows ([Retro-068](../retros/retro-068-a-slice-end-the-walk-could-not-read-kept-the-whole-axis.md)'s
cut). A re-export adds driver code only. `infer` is unchanged.

* **One loop for both models.** ADR-110's `02_unified_stream.lua` moved to
  `transducer_driver/11_buffered_stream.lua`. Per leaf: the step phase (`S_STEP_PHASE`), whether it takes a
  language prompt (`S_STEP_PROMPT`) and the mask's three sizes (`S_STEP_MASK`), and the default sizes.
  Parakeet's are `encoder`, no, no, 10/2/2 (the script's own). The unified model's are `encoder_step`, yes,
  yes, 5.6/0.56/0.56. The call site names its phase by a constant, so it is declared (`ComputedCall`) with each
  leaf's inputs.
* **The decode learned TDT.** `nemotron_asr_driver/00_stream_decode.lua` moved to
  `transducer_driver/10_stream_decode.lua` and reads the duration head as the one-pass decode does. NeMo's
  TDT label-looping state carries `time_jumps`: how far the last duration reached past the chunk's rows. The
  next chunk starts that far in. The loop keeps it as state `time_jump`; RNN-T never jumps.
* Everything else is ADR-110's, including the one liveness difference: a step runs once a sample past its end
  has arrived, and the step that reaches the clip's end runs in `stream_finish`.
* The oracle is `fixture_gen/reference_stream_nemo_buffered.py` (was `..._nemotron_unified.py`): the script's
  loop at batch 1 for any NeMo transducer. It uses the script's default decoding config, the mask only where
  the script sets one, and `none` for a model without a prompt. It also records each step's carried jump.

## Consequences

* **The stream is NeMo's stream.** `test_e2e_nemo_buffered_stream` (was `test_e2e_nemotron_unified_stream`)
  covers both:
  * **TDT:** jfk at four latencies (10/2/2, 2/1/1, 4/0.48/0.4, 1/0.16/0.08 s), plus cosyvoice3_ref, voxtral_tts_ref
    and 60 s of joined clips at the default. Every case ends on NeMo's ids. Every step's rows are within
    4.1e-05 of NeMo's relative to their absmax. Every step's carried jump is NeMo's (144 steps, 42 jumps).
    887/887.
  * **RNN-T:** jfk at 10/2/2 and 1/0.16/0.08, plus voxtral_tts_ref and long60. 518/518, rows within 6.7e-06.
* **Ids alone cannot see the jump.** A driver that reset `time_jump` to 0 ended on every case's ids: the
  frames a dropped jump re-reads were skipped by a blank's duration and decode as blanks again. The gate
  therefore pushes one encoder frame at a time and compares the state after each step with NeMo's
  `jumps.npy`. That sabotage fails 23 checks on jfk's four latencies, one per carried jump. Reading the
  jump forward (`t + n_frames`) cuts jfk to "And so my".
* **The shared code was re-gated where it already ran:** the unified model re-exported, 299/299 on four of
  its cases (`en` and `auto`); Nemotron 3.5's cache-aware stream (the decode only), 43/43 on two clips.
* **At the default, the stream is the one pass on short clips.** At 10/2/2 TDT's streamed ids are the one
  pass's on jfk and cosyvoice3_ref; on 60 s the stream writes 269 ids against the one pass's 276.
* **Below about 1 s of right context it is not usable, and that is the model.** TDT at 4/0.48/0.4 writes
  "And so my father Americans." and stops; at 1/0.16/0.08 it writes nothing. RNN-T at 1/0.16/0.08 writes "good for you can do for your country". Decoding
  NeMo's joined chunk rows once, without the carried state (the script's `simulated` mode), gives the same
  text. So the rows themselves are wrong: an offline full-attention encoder sees almost no future at a
  buffer's end. `stream_open` does not refuse these sizes, because NeMo does not, and the gate pins them as
  NeMo writes them. The script's recommended settings keep 2 s of right context or more (10-2-2,
  10-0.16-3.84).
* **Speed.** Dev box CPU (4 cores), F32, 100 ms pushes, at 10/2/2: RNN-T 0.61x real time on jfk (RTF 1.64),
  0.93x on voxtral_tts_ref, 0.50x on 60 s; TDT about 0.85x on jfk. Each step re-encodes up to 14 s for 2 s of
  new audio, which is the reference's design. A GPU run has not been done.
* **Parakeet's buffer is bounded by the encoder phase's traced range (20 s)**, so the script's long-file
  recipe (10/10/5, 25 s) is refused. Raising `MAX_SECONDS` is a re-export, if anyone asks.
* Publishing needs a re-export on an engine with the door, rc21 or later. The published files give the same
  text in one call, and the live door refuses them.
* **CTC and Canary are not this.** They are separate rows with their own references (see Context).

## Related

* `loom-exporter/loom_exporter/transducer_driver/10_stream_decode.lua`, `11_buffered_stream.lua`,
  `parakeet_export.py` (`buffered_stream_constants`, `stream_facts`), `nemotron_unified_export.py`,
  `transducer_export.py` (`stream_decode_constants`, `stream_decode_fragment`),
  `tests/ci/test_transducer_buffered_stream.py`, `fixture_gen/reference_stream_nemo_buffered.py`
* `tests/gate/test_e2e_nemo_buffered_stream.cpp`
