---
type: adr
status: accepted
date: 2026-11-12
tags: [streaming, live-api, stream-state, tts, audio-out, cosyvoice3, flow-matching, hift]
supersedes: []
---

# ADR-112: CosyVoice3 Streams the Way Its Reference Does, by Re-Running the Flow a Hop at a Time

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Puts CosyVoice3 behind
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door. Like
[ADR-110](adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md), the stream is the
reference's own streaming mode, so it is not `infer`'s output.

## Context

CosyVoice3's one pass is LM -> flow -> vocoder, each over the whole sentence: the Qwen2 LM draws every
speech token, the flow (a DiT under a 10-step guided Euler ODE) turns all of them into a mel, and
CausalHiFT turns that into a wave. On the 2-core dev box a 9 s sentence takes 141 s before its first
sample.

Upstream's `tts(stream=True)` does not carry state from hop to hop; it RE-RUNS. Each hop it runs
`flow.inference(streaming=True)` over the voice's prompt and every token drawn so far, keeps the mel past
what it already had, runs HiFT over ALL of the mel and hands out only the samples it had not. The hop is
counted in tokens: 25 plus whatever puts the prompt and the first hop on a 25-token boundary, then 50,
then 100 from there on. Every hop but the last also waits for 3 look-ahead tokens, which the flow's
pre-lookahead conv reads as context. `streaming=True` gives the DiT a block-causal mask of 50 frames
(one hop of 25 tokens), the chunk size the checkpoint was trained with. The non-streaming call uses full
attention. On the oracle sentence the two mels differ by 4.85.

What a stream must equal was measured in PyTorch before choosing:

* At f64, upstream's re-run loop is its streaming-mode one pass EXACTLY, mel for mel (max|d| 0). Each
  hop ends on a chunk boundary, the position conv and the chunk mask are causal, and the flow's noise is
  one fixed table sliced from the start.
* At f32 the two differ by 4.7e-3 on the mel: the ten guided Euler steps amplify the re-run's different
  rounding. CausalHiFT's sine source integrates F0 into a phase that reaches about 1e6 rad, and that
  turns the difference into about 1.08 on the wave. So a stream is compared with a one pass on the MEL,
  and the vocoder is checked on its own. Streamed over one fixed mel, HiFT is its one call within 1.1e-05.

## Options

1. **Carry the flow's state.** No such state exists. The ODE integrates the whole grid ten times, and
   each step attends over the prompt and every chunk before it. Carrying KV per chunk per Euler step per
   guidance branch is a computation upstream never runs, so nothing could check it.
2. **Re-run, as upstream does.** The flow is called with a growing token count and the DiT gets its
   chunk mask. This is export work plus driver work, and upstream's loop is its oracle.

## Decision

**2.** In the exporter:

* The estimator takes `chunk_ids [1, T]`, one per frame, and builds the mask from them in-graph:
  `clamp(relu(c_k - c_q), 0, 1) * -1e30`, so a frame sees its own chunk and the ones before. In
  streaming mode each id is `floor(frame / 50)`; otherwise every id is 0, which gives full attention. One
  graph serves both modes. `infer` keeps the non-streaming mode by default, and `streaming = 1` selects
  the other. The hop, its growth and ceiling, and the chunk are read off `CosyVoice3Model` by a CI test,
  so a checkpoint whose `static_chunk_size` is not 50 frames fails the export.
* The LM's loop moves to `cosyvoice3_driver/00_stream.lua` (`cosyvoice3_lm_begin` + `cosyvoice3_lm_step`),
  and `infer` runs the same loop, so the one-pass gate covers the live loop. Each step feeds the
  previous token BEFORE it samples, so no graph output is read across a call. This is the same rule as
  ADR-111.
* `stream_feed` is one hop. It draws tokens until the hop and its look-ahead are there or the LM stops,
  runs the flow over the prompt plus every kept token with the encoder rows past the hop dropped (on the
  last hop, the encoder's own zero pad), and appends the new mel. It then runs HiFT over all of the mel
  and hands out the samples that no later frame can change: all but the last 8 frames' (F0 condnet's
  causal pad 3, `conv_pre`'s look-right 4 and the iSTFT's last hop 1), and all of them on the last hop.
  These are read off the checkpoint at export.
* The noise belongs to the stream and is kept: the flow's draw is frame-major, so a prefix serves every
  re-run as upstream's `rand_noise[:, :, :T]` does, and the sine source's draw is per frame. A caller
  can pin the LM's draws, both noises and `return_mel` (the hop's mel instead of samples). This is how
  the gate compares a stream with a one pass. A text of several sentence chunks streams them one after
  another and refuses pins, because a pin is one generation.
* State between feeds: the text's chunks, the knobs, the LM's loop (prompt rows, step, `n_past`, RAS
  history, silence run, kept tokens), the hop, the mel so far and the samples handed out. The LM's
  history is the model's KV cache, leased or owned (ADR-099).

## Consequences

* Dev box CPU, the oracle sentence (223 kept tokens, 446 frames, 8.92 s): the stream gate passes 46/46.
  * The stream runs 4 hops; the first piece is 32,640 samples (1.36 s) and is ready after 46.5 s,
    against 141.3 s for the one pass.
  * The stream's mel is within 9.1e-4 of `infer` `streaming = 1` and within 2.0e-3 of upstream's own
    f32 re-run loop, which is itself 4.7e-3 from its one pass.
  * The joined wave has the one pass's 214,080 samples. HiFT run once over the stream's mel is within
    5.5e-06 of the joined wave.
  * `infer` `streaming = 1` is within 3.0e-3 of the reference mel, and full attention within 2.9e-3 of
    its own reference and 4.85 from the stream.
  * Each stream, after a reset and interleaved with another stream and a one-pass call under both
    cache modes, is bit-identical to itself. The old v5 file is refused by the door.
* Sabotaged by a same-length byte patch of the driver in the GGUF (`VOCODER_RIGHT_FRAMES` -> `0`, so each
  hop hands out samples the next one rewrites): the first piece grows by 8 frames to 36,480 samples, the
  joined wave is 0.591 off one vocoder call over the same mel, and the gate fails exactly that check
  (45/46). The mel checks still pass, because the patch touches only the vocoder's output.
* The one pass is unchanged by moving the LM loop: `test_e2e_cosyvoice3_lua_driver` 108/108 on the
  re-export (LM 76/76 tokens, flow mel 1.66e-3).
* Graded in PyTorch (`test_cosyvoice3_export.py`): the estimator against a tiny random DiT in either
  mode, the chunk mask's visibility, and the stream cut against `CosyVoice3Model`'s source.
* Re-running costs: hop k runs the flow and HiFT over everything so far, so a stream spends more in total
  than one pass, as upstream's does. The total was not measured. It is not real time on this box
  either: the one pass is 0.063x real time (RTF 15.8).
* `tests/support/npy_fixture.h` now refuses any fixture that is not little-endian float32. The gate's
  first run fed the LM text ids that were int32 bytes read as floats (denormals near 1e-43). The first
  sign was a flow grid four tokens short, 30 lines of driver later.
* Re-export needed: the estimator gains an input, and a file exported before this has no `stream_open`.
  Publishing waits for an rc21+ engine like the other streamed re-exports.

## Related

* `loom-exporter/loom_exporter/cosyvoice3_export.py` (`chunk_mask`, `_dit_attention`, `EstimatorPhase`,
  `STREAM_*`), `cosyvoice3_driver/00_stream.lua`, `01_lm.lua`, `02_flow_plan.lua`
* `tests/gate/test_e2e_cosyvoice3_stream.cpp` (`LOOM_COSYVOICE3_STREAM_GGUF`, `_REF_DIR`, `_OLD_GGUF`), its
  oracle `scripts/cosyvoice3_stream_reference.py` (`--f64` prints the f64 spread above),
  `tests/gate/test_e2e_cosyvoice3_lua_driver.cpp`
* Upstream: `CosyVoice/cosyvoice/cli/model.py` (`CosyVoice3Model.tts`, `token2wav`),
  `flow/flow.py` (`CausalMaskedDiffWithDiT.inference`), `flow/DiT/dit.py`, `hifigan/generator.py`
  (`CausalHiFTGenerator`)
