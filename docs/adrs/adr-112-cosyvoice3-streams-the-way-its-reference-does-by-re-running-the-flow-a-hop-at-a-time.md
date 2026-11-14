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
* CUDA, RTX 5090 (`LOOM_TEST_DEVICE=gpu`): the stream gate passes 44/44.
  * The mel is within 3.7e-3 of the one pass and 4.5e-3 of upstream's loop.
  * The joined wave is bit-identical to one vocoder call over the same mel.
  * Both cache modes interleave bit-identically, the first check of the per-stream mode on a discrete GPU.
  * The first piece comes after 0.5 s and the one pass takes 1.2 s, 7.3x real time (RTF 0.137).

  Three engine fixes it took, each found here:
  * `RSQRT` lowered to ggml's own ops. Its CPU-only custom op had put a host round trip into every
    RMSNorm and made the LM 79 ms a token (0.38x real time); it is now ~3 ms.
  * `ggml-0063`: a `GGML_PREC_F32` product skips TF32 (ADR-069's 2026-11-12 amendment). With TF32, the
    LM's pinned draws moved and the mel was 0.40 off.
  * `ggml-0064`: a prefix-stable cumsum. The vocoder's sine phase is one, and the stream re-runs it over
    a growing mel. Before it, the join was 5.9e-3 from one call.

  `tests/gate/test_gpu_cumsum_prefix_stable` holds the last one: it fails on 223 of 250 rows without the
  patch.
* The one-pass gate passes 139/139 on the dev box's CPU, the workstation's CPU and the RTX 5090.
  * Its free-running arms were graded by waveform rmse, which the vocoder's sine phase makes meaningless
    across machines. On the cloned voice the reference's own f32 run is 0.249, worse than silence
    (0.205). The workstation's CPU (0.104) and the 5090 (0.158) failed the 6e-2 bound with every phase at
    its floor, while a waveform 3 dB down passed it.
  * The arms are now graded by log-mel distance against the f64 waveform (`tests/support/log_mel.h`,
    bound 1.4 dB). Builds score 0.07-0.68 dB and the reference's own f32 run 1.04 dB. A 10 ms shift
    scores 1.7-3.7 dB, a level change of 3 dB scores 3.0 dB, and silence over 100 dB.
  * The clone also gained the teacher-forced mel check it lacked (2.3e-3 to 3.3e-3).
  * A 3 dB level sabotage reddens both free-running arms (3.1 and 3.3 dB).
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

## Amendment 2026-11-12: a seed is the same take both ways

The first loom-py run of the stream found that a seed alone did not give `infer` `streaming = 1`'s audio.
Seed 7 came out 7.32 s streamed and 6.92 s in one pass, though loom-py's docstring promised the join.
Both paths seed one generator, but `infer` drew every LM uniform, then the whole grid's flow noise, then
the whole wave's sine noise, while the stream interleaved them hop by hop. So after the first hop the LM
sampled from other uniforms. Every check above pins `draws`, `noise` and `nsf_noise`, so none could see it.

* Options: (a) document that only a fully pinned stream joins; (b) give each phase its own generator,
  which the bridge does not have (one per stream, ADR-099); (c) make both paths draw in one order. Chose (c),
  driver only.
* The LM's uniforms are one block drawn when its loop begins: two per step of the whole budget,
  `max_token_text_ratio` x the text's tokens, used or not.
* The flow noise and the sine noise are drawn a FRAME at a time in frame order (80 gaussians, then
  9 x 480 uniforms for a generated frame) by `cosyvoice3_draw_frames`, which `infer` calls for the whole
  grid and the stream for a hop's worth. Per frame, so the cut cannot change the order. A pinned one is
  not drawn, in both.
* Cost: a seed's audio from `infer` moves on re-export, in both modes. Old files keep theirs, as with
  ADR-104's SNAC. A short `draws` pin now throws instead of silently falling back to the generator.
* Gate section 6: a seed alone, stream against one pass, mel and samples. RTX 5090 49/49 (seed only:
  438/438 frames, mel 2.0e-3, 210,240 samples both ways); workstation CPU 47/47 (7.6e-4). The previous
  export fails it (44/47: 204,480 against 205,440 samples). The one-pass gate stays 139/139 on both;
  its arms pin their draws.
* loom-py, seed only: 390/390 frames, mel 1.65e-3 (CUDA) / 8.8e-4 (CPU); waveform log-mel 0.17 / 0.11 dB
  mean, against 2.8 dB between the two attention modes. 5090: one pass 9.5x real time (RTF 0.105), first
  piece 0.31 s, whole stream 3.9x (RTF 0.256).

## Related

* `loom-exporter/loom_exporter/cosyvoice3_export.py` (`chunk_mask`, `_dit_attention`, `EstimatorPhase`,
  `STREAM_*`), `cosyvoice3_driver/00_stream.lua`, `01_lm.lua`, `02_flow_plan.lua`
* `tests/gate/test_e2e_cosyvoice3_stream.cpp` (`LOOM_COSYVOICE3_STREAM_GGUF`, `_REF_DIR`, `_OLD_GGUF`), its
  oracle `scripts/cosyvoice3_stream_reference.py` (`--f64` prints the f64 spread above),
  `tests/gate/test_e2e_cosyvoice3_lua_driver.cpp`
* Upstream: `CosyVoice/cosyvoice/cli/model.py` (`CosyVoice3Model.tts`, `token2wav`),
  `flow/flow.py` (`CausalMaskedDiffWithDiT.inference`), `flow/DiT/dit.py`, `hifigan/generator.py`
  (`CausalHiFTGenerator`)
