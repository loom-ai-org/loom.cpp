---
type: adr
status: accepted
date: 2026-11-12
tags: [streaming, live-api, stream-state, tts, audio-out, voxcpm2, audiovae]
supersedes: []
---

# ADR-111: VoxCPM2 Streams a Patch at a Time Through a State-Carrying VAE Step

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Puts VoxCPM2 behind
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door, cut the way
[ADR-101](adr-101-pocket-tts-streams-a-frame-at-a-time-through-a-state-carrying-decoder-step.md) cut
Pocket-TTS.

## Context

VoxCPM2's one pass draws every patch (4 latents, 160 ms of 48 kHz audio) and then runs `vae_decode`
over all of them in one call. On the 2-core dev box a patch costs about 2.4 s, so a 3 s clip's first
sample is ready only when the whole clip is.

Upstream's `generate_streaming` hands each patch to `AudioVAE.streaming_decode()` as soon as the DiT has
drawn it. That decoder's state is only its causal convolutions: each `CausalConv1d` keeps its last
`2 * padding - output_padding` inputs, and each `CausalTransposeConv1d` re-convolves its last input frame
and drops what that frame alone makes. Everything else in the decoder (snake, the sample-rate scale and
bias, tanh) is pointwise in time. So the streamed waveform is the one-pass waveform, up to f32 summation
order, in zero-shot mode. In continuation mode upstream's stream never decodes the prompt's patches, so
there it is not the same waveform. This export has no prompt audio, so that case does not arise.

## Options

1. **Recompute with left context**, the way Dia's DAC windows do (ADR-103). That needs no state, but the
   decoder's receptive field at 48 kHz spans dozens of patches, so each piece would redo most of the clip.
2. **Every state as an explicit graph input and output**, kept on the device by `loom.keep_state`. This
   is ADR-101's `mimi_step`, without the attention. Export work only.

## Decision

**2.** Two new graphs beside `vae_decode`, which is unchanged:

* `vae_step(latents [1, n, 64], state_1..state_26)` -> `(samples [1920 n], state_1'..state_26')`, for `n`
  up to 64 latents (16 patches). The 26 states, in the decoder's order: the depthwise k7 over the
  latents, then per upsampling block the transposed conv's `k - s` overlap and the three dilated units'
  k7s (`6d` inputs), then the last k7. The 1x1s keep nothing. A causal conv's state is taken off the
  JOINED input, so a call shorter than the state (the dilation-9 unit's 54 samples against one patch's
  32 at the first block) keeps the older samples too, as upstream does. The transposed conv is spelled
  as an overlap-add, not as upstream's re-convolution: the same sums, and every slice is a fixed width
  counted from one end, so the latent count stays one axis. The weights are `vae_decode`'s, aliased, so
  the file does not grow.
* `vae_init(zero)` -> the 26 states as zeros on the device: the zeros `CausalConv1d` pads with, and no
  overlap yet.
* The driver's one loop (`voxcpm2_driver/00_stream.lua`, `voxcpm_begin` + `voxcpm_step`) is what `infer`
  runs too, so the one-pass gate covers the live loop. Each step first brings the two LMs up to the
  patch it was handed (the prefill on the first step), then draws, integrates and tests the stop head.
  That makes it safe to cut anywhere: no graph output is read across a call. The original loop ran the
  feedback rows at the end of a step and read them at the start of the next. Between feeds everything
  else is state values (ids, knobs, pinned draws, teacher patches, `n_past`, the conditioning patch), and
  the LMs' history is the model's KV cache (ADR-099).
* `stream_feed` draws `piece_patches` patches and decodes them in one `vae_step`. The default grows: 1
  patch, then 2, 4 and 8 from then on. `0` is the whole clip through `vae_decode`.
* The stream never retries a badcase (a run that spends its whole ratio budget), as upstream's streaming
  mode does not: a piece already handed out cannot be taken back. `infer` still retries.

## Consequences

* Dev box CPU, "The history of the city of Rome begins on the Palatine hill.": 20 patches (3.2 s) in 5
  pieces, the first (160 ms) after 3.38 s against 48.7 s for the one pass. The join is 153,600 samples,
  the one pass's count, and at most 7.0e-07 from it. Each stream, after a reset and interleaved with
  another stream and a one-pass call under both cache modes, is bit-identical to itself. Speech-stream
  gate 23/23, the old v5 file refused by the door.
* One pass is unchanged by moving the loop: `test_e2e_voxcpm2_lua_driver` 43/43 on the re-export
  (teacher-forced patches max 1.4e-05, rmse 1.2e-06; free-running to the same stop step).
* Graded in PyTorch on a tiny random VoxCPM2 (every parameter randomised): pieces of 1x7, 4+1+2 and 2+5
  latents match the one-call decoder and upstream's own `streaming_decode()` within 1e-5. A second
  piece decoded from a fresh state is off by more than 1e-2 (`test_voxcpm2_export.py`).
* Sabotaged by a same-length byte patch of the driver in the GGUF (`vae_started == 0` -> `>= 0`, so every
  piece is decoded from `vae_init`'s fresh state): the join is 0.582 off one pass, and the gate fails
  exactly its two stream-versus-one-pass checks (19/21). The self-consistency checks still pass, as they
  should: the sabotaged stream is wrong the same way every time.
* Not real time on this box: VoxCPM2 is 0.066x real time (RTF 15.2) on 2 cores, and the stream cannot be
  faster than the loop. The door only moves the first audio forward. A GPU run is still to do, as for
  every stream.
* Re-export needed: a file exported before this has no `stream_open` and is refused by the door. Publishing
  waits for an rc21+ engine like the other streamed re-exports.

## Related

* `loom-exporter/loom_exporter/voxcpm2_export.py` (`VAEStepPhase`, `VAEInitPhase`, `_conv_step`,
  `_conv_transpose_step`), `voxcpm2_driver/00_stream.lua`, `voxcpm2_driver/01_generate.lua`
* `tests/gate/test_e2e_speech_stream.cpp` (`LOOM_STREAM_TTS_TEXT_A/_B`), `tests/gate/test_e2e_voxcpm2_lua_driver.cpp`
* Upstream: `VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py` (`StreamingVAEDecoder`),
  `model/voxcpm2.py` (`_generate(streaming=True)`)
