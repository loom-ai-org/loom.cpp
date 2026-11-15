---
type: adr
status: accepted
date: 2026-11-09
tags: [streaming, live-api, stream-state, tts, audio-out, pocket-tts, mimi]
supersedes: []
---

# ADR-101: Pocket-TTS Streams a Frame at a Time Through a State-Carrying Decoder Step

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Takes
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door from a sentence
chunk per piece down to a latent frame (80 ms) for Pocket-TTS, all four sizes.

## Context

ADR-100's Pocket-TTS piece is a sentence chunk: the first audio waits for the whole first sentence (3.65 s
of 11 s on the dev box). Upstream's `generate_audio_stream` decodes Mimi per frame, carrying the decoder's
`mimi_state` between calls: each causal conv's last `(k - 1) d + 1 - s` inputs, each transposed conv's
`k - s` overlap, and the 2-layer transformer's cache, masked to a window of 250 positions (16 per frame).
Our `mimi_decoder` runs a chunk's frames in one call from a fresh state.

## Options

1. **Recompute with left context**, as Kyutai's Mimi encoder does: the decoder's receptive field is finite
   (2 x 249 positions, about 31 frames), so a frame decoded with 32 frames before it is the streamed one.
   No state at all, and about 33 times Mimi's work per frame.
2. **The engine's KV cache for the attention**, a private ring per module. A private cache today takes the
   file's LM geometry (6 x 1024, linear), not Mimi's (2 x 512, a 250 ring): engine work for one model.
3. **Every state as an explicit graph input and output**, kept on the device between calls by
   `loom.keep_state`, Zipformer streaming's shape (ADR-097/098). Export work only.

## Decision

**3.** Two new graphs beside `mimi_decoder`, which is unchanged:

* `mimi_step(latents [1, n, 32], positions [16 n], key_positions [250 + 16 n], state_1..state_13)` ->
  `(samples [1920 n], state_1'..state_13')`. Thirteen states, 526,720 floats (2 MB, nearly all keys and
  values). `n` is any count up to 64: every slice has a fixed width counted from one end, so the frame
  count stays one axis. Keys sit at `key_positions`, the state's first; a negative one is a slot no frame
  wrote, and the mask drops it, so a chunk's first call reads what a fresh decoder reads.
* `mimi_init(zero)` -> the thirteen states, broadcast from one number on the device: a fresh state needs
  no 2 MB array from Lua, and the file grows 42 KB, not 2 MB.
* The driver's `stream_feed` takes `piece_frames` latent frames per call and decodes them in one
  `mimi_step`. The default grows, 1 frame, then 2, 4 and 8 from then on: one frame per call costs about 20%
  over the one-pass decode, eight about 5% (ggml's `CONV_TRANSPOSE_1D` and the matmuls on 16 rows pay per
  call). `piece_frames = 0` is ADR-100's sentence chunk, which `infer` asks for, so one-pass audio is
  bit-identical to the export before this.
* Between feeds the open chunk's loop (step, EOS countdown, `n_past`, the last latent) is state values;
  the flow LM's history stays in the model's KV cache, leased or owned per ADR-099.

## Consequences

* Dev box CPU, three sentences: Pocket-TTS's first audio after 0.20-0.24 s (was 3.65 s), 2m's after 0.02 s
  (was 0.44 s). `loom_cli --stream 1`: 15.36 s of audio in 9.14 s = 1.68x real time (RTF 0.595), the one
  pass 8.95-9.03 s.
* Graded: in PyTorch, groups of 1, 3, 1, 7, 2 and 26 frames match the one-call decoder within 2.7e-06 and
  upstream's own per-frame decode within 3.5e-06. In the engine, teacher-forced against the reference
  at 1 and 3 frames per step, max 2.3e-05, rmse 5.7e-07, the one-call arm's numbers (gate 82/82). The
  stream gate: Pocket 31 pieces, 1.2e-06 from one pass; 2m 3.3e-06; each stream alone, after a reset and
  interleaved with another and a one-pass call, bit-identical to itself.
* So a Pocket-TTS stream is no longer `infer` bit for bit: the step sums the same terms in another order.
  The stream gate now checks sample count exactly and values to 1e-4, and bit-identity where it holds (a
  stream against itself; Soprano against one pass).
* Sabotaged: every piece decoded from a fresh state is 1.39 off one pass and fails only the
  stream-versus-one-pass checks (19/21). Reading the state from the module's retained outputs instead of
  the stream's makes `mimi_step` read the buffers it writes, and fails everything, alone included.
  Without the transposed conv's bias correction, the hermetic conv test fails 3/3.
* Not covered: the codec pairs and every other audio-out row; a GPU run.

## Related

* `loom-exporter/loom_exporter/pocket_tts_export.py` (`MimiStepPhase`, `MimiInitPhase`, `_conv_step`,
  `_conv_transpose_step`), `pocket_tts_driver/00_stream.lua`
* `tests/gate/test_e2e_speech_stream.cpp`, `tests/gate/test_e2e_pocket_tts_lua_driver.cpp` (arm 1b)
