---
type: adr
status: accepted
date: 2026-11-09
tags: [engine, streaming, audio-out, codec, lstm, noise, snac, encodec, musicgen]
supersedes: []
---

# ADR-104: A Recurrence Carries Across Pushes, and Noise Belongs to Its Frame

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Extends
[ADR-103](adr-103-a-module-may-declare-a-ring-kv-cache-of-its-own.md)'s windows over a non-causal codec to
the two codecs it left, and [ADR-102](adr-102-a-codec-pair-streams-as-two-doors-and-a-causal-codec-carries-its-state.md)'s
pairs to MusicGen.

## Context

Three rows remained among the codecs: SNAC 24 kHz, EnCodec 32 kHz and MusicGen Small, the talker that decodes
through EnCodec. DAC streams by windows over its one-pass graph, with context covering its receptive field
either side (ADR-103). The other two codecs each break that recipe in one place:

* **SNAC is stochastic.** Four NoiseBlocks add `randn * linear(x)`. `infer` draws the noise per call,
  block-major (all of block 0's frames, then block 1's, ...). A window decodes some frames again, and it has to
  give them the noise they got the first time.
* **EnCodec has an LSTM between two convolution stacks.** The convolutions are not causal but are short:
  measured at f64, `pre` needs 3 frames each side and `post` needs 1. The LSTM is causal, but windows cannot
  stand in for its state: with 8, 16 and 32 frames of left context it is still 1.05, 0.97 and 0.94 away from one
  call.

MusicGen's decode loop has the shape of Dia's: four codebooks on a delay pattern, and guidance from a
second, unconditional decoder stream.

## Options

SNAC's noise:
1. **The stream draws its own noise** as it goes. One pass and the stream then differ, and only in
   distribution, so the gate cannot compare them.
2. **Noise keyed to frames.** Both paths draw frame-major (frame 0's draws for every block, then frame 1's)
   from a generator of the stream's own, and a frame's noise is kept beside its codes.
3. **The caller hands in the noise.** That works for an oracle, but a live caller does not have the frames
   ahead of time.

EnCodec's LSTM:
1. **Windows round the whole chain.** O(1) error, as measured above.
2. **Carry h and c across pushes.** `run_recurrent` continues a sweep from the state an earlier call left,
   with windows only round `pre` and `post`.
3. **Step the cell from Lua** with h and c as state values. This needs no engine change, but adds a Lua call
   per frame per layer and marshals the state on each one.

## Decision

SNAC **2**, EnCodec **2**, and MusicGen cut as Dia is.

* `loom.run_recurrent(module, seq, n, in, hidden, reverse, carry)`: when `carry` is a table, the sweep starts
  from its `h` (and `c`) instead of zeros and writes the final state back into it. Two calls with one carry
  equal one call over both halves.
* `CodecStream` always gives its stream a generator of its own. The driver's `stream_open` seeds it, from
  `inputs.seed` or the file's default.
* SNAC (exporter): `StreamingCodecExportConfig` with 3 coarse frames of context each side. The window fragment
  draws `NOISE_PER_FRAME` per arriving frame. `infer` is one call whose noise is drawn the same frame-major way
  (`window_one_pass`), so a stream and one call with the same seed decode the same waveform. A caller's own
  `noise_i` arrays are still accepted.
* EnCodec: `encodec_driver/00_stream.lua` handles each push in three steps. `pre` runs over a window with 3
  frames of left context. Rows whose right context has arrived go through each LSTM layer with its carry.
  `post` runs over a window with 1 frame of context each side. `infer` is unchanged.
* MusicGen: `stream_open` runs T5 and keeps both cross-attention K/V in the stream's state. Each
  `stream_feed` makes one decoder step and hands back frame `t` once column `t + 3` is drawn. `infer` is
  the stream run in one call.

## Consequences

* **SNAC's one pass moves** for a given seed, because the draw order changed. It is the same distribution
  and the same file contract. Against PyTorch at f64 with the same noise handed in, it is 2.9e-06 (another
  `noise_0`: 0.025). The stream decodes 129 real frames (JFK, encoded by SNAC) 2.2e-06 from one pass,
  19/19. Sabotage: 1 frame of context gives 0.35, and a stream seeded one off gives 0.19. It runs 3 coarse
  frames (256 ms) behind the codes.
* **EnCodec's one pass is bit-identical to the published file**, and the old file is refused by the stream
  door. The stream decodes 550 real frames 6.3e-07 from one pass, in uneven pushes and in pushes of one
  frame (22/22, both cache modes). Sabotage: a post margin of 0 gives 0.48. It runs 4 frames (80 ms) behind
  the codes.
* **MusicGen's codes are identical to the published file's** for greedy decoding with guidance, sampling with
  seed 7, and greedy decoding without guidance (160 codes each). Over 40 frames, the talker's stream equals
  its one pass and the pair is 6.0e-08 from decoding the codes in one pass (15/15). `loom_cli --stream 1` on
  the 4-core dev box CPU with guidance: first audio after 2.5 s, 5.0 s of audio in 56 s = 0.09x real time
  (RTF 11.3), so it is live only on a GPU.
* Pinned in CI: `test_gru_sweep` (an LSTM cell against numpy; two calls with one carry, LSTM and GRU; a dropped
  `c` changes the answer; a carry of the wrong width is refused). With the carry never read, both split cases
  fail.
* Found on the way and fixed here. `loom.set_state` stored a table with named keys as its empty
  positional part, so the first carry vanished between pushes
  ([Retro-104](../retros/retro-104-a-state-value-that-dropped-its-named-keys.md)); it now refuses one.
  `PAD_1D_REFLECT` on an input no longer than its pad aborted the process inside ggml; it now throws (EnCodec
  on under 4 frames). Three CI tests (`test_gru_sweep`, `test_spm_prefix_vocab`, `test_transcribe_wordpiece`)
  ended in `return 0` and passed with failed checks; they now report them.
* Every file needs a re-export (rc21+). An older engine ignores the `carry` argument, and EnCodec's stream
  then gives wrong audio after its first push. Its `infer` is unaffected.

## Related

* `src/core/lua_bridge.cpp` (`run_recurrent_impl`, `cell_sweep`, `l_set_state`), `src/core/speech_stream.cpp`,
  `src/ops/primitives_basic.cpp` (`op_pad_1d_reflect`); `tests/ci/test_gru_sweep.cpp`,
  `tests/ci/test_speech_stream.cpp`, `tests/ci/test_pad_reflect_lowering.cpp`; gates `test_e2e_codec_stream`,
  `test_e2e_codec_pair_stream` (T5 text for MusicGen)
* `loom-exporter/loom_exporter/codec_stream_driver/00_window_stream.lua`, `01_window_one_pass.lua`,
  `encodec_driver/00_stream.lua`, `musicgen_driver/00_stream.lua`, `codec_stream_export.py`, `encodec_export.py`
