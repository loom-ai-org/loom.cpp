---
type: adr
status: accepted
date: 2026-11-13
tags: [streaming, live-api, stream-state, tts, audio-out, chatterbox, flow-matching, hift]
supersedes: []
---

# ADR-117: Chatterbox Streams a Hop at a Time by Re-Running Its Full-Attention Flow

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Puts Chatterbox behind
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door, with
[ADR-112](adr-112-cosyvoice3-streams-the-way-its-reference-does-by-re-running-the-flow-a-hop-at-a-time.md)'s
re-run cut.

## Context

Chatterbox is T3 (a guided Llama-520M drawing 25 Hz speech tokens), then S3Gen: a conformer and a 10-step
guided ODE over a U-Net in-fill a mel after the voice's prompt frames, and HiFT vocodes it. S3Gen is
CosyVoice2's flow and HiFT, and keeps CosyVoice2's streaming hook: `flow_inference(finalize=False)` runs
the encoder over every token so far and drops the mel of the last 3, which the pre-lookahead conv reads as
context. Chatterbox ships no loop around it; the `S3GenStreamer` a docstring names is not in the release.
On the dev box an 8 s sentence takes about 2 minutes, all of it before the first sample.

Two things differ from CosyVoice3 (ADR-112):

* **No streaming mode.** Both the encoder and the U-Net attend over the whole sequence (`static_chunk_size`
  0), so a hop's mel is what the flow makes without the tokens after it, and no one pass equals a
  re-run loop. Measured in PyTorch before choosing (200 tokens, 400 frames, guided greedy, one noise
  table sliced from the start): the loop's mel is a mean 0.085 from the one pass (max 2.6, at the first
  hop's start), while the one pass under another noise table is 0.378 from it (max 4.8). At f64 both runs
  are within 1.4e-05 of f32, so the distance is the attention's, not rounding. On the waveforms (a scoping
  run, another table) the loop was 1.59 dB of log-mel distance from the one pass and another draw 3.40 dB,
  and Whisper-small transcribed all three exactly.
* **The hook does not run as shipped.** `CausalMaskedDiffWithXvec.inference` trims `h` for `finalize=False`
  but builds the decoder's mask from the untrimmed lengths, and `solve_euler` fails (364 against 370
  frames). The mask is all ones for one utterance; CosyVoice2 builds it from `h`, and so does the oracle.

## Options

1. **Sentence by sentence, cut by the host** (ADR-100's option 3): what Chatterbox had. A sentence's
   latency, and prosody restarts at each one.
2. **Chunk the flow with a mask it was never run with** (CosyVoice2's chunk mask). The U-Net may have
   been trained with one, but the release runs it unmasked, so nothing upstream could grade it.
3. **CosyVoice2's re-run loop over Chatterbox's hook**: each hop the flow over every token so far, the new
   mel kept, HiFT over all of it. Driver work only, its PyTorch twin is the oracle, and the distance from
   the one pass is measured, not assumed.

## Decision

**3.** A driver change, and one engine fix its gate found (Consequences).

* `stream_feed` is one hop: T3 draws until the hop and 3 look-ahead tokens are there (or it stops), the flow
  runs over the prompt and every token so far with the look-ahead's rows dropped (on the last hop, all of
  them), and the new mel is appended. The hop is CosyVoice2Model's (25 tokens, doubling to 100), read off
  its source by a CI test. CosyVoice2's first-hop pad onto its chunk boundary is dropped: there is no chunk
  mask here, so it would only delay the first piece. `piece_tokens` fixes the hop; `0` is the whole turn
  in one hop, which `infer` runs, so the one-pass gate grades the stream's code.
* HiFT re-runs over all the mel and hands out all but the last `VOCODER_RIGHT_FRAMES` frames' samples.
  HiFT is not causal, and its reach is counted at export: the vocoder over a mel whose tail is NaN, and
  the first NaN sample. That is the graph's structural reach, 19.25 frames, so 20. A value probe (two mels
  differing in the tail) measures the data's instead: about 13 frames when F0 is under the voicing
  threshold and the source ignores it, about 14 to 1e-12 on a voiced sentence, where samples up to 19
  frames back move by 1e-15. A CI test on a random, voiced HiFT at f64 has the 20-frame cut bit-identical
  to one call and a 19-frame one not.
* T3 is cut per step: the previous draw is fed back before the next is drawn. The draws follow ADR-112's
  amendment: T3's uniforms are one block (one per step of the budget), then the harmonic phases, then a
  frame at a time the flow's noise and the sine source's. So a seed draws the same tokens both ways, and
  every re-run starts from the same noise.

## Consequences

* Stream gate `test_e2e_chatterbox_stream` 49/49 on the dev-box CPU (guided greedy, 200 tokens, the
  draws pinned to the oracle's):
  * The one pass's mel is 1.8e-05 from PyTorch's; the stream's, in 4 hops of 50, 100, 200 and 50 frames,
    is 1.3e-05 from the same loop in PyTorch, and a mean 0.085 from the one pass against 0.378 for another
    noise draw (checked: nearer).
  * The joined wave has the one pass's 192,000 samples in 4 pieces; `infer` with `piece_tokens = -1` is
    the stream bit for bit; HiFT run once over the stream's mel is within 3.2e-06 of the join.
  * Each stream after a reset, and interleaved with a sampled stream and a one-pass call under both cache
    modes, is bit-identical to itself. A seed alone gives the same tokens both ways (176 frames). The v5
    file is refused.
  * The first 0.6 s is ready after 38.6 s, against 134 s for the one pass (8 s of audio, 0.06x real time,
    RTF 16.8); the whole stream takes 254 s, since every hop re-runs the flow and HiFT over everything so
    far, as CosyVoice2's loop does.
* Sabotage: an in-place same-length byte patch `VOCODER_RIGHT_FRAMES = 20` -> `02` (each hop hands out
  samples the next one rewrites) grows the first piece from 14,400 to 23,040 samples, puts the join 0.476
  off one vocoder call over the same mel, and fails exactly that check (48/49); the mel checks still pass,
  the patch touching only the vocoder's output. Reverted, sha256 verified.
* Workstation (2026-11-13; both gates, the same re-export, sha256-checked copies; the old file refused):
  * CPU, 24 cores: one-pass gate 45/45 (2.77e-05); stream gate 49/49, the mel 1.8e-05 from the PyTorch
    loop, HiFT streamed 8.3e-06 from one call. The first 0.6 s after 4.8 s against 20.2 s for the one pass
    (8 s of audio, 0.40x real time, RTF 2.5); the whole stream 29.9 s (RTF 3.7).
  * CUDA, RTX 5090 (`LOOM_TEST_DEVICE=gpu`, which the one-pass gate now honours): one-pass gate 45/45
    (2.8e-04 against its 2e-3 bound); stream gate 49/49, the mel 7.4e-05 from the PyTorch loop, HiFT
    streamed bit-identical to one call, both cache modes interleaving bit-identically. The one pass takes
    1.8 s, 4.4x real time (RTF 0.225); the first piece is ready after 0.6 s and the whole stream after
    3.6 s, 2.2x real time (RTF 0.45). Since [ADR-118](adr-118-chatterbox-runs-each-guidance-pair-as-one-batch-of-two.md)
    (both guidance pairs one batch of two): the one pass 1.3 s, the first piece 0.5 s, the stream 2.7 s; on the
    CPU at 16 threads the one pass 16.3 s.
* One-pass gate `test_e2e_chatterbox_lua_driver` 45/45 on the re-export, 2.80e-05 from the reference
  (the v5 file gives 2.82e-05 on the same build).
* Its first run failed one check, 48/49: in park mode the sampled stream was not bit-identical to itself.
  The cause was the engine's, not the driver's: a graph built after a larger one of the same topology took
  that one's allocation plan, and the CPU conv fusion's overlap check made HiFT round two ways
  ([Retro-105](../retros/retro-105-a-graph-laid-out-by-the-graph-before-it.md)). `GraphBuilder` now plans
  every rebuild for its own graph; `test_graph_layout_is_its_own` holds it (12 of 22 nodes move without it).
* Re-export needed: the driver is in the GGUF. Publishing waits for an rc21+ engine like the other streamed
  re-exports, and this one needs the allocation fix too: on an older engine its interleaved streams can
  differ from themselves at 1e-07.

## Related

* `loom-exporter/loom_exporter/chatterbox_export.py` (`vocoder_right_frames`, `STREAM_*`),
  `chatterbox_driver/00_stream.lua`, `chatterbox_driver/01_generate.lua`, `tests/ci/test_chatterbox_export.py`
* `tests/gate/test_e2e_chatterbox_stream.cpp` (`LOOM_CHATTERBOX_STREAM_GGUF`, `_REF_DIR`, `_OLD_GGUF`), its
  oracle `scripts/chatterbox_stream_reference.py`, `tests/gate/test_e2e_chatterbox_lua_driver.cpp`
* `src/core/graph_builder.cpp` (the per-rebuild plan), `tests/ci/test_graph_layout_is_its_own.cpp`
* Upstream: `chatterbox/models/s3gen/flow.py` (`CausalMaskedDiffWithXvec.inference`), `s3gen.py`;
  `CosyVoice/cosyvoice/cli/model.py` (`CosyVoice2Model.tts`, `token2wav`)
