---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, gpu, cuda, performance, conv, stft, stream-state, nemo]
supersedes: []
---

# ADR-125: A Buffered Stream Runs on the GPU Without Host Round Trips

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). It speeds up the NeMo streams of
[ADR-110](adr-110-the-unified-asr-streams-the-way-nemos-buffered-streaming-does.md),
[ADR-122](adr-122-parakeet-streams-the-way-nemos-buffered-script-does.md),
[ADR-123](adr-123-a-nemo-ctc-model-streams-the-way-framebatchasr-does.md) and
[ADR-124](adr-124-canary-streams-the-way-nemos-aed-streaming-script-does.md) without changing what they
compute.

## Context

NeMo's buffered streams re-run the whole encoder over the buffer every step (up to 14 s for 2 s of new audio).
For an offline-trained model that is not optional: the mel is normalised over the buffer and the attention is
full, so no encoder state survives the window sliding. The user's answer was to keep upstream's algorithm and
make it fast on the GPU. Canary's 60 s stream ran at 11.5x real time on an RTX 5090. A bench split each push:

* **Pushes that ran no model were 38% of the time** (3.4 ms each). The drivers kept their audio as one
  stream-state array, and `loom.state`/`loom.set_state` moved all of it into Lua and back on every 100 ms
  push.
* **The encoder took 87 ms per 14 s buffer.** The profile gave 43% to `CONV_2D`: the conv module's two 1x1
  convolutions per layer (64 a pass), which ggml's CUDA direct convolution ran at 0.9 ms each.
* **The mel front end split the graph.** The STFT is a CPU FFT custom op (ADR-080), so the signal went to the
  host and both halves of the transform came back on every encoder call. It was a quarter of what remained.

## Options

1. **Change the stream**: carry encoder state, or a shorter left context. Not upstream's stream (see Context).
2. **TF32 for the F32 products**: faster, but ggml-0063 removed it because it broke tensor gates.
3. **Remove the three costs, each on its own**, without changing any result.

## Decision

Option **3**.

* **A pointwise convolution on a device is a matrix product.** In `op_conv_1d`, K = 1, stride 1, no padding
  or dilation, batch 1, an unfolded F32/F16 kernel, and a primary whose tensors are not host memory: the
  activation is transposed to `[IC, IL]` and multiplied by the kernel read as `[IC, OC]`. That gives `[IL, OC]`,
  the layout the next op reads. The CPU and host-memory accelerators keep the direct sweep that their gates
  and benchmarks are pinned against.
* **An STFT on a device is two products with a windowed DFT basis**, built in the graph: `im2col` frames the
  signal and `re = frames . (w cos)`, `im = -frames . (w sin)`. The phase `j*k mod n` is formed exactly in F32
  integers (exact while n² < 2^24; the op takes this path up to n_fft 4096), so the two transforms agree to a
  few ulps. Batch 1 only. The CPU keeps the FFT.
* **A stream-state array grows and is read in place**: `loom.append_state`, `loom.state_slice`,
  `loom.drop_state`, `loom.state_size`, all generic. The four NeMo stream drivers keep their audio this way:
  a push converts its own samples, and a step converts its window.

## Consequences

* **Every result is unchanged.** Re-exported with the new drivers, all seven NeMo streams pass their gates on
  the 5090 and on the workstation CPU:
  * Parakeet-TDT 883/883, RNN-T 514/514, the unified model 1677/1677;
  * Conformer-CTC small and both Citrinets 1763/1763 each;
  * Canary's 23 cases 883/883.
  
  The old-file sections were not run there, which is the count difference from the ADRs above. Canary's
  first CUDA run, with the lowerings and the old driver, was 883/883 too. The engine CI covers the bindings,
  including refusals and a one-pass call not seeing a stream's entries.
* **Speed** (RTX 5090; streams in 80 or 100 ms pushes):

  | stream | before (CUDA) | after (CUDA) | after (285K CPU) |
  |---|---|---|---|
  | Canary, 60 s, AlignAtt 10/2/2 | 11.5x real time | **52x** | -- |
  | Canary, 60 s, wait-k 10/2/2 | -- | **75x** | -- |
  | Canary gate, 457 s | 18.6x | **47x** | 5.3x |
  | Parakeet-TDT gate, 118 s | -- | **76x** | 6.1x |
  | Parakeet-RNNT gate, 93 s | -- | **71x** | 3.5x |
  | unified gate (0.56 s chunks), 128 s | -- | **29x** | 1.6x |
  | Conformer-CTC / Citrinet-1024 / -256-ls, 256 s | -- | **232x / 175x / 309x** | 23x / 12x / 24x |

  A Canary step now costs the 14 s encoder (20 ms, from 87) plus the decoder (1.04 ms a token). A push that
  runs no step costs 0.01 ms (from 3.4).
* **The one pass gains too**: every NeMo encoder's pointwise convolutions and front end now run on the GPU.
* Vulkan and Metal take the same lowerings (the conditions are about memory, not vendor) and were not
  measured.
* Publishing needs re-exports (the drivers call the new bindings) on an rc21+ engine, which the streams needed
  anyway.

## Related

* `src/ops/primitives_conv.cpp` (`op_conv_1d`, `op_stft`, `primary_is_device`), `src/core/lua_bridge.cpp`
  (`l_append_state` and the three beside it), `StreamState::value_for_write`, `tests/ci/test_transcribe_stream.cpp`
* `loom-exporter/loom_exporter/transducer_driver/11_buffered_stream.lua`, `nemo_ctc_driver/01_stream.lua`,
  `canary_driver/10_stream.lua`
* The three gates take `LOOM_TEST_DEVICE=gpu` now.
