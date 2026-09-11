---
type: adr
status: accepted
date: 2026-09-11
tags: [exporter, engine, model-coverage, family-11, contracts]
---

# ADR-029: A Multi-Rate Codec Keeps One Row Per Coarsest Frame

## Context

Family 11 takes `[1, n_frames, n_codebooks]` — codes frame-major, one row per codec frame, as many
columns as the codec has codebooks. That contract was written against DAC, where every codebook runs
at one rate, and [Epic-03](../epics/epic-03-model-coverage.md) named SNAC as the leaf that would test
it: `vq_strides = [4, 2, 1]` puts its three codebooks at three different frame rates, so codebook 0
emits one id where codebook 2 emits four. SNAC's own `decode` takes a **list of three tensors of
different lengths**, and there is no arrangement of those three into one rectangle that is also
"n_codebooks wide".

Two other things about this checkpoint had no precedent in the family:

* Its decoder is **stochastic**. `NoiseBlock` computes `x + randn(B, 1, T) * linear(x)` at four
  points, so the reference model returns a different waveform for the same codes on every call.
* Its `Snake1d` is a `@torch.jit.script` function whose body reshapes through `x.shape[i]`.

## Options

**The row.**

1. **One input per codebook** — three arrays, three lengths. Expressible: `GraphBuilder` resolves one
   dynamic-length symbol per topology, but `declared_axes` carries fixed multiples of it, which is
   exactly what Kokoro's vocoder phase does for four inputs at once.
2. **One row per FINEST frame**, with the coarse codebooks' ids repeated across the rows they span.
   Rectangular, one length, and every coarse id written four times.
3. **One row per COARSEST frame**, `sum(coarse // stride)` ids wide — 7 for SNAC.

**The noise.**

1. Leave it in the trace. A traced `randn` is a **constant** at the traced length: wrong at every
   other length, and the same "audio" forever.
2. Hoist it to four extra graph INPUTS, drawn by the DRIVER. This is the pattern every stochastic
   model here already uses — `loom.seed_rng`/`loom.gaussian_array`/`loom.uniform_array` on the Lua
   bridge, one shared `std::mt19937` — and the synthesized flow-matching driver already draws a
   dynamically-sized array from it. The four lengths are exact multiples of the root axis (32, 256,
   1024 and 2048 times `n_codes`), which is what `declared_axes` carries, as Kokoro's vocoder phase
   does for four inputs at once. The caller's contract is untouched: still one `codes` array in.
3. Add an RNG primitive *inside* a topology.
4. Drop the term, which leaves `E[output]` exactly, since the noise is zero-mean and enters linearly.

## Decision

**The row is one coarsest frame, `sum(coarse // stride)` ids wide, level-major** — codebook 0's id,
then codebook 1's ids for the sub-frames it spans in order, then codebook 2's. The wrapper slices the
row back into one tensor per codebook and lets the model's own `from_codes` repeat each level up to
the finest rate.

**The noise is kept, as four graph INPUTS the driver draws** (Option 2). `loom.seed_rng` seeds once
from `inputs.seed`, defaulting to a fixed 1234, and each stage's array is
`loom.gaussian_array(multiple * n_codes)`. A caller may hand the arrays in instead, which is what
keeps this family's oracle exact rather than distributional.

*This reverses the first version of this ADR, which dropped the term. The measurement that said it
was inaudible is below, and so is the listening test that overruled it.*

Two consequences are stated explicitly because they are what a caller reads:

* **`codec.n_codebooks` is CODE STREAMS PER FRAME, not the quantizer count.** It is 7 here for 3
  codebooks. The key keeps the meaning it was given and documented with — it is the pairing check
  between a family-10 LM and its codec, and loom-py's `test_codec_pair` uses it as the row width. A
  key that reported 3 would read true and break that pair.
* **`codec.frame_rate` is the rate of the ROWS** — 11.72 Hz for SNAC, the coarsest codebook's, not the
  codec's own 46.875.

## Why

**Option 3 is the one formula.** `sum(coarse // stride)` is `n_codebooks` exactly when every stride is
1, so a uniform codec is not a special case anywhere downstream: DAC's width, frame rate and driver
divisor all fall out of the same three lines, and `vq_strides` is `[1] * n_codebooks` for a codec that
does not have them. Option 1 hands a caller three arrays to keep in step and buys nothing — SNAC's
`decode` takes a list only because Python has lists. Option 2 sends four copies of every coarse id
across the boundary and makes the row width a lie in the other direction.

It is also **the layout the caller already has**: an AR LM over SNAC emits exactly these 7 ids per
step. (Which ORDER it emits them in is the LM's business — Orpheus interleaves depth-first — and
rearranging is the caller's job for the same reason the delay pattern is,
[ADR-020](adr-020-audio-codes-is-its-own-modality.md).)

**Option 1 is a silent wrong answer and Option 3 is ruled out on its own terms** — a topology is a pure dataflow graph
`GraphBuilder` builds once and reuses, so no node in it can produce fresh randomness, which is why
`topology_ops._op_random` raises and points at the host RNG instead. **So the choice was Option 2
against Option 4, and it was decided by ear after the numbers said the wrong thing.**

Every number said drop it. Removing the term moves the waveform **2.4% in relative RMS**, where two
decodes of the same codes under different seeds differ by **3.1%** — the deterministic decode sits
inside the reference model's own sample-to-sample spread, nearer the centre of it than any sample is.
The ASR oracle reads 22/22 either way. And Option 2 costs a driver component this family did not
have, four declared axes, ~1.6 noise draws per output sample across the Lua boundary, and a
**stochastic codec**, which every gate that compares two runs or two backends then has to seed first.

That is a fair price for an audible difference and a bad one for an inaudible one, so the measurement
decided it — and it has now been made at the level that can, since
what the noise plausibly buys is breath on unvoiced sound, which neither relative RMS nor a word-level
transcript can see.

**The measurement.** Both arms reconstruct the same source, so the SOURCE is the reference. Native
24 kHz material (no resampling): 6 s of speech and, as a second content type, 6 s of music. The mean
arm against five seeds of the noisy one, compared on the mean band energy of the frames with the
largest share of energy above 4 kHz — the noisiest frames of each clip, ranked rather than
threshold-picked.

| band | 0-1k | 1-2k | 2-4k | 4-6k | 6-8k | 8-10k | 10-12k |
|---|---|---|---|---|---|---|---|
| **noisy − mean**, speech, noisiest frames (dB) | 0.01 | 0.01 | 0.02 | 0.19 | 0.19 | −0.02 | 0.12 |
| **noisy − mean**, speech, tonal frames (dB) | −0.00 | 0.08 | 0.07 | 0.09 | −0.27 | 0.15 | **1.75** |
| source band level, dB re: the clip's own full-band energy | +10.4 | −2.1 | −10.3 | −16.1 | −19.4 | −22.1 | **−21.5** |

Every band where the two arms differ by more than 0.2 dB is more than 20 dB down. The one real gap is
the top octave on voiced frames — 1.75 dB at 10–12 kHz, which is breath, exactly where the noise was
predicted to matter — in a band carrying under 1% of the clip's energy. For scale, the two arms'
pointwise log-spectral distance to the source is 9.98 dB (mean) against 9.81 dB (noisy, five-seed
spread 0.01 dB), so the term is worth about 1.7% of the gap the codec itself leaves.

**The listening test disagreed, and it wins.** Given the source and both decodes blind, a listener
reported the deterministic one as *"less sharp, slightly more artificial"* — unprompted, and in the
direction the table predicted but at a size the table dismissed. 1.75 dB in the top octave is not
0.02 dB of broadband level; it is the presence of breath, which is a timbre cue rather than an energy
one, and reading "21 dB down" as "inaudible" was the error. A dB table cannot rank two waveforms for
naturalness, and this one was never asked to — it was asked whether anything was there to hear, it
said *a little, up high*, and that turned out to be enough.

**The general lesson is [Retro-043](../retros/retro-043-the-band-was-20db-down-and-audible.md).** For
anything whose output a person listens to, a spectral measurement scopes the question and a listener
answers it. The same finding one family over is Retro-006, where Kokoro matched PyTorch at cosine
0.996 and shipped unintelligible.

## Consequences

* Family 11's contract is unchanged for DAC and for any future uniform codec — the same GGUF keys,
  the same driver, the same `Len(codes) / width` divisor.
* The stride list itself is not written to the file. `hparams()` writes scalars a host reads with
  `hparam_u32`/`hparam_f32` and `write_gguf` refuses a list by design; what the strides describe is
  the order of the columns within a row, which is documentation rather than a number. The model card
  carries it.
* **`DriverInputs` gains a third host-computed binding kind**, `NOISE`, beside `POSITION` and `MASK`
  — and unlike those two it is named by the EXPORT rather than by a set of input names, because the
  per-input length ratio is not recoverable from a name. Any future family with a stochastic leaf gets
  it for free.
* **This codec is stochastic, and therefore seeded.** `inputs.seed` defaults to 1234 so two runs of
  one file agree and a gate can compare them; a caller who wants variation passes a seed. A caller may
  also pass the noise arrays themselves, which is how the oracle stays exact — both sides get the same
  draw and the comparison is `max|Δ| 1.2e-06` rather than a distribution test.
* The noise costs ~1.6 draws per output sample across the Lua boundary. Measured at export: no
  material change in decode time against the deterministic build.
* DAC's export does not move: the multiples are walked off the real decoder's own `NoiseBlock`s, a
  codec with none declares empty, and empty emits exactly the driver it always emitted.
* A caller cannot ask this file for a sub-coarse-frame number of frames. 11.72 Hz is the resolution of
  the contract; SNAC's own encoder pads to a multiple of `vq_strides[0]` for the same reason.
* `CodecFamily.decode` now takes the CALLER's layout and each member adapts it — DAC's transpose moved
  out of the wrapper. The adaptation was never shared; two leaves made it look that way.
* Snake is patched to the same arithmetic without its reshapes (see
  [Retro-042](../retros/retro-042-a-converters-op-default-changed-the-function.md) for the other patch
  this export needed, which was not SNAC's doing).
