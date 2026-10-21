---
type: retro
date: 2026-10-16
domain: export
tags: [exporter, quantization, keep-float, multi-phase, silero, q8_0_pc, refactor]
---

# Retro-088: A Family Setting the Phase Packers Never Saw

## The Issue

ADR-079 P4.33's first Silero VAD `Q8_0_PC` export moved jfk's speech probability by up to 0.16 and flipped
frames at both decision thresholds. A bisection on x86 (re-exporting with chosen weights kept at F32) found
the front end and the head responsible and the LSTM harmless. The fix looked like one line: the family's
`backend_kwargs()` gaining `keep_float`, as WakeHuBERT's already does for its DFT basis.

The export then reported "kept F32 by the family: frontend.model_stft_re" **and wrote that tensor as
Q8_0.** Nothing raised, and the summary line said the opposite of what the file held.

## Root Cause Analysis

**P5.0 moved packing into the phases and left `keep_float` where packing used to be.** A multi-phase export
used to quantize once, in the output exporter's `write_gguf`, which receives `backend_kwargs()`. P5.0 made
each phase pack its own weights as it converts (to bound peak memory), so `convert_phase` builds its own
`LoomGGUFExporter(None, quantize=quantize)`. `quantize` was threaded through to it, and through the
isolated worker's argv. `keep_float` was not. By the time the output exporter ran, every weight was in
`packing.raw_dtypes` already, so its `pack_weights` returned early, before both the `keep_float` rule and
its own check that every listed name exists.

**The summary line read the configuration, not the file.** "kept F32 by the family: ..." prints
`self.keep_float` from the output exporter, which did have the setting, so the report stayed right while
the behaviour was wrong.

**No family had hit it.** WakeHuBERT, the one `keep_float` user, is a single-topology export, so its packing
still happens in `write_gguf`. Its published cards keep their mel basis F32 (checked: `model_mel_basis`
is F32 in the Q8_0 and Q4_0 files). Silero was the first multi-phase family to need it.

## Takeaways

* **When a refactor moves where work happens, move every input of that work.** The P5.0 diff threaded
  `quantize`, the setting it was about, to the new site. The other settings the old site read are the ones
  to list and check.
* **A report should be computed from the artifact.** The summary line now lists only the names the
  packing record says were written F32. The test added here (`test_a_family_keep_float_reaches_every_phase`,
  in-process and isolated) reads the GGUF back, and it fails with the threading removed.
* **A typo check that runs after the work is done checks nothing.** The "names a weight this export does
  not write" guard sat behind the same early return. `MultiPhase` now checks the names against what the
  phases held.
