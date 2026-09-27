---
type: adr
status: accepted
date: 2026-09-25
tags: [exporter, model-coverage, family-10, remote-code, verification, moss]
supersedes: []
---

# ADR-051: A Remote-Code Stack of a Known Architecture Is Re-Hosted in Transformers' Class, and Asserted

## Context

MOSS-TTS-Local-Transformer-v1.5 ships as Hugging Face remote code. Its 36-layer backbone is
`MossQwen3Model`, the authors' own implementation. Its attention builds boolean masks from shapes
(`arange` over the query and key lengths), and it chooses between flash, SDPA and eager paths. None of
that is the spelling `passes.fuse_loom_attention` recognises. That pass is what turns a traced
attention block into a KV-cached `ATTENTION` node with GQA.

The arithmetic, however, is Qwen3's exactly: q/k RMSNorm before rotate-half RoPE at θ = 1e6, SwiGLU,
pre-norm, and the same parameter names. And transformers 4.57 has its own `Qwen3Model`, which
`causal_lm_export` already traces, fuses and caches for Qwen3-0.6B.

## Options

1. **Patch the remote code for the trace**, as `qwen3_tts_export.install_patches` does for Qwen3-TTS:
   class-level rewrites until the attention matches the fusion window. Every patch is a chance to be
   subtly wrong, and Qwen3-TTS needed four, plus materialising GQA into MHA (+277 MB).
2. **Re-spell the stack by hand**, as Pocket-TTS's Mimi decoder was. That is fine for 2 layers, and a
   second implementation of a 36-layer LM for no gain.
3. **Load the weights into `transformers.Qwen3Model`** (`load_state_dict(strict=True, assign=True)`:
   same names, same tensors, no copy) and trace that.

## Decision

**Option 3, with the equivalence ASSERTED at export time.** `moss_tts_export.check_global_equivalence`
runs both stacks on the same random prefix before anything is traced, and refuses the export past a
tolerance. It is cheap even at 4B (six positions), so it runs on the real weights on every export,
rather than once on a developer's machine.

It measured **0.0**: bit-identical, on the tiny test checkpoint and on the real 4.55B one. The fused,
cached, GQA-native attention then comes for free. The KV cache stays at 8 K/V heads, with no
materialisation.

## Consequences

* **This is the rule for the next one.** When a remote-code module is a known architecture, re-host
  it in transformers' own class and assert the equivalence. Patch or re-spell only what has no
  transformers counterpart; here that is the one-layer local GPT-2 with RoPE, re-spelled in
  `_LocalWrapper` and pinned against the reference's arithmetic in CI.
* The assertion makes a wrong premise loud. If a future MOSS release changes the backbone (a
  different norm placement, say), the export refuses instead of producing a plausible model.
* Verified end to end: greedy codes are identical to the reference's `generate` at 38/38 and 40/40
  frames, and sampled codes with pinned draws at 34/34 and 60/60, including the frame where both
  decide to stop.

## Related

[Epic-03](../epics/epic-03-model-coverage.md) family 10,
[ADR-003](adr-003-per-model-complexity-in-the-exporter.md),
[ADR-049](adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md) (its codec).
