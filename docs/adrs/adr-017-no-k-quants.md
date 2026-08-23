---
type: adr
status: accepted
date: 2026-08-20
tags: [quantization, gguf, tooling, scope]
---

# ADR-017: K-Quants Are Out of Scope for This Toolchain

## Context

The obvious request for a size-constrained edge engine is `Q4_K_M`, the quantization most llama.cpp
users know by name.

## Options Considered

1. **Support K-quants.** Requires writing quantization code the Python GGUF writer does not have.
2. **Stay with the block-32 types the writer can produce.**

## Decision

**Do not spend time on K-quants.**

* `Q4_K_M` is **not a tensor type at all** — it is a llama.cpp mixed-precision *recipe*.
* The real type `Q4_K` exists, but `gguf.quants` raises `NotImplementedError` for **every** K-quant
  (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), so this toolchain cannot write one.
* K-quants use block **256**, where only 9 of 132 VITS convolution kernels would align even after the
  layout fix that makes block-32 quantization possible at all.

Writable today: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ1_0, TQ2_0.
`main_export.quantize_choices()` derives the offered list by **probing the writer**, for exactly this
reason — so the CLI cannot offer something that will fail at write time.

## Consequences

* **Positive:** the offered list is always true, and no effort goes into a format the writer cannot
  produce.
* **Negative:** loom cannot match a llama.cpp size/quality point users may ask for by name. The honest
  answer is Q4_0 or Q8_0.
* **Revisit when:** `gguf.quants` grows K-quant support upstream. The block-256 alignment finding would
  still need re-checking against the model in question.

## Related

* Epic: [Epic-05: Edge Performance](../epics/epic-05-edge-performance.md)
