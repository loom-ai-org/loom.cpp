---
type: retro
date: 2026-08-12
domain: exporter
tags: [tts, tokenizer, padding, masking, tracing]
---

# Retro-005: A Fixed Text Length Made Supertonic's Tokenizer Near-Unusable

## The Issue

Supertonic's export traced text at `T_TEXT_FIXED = 10`, so `infer` took exactly 10 `txt_ids` — and
`<en>` + the pipeline's inserted final period + `</en>` is already 10 ids for the **empty** string.
`"hello world"` encodes to 21. The vocabulary in the file was therefore good for encoding and
inspection while synthesis still effectively took raw ids. It shipped as a "Known limitations" section
on the model card rather than silently.

## Root Cause Analysis

Two layers. The surface cause is a traced constant. The deeper one is why simply raising it does not
work: the export built `txt_msk` as all-ones, so padding was not inert. The mechanism is
`ConvNextBlock`'s **replicate** pad, which on a masked tensor replicates a zero column where the
unpadded reference replicates the last real one — measured at **97% relative error** on `txt_emb`
before any of it was exported.

## Resolution & Lesson Learned

The axis is padded and traced at five widths up to 512, `txt_msk` became a real input, and the driver
picks a width and pads. Fixed by `_edge_fill`, with **no engine change**.

* **Actionable takeaway 1 — a traced constant is a contract.** Anything the trace pins becomes a limit
  the host cannot see. Trace the axis, don't pin it.
* **Actionable takeaway 2 — padding is only inert if every op respects the mask.** A replicate/reflect
  pad reads neighbouring values, so it reads padding. Verify against the unpadded reference at the
  tensor level before assuming a mask is enough. See
  [Retro-006](retro-006-kokoro-shipped-noise.md) for the sibling failure.

---

## Full record (verbatim from the ledger)


Found while wiring the above. `T_TEXT_FIXED = 10` (`supertonic_export.py`), so `infer` took exactly 10
`txt_ids` — and `<en>` + the pipeline's inserted final period + `</en>` is 10 ids for the EMPTY string.
Every real sentence overflowed: `"hello world"` encodes to 21. So the vocabulary in the file was good for
encoding and inspection, and synthesis still effectively took ids directly. Shipped as a "Known
limitations" section on the model card rather than silently.

**Done as P4.6/P4.6a** (Exporter section), which is where the measurements live. The axis is padded and
traced at five widths up to 512, `txt_msk` is a real input, and the driver picks a width and pads — so `infer` takes any count up to `txt_len`. The
correction the scoping carried was right (raising the constant *alone* is a trap, because the export
built `txt_msk` as all-ones) but named the wrong mechanism for *why* padding is not inert: it is
`ConvNextBlock`'s **replicate** pad, which on a masked tensor replicates a zero column where the
unpadded reference replicates the last real one. Measured at 97% relative error on `txt_emb` before any
of it was exported; fixed by `_edge_fill`, with no engine change.

