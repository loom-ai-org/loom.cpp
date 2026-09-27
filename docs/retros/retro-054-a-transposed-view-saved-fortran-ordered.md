---
type: retro
date: 2026-09-24
domain: testing
tags: [fixtures, npy, gate, family-9, chatterbox, layout]
---

# Retro-054: A Transposed View Saved Fortran-Ordered, and the Reader Read It Transposed

## The Issue

Chatterbox's gate failed with max |d| **1.17** and a peak clipped at 0.99. The same export, driven by a
scratch runner from the same reference, had matched it to **2.5e-05** an hour earlier. The waveform
had the right length, so T3's decode had agreed; the audio was wrong.

## Root Cause

The reference script now wrote the flow's initial noise frame-major, as the transpose
`draws["flow_noise"][0].T` of the channel-major draw. `np.asarray(...).astype(np.float32)` keeps a
view's memory layout, so the saved array was Fortran-contiguous, and `np.save` recorded
`'fortran_order': True` in the header. `tests/support/npy_fixture.h`'s `read_npy_f32` parsed only the
shape and read the bytes in C order. It returned an array of the right length and the right declared
shape, transposed. The ODE then started from a scrambled noise field: a valid draw, but not the
reference's.

The earlier scratch run had written the same transpose with `.tofile()`, which always writes C order.
That is why it passed.

## The Fix

* The script saves with `np.ascontiguousarray(..., dtype=np.float32)`.
* `read_npy_f32` **refuses** a Fortran-ordered file (`LOOM_CHECK` on the header). A transposed read
  cannot be caught by any size or shape check, so the reader has to reject it.
* An audit of `loom-engine-artifacts/v5` found one other Fortran-ordered fixture:
  `f5_tts_ref/mel.npy`. F5-TTS's gate does not read it.

## Takeaways

* **`astype` keeps layout, and `np.save` records it.** Any generator that saves `.T`, `transpose` or
  `swapaxes` output needs `ascontiguousarray`.
* **A minimal reader must reject what it does not implement.** "C-order only" was documented in the
  header's comment and enforced nowhere.
* **The gate failed on this before it ever passed**, which was the evidence that it can go red. The
  flow-guidance sabotage arm (0.897) is the deliberate version of the same check.
