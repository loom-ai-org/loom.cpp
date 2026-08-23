---
type: retro
date: 2026-08-14
domain: backends
tags: [device-selection, gpu, vulkan, defaults]
---

# Retro-007: `"gpu"` Chose the iGPU Over an RTX 5090

## The Issue

P4.8e left ties *within* a device rank falling back to registration order, with the note that "a caller
with two such devices should name the one it wants". Installing the Vulkan wheel on a workstation
produced the first machine that actually had the tie, and the outcome was worse than that wording
suggests: it is not an arbitrary pick between equals, it is **reliably the wrong one**.

## Root Cause Analysis

Registration order is not a preference ordering. On a machine with both an Intel integrated GPU and a
discrete RTX 5090, the integrated device registers first.

## Resolution & Lesson Learned

Within a rank, discrete beats integrated — derived at run time from what the device reports about
itself rather than from a name the engine knows. See
[ADR-010](../adrs/adr-010-device-selection-by-kind.md).

**Actionable takeaway:** "ties fall back to arbitrary order, callers should be explicit" is only
acceptable when the arbitrary order is actually arbitrary. Check what the fallback does on real
hardware before documenting it as a caller's problem.

---

## Full record (verbatim from the ledger)


P4.8e left one thing open and named it: ties WITHIN a rank fall back to registration order, "and a
caller with two such devices should name the one it wants". Installing the Vulkan wheel on the
workstation produced the first machine that actually has the tie, and the outcome is worse than the
wording suggests -- it is not an arbitrary pick between equals, it is reliably the wrong one.

    Vulkan0 | Intel(R) Graphics (ARL)
    Vulkan1 | NVIDIA GeForce RTX 5090

`auto` and `gpu` both resolved to **Vulkan0**. The cost is legible without reading a device name:
`Vulkan1` generated 24 tokens in 8.35 s, while `auto` needed roughly 13 s for **eight**.

### Why neither existing signal separated them

`probe_tiers`, on the two-device Vulkan build:

    device     type   buft_is_host   host_buffer  device_id       kernel says
    Vulkan0    IGPU   false          true         0000:00:02.0    0x030000  <- display controller
    Vulkan1    GPU    false          true         0000:02:00.0    0x030000  <- display controller

Both are non-host, so P4.8e's rank ties. Both are PCI class 0x03 -- they are both genuinely display
controllers -- so P4.8h's kernel confirmation ties too. Registration order then decided, and it put the
integrated one first.

**The information needed was there and P4.8e had thrown it away.** That entry collapsed the tiers by
keying rank on where memory lives and dropping `ggml_backend_dev_type` entirely. Right for deciding
ranks -- the type is what misled rank 1 into existing at all -- and wrong for ordering within one,
where GPU-versus-IGPU is exactly the distinction wanted.

### The fix, and why the order of its two parts is load-bearing

The tie-break key becomes a pair: **kernel-confirmed first, discrete-before-integrated second.**

Putting the type first would reintroduce the defect the kernel check exists to prevent. `ggml-openvino`
reports `GPU` while driving an NPU or a CPU, and supplies no `device_id` (P4.8d) -- so a type-first key
would rank it above a genuine, kernel-confirmed iGPU. Confirmation first sorts it to `(1, 0)`, behind
anything the kernel vouches for, and this machine's two GPUs to `(0, 0)` and `(0, 1)`.

That composition is also why the check below matters more than it did last time: a wrong ORDERING of
the two parts still produces the right answer on this machine, and only goes wrong on a machine that
also has OpenVINO installed. Getting the observable case right is not evidence that the key is right.

### Verified on the hardware that exhibits it, including made to fail

The base wheel rebuilt through cibuildwheel and installed with `rt-vulkan` into a clean venv, 16 tokens
each:

| spec | correct ordering | ordering INVERTED |
|---|---|---|
| `auto` | **0.88 s** | 13.01 s |
| `gpu` | **0.87 s** | 12.47 s |
| `Vulkan0` (Intel iGPU) | 12.45 s | — |
| `Vulkan1` (RTX 5090) | 1.11 s | — |

The timings identify the selection without reading a device name, and the inverted build is what makes
the first column mean something: flipping discrete-and-integrated sends `auto` back to the iGPU, so the
comparison demonstrably drives the choice rather than agreeing with registration order by coincidence.
That coincidence is exactly what P4.8h nearly shipped, where `auto -> CUDA0` looked like proof.

**What this did NOT verify, and the entry should not be read as claiming it:** that the two parts are
in the right ORDER. Both orderings select the 5090 here, because nothing on this machine claims to be a
GPU without being one. Only a box with OpenVINO installed beside a real iGPU would separate them, and
there is none. The confirmation-before-type argument stays reasoned from `ggml-openvino` reporting
`GPU` with no `device_id`, not measured.

