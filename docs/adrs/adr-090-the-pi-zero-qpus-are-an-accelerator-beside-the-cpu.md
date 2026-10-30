---
type: adr
status: accepted
date: 2026-10-28
domain: backends
tags: [vc4, videocore-iv, qpu, raspberry-pi-zero, armv6, accel, mailbox, device-selection]
---

# ADR-090: The Pi Zero's QPUs Are an Accelerator Beside the CPU

P4.44 in [Epic-04](../epics/epic-04-backends-and-accelerators.md). Code: `backends/ggml-vc4/`. The
hardware trap found building it is [Retro-095](../retros/retro-095-an-emulator-that-allows-what-the-hardware-breaks.md);
what validating it on the shipped models found is [Retro-096](../retros/retro-096-an-accelerator-primary-rewrote-the-graphs-it-did-not-touch.md).
The Pi 4's GPU went the other way (ADR-089: V3DV is correct but slower than the CPU on every op, so it
is opt-in).

## Context

ARMv6 is the engine's floor (ADR-026) and its one core is the slowest target loom ships. The same
chip has a VideoCore IV with 12 QPUs: 16-wide SIMD at 300 MHz, 28.8 GFLOPS of dual-issued fmul+fadd
(no FMA), 26.2 measured on the board. No ggml backend drives it: Mesa's
vc4 driver is GLES2 only (no compute, no Vulkan), so the route is the firmware mailbox
(`execute_qpu`), with GPU memory from `/dev/vcsm-cma`.

Measured on a Pi Zero W (2026-10-27/28): a launch costs 130-150 us; the backend's F32 MUL_MAT runs
VITS-sized shapes 6.8-8.9x faster than ggml's ARMv6 CPU kernel through `ggml_backend_sched`. Under
loom's `"auto"`, Pocket-TTS 7m F32 offloads 144 MUL_MATs (7.55 GFLOP) and stays deterministic, and
end to end it is a wash: its Mimi convolutions dominate and stay on the CPU.

## Decision

1. **`ggml-vc4` is an ACCEL device, used with the CPU, never alone** -- ggml's own definition of
   ACCEL ("used together with the CPU backend, e.g. BLAS or AMX"). It takes only the F32 MUL_MATs it
   wins (at least 2^18 flops; below that the launch costs more than the CPU's work) whose weights are
   leaves in its own buffer, and declines everything else.
2. **Its buffer type is host memory**, allocated from the GPU's contiguous pool: a dma-heap
   (`linux,cma`) buffer mapped write-back cached by the CPU and imported into vcsm-cma for its bus
   address. The CPU computes on the same tensors, so loom's ranking puts it at rank 1 (a host-memory
   offload device, like BLAS), `"auto"` picks VC4 + CPU with no engine change, and a split between
   them copies nothing. Coherence is explicit per launch: clean the weights and the packed
   activations, flush dst before and invalidate it after (the ARM1176's L1 is the only cache between
   the two; the QPUs use the 0x8 bus alias, coherent with the VideoCore L2).
3. **The kernel reads the weights in place** through the TMU, packs the activations k-major into the
   uniform stream, and writes dst directly (horizontal VPM, VDW depth = valid m), m-major tile order.
   Completion is the kernel's own done word per QPU, never `execute_qpu`'s return.
4. **It is part of this repo, built with ggml's `ggml_add_backend_library`**: `LOOM_GGML_VC4`, ON by
   default on 32-bit ARM Linux, a MODULE `libggml-vc4.so` beside the other backends under
   `GGML_BACKEND_DL` (so the linux_armv6l wheel ships it). ggml's registry knows only its own backend
   names, so `ensure_backends_loaded` loads it by path from every directory it sweeps, or registers
   it directly in a linked build.
5. **The room check covers rank 1.** The pool is 64 MB by default and VC4 reports it (CmaFree), so a
   model that does not fit stays on the CPU with the reason in `selection_note`, rather than failing
   to allocate. BLAS, also rank 1, reports no memory and is unaffected.
6. **The CPU's extra buffer types still apply beside it.** `GgufModel::load` used to repack only on a
   plain CPU primary, so the first `"auto"` that picked VC4 took `CPU_ARMV6_I8` away from every
   Q8_0_PC model on the Pi Zero -- the int8 kernels of ADR-079 -- while the QPUs declined every one of
   those MUL_MATs anyway. A hybrid whose primary is host memory now repacks on its CPU fallback: those
   buffer types are not host memory (every repacking one ggml defines says so), so neither VC4 nor
   BLAS can claim a repacked weight and the scheduler hands its nodes to the CPU. A primary with its
   own memory still repacks nothing.
7. **Lowering decisions answer for the CPU beside it, too.** `backend_can_run` -- which picks between a
   native op and a composition built for a GPU that lacks it (P4.7e) -- asked only the primary, so with
   VC4 primary every ATAN got the polynomial instead of libm, and VITS Q8_0_PC's folded convolutions
   became im2col + mul_mat: 19.7 s on the CPU, 112 s under `"auto"`, the QPUs taking none of it
   (profiled: the im2col MUL_MATs alone 93 s). A host-memory primary now also accepts what its CPU fallback runs,
   because a node it declines runs there on the same memory with no copy. BLAS as primary (the macOS
   base wheel's `"auto"`) gets the same rule and so the CPU's lowerings.

## Alternatives considered

* **A GPU-type device that runs whole graphs.** The QPUs have no general compute API to cover ggml's
  op set, and a device buffer would make every CPU/QPU switch a copy; an accelerator over host memory
  makes every op the CPU keeps free.
* **Mesa / DRM (`vc4-kms-v3d`).** No compute path exists there, and under KMS the firmware refuses
  `qpu_enable` and `execute_qpu` wedges it until a reboot. So the backend requires the legacy firmware
  graphics and reports no device whenever the vc4 DRM module is loaded -- refused before any call.
* **vcsm-cma's own `cached` allocation.** Its dmabuf maps uncached whatever the flag says (a
  read-modify-write pass measured at uncached speed), which would slow every op the CPU runs on those
  tensors. dma-heap memory imported into vcsm is cached, with range cache maintenance through
  `VC_SM_CMA_IOCTL_MEM_CLEAN_INVALID2`.
* **Weights repacked into the uniform stream** (`qpu_gemm.h`, the first kernel). Exact, but it needs
  a second copy of every weight in GPU memory and a pack per launch; reading them in place needs a
  tail of a few words after each leaf (`get_alloc_size`), which ggml allows only for leaves.
* **n-major tile order.** Four weight blocks per slice share one TMU cache: 3.6x slower.
* **Trusting `execute_qpu`'s return.** It returns early for 8 and 12 QPUs (5 of 8 runs) while the
  QPUs still run.

## Consequences

* On a Pi Zero / Pi 1-3 booted without `vc4-kms-v3d`, `"auto"` is VC4 + CPU; with KMS, or anywhere
  else, the backend registers with no device and nothing changes. `GGML_VC4_STATS=1` prints what the
  QPUs took and declined.
* A QPU program that does not finish leaves the QPUs stuck, and `qpu_enable(0)` then wedges the
  firmware; the backend aborts after 2 s and says a reboot is needed rather than attempt recovery.
  Every kernel loop is bounded. A soft reboot is not enough: with the QPUs stuck the board did not
  come back from `systemctl reboot` (2026-10-28), so it takes a power cycle.
* Decisions 6 and 7 also change BLAS-primary runs (macOS base wheel, `"auto"`): a quantized folded
  convolution now takes the CPU's direct packed sweep instead of im2col + a BLAS GEMM. Measured on an
  M1 Pro (2026-10-28, BLAS without Metal, main d063024 against this branch, ABBA, median of 10, 40
  phonemes): VITS Q4_0 under `"auto"` 104-106 ms -> 42.4-43.7 ms (2.45x), and main's `"auto"` had
  synthesised a different utterance -- 55 808 samples where the CPU gives 56 320 -- which this branch
  does not (56 320). VITS F32 under `"auto"` 45.8 -> 46.5 ms with identical output (no folded kernels
  there), and both models under `cpu` identical in output and time. Decision 6 places nothing on an
  M1: no per-channel int8 buffer type is registered on a core with DOTPROD.
* QPU float rounding is not IEEE (no denormals), so the board gate compares with a tolerance; the
  emulator tests are bit-exact against an fmul-then-fadd reference.
* Tests: `test_vc4_qpu_isa` (encoder against a dump cross-checked with py-videocore's assembler),
  `test_vc4_gemm_emu` and `test_vc4_gemm_wt_emu` (both kernels in an emulator that refuses the hazards
  the reference guide warns about), each with a legal-but-wrong sabotage; `test_vc4_backend` on a
  board (exits 77 elsewhere; `cpu-only` checks the harness and the registration in CI).
* Open (hub P4.44): Q8_0 / Q4_0 weights in the kernel (what the Pi Zero models ship as), convolutions
  as GEMM (the TMU gather can do the im2col addressing), and the tile order for large N (128x3656x384
  ran at 2.2 GFLOPS). The convolutions and per-channel Q8_0 were taken up by P4.45:
  [ADR-091](adr-091-the-qpus-take-convolutions-and-per-channel-int8.md).
