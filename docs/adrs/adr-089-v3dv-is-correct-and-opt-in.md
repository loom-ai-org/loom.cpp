---
type: adr
status: accepted
date: 2026-10-28
domain: backends
tags: [vulkan, v3dv, broadcom, raspberry-pi-4, workgroup-size, specialization-constants, device-selection]
---

# ADR-089: The Raspberry Pi 4's GPU Runs Correctly and Is Opt-In

P4.43 in [Epic-04](../epics/epic-04-backends-and-accelerators.md#7-p443-v3dv-on-the-raspberry-pi-4-2026-10-28).
Built as `ggml-0054` and `ggml-0055` (`cmake/patches/UPSTREAM.md` PRs 50 and 51). The lesson from the
measurement is [Retro-094](../retros/retro-094-a-speedup-measured-against-a-run-with-different-output.md).

## Context

Mesa's V3DV now exposes the Pi 4's V3D 4.2 as a Vulkan 1.3 device, and the `loom-py-rt-vulkan` aarch64
wheel registers it. Out of the box it could not run a graph: device init threw on 16 KB of shared memory
(an IQ type's matmul tile), 32 shaders asked for 512 invocations where V3D allows 256 -- invalid usage
that Mesa compiles into wrong results -- and Mesa's compiler never finished the F32 matrix-vector shader.
With those fixed it is correct, and measured on the board against its four A72s it is far slower:

| Pi 4, V3DV opted in | GPU | CPU (4 threads) | output |
|---|---|---|---|
| VITS (piper) F32 synthesis | 64.9 s | 0.836 s | same audio, max abs diff 1.0e-5 |
| Qwen3-0.6B Q8_0_PC, 24 tokens | 509 s | 4.18 s | same 24 tokens |
| Qwen3-0.6B F32 | does not fit (2.38 GB of weights, 1.99 GB device) | 16.6 s | |

CONV_2D is 87% of the VITS GPU time at 0.30 GFLOPS with register spills; ~1100 elementwise dispatches
cost ~2.5 ms each; Q8_0 matrix-vector work is 99% of Qwen3's. V3D's peak is 32 GFLOPS (8 QPUs x 8
flops/cycle x 500 MHz, no FMA), a third of the A72s', which already sustain over 20 GFLOPS on VITS.

loom ranks a device with its own buffer type (rank 0) above the CPU, so a working V3DV would be what
`"auto"` picks on every Pi 4 with the Vulkan wheel installed.

## Decision

1. **Make V3DV correct** (`ggml-0055`): the thirty-two shaders' workgroup size becomes a
   specialization constant whose default is the old literal, and one function fits it to the device
   when the pipeline is initialised, reading the SpecIds and defaults from the SPIR-V, fitting every
   spec-sized dimension (outer ones first) and scaling the dispatch denominators with them.
   No device that allows 512 invocations sees a change. `V3D_DEBUG=noloopunroll` for Mesa's compiler.
2. **Leave V3DV out of ggml-vulkan's default device list**, beside the existing "unsupported" and
   "CPU-type" exclusions; `GGML_VK_VISIBLE_DEVICES` opts in. On a Pi 4, `"auto"` is then the CPU and
   `"gpu"` says there is none, which is true of every op measured.

## Alternatives considered

* **Rank it below the CPU in loom.** ggml has no property for "slower than the host", so this needs a
  device-name list in `backend.cpp`, which ranks by what a device is, never by name (ADR-010).
* **A CPU+V3DV hybrid**, V3DV as an assist the way host-memory accelerators are (user, 2026-10-28). It
  buys nothing: no op measured is faster on V3D, its buffers are not host memory so every switch costs a
  copy plus a ~ms submission, and the weights of GPU ops would have to be placed per op. The only real
  use, a second model run concurrently on the GPU, is a deliberate choice -- the opt-in.
* **Fit each pipeline at its call site.** Hundreds of generated variants share the thirty files; one
  SPIR-V-driven function cannot miss one and covers future shaders and `soft_max`'s `wg512` variant.
* **A V3D conv2d tile** (256 invocations, `TS_K = 2`, 64x32): spills halved, VITS 1.5% slower,
  interleaved A/B. Measured out.

## Consequences

* A Pi 4 with `loom-py-rt-vulkan` keeps running on the CPU, as it effectively did (rc19's V3DV threw
  at init); `GGML_VK_VISIBLE_DEVICES=0` plus `device="Vulkan0"` runs a model on the GPU, correctly.
* Three traps worth knowing for any shader change: a bare `local_size_x_id` defaults to 1, not to a
  constant declared with the same id; a dispatch that counts workgroups (SET_ROWS) must follow the
  fitted size too -- the KV-cache writes were wrong, and Qwen3's tokens with them, until it did; and a
  dispatch may read `wg_denoms` before requesting the pipeline (FILL), so they are fitted at
  initialisation. Forcing the fit on a GPU that does not need it (RADV, limit pretended 256, scratch)
  is what separated these from V3D's own failures.
* A fitter that skips what it does not handle hides it: the first version fitted only one-dimensional
  workgroups and passed over `ssm_conv`'s 32x16 without a word, so SSM_CONV failed all 45 cases on V3D
  until the every-op sweep found it. Every spec-sized dimension is fitted now (32x8), and a workgroup
  still over the limit is logged.
* Unchanged where 512 fits: on an AMD Vega 3 (RADV) every op passes, 17 490 cases, and the
  device-parity gates run 2 069 nodes on it with no fallback.
* Open on V3D, from the every-op sweep (Pi 4, 2026-10-28), none on the VITS or Qwen3 path:
  * `ARGSORT` from 1 023 columns, its large path: 60 of 98 cases.
  * The fused L2_NORM inside `GATED_DELTA_NET`: 2 of 36, 1.9e-6 against a 1e-7 tolerance.
  * `MUL_MAT_ID` over tolerance in 36 cases, every one at n = 129 (m = 512, k = 256) and of a
    32-element block type: q2_0 7, q4_0 8, q8_0 4, mxfp4 9, nvfp4 8 of 12 each, NMSE 5.6e-4 to 4.3e-3
    against 5e-4. Every other shape passes, as do f32, f16, q1_0, q4_1, q4_K and iq2_xxs at n = 129,
    and the same cases pass on RADV with the fit forced -- so it is V3D's arithmetic or Mesa's, not the
    workgroup fit.
  * Not covered: `MUL_MAT` (255 cases, none failing) and `MUL_MAT_ID` (887 passing) hit the sweep's
    30-minute limit per op before their last cases.
