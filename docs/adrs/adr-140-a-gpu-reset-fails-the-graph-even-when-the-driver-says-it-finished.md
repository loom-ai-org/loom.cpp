---
type: adr
status: accepted
date: 2026-11-19
domain: backends
tags: [vulkan, v3dv, raspberry-pi, gpu-reset, device-lost, correctness]
supersedes: []
---

# ADR-140: A GPU Reset Fails the Graph, Even When the Driver Says It Finished

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md). Reported in
[loom.cpp#124](https://github.com/loom-ai-org/loom.cpp/issues/124). The device is the opt-in V3DV path of
[ADR-089](adr-089-v3dv-is-correct-and-opt-in.md).

## Context

On a Raspberry Pi 4 (V3D 4.2, Mesa v3dv), the kernel's hang check kills a compute job that never
finishes and resets the GPU ("Resetting GPU for hang"). Vulkan still reports success:
`vkWaitForFences` and `vkDeviceWaitIdle` return `VK_SUCCESS`, and the output buffers hold whatever was
in them. A loom call would hand back wrong audio or text with no error.

The reporter saw this on kernel 6.12.75 and Mesa 25.3.6, and traced it to two places. In the kernel, a
job whose dependency failed still runs, so the no-op job carrying the user's fence signals success. In
Mesa, `vk_drm_syncobj` never reads a signalled fence's error status. Reproduced here on the bench Pi 4
(kernel 6.18.34, Mesa 25.0.7) with a planted infinite loop: `VK_SUCCESS` after 1.04 s. Mesa main still
does not read the status.

ggml-vulkan already throws on `VK_ERROR_DEVICE_LOST` (`VK_CHECK`, and `getFenceStatus` in
`ggml_vk_wait_for_fence`). The exception reaches the caller as an error through the bridge. The driver
just never returns that error.

## Decision

**Ask the kernel instead of the driver.** `loom/core/gpu_reset_watch.h`:

1. `Device::open` arms the watch when the selected device's description names a V3D GPU. On any other
   device the watch stays unarmed, and the check costs one relaxed atomic load.
2. Armed, `GraphBuilder::compute` reads the reset count before and after every graph, and a count that
   moved throws `DeviceLostError` (a new `loom::Error`). Every graph runs through that call, and
   `ggml_backend_graph_compute` returns only after the GPU has finished, so the check covers each one.
3. The count is the kernel's `DRM_V3D_PARAM_GLOBAL_RESET_COUNTER` (Linux 6.18+), read with one ioctl on
   the v3d render node, which needs only the `render` group. On an older kernel, the fallback is the
   time of the last "Resetting GPU for hang" line in the kernel log (`klogctl`), which needs
   `dmesg_restrict=0` or the `adm` group. With neither, arming prints a warning that resets cannot be
   seen.
4. The counter is global. A reset caused by another process on the same GPU also fails the graph
   running here. That is the safe answer, because the reset may have taken this process's queued jobs
   with it, and the message says the call can be retried.

## Options not taken

* **The reporter's per-request kernel-log comparison** (about 2 ms a read). It works on any kernel,
  which is why it is the fallback. Per graph it would cost seconds on a model that runs 1 500 graphs
  (ADR-136's sweeps), and it needs log access, which the counter does not.
* **Patching the kernel and Mesa.** These are the right fixes (the reporter carries both), but they are
  upstream's to take. An engine running on the kernel and Mesa a user already has must not depend on
  them.
* **Dropping V3DV.** The device is already opt-in and slower than the CPU (ADR-089). The guard costs
  nothing elsewhere, and it keeps the opt-in honest.
* **A canary write at the end of each command buffer.** Its job runs after the reset like any other
  dependent job, and it signals success.

## Consequences

* Measured on the Pi 4 with `GGML_VK_VISIBLE_DEVICES=0`, with graphs looping on V3DV for 5 s while a
  second process plants a hang: one call refused at 1.93 s ("reset count 1 -> 2"), and 15 357 ran.
  The control run with no hang refused 0 of 11 843. The same loop on the CPU device stays unarmed, and
  the hang does not affect it.
* `tests/ci/test_gpu_reset_watch` drives a stand-in counter through a real driver call. It checks:
  unarmed and non-V3D devices, a still count, a moving count, an unreadable source, the typed error,
  and the log parse on the Pi's own lines. Sabotaging the `verify` call or the log timestamp fails it.
* Hosts see an ordinary engine error, `loom.LoomError` in loom-py, naming the reset. Nothing else
  changes for any other device.
