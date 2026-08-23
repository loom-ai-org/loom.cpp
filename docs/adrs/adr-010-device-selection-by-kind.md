---
type: adr
status: accepted
date: 2026-08-14
tags: [device-selection, ggml, npu, correctness, defaults]
---

# ADR-010: Devices Rank by Kind, Not by Registration Order

## Context

`Device::open("gpu")` has to turn a generic request into one concrete device. The original scheme
ranked devices in four tiers, with `GGML_BACKEND_DEVICE_TYPE_ACCEL` as rank 1 — the assumption being
that an NPU or other accelerator would register as `ACCEL` and should be preferred.

Two findings broke that. First, **no NPU registers as `ACCEL`**; the pinned header says why —
`ACCEL` means "accelerator devices intended to be used together with the CPU backend (e.g. BLAS or
AMX)". Second, registration order is not a preference ordering: on the first machine that actually had
two GPUs, `"gpu"` reliably chose an Intel iGPU over an RTX 5090
([Retro-007](../retros/retro-007-gpu-chose-the-integrated-gpu.md)).

## Options Considered

1. **Keep four tiers and document the tie as the caller's problem.** The tie is not arbitrary; it is
   reliably wrong.
2. **Rank by a table of known device names.** Requires the engine to know every backend by name, which
   is exactly what `loom::Device` avoids.
3. **Rank by what the device reports about itself**, at run time.

## Decision

**Three tiers, not four** — the tier that could not exist is deleted. Within a rank, **discrete beats
integrated**, derived at run time from what the device reports rather than from any name the engine
knows.

`"npu"` **always throws**, because `ggml` has no NPU device type to resolve it to. That is a loud
failure by choice: silently returning something else would be the same class of defect as picking the
iGPU.

## Consequences

* **Positive:** `"gpu"` picks the fastest device present, on hardware nobody tested against.
* **Positive:** no backend names in the engine; a device that did not exist when this was written still
  ranks correctly.
* **Negative:** "discrete" is a property the backend must report honestly. A device that misreports
  ranks wrong, and the engine has no second opinion.
* **Negative:** an NPU is currently unreachable through the generic spec, by design. When `ggml` grows
  a device type for it, this ADR needs revisiting rather than extending.

## Related

* Epic: [Epic-04: Backends and Accelerators](../epics/epic-04-backends-and-accelerators.md)
* Ledger record, verbatim:

### P4.8b — item 2 is not a preference problem, it is a correctness one


The scoping above says that with two accelerators of different kinds `"gpu"` meaning "the first
GPU/iGPU/accelerator registered" **"stops being a sensible default"**. That understates it, and this is
now measured rather than reasoned about.

A second offload device turns out to be available on this machine with no new hardware: **`ggml-blas`
registers as `GGML_BACKEND_DEVICE_TYPE_ACCEL`** (`ggml-blas.cpp:354`) -- the same type an NPU
registers as, and one of the three `is_offload_device()` already accepts. `-DGGML_BLAS=ON
-DGGML_BLAS_VENDOR=OpenBLAS` alongside `-DGGML_VULKAN=ON` gives a three-device registry: an iGPU, an
ACCEL, and a CPU. `test_device_selection` passes 40/40 against it.

What that registry reveals is that **the first offload device is not a stable notion**. Registration
order differs between link modes, because they are two different orderings in ggml: a linked build
registers in the `#ifdef` sequence of `ggml_backend_registry`'s constructor (Vulkan at
`ggml-backend-reg.cpp:125`, BLAS at :155), while a `GGML_BACKEND_DL` build registers in the call
sequence of `ggml_backend_load_all` (:566), where `blas` is FIRST and `vulkan` is ninth. Built both
ways from the same source on the same machine:

| spec | linked build | `GGML_BACKEND_DL` build |
|---|---|---|
| `"auto"` | `Vulkan0` | **`BLAS`** |
| `"gpu"` | `Vulkan0` | **`BLAS`** |

Three things follow, and the third is why this is filed as a defect rather than a nicety.

1. **`"gpu"` resolving to BLAS is wrong on its face.** BLAS is not a GPU. The spec exists so that a
   caller asserting something about the machine gets an error instead of a silent CPU run -- and here
   it gets neither: it gets a device that is not what was asked for.
2. **The cost is the accelerator NOT used, not the fallback itself.** BLAS implements roughly
   `MUL_MAT`/`OUT_PROD`, so nearly every node in a loom graph goes to the CPU -- correctly, via the
   `{primary, CPU}` pair the engine already builds. That part is cheap and is worth being precise
   about rather than alarmed by: `ggml_backend_blas_device_get_buffer_type` returns
   `ggml_backend_cpu_buffer_type()` (`ggml-blas.cpp:380`), so BLAS tensors are ordinary host memory
   and a BLAS/CPU split moves no data. This is nothing like the 453 splits P4.7 measured on Vulkan,
   where every boundary was a real device transfer.

   The damage is that the machine's ACTUAL accelerator is silently skipped. On this box that is the
   difference between the Vulkan iGPU -- 2.74x on Qwen3 after the P4.7a fusion -- and a CPU run with
   OpenBLAS doing the matmuls, reported to the caller as an accelerator either way.
3. **DL is what the wheels ship** (P4.8a). So this is not a hypothetical about exotic hardware: any
   user who installs two accelerator packages gets a different `device="auto"` than the `loom_cli`
   the behaviour was tested with. A CUDA box that also has OpenBLAS present is the ordinary case.

## The selection half, FIXED (2026-08-14)

`is_offload_device`/`first_offload_device` are gone, replaced by a **rank over what a device IS**, so
registration order is no longer an input:

|   | |
|---|---|
| 0 | GPU / iGPU |
| 1 | an accelerator with its own memory -- a discrete NPU |
| 2 | an accelerator in host memory -- BLAS; possibly an NPU on a UMA SoC |
| 3 | the CPU |

`"auto"` takes the best rank present and therefore cannot fail. `"gpu"` is rank 0 **only** and throws
otherwise; `"npu"` (spelled `"accel"` too) is rank 1 only and throws otherwise. The two specs partition
rather than overlap, which is the actual repair: `"gpu"` can no longer answer with a non-GPU.

**The discrete/host split is what makes rank 1 and 2 different, and it is derived at run time from
`ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(dev))`** -- no backend names anywhere, which is
the same principle the device layer already followed. Two traps found while building it, both recorded
in the code: `ggml_backend_dev_props::caps.host_buffer` is NOT this question (it means "can hand out
pinned staging buffers", and measures true for Vulkan and false for BLAS -- exactly inverted); and the
probe is about ADDRESS SPACES, not packaging, so an iGPU on UMA hardware answers false and is treated
as discrete, correctly, because a split against it still costs a memcpy.

Verified on the three-device DL build, which is the configuration that produced the defect:

```
registry order: [0] BLAS  [1] Vulkan0  [2] CPU
"auto" -> Vulkan0    "gpu" -> Vulkan0    "cpu" -> CPU
```

`test_device_selection` covers it: 38/38 on a CPU-only build, 46/46 on the three-device DL build. The
load-bearing assertion is that `"auto"` equals what `"gpu"` would have returned whenever a GPU exists
-- under the old rule it returned BLAS with the GPU sitting right there.

**Still not solved: ties WITHIN a rank.** Two GPUs are separated by registration order, because nothing
about a machine says CUDA0 should beat Vulkan0. Documented in `backend.h` as "name one".

**[Superseded by P4.8f — the conversion below was completed in `3cb5723`, and the run-time half of this
warning turned out to be a silent skip rather than a link error.]**

**Prerequisite discovered, and it is bigger than it looks: the test suite does not build under
`GGML_BACKEND_DL`.** 109 test files call `ggml_backend_cpu_init()` directly, and that symbol lives in
the CPU backend, which a DL build dlopens rather than links. Only 5 files go through
`tests/support/ggml_test_helpers.h`, so this is not one edit. `test_device_selection.cpp` was converted
(`ggml_backend_dev_init(ggml_backend_dev_by_type(CPU), nullptr)`, which is correct in both link modes)
because the DL build is the only place the ranking can be tested against the order it defeats. The
other 108 are a mechanical follow-up, and until they are done **the gate suite cannot run against the
configuration the wheels ship** -- which is worth fixing before CUDA, since `test_e2e_device_parity` is
the evidence the device layer generalizes.


### P4.8e — the tier that could not exist, deleted; and the kernel breaks the tie

P4.8d found that no NPU registers as `ACCEL`. This is the correction, and the reason turned out to be
in the pinned header the whole time rather than in any backend:

```c
// accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
GGML_BACKEND_DEVICE_TYPE_ACCEL,
```

**ggml DEFINES `ACCEL` as the BLAS/AMX co-processor role** — a thing used *together with* the CPU,
which is P4.8b's rank **2** — while `GPU` is defined as "GPU device using dedicated memory". By that
taxonomy a discrete NPU that runs whole graphs out of its own memory **is** a GPU, and all three of
ggml's accelerator backends agree: `ggml-openvino` (:751), `ggml-hexagon` (:3917), and `ggml-et`, the
one new backend directory at master. So P4.8b's rank 1 was not waiting for hardware; it was a
misreading of the enum, and it could never have had a member.

Checked rather than assumed, because "upstream will fix it" was the live alternative: ggml master
`8846b79e` is **154 commits and one month past** the pinned `v0.16.0`, and both NPU backends are
byte-identical at HEAD on exactly these lines. The enums are identical too, comments included. This is
upstream's position, not an immaturity to wait out — which also retires the idea of a temporary
name-matching hack, since a temporary fix needs an expiry condition and there is none.

### The ranking, collapsed to what has behavioural consequences

|   | |
|---|---|
| 0 | an offload device with its own memory — a split against it costs a copy |
| 1 | an offload device in host memory — BLAS; a split against it copies nothing |
| 2 | the CPU |

`primary_rank` no longer switches on the type to decide the tier, only to separate the CPU from
everything else; the memory question it was already asking does the rest. The assist rule survives
unchanged apart from its numbering, because the two ranks it names are exactly the two that question
separates.

### `"npu"` now always throws, and the message is the feature

Resolving it was never right: it answered "no NPU with its own memory is available" on the one machine
whose NPU was running. Deleting the spelling is not right either — a caller who types it has a real
question and deserves an answer rather than `unknown device 'npu'`. So it is a recognised spec that
raises, naming the cause (`ggml does not report NPU identity`), the evidence (all three backends
register as GPU), and the thing that actually works (name the device; for OpenVINO also set
`GGML_OPENVINO_DEVICE=NPU`, since that is what picks its target).

**And it consults `/dev/accel` to avoid lying by omission.** The Linux accel subsystem (`drivers/accel`
— `intel_vpu`, `habanalabs`, `qaic`) is where accelerators live and where a GPU never appears: on the
workstation `accel0` is `intel_vpu` while both GPUs are `/dev/dri/card{0,1}`. So when the kernel does
know about an accelerator the message says so, and a user with an NPU is not told they have none. It
is diagnostic only — it answers "does this machine have an accelerator", never "is THIS ggml device
one", which is the question nothing can answer.

### Ties within a rank, which the collapse made urgent

Merging the tiers put a real GPU and an NPU-shaped backend in the same rank, so `"auto"` on a CUDA +
OpenVINO box would go back to registration-order roulette — the exact defect P4.8b was filed for. The
tie-break is the first *positive* signal found in this whole thread:

```
device     type   buft_is_host  host_buffer  device_id       kernel says
CUDA0      GPU    false         true         0000:02:00.0    0x030000  <- display controller
OPENVINO0  GPU    false         false        (null)          -
```

`ggml_backend_dev_props::device_id` is the device's PCI address, and **null is a reliable reading**
rather than uninitialised memory, because `ggml_backend_dev_get_props` memsets the struct before the
backend fills it. sysfs then says what that address is: base class `0x03` is a display controller, and
the Intel NPU at `0000:00:0b.0` is `0x12`, a processing accelerator. The kernel maintains exactly the
taxonomy ggml's backends stopped maintaining, and it is the one authority here that is not a
self-report.

**It promotes on positive evidence and never demotes on absence**, which is the whole of its safety.
Only `ggml-cuda` and `ggml-vulkan` populate `device_id`; `ggml-metal`, `ggml-sycl`, `ggml-opencl`,
`ggml-webgpu` and `ggml-cann` are real GPU backends that leave it null, and there is no sysfs at all
off Linux. So an unconfirmable device keeps precisely the standing it had before — where nothing is
confirmable, registration order still decides, exactly as it did yesterday.

### Verified, and the tie-break was made to fail on purpose first

* **CPU-only build**: `test_device_selection` 49/49 (was 38 — the new checks are the unconditional
  `"npu"`/`"accel"` raise and its message).
* **Three-device DL build** (Vulkan + BLAS + CPU, the configuration that produced P4.8b's defect):
  64/64, and `probe_chain` still shows `auto -> Vulkan0, assists=1, order Vulkan0 BLAS CPU` **with
  BLAS registering first** — the collapse did not cost the assist chain.
* **CUDA + OpenVINO on the workstation** — the configuration this tie-break exists for: 61/61,
  `ci` 58/58, and both device-parity gates still pass. `auto` and `gpu` resolve to `CUDA0`.
* The `"npu"` message degrades as designed: no `/dev/accel` on the dev box, so the accelerator clause
  is absent there, and present on the workstation as `[accel0 (intel_vpu)]`.

**`auto -> CUDA0` on that box proves nothing by itself, and nearly shipped as if it did.** CUDA
registers *first* in both link modes — the `#ifdef` sequence puts it ahead of OpenVINO, and
`ggml_backend_load_all` calls `cuda` 11th against `openvino` 21st — so the old registration-order rule
gives the same answer, and the whole tie-break could have been dead code behind a green result.

So it was inverted (`kernel_confirms_gpu(dev) ? 1 : 0`) and re-measured: `auto` and `gpu` both flipped
to **`OPENVINO0`**, with the registry order unchanged. The tie-break demonstrably overrides
registration order, and CUDA0 wins for the reason claimed rather than by luck.

**A trap worth recording from doing that:** restoring the file with `rsync -a` and rebuilding did NOT
revert the binary. rsync preserves mtimes, so the restored source was *older* than the object compiled
from the inverted version and ninja considered it up to date — the probe kept answering `OPENVINO0`
from a stale `.o` after a build that reported success. `touch` the file after any rsync-based revert,
or the next measurement is of the code you thought you removed.

`probe_tiers` gained the `device_id` and kernel-class columns, since it is the tool that answers this
question for the next backend, and `tools/debug/README.md`'s "first question to point them at" is
rewritten — that question is now answered.

