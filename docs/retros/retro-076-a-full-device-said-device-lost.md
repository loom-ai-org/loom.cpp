---
type: retro
date: 2026-10-09
domain: backends
tags: [backends, vulkan, radv, memory, device-selection, moss-tts, igpu]
---

# Retro-076: A Full Device Said "Device Lost"

## The Issue

The MOSS-TTS model card failed on Vulkan on the dev box's Radeon Vega 3 iGPU (RADV RAVEN2) with
`loom.run_subgraph: vk::Queue::submit: ErrorDeviceLost`, about 15 minutes into the card. It failed
the same way on the rc15 package and on the ggml v0.26.0 bump. It passed on the RTX 5090 under both
Vulkan and CUDA. Larger single models pass on the same Vega (Voxtral-4B, 15 GB), and so did each half
of this pair on its own (the codec card, 3 min).

## Root Cause

**The device was out of memory, and nothing said so until the first command submission.** The card
holds the talker (`moss-tts-local-transformer-v1.5`, 16.8 GB F32) and the codec
(`moss-audio-tokenizer-v2`, 4.3 GB F32) at once. That is 21.1 GB of weights. The Vega can make at most
19.97 GB resident: GTT 17.8 GB plus a 2 GB VRAM carve-out (`/sys/class/drm/card0/device/mem_info_*`),
which is what `loom.devices()` reports as `memory_total`.

RADV accepted every allocation past that point; resident memory even went *down* as the codec loaded
(17.6 -> 17.2 GB) while the kernel evicted. The codec's first graph then failed at submit, and RADV
printed the reason:

```
radv/amdgpu: Not enough memory for command submission.
ggml_vulkan: device lost on Vulkan0
```

The failure in the talker's line of the log is a consequence: a lost device stays lost for the
process. The 15 minutes were the talker generating; loading the talker without running it, then
decoding 50 frames through the codec, reproduces it in 42 s.

Nothing in the engine looked at free memory before loading: `GgufModel::load` handed the weights to
`ggml_backend_alloc_ctx_tensors`, and `Device::open("auto")` ranked devices by kind alone.

## The Fix

ggml's `memory_free` for Vulkan follows `VK_EXT_memory_budget`: 19.80 GB before the codec, 15.48 GB
after it. Both decisions now read it:

* **`GgufModel::load` refuses up front** when the primary is an offload device that reports less free
  memory than the weights need. The requested size is computed from each tensor's buffer-type size
  plus alignment, which is what the allocation would take. The message gives both numbers and the
  way out. A device that reports no memory is not checked.
* **`Device::open(spec, weight_bytes)` makes `auto` skip** an offload device that cannot hold the
  weights. The next rank wins, and the CPU is never skipped. `selection_note()` records why;
  `loom_cli` prints it and loom-py raises it as a `RuntimeWarning`. A device named explicitly is never
  second-guessed: under the refusal above it gets the error instead. `GgufModel::weight_bytes()` on
  a `load_metadata` model supplies the size without allocating.

On the Vega, the card now runs the talker on Vulkan0 and the codec on the CPU, with the warning
`'auto' chose CPU: Vulkan0 (...) has 2.39 GB free and the weights need 4.31 GB`. `device="Vulkan0"`
for the codec fails at load, in seconds.

**Not covered:** the check counts weights only. A model whose weights fit with too little room left
for its compute buffers and caches can still overrun the same way. That needs the scheduler's
reserved sizes, which are known only once a graph is built.

## Takeaway

**On a GPU, `ErrorDeviceLost` is a symptom, not a diagnosis. Read the driver's own stderr line before
the ggml one, and add up what is resident before suspecting a shader.** The memory hypothesis was
testable in a minute: sysfs gave the pool sizes, and two small Python runs (one model, then both)
separated "too big together" from "a dispatch that hangs". And **a device that over-commits will not
refuse the allocation that breaks it**, so the engine has to ask before it allocates.
