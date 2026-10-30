---
type: retro
date: 2026-10-29
domain: backends
tags: [vc4, ggml-backend-dl, dlopen, symbols, testing, device-selection, raspberry-pi-zero]
---

# Retro-101: A Board Test That Linked What Production Loads

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.46); the decision is
[ADR-092](../adrs/adr-092-the-qpus-take-the-epilogue-q4-0-and-the-pointwise-convolutions.md).

## Issue

P4.46's ggml-vc4 passed `test_vc4_backend` on the Pi Zero -- every new path placed on the QPUs, 40
nodes folded into convolutions, ALL OK. The model A/B that followed, through loom-py, ran all 13 models
on the CPU in both P4.46 arms, at the CPU's speed, with outputs identical to the CPU's and no selection
note: nothing in the results said the QPUs had not been there.

## Root cause

The fallback for nodes the backend claims and cannot fold calls ggml's registry to find the CPU
backend (`ggml_backend_dev_by_type`), and the registry is libggml's, not libggml-base's -- the one
ggml library a backend module links. Under `GGML_BACKEND_DL` ggml's loader dlopens the module, the
symbol does not resolve, and the loader drops the module with one line on stderr
(`failed to load ... undefined symbol: ggml_backend_dev_by_type`). With no VC4 device, "auto" has one
choice and makes it silently. The board test never saw it: it links loom's engine and libggml into
the executable, so the symbol was already in the process when the module was opened.

The fix finds the function at run time (`dlopen("libggml.so", RTLD_NOLOAD)` + `dlsym`), leaving no
reference to resolve at load.

## Takeaway

**A test must load a plugin the way production loads it, or check what production's loader checks.**
Linking more into the test executable than the deployed process has is a convenience that hides
exactly the failures that only happen at load. Where the test cannot be made to load the plugin the
production way, assert the property the loader enforces: `test_vc4_module_symbols` now requires every
unversioned import of `libggml-vc4.so` to be defined by `libggml-base.so`, in every DL build -- it
fails on the dropped module and passes on P4.45's. And an A/B whose arms can silently fall back must
check the device each arm ran on before comparing times: "identical output, same time" was the
symptom, not a result.

## Record

* `~/p446/out1/stderr-*-p446.txt`: `load_backend: failed to load /home/pi/p446/pkg/loom/libggml-vc4.so:
  ... undefined symbol: ggml_backend_dev_by_type`; every P4.46 RESULT line `device=CPU ... note=''`.
* `cmake -P tests/ci/vc4_module_symbols.cmake` on the dropped module: FATAL_ERROR naming
  `ggml_backend_dev_by_type`; on P4.45's module: 16 imports, all in libggml-base; on the fixed x86 DL
  build: 24.
