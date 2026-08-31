---
type: retro
date: 2026-08-31
domain: packaging
tags: [porting, cross-platform, reading-vs-running, blocker-scoping]
---

# Retro-024: A Blocker Read From One Half Of An Agreement

## The Issue

P4.10 (macOS wheels) was scoped by reading pinned sources, and scoped carefully: **four blockers,
established twice, re-verified against the current pins the day before the work started.** The fourth
was flagged as "the one that would burn a day of CI" and the item carried an explicit instruction —
*do not start by writing a workflow, start by settling this.*

The first macOS build settled it in the other direction. **Blocker 4 did not exist**, and never had.
Meanwhile three blockers that no amount of reading had produced appeared within the first two builds,
and each of them was real.

## Root Cause

Blocker 4 said: `ggml-backend-reg.cpp`'s `backend_filename_extension()` returns `.dll` on `_WIN32`
and `.so` otherwise, with no `__APPLE__` case; CMake emits `.dylib` on macOS; therefore a
`GGML_BACKEND_DL` build produces backends its own loader will never find, which means **zero devices,
including no CPU**, silently.

Every clause of that is true except the second, and the second is not about macOS — it is about
**what kind of CMake target a backend is**. ggml builds each one with `add_library(${backend} MODULE
...)`. Darwin gives *module* libraries the suffix `.so` (`CMAKE_SHARED_MODULE_SUFFIX`); only *shared*
libraries get `.dylib`. The loader and the build already agreed, and they agreed on `.so`.

So the failure was not in the reading. It was in reading **one half of a two-party agreement and
predicting the other half from a general rule** — "CMake emits `.dylib` on macOS" — that happens not
to apply to the party in question. The evidence looked like two files disagreeing; it was one file
plus an assumption.

The three that were missed share the opposite property. Each is invisible on Linux for a *structural*
reason, so no amount of reading the macOS side would surface them — only running would:

* **LuaJIT's sub-`make` inherited the outer make's jobserver** and died on closed descriptors. Not a
  macOS issue at all: any `Unix Makefiles` build anywhere. Hidden because everything that matters
  uses Ninja.
* **`loom::Error`'s typeinfo did not coalesce** across the dylib boundary under pybind11's
  `-fvisibility=hidden`, because Mach-O has a two-level namespace and Apple's libc++ compares
  `type_info` by address. `loom.LoomError` quietly became `RuntimeError`. Hidden because ELF's flat
  namespace merges the weak symbols.
* **`/tmp` and `/var` are symlinks on macOS**, so a test comparing `abspath` against a `resolve()`d
  path failed on two spellings of the same directory. Hidden because Linux has no such symlink.

A fourth of the same kind: the wheel's **filename tag** comes from `MACOSX_DEPLOYMENT_TARGET` in the
environment, while `CMAKE_OSX_DEPLOYMENT_TARGET` decides what the binaries actually require. Setting
only the CMake half produced binaries good for 11.0 inside a wheel named `macosx_15_0_arm64`.

## Takeaway

**A loader and a build are two halves of one agreement, and reading one half predicts nothing about
the other.** When a port hinges on "these two components will disagree", find the second component's
own statement — here, the `add_library` line — rather than deriving it from what the platform
"normally" does. The general rule was correct and did not apply.

**Scoping by reading finds the blockers that are written down; running finds the ones that are
structural.** The four scoped blockers cost, between them, three small CMake changes. The three
unscoped ones cost the debugging. That is not an argument against scoping — three of four were real
and the fixes were ready — but it does mean the estimate a scoping produces is an estimate of the
*known* half, and a port should be planned as though the unknown half is the same size.

**The corollary for the next port (Windows).** The same three shapes will recur in Windows spellings:
a build-system assumption that is really a target-type question, a symbol-visibility rule that differs
(`__declspec(dllexport)` and no weak symbols at all), and paths that are the same location under two
names. Expect to find them by building, and budget for it.

## The Record

Blocker 4's disproof, from the runtime directory of the first successful macOS build:

```
libggml-base.dylib   libggml.dylib   libloom_engine.dylib      <- SHARED, linked, .dylib
libggml-cpu-apple_m1.so   libggml-cpu-apple_m2_m3.so           <- MODULE, dlopened, .so
libggml-cpu-apple_m4.so   libggml-blas.so   libggml-metal.so
```

followed by `loom.devices()` reporting `CPU (Apple M1 Pro)` — the thing blocker 4 predicted would be
absent. No `SUFFIX ".so"` override was written, here or upstream.

The consequence that survives is the *asymmetry*, which the original blocker 2 had only half of: the
wheel's install rule has to match **both** suffixes, because the linked libraries really are `.dylib`
and the backends beside them really are `.so`.

Full record in [Epic-08 §4](../epics/epic-08-packaging-and-release.md).
