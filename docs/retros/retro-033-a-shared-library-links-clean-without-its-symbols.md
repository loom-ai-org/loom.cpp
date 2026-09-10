---
type: retro
date: 2026-09-04
domain: packaging
tags: [armv6, arm32, atomics, linking, cmake, wheels, ggml]
---

# Retro-033: A Shared Library Links Clean Without Its Symbols

## Issue

The first ARMv6 build of the engine compiled all 156 objects, linked `libggml-base.so`,
`libggml-cpu.so`, `libggml.so` and `libloom_engine.so` without a word, and then died on step 74 of 74:

```
[73/74] Linking CXX shared library libloom_engine.so
[74/74] Linking CXX executable tools/loom_cli/loom_cli
FAILED: tools/loom_cli/loom_cli
/usr/bin/ld: _deps/ggml-build/src/libggml-base.so.0.19.0: undefined reference to `__atomic_fetch_add_8'
```

The symbol belongs to a library built four steps earlier. `ld` blames it while linking something else.

## Root cause

`ggml_graph_next_uid()` (`ggml.c`) is a `__atomic_fetch_add` on a `static uint64_t`. Every 64-bit
host inlines that, and so does every 32-bit CPU with a double-word exclusive — i686 has `cmpxchg8b`,
ARMv7 has `LDREXD`. **ARMv6 has `LDREX`/`STREX` and no `LDREXD`**, so GCC emits a call to libatomic's
`__atomic_fetch_add_8` instead. Nothing in the tree links libatomic, because on every other target
there has never been a call to link.

The interesting half is not the missing library, it is **when you find out**. A shared object is
permitted to carry undefined symbols — the loader resolves them from whatever else is mapped — so
`ld -shared` says nothing. Only an *executable* must be fully resolved at link time, which is why the
error arrives at the first one and points backwards at a library that already "succeeded".

**A build that links no executable at all never reaches the error.** That is exactly the wheel build:
`_loom.cpython-311-arm-linux-gnueabihf.so` is a shared object too. Had ARMv6 been approached from the
packaging side first — which was the plan — the wheel would have built, published, installed, and
failed at `import loom` with an undefined-symbol `dlopen` error, on a user's board, with nothing in
CI having gone red.

## Takeaway

**A clean `-shared` link is not evidence that a library's symbols exist.** When a build is
cross-checking a new platform, make sure something in it links an *executable*, or the first honest
answer comes from `dlopen` on somebody else's machine. `ldd -r` and `readelf --dyn-syms` answer the
same question deliberately.

**Probe the toolchain, do not test the architecture name.** The fix is
`check_c_source_compiles` — which compiles *and links* — around exactly the construct in question, and
libatomic added when the probe fails. That covers ARMv5, MIPS32 and rv32 for free, keeps no list to
maintain in `cmake/Dependencies.cmake`, and adds nothing on the platforms that inline it. The
architecture-name version of this fix would have been shorter, wrong twice, and silent about it.

**The dependency goes on the library that has the undefined symbol**, `PUBLIC`, not on our own
executables. Put it there and the DT_NEEDED is where a reader expects it and every downstream
target — tools, tests, the Python extension — cannot forget.

## Record

```
$ readelf -A tools/loom_cli/loom_cli | grep Tag_
  Tag_CPU_arch: v6
  Tag_FP_arch: VFPv2
  Tag_ABI_VFP_args: VFP registers

$ cmake -B build                       # after the fix, on the target
-- 64-bit atomics need libatomic on this target: /usr/lib/arm-linux-gnueabihf/libatomic.so.1

$ ldd _deps/ggml-build/src/libggml-base.so.0.19.0 | grep atomic
	libatomic.so.1 => /lib/arm-linux-gnueabihf/libatomic.so.1

$ cmake -B build                       # x86-64, unchanged
-- Performing Test LOOM_HAVE_BUILTIN_ATOMIC64 - Success
```
