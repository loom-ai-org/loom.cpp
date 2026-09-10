---
type: retro
date: 2026-09-04
domain: packaging
tags: [armv6, arm32, raspberry-pi, libstdc++, filesystem, verification]
---

# Retro-034: The Board Was Running a libstdc++ Nobody Built Against

## Issue

With the ARMv6 port otherwise finished — engine green, wheel built, `ctest -L ci` 81/81 and
`pytest tests/ci` 95/4 under emulation — the reference board started segfaulting. Not always: the
same script would run to completion seven times and then fail eight times in a row, and a copy of it
with one comment line added behaved differently from the original. `loom.devices()` was clean 12/12
and `Model.from_file` clean 20/20 while the full script was 10/10 dead.

The core dump put it in ggml, at startup:

```
Program terminated with signal SIGSEGV.
#0  ggml_backend_load_best(char const*, bool, char const*)  from .../loom/libggml.so
#1  ggml_backend_load_all_from_path ()                      from .../loom/libggml.so
#2  loom::(anonymous namespace)::ensure_backends_loaded()   from .../loom/libloom_engine.so
#3  loom::Device::open(std::string const&)                  from .../loom/libloom_engine.so
```

Backend discovery — the directory walk that finds `libggml-*.so`. Not a model, not a kernel, not
arithmetic. Nothing that had been ported.

## Root cause

Not this project's, and not ARMv6's. **The board had a third-party `libstdc++.so.6.0.32` installed
into `/usr/local/lib/arm-linux-gnueabihf`, and that directory is first in `/etc/ld.so.conf.d`** — so
every C++ program on the machine loads it in preference to Raspbian's own 6.0.30. `dpkg -S` names the
package that put it there; it has nothing to do with anything here.

The disassembly at the faulting instruction says exactly what went wrong. It is the inlined
`_Sp_counted_base::_M_release()` for the `shared_ptr` inside `std::filesystem::directory_iterator`,
on the `__libc_single_threaded` fast path:

```
ldr r6, [sp, #156]     ; the control block
ldr r3, [r6, #28]      ; use_count--   (non-atomic: single-threaded)
sub r2, r3, #1
str r2, [r6, #28]
cmp r3, #1
ldr r3, [r6]           ; last reference -> load the vptr to destroy it
ldr r3, [r3, #8]       ; <-- SIGSEGV: r3 = 0x261f, not a pointer
```

`0x261f` is what glibc's tcache leaves in a freed chunk. The control block had already been freed:
the iterator's shared state is created inside libstdc++ and released by code inlined from the headers
we compiled against, and those two were not the same libstdc++.

**The proof is fifteen lines and does not mention loom.** Compiled with the board's own `g++`:

```cpp
#include <filesystem>
namespace fs = std::filesystem;
int main(int argc, char ** argv) {
    for (int round = 0; round < 200; ++round) {
        std::error_code ec;
        for (const auto & e : fs::directory_iterator(argv[1])) { if (e.is_regular_file(ec)) {} }
    }
}
```

```
--- default loader (/usr/local libstdc++ 6.0.32) ---   RC=139  RC=139  RC=139
--- LD_PRELOAD Raspbian libstdc++ 6.0.30 ---           RC=0    RC=0    RC=0
```

And the same A/B on the actual wheel: 5/5 dead, then 5/5 alive.

## Fix, on the board

The stray library could not simply be deleted: the binary that shipped it genuinely needs
`GLIBCXX_3.4.32`, which Raspbian bookworm's 6.0.30 does not provide, and its service is enabled and
running. So it was made **private to the program that needs it** rather than global —
`/usr/local/lib/bluetooth-km-switch/`, out of the loader's search path, with a systemd drop-in giving
that unit an `LD_LIBRARY_PATH`. The service still starts; every other C++ program on the board is
back on the distribution's libstdc++; the fifteen-line reproducer passes; the wheel runs.

## Takeaway

**Before blaming a new architecture, ask which libraries the board is actually loading.** The port had
one genuine defect ([Retro-033](retro-033-a-shared-library-links-clean-without-its-symbols.md)) and
this was not it, but everything about the symptom said "32-bit memory bug": a use-after-free, in
`std::filesystem`, reachable only on the one machine of that ISA, sensitive to layout in the way real
corruption is. `ldd` on the shipped `.so` would have named it in one command, and did — an hour later
than it should have. Add it to the first things checked on a board that is new to this project:
`ldd`, `/etc/ld.so.conf.d`, and whether anything in `/usr/local` is shadowing the toolchain.

**A reproducer that removes your own code is worth the ten minutes.** "loom segfaults on the Pi Zero"
and "`std::filesystem` segfaults on the Pi Zero" are different bug reports with different owners, and
the second one is the true one. Until that program existed, every experiment was an attempt to find
the pattern in *our* behaviour — script length, invocation style, detachment, faulthandler — and each
one produced a plausible, wrong theory, because the real variable was not in the search space.

**Emulation could not have caught it.** The container has exactly one libstdc++, the one the wheel was
built against, which is the whole point of building in the target's userland — and is precisely why
the emulated gate is not a substitute for the board.
