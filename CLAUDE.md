# loom.cpp — orientation

The runtime. One of **three repos**, all under `github.com/loom-ai-org` and, on a dev machine,
side by side under one parent directory:

| | |
|---|---|
| `loom.cpp` | this repo — the ggml engine, its primitives, its Lua bridge |
| `loom-exporter` | turns a PyTorch checkpoint into a GGUF this engine runs |
| `loom-py` | Python bindings; vendors this repo as a submodule at `vendor/loom.cpp` |

## The one idea

**The engine hardcodes no model.** A model is a GGUF that carries its own *graph topologies* as JSON
metadata and its own *driver script* as embedded Lua, alongside the weights those describe; the engine
parses them and builds the ggml graph at run time. Adding an architecture is an export, not a C++
patch — see [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

The corollary is a standing rule: **per-model complexity belongs in the exporter, not here.** The
engine targets edge devices, and a new family should cost a Python change. C++ here is for what is
per-*task* rather than per-model — tokenizers, CTC decode, the caches.

## Build and test

```sh
cmake -B build && cmake --build build -j"$(nproc)"
ctest --test-dir build -L ci      # hermetic, ~5 s, what CI runs
ctest --test-dir build -L gate    # real exported checkpoints; skips cleanly without them
```

**A test's directory is which class it is in**, and both are also ctest labels:

* `tests/ci/` (33 executables) needs nothing but this repo, a toolchain and `gguf`+`numpy` — every
  fixture it reads is generated from `tests/fixtures/*.py` by a ctest step that runs first.
* `tests/gate/` (80) compares against real exported models. Each exits **77** when its fixture is
  absent, which ctest reports as Skipped, so a developer with none of them still gets a green suite
  meaning *nothing hermetic broke*.

Gate fixtures come from one variable, `LOOM_FIXTURES`, whose layout is a *derived rule* rather than a
table: drop `LOOM_`, lowercase, `_GGUF` means a file and `_DIR` is dropped. The rule is implemented
twice on purpose — `tests/support/fixtures.h` and `scripts/fixtures.py` — and pinned from both ends by
`tests/ci/test_fixture_resolution.cpp`. Each test's own historical variable
(`LOOM_KOKORO_MIL_GGUF`, …) is still checked **first** and wins. `scripts/fixtures.py status` says
what is present.

## Conventions worth knowing before changing anything

* **`BACKLOG.md` is the project ledger for all three repos** — every work item, what was measured, and
  what was found. Code comments reference it by item (`BACKLOG.md P4.3e`). Read the relevant entry
  before touching an area; add to it when you finish something.
* **Comments say why, not what.** The codebase's density is deliberate: where a line is the way it is
  because an alternative was tried and failed, the comment records the failure and the measurement.
  Match that.
* **A gate that cannot fail proves nothing.** Before trusting a byte-identity or reference comparison,
  confirm it can go red — the sweep has produced a false pass before.
* **Tensor oracle, not token oracle.** A wrong encoder still decodes a plausible transcript; compare
  tensors before believing token agreement.
