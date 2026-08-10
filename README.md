<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="loom.cpp" width="96">
  </picture>
</p>

<h1 align="center">loom.cpp</h1>

An inference engine built on [`ggml`](https://github.com/ggml-org/ggml) that hardcodes no model. A
model is a GGUF that carries its own **graph topologies** as JSON metadata and its own **driver
script** as embedded Lua, alongside the weights those describe; the engine parses them and builds the
compute graph at run time. Adding a model architecture is an export, not a C++ patch.

That is the whole architectural bet, and it is why this repo stays small: it targets edge devices, so
every per-model decision belongs in the exporter where it costs a Python change instead of a
specialized C++ driver. See [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## The three repos

| | |
|---|---|
| [**loom.cpp**](https://github.com/loom-ai-org/loom.cpp) | this one — the runtime, its primitives and its Lua bridge |
| [**loom-exporter**](https://github.com/loom-ai-org/loom-exporter) | turns a PyTorch checkpoint into a GGUF this engine runs |
| [**loom-py**](https://github.com/loom-ai-org/loom-py) | Python bindings, with the engine as a submodule |

## Building

```sh
cmake -B build
cmake --build build -j"$(nproc)"
```

Dependencies (`ggml`, `nlohmann_json`, LuaJIT) are fetched by CMake; nothing else is needed to build
and run the hermetic suite.

## Testing

Two classes of test, and a test's own directory is which class it is in.

```sh
ctest --test-dir build -L ci      # hermetic: builds its own fixtures, seconds, what CI runs
ctest --test-dir build -L gate    # real exported checkpoints; skips cleanly without them
```

`tests/ci/` needs nothing but this repo, a toolchain and `gguf`+`numpy` — every fixture it reads is
generated from `tests/fixtures/*.py` by a ctest step that runs first. `tests/gate/` compares against
real models: gigabytes that cannot live in git and hours that cannot run in CI, so each of those tests
exits 77 (Skipped) when its fixture is absent, and a developer with none of them still gets a green
suite that means *nothing hermetic broke*.

Point the gate suite at its fixtures with one variable:

```sh
export LOOM_FIXTURES=~/loom-fixtures
scripts/fixtures.py status    # what the 77 gate tests want, and what you have
scripts/fixtures.py fetch     # from the published fixture repo
```

Every fixture also still honours the per-test variable it always had
(`LOOM_KOKORO_MIL_GGUF`, …), which wins over the root — pointing one test at one artifact you just
rebuilt is what you do while working on it.

## Documentation

| | |
|---|---|
| [`BACKLOG.md`](BACKLOG.md) | the project ledger: every work item, what was measured, and what was found |
| [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) | the data-driven design and why the engine hardcodes nothing |
| [`docs/KV-CACHE.md`](docs/KV-CACHE.md) | how a cached attention block reaches the engine from an export |
| [`docs/LOOM_PROCEDURAL_GENERALIZATION.md`](docs/LOOM_PROCEDURAL_GENERALIZATION.md) | the embedded-Lua orchestration blueprint |

## Licence

MIT — see [`LICENSE`](LICENSE).
