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

* `tests/ci/` (35 executables) needs nothing but this repo, a toolchain and `gguf`+`numpy` — every
  fixture it reads is generated from `tests/fixtures/*.py` by a ctest step that runs first.
* `tests/gate/` (84) compares against real exported models. Each exits **77** when its fixture is
  absent, which ctest reports as Skipped, so a developer with none of them still gets a green suite
  meaning *nothing hermetic broke*.

Gate fixtures come from one variable, `LOOM_FIXTURES`, whose layout is a *derived rule* rather than a
table: drop `LOOM_`, lowercase, `_GGUF` means a file and `_DIR` is dropped. The rule is implemented
twice on purpose — `tests/support/fixtures.h` and `scripts/fixtures.py` — and pinned from both ends by
`tests/ci/test_fixture_resolution.cpp`. Each test's own historical variable
(`LOOM_KOKORO_MIL_GGUF`, …) is still checked **first** and wins. `scripts/fixtures.py status` says
what is present.

## The knowledge base, and how to keep it

**Documentation for all three repos lives in `loom.cpp/docs/`, in four tiers.** `BACKLOG.md` was one
9,000-line ledger; it is now a redirect with a map from every old section to its new home.

| tier | path | holds | size |
|---|---|---|---|
| **Hub** | [`docs/backlog/active-index.md`](docs/backlog/active-index.md) | **open work only**, one line each | one screen per domain |
| **Epic** | [`docs/epics/`](docs/epics/) | what a domain is, how it works, and its planned work | ~100 lines, except where a live plan lives there |
| **ADR** | [`docs/adrs/`](docs/adrs/) | why a choice was made: context, options, decision, consequences | ~60 lines |
| **Retro** | [`docs/retros/`](docs/retros/) | what broke: issue, root cause, takeaway — then the verbatim record | ~50 lines + record |

`docs/archive/` holds per-commit detail for work closed before 2026-08-10. **Unmaintained. Never cite
it as current, never add to it.**

### Where does this belong?

Ask in this order and stop at the first yes.

1. **Is it work still to be done?** → the hub, as one line, linked to the epic or ADR that gives it
   context. Nothing else goes in the hub — no explanations, no history.
2. **Does it explain why a technical choice was made, with alternatives that were considered?** → an
   **ADR**. If you can name what you *didn't* do and why, it is an ADR.
3. **Was something wrong, debugged, and understood?** → a **retro**. The test is whether there is a
   takeaway that changes how the next person works. A bug with no transferable lesson is a git commit,
   not a retro.
4. **Does it describe how a domain works today?** → the relevant **epic**.
5. **None of the above?** → it is a commit message.

### The protocol

* **Item IDs are the existing `P`-numbers** (`P4.15b`, `P4.3e`). Code comments across all three repos
  cite them. Do **not** renumber; new items continue the scheme. ADRs and retros get their own
  sequential numbers.
* **When you finish an item, it leaves the hub.** Move its decision into an ADR, its lesson into a
  retro, its architecture into the epic — then delete the line. "Done" is not a state the hub tracks.
* **Nothing closed and older than about two weeks stays tracked anywhere as an item.** Its knowledge
  has to have landed in a tier by then, or it is lost.
* **One file, one thing.** A retro covers one failure; an ADR covers one decision. If you are writing
  "and also", start a second file.
* **Every new file gets YAML frontmatter** (`type`, `status`/`date`, `domain`, `tags`) so the set stays
  queryable, and links to its neighbours: an ADR names the epic it serves, an epic names its ADRs and
  retros, the hub links out to both.
* **Update `last_updated` on an epic or the hub when you change it.**
* **Check links before committing:** every relative link and `#anchor` in `docs/` must resolve.

### Standing rules these tiers encode

* **Comments say why, not what.** Where a line is the way it is because an alternative was tried and
  failed, the comment records the failure and the measurement. Match that density.
* **A gate that cannot fail proves nothing.** Sabotage it and confirm it goes red before trusting a
  byte-identity or reference comparison. This has produced a false pass before —
  [ADR-015](docs/adrs/adr-015-ci-and-gate-test-classes.md).
* **Tensor oracle, not token oracle.** A wrong encoder still decodes a plausible transcript.
* **ASR oracle for TTS.** Cosine 0.996 against PyTorch and unintelligible output are compatible states
  — [Retro-006](docs/retros/retro-006-kokoro-shipped-noise.md).
* **Before opening a performance item**, read
  [Retro-012](docs/retros/retro-012-optimizations-that-were-measured-out.md). The register of
  measured-out ideas exists so they are not re-proposed.
* **Per-model complexity belongs in the exporter, not here** —
  [ADR-003](docs/adrs/adr-003-per-model-complexity-in-the-exporter.md). C++ in this repo is for what is
  per-*task*: tokenizers, CTC decode, the caches.

## Where to start

Read [`docs/backlog/active-index.md`](docs/backlog/active-index.md) for what is open, then the epic for
the area you are touching. Read the relevant entry before changing anything in an area; add to the
right tier when you finish.
