---
type: adr
status: accepted
date: 2026-08-10
tags: [repository-layout, packaging, submodule, project-structure]
---

# ADR-011: One Repository Becomes Three

## Context

The engine, the PyTorch-side exporter and the Python bindings had lived in one repository. They have
different audiences, different languages, different dependency weight (the exporter pulls
`coremltools`, `torch` and family; the engine pulls a toolchain) and different release cadences.

## Options Considered

1. **Stay monolithic.** One history, one clone, and every consumer of the engine inherits the
   exporter's Python dependency tree.
2. **Split into three**, side by side on disk and under one GitHub organisation.

## Decision

Three repositories under `github.com/loom-ai-org`:

| repo | what it holds |
|---|---|
| `loom.cpp` | the `ggml` engine, its primitives, its Lua bridge |
| `loom-exporter` | turns a PyTorch checkpoint into a GGUF this engine runs |
| `loom-py` | Python bindings; vendors `loom.cpp` as a submodule at `vendor/loom.cpp` |

**The ledger stays in `loom.cpp` and remains the ledger for all three.** Items in it describe exporter
work, and splitting the record would split the reasons anything is the way it is.

## Consequences

* **Positive:** the engine ships without a Python dependency tree; the exporter can move at its own
  pace.
* **Positive:** `loom-py` vendoring `loom.cpp` as a submodule pins exactly which engine a wheel was
  built against.
* **Negative:** a change spanning engine and exporter is now two PRs in two repos, and the byte-identity
  sweep has to be run with both trees at matching revisions.
* **Negative:** documentation has to say which repo it is talking about. This is why the ledger and
  knowledge base live in one place — see the process note in
  [`CLAUDE.md`](../../CLAUDE.md) and the [backlog hub](../backlog/active-index.md).

## Related

* Epic: [Epic-08: Packaging and Release](../epics/epic-08-packaging-and-release.md)
* Ledger record, verbatim:


The engine, the exporter and a new Python binding are now three repos under
`github.com/loom-ai-org`, side by side on disk. **This file stays in `loom.cpp` and remains the ledger
for all three** — items here still describe exporter work, and splitting the ledger would split the
record of why anything is the way it is.

| repo | what it holds | history |
|---|---|---|
| `loom.cpp` | `src/ include/ tests/ scripts/ docs/`, this file | the original repo, transferred to the org |
| `loom-exporter` | `loom_exporter/ tools/ fixture_gen/ docs/` | 157 commits, `git filter-repo` by path, so `--follow` crosses the move |
| `loom-py` | pybind11 bindings, engine as a submodule | new |

**`tools/loom_mil_compiler` became `loom_exporter`**, at the repo root rather than under `tools/`: the
repo is loom-exporter, the CLI is `loom-export`, and the package now agrees with both. The old name
described the implementation where every other name describes the job.

**Every repo now splits its tests the same way, and a test's directory is which class it is in.**
`tests/ci/` is hermetic and is what GitHub Actions runs; `tests/gate/` needs real checkpoints and skips
cleanly without them. The engine's are labelled too, so `ctest -L ci` selects them without knowing any
names. The exporter's gate is the byte-identity sweep, which had lived only as a recipe outside the
repo and is now a test with its own can-this-fail check.

**`LOOM_FIXTURES` replaced sixty-nine per-test environment variables** in the engine. The layout is a
*derived rule* — drop `LOOM_`, lowercase, `_GGUF` means a file, `_DIR` is dropped — implemented twice
on purpose (`tests/support/fixtures.h`, `scripts/fixtures.py`) and pinned from both ends by
`tests/ci/test_fixture_resolution.cpp`. The migration was safe for a stated reason rather than a hoped
one: `fixture_env` consults each test's own variable FIRST and is shaped exactly like `std::getenv`,
so the 130 rewritten call sites kept every null check and skip already around them.

**What the split kept breaking, and the fix.** Code that located a sibling with
`Path(__file__).resolve().parent.parent` — which meant `tools/` before and something else after. It
took out 55 exporter tests at once. `loom_exporter/paths.py` now holds that relationship once. One
genuinely cross-repo check survives: the exporter's pre-tokenizer names against this repo's
`bpe_vocab.cpp`, found via `LOOM_CPP_ROOT` or the sibling checkout, skipping cleanly when absent.

**Two real defects surfaced only because a second consumer appeared.** `loom-export` had been invoking
`python -m tools.loom_mil_compiler.main_export`, a module path that stopped existing when the package
moved — the CLI was simply broken and nothing noticed, because the sweep calls the module directly.
And `loom/loom.h` included `kv_cache.h` but not `conv_state_cache.h`, so a host on the umbrella header
could load LFM2, tokenize for it, and fail on the first `SHORT_CONV` node; `loom_cli` never noticed
because it includes the concrete header. Both are the same lesson: a surface with one caller is not a
surface that has been tested.

