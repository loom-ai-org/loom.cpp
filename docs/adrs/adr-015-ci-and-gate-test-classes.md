---
type: adr
status: accepted
date: 2026-07-01
tags: [testing, ci, fixtures, verification, gates]
---

# ADR-015: Two Test Classes — Hermetic `ci`, Checkpoint-Backed `gate`

## Context

Verifying a data-driven engine needs both kinds of test: fast hermetic checks that any contributor can
run with nothing but a toolchain, and numerical comparisons against real exported checkpoints — which
are gigabytes, cannot be committed, and will not be present on most machines.

A suite that fails without the checkpoints is unusable; a suite that silently skips them proves
nothing.

## Options Considered

1. **One suite, skipping silently when fixtures are missing.** Green means nothing.
2. **One suite, failing when fixtures are missing.** Nobody without the models can contribute.
3. **Two classes, distinguished by directory and by ctest label.**

## Decision

**A test's directory is which class it is in**, and both are also ctest labels:

* **`tests/ci/`** needs nothing but this repo, a toolchain and `gguf`+`numpy`. Every fixture it reads is
  generated from `tests/fixtures/*.py` by a ctest step that runs first.
* **`tests/gate/`** compares against real exported models. Each exits **77** when its fixture is absent,
  which ctest reports as Skipped — so a developer with none of them still gets a green suite meaning
  *nothing hermetic broke*.

Gate fixtures come from one variable, `LOOM_FIXTURES`, whose layout is a **derived rule** rather than a
table: drop `LOOM_`, lowercase, `_GGUF` means a file and `_DIR` is dropped. The rule is implemented
twice on purpose — `tests/support/fixtures.h` and `scripts/fixtures.py` — and pinned from both ends by
`tests/ci/test_fixture_resolution.cpp`. Each test's own historical variable still wins if set.

**Three standing rules come with this:**

1. **A gate that cannot fail proves nothing.** Before trusting a byte-identity or reference comparison,
   sabotage it and confirm it goes red. This is not a nicety: `LOOM_CHECK` only *counts* failures, and
   two ASR gates ended with `return 0` instead of `LOOM_TEST_REPORT_AND_RETURN()`, so every check in
   them was decorative — found only because a new test was deliberately sabotaged and did not go red.
2. **Tensor oracle, not token oracle.** A wrong encoder still decodes a plausible transcript. Compare
   tensors before believing token agreement.
3. **ASR oracle for TTS.** Cosine similarity against PyTorch is not a shipping gate; transcribe the
   audio and check the peak. See [Retro-006](../retros/retro-006-kokoro-shipped-noise.md).

Exporter changes are additionally gated on a byte-identity sweep — see
[ADR-004](adr-004-mil-as-the-single-export-path.md) and
[Retro-015](../retros/retro-015-export-snapshot-sweeps.md).

## Consequences

* **Positive:** a contributor with no checkpoints gets a meaningful green in seconds.
* **Positive:** the fixture-resolution rule is pinned from both language sides, so C++ and Python cannot
  drift about where a fixture lives.
* **Negative:** "green" has two meanings, and a reader must check whether the gate class ran or skipped.
  `scripts/fixtures.py status` says what is present.
* **Negative:** the gate only tests the build configuration it ran in — see
  [Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md).

## Related

* Epic: [Epic-01: Inference Engine Core](../epics/epic-01-inference-engine-core.md)
