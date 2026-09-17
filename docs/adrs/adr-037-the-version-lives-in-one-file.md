---
type: adr
status: accepted
date: 2026-09-17
tags: [packaging, release, loom-py, epic-08]
supersedes: []
---

# ADR-037: The Version Lives in One File, and the Derived Copies Are Text a Test Proves

## Context

loom-py publishes **four** packages — `loom-py-rt` and the `-cuda`/`-vulkan`/`-metal` backends — and
they pin each other by **exact** version, in a circle. That pin is not style:
[ADR-009](adr-009-backends-as-dynamic-libraries.md) makes a backend a `libggml-<backend>.so` dlopened
beside the base wheel's `libggml-base.so`, and ggml promises no ABI across revisions, so any base
release that moves the ggml pin invalidates every backend wheel published before it.

A circular exact-pin set means one release number is **eleven strings in five files, in two
spellings**: `version = "1.0.0-rc10"` (PEP 621's readable form) and `== 1.0.0rc10` (PEP 440's normal
form, the only one a specifier resolves with). Bumping any subset publishes a package that resolves
against a version nobody released.

The count was recorded in this ledger three times and was wrong twice — "seven strings across three
files" (Epic-08), then "TEN strings in FOUR files" (the hub) — because the set grows with the
packaging: `rt-metal` added two, and P7's ARMv6 wheel added a fifth file that is not a package at all.
PyPI refuses the `linux_armv6l` tag, so the Pi Zero wheel is a GitHub release asset installed by URL,
and **the version is in the filename** — `releases/latest/download/loom_py_rt-1.0.0rc10-…whl`. A
missed bump there is a 404 the moment the release ships, visible only to the users with the smallest
boards. A number nobody can count by hand is one nobody should have to.

## Decision

**`VERSION` at loom-py's repo root is the version. Every other copy is derived, written only by
`packaging/version.py`, and proved by a CI test.**

```sh
python packaging/version.py --set 1.0.0-rc11   # writes VERSION and all eleven derived copies
python packaging/version.py                    # the check: is every copy in step?
```

The tool holds a table of `Site` rows — path, pattern, which spelling belongs there — derives the
PEP 440 form by dropping one hyphen, and requires **exactly one match per site**, because a pattern
that silently stops matching would make the check pass while checking nothing.

**The derived copies stay literal text.** scikit-build-core can read a version out of a file at build
time, but the pins live in `dependencies`/`optional-dependencies`, and the three backend wheels are
built by cibuildwheel from a **staged tree** (`packaging/stage.py`) holding only the package directory
— so a build-time read of a repo-root file is unavailable to exactly the builds that need it most.
Static text a test proves correct beats dynamic text that resolves in only some of the builds.

**The test is what makes `VERSION` authoritative**, not the tool:
`tests/ci/test_version_consistency.py`, one assertion per site, run by `ci.yml` on every push **and
inside every wheel cibuildwheel builds** (`test-command` runs `pytest {project}/tests/ci`), which is
the last moment before an upload. It imports no `loom`: this is a property of the source tree and
should still fail on a checkout where the extension was never built.

**The twelfth copy lives outside the tree — the git tag** — so `wheels.yml` gains a `version-guard`
job that `publish-pypi` needs, running the check with `--expect-tag`. A tag disagreeing with `version`
re-publishes the *previous* number, and `skip-existing: true` then swallows that as a success.

## Alternatives not taken

* **A compatible-release range (`~=`) instead of the exact pin**, which would make most of the copies
  unnecessary. Rejected in ADR-009's terms: it lets pip pair a backend with a base wheel built against
  a different ggml, and the failure surfaces as a missing accelerator in `loom.devices()` — or a
  symbol error at model load, or nothing at all.
* **Dynamic metadata (scikit-build-core's regex provider, or setuptools-scm's git-tag versioning).**
  It covers `version` and not the pins, it is unavailable in the staged backend builds, and a release
  branch is the worst place to find out which of the nine wheel jobs disagrees.
* **Deriving the version from the git tag.** The tag is the *claim*; the tree is what is uploaded.
  Making the tag authoritative moves the single point of failure rather than removing it, and CI tests
  a branch, where no tag exists.

## Consequences

* A bump is one edit and one command, and a partial bump cannot be pushed green.
* **Adding a backend package now touches the table too** — two `Site` rows for the new pyproject and
  one for its extra. `packaging/README.md`'s "Adding a backend" list says so as step 5; the test needs
  no change, being driven by the table.
* The ledger no longer carries an arithmetic that goes stale. It carries the file name.
* `VERSION` is plain text on purpose: `cat VERSION` answers "what is this tree?" without a toolchain,
  and CMake or a shell script can read it if one ever needs to.

Serves [Epic-08](../epics/epic-08-packaging-and-release.md). Shipped on loom-py's `release/1.0.0-rc10`
branch with the rc10 bump itself, which is the first bump performed by the tool.
