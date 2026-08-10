#!/usr/bin/env python3
"""What the gate suite needs, whether it is here, and where to get it.

`tests/gate/` compares against real exported checkpoints: GGUFs the loom-exporter produced and
directories of reference tensors a HuggingFace forward pass wrote. They are gigabytes, they take hours
to rebuild, and several of them derive from checkpoints under licences that are not this repo's to
redistribute. So they are not in git, and a developer with none of them still gets a green suite --
every gate test exits 77 (Skipped) when its fixture is absent.

This script is the other half of that arrangement: it says what is missing, verifies what is present,
and fetches what it can.

    scripts/fixtures.py status          # what the gate suite wants, and what you have
    scripts/fixtures.py verify          # checksum everything present against the manifest
    scripts/fixtures.py record NAME...  # write the checksums of fixtures you just built
    scripts/fixtures.py fetch [NAME...] # download from the published fixture repo

**The layout is a rule, not a table** -- the same rule `tests/support/fixtures.h` applies, and
`tests/ci/test_fixture_resolution.cpp` pins both ends of it. Drop `LOOM_`, lowercase, a `_GGUF`
suffix means a file and a `_DIR` suffix is dropped as noise. `LOOM_KOKORO_MIL_GGUF` lives at
`$LOOM_FIXTURES/kokoro_mil.gguf`; `LOOM_KOKORO_ALBERT_REF_DIR` at `$LOOM_FIXTURES/kokoro_albert_ref/`.
Two implementations of one rule is one more than ideal; a hand-maintained mapping of sixty-eight
variables to sixty-eight paths would have been sixty-eight more things to drift.

**`manifest.json` is derived from the tests that read it.** Its `used_by` lists come from scanning
`tests/gate/` for `fixture_env(...)` calls, so the manifest cannot claim a fixture nobody wants or
miss one somebody does -- `scripts/fixtures.py scan` regenerates it and is what keeps that true.
Checksums arrive later, from `record`, once a fixture actually exists on some machine.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "fixtures" / "manifest.json"
GATE = REPO / "tests" / "gate"


def relpath_for(var: str) -> str:
    """`tests/support/fixtures.h`'s `fixture_relpath`, in Python.

    Kept deliberately literal rather than clever, because the C++ side is the one that decides whether
    a test runs and this one only has to agree with it. `test_fixture_resolution.cpp` checks the same
    five cases on the other side.
    """
    stem = var[5:] if var.startswith("LOOM_") else var
    if stem.endswith("_GGUF"):
        return stem[:-5].lower() + ".gguf"
    if stem.endswith("_DIR"):
        stem = stem[:-4]
    return stem.lower()


def fixtures_root() -> Path:
    root = os.environ.get("LOOM_FIXTURES")
    if not root:
        sys.exit(
            "LOOM_FIXTURES is not set. It names the one directory holding every gate fixture; the\n"
            "gate suite reads it, this script writes it. Pick a path with room for tens of gigabytes:\n"
            "    export LOOM_FIXTURES=~/loom-fixtures"
        )
    return Path(root)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} does not exist yet -- run `scripts/fixtures.py scan` to derive it.")
    return json.loads(MANIFEST.read_text())


def scan_gate_tests() -> dict:
    """Which fixture each gate test asks for, read off the tests themselves."""
    used: dict[str, set[str]] = {}
    for path in sorted(GATE.glob("*.cpp")):
        text = path.read_text()
        names = set(re.findall(r'fixture_env\("(LOOM_[A-Z0-9_]+)"\)', text))
        # One test names its variables through a table of structs rather than inline, so the literal
        # scan misses them. Picking the string literals out of the whole file catches those without
        # pretending to parse C++.
        if "fixture_env(fam." in text or "fixture_env(entry." in text:
            names |= set(re.findall(r'"(LOOM_[A-Z0-9_]+)"', text))
        for name in names:
            used.setdefault(name, set()).add(path.stem)
    return used


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(path: Path) -> dict:
    """A checksum for a file; for a directory, a checksum over its sorted (relpath, sha256) listing --
    so a reference directory is one comparable value the way a GGUF is, and a fixture that gained or
    lost a file is a mismatch rather than a silent pass."""
    if path.is_file():
        return {"kind": "file", "sha256": sha256_of(path), "bytes": path.stat().st_size}
    listing, total = [], 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        listing.append(f"{child.relative_to(path)}:{sha256_of(child)}")
        total += child.stat().st_size
    return {
        "kind": "dir",
        "sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
        "bytes": total,
        "files": len(listing),
    }


def cmd_scan(_args) -> int:
    used = scan_gate_tests()
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old_entries = previous.get("fixtures", {})
    entries = {}
    for var in sorted(used):
        kept = old_entries.get(var, {})
        entries[var] = {
            "path": relpath_for(var),
            "used_by": sorted(used[var]),
            # Carried forward: these are facts about an artifact, not about the tests, and rescanning
            # the tests must not discard them.
            "sha256": kept.get("sha256"),
            "bytes": kept.get("bytes"),
            "kind": kept.get("kind"),
            "produced_by": kept.get("produced_by"),
        }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "comment": "Derived from tests/gate/ by scripts/fixtures.py scan. Do not hand-edit `path` or "
                   "`used_by`; `sha256`/`bytes`/`produced_by` are filled in by `record`.",
        "repo": previous.get("repo", "loom-ai-org/loom-fixtures"),
        "revision": previous.get("revision", "main"),
        "fixtures": entries,
    }, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)}: {len(entries)} fixtures across "
          f"{len({t for ts in used.values() for t in ts})} gate tests")
    return 0


def cmd_status(_args) -> int:
    manifest = load_manifest()
    root = fixtures_root()
    present = missing = 0
    for var, entry in manifest["fixtures"].items():
        path = root / entry["path"]
        mark = "present" if path.exists() else "MISSING"
        if path.exists():
            present += 1
        else:
            missing += 1
        print(f"  {mark:8} {entry['path']:44} {var:44} ({len(entry['used_by'])} test(s))")
    print(f"\n{present} present, {missing} missing, under {root}")
    print(f"{'all gate fixtures are here' if not missing else f'`ctest -L gate` will skip {missing} fixtures worth of tests'}")
    return 0


def cmd_verify(_args) -> int:
    manifest = load_manifest()
    root = fixtures_root()
    checked = bad = unrecorded = 0
    for var, entry in manifest["fixtures"].items():
        path = root / entry["path"]
        if not path.exists():
            continue
        if not entry.get("sha256"):
            unrecorded += 1
            continue
        actual = measure(path)
        checked += 1
        if actual["sha256"] != entry["sha256"]:
            bad += 1
            print(f"  MISMATCH {entry['path']}\n    manifest {entry['sha256']}\n    actual   {actual['sha256']}")
    print(f"{checked} verified, {bad} mismatched, {unrecorded} present but never recorded")
    return 1 if bad else 0


def cmd_record(args) -> int:
    manifest = load_manifest()
    root = fixtures_root()
    targets = args.names or list(manifest["fixtures"])
    recorded = 0
    for var in targets:
        entry = manifest["fixtures"].get(var)
        if entry is None:
            print(f"  unknown fixture {var}")
            continue
        path = root / entry["path"]
        if not path.exists():
            continue
        entry.update(measure(path))
        if args.produced_by:
            entry["produced_by"] = args.produced_by
        recorded += 1
        print(f"  recorded {entry['path']} ({entry['bytes']:,} bytes)")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{recorded} fixture(s) recorded in {MANIFEST.relative_to(REPO)}")
    return 0


def cmd_fetch(args) -> int:
    manifest = load_manifest()
    root = fixtures_root()
    repo_id = manifest.get("repo")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("fetch needs `pip install huggingface_hub`.")

    targets = args.names or list(manifest["fixtures"])
    wanted = [manifest["fixtures"][v]["path"] for v in targets if v in manifest["fixtures"]]
    wanted = [p for p in wanted if not (root / p).exists()]
    if not wanted:
        print("nothing to fetch: every requested fixture is already present")
        return 0
    print(f"fetching {len(wanted)} fixture(s) from {repo_id} into {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=manifest.get("revision", "main"),
            local_dir=str(root), allow_patterns=[f"{p}*" for p in wanted],
        )
    except Exception as error:  # noqa: BLE001 -- the point is to name the repo in the message
        sys.exit(
            f"could not fetch from {repo_id!r}: {error}\n\n"
            f"That dataset repo is where these fixtures are meant to be published; until it exists, "
            f"build them locally with loom-exporter and register them with\n"
            f"    scripts/fixtures.py record"
        )
    return cmd_verify(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="re-derive the manifest from tests/gate/").set_defaults(fn=cmd_scan)
    sub.add_parser("status", help="what is present under $LOOM_FIXTURES").set_defaults(fn=cmd_status)
    sub.add_parser("verify", help="checksum what is present").set_defaults(fn=cmd_verify)
    rec = sub.add_parser("record", help="record checksums of fixtures you built")
    rec.add_argument("names", nargs="*")
    rec.add_argument("--produced-by", help="the command that built them, for the manifest")
    rec.set_defaults(fn=cmd_record)
    fetch = sub.add_parser("fetch", help="download from the published fixture repo")
    fetch.add_argument("names", nargs="*")
    fetch.set_defaults(fn=cmd_fetch)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
