#!/usr/bin/env python3
"""Snapshot a GGUF's *structure* into a directory of diffable text files, so an exporter change can be
proven output-preserving instead of merely believed to be.

This is the gate every refactor in BACKEND.md was held to, and it is the reason a mechanical extraction
bug (three silently dropped `LESS` nodes in Kokoro's decoder_vocoder topology) was caught at all -- no
existing test covered it, because the emitted model was still structurally valid, just wrong.

Writes per model: `kv.txt` (every metadata KV, sorted), one pretty-printed key-sorted `.json` per
`model.graph_topology.*` entry, the embedded `model.driver_script` Lua verbatim, and `tensors.txt` with
one `name / shape / dtype / sha256-of-data` line per tensor. Everything is sorted and line-oriented, so
`diff -r` over two snapshot dirs is the whole comparison.

Typical use -- prove a change is output-preserving:

    git archive HEAD | tar -x -C /tmp/baseline        # or any known-good tree
    (cd /tmp/baseline && python3 export_vits_mil.py)  # re-run the exports there
    python3 tools/loom_mil_compiler/snapshot_gguf.py /tmp/snap_base /tmp/baseline/*.gguf
    python3 export_vits_mil.py                        # then in the working tree
    python3 tools/loom_mil_compiler/snapshot_gguf.py /tmp/snap_new  *.gguf
    diff -r /tmp/snap_base /tmp/snap_new              # must be empty

Note the `.gguf` files committed in the repo tree are `.gitignore`d build outputs and are routinely
stale -- regenerate the baseline, never diff against them.

Usage: snapshot_gguf.py <out_dir> <model.gguf> [more.gguf ...]
"""
import hashlib
import json
import sys
from pathlib import Path

from gguf import GGUFReader


def snapshot(path: Path, out_dir: Path):
    r = GGUFReader(str(path))
    stem = path.stem
    base = out_dir / stem
    base.mkdir(parents=True, exist_ok=True)

    kv_lines = []
    for key, field in r.fields.items():
        if key.startswith("GGUF."):
            continue
        val = field.contents()
        if isinstance(val, str) and (key.startswith("model.graph_topology") or key.endswith("driver_script")):
            # Pretty-print structured payloads into their own files for readable diffs.
            fname = key.replace(".", "_")
            try:
                parsed = json.loads(val)
                (base / f"{fname}.json").write_text(json.dumps(parsed, indent=2, sort_keys=True))
            except (json.JSONDecodeError, TypeError):
                (base / f"{fname}.txt").write_text(val)
            kv_lines.append(f"{key} = <file {fname}> sha256={hashlib.sha256(val.encode()).hexdigest()[:16]}")
        else:
            kv_lines.append(f"{key} = {val!r}")
    (base / "kv.txt").write_text("\n".join(sorted(kv_lines)) + "\n")

    tensor_lines = []
    for t in r.tensors:
        digest = hashlib.sha256(t.data.tobytes()).hexdigest()[:16]
        tensor_lines.append(f"{t.name}\t{list(t.shape)}\t{t.tensor_type.name}\t{digest}")
    (base / "tensors.txt").write_text("\n".join(sorted(tensor_lines)) + "\n")
    print(f"snapshot {path} -> {base} ({len(tensor_lines)} tensors, {len(kv_lines)} kvs)")


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    out_dir = Path(sys.argv[1])
    for p in sys.argv[2:]:
        snapshot(Path(p), out_dir)


if __name__ == "__main__":
    main()
