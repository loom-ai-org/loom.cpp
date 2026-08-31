#!/usr/bin/env python3
"""What a GGUF's weights ARE, bucketed by what reads them -- and how much of them is zeros.

WHY THIS EXISTS (P4.28). A quantized export reports a coverage percentage, and that number answers
"what fraction of the float bytes moved" and NOT "should those bytes have been there". VITS reported a
healthy 73% with no warning while **62% of the file it wrote was zeros** -- 18.9 MB of constant-folded
relative-position padding that no quantization gate could ever reach, because the tensors are a VIEW's
source and only a mul_mat's FIRST operand is eligible to be packed.

THE METHOD, WHICH IS THE TRANSFERABLE PART. Bucket the tensors a file left as F32 by **what reads them
and in which operand position**, not by name. VITS's buckets came out 86.8% "second operand / GET_ROWS
/ other" -- and a bucket named for what it is *not* is exactly where a thing nobody is looking at hides.

    scripts/weight_census.py model.gguf              # one file, both analyses
    scripts/weight_census.py hf-models/*/*.gguf      # the sweep: which model has a problem at all

Run with any python that has `gguf` and `numpy` (`~/.venvs/piper/bin/python3`).
"""
import collections
import json
import math
import os
import sys

import numpy as np
from gguf import GGUFReader

# The ops whose FIRST input the engine can read in a packed form -- keep in step with
# `LoomGGUFExporter.PACKED_WEIGHT_FIRST_OPS`. A weight outside this set is not quantizable wherever it
# sits, which is the distinction the buckets below exist to draw.
PACKED_FIRST = ("MUL_MAT", "CONV_1D", "CONV_2D", "CONV_1D_DW", "CONV_2D_DW", "SHORT_CONV")
FOLDABLE = ("CONV_1D", "CONV_2D")
ZERO_MIN_BYTES = 200_000     # below this a zero-heavy tensor is a bias vector, not a finding
ZERO_SHARE = 0.5


def topologies(reader):
    kv = {f.name: f for f in reader.fields.values()}
    for name in kv:
        if name.startswith("model.graph_topology"):
            f = kv[name]
            yield name, json.loads(str(bytes(f.parts[f.data[0]]), "utf-8"))


def census(path):
    reader = GGUFReader(path)
    use = collections.defaultdict(set)
    for _, topo in topologies(reader):
        for node in topo.get("nodes", []):
            for pos, name in enumerate(node.get("inputs") or []):
                use[name].add((node.get("op"), pos))

    f32_total = sum(t.n_bytes for t in reader.tensors if t.tensor_type.name == "F32")
    buckets, byte_counts = collections.Counter(), collections.Counter()
    zero_heavy = []
    for t in reader.tensors:
        if t.tensor_type.name != "F32":
            continue
        u = use.get(t.name, set())
        first = {op for op, pos in u if pos == 0}
        if not u:
            key = "unreferenced (driver weights)"
        elif first & set(FOLDABLE):
            key = "dense conv kernel, unaligned even folded"
        elif first & {"CONV_1D_DW", "CONV_2D_DW", "SHORT_CONV"}:
            key = "depthwise conv (never folded)"
        elif {"CONV_TRANSPOSE_1D", "CONV_TRANSPOSE_2D"} & {op for op, _ in u}:
            key = "CONV_TRANSPOSE kernel (native op, no mul_mat)"
        elif first & {"MUL_MAT"}:
            key = "MUL_MAT first operand, unaligned"
        elif len(t.shape) < 2 or int(t.shape[0]) == 1:
            key = "1-D / scalar (norms, biases)"
        else:
            key = "second operand / GET_ROWS / other"
        buckets[key] += 1
        byte_counts[key] += t.n_bytes

        if t.n_bytes >= ZERO_MIN_BYTES:
            share = float((np.array(t.data).reshape(-1) == 0).mean())
            if share > ZERO_SHARE:
                zero_heavy.append((t.n_bytes, t.name, share, [int(d) for d in t.shape], sorted(u)))

    return f32_total, buckets, byte_counts, zero_heavy


def main(paths):
    one = len(paths) == 1
    for path in paths:
        f32_total, buckets, byte_counts, zero_heavy = census(path)
        zb = sum(b for b, *_ in zero_heavy)
        name = os.path.basename(path)
        if one:
            print(f"{name}: {f32_total / 1e6:.1f} MB still F32\n")
            print(f"  {'bucket':44s} {'tensors':>8s} {'MB':>9s} {'%':>6s}")
            for key, b in byte_counts.most_common():
                print(f"  {key:44s} {buckets[key]:8d} {b / 1e6:9.3f} "
                      f"{100 * b / max(f32_total, 1):5.1f}%")
            print(f"\n  zero-heavy constants (>{ZERO_MIN_BYTES // 1000} KB, >{ZERO_SHARE:.0%} zeros): "
                  f"{zb / 1e6:.2f} MB, {100 * zb / max(f32_total, 1):.1f}% of the float weights")
            for b, tname, share, shape, u in sorted(zero_heavy, reverse=True)[:8]:
                print(f"    {b / 1e6:8.3f} MB  {100 * share:5.1f}% zeros  {tname:44s} ne={shape} "
                      f"read by {u}")
        else:
            top = "; ".join(f"{n} {100 * s:.1f}%z {b / 1e6:.1f}MB"
                            for b, n, s, _, _ in sorted(zero_heavy, reverse=True)[:2])
            print(f"{name:34s} {f32_total / 1e6:9.1f} MB F32  zero-heavy {zb / 1e6:8.2f} MB "
                  f"{100 * zb / max(f32_total, 1):5.1f}%  {top}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
