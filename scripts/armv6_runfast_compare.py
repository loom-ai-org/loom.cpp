#!/usr/bin/env python3
"""ADR-078 gates 2 and 4: compare the board's IEEE and RunFast runs, model by model.

  scripts/armv6_runfast_compare.py <dir pulled from the Pi: gate.log, models.txt, out/>

Gate 2 (numerics): every run's output against the same model's IEEE run of the same round; also each
mode against itself across rounds (the engine must be deterministic, or the comparison means nothing).
Gate 4 (speed): the infer time per mode, two rounds, ABBA.
"""
import re
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
runs = {}
for line in (d / "gate.log").read_text().splitlines():
    m = re.match(r"RESULT kind=(\S+) model=(\S+) runfast=(\S+) infer=([\d.]+)s n=(\d+) sha=(\S+)", line)
    if m:
        runs.setdefault(m.group(2), []).append((m.group(3), float(m.group(4)), m.group(6)))

names = [l.split()[3] for l in (d / "models.txt").read_text().splitlines() if l.strip()]
ggufs = [Path(l.split()[1]).name for l in (d / "models.txt").read_text().splitlines() if l.strip()]


def load(name, mode, rnd):
    f = d / "out" / f"{name}.rf{mode}.r{rnd}.f32"
    if f.exists():
        return np.fromfile(f, np.float32)
    t = d / "out" / f"{name}.rf{mode}.r{rnd}.txt"
    return t.read_text() if t.exists() else None


print(f"{'model':12s} {'IEEE s':>15s} {'RunFast s':>15s} {'speed-up':>8s}   numerics (RunFast vs IEEE, same round)")
for name, gguf in zip(names, ggufs):
    r = runs.get(gguf, [])
    t0 = [t for m, t, _ in r if m == "0"]
    t1 = [t for m, t, _ in r if m == "1"]
    det = all(load(name, m, 1) is not None and
              (np.array_equal(load(name, m, 1), load(name, m, 2)) if isinstance(load(name, m, 1), np.ndarray)
               else load(name, m, 1) == load(name, m, 2)) for m in (0, 1))
    a, b = load(name, 0, 1), load(name, 1, 1)
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            num = f"LENGTH DIFFERS {a.shape} vs {b.shape}"
        else:
            diff = np.abs(a.astype(np.float64) - b)
            ndiff = int((a.view(np.uint32) != b.view(np.uint32)).sum())
            num = (f"{ndiff}/{a.size} elements differ, max |d| {diff.max():.3g}, "
                   f"max |d|/max|ref| {diff.max() / max(np.abs(a).max(), 1e-30):.3g}")
    else:
        num = "text identical" if a == b else f"TEXT DIFFERS: {a!r} vs {b!r}"
    sp = (np.mean(t0) / np.mean(t1)) if t0 and t1 else float("nan")
    print(f"{name:12s} {'/'.join(f'{t:.2f}' for t in t0):>15s} {'/'.join(f'{t:.2f}' for t in t1):>15s} "
          f"{sp:7.2f}x   {num}{'' if det else '   [NOT DETERMINISTIC across rounds]'}")
