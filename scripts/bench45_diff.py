#!/usr/bin/env python3
"""Compare two bench45 dumps (qemu and the board, say) case by case.

ADR-078 gate 1. For every instruction and both modes it counts the cases whose result or flags differ,
classifies each by what its operands and the two results are (NaN, denormal, zero, normal, inf), and
prints the first few with their bits, so a disagreement can be named rather than just counted.

  python3 scripts/bench45_diff.py qemu.bin board.bin [cases per instruction, default 20000]
"""
import collections
import struct
import sys

import numpy as np

OPS = ["vadd.f32", "vsub.f32", "vmul.f32", "vnmul.f32", "vdiv.f32", "vmla.f32", "vmls.f32", "vnmla.f32",
       "vsqrt.f32", "vabs.f32", "vneg.f32", "vcvtr.s32.f32", "vcvt.s32.f32", "vcmp.f32", "vcmpe.f32",
       "vcvt.f64.f32", "vcvt.f32.f64", "vadd.f64", "vmul.f64", "vdiv.f64", "vmla.f64", "vsqrt.f64"]
F64_IN = {"vcvt.f32.f64", "vadd.f64", "vmul.f64", "vdiv.f64", "vmla.f64", "vsqrt.f64"}
F64_OUT = F64_IN - {"vcvt.f32.f64"}
FLAG = {0x80: "IDC", 0x10: "IXC", 0x08: "UFC", 0x04: "OFC", 0x02: "DZC", 0x01: "IOC"}


def cls(u, f64):
    exp, man = ((u >> 52) & 0x7ff, u & 0xfffffffffffff) if f64 else ((u >> 23) & 0xff, u & 0x7fffff)
    top = 0x7ff if f64 else 0xff
    qbit = 1 << (51 if f64 else 22)
    if exp == top:
        return "inf" if man == 0 else ("qnan" if man & qbit else "snan")
    if exp == 0:
        return "zero" if man == 0 else "denorm"
    return "norm"


def flags(f):
    return "|".join(n for b, n in FLAG.items() if f & b) or "-"


def read(path, n):
    raw = open(path, "rb").read()
    per = n * (3 * 8 + 2 * (8 + 4))
    assert len(raw) == per * len(OPS), (len(raw), per * len(OPS))
    out = []
    for o in range(len(OPS)):
        b = raw[o * per:(o + 1) * per]
        ins = np.frombuffer(b, np.uint64, 3 * n).reshape(n, 3)
        off = 3 * n * 8
        modes = []
        for _ in range(2):
            r = np.frombuffer(b, np.uint64, n, off); off += 8 * n
            f = np.frombuffer(b, np.uint32, n, off); off += 4 * n
            modes.append((r, f))
        out.append((ins, modes))
    return out


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    A, B = read(a_path, n), read(b_path, n)
    for o, name in enumerate(OPS):
        ins = A[o][0]
        assert (ins == B[o][0]).all(), f"{name}: the two runs drew different operands"
        for m, mode in enumerate(("ieee", "runfast")):
            (ra, fa), (rb, fb) = A[o][1][m], B[o][1][m]
            dr, df = np.nonzero(ra != rb)[0], np.nonzero(fa != fb)[0]
            if not len(dr) and not len(df):
                continue
            f64i, f64o = name in F64_IN, name in F64_OUT
            nin = 1 if name in ("vsqrt.f32", "vsqrt.f64", "vabs.f32", "vneg.f32", "vcvtr.s32.f32",
                                "vcvt.s32.f32", "vcvt.f64.f32", "vcvt.f32.f64") else (3 if "mla" in name or "mls" in name else 2)
            kinds = collections.Counter()
            for i in dr:
                k = ",".join(cls(int(ins[i][j]), f64i) for j in range(nin))
                outk = "" if name.startswith(("vcvt.s32", "vcvtr", "vcmp", "vcvt.f64")) else \
                    f" -> {cls(int(ra[i]), f64o)}/{cls(int(rb[i]), f64o)}"
                kinds[k + outk] += 1
            fkinds = collections.Counter(f"{flags(int(fa[i]))} / {flags(int(fb[i]))}" for i in df)
            print(f"{name:14s} {mode:7s} results differ {len(dr):6d}  flags differ {len(df):6d}")
            for k, c in kinds.most_common(6):
                print(f"    result  {c:6d}  {k}")
            for k, c in fkinds.most_common(4):
                print(f"    flags   {c:6d}  {k}")
            for i in dr[:3]:
                w = 16 if f64i else 8
                print("    e.g. " + " ".join(f"{int(ins[i][j]) & ((1 << (4 * w)) - 1):0{w}x}" for j in range(nin))
                      + f"  ->  {int(ra[i]):x} / {int(rb[i]):x}")


if __name__ == "__main__":
    main()
