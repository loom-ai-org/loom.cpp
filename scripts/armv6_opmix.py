# ADR-077 gate 1's cross-model analysis: one model's board profile, by op class, with the FLOPs of
# every GEMM-shaped class, so a class carries seconds AND a rate.
#
#   python3 scripts/armv6_opmix.py <board.prof> <shapes.prof> [--nodes]
#
# <board.prof>  $LOOM_PROFILE written on the Pi Zero with LOOM_PROFILE_NODES=1 (armv6_opmix_profile.sh):
#               the times. Each node bucket loses the per-node floor (the profiler's own cost).
# <shapes.prof> the same model and input on any machine with LOOM_PROFILE_NODES=src: the first two
#               sources of every node, which the board's engine does not print. The graphs are the
#               same on both (an unmatched bucket is reported), so a bucket's (op, name, ne) finds them.
#               A board bucket that matches several shape buckets is split by their calls.
import argparse
import collections
import re

ROW = re.compile(r"^\s*([\d.]+)\s+(\d+)\s+(\S+)\s+(.*?)\s+(-?\d+,-?\d+,-?\d+,-?\d+)(?:\s+(\S+)\s+(\S+))?\s*$")

# (class, what int8 could do to it). The classes ADR-077 names, in its order.
CLASSES = ["gemm", "gemv", "conv_k", "conv_dw", "conv_transpose", "dft", "attn_mm", "softmax", "norm",
           "ew_bcast", "ew_act", "ew_same", "ew_other", "layout", "other"]
ELEMENTWISE = {"ADD", "ADD1", "MUL", "SUB", "DIV", "SCALE", "UNARY", "GLU", "SQR", "SQRT", "LOG", "EXP",
               "CLAMP", "LEAKY_RELU", "SUM_ROWS", "SUM", "MEAN", "ARGMAX", "SIN", "COS", "ABS", "NEG",
               "STEP", "SGN", "ADD_ID", "COUNT_EQUAL", "TANH", "SIGMOID", "ADD_REL_POS", "SOFTPLUS"}
LAYOUT = {"CONT", "CPY", "PERMUTE", "RESHAPE", "VIEW", "TRANSPOSE", "CONCAT", "PAD", "GET_ROWS", "REPEAT",
          "SET_ROWS", "ARANGE", "IM2COL", "ROLL", "PAD_REFLECT_1D", "GET_ROWS_BACK", "REPEAT_BACK", "DUP",
          "TIMESTEP_EMBEDDING", "FILL"}
NORM = {"NORM", "RMS_NORM", "GROUP_NORM", "L2_NORM"}


def parse(path, with_srcs):
    floor, rows, in_nodes = 0.0, [], False
    for line in open(path):
        if line.startswith("per-node floor"):
            # The profiler's floor is its cheapest computing node, which is the dispatch cost only
            # where a graph HAS a trivial node: WakeHuBERT's 3.5 ms is a real op. Capped at the
            # boards' 0.044-0.057 ms over models that do.
            floor = min(float(line.split()[2]) / 1e3, 60e-6)
        if line.split()[:4] == ["ms", "calls", "op", "name"]:
            in_nodes = True
            continue
        if not in_nodes:
            continue
        m = ROW.match(line)
        if not m:
            continue
        ms, calls, op, name, ne = float(m[1]), int(m[2]), m[3], m[4], tuple(int(v) for v in m[5].split(","))
        srcs = (m[6], m[7]) if with_srcs and m[6] else None
        rows.append(dict(s=ms / 1e3, calls=calls, op=op, name=name, ne=ne, srcs=srcs))
    return floor, rows


def src_shape(s):
    if s is None or s == "-":
        return None, None
    typ, ne = s.split(":")
    return typ, tuple(int(v) for v in ne.split(","))


def classify(op, ne, srcs):
    """(class, GFLOP per call) for one node."""
    t0, a = src_shape(srcs[0]) if srcs else (None, None)
    t1, b = src_shape(srcs[1]) if srcs else (None, None)
    n_out = ne[0] * ne[1] * ne[2] * ne[3]
    if op == "MUL_MAT":
        flop = 2.0 * a[0] * n_out if a else 0.0
        # A weight has no batch axes; attention's QK^T / AV carry the heads in ne2. A weight GEMM with
        # one or two columns (a decode step, a recurrent step) streams the whole weight per call and is
        # bound by DRAM, not the MACs: its own class.
        if a and a[2] * a[3] == 1:
            return ("gemv" if ne[1] * ne[2] * ne[3] <= 2 else "gemm"), flop
        return "attn_mm", flop
    if op == "MUL_MAT_ID":
        return "gemm", 2.0 * a[0] * n_out if a else 0.0
    if op in ("CONV_2D", "CONV_1D"):
        if a and t0 not in ("f32", "f16"):
            # conv_2d_direct_packed (ggml-0013): a quantized kernel is folded to [IC * KW * KH, OC].
            k = a[0] // b[2] if b else 0
            flop = 2.0 * a[0] * n_out
        else:
            k = a[0] * a[1] if a else 0
            flop = 2.0 * (a[0] * a[1] * a[2]) * n_out if a else 0.0
        return ("gemm" if k == 1 else "conv_k"), flop   # a 1x1 conv is a GEMM in any precision
    if op in ("CONV_2D_DW", "CONV_1D_DW", "CONV_2D_DW_DIRECT"):
        return "conv_dw", 2.0 * (a[0] * a[1]) * n_out if a else 0.0
    if op == "CONV_TRANSPOSE_1D":
        # kernel [K, OC, IC], input [L, IC]: every input sample scatters K x OC products per IC.
        return "conv_transpose", 2.0 * a[0] * a[1] * a[2] * b[0] if a and b else 0.0
    if op in ("SOFT_MAX", "FLASH_ATTN_EXT"):
        return ("softmax" if op == "SOFT_MAX" else "attn_mm"), 0.0
    if op in NORM:
        return "norm", 0.0
    if op in ELEMENTWISE:
        # Split by what removes it: a broadcast ADD/MUL (a bias, a mask) and an activation fold into the
        # producing op's epilogue in either precision; a same-shape one (a residual, a gate) needs a
        # fused epilogue with a second input; the rest (reductions, sin/cos) stays.
        if op in ("UNARY", "LEAKY_RELU", "GLU", "CLAMP", "TANH", "SIGMOID", "SOFTPLUS"):
            return "ew_act", 0.0
        if op in ("ADD", "MUL", "SUB", "DIV") and a and b:
            return ("ew_bcast" if b[0] * b[1] * b[2] * b[3] < a[0] * a[1] * a[2] * a[3] else "ew_same"), 0.0
        return "ew_other", 0.0
    if op in LAYOUT:
        return "layout", 0.0
    return "other", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board")
    ap.add_argument("shapes")
    ap.add_argument("--nodes", action="store_true", help="also list the heaviest GEMM-class buckets")
    ap.add_argument("--json", help="also write {class: [seconds, GFLOP, M output elements]} here")
    args = ap.parse_args()
    floor, board = parse(args.board, False)
    _, shapes = parse(args.shapes, True)
    by_key = collections.defaultdict(list)
    for r in shapes:
        by_key[(r["op"], r["name"], r["ne"])].append(r)

    sec = collections.Counter()
    elems = collections.Counter()   # output elements, summed over calls: what an island (de)quantizes
    flop = collections.Counter()
    calls = collections.Counter()
    unmatched, total, heavy = 0.0, 0.0, []
    for r in board:
        s = max(0.0, r["s"] - floor * r["calls"])
        total += s
        cands = by_key.get((r["op"], r["name"], r["ne"]))
        if not cands:
            cands = [dict(calls=r["calls"], srcs=None)]
            if r["op"] in ("MUL_MAT", "CONV_2D", "CONV_2D_DW", "CONV_TRANSPOSE_1D", "MUL_MAT_ID"):
                unmatched += s
        n = sum(c["calls"] for c in cands)
        for c in cands:
            share = c["calls"] / n
            cls, f = classify(r["op"], r["ne"], c["srcs"])
            sec[cls] += s * share
            elems[cls] += r["ne"][0] * r["ne"][1] * r["ne"][2] * r["ne"][3] * r["calls"] * share
            flop[cls] += f * r["calls"] * share
            calls[cls] += r["calls"] * share
            if f:
                heavy.append((s * share, r["calls"] * share, cls, r["op"], r["ne"], c["srcs"], f))
            # A DFT basis as a convolution: one input channel and a kernel of >= 256 taps (STFT), or
            # a transposed conv with one output channel and >= 256 taps (iSTFT). An FFT replaces it
            # in either precision, so the projections take it out of the GEMM classes.
            a = src_shape(c["srcs"][0])[1] if c["srcs"] else None
            b = src_shape(c["srcs"][1])[1] if c["srcs"] else None
            if a and b and ((r["op"] == "CONV_2D" and b[2] == 1 and a[0] >= 256)
                            or (r["op"] == "CONV_TRANSPOSE_1D" and a[1] == 1 and a[0] >= 256)):
                n_el = r["ne"][0] * r["ne"][1] * r["ne"][2] * r["ne"][3] * r["calls"] * share
                sec["dft"] += s * share
                sec[cls] -= s * share
                flop[cls] -= f * r["calls"] * share
                elems[cls] -= n_el
                elems["dft"] += n_el

    print(f"{'class':16} {'s':>8} {'%':>6} {'calls':>8} {'GFLOP':>8} {'GOP/s':>7}")
    for cls in CLASSES:
        if not sec[cls]:
            continue
        rate = flop[cls] / sec[cls] / 1e9 if flop[cls] and sec[cls] else 0
        print(f"{cls:16} {sec[cls]:8.3f} {100 * sec[cls] / total:5.1f}% {calls[cls]:8.0f} "
              f"{flop[cls] / 1e9:8.3f} {rate:7.3f}")
    print(f"{'total':16} {total:8.3f}   (floor {floor * 1e3:.3f} ms per node removed)")
    if args.json:
        import json
        json.dump({c: [sec[c], flop[c] / 1e9, elems[c] / 1e6] for c in CLASSES if sec[c]}, open(args.json, "w"), indent=1)
    if unmatched:
        print(f"WARNING: {unmatched:.3f} s of GEMM-shaped nodes found no shapes (FLOPs undercounted)")
    if args.nodes:
        heavy.sort(key=lambda h: -h[0])
        print(f"\n{'s':>8} {'calls':>6} {'GOP/s':>6}  class / op / ne / srcs")
        for s, c, cls, op, ne, srcs, f in heavy[:25]:
            print(f"{s:8.3f} {c:6.0f} {f * c / s / 1e9 if s else 0:6.3f}  {cls} {op} {ne} {srcs}")


if __name__ == "__main__":
    main()
