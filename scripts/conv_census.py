"""Every convolution a GGUF's topologies issue, with its real shapes -- and, optionally, the same
census of an ONNX graph to line the two up against each other.

WHY THIS EXISTS (BACKLOG.md P4.15d). `$LOOM_PROFILE` buckets nodes by `(op, ne0, ne1)`, and a 1-D
convolution's output is `[OL, 1, OC]`, so every convolution at a given activation length collapses into
ONE row -- 93 of VITS's convolutions land in a single `CONV_2D ne0=100 ne1=1` bucket whose weight shapes
are gone. That is fine for "where does the time go" and useless for "is loom running the same work as
the other engine", which is the question that gates every per-shape ratio in P4.16. The topology JSON in
the GGUF has the answer already: it lists every CONV_1D with its attrs, so the shapes can be recovered
without running anything.

WHAT IT DOES. Propagates shapes through a topology's node list symbolically (the same job
`GraphBuilder` does at run time, in 80 lines of Python and only far enough to size the convolutions) and
prints one row per convolution: activation length in, IC, K, OC, length out, stride/pad/dilation, and
the arithmetic. It knows the ops the exported models actually use and RAISES on anything else rather
than guessing a shape -- a census that silently invented a length would be worse than no census.

WHAT VALIDATES IT. Every count and length it produces for VITS matches a number that was measured
independently: 153 CONV_1D + 12 CONV_1D_DW + 3 CONV_TRANSPOSE_1D (`$LOOM_PROFILE`'s own totals), the
93/41/6/6/7/3 per-scale split of P4.16's table, and the 73472-sample output the driver synthesises for
the reference utterance. If a change here breaks one of those, the propagator is wrong, not the table.

AND `--shared-prefix`, WHICH IS WHAT ACTUALLY FOUND THE BUG. It reports every pair of topologies in a
file that begins with the same computation -- structural isomorphism with weights compared by identity,
not by shape. VITS's `stats` and `logw` share 469 nodes; Matcha's `encoder_mu` and `encoder_logw` share
642. A hit is not automatically a bug (Supertonic emits one topology per length bucket and the driver
runs exactly one of them); it is a bug when the driver runs both per inference.

USAGE

    python3 scripts/conv_census.py <model.gguf> --syms n_tokens=100 --syms flow_vocoder:n_tokens=287
    python3 scripts/conv_census.py <model.gguf> ... --onnx <model.onnx>   # needs `pip install onnx`
    python3 scripts/conv_census.py <model.gguf> --shared-prefix           # needs no symbols at all

`--syms name=value` binds a symbol for every topology; `--syms topo:name=value` binds it for one. VITS
needs the two-form spelling above because its `flow_vocoder` runs at the DURATION-EXPANDED length
(y_length = 287 for the reference utterance) while its text-side topologies run at the token count --
the driver script is what decides that, and no amount of staring at the topology JSON will reveal it.

The `--onnx` census is deliberately static: it reads the graph's Conv/ConvTranspose nodes and their
initializer shapes, no session and no profiling, because "which nodes exist" is a property of the graph.
It also counts MatMul nodes and reports whether each has a constant operand, which is how P4.15c's
premise ("onnxruntime carries 26 of loom's convolutions as MatMul") was checked -- and refuted.
"""
import argparse
import collections
import json
import math
import re
import sys

import gguf

# The engine's own expression grammar (src/core/symbol_env.cpp): + - * / floor() sqrt(), identifiers,
# numbers, and a '$' sigil that is optional even mid-expression. Anything outside this character set is
# rejected rather than handed to eval().
_SAFE = re.compile(r"^[0-9A-Za-z_+\-*/(). $]*$")


def ev(expr, syms):
    """One shape attribute as an integer, resolving symbols. Mirrors `SymbolEnv::eval_expr`."""
    if isinstance(expr, (int, float)):
        return int(expr)
    text = str(expr)
    if not _SAFE.match(text):
        raise ValueError(f"expression outside symbol_env.cpp's grammar: {text!r}")
    text = text.replace("$", "")
    text = re.sub(r"\bfloor\(", "math.floor(", text)
    text = re.sub(r"\bsqrt\(", "math.sqrt(", text)
    try:
        return int(eval(text, {"__builtins__": {}}, dict(syms, math=math)))
    except NameError as e:
        # An unbound shape symbol is the most common way to hold this tool wrong -- every model names
        # its own (n_tokens, n_samples, n_kv, n_enc_frames ...) and only its driver script knows which
        # value each phase runs at.
        raise SystemExit(f"{e.name!r} is not bound: add --syms {e.name}=<value> "
                         f"(or --syms <topology>:{e.name}=<value>) -- see the driver script for the "
                         f"value each phase runs at") from None


def pad4(ne):
    """ggml never leaves an axis at 0: a shape is always four dimensions with 1s in the unused ones."""
    dims = [int(d) for d in ne] + [1] * 4
    return dims[:4]


def load(path):
    """The topologies and the tensor shapes of one GGUF, alias table applied."""
    reader = gguf.GGUFReader(path)
    topologies = {}
    for field in reader.fields.values():
        if field.name.startswith("model.graph_topology."):
            name = field.name[len("model.graph_topology."):]
            topologies[name] = json.loads(str(field.contents()))
        elif field.name == "model.graph_topology":
            topologies[""] = json.loads(str(field.contents()))
    tensors = {t.name: [int(d) for d in t.shape] for t in reader.tensors}

    # `loom.tensor_alias.*` -- a topology may name a weight the file stores once under another name
    # (VITS's `logw` reaches into `stats` for all 137 of its text-encoder weights). GgufModel resolves
    # these at load; so does this, or half the convolutions would report an unknown operand.
    names, targets = [], []
    for field in reader.fields.values():
        if field.name == "loom.tensor_alias.names":
            names = [str(v) for v in field.contents()]
        elif field.name == "loom.tensor_alias.targets":
            targets = [str(v) for v in field.contents()]
    for alias, target in zip(names, targets):
        if target in tensors:
            tensors[alias] = tensors[target]
    return topologies, tensors


def conv_out_len(il, k, s, p, d):
    """ggml_calc_conv_output_size, which is what op_conv_1d's lowering ends up calling."""
    return (il + 2 * p - d * (k - 1) - 1) // s + 1


def propagate(topology, tensors, syms, on_conv):
    """Walks one topology, calling `on_conv(index, node, op, info)` for every convolution.

    Only shapes are computed -- this is not a second implementation of the engine, and it exists to
    answer a counting question. Unknown ops raise: see the module docstring.
    """
    shapes = {name: pad4(ne) for name, ne in tensors.items()}
    for declared in topology["inputs"]:
        shapes[declared["name"]] = pad4([ev(d, syms) for d in declared["shape"]])

    for index, node in enumerate(topology["nodes"]):
        op = node["op"]
        attrs = node.get("attrs", {})
        ins = []
        for name in node["inputs"]:
            if name not in shapes:
                raise KeyError(f"{op} (node {index}) reads unknown tensor {name!r}")
            ins.append(shapes[name])
        out = None

        if op in ("RESHAPE", "VIEW", "REPEAT"):
            dims = [ev(d, syms) for d in attrs["shape"]]
            if -1 in dims:   # the numpy-style inferred axis the exporter emits for a flatten
                known = 1
                for d in dims:
                    if d != -1:
                        known *= d
                total = 1
                for d in ins[0]:
                    total *= d
                dims[dims.index(-1)] = total // known
            out = dims
        elif op == "PERMUTE":
            # ggml_permute(a, ax0..ax3): source axis i becomes destination axis attrs["axes"][i].
            out = [0] * 4
            for src_axis, dst_axis in enumerate(attrs["axes"]):
                out[dst_axis] = ins[0][src_axis]
        elif op in ("CONT", "LAYER_NORM", "RMS_NORM", "GROUP_NORM", "SOFTMAX", "RELU", "GELU", "SILU",
                    "TANH", "SIGMOID", "LEAKY_RELU", "SOFTPLUS", "EXP", "LOG", "NEG", "ABS", "SQRT",
                    "SQR", "SIN", "COS", "ATAN", "FLOOR", "CLAMP", "SCALE", "CAST", "NOT", "CUMSUM",
                    "RQ_SPLINE_INVERSE"):
            out = list(ins[0])
        elif op in ("ADD", "MUL", "SUB", "DIV", "FLOOR_DIV", "EQUAL", "NOT_EQUAL", "LESS", "GREATER",
                    "GREATER_EQUAL", "LESS_EQUAL", "SELECT", "MAX", "MIN", "MAXIMUM", "MINIMUM"):
            out = [max(axis) for axis in zip(*ins)]   # ggml broadcasts the smaller operand
        elif op == "CONCAT":
            out = list(ins[0])
            out[int(attrs["dim"])] = sum(i[int(attrs["dim"])] for i in ins)
        elif op in ("PAD_1D", "PAD_1D_REFLECT"):
            out = list(ins[0])
            out[0] += ev(attrs["lp0"], syms) + ev(attrs["rp0"], syms)
        elif op == "RANGE_1D":
            span = ev(attrs["end"], syms) - ev(attrs["start"], syms)
            out = [-(-span // max(1, ev(attrs.get("step", 1), syms))), 1, 1, 1]
        elif op == "INTERPOLATE_1D":
            out = list(ins[0])
            out[0] = ev(attrs["ne0"], syms)
        elif op == "POOL_1D":
            out = list(ins[0])
            out[0] = conv_out_len(out[0], ev(attrs["k0"], syms), ev(attrs["s0"], syms),
                                  ev(attrs["p0"], syms), 1)
        elif op == "MEAN":
            out = [1] + list(ins[0][1:])
        elif op == "REDUCE_SUM":
            out = list(ins[0])
            axis = int(attrs.get("axis", 0))
            if attrs.get("keep_dims"):
                out[axis] = 1
            else:
                out = out[:axis] + out[axis + 1:] + [1]
        elif op == "GET_ROWS":
            out = [ins[0][0], ins[1][0], ins[1][1], 1]
        elif op == "MUL_MAT":
            out = [ins[0][1], ins[1][1], ins[1][2], ins[1][3]]
        elif op in ("CONV_1D", "CONV_1D_DW"):
            k, kernel_ic, kernel_oc = ins[0][0], ins[0][1], ins[0][2]
            il, ic, n = ins[1][0], ins[1][1], ins[1][2]
            s0 = ev(attrs["s0"], syms)
            p0 = ev(attrs["p0"], syms)
            d0 = ev(attrs["d0"], syms)
            ol = conv_out_len(il, k, s0, p0, d0)
            # Depthwise carries one filter per channel, so its kernel's ne[1] is 1 and OC is the
            # activation's own channel count.
            oc = ic if op == "CONV_1D_DW" else kernel_oc
            out = [ol, oc, n, 1]
            on_conv(index, node, op, dict(k=k, ic=ic, oc=oc, il=il, ol=ol, n=n, s0=s0, p0=p0, d0=d0,
                                          kernel_ic=kernel_ic, weight=node["inputs"][0]))
        elif op in ("CONV_2D", "CONV_2D_DW"):
            kw, kh, kernel_ic, kernel_oc = ins[0]
            w, h, c, n = ins[1]
            s0, s1 = ev(attrs["s0"], syms), ev(attrs["s1"], syms)
            p0, p1 = ev(attrs["p0"], syms), ev(attrs["p1"], syms)
            d0, d1 = ev(attrs["d0"], syms), ev(attrs["d1"], syms)
            ow, oh = conv_out_len(w, kw, s0, p0, d0), conv_out_len(h, kh, s1, p1, d1)
            oc = c if op == "CONV_2D_DW" else kernel_oc
            out = [ow, oh, oc, n]   # op_conv_2d permutes back to [OW, OH, OC, N]
            on_conv(index, node, op, dict(k=kw * kh, ic=c, oc=oc, il=w * h, ol=ow * oh, n=n, s0=s0,
                                          p0=p0, d0=d0, kernel_ic=kernel_ic, weight=node["inputs"][0]))
        elif op == "CONV_TRANSPOSE_1D":
            k, oc, ic = ins[0][0], ins[0][1], ins[0][2]   # ggml stores a transpose kernel [K, OC, IC]
            il, act_ic, _ = ins[1][0], ins[1][1], ins[1][2]
            s0 = ev(attrs["s0"], syms)
            ol = (il - 1) * s0 + k                        # ggml forces p0=0, d0=1 for this op
            out = [ol, oc, 1, 1]
            on_conv(index, node, op, dict(k=k, ic=act_ic, oc=oc, il=il, ol=ol, n=1, s0=s0, p0=0, d0=1,
                                          kernel_ic=ic, weight=node["inputs"][0]))
        else:
            raise NotImplementedError(
                f"{op} (node {index}): no shape rule here. Add one -- guessing would silently "
                f"mis-size every convolution downstream of it.")

        for name in node["outputs"]:
            shapes[name] = pad4(out)
    return shapes


def macs(row):
    """Multiply-accumulates, x2 for FLOPs. Depthwise contracts over K alone, not over IC."""
    if row["op"] == "CONV_1D_DW":
        return row["ol"] * row["oc"] * row["k"]
    return row["ol"] * row["oc"] * row["ic"] * row["k"]


# A convolution's identity for comparison purposes: what it computes, independent of where it sits.
# Length is deliberately NOT in the key -- two engines that agree on the graph can still disagree on a
# padded length by a couple of frames (onnxruntime's pinned y_length is 282 against loom's 287).
def shape_key(row):
    return (row["op"], row["k"], row["ic"], row["oc"], row["d0"], row["s0"])


def loom_census(path, syms_by_topo, default_syms):
    topologies, tensors = load(path)
    rows = []
    for name, topology in topologies.items():
        syms = dict(default_syms)
        syms.update(syms_by_topo.get(name, {}))
        def record(index, node, op, info, name=name):
            rows.append(dict(topo=name, index=index, node=node["outputs"][0], op=op, **info))
        propagate(topology, tensors, syms, record)
    return rows


def shared_prefix(topo_a, topo_b, alias, tensor_names):
    """How many leading nodes of two topologies are THE SAME COMPUTATION, not merely the same shape.

    Structural isomorphism, because SSA value names differ between topologies (`_11` here is `_55`
    there): a bijection between value names is built as the walk proceeds. A WEIGHT, though, has to be
    literally the same tensor after `loom.tensor_alias` resolution -- without that rule Kokoro's f0 and
    n predictor blocks, which are structurally identical with different weights, read as duplicated
    work when they are nothing of the kind. This is what found P4.15d's duplicated text encoder.
    """
    forward, backward = {}, {}

    def same(x, y):
        x, y = alias.get(x, x), alias.get(y, y)
        if x in tensor_names or y in tensor_names:
            return x == y
        if forward.get(x, y) != y or backward.get(y, x) != x:
            return False
        forward[x], backward[y] = y, x
        return True

    n = 0
    for a, b in zip(topo_a["nodes"], topo_b["nodes"]):
        if a["op"] != b["op"] or a.get("attrs", {}) != b.get("attrs", {}):
            break
        if len(a["inputs"]) != len(b["inputs"]) or len(a["outputs"]) != len(b["outputs"]):
            break
        if not all(same(x, y) for x, y in zip(a["inputs"], b["inputs"])):
            break
        if not all(same(x, y) for x, y in zip(a["outputs"], b["outputs"])):
            break
        n += 1
    return n


def report_shared_prefixes(path):
    """Every pair of topologies in one GGUF that starts with the same computation. See P4.15d/P4.15f.

    A hit is not automatically a bug: a model whose exporter emits one topology per LENGTH BUCKET
    (Supertonic's `vfe_128`/`vfe_256`/...) shares a prefix between variants of which the driver runs
    exactly one. A hit is a bug when the DRIVER runs both per inference -- read the driver script.
    """
    import itertools

    reader = gguf.GGUFReader(path)
    topologies, _ = load(path)
    names, targets = [], []
    for field in reader.fields.values():
        if field.name == "loom.tensor_alias.names":
            names = [str(v) for v in field.contents()]
        elif field.name == "loom.tensor_alias.targets":
            targets = [str(v) for v in field.contents()]
    alias = dict(zip(names, targets))
    tensor_names = {t.name for t in reader.tensors}

    print(f"\n{len(topologies)} topologies; pairs sharing a leading computation:")
    found = False
    for a, b in itertools.combinations(sorted(topologies), 2):
        n = shared_prefix(topologies[a], topologies[b], alias, tensor_names)
        if n < 8:
            continue
        found = True
        heavy = collections.Counter(node["op"] for node in topologies[a]["nodes"][:n])
        heavy = {op: c for op, c in heavy.items() if "CONV" in op or op == "MUL_MAT"}
        print(f"  {a} | {b}: {n} nodes of {len(topologies[a]['nodes'])}/"
              f"{len(topologies[b]['nodes'])}  {heavy}")
    if not found:
        print("  none")


def onnx_census(path):
    """Conv/ConvTranspose nodes of an ONNX graph, plus a MatMul tally. Static: no session, no run."""
    import onnx   # optional dependency, imported only for --onnx

    model = onnx.load(path, load_external_data=False)
    initializers = {i.name: list(i.dims) for i in model.graph.initializer}
    rows, matmuls = [], []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            matmuls.append((node.name, [i in initializers for i in node.input]))
            continue
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        attrs = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
        weight = initializers[node.input[1]]
        group = attrs.get("group", 1)
        if node.op_type == "ConvTranspose":
            op, oc, ic, k = "CONV_TRANSPOSE_1D", weight[1], weight[0], weight[2]
        else:
            op = "CONV_1D_DW" if group != 1 else "CONV_1D"
            oc, ic, k = weight[0], weight[1] * group, weight[2]
        rows.append(dict(op=op, k=k, ic=ic, oc=oc, d0=(attrs.get("dilations") or [1])[0],
                         s0=(attrs.get("strides") or [1])[0], name=node.name,
                         module=node.name.strip("/").split("/")[0]))
    return rows, matmuls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--syms", action="append", default=[],
                    metavar="[TOPO:]NAME=VALUE",
                    help="bind a shape symbol, globally or for one topology (repeatable)")
    ap.add_argument("--onnx", metavar="MODEL.ONNX",
                    help="also census an ONNX graph and diff the two, shape by shape")
    ap.add_argument("--quiet", action="store_true", help="rollups only, no per-node table")
    ap.add_argument("--shared-prefix", action="store_true",
                    help="report topology pairs that begin with the same computation (needs no --syms)")
    args = ap.parse_args(argv)

    if args.shared_prefix:
        report_shared_prefixes(args.gguf)
        if not args.syms:
            return 0

    default_syms, syms_by_topo = {"n_past": 0}, collections.defaultdict(dict)
    for spec in args.syms:
        scope, _, binding = spec.rpartition(":")
        name, _, value = binding.partition("=")
        target = syms_by_topo[scope] if scope else default_syms
        target[name] = int(value)

    rows = loom_census(args.gguf, syms_by_topo, default_syms)

    if not args.quiet:
        print(f"{'topology':14s} {'op':18s} {'IL':>7s} {'IC':>5s} {'K':>3s} {'OC':>5s} {'OL':>7s} "
              f"{'s':>2s} {'p':>3s} {'d':>2s} {'MFLOP':>9s}  node")
        for r in rows:
            print(f"{r['topo']:14s} {r['op']:18s} {r['il']:7d} {r['ic']:5d} {r['k']:3d} {r['oc']:5d} "
                  f"{r['ol']:7d} {r['s0']:2d} {r['p0']:3d} {r['d0']:2d} "
                  f"{2 * macs(r) / 1e6:9.1f}  {r['node']}")
        print()

    by_op = collections.Counter(r["op"] for r in rows)
    total_flops = sum(2 * macs(r) for r in rows)
    print(f"loom: {len(rows)} convolutions  " +
          "  ".join(f"{op} {n}" for op, n in sorted(by_op.items())) +
          f"   {total_flops / 1e9:.2f} GFLOP")

    print("\nby (topology, activation length):")
    per_scale = collections.Counter()
    scale_flops = collections.Counter()
    for r in rows:
        per_scale[(r["topo"], r["il"])] += 1
        scale_flops[(r["topo"], r["il"])] += 2 * macs(r)
    for key in sorted(per_scale, key=lambda k: -scale_flops[k]):
        print(f"  {key[0]:14s} L={key[1]:<7d} {per_scale[key]:4d} convs  "
              f"{scale_flops[key] / 1e6:9.1f} MFLOP")

    if not args.onnx:
        return 0

    onnx_rows, matmuls = onnx_census(args.onnx)
    loom_by_shape = collections.Counter(shape_key(r) for r in rows)
    onnx_by_shape = collections.Counter(shape_key(r) for r in onnx_rows)

    print(f"\nonnx: {len(onnx_rows)} convolutions, {len(matmuls)} MatMul "
          f"({sum(1 for _, w in matmuls if any(w))} with a constant operand -- "
          f"a MatMul with none is an activation-by-activation product, NOT a convolution)")

    print(f"\n{'op':20s} {'K':>3s} {'IC':>5s} {'OC':>5s} {'d':>3s} {'s':>2s} {'loom':>6s} {'onnx':>6s} "
          f"{'diff':>6s}")
    loom_total = onnx_total = 0
    for key in sorted(set(loom_by_shape) | set(onnx_by_shape)):
        l, o = loom_by_shape[key], onnx_by_shape[key]
        loom_total += l
        onnx_total += o
        print(f"{key[0]:20s} {key[1]:3d} {key[2]:5d} {key[3]:5d} {key[4]:3d} {key[5]:2d} "
              f"{l:6d} {o:6d} {l - o:6d}" + ("   <-- differs" if l != o else ""))
    print(f"{'TOTAL':20s} {'':3s} {'':5s} {'':5s} {'':3s} {'':2s} {loom_total:6d} {onnx_total:6d} "
          f"{loom_total - onnx_total:6d}")

    print("\nonnx convolutions by module:")
    by_module = collections.Counter(r["module"] for r in onnx_rows)
    for module, n in by_module.most_common():
        print(f"  {module:14s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
