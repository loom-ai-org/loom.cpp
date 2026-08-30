# onnxruntime's convolution time broken down by SHAPE, to sit beside loom's `$LOOM_PROFILE` output.
#
# NOT part of the build. It is the other half of P4.16's table (Epic-05 §5): `scripts/conv_census.py`
# says which convolutions each engine ISSUES, this says what onnxruntime's cost, and `$LOOM_PROFILE`
# says what loom's cost. It lived only on the Pi until 2026-08-30, which is the failure
# [Retro-018](../docs/retros/retro-018-a-table-of-ratios-nobody-could-re-derive.md) is about -- a
# published ratio whose other half nobody else can re-derive.
#
#   python3 scripts/prof_onnx_conv_shapes.py <miro_en-GB.onnx> [threads] [nrun]
#
# THREE THINGS IT DOES ON PURPOSE, each of which is load-bearing:
#
#  * SHARES ONLY, apportioned over a separately measured UN-PROFILED wall. Profiling costs onnxruntime
#    ~1.18x, so its absolute event durations are not comparable to anything; the fraction of the run a
#    shape accounts for is. loom's side of the table has to be treated identically or the two columns
#    are not the same quantity.
#  * IT PINS THE DURATION PREDICTOR (`scales = [0, 1, 0]`). VITS's is stochastic, both engines are
#    near-linear in output samples, and unpinned this profiles a different utterance every run
#    (Retro-010). The sample count is printed so it can be checked against loom's.
#  * IT DROPS THE FIRST RUN and splits the rest BY ORDER, because this build's profile events carry no
#    `run_index` -- the first is warm-up and would otherwise be averaged in.
#
# AND ONE THING TO KNOW BEFORE READING THE OUTPUT: a node-by-node profile cannot see FUSION. Compare
# `FusedConv` counts against `scripts/conv_census.py`'s plain-Conv counts before concluding that two
# engines issue different work -- and on loom's side the equivalent blindness costs 135 ms of a 1137 ms
# VITS synthesis, measured with `GGML_CPU_DISABLE_FUSION=1`.
import collections
import json
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

# The same sentence scripts/bench_vits_loom.cpp speaks, as piper phoneme ids from miro_en-GB.onnx.json.
IDS = [1,20,0,120,0,18,0,74,0,8,0,3,0,23,0,39,0,26,0,3,0,22,0,33,0,122,0,3,0,96,0,120,0,102,0,32,
       0,17,0,14,0,100,0,26,0,3,0,41,0,59,0,3,0,23,0,59,0,25,0,28,0,22,0,120,0,33,0,122,0,32,0,
       50,0,8,0,3,0,25,0,14,0,74,0,3,0,19,0,88,0,120,0,61,0,26,0,17,0,13,0,2]


def session(model, threads, profile):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.enable_profiling = profile
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])


def feed():
    x = np.array([IDS], dtype=np.int64)
    return {"input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)}


def shape_of(event):
    dims = []
    for tensor in event["args"].get("input_type_shape", []):
        dims.extend(tensor.values())
    return dims


def main(argv):
    model = argv[1]
    threads = int(argv[2]) if len(argv) > 2 else 4
    nrun = int(argv[3]) if len(argv) > 3 else 5

    plain, f = session(model, threads, False), feed()
    n_samples = np.asarray(plain.run(None, f)[0]).size          # warm, and the equal-work invariant
    wall = statistics.median(
        [(lambda t0: (plain.run(None, f), time.perf_counter() - t0)[1])(time.perf_counter())
         for _ in range(9)])
    print(f"onnx   vits  samples={n_samples}  un-profiled wall {wall * 1e3:.1f} ms  "
          f"({threads} thread{'s' * (threads != 1)})\n")

    profiled = session(model, threads, True)
    for _ in range(nrun):
        profiled.run(None, f)
    events = json.load(open(profiled.end_profiling()))
    kernels = [e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
    per_run = len(kernels) // nrun
    measured = kernels[-per_run * (nrun - 1):]                  # drop run 1, keep the rest
    total = sum(e["dur"] for e in measured)

    agg = collections.defaultdict(lambda: [0, 0])
    for e in measured:
        if e["args"]["op_name"] not in ("Conv", "FusedConv", "ConvTranspose"):
            continue
        dims = shape_of(e)
        key = (e["args"]["op_name"], tuple(dims[0] if dims else []), tuple(dims[1] if len(dims) > 1 else []))
        agg[key][0] += e["dur"]
        agg[key][1] += 1

    conv = sum(v[0] for v in agg.values())
    print(f"{'op':<14}{'activation':>18}{'weight':>18}{'calls':>7}{'ms@wall':>9}{'%conv':>7}")
    for (op, act, weight), (dur, calls) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{op:<14}{str(list(act)):>18}{str(list(weight)):>18}{calls // (nrun - 1):>7}"
              f"{dur / total * wall * 1e3:>9.1f}{dur / conv * 100:>6.1f}%")
    print(f"\nconvolution total: {conv / total * wall * 1e3:.1f} ms of {wall * 1e3:.1f} ms wall "
          f"({conv / total * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        print("usage: prof_onnx_conv_shapes.py <miro_en-GB.onnx> [threads] [nrun]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv))
