#!/usr/bin/env python3
r"""WakeHuBERT tiny: loom's four GGUF precisions against upstream's ONNX files, timed per LAUNCH.

Both engines are driven the way a user drives them from Python: loom through loom-py's
`model.infer(waveform=...)` (the list marshalling into the driver is part of what a caller pays),
onnxruntime through `InferenceSession.run` on a numpy array. One launch loads one model, warms up, and
prints the median of `nrun` timed calls; `--sweep` runs many launches with a settle between them and the
arm order shuffled per round, and reports the mean of the per-launch medians (P4.30b's protocol:
placement is chosen once per process, so only a per-launch sample averages it out).

One launch:

    python3 scripts/bench_wakehubert.py run loom <file.gguf> <seconds> <threads> <nrun>
    python3 scripts/bench_wakehubert.py run onnx <file.onnx> <seconds> <threads> <nrun>

The sweep (each arm's interpreter is named, since loom-py and onnxruntime may live in different venvs):

    python3 scripts/bench_wakehubert.py sweep --rounds 7 --threads 1 2 --seconds 2.5 30 \
        --arm "loom-f32=<py> <gguf>" ... --arm "onnx-int8=<py> <onnx>"

`--threads` is LOOM_N_THREADS on the loom side and `intra_op_num_threads` on the onnx side (inter-op 1).
The input is samples/jfk.wav tiled to the requested length, so both engines see the same speech.
"""
import argparse
import os
import random
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

JFK = Path(__file__).resolve().parents[1] / "samples" / "jfk.wav"
LINE = re.compile(r"^BENCH (\S+) seconds=(\S+) threads=(\d+) median_ms=(\S+) frames=(\d+) rss_mb=(\S+)$")


def clip(seconds):
    with wave.open(str(JFK)) as w:
        raw = w.readframes(w.getnframes())
    pcm = [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
    n = int(seconds * 16000)
    return (pcm * (n // len(pcm) + 1))[:n]


def run(engine, path, seconds, threads, nrun):
    samples = clip(seconds)
    if engine == "loom":
        os.environ["LOOM_N_THREADS"] = str(threads)
        import loom
        model = loom.Model.from_file(path)
        call = lambda: model.infer(waveform=samples)
        frames = lambda out: len(out) // 128
    else:
        import numpy as np
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        x = np.asarray(samples, dtype=np.float32)[None]
        call = lambda: sess.run(None, {"waveform": x})[0]
        frames = lambda out: out.shape[1]
    for _ in range(3):
        out = call()
    times = []
    for _ in range(nrun):
        t0 = time.perf_counter()
        out = call()
        times.append(time.perf_counter() - t0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"BENCH {engine} seconds={seconds} threads={threads} median_ms={1e3 * statistics.median(times):.3f} "
          f"frames={frames(out)} rss_mb={rss:.1f}", flush=True)


def sweep(args):
    arms = [a.split("=", 1) for a in args.arm]
    results = {}
    for seconds in args.seconds:
        nrun = max(3, int(args.budget / max(seconds, 0.5)))
        for threads in args.threads:
            for rnd in range(args.rounds):
                order = list(arms)
                random.Random(rnd * 1000 + threads).shuffle(order)
                for name, spec in order:
                    py, path = shlex.split(spec)
                    engine = "onnx" if path.endswith(".onnx") else "loom"
                    time.sleep(args.settle)
                    cmd = [py, __file__, "run", engine, path, str(seconds), str(threads), str(nrun)]
                    out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
                    match = next((LINE.match(l) for l in out.stdout.splitlines() if LINE.match(l)), None)
                    if out.returncode or match is None:
                        sys.exit(f"{name} failed:\n{out.stdout}\n{out.stderr}")
                    results.setdefault((seconds, threads, name), []).append(
                        (float(match.group(4)), int(match.group(5)), float(match.group(6))))
                print(f"  {seconds} s, {threads} thr, round {rnd + 1}/{args.rounds} done", file=sys.stderr)
    for seconds in args.seconds:
        for threads in args.threads:
            print(f"\n{seconds} s of audio, {threads} thread(s), {args.rounds} launches per arm "
                  f"(mean of per-launch medians; p10-p90 across launches):")
            base = None
            for name, _ in arms:
                rows = results[(seconds, threads, name)]
                ms = [r[0] for r in rows]
                frames = {r[1] for r in rows}
                mean = statistics.mean(ms)
                if name == args.baseline:
                    base = mean
                q = statistics.quantiles(ms, n=10) if len(ms) > 2 else [min(ms)] * 9
                rtf = mean / 1000.0 / seconds
                print(f"  {name:12s} {mean:9.2f} ms  [{q[0]:8.2f} - {q[-1]:8.2f}]  x{1 / rtf:7.0f} real time  "
                      f"frames {sorted(frames)}  peak RSS {max(r[2] for r in rows):6.1f} MB")
            if base:
                print(f"  (vs {args.baseline}: " + ", ".join(
                    f"{n} {base / statistics.mean([r[0] for r in results[(seconds, threads, n)]]):.2f}x"
                    for n, _ in arms if n != args.baseline) + "; >1 = faster than the baseline)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("engine", choices=["loom", "onnx"])
    r.add_argument("path")
    r.add_argument("seconds", type=float)
    r.add_argument("threads", type=int)
    r.add_argument("nrun", type=int)
    s = sub.add_parser("sweep")
    s.add_argument("--arm", action="append", required=True, help="NAME=<python> <model file>")
    s.add_argument("--rounds", type=int, default=7)
    s.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    s.add_argument("--seconds", type=float, nargs="+", default=[2.5, 30.0])
    s.add_argument("--budget", type=float, default=30.0, help="seconds of audio per launch, which sets nrun")
    s.add_argument("--settle", type=float, default=1.0)
    s.add_argument("--baseline", default="onnx-int8")
    a = p.parse_args()
    if a.mode == "run":
        run(a.engine, a.path, a.seconds, a.threads, a.nrun)
    else:
        sweep(a)


if __name__ == "__main__":
    main()
