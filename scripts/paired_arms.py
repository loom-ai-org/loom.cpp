#!/usr/bin/env python3
"""Two arms of one benchmark, compared as PAIRED rounds rather than as two medians.

An arm is either a BUILD of one shared library (`--lib` plus two `--arm NAME=PATH`) or a VALUE of one
environment variable (`--env VAR` plus two `--arm NAME=VALUE`).  The second form is what a run-time
kill switch inside the thing being measured is for: one binary, one allocation, one page cache, and a
branch flipped between the two halves of a pair.  P4.26 measured `ggml-0012` that way after P4.22 had
measured it by building two whole trees, which cannot make that claim.

WHY THIS EXISTS.  P4.18's `ggml-0011` is a patch to `sgemm.cpp`, so the honest baseline is "the same
tree with that patch reverse-applied", not an older commit -- and the two arms are then two builds of
`libggml-cpu.so` that differ in one function.  Comparing them by running each thirty times and taking
two medians does not work on a thermally noisy box: Epic-05 measured a 2.2x swing in one binary at one
shape twenty minutes apart, and the effects being tested here are 1.2x.

A PAIRED test fixes what two independent medians cannot.  Both arms run back to back inside one round,
the RATIO is recorded per round, and the report is the median ratio with its p10/p90.  Drift moves both
halves of a pair together and cancels; two independent minima do not.  The arm order flips every round
(ABBA), so a systematic first-slot advantage cancels too.

WHAT IT SWAPS.  One file, in place -- `--lib` is the real library file (not a symlink), `--arm` names a
replacement to copy over it before each run.  Every arm's file is restored to the one that was there at
the start when the run ends, including on Ctrl-C: leaving a developer's build tree holding a benchmark
arm is a trap that outlives the benchmark.

    ./scripts/paired_arms.py \
        --lib build/_deps/ggml-build/src/libggml-cpu.so.0.19.0 \
        --arm base=/tmp/arms/libggml-cpu.base.so \
        --arm patched=/tmp/arms/libggml-cpu.patched.so \
        --rounds 15 --label conformer -- \
        ./bench_asr_loom model.gguf clip.wav 3

The command is run once per arm per round; the number it reports is the last float on a line matching
`--metric` (default: the `median` field `bench_asr_loom` prints).  A command that fails, or whose output
carries no such number, aborts rather than being silently dropped -- a paired test with holes in it is
back to being two independent medians.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


def parse_metric(text, pattern):
    """The number `pattern` selects, from the last line that matches it.

    A capture group in the pattern IS the number; without one, the last float on the line is taken.
    The group is not a nicety -- `bench_asr_loom` prints `median 0.68 s min 0.65 s (n=3)`, and the
    first draft of this read that as 3.
    """
    value = None
    for line in text.splitlines():
        m = re.search(pattern, line)
        if not m:
            continue
        if m.groups():
            value = float(m.group(1))
        else:
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if numbers:
                value = float(numbers[-1])
    if value is None:
        raise RuntimeError("no line matching %r in:\n%s" % (pattern, text))
    return value


def percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", help="the library file each arm is copied over (library-arm mode)")
    ap.add_argument("--env", metavar="VAR",
                    help="environment variable each arm sets instead (env-arm mode)")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH_OR_VALUE",
                    help="an arm; give it exactly twice, first one is the baseline")
    ap.add_argument("--between", metavar="CMD",
                    help="shell command run before every arm -- on the reference Pi this is the "
                         "cool-to-a-fixed-temperature wait, which has to happen inside the pairing "
                         "rather than around it")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--metric", default=r"median\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    help="regex selecting the number; a capture group IS the number, else the "
                         "last float on the matching line")
    ap.add_argument("--label", default="")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="after --, the command to run once per arm per round")
    args = ap.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        ap.error("no command given (put it after --)")
    if bool(args.lib) == bool(args.env):
        ap.error("give exactly one of --lib (library arms) and --env (environment arms)")
    arms = []
    for spec in args.arm:
        name, sep, value = spec.partition("=")
        if not sep:
            ap.error("--arm wants NAME=PATH or NAME=VALUE, got %r" % spec)
        arms.append((name, value))
    if len(arms) != 2:
        ap.error("exactly two arms, so that a round is a pair")

    # The library as it stands now, restored whatever happens -- see the docstring.
    saved = None
    if args.lib:
        saved = tempfile.NamedTemporaryFile(delete=False, suffix=".so").name
        shutil.copy2(args.lib, saved)
    samples = {name: [] for name, _ in arms}
    ratios = []
    try:
        for r in range(args.rounds):
            # ABBA: the baseline runs first on even rounds and second on odd ones.
            order = arms if r % 2 == 0 else arms[::-1]
            for name, value in order:
                env = None
                if args.lib:
                    shutil.copy2(value, args.lib)
                else:
                    env = dict(os.environ, **{args.env: value})
                if args.between:
                    subprocess.run(args.between, shell=True, check=True)
                proc = subprocess.run(command, capture_output=True, text=True, env=env)
                if proc.returncode != 0:
                    raise RuntimeError("arm %s failed (%d):\n%s\n%s"
                                       % (name, proc.returncode, proc.stdout, proc.stderr))
                samples[name].append(parse_metric(proc.stdout + proc.stderr, args.metric))
            a, b = arms[0][0], arms[1][0]
            ratios.append(samples[a][-1] / samples[b][-1])
            print("  round %2d/%d  %s %.4f  %s %.4f  ratio %.3f"
                  % (r + 1, args.rounds, a, samples[a][-1], b, samples[b][-1], ratios[-1]),
                  file=sys.stderr, flush=True)
    finally:
        if saved:
            shutil.copy2(saved, args.lib)
            os.unlink(saved)

    ratios.sort()
    a, b = arms[0][0], arms[1][0]
    print("\n%s%s / %s over %d paired rounds" % (args.label + ": " if args.label else "", a, b,
                                                 len(ratios)))
    for name, _ in arms:
        v = sorted(samples[name])
        print("  %-10s median %.4f   min %.4f   max %.4f" % (name, statistics.median(v), v[0], v[-1]))
    print("  ratio      p10 %.3f   MEDIAN %.3f   p90 %.3f" % (percentile(ratios, 0.10),
                                                              statistics.median(ratios),
                                                              percentile(ratios, 0.90)))
    # Epic-05's rule for this box, restated where it gets read: a p10 that crosses 1.0 means the
    # measurement did not resolve the effect, whatever the median says.
    if percentile(ratios, 0.10) < 1.0 < percentile(ratios, 0.90):
        print("  ^ p10/p90 straddle 1.0 -- WEAK, report as unresolved rather than as a number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
