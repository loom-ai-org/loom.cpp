#!/usr/bin/env python3
r"""Two arms of one benchmark, compared as PAIRED rounds rather than as two medians.

An arm is a BUILD of one shared library (`--lib` plus two `--arm NAME=PATH`), a VALUE of one
environment variable (`--env VAR` plus two `--arm NAME=VALUE`), or a COMMAND of its own (`--cmd` plus
two `--arm NAME=SHELL_COMMAND`).  The second form is what a run-time kill switch inside the thing being
measured is for: one binary, one allocation, one page cache, and a branch flipped between the two
halves of a pair.  P4.26 measured `ggml-0012` that way after P4.22 had measured it by building two
whole trees, which cannot make that claim.

`--cmd` is the form for TWO ENGINES rather than two builds of one -- loom against onnxruntime, where
the arms share no binary and no metric line.  Each round is one launch of each, which is the whole
point on a hybrid part: thread placement is chosen once per PROCESS and then sticks, so a within-
process median cannot average it out and only a per-launch sample can (P4.30b, Retro-018).  Give
`--metric` twice there, once per arm in the order the arms are declared, because the two engines do
not print the same line -- and name the ratio explicitly with `--ratio NUM/DEN` so a seconds metric
(smaller is better) and a tok/s one (larger is better) both come out as ">1 means the first engine
won".

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

    ./scripts/paired_arms.py --cmd --rounds 9 --label "285K TTS @24" \
        --arm loom="LOOM_N_THREADS=24 ./bench_vits_loom vits.gguf 9" \
        --arm onnx="python3 scripts/bench_onnx_tasks.py vits miro_en-GB.onnx 24 9" \
        --metric 'loom   vits.*median\s+([0-9.]+)' \
        --metric 'onnx   vits.*median\s+([0-9.]+)' \
        --ratio onnx/loom

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
    ap.add_argument("--cmd", action="store_true",
                    help="each arm's value IS its shell command (command-arm mode) -- for two "
                         "engines rather than two builds of one")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH_OR_VALUE",
                    help="an arm; give it exactly twice, first one is the baseline")
    ap.add_argument("--between", metavar="CMD",
                    help="shell command run before every arm -- on the reference Pi this is the "
                         "cool-to-a-fixed-temperature wait, which has to happen inside the pairing "
                         "rather than around it, and on a hybrid x86 part `sleep 1` is the settle "
                         "that stops the second arm inheriting the first one's placement (P4.30b: "
                         "worth 1.41x -> 1.20x on one README cell)")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--metric", action="append", metavar="REGEX",
                    help="regex selecting the number; a capture group IS the number, else the last "
                         "float on the matching line. Give it once for both arms, or twice -- one per "
                         "arm, in the order the arms are declared -- when they do not print alike. "
                         "Default: %(default)s")
    ap.add_argument("--ratio", metavar="NUM/DEN",
                    help="which arm divides which, by name. Default is the first arm over the "
                         "second, which is right for two builds of one binary reporting the same "
                         "metric and wrong the moment an arm reports tok/s instead of seconds")
    ap.add_argument("--label", default="")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="after --, the command to run once per arm per round")
    args = ap.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if sum(bool(m) for m in (args.lib, args.env, args.cmd)) != 1:
        ap.error("give exactly one of --lib (library arms), --env (environment arms) and --cmd "
                 "(command arms)")
    if args.cmd:
        if command:
            ap.error("--cmd takes each arm's command from its --arm, so there is nothing after --")
    elif not command:
        ap.error("no command given (put it after --)")

    arms = []
    for spec in args.arm:
        name, sep, value = spec.partition("=")
        if not sep:
            ap.error("--arm wants NAME=PATH, NAME=VALUE or NAME=COMMAND, got %r" % spec)
        arms.append((name, value))
    if len(arms) != 2:
        ap.error("exactly two arms, so that a round is a pair")

    # One metric for both arms, or one each. Two engines do not print the same line, and quietly
    # reusing arm 1's regex on arm 2 would either fail loudly (fine) or match the wrong line (not):
    # bench_lm_loom prints tok/s TWICE, for `infer_with_past` and for the re-fed `infer` arm that is
    # 2.8x slower, and parse_metric takes the LAST match.
    default_metric = r"median\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    metrics = args.metric or [default_metric]
    if len(metrics) == 1:
        metrics = metrics * 2
    if len(metrics) != 2:
        ap.error("--metric takes one regex for both arms or two, one per arm; got %d" % len(metrics))
    metric_of = {name: metrics[i] for i, (name, _) in enumerate(arms)}

    # Which arm divides which. Named rather than positional because the direction is not recoverable
    # from the numbers: 0.6 is a win for a seconds metric and a loss for a tok/s one.
    names = [name for name, _ in arms]
    if args.ratio:
        num, sep, den = args.ratio.partition("/")
        if not sep or num not in names or den not in names or num == den:
            ap.error("--ratio wants NUM/DEN naming both arms, e.g. %s/%s" % (names[0], names[1]))
    else:
        num, den = names[0], names[1]

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
                env, run, shell = None, command, False
                if args.lib:
                    shutil.copy2(value, args.lib)
                elif args.env:
                    env = dict(os.environ, **{args.env: value})
                else:
                    # A command arm is run through a shell so that an arm can carry its own
                    # environment inline ("LOOM_N_THREADS=24 ./bench_vits_loom ..."), which is how the
                    # thread count gets into one side of a two-engine pair.
                    run, shell = value, True
                if args.between:
                    subprocess.run(args.between, shell=True, check=True)
                proc = subprocess.run(run, capture_output=True, text=True, env=env, shell=shell)
                if proc.returncode != 0:
                    raise RuntimeError("arm %s failed (%d):\n%s\n%s"
                                       % (name, proc.returncode, proc.stdout, proc.stderr))
                samples[name].append(parse_metric(proc.stdout + proc.stderr, metric_of[name]))
            ratios.append(samples[num][-1] / samples[den][-1])
            a, b = arms[0][0], arms[1][0]
            print("  round %2d/%d  %s %.4f  %s %.4f  %s/%s %.3f"
                  % (r + 1, args.rounds, a, samples[a][-1], b, samples[b][-1], num, den, ratios[-1]),
                  file=sys.stderr, flush=True)
    finally:
        if saved:
            shutil.copy2(saved, args.lib)
            os.unlink(saved)

    ratios.sort()
    print("\n%s%s / %s over %d paired rounds" % (args.label + ": " if args.label else "", num, den,
                                                 len(ratios)))
    medians, means = {}, {}
    for name, _ in arms:
        v = sorted(samples[name])
        medians[name] = statistics.median(v)
        means[name] = statistics.fmean(v)
        print("  %-10s median %.4f   mean %.4f   min %.4f   max %.4f   spread %.2fx"
              % (name, medians[name], means[name], v[0], v[-1],
                 v[-1] / v[0] if v[0] else float("nan")))
    print("  ratio      p10 %.3f   MEDIAN %.3f   p90 %.3f" % (percentile(ratios, 0.10),
                                                              statistics.median(ratios),
                                                              percentile(ratios, 0.90)))
    # The ratio OF THE MEDIANS, which is a different number from the median of the ratios and is the
    # one a published table of per-engine cells is quoting. Both are printed because they answer
    # different questions: the paired median says whether the effect resolves, and this says what the
    # table should read.
    print("  medians    %s/%s %.3f" % (num, den, medians[num] / medians[den]))
    print("  means      %s/%s %.3f" % (num, den, means[num] / means[den]))

    # A median is the wrong estimator for a BIMODAL arm, and this is where that gets noticed rather
    # than published. On the Core Ultra 9 285K at 24 threads both engines' VITS time splits into two
    # per-launch modes about 1.4x apart at roughly 50/50 (P4.30b), so the median is a coin flip
    # between them and moves the cell by that whole 1.4x. The mean does not. The test below is a
    # cheap one -- a gap in the middle of the sorted samples wider than a fifth of their range -- and
    # it is a prompt to look, not a diagnosis.
    for name, _ in arms:
        v = sorted(samples[name])
        if len(v) < 6 or v[-1] <= v[0]:
            continue
        # Only splits with real weight on BOTH sides -- a lone outlier is not a mode, and a warning
        # that fires on one is a warning nobody reads. 15% of the launches, and never fewer than two.
        floor = max(2, int(0.15 * len(v)))
        gaps = [(v[i + 1] - v[i], i) for i in range(len(v) - 1)
                if i + 1 >= floor and len(v) - (i + 1) >= floor]
        gap, at = max(gaps) if gaps else (0.0, 0)
        if gap > 0.2 * (v[-1] - v[0]):
            lo, hi = v[:at + 1], v[at + 1:]
            print("  ^ %s looks BIMODAL: %d launches near %.4f, %d near %.4f (%.2fx apart) -- a "
                  "median over that is unstable, quote the mean"
                  % (name, len(lo), statistics.median(lo), len(hi), statistics.median(hi),
                     statistics.median(hi) / statistics.median(lo)))
    # Epic-05's rule for this box, restated where it gets read: a p10 that crosses 1.0 means the
    # measurement did not resolve the effect, whatever the median says.
    if percentile(ratios, 0.10) < 1.0 < percentile(ratios, 0.90):
        print("  ^ p10/p90 straddle 1.0 -- WEAK, report as unresolved rather than as a number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
