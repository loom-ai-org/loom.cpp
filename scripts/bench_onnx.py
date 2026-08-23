# onnxruntime's side of the loom comparison: wall time with the duration predictor PINNED, and the
# per-op / per-shape profile that BACKLOG.md P4.16's table is built from. Not part of the build; run it
# on the Pi in the venv that has onnxruntime + phoonnx (`~/test` there).
#
#   source ~/test/bin/activate && python3 scripts/bench_onnx.py            # wall time, both harnesses
#   source ~/test/bin/activate && python3 scripts/bench_onnx.py --shapes   # per-shape conv profile
#
# THE PINNING IS THE POINT. VITS's duration predictor is stochastic and phoonnx does not seed it, so
# consecutive runs synthesise 72k-76k samples -- and both engines are near-linear in output samples, so
# an unpinned comparison times two different utterances. It cost this project a baseline (1.024 s) that
# stood for weeks and made every ratio derived from it ~4% optimistic. Pinned, onnxruntime is 1.044 s at
# 72192 samples, 1.063 s normalised to loom's 73472, repeatable to 0.5%.
#
# And profiling costs onnxruntime ~1.18x, so only the SHARES are used -- apportioned over an un-profiled
# wall time measured in the same process. This build's events carry no run_index, so runs split by ORDER.
import json, statistics, sys, time
import numpy as np
import onnxruntime as ort

MODEL = "/home/pi/pipertts-en-gb-miro/miro_en-GB.onnx"
PHONEMES = "hˈeɪ, kæn juː ʃˈʌtdaʊn ðə kəmpjˈuːtɐ, maɪ fɹˈɛnd?"
LOOM_SAMPLES = 73472          # what loom synthesises for the same utterance at seed 0
NRUN = 9

def phoneme_ids():
    from phoonnx.voice import TTSVoice
    return TTSVoice.load(MODEL), None

def session(profile):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.enable_profiling = profile
    return ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])

def feed(ids):
    x = np.array([ids], dtype=np.int64)
    return {"input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)}   # noise, length, noise_w -- PINNED

def timed(fn, n=NRUN):
    fn()                                                    # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), out

def main():
    voice, _ = phoneme_ids()
    ids = [int(i) for i in voice.phonemes_to_ids(list(PHONEMES))]
    s, f = session(False), None
    f = feed(ids)

    wall, lo, out = timed(lambda: s.run(None, f))
    n = np.asarray(out[0]).size
    print(f"raw session.run : median {wall:.4f}s  min {lo:.4f}s  samples={n}")
    print(f"                  {wall/n*1e6:.3f} us/sample -> {wall/n*LOOM_SAMPLES:.4f}s at loom's {LOOM_SAMPLES}")

    from phoonnx.config import SynthesisConfig
    cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)
    pwall, plo, pout = timed(lambda: voice.phoneme_ids_to_audio(ids, cfg))
    pn = pout.size
    print(f"via phoonnx     : median {pwall:.4f}s  min {plo:.4f}s  samples={pn}"
          f"  -> {pwall/pn*LOOM_SAMPLES:.4f}s at loom's {LOOM_SAMPLES}")
    if len(sys.argv) > 1 and sys.argv[1] == "--shapes":
        shapes(ids, wall)

def shapes(ids, wall, nrun=5):
    s2, f = session(True), feed(ids)
    for _ in range(nrun):
        s2.run(None, f)
    ev = json.load(open(s2.end_profiling()))
    kern = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
    per = len(kern) // nrun
    last = kern[-per * (nrun - 1):]                          # drop the first, warm run
    tot = sum(e["dur"] for e in last)

    def dims(e):
        out = []
        for t in e["args"].get("input_type_shape", []):
            for _, v in t.items():
                out.append(v)
        return out

    agg = {}
    for e in last:
        op = e["args"]["op_name"]
        if op not in ("Conv", "FusedConv", "ConvTranspose"):
            continue
        d = dims(e)
        key = (op, tuple(d[0]) if d else (), tuple(d[1]) if len(d) > 1 else ())
        a = agg.setdefault(key, [0, 0]); a[0] += e["dur"]; a[1] += 1

    conv = sum(v[0] for v in agg.values())
    print(f"\nprofiled {per} nodes/run over {nrun - 1} runs; shares apportioned over {wall*1e3:.1f} ms\n")
    print(f"{'op':<14}{'activation':>18}{'weight':>18}{'calls':>7}{'ms@wall':>9}{'%conv':>7}")
    for (op, x, w), (us, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{op:<14}{str(list(x)):>18}{str(list(w)):>18}{c//(nrun-1):>7}"
              f"{us/tot*wall*1e3:>9.1f}{us/conv*100:>6.1f}%")
    print(f"\nconvolution total: {conv/tot*wall*1e3:.1f} ms of {wall*1e3:.1f} ms wall ({conv/tot*100:.1f}%)")

if __name__ == "__main__":
    main()
