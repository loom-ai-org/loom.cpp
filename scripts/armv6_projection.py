# ADR-077 gate 1's output table: per model, the seconds left under
#   (a) the F32 items       RunFast (already in), the equal-effort F32 kernels, epilogue fusion
#   (b) (a) + dynamic int8  option 1: int8 weight GEMMs/GEMVs with a per-call activation quantize
#   (c) full static int8    option 2: int8 tensors between ops, calibrated scales, F32 islands
# from each model's board profile by class (scripts/armv6_opmix.py --json) and the per-class rates
# measured on the Pi Zero W at equal effort (bench_xnnpack citrinet, bench48, bench49, bench43).
#
#   python3 scripts/armv6_projection.py <dir of <model>.json>
#
# A projection, not a measurement: every isolated-op number is a ceiling (feedback in Retro-012's
# register), so the table orders the options and sizes the gaps; the end-to-end number is gate 4's.
# Each factor below names its source; the ones marked PROJECTED have no measurement behind them.
import json
import os
import sys

# Rates in GOP/s on the board, one thread, RunFast. GEMM regimes:
#   R1  Citrinet's 1x1 convs (K = N = 256, T = 138-1101), bench_xnnpack citrinet mode: ggml F32 0.50,
#       loom F32 asm (vector tile, TB 24-32) 0.71-0.72, loom int8 dynamic 1.04-1.05 (CF) / 1.11 (CL),
#       int8 GEMM alone 1.19-1.20.
#   R2  the vocoders' direct convs (bench49, 7 shapes, TB = 64, two rounds within 1%): F32 asm
#       1.10-1.11, int8 dynamic (asm quantize) 1.17-1.18, int8 GEMM alone 1.44-1.46.
#   R4  GEMV (bench48, Moonshine's 288 x 32768): F32 at the stream bound 0.158, int8 asm 0.339.
RATES = {
    "R1": {"f32": 0.715, "dyn": 1.05, "static": 1.20},
    "R2": {"f32": 1.105, "dyn": 1.175, "static": 1.45},
    "R4": {"f32": 0.158, "dyn": 0.339, "static": 0.339},
}
# Depthwise (bench43, Citrinet's 107 dw convs): ggml 3.96 s, F32 blocked RunFast 0.50 s, int8 C 0.68 s.
DW = {"f32": 0.50 / 3.96, "int8": 0.68 / 3.96}
# Per output element of an F32 island's boundary: quantize (amax + convert, bench_xnnpack `loom qcl`:
# 0.366 s over 11.6 M elements) + dequantize (PROJECTED at half of that).
Q_NS, DQ_NS = 31.5, 16.0
# The vocoder TTS models: conv_k and conv_transpose run as R2; everyone else's as R1.
VOCODERS = {"amy", "de051", "vits", "heart", "heartnano"}
# Small-T GEMMs (ner T = 19): PROJECTED at R1's int8 rates. Each call reads its int16-held weights
# once (4.7 MB for the FFN at 338 MB/s, 14 ms, against ~80 ms of MACs), so they stay compute-bound.
SMALL_T = {"ner"}


def project(name, d):
    g = lambda c, i=0: d.get(c, [0, 0, 0])[i]
    out = {}
    now = sum(v[0] for v in d.values())
    vocoder = name in VOCODERS
    conv_regime = "R2" if vocoder else "R1"

    def gemm_s(regime, mode, gflop, cur):
        return min(cur, gflop / RATES[regime][mode]) if gflop else cur

    for mode in ("f32", "dyn", "static"):
        s = 0.0
        # weight GEMMs and 1x1 convs; conv_k / conv_transpose by regime. distilbert-ner's F32 weights
        # (265 MB) do not fit the board beside anything else, so its (a) keeps today's Q4_0 GEMMs.
        if name in SMALL_T and mode == "f32":
            s += g("gemm")
        else:
            s += gemm_s("R1", mode, g("gemm", 1), g("gemm"))
        s += gemm_s(conv_regime, mode, g("conv_k", 1), g("conv_k"))
        s += gemm_s(conv_regime, mode, g("conv_transpose", 1), g("conv_transpose"))   # PROJECTED (R2 shape)
        s += gemm_s("R4", mode, g("gemv", 1), g("gemv"))
        s += g("dft")                                   # an FFT removes most of it in any precision: kept, listed
        dw = g("conv_dw")
        if mode == "static":                            # int8 dw, or an F32 island: whichever is cheaper
            s += min(dw * DW["int8"], dw * DW["f32"] + g("conv_dw", 2) * 1e6 * (Q_NS + DQ_NS) * 1e-9)
        else:
            s += dw * DW["f32"]
        # attention: F32 in (a) and (b) (option 1 quantizes weights only); int8 at R1's static/F32-asm
        # ratio in (c), PROJECTED -- the head dims (36-64) are far below R1's K.
        s += g("attn_mm") * (RATES["R1"]["f32"] / RATES["R1"]["static"] if mode == "static" else 1.0)
        # islands: softmax and norms stay F32; in (c) each pays a dequantize in and a quantize out
        for c in ("softmax", "norm"):
            s += g(c)
            if mode == "static":
                s += g(c, 2) * 1e6 * (Q_NS + DQ_NS) * 1e-9
        # elementwise: biases, masks and activations fold into the producer's epilogue in every
        # option (Citrinet's mask is the identity at batch 1); residuals stay, at half in int8 (a
        # quarter of the bytes, PROJECTED); the rest stays.
        s += g("ew_same") * (0.5 if mode == "static" else 1.0) + g("ew_other")
        # layout: a quarter of the bytes in static int8, PROJECTED at 0.4x
        s += g("layout") * (0.4 if mode == "static" else 1.0) + g("other")
        out[mode] = s
    return now, out, g("dft")


# Seconds of audio each run covers, for the real-time factor: jfk (11.0 s) and jfk3 (3.0 s) for the
# recognizers; the TTS outputs' sample counts at 24 kHz (sanoTTS) and 22.05 kHz (piperlite, VITS).
AUDIO = {"ct-f32": 11.0, "ct-q8": 11.0, "mt-f32": 11.0, "mt-q8": 11.0, "vad": 11.0, "wake": 11.0,
         "conformer": 3.0, "heartnano": 73472 / 24000, "heart": 76288 / 24000, "amy": 130048 / 22050,
         "de051": 97024 / 22050, "vits": 70400 / 22050}


def main():
    d = sys.argv[1]
    order = ["ct-f32", "ct-q8", "mt-f32", "mt-q8", "heartnano", "heart", "amy", "de051", "vits", "conformer",
             "ner", "vad", "wake"]
    print(f"{'model':10} {'today':>8} {'(a) F32':>8} {'(b) dyn':>8} {'(c) stat':>8} {'a/today':>8} {'b vs a':>7} "
          f"{'c vs a':>7} {'c vs b':>7} {'of a: DFT':>9}   RTF today / a / b / c")
    for m in order:
        p = os.path.join(d, m + ".json")
        if not os.path.exists(p):
            continue
        now, o, dft = project(m, json.load(open(p)))
        print(f"{m:10} {now:8.2f} {o['f32']:8.2f} {o['dyn']:8.2f} {o['static']:8.2f} {now / o['f32']:7.2f}x "
              f"{o['f32'] / o['dyn']:6.2f}x {o['f32'] / o['static']:6.2f}x {o['dyn'] / o['static']:6.2f}x {dft:9.2f}"
              + (f"   {now / AUDIO[m]:.2f} / {o['f32'] / AUDIO[m]:.2f} / {o['dyn'] / AUDIO[m]:.2f} / "
                 f"{o['static'] / AUDIO[m]:.2f}" if m in AUDIO else ""))


if __name__ == "__main__":
    main()
