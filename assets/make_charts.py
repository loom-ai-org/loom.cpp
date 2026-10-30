#!/usr/bin/env python3
"""The README's two benchmark charts, as light and dark SVGs (`assets/charts/*.svg`).

    python3 assets/make_charts.py

The numbers are copied from docs/BENCHMARKS.md, which holds their protocol and provenance; change them
there first, then here. Plain SVG with no dependencies, so the charts stay reviewable as text.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "charts"

# docs/BENCHMARKS.md, "Raspberry Pi 4": onnxruntime time / loom time, 4 threads, paired medians.
PI4 = [  # model, task, F32 vs onnxruntime FP32, Q8_0_PC vs onnxruntime int8
    ("VITS (piper)", "TTS", 1.34, 5.03),
    ("Pocket TTS", "TTS", 2.06, 3.27),
    ("Qwen3-0.6B", "LM", 1.06, 3.45),
    ("Moonshine tiny", "ASR", 1.13, 1.75),
    ("whisper-small", "ASR", 1.00, 1.17),
    ("Citrinet-256", "ASR", 0.65, 9.31),
]

# docs/BENCHMARKS.md, "Raspberry Pi Zero W: CPU against CPU+VC4": real-time factor of each device's
# faster precision, 2026-10-29/30 (P4.47, loom.cpp #106 at 6ffd561). CPU is the ARM11 alone; CPU+VC4 adds the GPU's 12 QPUs (ggml-vc4, "auto").
PIZERO = [  # model, task, CPU RTF, CPU+VC4 RTF
    ("WakeHuBERT tiny", "wake word", 0.17, 0.12),
    ("Silero VAD", "VAD", 0.18, 0.16),
    ("sanoTTS heartnano", "TTS", 0.22, 0.19),
    ("sanoTTS heart", "TTS", 0.54, 0.29),
    ("Citrinet-256", "ASR", 0.65, 0.31),
    ("sanoTTS thorsten (de)", "TTS", 0.72, 0.35),
    ("sanoTTS amy", "TTS", 2.30, 0.69),
    ("Moonshine tiny", "ASR", 1.80, 1.17),
    ("Conformer-CTC small", "ASR", 1.90, 1.39),
    ("Pocket TTS 2m", "TTS", 1.84, 1.39),
    ("VITS piper (miro)", "TTS", 5.83, 1.74),
    ("Pocket TTS 7m", "TTS", 3.94, 2.92),
    ("Pocket TTS 18m", "TTS", 4.85, 3.82),
]

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0", a="#0969da", b="#bf3989",
                  good="#1a7f37", ref="#cf222e"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#3d444d", a="#4493f8", b="#db61a2",
                 good="#3fb950", ref="#f85149"),
}
FONT = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""


def text(x, y, s, fill, size=13, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT}>{s}</text>')


def pi4_chart(t):
    w, left, right, row, top = 760, 170, 40, 46, 78
    h = top + row * len(PI4) + 40
    panel = (w - left - right - 30) / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" rx="10" fill="{t["bg"]}"/>',
           text(20, 30, "Raspberry Pi 4 — loom vs onnxruntime", t["fg"], 17, weight="600"),
           text(20, 52, "speedup over onnxruntime 1.28 at the same precision · 4 threads · higher is better",
                t["muted"], 12)]
    for p, (title, scale, color, idx) in enumerate([("F32 vs onnxruntime FP32", 2.5, t["a"], 2),
                                                    ("Q8_0_PC vs onnxruntime int8", 10.0, t["b"], 3)]):
        x0 = left + p * (panel + 30)
        out.append(text(x0, top - 6, title, t["fg"], 12, weight="600"))
        for k in range(0, int(scale) + 1, 1 if scale <= 3 else 2):
            gx = x0 + panel * k / scale
            out.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{h - 34}" stroke="{t["grid"]}" stroke-width="1"/>')
            out.append(text(gx, h - 18, f"{k}x", t["muted"], 11, "middle"))
        one = x0 + panel / scale
        out.append(f'<line x1="{one:.1f}" y1="{top - 2}" x2="{one:.1f}" y2="{h - 34}" stroke="{t["ref"]}" '
                   f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        for i, r in enumerate(PI4):
            v = r[idx]
            y = top + 10 + i * row
            bw = panel * min(v, scale) / scale
            out.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bw:.1f}" height="22" rx="4" fill="{color}"/>')
            label = f"{v:.2f}x"
            if bw > panel - 46:
                out.append(text(x0 + bw - 6, y + 16, label, "#ffffff", 12, "end", "600"))
            else:
                out.append(text(x0 + bw + 6, y + 16, label, t["fg"], 12, weight="600"))
    for i, (name, task, *_ ) in enumerate(PI4):
        y = top + 10 + i * row
        out.append(text(left - 12, y + 11, name, t["fg"], 13, "end", "600"))
        out.append(text(left - 12, y + 25, task, t["muted"], 11, "end"))
    out.append("</svg>")
    return "\n".join(out)


def pizero_chart(t):
    w, left, right, row, top, scale = 760, 200, 56, 46, 96, 6.0   # Pocket TTS 18m's CPU bar is 4.85, VITS's 5.85
    h = top + row * len(PIZERO) + 40
    span = w - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" rx="10" fill="{t["bg"]}"/>',
           text(20, 30, "Raspberry Pi Zero W — one 1 GHz ARM11 core, and its GPU's 12 QPUs", t["fg"], 17, weight="600"),
           text(20, 52, "real-time factor: seconds of compute per second of audio · lower is better · "
                        "left of the line is faster than real time", t["muted"], 12)]
    for k, (label, color) in enumerate([("CPU", t["muted"]), ("CPU + VC4", t["good"])]):
        x = 20 + k * 110
        out.append(f'<rect x="{x}" y="66" width="12" height="12" rx="2" fill="{color}"/>')
        out.append(text(x + 18, 76, label, t["fg"], 12))
    for k in range(int(scale) + 1):
        gx = left + span * k / scale
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{h - 34}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(gx, h - 16, f"{k:g}", t["muted"], 11, "middle"))
    one = left + span / scale
    out.append(f'<line x1="{one:.1f}" y1="{top - 8}" x2="{one:.1f}" y2="{h - 34}" stroke="{t["ref"]}" '
               f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    out.append(text(one + 6, top - 10, "real time", t["ref"], 11))
    for i, (name, task, cpu, vc4) in enumerate(PIZERO):
        y = top + 4 + i * row
        for k, (v, color) in enumerate([(cpu, t["muted"]), (vc4, t["good"])]):
            yy = y + k * 18
            bw = span * min(v, scale) / scale
            out.append(f'<rect x="{left}" y="{yy:.1f}" width="{bw:.1f}" height="15" rx="3" fill="{color}"/>')
            label = f"{v:.2f}" + (" ›" if v > scale else "")
            if v > scale:
                out.append(text(left + bw - 6, yy + 12, label, t["bg"], 11, "end", "600"))
            else:
                out.append(text(left + bw + 6, yy + 12, label, t["fg"], 11, weight="600"))
        out.append(text(left - 12, y + 15, name, t["fg"], 13, "end", "600"))
        out.append(text(left - 12, y + 29, task, t["muted"], 11, "end"))
    out.append("</svg>")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    for name, t in THEMES.items():
        (OUT / f"pi4-vs-onnxruntime-{name}.svg").write_text(pi4_chart(t) + "\n")
        (OUT / f"pi-zero-rtf-{name}.svg").write_text(pizero_chart(t) + "\n")


if __name__ == "__main__":
    main()
