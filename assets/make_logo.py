#!/usr/bin/env python3
"""Renders the loom.cpp mark to SVG, from the geometry the design artifact defines.

`Loom.cpp-Logo.dc.html` is the source of truth for the mark, but it is a Claude artifact: the icon
exists there only as `React.createElement` calls evaluated in a browser, so nothing can `<img src>` it.
This script is that geometry written out as a file -- same 84x84 box, same three warp threads at
x = 12/35/58, same radii and stroke widths -- so the asset in each README is reproducible from the
design rather than traced by hand from a screenshot.

**The colours are converted from oklch to sRGB hex, and that is not a preference.** The artifact
specifies every colour in oklch, which browsers handle and the SVG rasterisers behind GitHub's image
proxy historically do not -- an unconverted file renders black-on-black in exactly the place a logo
most needs to work. The conversion below is the standard oklch -> Oklab -> linear sRGB -> sRGB pipeline,
so the hex values are the same colours a browser would have shown.

    python3 assets/make_logo.py            # rewrites logo.svg and logo-dark.svg in place
"""
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- the artifact's palette, verbatim -------------------------------------------------------------
RUST = (58, 0.15, 35)
OCHRE = (70, 0.13, 75)
INDIGO = (45, 0.09, 265)
CREAM = (97, 0.012, 75)
INK = (30, 0.02, 60)
WHITE = (99, 0.0, 0)
DARK_GROUND = (22, 0.02, 60)


def oklch_to_hex(lightness_pct: float, chroma: float, hue_deg: float) -> str:
    """oklch -> sRGB hex, via Oklab and linear sRGB (Björn Ottosson's matrices)."""
    lightness = lightness_pct / 100.0
    a = chroma * math.cos(math.radians(hue_deg))
    b = chroma * math.sin(math.radians(hue_deg))

    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3

    linear = (
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )

    def gamma(channel: float) -> int:
        channel = max(0.0, min(1.0, channel))
        encoded = 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, encoded)) * 255)

    return "#{:02x}{:02x}{:02x}".format(*(gamma(c) for c in linear))


def mark(threads, ground, stroke, title: str) -> str:
    """The icon: a rounded square with three vertical warp threads across it.

    `strip = 15, gap = 8, start = 12` puts the threads at x = 12, 35, 58 and gives each a height of
    `15 * 3 + 8 * 2 = 61` -- the artifact's own arithmetic, kept as arithmetic so it stays checkable.
    """
    strip, gap, start = 15, 8, 12
    span = strip * 3 + gap * 2
    positions = [start + i * (strip + gap) for i in range(3)]
    stroke_attr = "none" if stroke is None else stroke
    rects = "\n".join(
        f'  <rect x="{x}" y="{start}" width="{strip}" height="{span}" rx="3" '
        f'fill="{colour}" stroke="{stroke_attr}" stroke-width="1.25"/>'
        for x, colour in zip(positions, threads)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84" viewBox="0 0 84 84" '
        f'role="img" aria-label="{title}">\n'
        f'  <title>{title}</title>\n'
        f'  <rect x="2" y="2" width="80" height="80" rx="16" fill="{ground}" '
        f'stroke="{stroke_attr}" stroke-width="2"/>\n'
        f"{rects}\n"
        f"</svg>\n"
    )


def main() -> None:
    colour = mark(
        threads=[oklch_to_hex(*RUST), oklch_to_hex(*OCHRE), oklch_to_hex(*INDIGO)],
        ground=oklch_to_hex(*CREAM), stroke=oklch_to_hex(*INK), title="loom.cpp",
    )
    # The artifact's "on dark" lockup: white threads on the dark ground, no stroke.
    dark = mark(
        threads=[oklch_to_hex(*WHITE)] * 3,
        ground=oklch_to_hex(*DARK_GROUND), stroke=None, title="loom.cpp",
    )
    (HERE / "logo.svg").write_text(colour)
    (HERE / "logo-dark.svg").write_text(dark)
    for name, spec in (("rust", RUST), ("ochre", OCHRE), ("indigo", INDIGO),
                       ("cream", CREAM), ("ink", INK), ("dark ground", DARK_GROUND)):
        print(f"  {name:12} oklch({spec[0]}% {spec[1]} {spec[2]})  ->  {oklch_to_hex(*spec)}")
    print(f"wrote {HERE/'logo.svg'} and {HERE/'logo-dark.svg'}")


if __name__ == "__main__":
    main()
