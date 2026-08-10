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
DARK_GROUND = (22, 0.02, 60)
# The design's monochrome lockups use these two as their threads; see the note in main() for why
# nothing here writes those files.
WHITE = (99, 0.0, 0)
NEAR_WHITE = (99, 0.005, 75)


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


# --- the README header's optical compensation ------------------------------------------------------
#
# `logo-inline*.svg` is the same mark in a taller box, and the extra space is not decoration.
#
# GitHub strips `style` from markdown (verified: it replaces the attribute with its own
# `max-width:100%`), so the only vertical-alignment lever left on an inline image is the legacy
# `align="middle"`. That aligns the image's box centre with `baseline + x-height/2` -- the x-height,
# not the cap-height -- so beside a heading whose visual centre is `baseline + cap-height/2` the mark
# hangs low by `(cap - x) / 2`. At the mark's full size the drop is plainly visible.
#
# Padding the box BELOW the mark raises the mark within it by half the padding, which cancels exactly.
# The arithmetic, with the image rendered `WIDTH_PX` wide beside a `FONT_PX` heading:
#
#     mark centre sits above box centre by   WIDTH_PX * (vb_height - 84) / 168
#     it needs to sit above it by            (cap - x) / 2
#     so                                     vb_height = 84 + 84 * (cap - x) / WIDTH_PX
#
# **`DROP_PX` is measured, not derived, and the difference matters.** The obvious model says the drop
# is `(cap - x) / 2`, about 2.8px for a 32px heading -- and correcting by that much fixed only a third
# of it. The rest comes from the `<picture>` wrapper the dark-mode switch requires: `align="middle"`
# positions the `<img>` inside its parent inline box, and the *picture* then sits on the text baseline
# itself, so the real drop is larger than any font metric predicts. Rather than model that, it was
# measured: render the header, find the mark's and the wordmark's ink centres, and read off the gap.
#
# 9px at `WIDTH_PX = 52`, confirmed by sweeping the box height (84 -> +9.0px, 100 -> +4.0, 108 -> +2.0,
# 113 -> 0.0, 118 -> -2.0), which is also what makes the linear relationship above trustworthy: the
# measured points sit on it.
#
# **This variant exists so that `logo.svg` does not have to.** A compensation for one consumer's
# typography does not belong in the mark itself, which is used at other sizes and in other places.
WIDTH_PX = 52.0       # what the READMEs set on the header image
DROP_PX = 9.0         # how far the mark hangs below the wordmark without compensation


def inline_box_height() -> float:
    """The taller viewBox that puts the mark's optical centre on the heading's."""
    return 84.0 + 168.0 * DROP_PX / WIDTH_PX


def mark(threads, ground, stroke, title: str, box_height: float = 84.0) -> str:
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
        f'<svg xmlns="http://www.w3.org/2000/svg" width="84" height="{box_height:g}" '
        f'viewBox="0 0 84 {box_height:g}" role="img" aria-label="{title}">\n'
        f'  <title>{title}</title>\n'
        f'  <rect x="2" y="2" width="80" height="80" rx="16" fill="{ground}" '
        f'stroke="{stroke_attr}" stroke-width="2"/>\n'
        f"{rects}\n"
        f"</svg>\n"
    )


def main() -> None:
    threads = [oklch_to_hex(*RUST), oklch_to_hex(*OCHRE), oklch_to_hex(*INDIGO)]
    colour = mark(threads=threads, ground=oklch_to_hex(*CREAM), stroke=oklch_to_hex(*INK),
                  title="loom.cpp")
    # The same threads on the dark ground, and this is the one place these files deviate from the
    # artifact -- deliberately.
    #
    # The design's "on dark" lockup is MONOCHROME (white threads, `iconMonoSvgDark`), which is the
    # right call for a single-colour context. It is the wrong call for a README, because a README's
    # dark variant is what most people see: `<picture>` serves it on `prefers-color-scheme: dark`, so
    # shipping the monochrome there means the brand's colours never appear for the majority of
    # readers. Recolouring the ground instead keeps rust/ochre/indigo in both themes, each on a ground
    # that suits its page.
    #
    # The alternative -- serving the cream-tiled mark everywhere -- also works and reads fine, but it
    # puts a bright tile on a dark page where this sits in it. The monochrome variants the design
    # defines are still one call away: `mark([ink]*3, near_white, ink)` and `mark([white]*3, dark,
    # None)`; nothing here uses them, so nothing here writes them.
    dark = mark(threads=threads, ground=oklch_to_hex(*DARK_GROUND), stroke=None, title="loom.cpp")
    (HERE / "logo.svg").write_text(colour)
    (HERE / "logo-dark.svg").write_text(dark)

    # The same two marks in the taller box the README header needs -- see `inline_box_height`.
    box = inline_box_height()
    (HERE / "logo-inline.svg").write_text(
        mark(threads=threads, ground=oklch_to_hex(*CREAM), stroke=oklch_to_hex(*INK),
             title="loom.cpp", box_height=box))
    (HERE / "logo-inline-dark.svg").write_text(
        mark(threads=threads, ground=oklch_to_hex(*DARK_GROUND), stroke=None,
             title="loom.cpp", box_height=box))
    print(f"  inline box  84 x {box:.1f} units "
          f"({box - 84:.1f} of bottom padding, cancelling the align=middle drop)")
    for name, spec in (("rust", RUST), ("ochre", OCHRE), ("indigo", INDIGO),
                       ("cream", CREAM), ("ink", INK), ("dark ground", DARK_GROUND)):
        print(f"  {name:12} oklch({spec[0]}% {spec[1]} {spec[2]})  ->  {oklch_to_hex(*spec)}")
    print(f"wrote {HERE/'logo.svg'} and {HERE/'logo-dark.svg'}")


if __name__ == "__main__":
    main()
