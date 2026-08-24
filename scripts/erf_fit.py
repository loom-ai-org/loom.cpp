#!/usr/bin/env python3
"""Derive and bound the erf approximation in cmake/patches/ggml-0010-gelu-erf-simd.patch.

NOT part of the build. Kept because the six-plus-six coefficients in that patch are otherwise
unexplainable magic numbers, and the next person to want to change the degree, move the clamp, or
answer "is this fit good enough" needs the thing that produced them.

    python3 scripts/erf_fit.py            # refit, print the C arrays, report the fit error

WHAT IT FITS.  erf(z) ~ z * P(w)/Q(w) with w = z*z, P and Q of degree 5, on z in [0, 4].  A rational
rather than a polynomial because a polynomial needs degree 14 in w to reach 6e-07 (14 FMAs, worse than
this) -- erf's tail is not something a polynomial approximates cheaply.  Degree 5/5 because 4/4 is
5.1e-06 and 6/6 does not converge under this fit.

WHAT THIS SCRIPT DOES **NOT** GIVE YOU: a bound.  It is iteratively reweighted least squares, not a
Remez exchange, so there is no equioscillation certificate, and the number it prints is a maximum over
a grid.  It is also in float64, and the shipped kernel is float32 -- where rounding in the Horner
evaluation turns out to dominate the fit error by about 25x, which is why tightening these
coefficients would buy nothing.

**The bound comes from enumeration instead, and it is stronger than any certificate would be:** the
input type is finite, so the patch's accuracy claim is a sweep over all 2^32 float32 values against a
double-precision erf (about 30 s on 24 cores).  See ggml-0010's header for those numbers, and
epic-05's operating notes for why that is the right instrument for a float32 kernel.
"""
import numpy as np
from math import erf

CLAMP, DP, DQ = 4.0, 5, 5
SQRT_2_INV = 0.70710678118654752440084436210484


def fit(clamp=CLAMP, dp=DP, dq=DQ, npts=400_001, iters=80):
    """erf(z) ~ z*P(w)/Q(w). Linearise as z*P(w) - erf*Q(w) = 0 with q0 fixed at 1, then reweight by
    1/Q -- which is what turns a least-squares fit of the NUMERATOR into one of the ratio."""
    z = np.linspace(1e-7, clamp, npts)
    w = z * z
    y = np.array([erf(v) for v in z])
    cols = [z * w**i for i in range(dp + 1)] + [-y * w**i for i in range(1, dq + 1)]
    A = np.stack(cols, 1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(iters):
        q = np.r_[1.0, sol[dp + 1:]]
        Qv = np.polyval(q[::-1], w)
        sol, *_ = np.linalg.lstsq(A / Qv[:, None], y / Qv, rcond=None)
    p, q = sol[:dp + 1], np.r_[1.0, sol[dp + 1:]]
    approx = z * np.polyval(p[::-1], w) / np.polyval(q[::-1], w)
    return p, q, np.abs(approx - y).max()


def gelu_f32(x, p, q):
    """The shipped kernel's arithmetic, in float32 throughout -- including the clamp and the
    saturation of erf to [-1, 1], both of which are load-bearing (see the patch header)."""
    f = np.float32
    x = f(x)
    z = np.clip(f(x * f(SQRT_2_INV)), f(-CLAMP), f(CLAMP))
    w = f(z * z)
    P, Q = f(p[DP]), f(q[DQ])
    for i in range(DP - 1, -1, -1):
        P, Q = f(f(P * w) + f(p[i])), f(f(Q * w) + f(q[i]))
    e = np.clip(f(z * f(P / Q)), f(-1.0), f(1.0))
    return f(f(0.5) * x * f(f(1.0) + e))


if __name__ == "__main__":
    p, q, err = fit()
    print(f"fit (float64, grid): max abs err on erf over [0, {CLAMP}] = {err:.3e}\n")
    for name, c in (("ggml_erf_rat_p", p), ("ggml_erf_rat_q", q)):
        print(f"static const float {name}[6] = {{" + ", ".join(f"{v:.9e}f" for v in c) + "};")

    # The f32 story, on a grid -- the real bound is the exhaustive sweep, see the docstring.
    x = np.float32(np.linspace(-8, 8, 1_000_001))
    ref = np.array([0.5 * float(v) * (1.0 + erf(float(v) * SQRT_2_INV)) for v in x])
    cand = gelu_f32(x, p, q).astype(np.float64)
    libm = np.array([np.float32(0.5) * np.float32(v)
                     * (np.float32(1.0) + np.float32(erf(float(v) * SQRT_2_INV))) for v in x])
    print(f"\ngelu in f32 over |x| <= 8, vs a float64 reference:")
    print(f"  this fit      max abs {np.abs(cand - ref).max():.3e}")
    print(f"  erff() (ggml) max abs {np.abs(libm - ref).max():.3e}")
    print("  -> the fit error is not what limits this; f32 Horner rounding is.")
