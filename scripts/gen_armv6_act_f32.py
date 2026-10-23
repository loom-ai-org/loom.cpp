# ADR-079 P4.32f f2 step 3a: the ARMv6 activation kernels, generated into
# ggml/src/ggml-cpu/arch/arm/armv6-act-f32.inc (a C++ include holding the declarations and the asm as one
# top-level asm block, as armv6-conv-f32.inc is).
#
#   python3 scripts/gen_armv6_act_f32.py <ggml>/src/ggml-cpu/arch/arm/armv6-act-f32.inc
#
# GELU_ERF, BIT-IDENTICAL TO ggml-0010's SCALAR TAIL (what ARMv6 ran: there is no SIMD body there):
#
#     z = clamp4(x * SQRT_2_INV);  w = z*z
#     p = ((((P5 w + P4) w + P3) w + P2) w + P1) w + P0      q likewise with Q
#     e = clamp1(z * (p / q));     y = (0.5 x) * (1 + e)
#
# with every operation the same IEEE single operation in the same order, so each element rounds exactly
# as the scalar tail does (VFPv2's vmla is not fused, so GCC's contraction of `P5*w + P4` into one is the
# same two roundings). The scalar tail paid ~200 ns an element on a Pi Zero W: each Horner step waits on
# the one before it, and each of the four clamps is a vcmpe + vmrs that drains the VFP pipeline.
#
# Here, 8 elements a pass (scripts/armv6_shortvec.py: LEN = 4, STRIDE = 1, the thread's RunFast):
#   A  z = x * S for two vectors, stored to an 8-float scratch `s`
#   B  the clamp to [-4, 4] IN THE INTEGER UNIT, in place on `s` (below)
#   C  w = z*z (s16, s20); p (s8, s12) and q (s24, s28) by Horner, interleaved: four independent vector
#      chains, since a vector op waits ~14 cycles for the previous one on its destination (P4.32e)
#   D  d = p / q;  e = z * d (z reloaded from `s`), stored to `s`
#   E  the clamp to [-1, 1], in place on `s`
#   F  y = (x * 0.5) * (e + 1) -- x reloaded
# Bank 0 holds the scalars: the twelve coefficients stream through s0-s7 as the levels use them.
#
# THE INTEGER CLAMP is exactly `v = v < c ? v : c; v = v > -c ? v : -c`: |v| >= c gives c with v's sign
# (inf included, and v = +-c itself is unchanged), a NaN of either sign gives +c (both compares false), and
# everything else is kept -- a denormal too, which the VFP's compares see as 0 under flush-to-zero.
#
# SIGMOID, SILU AND TANH, NOT bit-identical to glibc (whose expf and tanhf ARMv6 called per element, 123
# and 437 ns): an own e^v, Cephes' single-precision construction -- Cody-Waite reduction v = n ln2 + r with
# ln2 split in two, e^r = 1 + r + r^2 (((((C0 r + C1) r + C2) r + C3) r + C4) r + C5), then 2^n added to
# the exponent in the integer unit, which also takes the range cases: |v| >= 88.37 (where n would reach
# 128) gives inf or 0 by sign, a NaN passes through. Then
#     sigmoid  1 / (e^-x + 1)            silu  x / (e^-x + 1)
#     tanh     |x| < 0.625: x + x^3 P(x^2) (Cephes' minimax), else 1 - 2 / (e^2|x| + 1); x's sign on both
# Measured against f64 over every 11th float with |x| <= 90: sigmoid 2.40 ULP (glibc 2.39), tanh 1.30 ULP
# (glibc 2.16); silu is sigmoid's quotient with x on top.
#
# ELU (P4.41, ggml-0052): the same exp with v = x, and expm1 taken from its polynomial before the 1 is
# added wherever the reduction leaves n = 0 -- see elu().
#
# Functions: ggml_armv6_{gelu_erf,sigmoid,silu,tanh,elu}_f32(args), args = { x, y, s, n8 }; `s` is 32 floats
# of scratch here (gelu_erf uses 8).
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armv6_shortvec as sv  # noqa: E402

FIELDS = [("const float *", "x", "input, 8 * n8 floats"),
          ("float *", "y", "output, 8 * n8 floats; may be x"),
          ("float *", "s", "32 floats of scratch, not overlapping x or y"),
          ("int32_t", "n8", "blocks of 8, >= 1")]
OFF = {name: 4 * i for i, (_, name, _) in enumerate(FIELDS)}

# ggml-0010's constants, as the f32 values ggml compiles them to
SQRT_2_INV = 0.70710678118654752440084436210484
P = [1.128379046e+00, 1.923043769e-01, 5.298133013e-02, 3.844561124e-03, 2.861880010e-04, 2.027359838e-06]
Q = [1.000000000e+00, 5.037572164e-01, 1.148780232e-01, 1.512413709e-02, 1.181028205e-03, 3.792931260e-05]


def f32hex(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0]


L = []
e = L.append


def clamp_pass(cb_reg):
    """In place on the 8 floats at r2 (the scratch): r7 = the bound's bits, r8 = 0x7f800000."""
    for k in range(8):
        e(f"    ldr r4, [r2, #{4 * k}]")
        e("    bic r5, r4, #0x80000000")
        e(f"    cmp r5, {cb_reg}")
        e("    andhs r4, r4, #0x80000000")
        e(f"    orrhs r4, r4, {cb_reg}")
        e("    cmp r5, r8")
        e(f"    movhi r4, {cb_reg}")
        e(f"    str r4, [r2, #{4 * k}]")


def gelu_erf():
    name = "ggml_armv6_gelu_erf_f32"
    e(f"    .global {name}\n    .type {name}, %function\n    .align 4\n{name}:")
    e("    push {r4-r9}")
    e("    vpush {d8-d15}")
    e(f"    ldr r1, [r0, #{OFF['y']}]")
    e(f"    ldr r2, [r0, #{OFF['s']}]")
    e(f"    ldr r3, [r0, #{OFF['n8']}]")
    e(f"    ldr r0, [r0, #{OFF['x']}]")
    e("    adr r6, 9f")                      # the constant table
    e("    ldr r8, [r6, #68]")              # 0x7f800000: not an ARM immediate (odd rotation)
    sv.enter(e)
    e("1:")
    e("    pld [r0, #64]")
    # A: z = x * S
    e("    vldr s0, [r6]")                   # S
    e("    vldmia r0, {s8-s15}")
    e("    vmul.f32 s8, s8, s0")
    e("    vmul.f32 s12, s12, s0")
    e("    vstmia r2, {s8-s15}")
    # B: clamp z to [-4, 4]
    e("    ldr r7, [r6, #4]")                # bits of 4.0
    clamp_pass("r7")
    # C: w, then p and q by Horner. Table at r6 + 12: P5 Q5 P4 Q4 P3 Q3 P2 Q2 P1 Q1 P0 Q0.
    e("    vldmia r2, {s8-s15}")
    e("    add r9, r6, #12")
    e("    vldmia r9!, {s0-s7}")             # P5 Q5 P4 Q4 P3 Q3 P2 Q2
    e("    vmul.f32 s16, s8, s8")
    e("    vmul.f32 s20, s12, s12")
    e("    vmul.f32 s8, s16, s0")            # p = w * P5
    e("    vmul.f32 s24, s16, s1")           # q = w * Q5
    e("    vmul.f32 s12, s20, s0")
    e("    vmul.f32 s28, s20, s1")
    slots = [(2, 3), (4, 5), (6, 7), (0, 1), (2, 3)]   # (P_k, Q_k) for k = 4 .. 0
    for lvl, (sp, sq) in enumerate(slots):
        if lvl == 3:
            e("    vldmia r9, {s0-s3}")      # P1 Q1 P0 Q0 (P5..Q4's slots are dead)
        e(f"    vadd.f32 s8, s8, s{sp}")
        e(f"    vadd.f32 s24, s24, s{sq}")
        e(f"    vadd.f32 s12, s12, s{sp}")
        e(f"    vadd.f32 s28, s28, s{sq}")
        if lvl < 4:
            e("    vmul.f32 s8, s8, s16")
            e("    vmul.f32 s24, s24, s16")
            e("    vmul.f32 s12, s12, s20")
            e("    vmul.f32 s28, s28, s20")
    # D: d = p / q, e = z * d
    e("    vdiv.f32 s8, s8, s24")
    e("    vdiv.f32 s12, s12, s28")
    e("    vldmia r2, {s16-s23}")            # z, clamped
    e("    vmul.f32 s8, s16, s8")
    e("    vmul.f32 s12, s20, s12")
    e("    vstmia r2, {s8-s15}")
    # E: clamp e to [-1, 1]
    e("    ldr r7, [r6, #8]")                # bits of 1.0
    clamp_pass("r7")
    # F: y = (x * 0.5) * (e + 1)
    e("    vldr s0, [r6, #60]")              # 0.5
    e("    vldr s1, [r6, #64]")              # 1.0
    e("    vldmia r2, {s8-s15}")
    e("    vldmia r0!, {s16-s23}")
    e("    vmul.f32 s16, s16, s0")
    e("    vmul.f32 s20, s20, s0")
    e("    vadd.f32 s8, s8, s1")
    e("    vadd.f32 s12, s12, s1")
    e("    vmul.f32 s16, s16, s8")
    e("    vmul.f32 s20, s20, s12")
    e("    vstmia r1!, {s16-s23}")
    e("    subs r3, r3, #1")
    e("    bne 1b")
    sv.leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r9}")
    e("    bx lr")
    e("    .align 2")
    e("9:")
    table = [f32hex(SQRT_2_INV), f32hex(4.0), f32hex(1.0)]
    for k in range(5, -1, -1):
        table += [f32hex(P[k]), f32hex(Q[k])]
    table += [f32hex(0.5), f32hex(1.0), "0x7f800000"]
    for t in table:
        e(f"    .word {t}")
    e(f"    .size {name}, .-{name}")
    return name


# Cephes' expf and tanhf constants
LOG2E, MAGIC, LN2HI, LN2LO = 1.44269504088896341, 12582912.0, 0.693359375, -2.12194440e-4
C = [1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1]
T = [-5.70498872745e-3, 2.06390887954e-2, -5.37397155531e-2, 1.33314422036e-1, -3.33332819422e-1]
EXP_TABLE = [LOG2E, MAGIC, LN2HI, LN2LO] + C + [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0] + T + [0.625]
# word offsets in the table
O_C, O_ONE, O_TWO, O_T, O_SMALL = 4, 10, 14, 18, 23


def exp_core(neg, tanh, elu=False):
    """e^v for the 8 floats at r0 into s[8..15] (scratch r2 + 32), v = -x (neg), 2|x| (tanh) or x (elu). The
    block's t = v log2e + magic goes to s[0..7] (scratch r2) for the integer pass. Leaves r9 at r2 + 32.
    With `elu`, q = r + r^2 P(r) -- e^r - 1 before the 1 is added -- also goes to scratch r2 + 64."""
    e("    vldmia r0, {s16-s23}")
    if neg:
        e("    vneg.f32 s16, s16")
        e("    vneg.f32 s20, s20")
    if tanh:
        e("    vabs.f32 s16, s16")
        e("    vabs.f32 s20, s20")
        e("    vadd.f32 s16, s16, s16")
        e("    vadd.f32 s20, s20, s20")
    e("    vldmia r6, {s0-s7}")              # LOG2E MAGIC LN2HI LN2LO C0 C1 C2 C3
    e("    vmul.f32 s24, s16, s0")           # t = v * log2e + magic
    e("    vmul.f32 s28, s20, s0")
    e("    vadd.f32 s24, s24, s1")
    e("    vadd.f32 s28, s28, s1")
    e("    vstmia r2, {s24-s31}")
    e("    vsub.f32 s24, s24, s1")           # nf = t - magic
    e("    vsub.f32 s28, s28, s1")
    e("    vmls.f32 s16, s24, s2")           # r = v - nf ln2hi
    e("    vmls.f32 s20, s28, s2")
    e("    vmls.f32 s16, s24, s3")           # r = r - nf ln2lo
    e("    vmls.f32 s20, s28, s3")
    e("    vmul.f32 s8, s16, s4")            # p = C0 r + C1, then Horner
    e("    vmul.f32 s12, s20, s4")
    e("    vmul.f32 s24, s16, s16")          # r^2, on the side
    e("    vmul.f32 s28, s20, s20")
    e("    vadd.f32 s8, s8, s5")
    e("    vadd.f32 s12, s12, s5")
    for k in (6, 7):
        e("    vmul.f32 s8, s8, s16")
        e("    vmul.f32 s12, s12, s20")
        e(f"    vadd.f32 s8, s8, s{k}")
        e(f"    vadd.f32 s12, s12, s{k}")
    e(f"    add r9, r6, #{4 * (O_C + 4)}")
    e("    vldmia r9, {s0-s1}")              # C4 C5
    for k in (0, 1):
        e("    vmul.f32 s8, s8, s16")
        e("    vmul.f32 s12, s12, s20")
        e(f"    vadd.f32 s8, s8, s{k}")
        e(f"    vadd.f32 s12, s12, s{k}")
    e(f"    vldr s2, [r6, #{4 * O_ONE}]")
    e("    vmul.f32 s8, s8, s24")            # p = p r^2 + r + 1
    e("    vmul.f32 s12, s12, s28")
    e("    vadd.f32 s8, s8, s16")
    e("    vadd.f32 s12, s12, s20")
    if elu:
        e("    add r9, r2, #64")
        e("    vstmia r9, {s8-s15}")         # q = e^r - 1, kept for n = 0 (no cancellation there)
    e("    vadd.f32 s8, s8, s2")
    e("    vadd.f32 s12, s12, s2")
    e("    add r9, r2, #32")
    e("    vstmia r9, {s8-s15}")
    # The integer pass: e = p * 2^n, n = bits(t) - bits(magic), added to p's exponent field. Both ends of the
    # range are taken here, and they are NOT symmetric: p's exponent field is 126 or 127, so n has to stay in
    # [-125, 127] for the sum to be a normal float. Above, v >= 88.37 keeps n <= 127 and gives inf. Below,
    # v <= -86.98 (n <= -126 from -86.99) gives 0: e^v is then within 2^-125 of zero, and a sum past the
    # field's bottom WRAPS INTO THE SIGN BIT and comes out a negative NaN -- which a symmetric |v| >= 88.37
    # bound did for every x in (86.98, 88.37], and turned heart's SILU outputs to NaN.
    # r7 = bits of the |x| from which v >= 88.37 (88.37 for sigmoid/silu's v = -x, 44.185 for tanh's 2|x|),
    # r12 = bits(86.98) (sigmoid/silu: v = -x <= -86.98), r8 = 0x7f800000, r10 = bits(magic).
    for k in range(8):
        e(f"    ldr r4, [r0, #{4 * k}]")      # x
        e(f"    ldr r5, [r2, #{4 * k}]")      # t
        e(f"    ldr r3, [r9, #{4 * k}]")      # p
        e("    sub r5, r5, r10")
        e("    add r3, r3, r5, lsl #23")
        e("    bic r5, r4, #0x80000000")
        if elu:
            e("    cmp r5, r12")
            e("    bichs r3, r3, r4, asr #31")   # |x| >= 86.98: 0 for x < 0 (v <= -86.98), kept for x > 0
            e("    cmp r5, r7")
            e("    bichs r3, r8, r4, asr #31")   # |x| >= 88.37: inf for x > 0, 0 for x < 0
        elif neg:
            e("    cmp r5, r12")
            e("    andhs r3, r3, r4, asr #31")   # |x| >= 86.98: 0 for x > 0 (v <= -86.98), kept for x < 0
            e("    cmp r5, r7")
            e("    andhs r3, r8, r4, asr #31")   # |x| >= 88.37: inf for x < 0 (v >= 88.37), 0 for x > 0
        else:
            e("    cmp r5, r7")
            e("    movhs r3, r8")                # v = 2|x| >= 88.37: inf (v is never negative here)
        e("    cmp r5, r8")
        e("    movhi r3, r4")                    # NaN in, NaN out
        e(f"    str r3, [r9, #{4 * k}]")


def act(kind):
    name = f"ggml_armv6_{kind}_f32"
    e(f"    .global {name}\n    .type {name}, %function\n    .align 4\n{name}:")
    e("    push {r4-r11}")
    e("    vpush {d8-d15}")
    e(f"    ldr r1, [r0, #{OFF['y']}]")
    e(f"    ldr r2, [r0, #{OFF['s']}]")
    e(f"    ldr r11, [r0, #{OFF['n8']}]")
    e(f"    ldr r0, [r0, #{OFF['x']}]")
    e("    ldr r6, 5f")                      # the table, PC-relative (adr cannot reach it from tanh's body)
    e("4:  add r6, pc, r6")
    e("    ldr r7, 8f")
    e("    ldr r8, 7f")
    e("    ldr r10, 6f")
    sv.enter(e)
    if kind != "tanh":
        e("    ldr r12, 3f")                     # bits(86.98), after enter (which uses r12)
    e("1:")
    e("    pld [r0, #64]")
    if kind == "tanh":
        # the small-|x| branch first: s = x + x^3 P(x^2), Cephes' order, to the scratch at r2 + 64
        e("    vldmia r0, {s16-s23}")
        e(f"    add r9, r6, #{4 * O_T}")
        e("    vldmia r9, {s0-s4}")          # T0 .. T4
        e("    vmul.f32 s24, s16, s16")      # z = x*x
        e("    vmul.f32 s28, s20, s20")
        e("    vmul.f32 s8, s24, s0")        # T0 z + T1, then Horner
        e("    vmul.f32 s12, s28, s0")
        e("    vadd.f32 s8, s8, s1")
        e("    vadd.f32 s12, s12, s1")
        for k in (2, 3, 4):
            e("    vmul.f32 s8, s8, s24")
            e("    vmul.f32 s12, s12, s28")
            e(f"    vadd.f32 s8, s8, s{k}")
            e(f"    vadd.f32 s12, s12, s{k}")
        e("    vmul.f32 s8, s8, s24")        # * z, * x, + x
        e("    vmul.f32 s12, s12, s28")
        e("    vmul.f32 s8, s8, s16")
        e("    vmul.f32 s12, s12, s20")
        e("    vadd.f32 s8, s8, s16")
        e("    vadd.f32 s12, s12, s20")
        e("    add r9, r2, #64")
        e("    vstmia r9, {s8-s15}")
    exp_core(kind != "tanh", kind == "tanh")
    e("    vldmia r9, {s8-s15}")             # e
    e(f"    vldr s2, [r6, #{4 * O_ONE}]")
    e("    vadd.f32 s8, s8, s2")             # e + 1
    e("    vadd.f32 s12, s12, s2")
    if kind == "sigmoid":
        e(f"    add r9, r6, #{4 * O_ONE}")
        e("    vldmia r9, {s0-s3}")          # 1.0 x 4, a VECTOR numerator (bank-0 Sn is a vector)
        e("    vdiv.f32 s16, s0, s8")
        e("    vdiv.f32 s20, s0, s12")
    elif kind == "silu":
        e("    vldmia r0, {s16-s23}")
        e("    vdiv.f32 s16, s16, s8")
        e("    vdiv.f32 s20, s20, s12")
    if kind != "tanh":
        e("    add r0, r0, #32")
        e("    vstmia r1!, {s16-s23}")
    else:
        e(f"    add r9, r6, #{4 * O_ONE}")
        e("    vldmia r9, {s0-s7}")          # 1.0 x 4, 2.0 x 4: vector operands from bank 0
        e("    vdiv.f32 s16, s4, s8")        # l = 2 / (e + 1)
        e("    vdiv.f32 s20, s4, s12")
        e("    vsub.f32 s16, s0, s16")       # l = 1 - l
        e("    vsub.f32 s20, s0, s20")
        e("    add r9, r2, #32")
        e("    vstmia r9, {s16-s23}")
        # select: |x| < 0.625: s; else l; both with x's sign (s has it already except at -0, where x + x^3 P
        # is +0 and glibc's tanhf gives -0); NaN: x. Written straight to y (x[k] is read first).
        e(f"    ldr r10, [r6, #{4 * O_SMALL}]")
        for k in range(8):
            e(f"    ldr r4, [r0, #{4 * k}]")
            e(f"    ldr r3, [r9, #{4 * k}]")
            e(f"    ldr r5, [r2, #{64 + 4 * k}]")
            e("    and r12, r4, #0x80000000")
            e("    orr r3, r3, r12")
            e("    orr r5, r5, r12")                # x's sign on s too: x + x^3 P is +0 at x = -0
            e("    bic r12, r4, #0x80000000")
            e("    cmp r12, r10")
            e("    movlo r3, r5")
            e("    cmp r12, r8")
            e("    movhi r3, r4")
            e(f"    str r3, [r1, #{4 * k}]")
        e("    ldr r10, 6f")                 # bits(magic) again for the next block
        e("    add r0, r0, #32")
        e("    add r1, r1, #32")
    e("    subs r11, r11, #1")
    e("    bne 1b")
    sv.leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r11}")
    e("    bx lr")
    e("    .align 2")
    e(f"3:  .word {f32hex(86.98)}")
    e("5:  .word 9f - (4b + 8)")
    e("6:  .word 0x4b400000")               # bits(magic)
    e("7:  .word 0x7f800000")
    e(f"8:  .word {f32hex(44.185 if kind == 'tanh' else 88.37)}")
    e("9:")
    for v in EXP_TABLE:
        e(f"    .word {f32hex(v)}")
    e(f"    .size {name}, .-{name}")
    return name


def elu():
    """ELU (alpha 1): x for x >= +0, else expm1(x). expm1 is q = r + r^2 P(r) where the reduction leaves
    n = 0 (|x| < ~0.347, where e^x - 1 would cancel), else e^x - 1, which cannot cancel there (e^x <= 0.71);
    then x itself for |x| < 2^-24 (expm1 rounds to x there, and q would lose -0's sign) and for a NaN."""
    name = "ggml_armv6_elu_f32"
    e(f"    .global {name}\n    .type {name}, %function\n    .align 4\n{name}:")
    e("    push {r4-r11}")
    e("    vpush {d8-d15}")
    e(f"    ldr r1, [r0, #{OFF['y']}]")
    e(f"    ldr r2, [r0, #{OFF['s']}]")
    e(f"    ldr r11, [r0, #{OFF['n8']}]")
    e(f"    ldr r0, [r0, #{OFF['x']}]")
    e("    ldr r6, 5f")
    e("4:  add r6, pc, r6")
    e("    ldr r7, 8f")
    e("    ldr r8, 7f")
    e("    ldr r10, 6f")
    sv.enter(e)
    e("    ldr r12, 3f")
    e("1:")
    e("    pld [r0, #64]")
    exp_core(False, False, elu=True)
    e("    vldmia r9, {s8-s15}")             # e^x
    e(f"    vldr s2, [r6, #{4 * O_ONE}]")
    e("    vsub.f32 s8, s8, s2")             # e^x - 1
    e("    vsub.f32 s12, s12, s2")
    e("    vstmia r9, {s8-s15}")
    # select, straight to y: q where n = 0 (t is magic exactly), e^x - 1 elsewhere, x for a sign-clear x
    # (+0 and +NaN included), for |x| < 2^-24 and for a NaN
    for k in range(8):
        e(f"    ldr r4, [r0, #{4 * k}]")      # x
        e(f"    ldr r3, [r9, #{4 * k}]")      # e^x - 1
        e(f"    ldr r5, [r2, #{4 * k}]")      # t
        e("    cmp r5, r10")
        e(f"    ldreq r3, [r2, #{64 + 4 * k}]")
        e("    bic r5, r4, #0x80000000")
        e("    cmp r5, #0x33800000")
        e("    movlo r3, r4")
        e("    cmp r5, r8")
        e("    movhi r3, r4")
        e("    tst r4, #0x80000000")
        e("    moveq r3, r4")
        e(f"    str r3, [r1, #{4 * k}]")
    e("    add r0, r0, #32")
    e("    add r1, r1, #32")
    e("    subs r11, r11, #1")
    e("    bne 1b")
    sv.leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r11}")
    e("    bx lr")
    e("    .align 2")
    e(f"3:  .word {f32hex(86.98)}")
    e("5:  .word 9f - (4b + 8)")
    e("6:  .word 0x4b400000")               # bits(magic)
    e("7:  .word 0x7f800000")
    e(f"8:  .word {f32hex(88.37)}")
    e("9:")
    for v in EXP_TABLE:
        e(f"    .word {f32hex(v)}")
    e(f"    .size {name}, .-{name}")
    return name


e("    .pushsection .text\n    .arm\n    .fpu vfp")
names = [gelu_erf(), act("sigmoid"), act("silu"), act("tanh"), elu()]
e("    .popsection")
sv.lint(L)

H = ["// GENERATED by loom.cpp scripts/gen_armv6_act_f32.py (ADR-079 P4.32f f2) -- edit the generator, not this.",
     "// ARMv6 (VFP11) only: activation kernels on the VFP's short vectors, see vec.h / unary-ops.cpp.",
     "#pragma once", "#include <stdint.h>", "",
     "struct ggml_armv6_act_args {"]
for ty, name, what in FIELDS:
    H.append(f"    {ty + ' ' + name + ';':<24} // {OFF[name]:2d}: {what}")
H += ["};", ""]
for n in names:
    H.append(f'extern "C" void {n}(const ggml_armv6_act_args *);')
H += ["", "#ifndef GGML_ARMV6_ACT_DECLS_ONLY", "__asm__("]
H += [f'    "{line}\\n"' for chunk in L for line in chunk.split("\n")]
H += [");", "#endif", ""]
with open(sys.argv[1], "w") as f:
    f.write("\n".join(H))
