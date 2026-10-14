# ADR-077 gate 1 (the cross-model analysis): generates the F32 arm that the int8 tile
# (scripts/gen_armv6_i8_tile.py) is compared with AT EQUAL EFFORT -- asm, K unrolled, on the same shapes
# in bench_xnnpack.cpp's citrinet mode. 4 time steps x 4 output channels over one K chunk of KC = 128:
#   tile4x4_f32_kc_bias   accumulators start from the bias      (the first K chunk)
#   tile4x4_f32_kc_acc    accumulators start from Y's partials  (every later chunk)
#
# It runs the VFP11 in short-vector mode: the caller sets FPSCR LEN = 4, STRIDE = 1 (and RunFast). Then
# `vmla s16, s8, s0` is four MACs, s16-s19 += s8-s11 * s0, because a destination outside bank 0 (s0-s7)
# makes the op a vector and a bank-0 Sm makes that operand a scalar (bench44 `vfpvec 0`: 0.94 GMAC/s,
# against 0.49-0.69 for scalar vmla in RunFast). One vector vmla per channel per k:
#   s8-s11 / s12-s15   X[k][t0..t3], double-buffered (bank 1 holds two LEN-4 vectors)
#   s0-s3  / s4-s7     W[k][n0..n3] as scalars, double-buffered (W packed [N/4][K][4])
#   s16-s19, s20-s23, s24-s27, s28-s31   the accumulators of channels n0..n3, each over t0..t3
# A load issues to the LS pipe while a vector iterates in the FMAC pipe, so the loads for one k sit
# between the vmlas of the other. The bias start is a vector vmov from a bank-0 scalar (a broadcast).
# The caller must not run VFP arithmetic of its own while LEN != 0 (it would be vectorized too), so it
# sets FPSCR once around a GEMM whose loops are integer code. s16-s31 are callee-saved and are pushed.
#
# X is read from a PACKED block, [K][4] per tile (16 B rows), not from channels-first X in place: the
# first version read X[k][t0..t3] at a stride of T floats, a new 4 KB page every k at T = 1101, and ran
# jfk's 1x1 convs in 21.9 s against ggml F32's 12.0 s.
#   r0 = packed X [KC][4], r1 = W quad at this chunk ([KC][4], contiguous), r2 = bias[n0..n3] (bias
#   variant; unused by acc), r3 = Y + n0 * T + t0, [sp] = Y's row stride in bytes.
#   python3 scripts/gen_armv6_f32_tile.py tile_f32.S
import sys

KC = 128
L = []
e = L.append


def tile(name, from_bias):
    e(f"    .global {name}\n    .type {name}, %function\n{name}:")
    e("    push {r4, r5}")
    e("    vpush {d8-d15}")            # s16-s31 are callee-saved (AAPCS)
    e("    ldr r4, [sp, #72]")
    e("    mov r5, r3")
    if from_bias:
        e("    vldmia r2, {s0-s3}")
        e("    vmov.f32 s16, s0")
        e("    vmov.f32 s20, s1")
        e("    vmov.f32 s24, s2")
        e("    vmov.f32 s28, s3")
    else:
        for i, bank in enumerate(["{s16-s19}", "{s20-s23}", "{s24-s27}", "{s28-s31}"]):
            e(f"    vldmia r5, {bank}")
            if i < 3:
                e("    add r5, r5, r4")
        e("    mov r5, r3")
    e("    vldmia r0!, {s8-s11}")
    e("    vldmia r1!, {s0-s3}")
    bufs = [("s8", ["s0", "s1", "s2", "s3"], "{s8-s11}", "{s0-s3}"),
            ("s12", ["s4", "s5", "s6", "s7"], "{s12-s15}", "{s4-s7}")]
    for k in range(KC):
        x, w, _, _ = bufs[k % 2]
        _, _, xn, wn = bufs[(k + 1) % 2]
        more = k + 1 < KC
        e(f"    vmla.f32 s16, {x}, {w[0]}")
        if more:
            e(f"    vldmia r0!, {xn}")
        e(f"    vmla.f32 s20, {x}, {w[1]}")
        if more:
            e(f"    vldmia r1!, {wn}")
        e(f"    vmla.f32 s24, {x}, {w[2]}")
        e(f"    vmla.f32 s28, {x}, {w[3]}")
    for i, bank in enumerate(["{s16-s19}", "{s20-s23}", "{s24-s27}", "{s28-s31}"]):
        e(f"    vstmia r5, {bank}")
        if i < 3:
            e("    add r5, r5, r4")
    e("    vpop {d8-d15}")
    e("    pop {r4, r5}")
    e("    bx lr")
    e(f"    .size {name}, .-{name}")


e("    .text\n    .arm\n    .fpu vfp\n    .align 4")
tile("tile4x4_f32_kc_bias", True)
tile("tile4x4_f32_kc_acc", False)
e('    .section .note.GNU-stack,"",%progbits')
with open(sys.argv[1], "w") as f:
    f.write("\n".join(L) + "\n")
