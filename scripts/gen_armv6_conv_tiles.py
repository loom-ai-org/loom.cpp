# ADR-077 gate 1's cross-model analysis, regime R2: the vocoders' direct convolutions (VITS, piperlite,
# HiFi-GAN-style: k = 3-7, dilation 1-5, 12-64 channels, tens of thousands of samples). Generates both
# arms of scripts/bench49.cpp at the same effort -- asm, the reduction unrolled -- into conv_tiles.S
# plus conv_tiles.h, the table bench49 looks its shapes up in.
#
#   conv_f32_k{k}d{d}(xp, w, b, y, ystride, n_ic, xrow)          F32, 4 output channels x 4 time steps
#     The VFP11 in short-vector mode (the caller sets LEN = 4 and RunFast; see gen_armv6_f32_tile.py for
#     the register banks). Per input channel and tap j: X[ic][t0 + j*d .. +3] as two vldr.64 at an
#     immediate offset (one packed row per ic, xrow bytes apart), W[oc0..3][ic][j] as four scalars
#     (W packed [OC/4][IC][k][4]), four vector vmlas; loads for the next tap sit between them. The
#     accumulators start from the bias. r0 = xp, r1 = w, r2 = b, r3 = y, stack: ystride, n_ic, xrow.
#
#   conv_i8_c{IC}k{k}d{d}(xq, w, out)                               int8, 2 time steps x 4 output channels
#     ADR-077 gate 2's SMLAD tile (gen_armv6_i8_tile.py) with the reduction over (tap, input-channel
#     pair): X int16 channels-last [T_in][IC] (pairs along ic, one word per pair), W int8 held as int16
#     pairs, packed [OC/4][k][4][IC/2] words. Fully unrolled, every offset an immediate. out = int32[8].
import sys

F32_SHAPES = [(3, 1), (5, 1), (7, 1), (7, 3), (3, 3), (7, 5)]          # (k, d)
I8_SHAPES = [(12, 7, 1), (24, 7, 1), (24, 7, 3), (48, 3, 1), (48, 7, 1), (64, 7, 1), (24, 7, 5)]   # (IC, k, d)

L = []
e = L.append
e("    .text\n    .arm\n    .fpu vfp\n    .align 4")


def f32_tile(k, d):
    name = f"conv_f32_k{k}d{d}"
    e(f"    .global {name}\n    .type {name}, %function\n{name}:")
    e("    push {r4-r6}")
    e("    vpush {d8-d15}")
    e("    ldr r4, [sp, #76]")      # ystride (12 + 64 bytes pushed)
    e("    ldr r5, [sp, #80]")      # n_ic
    e("    ldr r6, [sp, #84]")      # xrow bytes
    e("    vldmia r2, {s0-s3}")
    e("    vmov.f32 s16, s0\n    vmov.f32 s20, s1\n    vmov.f32 s24, s2\n    vmov.f32 s28, s3")
    bufs = [("s8", ["s0", "s1", "s2", "s3"], ("d4", "d5"), "{s0-s3}"),
            ("s12", ["s4", "s5", "s6", "s7"], ("d6", "d7"), "{s4-s7}")]
    # One ic per loop iteration, k taps unrolled. Tap j of this ic preloads tap j+1; the last tap
    # preloads tap 0 of the NEXT row (r0 + xrow), which the loop's add then makes current. That load
    # runs one row past the last ic, so the packed block carries a spare row.
    e(f"    vldr d4, [r0]\n    vldr d5, [r0, #8]")
    e("    vldmia r1!, {s0-s3}")
    e("1:")
    for j in range(k):
        x, w, _, _ = bufs[j % 2]
        _, _, (dn0, dn1), wn = bufs[(j + 1) % 2]
        nxt = (j + 1) * d * 4 if j + 1 < k else None
        e(f"    vmla.f32 s16, {x}, {w[0]}")
        if nxt is not None:
            e(f"    vldr {dn0}, [r0, #{nxt}]")
            e(f"    vmla.f32 s20, {x}, {w[1]}")
            e(f"    vldr {dn1}, [r0, #{nxt + 8}]")
        else:
            e("    add r0, r0, r6")
            e(f"    vmla.f32 s20, {x}, {w[1]}")
            e(f"    vldr {dn0}, [r0]")
            e(f"    vldr {dn1}, [r0, #8]")
        e(f"    vldmia r1!, {wn}")
        e(f"    vmla.f32 s24, {x}, {w[2]}")
        e(f"    vmla.f32 s28, {x}, {w[3]}")
    if k % 2:
        # An odd k leaves the next row's tap 0 in the odd buffer; move it to the even one the loop
        # starts from (bank-0 and bank-1 moves: vmov of a bank-1 destination is a VECTOR op at LEN 4,
        # which copies all four lanes -- exactly what is wanted here).
        e("    vmov.f32 s8, s12")
        e("    vmov.f32 s0, s4")      # bank 0 is scalar: copy the four W scalars one by one
        e("    vmov.f32 s1, s5\n    vmov.f32 s2, s6\n    vmov.f32 s3, s7")
    e("    subs r5, r5, #1")
    e("    bne 1b")
    for i, bank in enumerate(["{s16-s19}", "{s20-s23}", "{s24-s27}", "{s28-s31}"]):
        e(f"    vstmia r3, {bank}")
        if i < 3:
            e("    add r3, r3, r4")
    e("    vpop {d8-d15}")
    e("    pop {r4-r6}")
    e("    bx lr")
    e(f"    .size {name}, .-{name}")
    return name


def i8_tile(ic, k, d):
    name = f"conv_i8_c{ic}k{k}d{d}"
    pw = ic // 2                    # words per (t, tap) row of X, and per W row
    c0 = ["r2", "r3", "r4", "r5"]
    c1 = ["r6", "r7", "r8", "r9"]
    e(f"    .global {name}\n    .type {name}, %function\n{name}:")
    e("    push {r2, r4-r11, lr}")
    pairs = [(j, p) for j in range(k) for p in range(pw)]

    def xo(j, p, row):              # byte offset of X[t0 + row + j*d][2p..2p+1]
        return ((j * d + row) * pw + p) * 4

    def wo(j, p, r):                # byte offset of W[oc r][tap j][2p..2p+1]
        return ((j * 4 + r) * pw + p) * 4

    j, p = pairs[0]
    e(f"    ldr r10, [r0, #{xo(j, p, 0)}]")
    e(f"    ldr r11, [r0, #{xo(j, p, 1)}]")
    e(f"    ldr r12, [r1, #{wo(j, p, 0)}]")
    e(f"    ldr lr, [r1, #{wo(j, p, 1)}]")
    for i, (j, p) in enumerate(pairs):
        first, last = i == 0, i == len(pairs) - 1

        def mac(acc, a, w):
            e(f"    smuad {acc}, {a}, {w}" if first else f"    smlad {acc}, {a}, {w}, {acc}")

        mac(c0[0], "r10", "r12"); mac(c1[0], "r11", "r12")
        e(f"    ldr r12, [r1, #{wo(j, p, 2)}]")
        mac(c0[1], "r10", "lr"); mac(c1[1], "r11", "lr")
        e(f"    ldr lr, [r1, #{wo(j, p, 3)}]")
        mac(c0[2], "r10", "r12"); mac(c1[2], "r11", "r12")
        if not last:
            jn, pn = pairs[i + 1]
            e(f"    ldr r12, [r1, #{wo(jn, pn, 0)}]")
        mac(c0[3], "r10", "lr")
        if not last:
            e(f"    ldr r10, [r0, #{xo(jn, pn, 0)}]")
        mac(c1[3], "r11", "lr")
        if not last:
            e(f"    ldr r11, [r0, #{xo(jn, pn, 1)}]")
            e(f"    ldr lr, [r1, #{wo(jn, pn, 1)}]")
    e("    pop {r10}")
    e("    stmia r10, {r2-r9}")
    e("    pop {r4-r11, pc}")
    e(f"    .size {name}, .-{name}")
    for (jj, pp) in pairs:
        for r in range(4):
            assert wo(jj, pp, r) < 4096 and xo(jj, pp, 1) < 4096, (ic, k, d)
    return name


f32_names = {(k, d): f32_tile(k, d) for k, d in F32_SHAPES}
i8_names = {(c, k, d): i8_tile(c, k, d) for c, k, d in I8_SHAPES}
e('    .section .note.GNU-stack,"",%progbits')
with open(sys.argv[1], "w") as f:
    f.write("\n".join(L) + "\n")
with open(sys.argv[2], "w") as f:
    f.write("// generated by scripts/gen_armv6_conv_tiles.py\n#pragma once\n#include <stdint.h>\n")
    f.write("typedef void (*conv_f32_fn)(const float*, const float*, const float*, float*, int, int, int);\n")
    f.write("typedef void (*conv_i8_fn)(const uint32_t*, const uint32_t*, int32_t*);\n")
    for n in f32_names.values():
        f.write(f'extern "C" void {n}(const float*, const float*, const float*, float*, int, int, int);\n')
    for n in i8_names.values():
        f.write(f'extern "C" void {n}(const uint32_t*, const uint32_t*, int32_t*);\n')
    f.write("struct ConvF32Tile { int k, d; conv_f32_fn fn; };\nstatic const ConvF32Tile kConvF32[] = {\n")
    for (k, d), n in f32_names.items():
        f.write(f"    {{{k}, {d}, {n}}},\n")
    f.write("};\nstruct ConvI8Tile { int ic, k, d; conv_i8_fn fn; };\nstatic const ConvI8Tile kConvI8[] = {\n")
    for (c, k, d), n in i8_names.items():
        f.write(f"    {{{c}, {k}, {d}, {n}}},\n")
    f.write("};\n")
