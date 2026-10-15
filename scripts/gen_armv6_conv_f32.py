# ADR-079 P4.32b: the ARMv6 direct-convolution kernels of ggml-0032, generated into
# ggml/src/ggml-cpu/arch/arm/armv6-conv-f32.inc (a C++ include holding the declarations and the asm as one
# top-level asm block, so the patch needs no build-system change).
#
#   python3 scripts/gen_armv6_conv_f32.py <ggml>/src/ggml-cpu/arch/arm/armv6-conv-f32.inc
#
# One function per kernel width k: ggml_armv6_conv_f32_k{k}(const ggml_armv6_conv_args *). It runs `nt`
# consecutive tiles over one chunk of input channels (b = NULL continues a chunk sum already in y), of 4 output channels x 4 time steps over a packed input window, the bench49 tile
# (gen_armv6_conv_tiles.py, 1.10-1.11 GOP/s on the vocoders' shapes) with two changes:
#   * the dilation is an argument (d4 = dilation * 4 bytes), not an immediate. Tap j+1's address is kept
#     in r7 and advanced by one `add` per tap, which issues while a vector vmla iterates;
#   * the tile loop is inside the function, inside one short-vector region (scripts/armv6_shortvec.py):
#     LEN = 4 is set once per call, and the region is integer loops and VFP code only (lint checks it).
#
# The VFP11 in short-vector mode (LEN = 4, STRIDE = 1, the thread's RunFast): a destination outside bank 0
# makes an op a vector of four, a bank-0 Sm makes that operand a scalar.
#   s8-s11 / s12-s15   X[ic][t0 + j*d .. +3], double-buffered across taps
#   s0-s3  / s4-s7     W[oc0..3][ic][j] as scalars, double-buffered (W packed [OC/4][IC][k][4])
#   s16-s19 ... s28-s31  the accumulators of oc0..oc3 over t0..t3, started from the bias by broadcasting
#                        a bank-0 scalar (vmov.f32 s16, s0 at LEN 4 writes s16-s19)
# Per ic, k taps unrolled; tap j preloads tap j+1, and the last tap preloads tap 0 of the NEXT window row.
# That load runs one row past the last ic, and the W preload four floats past the quad, so the caller's
# window carries a spare row and the packed weights four floats of padding. Every lane accumulates
# bias + sum over ic (outer) and j (inner) in order: a scalar loop in that order gives the same bits.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armv6_shortvec as sv  # noqa: E402

KS = list(range(1, 12))       # 1x1 convs included: the same tile with one tap

# Field offsets of ggml_armv6_conv_args, emitted into the header below and used by the asm.
FIELDS = [("const float *", "x", "window + 4 * tile: row ic at x + ic * xrow bytes"),
          ("const float *", "w", "this quad's packed weights [IC][k][4]"),
          ("const float *", "b", "bias of the quad's four channels; NULL: start from the partial sums in y"),
          ("float *", "y", "output of channel oc0 at the first tile; oc0 + r at y + r * ystride bytes"),
          ("int32_t", "ystride", "bytes between output channels"),
          ("int32_t", "n_ic", "input channels, >= 1"),
          ("int32_t", "xrow", "bytes between window rows"),
          ("int32_t", "nt", "tiles, >= 1: tile i reads x + 4 i and writes y + 4 i"),
          ("int32_t", "d4", "dilation * 4 bytes")]
OFF = {name: 4 * i for i, (_, name, _) in enumerate(FIELDS)}

L = []
e = L.append


def kernel(k):
    name = f"ggml_armv6_conv_f32_k{k}"
    e(f"    .global {name}\n    .type {name}, %function\n    .align 4\n{name}:")
    e("    push {r4-r11}")
    e("    vpush {d8-d15}")         # s16-s31 are callee-saved (AAPCS)
    e("    mov r11, r0")
    e(f"    ldr r8, [r11, #{OFF['nt']}]")
    e(f"    ldr r9, [r11, #{OFF['x']}]")
    e(f"    ldr r10, [r11, #{OFF['y']}]")
    e(f"    ldr r2, [r11, #{OFF['d4']}]")
    e(f"    ldr r6, [r11, #{OFF['xrow']}]")
    e(f"    ldr r4, [r11, #{OFF['ystride']}]")
    sv.enter(e)
    e("3:")
    e("    mov r0, r9")
    e(f"    ldr r1, [r11, #{OFF['w']}]")
    e(f"    ldr r5, [r11, #{OFF['n_ic']}]")
    e(f"    ldr r3, [r11, #{OFF['b']}]")
    e("    cmp r3, #0")
    e("    beq 4f")
    e("    vldmia r3, {s0-s3}")
    e("    vmov.f32 s16, s0\n    vmov.f32 s20, s1\n    vmov.f32 s24, s2\n    vmov.f32 s28, s3")
    e("    b 5f")
    # A later input-channel chunk: the accumulators continue from the partial sums the earlier ones stored.
    e("4:")
    e("    mov r3, r10")
    for i, bank in enumerate(["{s16-s19}", "{s20-s23}", "{s24-s27}", "{s28-s31}"]):
        e(f"    vldmia r3, {bank}")
        if i < 3:
            e("    add r3, r3, r4")
    e("5:")
    e("    vldr d4, [r0]\n    vldr d5, [r0, #8]")
    e("    add r7, r0, r2")
    e("    vldmia r1!, {s0-s3}")
    bufs = [("s8", ["s0", "s1", "s2", "s3"], ("d4", "d5"), "{s0-s3}"),
            ("s12", ["s4", "s5", "s6", "s7"], ("d6", "d7"), "{s4-s7}")]

    def one_ic(first):
        """The k taps of one input channel, starting from buffer `first`. Tap j preloads tap j+1 into the
        other buffer; the last tap preloads tap 0 of the next window row."""
        for j in range(k):
            x, w, _, _ = bufs[(first + j) % 2]
            _, _, (dn0, dn1), wn = bufs[(first + j + 1) % 2]
            e(f"    vmla.f32 s16, {x}, {w[0]}")
            if j + 1 < k:
                e(f"    vldr {dn0}, [r7]")
                e(f"    vmla.f32 s20, {x}, {w[1]}")
                e(f"    vldr {dn1}, [r7, #8]")
                e("    add r7, r7, r2")
            else:
                e("    add r0, r0, r6")
                e(f"    vmla.f32 s20, {x}, {w[1]}")
                e(f"    vldr {dn0}, [r0]")
                e(f"    vldr {dn1}, [r0, #8]")
                e("    add r7, r0, r2")
            e(f"    vldmia r1!, {wn}")
            e(f"    vmla.f32 s24, {x}, {w[2]}")
            e(f"    vmla.f32 s28, {x}, {w[3]}")

    if k % 2 == 0:
        # An even k ends each channel in the buffer the next one starts from.
        e("1:")
        one_ic(0)
        e("    subs r5, r5, #1")
        e("    bne 1b")
    else:
        # An odd k ends a channel in the OTHER buffer, so channels go in pairs (the second starting from
        # buffer 1), plus one alone at the end. The first version copied the buffers back after every
        # channel instead -- a vector vmov and four scalar ones, as many FMAC cycles as the four vmlas at
        # k = 1: Citrinet's 1x1 convolutions ran 0.53 GOP/s that way. Same operations, same order.
        e("    subs r5, r5, #2")
        e("    blt 6f")
        e("1:")
        one_ic(0)
        one_ic(1)
        e("    subs r5, r5, #2")
        e("    bge 1b")
        e("6:")
        e("    adds r5, r5, #2")
        e("    beq 7f")
        one_ic(0)
        e("7:")
    e("    mov r3, r10")
    for i, bank in enumerate(["{s16-s19}", "{s20-s23}", "{s24-s27}", "{s28-s31}"]):
        e(f"    vstmia r3, {bank}")
        if i < 3:
            e("    add r3, r3, r4")
    e("    add r9, r9, #16")
    e("    add r10, r10, #16")
    e("    subs r8, r8, #1")
    e("    bne 3b")
    sv.leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r11}")
    e("    bx lr")
    e(f"    .size {name}, .-{name}")
    return name


e("    .pushsection .text\n    .arm\n    .fpu vfp")
names = [kernel(k) for k in KS]
e("    .popsection")
sv.lint(L)

H = ["// GENERATED by loom.cpp scripts/gen_armv6_conv_f32.py (ADR-079 P4.32b) -- edit the generator, not this.",
     "// ARMv6 (VFP11) only: the direct-convolution tiles on the VFP's short vectors, see ops.cpp.",
     "#pragma once", "#include <stdint.h>", "",
     "struct ggml_armv6_conv_args {"]
for ty, name, what in FIELDS:
    H.append(f"    {ty + ' ' + name + ';':<24} // {OFF[name]:2d}: {what}")
H += ["};", ""]
for n in names:
    H.append(f'extern "C" void {n}(const ggml_armv6_conv_args *);')
H += ["", f"#define GGML_ARMV6_CONV_KMAX {max(KS)}",
      "typedef void (*ggml_armv6_conv_fn)(const ggml_armv6_conv_args *);",
      "static const ggml_armv6_conv_fn ggml_armv6_conv_f32[GGML_ARMV6_CONV_KMAX + 1] = {",
      "    nullptr, " + ", ".join(names) + ",", "};", "", "__asm__("]
H += [f'    "{line}\\n"' for chunk in L for line in chunk.split("\n")]
H += [");", ""]
with open(sys.argv[1], "w") as f:
    f.write("\n".join(H))
