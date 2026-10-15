# ADR-079 P4.32e: the ARMv6 depthwise-convolution kernels, generated into
# ggml/src/ggml-cpu/arch/arm/armv6-dw-f32.inc (a C++ include holding the declarations and the asm as one
# top-level asm block, as armv6-conv-f32.inc is).
#
#   python3 scripts/gen_armv6_dw_f32.py <ggml>/src/ggml-cpu/arch/arm/armv6-dw-f32.inc
#
# One channel, stride 1, dilation 1, K >= 2: y[t] = sum_k w[k] * x[t + k] over a zero-padded row, 8
# outputs a block, `nblk` blocks. A depthwise conv has no input channels to reuse a weight across, so the
# 4x4 tile of gen_armv6_conv_f32.py does not apply: what is reused is the input window, across taps.
#
# The VFP11 in short-vector mode (LEN = 4, STRIDE = 1, the thread's RunFast; scripts/armv6_shortvec.py):
#   s8-s15    a RING of 8 input values: at tap k it holds x[t + k .. t + k + 7], x[t + i] in slot i % 8.
#             A vector operand starting at slot k % 8 wraps around the bank (s13 at LEN 4 is s13, s14,
#             s15, s8), which is exactly x[t + k .. t + k + 3]: the window is renamed by the unroll,
#             never moved. After tap k's first vmla, slot k % 8 is dead and takes x[t + k + 8].
#   s0-s7     the taps' weights as scalars, w[k] in slot k % 8, loaded four taps ahead
#   s16-s23   y[t .. t + 7] over the EVEN taps (two vectors of four)
#   s24-s31   y[t .. t + 7] over the ODD taps
# Per tap: two vector vmla (8 MACs) and two scalar loads. The taps go in groups of 8 (the ring's period),
# then K % 8 more.
#
# Why two partial sums (board, Pi Zero W, RunFast): with one set of two accumulator vectors a tap cost 14
# cycles with or without its loads -- a vector vmla waits for the whole previous one on its destination,
# ~14 cycles, and two chains cover 8. Four chains run a vmla per ~5 cycles. The ring leaves no bank for
# more outputs, so the extra chains are the odd taps: y = (w0 x0 + w2 x2 + ...) + (w1 x1 + w3 x3 + ...),
# each partial sum in k order, started by a vmul (taps 0 and 1) rather than from +0.0. That is the
# kernel's contract to the bit (tests/ci/test_conv_1d_dw_sweep.cpp), and it differs from ggml's sweep,
# which sums all taps in order, at rounding level.
#
# `pf` walks one cache line per block (PLD): the caller points it at the row it reads next, which a row
# of 8 * nblk outputs covers. Rows come from DRAM cold (scripts/bench52.cpp, the jfk schedule): 0.37
# GMAC/s without it, 0.49 with it. Measured out: two lines a block (0.45) and a PLD on the output lines
# ahead (0.39; the stores stream without allocating, so it only added reads). Loading the next block's
# window while the partial sums finish bought 4-5%.
#
# Reads past the row: x up to 8 * nblk + K + 15 floats from the block start, w up to K + 4 floats. The
# caller passes padded copies. Functions: ggml_armv6_dw_f32[K >= 8][K % 8].
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armv6_shortvec as sv  # noqa: E402

FIELDS = [("const float *", "x", "the padded row at the first output: x[t + k] for output t, tap k"),
          ("const float *", "w", "the channel's K weights, readable to K + 4"),
          ("float *", "y", "8 * nblk outputs"),
          ("int32_t", "nblk", "blocks of 8 outputs, >= 1"),
          ("int32_t", "ngrp", "K / 8 (the function is chosen by K >= 8 and K % 8)"),
          ("const void *", "pf", "PLD one line here per block, advancing (the next row's source)")]
OFF = {name: 4 * i for i, (_, name, _) in enumerate(FIELDS)}

L = []
e = L.append


def tap(j, k, load_x, load_w):
    """Tap k of the kernel (slot k % 8), j = its offset from the group base: r5 = x + 8 g, r6 = w + 8 g."""
    s = k % 8
    acc = ("s16", "s20") if k % 2 == 0 else ("s24", "s28")
    op = "vmul.f32" if k < 2 else "vmla.f32"
    e(f"    {op} {acc[0]}, s{8 + s}, s{s}    {sv.WRAP_MARK}")
    if load_x:
        e(f"    vldr s{8 + s}, [r5, #{(j + 8) * 4}]")
    e(f"    {op} {acc[1]}, s{8 + (s + 4) % 8}, s{s}    {sv.WRAP_MARK}")
    if load_w:
        e(f"    vldr s{(s + 4) % 8}, [r6, #{(j + 4) * 4}]")


def kernel(big, r):
    name = f"ggml_armv6_dw_f32_{'g' if big else 'r'}{r}"
    if not big and r < 2:
        return None                  # K = 0 or 1: not dispatched
    e(f"    .global {name}\n    .type {name}, %function\n    .align 4\n{name}:")
    e("    push {r4-r9}")
    e("    vpush {d8-d15}")         # s16-s31 are callee-saved (AAPCS)
    e(f"    ldr r1, [r0, #{OFF['w']}]")
    e(f"    ldr r2, [r0, #{OFF['y']}]")
    e(f"    ldr r3, [r0, #{OFF['nblk']}]")
    e(f"    ldr r8, [r0, #{OFF['ngrp']}]")
    e(f"    ldr r9, [r0, #{OFF['pf']}]")
    e(f"    ldr r0, [r0, #{OFF['x']}]")
    sv.enter(e)
    e("    vldmia r0, {s8-s15}")
    e("    vldmia r1, {s0-s3}")
    e("3:")
    e("    pld [r9]")
    e("    add r9, r9, #32")
    e("    mov r5, r0")
    e("    mov r6, r1")
    if big:
        # The first group is peeled: its taps 0 and 1 start the partial sums.
        for j in range(8):
            tap(j, j, True, True)
        e("    add r5, r5, #32")
        e("    add r6, r6, #32")
        e("    subs r4, r8, #1")
        e("    beq 2f")
        e("1:")
        for j in range(8):
            tap(j, 8 + j, True, True)
        e("    add r5, r5, #32")
        e("    add r6, r6, #32")
        e("    subs r4, r4, #1")
        e("    bne 1b")
        e("2:")
    # The remainder loads only what a later tap of it reads: slot j's next x is first read by tap j + 1's
    # second vmla, its next w by tap j + 4.
    for j in range(r):
        tap(j, (8 if big else 0) + j, j + 1 < r, j + 4 < r)
    # The next block's window and first weights load while the partial sums finish (the ring and the
    # weight slots are dead after the last tap). After the last block that reads 8 floats past it, inside
    # the bound below.
    e("    add r0, r0, #32")
    e("    vldmia r0, {s8-s15}")
    e("    vldmia r1, {s0-s3}")
    e("    vadd.f32 s16, s16, s24")
    e("    vadd.f32 s20, s20, s28")
    e("    vstmia r2!, {s16-s23}")
    e("    subs r3, r3, #1")
    e("    bne 3b")
    sv.leave(e)
    e("    vpop {d8-d15}")
    e("    pop {r4-r9}")
    e("    bx lr")
    e(f"    .size {name}, .-{name}")
    return name


e("    .pushsection .text\n    .arm\n    .fpu vfp")
names = [[kernel(big, r) for r in range(8)] for big in (False, True)]
e("    .popsection")
sv.lint(L)

H = ["// GENERATED by loom.cpp scripts/gen_armv6_dw_f32.py (ADR-079 P4.32e) -- edit the generator, not this.",
     "// ARMv6 (VFP11) only: the depthwise-convolution kernels on the VFP's short vectors, see ops.cpp.",
     "#pragma once", "#include <stdint.h>", "",
     "struct ggml_armv6_dw_args {"]
for ty, name, what in FIELDS:
    H.append(f"    {ty + ' ' + name + ';':<24} // {OFF[name]:2d}: {what}")
H += ["};", ""]
for n in [n for row in names for n in row if n]:
    H.append(f'extern "C" void {n}(const ggml_armv6_dw_args *);')
H += ["", "typedef void (*ggml_armv6_dw_fn)(const ggml_armv6_dw_args *);",
      "// [K >= 8][K % 8]; K < 2 has no kernel.",
      "static const ggml_armv6_dw_fn ggml_armv6_dw_f32[2][8] = {"]
for row in names:
    H.append("    { " + ", ".join(n or "nullptr" for n in row) + " },")
# GGML_ARMV6_DW_DECLS_ONLY: the declarations alone, for a bench linked against a libggml-cpu that
# already defines the kernels (scripts/bench52.cpp -DBENCH52_GGML).
H += ["};", "", "#ifndef GGML_ARMV6_DW_DECLS_ONLY", "__asm__("]
H += [f'    "{line}\\n"' for chunk in L for line in chunk.split("\n")]
H += [");", "#endif", ""]
with open(sys.argv[1], "w") as f:
    f.write("\n".join(H))
