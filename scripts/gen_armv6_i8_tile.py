# ADR-077 gate 2: generates tile2x4_k256, the ARMv6 int8 GEMM tile that bench_xnnpack.cpp's citrinet
# mode times. 2 time steps x 4 output channels, int16 operand pairs, SMLAD, K = 256 (128 pairs) fully
# unrolled with immediate offsets.
#   r0 = X: two rows of 128 words (row 1 at +512 B); r1 = W: four rows (row j at +512*j B);
#   r2 = int32 out[8] (t0's four channels, then t1's).
# Registers: accumulators r2-r9 (r2 is saved and popped as the output pointer), a0/a1 = r10/r11,
# wA/wB = r12/lr. 14 instructions per K pair: 8 SMLADs and 6 loads. Every load is consumed at least
# 3 instructions later (the ARM1176's load-use latency), and the first pair uses SMUAD so nothing is
# zeroed. On L1-resident operands it runs at 1.95 GOP/s on a Pi Zero W. Written as asm because the
# C version of the same tile (gcc 12, -O3) took 9.2 s on jfk's 1x1 convs against this one's 6.8 s.
#   python3 scripts/gen_armv6_i8_tile.py tile.S
import sys

KW = 128
c0 = ["r2", "r3", "r4", "r5"]
c1 = ["r6", "r7", "r8", "r9"]
L = []
e = L.append
e("    .text\n    .arm\n    .align 4\n    .global tile2x4_k256\n    .type tile2x4_k256, %function\ntile2x4_k256:")
e("    push {r2, r4-r11, lr}")
e("    ldr r10, [r0]")
e("    ldr r11, [r0, #512]")
e("    ldr r12, [r1]")
e("    ldr lr, [r1, #512]")
for k in range(KW):
    first = k == 0
    last = k == KW - 1

    def mac(acc, a, w):
        e(f"    smuad {acc}, {a}, {w}" if first else f"    smlad {acc}, {a}, {w}, {acc}")

    o = 4 * k
    mac(c0[0], "r10", "r12"); mac(c1[0], "r11", "r12")
    e(f"    ldr r12, [r1, #{1024 + o}]")
    mac(c0[1], "r10", "lr"); mac(c1[1], "r11", "lr")
    e(f"    ldr lr, [r1, #{1536 + o}]")
    mac(c0[2], "r10", "r12"); mac(c1[2], "r11", "r12")
    if not last:
        e(f"    ldr r12, [r1, #{o + 4}]")
    mac(c0[3], "r10", "lr")
    if not last:
        e(f"    ldr r10, [r0, #{o + 4}]")
    mac(c1[3], "r11", "lr")
    if not last:
        e(f"    ldr r11, [r0, #{512 + o + 4}]")
        e(f"    ldr lr, [r1, #{512 + o + 4}]")
e("    pop {r10}")
e("    stmia r10, {r2-r9}")
e("    pop {r4-r11, pc}")
e("    .size tile2x4_k256, .-tile2x4_k256")
e('    .section .note.GNU-stack,"",%progbits')
with open(sys.argv[1], "w") as f:
    f.write("\n".join(L) + "\n")
