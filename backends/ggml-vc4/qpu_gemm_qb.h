// Block-scaled int8 MUL_MAT for the VideoCore IV QPUs: ggml's Q8_0 and Q4_0, one scale per 32 weights
// of a row -- what qpu_gemm_q8.h cannot take, since it applies one scale per row after the whole sum.
//
//     W = src0 [K, M], row m in blocks b of 32: q[m][32b + i] with scale d[m][b]
//     X = src1 [K, N] F32, packed per call
//     C[n][m] = sum_b d[m][b] * sum_{i<32} q[m][32b + i] * X[n][32b + i]
//
// The inner loop is qpu_gemm_q8.h's: offset-binary bytes u = q + 128, four to a TMU word, each read by
// the multiply that uses it through r4's 8-bit unpack, which yields u/255 (truncated) -- so X is packed
// x 255. What changes is that each block is summed on its own and scaled by its own d:
//
//   * a repacked row is [32 bytes | d as an F32] per block, 36 bytes: the TMU stream lane l reads then
//     brings each block's scale into r4 right after its last word, with no second stream to schedule;
//   * the block's partial sums p_r (one per column n0 + r, regfile A) start at -128 * sum_{k in b} x
//     -- a packed value the uniform stream carries before the block's 32 k -- so p_r ends as
//     sum q x exactly where qpu_gemm_q8.h subtracts 128 * sum x once per row;
//   * then T_r += p_r * d (regfile B), the flush.
//
// Per block: 16 (start the sums) + 8 x 65 (as qpu_gemm_q8.h) + 2 + 17 (flush) + 5 (loop) = 560
// instructions for 32 k, 70 per 4 k against the per-channel kernel's 65 and the F32 kernel's 68;
// the TMU reads 36 bytes per 32 k of a row, 28% of the F32 kernel's 128.
//
// Q4_0 is repacked into the same bytes (u = q4 - 8 + 128): its nibbles cannot be read by the unpack,
// and one kernel serves both types. The rows double (18 -> 36 bytes a block) in the copy the QPUs read.
//
// Output: like every kernel here, a horizontal VPM row per accumulator and one VDW for the tile --
// dst row n holds C[n][m0..]. GEMM_OUT_M writes the transpose instead (dst row m holds C[..][n0..]),
// with the accumulators written VERTICALLY into the VPM (as qpu_gemm.h does), so VPM row l is lane l's
// m: only the uniforms differ, so every int8 kernel has both. That is a 1x1 convolution's layout,
// [OC][T] (P4.46: Citrinet's pointwise convolutions).
#pragma once

#include "qpu_gemm.h"  // GEMM_TMU_AHEAD, Program, gemm_tiles

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vc4 {

constexpr int QB_BLOCK = 32;                      // k per block (ggml's QK8_0 = QK4_0)
constexpr int QB_ROW_BLOCK_BYTES = QB_BLOCK + 4;  // 32 offset-binary bytes, then the block's F32 scale

inline size_t gemm_qb_row_bytes(int K) { return gemm_row_stride(size_t(K / QB_BLOCK) * QB_ROW_BLOCK_BYTES); }
// The TMU runs GEMM_TMU_AHEAD words past the last row's last block.
inline size_t gemm_qb_tail_bytes() { return size_t(GEMM_TMU_AHEAD + 1) * 4; }
inline size_t gemm_qb_weight_bytes(int M, int K) { return size_t(M) * gemm_qb_row_bytes(K) + gemm_qb_tail_bytes(); }
// Per 16 columns and block: the 16 starting sums, then 32 k x 16 columns.
inline size_t gemm_qb_xpack_floats(int N, int K) { return size_t((N + 15) / 16) * (K / QB_BLOCK) * (QB_BLOCK + 1) * 16; }
inline size_t gemm_qb_xpack_bytes(int N, int K) { return gemm_qb_xpack_floats(N, K) * 4; }

// Register map: gemm_q8_kernel's, with the block sums p_r in ra0..ra15 and the totals T_r in rb15..rb30.
//   ra0..ra15 block sums   ra16 X block address   ra18 tiles left   ra19 seq
//   rb0 W row stride   rb1 blocks (loop count)   rb2 VPM write setup   rb3 tile VDW stride setup
//   rb5 constant 4     rb6 next tile params   rb7 tile dst address   rb8 tile VDW setup
//   rb9 valid m lanes - 1   rb11/12/13/14 done marker setups and address   rb15..rb30 totals
//   r1 product   r2 TMU address   r3 scratch / block counter   r4 TMU data   r0 scratch
inline Program gemm_qb_kernel() {
    const int ahead = GEMM_TMU_AHEAD;
    Program p;
    auto unif_to_a = [&](uint32_t ra) { p.mov(ra, 0, MUX_A, R_UNIFORM); };
    auto unif_to_b = [&](uint32_t rb) { p.mov(rb, 1, MUX_A, R_UNIFORM); };
    constexpr uint32_t T0 = 15;  // rb15: total of column n0

    unif_to_a(18);  // tiles
    unif_to_b(0);   // W row stride (bytes)
    unif_to_b(1);   // blocks per row
    unif_to_b(2);   // VPM write setup, row 0 (horizontal) or column 0 (vertical, GEMM_OUT_M)
    unif_to_b(11);  // done: VPM setup
    unif_to_b(13);  // done: VDW stride setup
    unif_to_b(12);  // done: VDW setup
    unif_to_b(14);  // done: address
    unif_to_a(19);  // seq
    p.raw(ldi(5, 1, 4));

    const size_t tile_loop = p.here();
    // Tile params: mvalid-1, W base (row m0), VDW stride setup, VDW setup, dst, X block, next.
    unif_to_b(9);
    p.mov(W_R0, 0, MUX_A, R_UNIFORM);
    unif_to_b(3);
    unif_to_b(8);
    unif_to_b(7);
    unif_to_a(16);
    unif_to_b(6);
    { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
      a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
    { Alu a; a.op_mul = M_MUL24; a.waddr_mul = W_R3; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R3; a.mul_b = MUX_B;
      a.raddr_b = 0; p.alu(a); }
    { Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R0; a.add_b = MUX_R3; p.alu(a); }
    p.mov(W_UNIFORMS_ADDR, 0, MUX_A, 16);
    for (uint32_t r = 0; r < 16; r++) p.raw(ldi(T0 + r, 1, 0));
    auto tmu_req = [&]() {
        Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R2;
        a.add_b = MUX_B; a.raddr_b = 5;
        a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R2; a.mul_b = MUX_R2;
        p.alu(a);
    };
    for (int i = 0; i < ahead; i++) tmu_req();
    p.nop(SIG_LDTMU0);
    p.mov(W_R3, 0, MUX_B, R_NOP, 1);

    const size_t block_loop = p.here();
    // Start the block's sums: r1 <- the uniform (-128 * the block's sum of x for column r), then
    // p_(r-1) <- r1 one instruction later; p15 is set by the first fmac below. v8min of a value with
    // itself is a move.
    for (uint32_t r = 0; r < 16; r++) {
        Alu a;
        a.op_mul = M_V8MIN; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_B; a.mul_b = MUX_B;
        a.raddr_b = R_UNIFORM;
        if (r > 0) { a.op_add = A_OR; a.waddr_add = r - 1; a.cond_add = C_ALWAYS; a.add_a = MUX_R1; a.add_b = MUX_R1; }
        p.alu(a);
    }
    for (int w = 0; w < QB_BLOCK / 4; w++) {
        tmu_req();
        for (int b = 0; b < 4; b++)
            for (uint32_t r = 0; r < 16; r++) {
                Alu a; a.sig = b == 3 && r == 15 ? SIG_LDTMU0 : SIG_NONE;
                a.waddr_add = (r + 15) & 15; a.cond_add = C_ALWAYS;
                if (w == 0 && b == 0 && r == 0) {  // p15 <- column 15's starting sum
                    a.op_add = A_OR; a.add_a = MUX_R1; a.add_b = MUX_R1;
                } else {
                    a.op_add = A_FADD; a.add_a = MUX_A; a.raddr_a = (r + 15) & 15; a.add_b = MUX_R1;
                }
                a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R4; a.mul_b = MUX_B;
                a.raddr_b = R_UNIFORM; a.pm = 1; a.unpack = 4 + b;
                p.alu(a);
            }
    }
    // r4 is the block's scale now (the last fmac popped it). The last product, then the request that
    // keeps `ahead` words in flight, then T_r += p_r * d with the write sides swapped (ws = 1: the add
    // writes regfile B), popping the next block's first word on the way out.
    { Alu a; a.op_add = A_FADD; a.waddr_add = 15; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = 15;
      a.add_b = MUX_R1; p.alu(a); }
    tmu_req();
    for (uint32_t i = 0; i <= 16; i++) {
        Alu a; a.ws = 1; a.sig = i == 16 ? SIG_LDTMU0 : SIG_NONE;
        if (i < 16) {
            a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_A; a.raddr_a = i; a.mul_b = MUX_R4;
        }
        if (i > 0) {
            a.op_add = A_FADD; a.waddr_add = T0 + i - 1; a.cond_add = C_ALWAYS; a.add_a = MUX_B; a.raddr_b = T0 + i - 1;
            a.add_b = MUX_R1;
        }
        p.alu(a);
    }
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = W_R3; a.cond_add = C_ALWAYS;
      a.add_a = MUX_R3; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, block_loop);
    p.nop(); p.nop(); p.nop();
    for (int i = 0; i < ahead - 1; i++) p.nop(SIG_LDTMU0);

    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 2);
    for (uint32_t r = 0; r < 16; r++) p.mov(W_VPM, 0, MUX_B, R_NOP, T0 + r);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 3);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 8);
    p.mov(W_VPM_ADDR, 1, MUX_B, R_NOP, 7);
    p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    p.mutex_release();

    p.mov(W_UNIFORMS_ADDR, 0, MUX_B, R_NOP, 6);
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = 18; a.cond_add = C_ALWAYS;
      a.add_a = MUX_A; a.raddr_a = 18; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, tile_loop);
    p.nop(); p.nop(); p.nop();

    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 11);
    p.mov(W_VPM, 0, MUX_A, 19);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 13);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 12);
    p.mov(W_VPM_ADDR, 1, MUX_B, R_NOP, 14);
    p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    p.mutex_release();
    p.host_interrupt();
    p.thread_end();
    return p;
}

// ggml rows -> the repacked weight. `Block` is ggml's block_q8_0 {d; int8 qs[32]} or block_q4_0
// {d; uint8 qs[16]} (element i is qs[i] & 15, element i + 16 is qs[i] >> 4, both less 8); `d_to_float`
// converts a block's d.
template <class Block, class D2F>
void gemm_qb_repack(const Block * rows, int M, int K, D2F d_to_float, uint8_t * out) {
    const int nb = K / QB_BLOCK;
    for (int m = 0; m < M; m++) {
        const Block * b = rows + size_t(m) * nb;
        uint8_t * o = out + size_t(m) * gemm_qb_row_bytes(K);
        for (int j = 0; j < nb; j++, o += QB_ROW_BLOCK_BYTES) {
            if constexpr (sizeof(b[j].qs) == 32) {
                for (int i = 0; i < 32; i++) o[i] = uint8_t(int(b[j].qs[i]) + 128);
            } else {
                for (int i = 0; i < 16; i++) {
                    o[i] = uint8_t((b[j].qs[i] & 15) + 120);
                    o[i + 16] = uint8_t((b[j].qs[i] >> 4) + 120);
                }
            }
            const float d = d_to_float(b[j].d);
            memcpy(o + QB_BLOCK, &d, 4);
        }
    }
    std::fill(out + size_t(M) * gemm_qb_row_bytes(K), out + gemm_qb_weight_bytes(M, K), uint8_t(0));
}

// X -> per 16 columns and block, the 16 starting sums -128 * sum_{k in b} x (summed in double, as
// gemm_q8_pack_x does: the block's sum cancels most of it) then 255 * x k-major. X[n][k] is
// x[n * ns + k * ks]: ks = 1 for a MUL_MAT's activations, ns = 1 for a 1x1 convolution's [IC][T] input.
inline void gemm_qb_pack_x(const float * x, size_t ns, size_t ks, int N, int K, float * out) {
    const int nb = K / QB_BLOCK;
    for (int c = 0; c < (N + 15) / 16; c++) {
        const int rows = std::min(16, N - c * 16);
        const float * xc = x + size_t(c) * 16 * ns;
        for (int j = 0; j < nb; j++) {
            float * sums = out;
            float * o = out + 16;
            double s[16] = {};
            for (int k = j * QB_BLOCK; k < (j + 1) * QB_BLOCK; k++, o += 16) {
                int r = 0;
                for (; r < rows; r++) {
                    const float v = xc[size_t(r) * ns + size_t(k) * ks];
                    o[r] = 255.f * v;
                    s[r] += v;
                }
                for (; r < 16; r++) o[r] = 0.f;
            }
            for (int r = 0; r < 16; r++) sums[r] = float(-128.0 * s[r]);
            out = o;
        }
    }
}

struct GemmQbProblem {
    int M, N, K;
    uint32_t w;             // repacked row 0 (bus); rows gemm_qb_row_bytes(K) apart
    uint32_t c, c_stride;   // dst row 0, bytes between dst rows (rows of n, or of m with GEMM_OUT_M)
    uint32_t xpack;         // this slice's packed X (gemm_qb_xpack_bytes(N, K) bytes)
};

inline size_t gemm_qb_uniform_words(const std::vector<GemmQbProblem> & probs, int nq) {
    size_t tiles = 0;
    for (auto & p : probs) tiles += size_t((p.M + 15) / 16) * ((p.N + 15) / 16);
    return size_t(nq) * (9 + 16) + tiles * 7;
}

// The per-QPU uniform streams, tiles in gemm_tiles' order, round-robin over min(nq_max, tiles) QPUs.
template <class CpuMap>
int gemm_qb_uniforms(CpuMap cpu, uint32_t unif, const std::vector<GemmQbProblem> & probs, int nq_max, uint32_t done,
                     uint32_t seq, std::vector<uint32_t> & starts, GemmOut out = GEMM_OUT_N, int panel = 0) {
    std::vector<GemmTile> tiles;
    for (size_t i = 0; i < probs.size(); i++) gemm_tiles(probs[i].M, probs[i].N, panel, int(i), tiles);
    const int nq = std::min<int>(nq_max, int(tiles.size()));
    starts.clear();
    if (nq == 0) return 0;
    const int K = probs[0].K;
    const size_t xblock = size_t(K / QB_BLOCK) * (QB_BLOCK + 1) * 16 * 4;
    const uint32_t done_vdw = 0x80000000u | 1u << 23 | 16u << 16 | 1u << 14;
    uint32_t at = unif;
    for (int q = 0; q < nq; q++) {
        starts.push_back(at);
        uint32_t * u = cpu(at);
        int i = 0, mine = 0;
        for (size_t t = q; t < tiles.size(); t += nq) mine++;
        u[i++] = uint32_t(mine);
        u[i++] = uint32_t(gemm_qb_row_bytes(K));
        u[i++] = uint32_t(K / QB_BLOCK);
        u[i++] = gemm_vpm_setup(out);
        u[i++] = gemm_vpm_setup(GEMM_OUT_N);
        u[i++] = 0xc0000000u;
        u[i++] = done_vdw;
        u[i++] = done + q * 64;
        u[i++] = seq;
        for (size_t t = q; t < tiles.size(); t += nq) {
            const GemmTile & T = tiles[t];
            const GemmQbProblem & P = probs[T.p];
            const int mvalid = std::min(16, P.M - T.m0), nvalid = std::min(16, P.N - T.n0);
            uint32_t stride_setup, vdw_setup, addr;
            gemm_tile_vdw(out, P.c, P.c_stride, T.m0, T.n0, mvalid, nvalid, stride_setup, vdw_setup, addr);
            u[i++] = uint32_t(mvalid - 1);
            u[i++] = P.w + uint32_t(T.m0) * uint32_t(gemm_qb_row_bytes(K));
            u[i++] = stride_setup;
            u[i++] = vdw_setup;
            u[i++] = addr;
            u[i++] = P.xpack + uint32_t(size_t(T.n0 / 16) * xblock);
            u[i] = at + uint32_t(i + 1) * 4;
            i++;
        }
        at = (at + uint32_t(i) * 4 + 63) & ~63u;
    }
    return nq;
}

}  // namespace vc4
