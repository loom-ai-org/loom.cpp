// F32 MUL_MAT for the VideoCore IV QPUs with the WEIGHTS read in place -- the kernel ggml-vc4 runs.
//
//     W = src0 [K, M] (row m: K floats, row stride w_stride bytes)   -- weights, read by the TMU
//     X = src1 [K, N]                                                 -- activations, packed per call
//     C = dst  [M, N], C[n][m] = sum_k W[m][k] * X[n][k]  (row stride c_stride bytes)
//
// It is qpu_gemm.h's kernel with the operands swapped, and the swap is what lets a backend keep
// weights where the model put them. Lane l <-> m0 + l gathers W[m0+l][k] through TMU0 (a 16-row
// gather, the access pattern qpu_gemm.h proved on hardware); accumulator ra_r <-> n0 + r takes
// X[n0+r][k] from the uniform stream, so only the activations are packed (k-major in 16-row blocks,
// zero-filled past N and past K), and they change every call anyway. Each accumulator is then one
// contiguous row of dst: a HORIZONTAL VPM write per accumulator and one VDW of nvalid rows x mvalid
// words writes the tile straight into dst, so a partial tile never touches its neighbours and there
// is no scratch C to compact.
//
// Per k: 16 x (fadd ra_(r-1) += r1 ; fmul r1 = r4 * uniform) with the ldtmu0 pop of column k+1 on
// the last one, after 1 x (iadd r2 += 4 ; tmu0_s = r2) at the top -- never an ldtmu and a TMU
// register write in one instruction (see qpu_gemm.h: on hardware that returned stale columns).
//
// Over-read: the TMU reads up to GEMM_U + GEMM_TMU_AHEAD words past a row's K; within W that is the
// next row (finite, multiplied by the packed zeros), past the last row the caller provides
// gemm_wt_tail_bytes() of readable memory.
#pragma once

#include "qpu_gemm.h"  // GEMM_U, GEMM_TMU_AHEAD

namespace vc4 {

inline size_t gemm_wt_tail_bytes() { return (GEMM_U + GEMM_TMU_AHEAD) * 4; }

// Register map.
//   ra0..ra15 accumulators   ra16 X block address   ra18 tiles left   ra19 seq
//   rb0 W row stride   rb1 k-loop count   rb2 VPM horizontal setup   rb3 tile VDW stride setup
//   rb5 constant 4     rb6 next tile params   rb7 tile dst address   rb8 tile VDW setup
//   rb9 valid m lanes - 1   rb11/12/13/14 done marker setups and address
//   r1 product   r2 TMU address   r3 scratch / k counter   r4 TMU data   r0 scratch
inline Program gemm_f32_wt_kernel() {
    const int ahead = GEMM_TMU_AHEAD;
    Program p;
    auto unif_to_a = [&](uint32_t ra) { p.mov(ra, 0, MUX_A, R_UNIFORM); };
    auto unif_to_b = [&](uint32_t rb) { p.mov(rb, 1, MUX_A, R_UNIFORM); };

    unif_to_a(18);  // tiles
    unif_to_b(0);   // W row stride
    unif_to_b(1);   // k-loop count = Kp / U
    unif_to_b(2);   // VPM horizontal write setup, row 0
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
    for (uint32_t r = 0; r < 16; r++) p.raw(ldi(r, 0, 0));
    p.raw(ldi(W_R1, 0, 0));
    auto tmu_req = [&]() {
        Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R2;
        a.add_b = MUX_B; a.raddr_b = 5;
        a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R2; a.mul_b = MUX_R2;
        p.alu(a);
    };
    for (int i = 0; i < ahead; i++) tmu_req();
    p.nop(SIG_LDTMU0);
    p.mov(W_R3, 0, MUX_B, R_NOP, 1);

    const size_t k_loop = p.here();
    for (int u = 0; u < GEMM_U; u++) {
        tmu_req();
        for (uint32_t r = 0; r < 16; r++) {
            Alu a; a.sig = r == 15 ? SIG_LDTMU0 : SIG_NONE;
            a.op_add = A_FADD; a.waddr_add = (r + 15) & 15; a.cond_add = C_ALWAYS;
            a.add_a = MUX_A; a.raddr_a = (r + 15) & 15; a.add_b = MUX_R1;
            a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R4; a.mul_b = MUX_B;
            a.raddr_b = R_UNIFORM;
            p.alu(a);
        }
    }
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = W_R3; a.cond_add = C_ALWAYS;
      a.add_a = MUX_R3; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, k_loop);
    p.nop(); p.nop(); p.nop();
    { Alu a; a.op_add = A_FADD; a.waddr_add = 15; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = 15;
      a.add_b = MUX_R1; p.alu(a); }
    for (int i = 0; i < ahead - 1; i++) p.nop(SIG_LDTMU0);

    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 2);
    for (uint32_t r = 0; r < 16; r++) p.mov(W_VPM, 0, MUX_A, r);
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

// One 2-D product inside a launch: a batch slice of a MUL_MAT. Addresses are bus addresses.
struct GemmWtProblem {
    int M, N, K;
    uint32_t w, w_stride;   // W row 0, bytes between rows
    uint32_t c, c_stride;   // dst row 0 (n = 0), bytes between dst rows
    uint32_t xpack;         // this slice's packed X (gemm_wt_xpack_bytes(N, K) bytes)
};

inline int gemm_wt_kp(int K) { return (K + GEMM_U - 1) / GEMM_U * GEMM_U; }
inline size_t gemm_wt_xpack_bytes(int N, int K) { return size_t((N + 15) / 16) * gemm_wt_kp(K) * 16 * 4; }

// Pack X rows (row n at x + n * x_stride floats) k-major in 16-row blocks, zero-padded to Kp and to
// a whole block of rows. k outer so the writes are sequential: the r-outer order scattered every
// write 64 bytes from the last and cost 45 ms for a 3656 x 384 X on the Pi Zero.
inline void gemm_wt_pack_x(const float* x, size_t x_stride, int N, int K, float* out) {
    const int Kp = gemm_wt_kp(K), nb = (N + 15) / 16;
    for (int b = 0; b < nb; b++) {
        float* o = out + size_t(b) * Kp * 16;
        const int rows = std::min(16, N - b * 16);
        const float* row[16];
        for (int r = 0; r < rows; r++) row[r] = x + size_t(b * 16 + r) * x_stride;
        for (int k = 0; k < K; k++, o += 16) {
            int r = 0;
            for (; r < rows; r++) o[r] = row[r][k];
            for (; r < 16; r++) o[r] = 0.f;
        }
        for (int k = K; k < Kp; k++, o += 16)
            for (int r = 0; r < 16; r++) o[r] = 0.f;
    }
}

// Words of uniforms gemm_wt_uniforms writes for these problems on nq QPUs (upper bound).
inline size_t gemm_wt_uniform_words(const std::vector<GemmWtProblem>& probs, int nq) {
    size_t tiles = 0;
    for (auto& p : probs) tiles += size_t((p.M + 15) / 16) * ((p.N + 15) / 16);
    return size_t(nq) * (9 + 16) + tiles * 7;
}

// Write the per-QPU uniform streams for `probs` (all must share K) at `unif`, round-robin over
// min(nq_max, tiles) QPUs. Returns the number of QPUs to launch; `starts` gets each stream's address.
template <class CpuMap>
int gemm_wt_uniforms(CpuMap cpu, uint32_t unif, const std::vector<GemmWtProblem>& probs, int nq_max,
                     uint32_t done, uint32_t seq, std::vector<uint32_t>& starts) {
    struct Tile { const GemmWtProblem* p; int m0, n0; };
    std::vector<Tile> tiles;
    // m-major: the tiles running at the same time share one 16-row block of W, which every QPU
    // gathers through the TMU. n-major order (each QPU a different W block) ran 3.6x slower on the
    // Pi Zero -- four 16-row working sets per slice's TMU cache instead of one.
    for (auto& p : probs)
        for (int m0 = 0; m0 < p.M; m0 += 16)
            for (int n0 = 0; n0 < p.N; n0 += 16) tiles.push_back({&p, m0, n0});
    const int nq = std::min<int>(nq_max, int(tiles.size()));
    starts.clear();
    if (nq == 0) return 0;
    const int K = probs[0].K, Kp = gemm_wt_kp(K);
    const uint32_t vpm_hsetup = 1u << 12 | 1u << 11 | 2u << 8;
    const uint32_t done_vdw = 0x80000000u | 1u << 23 | 16u << 16 | 1u << 14;
    uint32_t at = unif;
    for (int q = 0; q < nq; q++) {
        starts.push_back(at);
        uint32_t* u = cpu(at);
        int i = 0, mine = 0;
        for (size_t t = q; t < tiles.size(); t += nq) mine++;
        u[i++] = uint32_t(mine);
        u[i++] = 0;  // W row stride: per problem, patched below (one stride per launch)
        u[i++] = uint32_t(Kp / GEMM_U);
        u[i++] = vpm_hsetup;
        u[i++] = vpm_hsetup;
        u[i++] = 0xc0000000u;
        u[i++] = done_vdw;
        u[i++] = done + q * 64;
        u[i++] = seq;
        u[1] = probs[0].w_stride;
        for (size_t t = q; t < tiles.size(); t += nq) {
            const Tile& T = tiles[t];
            const GemmWtProblem& P = *T.p;
            const int mvalid = std::min(16, P.M - T.m0), nvalid = std::min(16, P.N - T.n0);
            u[i++] = uint32_t(mvalid - 1);
            u[i++] = P.w + uint32_t(T.m0) * P.w_stride;
            u[i++] = 0xc0000000u | (P.c_stride - uint32_t(mvalid) * 4);
            u[i++] = 0x80000000u | uint32_t(nvalid) << 23 | uint32_t(mvalid) << 16 | 1u << 14;
            u[i++] = P.c + uint32_t(T.n0) * P.c_stride + uint32_t(T.m0) * 4;
            u[i++] = P.xpack + uint32_t(size_t(T.n0 / 16) * Kp * 16 * 4);
            u[i] = at + uint32_t(i + 1) * 4;
            i++;
        }
        at = (at + uint32_t(i) * 4 + 63) & ~63u;
    }
    return nq;
}

}  // namespace vc4
