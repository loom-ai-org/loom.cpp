// Per-channel int8 MUL_MAT for the VideoCore IV QPUs -- Q8_0_PC weights (one scale per row) read
// four to a word, the int8 ADR-079 runs on the CPU, at F32 speed on the QPUs.
//
//     W = src0 [K, M] int8, row m with ONE scale s_m (ggml Q8_0 whose blocks in a row share d)
//     X = src1 [K, N] F32, packed per call
//     C[n][m] = s_m * sum_k q[m][k] * X[n][k]
//
// gemm_f32_wt_kernel with one change in the inner loop: the TMU word lane l pops holds FOUR weights
// of row m0 + l, and each of the four k-steps that follow reads its byte straight out of r4 through
// the 8-bit unpack (pm = 1, unpack 8a..8d). That unpack is a colour conversion: it yields u / 255
// as a float, not u, and it is unsigned -- measured on the Pi Zero (probe_unpack, 2026-10-28), u/255
// truncated to 24 bits for every u, integer ops included. So the stored byte is offset binary,
// u = q + 128, the activations are packed times 255, and each column's 128 * sum_k X[n][k] goes in
// after its block:
//
//     sum_k c(u) * (255 x)  ~=  sum_k u x  =  sum_k q x + 128 sum_k x
//     C[n][m] = s_m * (acc - 128 * sum_k x[n][k])
//
// c(u) * 255 is u to within 2^-23 relative. What the offset does cost is that the sum carries
// 128 * sum x, which the subtraction removes, so float rounding is relative to sum |u x| rather than
// sum |q x| -- about 128 / mean|q| (~3x for typical weights) more than an F32 GEMM of the same
// product; still far below Q8's own quantization error.
//
// Per word (4 k): 1 x (iadd r2 += 4 ; tmu0_s = r2) then 4 x 16 fmacs, the next word's ldtmu on the
// last: 65 instructions per 4 k against the F32 kernel's 68, and a quarter of its TMU traffic.
// Epilogue per tile: the scales gathered once through the TMU, then 17 instructions of
// (fsub r1 = ra_i - uniform ; fmul ra_(i-1) = r1 * r0) with the write sides swapped (ws = 1) so the
// mul ALU writes regfile A.
#pragma once

#include "qpu_gemm.h"  // GEMM_TMU_AHEAD, Program

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vc4 {

constexpr int Q8_WORDS_PER_ITER = 2;            // 8 k per loop iteration
constexpr int Q8_KSTEP = 4 * Q8_WORDS_PER_ITER;

inline int gemm_q8_kp(int K) { return (K + Q8_KSTEP - 1) / Q8_KSTEP * Q8_KSTEP; }
// Repacked weight: M rows of Kp offset-binary bytes, gemm_q8_stride(K) apart (gemm_row_stride), then
// M float scales, then the TMU's over-read.
inline uint32_t gemm_q8_stride(int K) { return uint32_t(gemm_row_stride(size_t(gemm_q8_kp(K)))); }
inline size_t gemm_q8_tail_bytes() { return size_t(GEMM_TMU_AHEAD + 1) * 4; }
inline size_t gemm_q8_scales_at(int M, int K) { return size_t(M) * gemm_q8_stride(K); }
inline size_t gemm_q8_weight_bytes(int M, int K) { return gemm_q8_scales_at(M, K) + size_t(M) * 4 + gemm_q8_tail_bytes(); }
inline size_t gemm_q8_xpack_bytes(int N, int K) { return size_t((N + 15) / 16) * (size_t(gemm_q8_kp(K)) + 1) * 16 * 4; }

// Epilogue (P4.46, qpu_gemm.h's Epilogue), for GEMM_OUT_M -- a 1x1 convolution, lanes its output
// channels, accumulators its positions: the bias is one TMU read a tile and 16 fadds, the residual one
// request per position as in qpu_conv1d.h, the mask 16 uniforms a tile (one per position). Read from
// the tile's own stream, after its scale address.
//
// Register map: gemm_f32_wt_kernel's, plus rb10 the tile's scale address.
inline Program gemm_q8_kernel(unsigned epi = 0) {
    const int ahead = GEMM_TMU_AHEAD;
    Program p;
    auto unif_to_a = [&](uint32_t ra) { p.mov(ra, 0, MUX_A, R_UNIFORM); };
    auto unif_to_b = [&](uint32_t rb) { p.mov(rb, 1, MUX_A, R_UNIFORM); };

    unif_to_a(18);  // tiles
    unif_to_b(0);   // W row stride (bytes)
    unif_to_b(1);   // loop count = Kp / Q8_KSTEP
    unif_to_b(2);   // VPM horizontal write setup, row 0
    unif_to_b(11);  // done: VPM setup
    unif_to_b(13);  // done: VDW stride setup
    unif_to_b(12);  // done: VDW setup
    unif_to_b(14);  // done: address
    unif_to_a(19);  // seq
    p.raw(ldi(5, 1, 4));

    const size_t tile_loop = p.here();
    // Tile params: mvalid-1, W base (row m0), VDW stride setup, VDW setup, dst, X block, next, scales.
    unif_to_b(9);
    p.mov(W_R0, 0, MUX_A, R_UNIFORM);
    unif_to_b(3);
    unif_to_b(8);
    unif_to_b(7);
    unif_to_a(16);
    unif_to_b(6);
    unif_to_b(10);
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
    for (int w = 0; w < Q8_WORDS_PER_ITER; w++) {
        tmu_req();
        for (int b = 0; b < 4; b++)
            for (uint32_t r = 0; r < 16; r++) {
                Alu a; a.sig = b == 3 && r == 15 ? SIG_LDTMU0 : SIG_NONE;
                a.op_add = A_FADD; a.waddr_add = (r + 15) & 15; a.cond_add = C_ALWAYS;
                a.add_a = MUX_A; a.raddr_a = (r + 15) & 15; a.add_b = MUX_R1;
                a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R4; a.mul_b = MUX_B;
                a.raddr_b = R_UNIFORM; a.pm = 1; a.unpack = 4 + b;
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

    // The tile's scales, lane l <-> s[m0 + min(l, mvalid - 1)], into r0.
    { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
      a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SHL; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_R3;
      a.add_b = MUX_B; a.raddr_b = small_int(2); p.alu(a); }
    { Alu a; a.op_add = A_ADD; a.waddr_add = W_TMU0_S; a.cond_add = C_ALWAYS; a.add_a = MUX_R3; a.add_b = MUX_B;
      a.raddr_b = 10; p.alu(a); }
    p.nop(SIG_LDTMU0);
    p.mov(W_R0, 0, MUX_R4);
    // C = (acc - 128 * sum x) * s: the column sums follow the X block in the uniform stream.
    for (uint32_t i = 0; i <= 16; i++) {
        Alu a; a.ws = 1;
        if (i < 16) {
            a.op_add = A_FSUB; a.waddr_add = W_R1; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = i;
            a.add_b = MUX_B; a.raddr_b = R_UNIFORM;
        }
        if (i > 0) {
            a.op_mul = M_FMUL; a.waddr_mul = i - 1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R1; a.mul_b = MUX_R0;
        }
        p.alu(a);
    }

    if (epi) {
        // The epilogue's params follow the tile's in its stream, and the next tile's follow them, so
        // the stream is read on from here and not switched again below. r3 is still min(l, valid-1)*4.
        p.mov(W_UNIFORMS_ADDR, 0, MUX_B, R_NOP, 6);
        p.nop(); p.nop();
        if (epi & EPI_BIAS) {
            { Alu a; a.op_add = A_ADD; a.waddr_add = W_TMU0_S; a.cond_add = C_ALWAYS; a.add_a = MUX_A;
              a.raddr_a = R_UNIFORM; a.add_b = MUX_R3; p.alu(a); }
            p.nop(SIG_LDTMU0);
            for (uint32_t r = 0; r < 16; r++) {
                Alu a; a.op_add = A_FADD; a.waddr_add = r; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = r;
                a.add_b = MUX_R4; p.alu(a);
            }
        }
        if (epi & EPI_RESIDUAL) {
            // r3 <- min(l, valid-1) * the residual's row bytes; then position r's word of lane l's row
            { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
              a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
            { Alu a; a.op_mul = M_MUL24; a.waddr_mul = W_R3; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R3; a.mul_b = MUX_A;
              a.raddr_a = R_UNIFORM; p.alu(a); }
            auto addr = [&](Alu & a) {
                a.op_add = A_ADD; a.waddr_add = W_R0; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = R_UNIFORM;
                a.add_b = MUX_R3;
            };
            auto request = [&](Alu & a) {
                a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R0; a.mul_b = MUX_R0;
            };
            { Alu a; addr(a); p.alu(a); }
            for (int r = 0; r < 4; r++) { Alu a; request(a); if (r < 3) addr(a); p.alu(a); }
            for (uint32_t r = 0; r < 16; r++) {
                { Alu a; a.sig = SIG_LDTMU0; if (r + 4 < 16) addr(a); p.alu(a); }
                { Alu a; a.op_add = A_FADD; a.waddr_add = r; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = r;
                  a.add_b = MUX_R4; if (r + 4 < 16) request(a); p.alu(a); }
            }
        }
        if ((epi & EPI_RELU) && (epi & EPI_MASK)) {
            // r1 <- max(ra_i, 0) ; ra_(i-1) <- r1 * mask[n0 + i - 1] (a uniform), the mul writing regfile A
            p.raw(ldi(W_R2, 0, 0));
            for (uint32_t i = 0; i <= 16; i++) {
                Alu a; a.ws = 1;
                if (i < 16) {
                    a.op_add = A_FMAX; a.waddr_add = W_R1; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = i;
                    a.add_b = MUX_R2;
                }
                if (i > 0) {
                    a.op_mul = M_FMUL; a.waddr_mul = i - 1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R1; a.mul_b = MUX_B;
                    a.raddr_b = R_UNIFORM;
                }
                p.alu(a);
            }
        } else if (epi & EPI_RELU) {
            for (uint32_t r = 0; r < 16; r++) {
                Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_FMAX; a.waddr_add = r; a.cond_add = C_ALWAYS; a.add_a = MUX_A;
                a.raddr_a = r; a.add_b = MUX_B; a.raddr_b = small_int(0);
                p.alu(a);
            }
        } else if (epi & EPI_MASK) {
            for (uint32_t r = 0; r < 16; r++) {
                Alu a; a.ws = 1; a.op_mul = M_FMUL; a.waddr_mul = r; a.cond_mul = C_ALWAYS; a.mul_a = MUX_A; a.raddr_a = r;
                a.mul_b = MUX_B; a.raddr_b = R_UNIFORM;
                p.alu(a);
            }
        }
    }

    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 2);
    for (uint32_t r = 0; r < 16; r++) p.mov(W_VPM, 0, MUX_A, r);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 3);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 8);
    p.mov(W_VPM_ADDR, 1, MUX_B, R_NOP, 7);
    p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    p.mutex_release();

    if (!epi) p.mov(W_UNIFORMS_ADDR, 0, MUX_B, R_NOP, 6);
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

// ggml Q8_0 rows -> the repacked weight. `rows` is M rows of K / 32 blocks {fp16 d; int8 qs[32]};
// `d_to_float` converts a block's d. Returns false (and writes nothing useful) if a row's blocks do
// not share one d: only per-channel weights have a per-row scale.
template <class Block, class D2F>
bool gemm_q8_repack(const Block * rows, int M, int K, D2F d_to_float, uint8_t * out) {
    const int nb = K / 32, Kp = gemm_q8_kp(K);
    float * scales = reinterpret_cast<float *>(out + gemm_q8_scales_at(M, K));
    for (int m = 0; m < M; m++) {
        const Block * b = rows + size_t(m) * nb;
        for (int j = 1; j < nb; j++)
            if (b[j].d != b[0].d) return false;
        uint8_t * o = out + size_t(m) * gemm_q8_stride(K);
        for (int j = 0; j < nb; j++)
            for (int i = 0; i < 32; i++) o[j * 32 + i] = uint8_t(int(b[j].qs[i]) + 128);
        for (int k = K; k < Kp; k++) o[k] = 128;
        scales[m] = d_to_float(b[0].d);
    }
    std::fill(out + gemm_q8_scales_at(M, K) + size_t(M) * 4, out + gemm_q8_weight_bytes(M, K), uint8_t(0));
    return true;
}

// X -> 255 * X, k-major in 16-row blocks zero-padded to Kp, each block followed by its 16 columns'
// 128 * sum_k x (summed in double: the subtraction cancels most of the accumulator). X[n][k] is
// x[n * ns + k * ks]: ks = 1 for a MUL_MAT's activations, ns = 1 for a 1x1 convolution's [IC][T].
inline void gemm_q8_pack_x(const float * x, size_t ns, size_t ks, int N, int K, float * out) {
    const int Kp = gemm_q8_kp(K), nb = (N + 15) / 16;
    for (int b = 0; b < nb; b++) {
        float * o = out + size_t(b) * (Kp + 1) * 16;
        const float * xb = x + size_t(b) * 16 * ns;
        const int rows = std::min(16, N - b * 16);
        double sum[16] = {};
        for (int k = 0; k < K; k++, o += 16) {
            int r = 0;
            for (; r < rows; r++) {
                const float v = xb[size_t(r) * ns + size_t(k) * ks];
                o[r] = 255.f * v;
                sum[r] += v;
            }
            for (; r < 16; r++) o[r] = 0.f;
        }
        for (int k = K; k < Kp; k++, o += 16)
            for (int r = 0; r < 16; r++) o[r] = 0.f;
        for (int r = 0; r < 16; r++) o[r] = float(128.0 * sum[r]);
    }
}
inline void gemm_q8_pack_x(const float * x, size_t x_stride, int N, int K, float * out) {
    gemm_q8_pack_x(x, x_stride, 1, N, K, out);
}

struct GemmQ8Problem {
    int M, N, K;
    uint32_t w, w_stride;   // repacked row 0 (bus), bytes between rows (gemm_q8_stride(K))
    uint32_t scales;        // M floats (bus)
    uint32_t c, c_stride;   // dst row 0 (n = 0), bytes between dst rows
    uint32_t xpack;         // this slice's packed X (gemm_q8_xpack_bytes(N, K) bytes)
};

// The epilogue's operands for GEMM_OUT_M (one problem): the bias, M floats (bus); the residual's row 0
// and the bytes between its rows of m (bus); the mask, N floats (host: they go into the uniforms).
struct GemmQ8Epi {
    unsigned flags = 0;
    uint32_t bias = 0;
    uint32_t residual = 0, residual_row = 0;
    const float * mask = nullptr;
};

inline size_t gemm_q8_uniform_words(const std::vector<GemmQ8Problem> & probs, int nq) {
    size_t tiles = 0;
    for (auto & p : probs) tiles += size_t((p.M + 15) / 16) * ((p.N + 15) / 16);
    return size_t(nq) * (9 + 16) + tiles * (8 + 1 + 17 + 16);
}

// As gemm_wt_uniforms, with each tile's scale address.
template <class CpuMap>
int gemm_q8_uniforms(CpuMap cpu, uint32_t unif, const std::vector<GemmQ8Problem> & probs, int nq_max,
                     uint32_t done, uint32_t seq, std::vector<uint32_t> & starts, GemmOut out = GEMM_OUT_N,
                     int panel = 0, const GemmQ8Epi & epi = {}) {
    std::vector<GemmTile> tiles;
    for (size_t i = 0; i < probs.size(); i++) gemm_tiles(probs[i].M, probs[i].N, panel, int(i), tiles);
    const int nq = std::min<int>(nq_max, int(tiles.size()));
    starts.clear();
    if (nq == 0) return 0;
    const int K = probs[0].K, Kp = gemm_q8_kp(K);
    const uint32_t done_vdw = 0x80000000u | 1u << 23 | 16u << 16 | 1u << 14;
    uint32_t at = unif;
    for (int q = 0; q < nq; q++) {
        starts.push_back(at);
        uint32_t * u = cpu(at);
        int i = 0, mine = 0;
        for (size_t t = q; t < tiles.size(); t += nq) mine++;
        u[i++] = uint32_t(mine);
        u[i++] = probs[0].w_stride;
        u[i++] = uint32_t(Kp / Q8_KSTEP);
        u[i++] = gemm_vpm_setup(out);
        u[i++] = gemm_vpm_setup(GEMM_OUT_N);
        u[i++] = 0xc0000000u;
        u[i++] = done_vdw;
        u[i++] = done + q * 64;
        u[i++] = seq;
        for (size_t t = q; t < tiles.size(); t += nq) {
            const GemmTile & T = tiles[t];
            const GemmQ8Problem & P = probs[T.p];
            const int mvalid = std::min(16, P.M - T.m0), nvalid = std::min(16, P.N - T.n0);
            uint32_t stride_setup, vdw_setup, addr;
            gemm_tile_vdw(out, P.c, P.c_stride, T.m0, T.n0, mvalid, nvalid, stride_setup, vdw_setup, addr);
            u[i++] = uint32_t(mvalid - 1);
            u[i++] = P.w + uint32_t(T.m0) * P.w_stride;
            u[i++] = stride_setup;
            u[i++] = vdw_setup;
            u[i++] = addr;
            u[i++] = P.xpack + uint32_t(size_t(T.n0 / 16) * (Kp + 1) * 16 * 4);
            u[i] = at + uint32_t(i + 2) * 4;
            i++;
            u[i++] = P.scales + uint32_t(T.m0) * 4;
            if (epi.flags & EPI_BIAS) u[i++] = epi.bias + uint32_t(T.m0) * 4;
            if (epi.flags & EPI_RESIDUAL) {  // a position past N reads the last one (and is discarded)
                u[i++] = epi.residual_row;
                for (int r = 0; r < 16; r++)
                    u[i++] = epi.residual + uint32_t(T.m0) * epi.residual_row + uint32_t(std::min(T.n0 + r, P.N - 1)) * 4;
            }
            if (epi.flags & EPI_MASK)
                for (int r = 0; r < 16; r++) memcpy(&u[i++], &epi.mask[std::min(T.n0 + r, P.N - 1)], 4);
        }
        at = (at + uint32_t(i) * 4 + 63) & ~63u;
    }
    return nq;
}

}  // namespace vc4
