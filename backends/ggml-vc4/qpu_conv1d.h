// 1-D convolution for the VideoCore IV QPUs, read in place: no im2col, no copy of the input.
//
//     x   [IC][T_in]  (row ic at x + ic * x_row bytes)        -- activations, gathered by the TMU
//     W   [OC][IC*KW] (k = ic * KW + kx, ggml's folded order)  -- weights, packed once into uniforms
//     dst [OC][T_out] (row oc at dst + oc * dst_row bytes)
//     dst[oc][t] = sum_k W[oc][k] * x[ic][t*s + kx*d - p],  zero outside [0, T_in)
//
// The GEMM kernel's roles, turned around: lane l <-> output position t0 + l gathers x through TMU0
// (16 consecutive floats per k when s = 1, so the TMU cache serves the KW - 1 shifted reads that
// follow), and accumulator ra_r <-> output channel oc0 + r takes W[oc0+r][k] from the uniform stream,
// a weight block packed once per weight ([oc block][k][16] F32, zero past OC). The im2col an F32 GEMM
// would need is 224 x 75008 floats -- 67 MB -- for one VITS vocoder convolution; here it is an address
// increment per k: d*4 within an input channel, and x_row - (KW-1)*d*4 to the next one.
//
// The QPUs take the INTERIOR -- every t whose KW taps all land inside x -- and the host computes the
// few padded positions at each end (conv1d_interior), so no lane ever needs a bounds test.
//
// Epilogue (P4.46, qpu_gemm.h's Epilogue): the bias is where the accumulators start, 16 uniforms per
// tile in place of 16 zeroing ldi -- free; the residual is one TMU request per output channel, 2
// instructions a channel, ~1% of a VITS vocoder tile; the mask one TMU read a tile.
//
// Output: an accumulator is one output channel's 16 positions, one horizontal VPM row. When dst's
// rows are close enough for the VDW stride field (13 bits: T_out up to ~2048) the tile is one DMA of
// nvalid rows, as in the GEMM kernel (CONV1D_OUT_BLOCK); a VITS vocoder row is 300 KB, so there each
// row is its own DMA, addressed by a uniform, a discard area for a channel past OC (CONV1D_OUT_ROWS).
//
// TMU pipeline: GEMM_TMU_AHEAD requests ahead, as in gemm_f32_wt_kernel, but never past the last k:
// the last P = ceil(ahead / KW) input channels are a peeled tail that issues no request beyond it,
// so nothing reads past x's last row and no drain is needed.
#pragma once

#include "qpu_gemm.h"  // GEMM_TMU_AHEAD, Program

#include <algorithm>
#include <cstring>
#include <vector>

namespace vc4 {

struct Conv1dShape {
    int IC, KW, s, d, p, T_in, T_out, OC;
};

// [lo, hi): the output positions whose taps are all inside x.
inline void conv1d_interior(const Conv1dShape & c, int & lo, int & hi) {
    lo = (c.p + c.s - 1) / c.s;
    const int last = c.T_in - 1 + c.p - (c.KW - 1) * c.d;  // t*s <= last
    hi = last < 0 ? 0 : std::min(c.T_out, last / c.s + 1);
    lo = std::min(lo, c.T_out);
    hi = std::max(hi, lo);
}

// Input channels per loop iteration (at least 4 k per branch, whose 4 instructions are overhead) and
// in the peeled tail (at least enough to cover the TMU look-ahead).
inline int conv1d_group(int KW) { return std::max(1, (GEMM_U + KW - 1) / KW); }
inline int conv1d_peel(int KW) { return (GEMM_TMU_AHEAD + KW - 1) / KW; }
// The tail length for IC input channels; the program depends on (KW, tail).
inline int conv1d_tail(int IC, int KW) {
    const int G = conv1d_group(KW), P = conv1d_peel(KW);
    return IC - (IC - P) / G * G;
}
inline bool conv1d_kernel_ok(int IC, int KW) { return KW >= 1 && IC - conv1d_peel(KW) >= conv1d_group(KW); }

// Register map.
//   ra0..ra15 accumulators   ra16 weight block address   ra18 tiles left   ra19 seq
//   rb0 lane stride s*4   rb1 loop count   rb2 VPM horizontal setup   rb4 next-channel step
//   rb5 tap step d*4      rb6 tile row addresses   rb8 tile VDW setup (row 0)   rb9 valid lanes - 1
//   rb10 constant 128
//   rb11/12/13/14 done marker setups and address
//   r0 x address / VDW setup   r1 product   r2 TMU address   r3 scratch / loop counter   r4 TMU data
enum Conv1dOut { CONV1D_OUT_ROWS, CONV1D_OUT_BLOCK };

inline bool conv1d_block_ok(uint32_t dst_row) { return dst_row <= 8191 + 4; }


inline Program conv1d_f32_kernel(int KW, int tail, Conv1dOut out = CONV1D_OUT_ROWS, unsigned epi = 0) {
    const int ahead = GEMM_TMU_AHEAD, G = conv1d_group(KW);
    Program p;
    auto unif_to_a = [&](uint32_t ra) { p.mov(ra, 0, MUX_A, R_UNIFORM); };
    auto unif_to_b = [&](uint32_t rb) { p.mov(rb, 1, MUX_A, R_UNIFORM); };

    unif_to_a(18);  // tiles
    unif_to_b(0);   // lane stride
    unif_to_b(1);   // loop count
    unif_to_b(2);   // VPM horizontal write setup, row 0
    unif_to_b(11);  // done: VPM setup
    unif_to_b(13);  // done: VDW stride setup
    unif_to_b(12);  // done: VDW setup
    unif_to_b(14);  // done: address
    unif_to_a(19);  // seq
    unif_to_b(5);   // tap step
    unif_to_b(4);   // next-channel step
    p.raw(ldi(10, 1, 128));

    const size_t tile_loop = p.here();
    // Tile params: valid lanes - 1, x address of lane 0's first tap, VDW setup, weight block, where
    // the epilogue's own params are, and with EPI_BIAS the 16 channels' biases. The epilogue's params
    // follow (16 residual row addresses, a mask address, then 16 row addresses or a VDW stride setup
    // and the tile's dst address), read after the k loop, and the next tile's params follow them.
    unif_to_b(9);
    p.mov(W_R0, 0, MUX_A, R_UNIFORM);
    unif_to_b(8);
    unif_to_a(16);
    unif_to_b(6);
    { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
      a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
    { Alu a; a.op_mul = M_MUL24; a.waddr_mul = W_R3; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R3; a.mul_b = MUX_B;
      a.raddr_b = 0; p.alu(a); }
    { Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R0; a.add_b = MUX_R3; p.alu(a); }
    if (epi & EPI_BIAS) {  // read before the uniform stream moves to the weights: the same 16 instructions
        for (uint32_t r = 0; r < 16; r++) unif_to_a(r);
        p.mov(W_UNIFORMS_ADDR, 0, MUX_A, 16);
    } else {
        p.mov(W_UNIFORMS_ADDR, 0, MUX_A, 16);
        for (uint32_t r = 0; r < 16; r++) p.raw(ldi(r, 0, 0));
    }
    p.raw(ldi(W_R1, 0, 0));
    // A TMU request for the tap whose kernel column is kx, then the step to the next tap's address.
    auto tmu_req = [&](int kx) {
        Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R2;
        a.add_b = MUX_B; a.raddr_b = kx == KW - 1 ? 4 : 5;
        a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R2; a.mul_b = MUX_R2;
        p.alu(a);
    };
    // One k: 16 x (fadd ra_(r-1) += r1 ; fmul r1 = r4 * uniform), the next tap's pop on the last.
    auto fmacs = [&](bool pop) {
        for (uint32_t r = 0; r < 16; r++) {
            Alu a; a.sig = r == 15 && pop ? SIG_LDTMU0 : SIG_NONE;
            a.op_add = A_FADD; a.waddr_add = (r + 15) & 15; a.cond_add = C_ALWAYS;
            a.add_a = MUX_A; a.raddr_a = (r + 15) & 15; a.add_b = MUX_R1;
            a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R4; a.mul_b = MUX_B;
            a.raddr_b = R_UNIFORM;
            p.alu(a);
        }
    };
    for (int q = 0; q < ahead; q++) tmu_req(q % KW);
    p.nop(SIG_LDTMU0);
    p.mov(W_R3, 0, MUX_B, R_NOP, 1);

    const size_t k_loop = p.here();
    for (int j = 0; j < G * KW; j++) {
        tmu_req((j + ahead) % KW);
        fmacs(true);
    }
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = W_R3; a.cond_add = C_ALWAYS;
      a.add_a = MUX_R3; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, k_loop);
    p.nop(); p.nop(); p.nop();
    for (int j = 0; j < tail * KW; j++) {
        if (j + ahead < tail * KW) tmu_req((j + ahead) % KW);
        fmacs(j + 1 < tail * KW);
    }
    { Alu a; a.op_add = A_FADD; a.waddr_add = 15; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = 15;
      a.add_b = MUX_R1; p.alu(a); }

    // Back to the tile params: the epilogue's come next.
    p.mov(W_UNIFORMS_ADDR, 0, MUX_B, R_NOP, 6);
    if (epi & (EPI_RESIDUAL | EPI_MASK)) {
        // r3 <- lane l's byte offset along a [.][T_out] row, min(l, valid - 1) * 4 (the three
        // instructions also cover the uniform read after the switch)
        { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
          a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
        { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SHL; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_R3;
          a.add_b = MUX_B; a.raddr_b = small_int(2); p.alu(a); }
        p.nop();
    }
    // r0 <- the next uniform (an address) + r3; the TMU request for it goes in the instruction after.
    auto addr = [&](Alu & a) {
        a.op_add = A_ADD; a.waddr_add = W_R0; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = R_UNIFORM;
        a.add_b = MUX_R3;
    };
    auto request = [&](Alu & a) {
        a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R0; a.mul_b = MUX_R0;
    };
    if (epi & EPI_RESIDUAL) {
        // ra_r += residual[oc0 + r][t0 + l]: four requests in flight, never a pop and a TMU write in
        // one instruction (qpu_gemm.h).
        { Alu a; addr(a); p.alu(a); }
        for (int r = 0; r < 4; r++) { Alu a; request(a); if (r < 3) addr(a); p.alu(a); }
        for (uint32_t r = 0; r < 16; r++) {
            { Alu a; a.sig = SIG_LDTMU0; if (r + 4 < 16) addr(a); p.alu(a); }
            { Alu a; a.op_add = A_FADD; a.waddr_add = r; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = r;
              a.add_b = MUX_R4; if (r + 4 < 16) request(a); p.alu(a); }
        }
    }
    if (epi & EPI_MASK) {
        { Alu a; addr(a); p.alu(a); }
        { Alu a; request(a); p.alu(a); }
        p.nop(SIG_LDTMU0);
    }
    if ((epi & EPI_RELU) && (epi & EPI_MASK)) {
        // r1 <- max(ra_i, 0) ; ra_(i-1) <- r1 * mask, the mul writing regfile A (ws = 1)
        for (uint32_t i = 0; i <= 16; i++) {
            Alu a; a.ws = 1;
            if (i < 16) {
                a.sig = SIG_SMALL_IMM; a.op_add = A_FMAX; a.waddr_add = W_R1; a.cond_add = C_ALWAYS; a.add_a = MUX_A;
                a.raddr_a = i; a.add_b = MUX_B; a.raddr_b = small_int(0);
            }
            if (i > 0) { a.op_mul = M_FMUL; a.waddr_mul = i - 1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R1; a.mul_b = MUX_R4; }
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
            a.mul_b = MUX_R4;
            p.alu(a);
        }
    }
    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 2);
    for (uint32_t r = 0; r < 16; r++) p.mov(W_VPM, 0, MUX_A, r);
    if (out == CONV1D_OUT_BLOCK) {
        p.mov(W_VPMVCD_SETUP, 1, MUX_A, R_UNIFORM);   // VDW stride setup
        p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 8);    // VDW setup, nvalid rows
        p.mov(W_VPM_ADDR, 1, MUX_A, R_UNIFORM);
        p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    } else {
        // The VPM DMA setup registers are shared by the 12 QPUs, so they are written only under the
        // mutex: the first version set this stride once at program start, outside it, and on the
        // Pi Zero 12 QPUs then hung where one QPU on one tile did not (2026-10-28).
        p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 13);  // VDW stride 0, for every row DMA below
        p.mov(W_R0, 0, MUX_B, R_NOP, 8);
        for (uint32_t r = 0; r < 16; r++) {
            p.mov(W_VPMVCD_SETUP, 1, MUX_R0);
            { Alu a; a.op_add = A_ADD; a.waddr_add = W_R0; a.cond_add = C_ALWAYS; a.add_a = MUX_R0; a.add_b = MUX_B;
              a.raddr_b = 10; p.alu(a); }
            p.mov(W_VPM_ADDR, 1, MUX_A, R_UNIFORM);
            p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
        }
    }
    p.mutex_release();

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

// Weight blocks: W [OC][K] (row oc at w + oc * K floats) -> [ceil(OC/16)][K][16], zero past OC.
inline size_t conv1d_wpack_floats(int OC, int K) { return size_t((OC + 15) / 16) * K * 16; }
// Writes in output order: the ARM1176 does not allocate cache lines on a write, so a store pattern
// that strides 64 bytes goes to the bus one word at a time (a chunked k-then-r order measured ~80 ns
// a weight on the Pi Zero); the 16 rows it reads stay in L1.
inline void conv1d_pack_w(const float * w, int OC, int K, float * out) {
    for (int b = 0; b < (OC + 15) / 16; b++)
        for (int k = 0; k < K; k++)
            for (int r = 0; r < 16; r++) {
                const int oc = b * 16 + r;
                *out++ = oc < OC ? w[size_t(oc) * K + k] : 0.f;
            }
}

// Right padding that makes every output position interior once x is staged as
// [p zeros | x | pr zeros]: the last tap of the last position lands inside.
inline int conv1d_pad_right(const Conv1dShape & c) {
    const int last = (c.T_out - 1) * c.s + (c.KW - 1) * c.d;  // in the padded coordinates
    return std::max(0, last + 1 - (c.p + c.T_in));
}

// One launch's addresses (bus): x row 0 / dst row 0 / the packed weights / a 64-byte discard area.
struct Conv1dProblem {
    Conv1dShape c;
    uint32_t x, x_row, dst, dst_row, wpack, discard;
};

// What the epilogue reads, for one launch: the biases (OC floats, host memory: they go into the
// uniform stream), the residual's row 0 at this launch's t = 0 and the bytes between its rows, the
// mask's element at t = 0 (bus addresses).
struct Conv1dEpi {
    unsigned flags = 0;
    const float * bias = nullptr;
    uint32_t residual = 0, residual_row = 0;
    uint32_t mask = 0;
};

constexpr int CONV1D_GLOBAL_WORDS = 11, CONV1D_TILE_WORDS = 5 + 16 + 16 + 16 + 1;  // ROWS, every epilogue; BLOCK uses 5 + 2

inline size_t conv1d_uniform_words(const Conv1dProblem & P, int nq) {
    int lo, hi;
    conv1d_interior(P.c, lo, hi);
    const size_t tiles = size_t((P.c.OC + 15) / 16) * ((hi - lo + 15) / 16);
    return size_t(nq) * (CONV1D_GLOBAL_WORDS + 16) + tiles * CONV1D_TILE_WORDS;
}

// The per-QPU uniform streams for the interior tiles, oc-block-major (the tiles running at once share
// one weight block). Returns the number of QPUs to launch (0: no interior); `starts` gets each stream.
template <class CpuMap>
int conv1d_uniforms(CpuMap cpu, uint32_t unif, const Conv1dProblem & P, int nq_max, uint32_t done, uint32_t seq,
                    std::vector<uint32_t> & starts, Conv1dOut out = CONV1D_OUT_ROWS, const Conv1dEpi & epi = {}) {
    const Conv1dShape & c = P.c;
    const int K = c.IC * c.KW, G = conv1d_group(c.KW), tail = conv1d_tail(c.IC, c.KW);
    int lo, hi;
    conv1d_interior(c, lo, hi);
    struct Tile { int b, t0; };
    std::vector<Tile> tiles;
    for (int b = 0; b < (c.OC + 15) / 16; b++)
        for (int t0 = lo; t0 < hi; t0 += 16) tiles.push_back({b, t0});
    const int nq = std::min<int>(nq_max, int(tiles.size()));
    starts.clear();
    if (nq == 0) return 0;
    const uint32_t vpm_hsetup = 1u << 12 | 1u << 11 | 2u << 8;
    const uint32_t done_vdw = 0x80000000u | 1u << 23 | 16u << 16 | 1u << 14;
    uint32_t at = unif;
    for (int q = 0; q < nq; q++) {
        starts.push_back(at);
        uint32_t * u = cpu(at);
        int i = 0, mine = 0;
        for (size_t t = q; t < tiles.size(); t += nq) mine++;
        u[i++] = uint32_t(mine);
        u[i++] = uint32_t(c.s) * 4;
        u[i++] = uint32_t((c.IC - tail) / G);
        u[i++] = vpm_hsetup;
        u[i++] = vpm_hsetup;
        u[i++] = 0xc0000000u;
        u[i++] = done_vdw;
        u[i++] = done + q * 64;
        u[i++] = seq;
        u[i++] = uint32_t(c.d) * 4;
        u[i++] = P.x_row - uint32_t((c.KW - 1) * c.d) * 4;
        for (size_t t = q; t < tiles.size(); t += nq) {
            const Tile & T = tiles[t];
            const int mvalid = std::min(16, hi - T.t0), nvalid = std::min(16, c.OC - T.b * 16);
            const uint32_t rows = out == CONV1D_OUT_BLOCK ? uint32_t(nvalid) : 1u;
            u[i++] = uint32_t(mvalid - 1);
            u[i++] = P.x + uint32_t(T.t0 * c.s - c.p) * 4;
            u[i++] = 0x80000000u | rows << 23 | uint32_t(mvalid) << 16 | 1u << 14;
            u[i++] = P.wpack + uint32_t(size_t(T.b) * K * 16 * 4);
            u[i] = at + uint32_t(i + 1 + (epi.flags & EPI_BIAS ? 16 : 0)) * 4;
            i++;
            if (epi.flags & EPI_BIAS)
                for (int r = 0; r < 16; r++) {
                    const float b = T.b * 16 + r < c.OC ? epi.bias[T.b * 16 + r] : 0.f;
                    memcpy(&u[i++], &b, 4);
                }
            if (epi.flags & EPI_RESIDUAL)  // a channel past OC reads the last one's row (and is discarded)
                for (int r = 0; r < 16; r++)
                    u[i++] = epi.residual + uint32_t(std::min(T.b * 16 + r, c.OC - 1)) * epi.residual_row + uint32_t(T.t0) * 4;
            if (epi.flags & EPI_MASK) u[i++] = epi.mask + uint32_t(T.t0) * 4;
            if (out == CONV1D_OUT_BLOCK) {
                u[i++] = 0xc0000000u | (P.dst_row - uint32_t(mvalid) * 4);
                u[i++] = P.dst + uint32_t(T.b * 16) * P.dst_row + uint32_t(T.t0) * 4;
            } else {
                for (int r = 0; r < 16; r++) {
                    const int oc = T.b * 16 + r;
                    u[i++] = oc < c.OC ? P.dst + uint32_t(oc) * P.dst_row + uint32_t(T.t0) * 4 : P.discard;
                }
            }
        }
        at = (at + uint32_t(i) * 4 + 63) & ~63u;
    }
    return nq;
}

// The positions outside [lo, hi), on the host, in the QPUs' order (k ascending, fmul then fadd), so the
// emulator test can hold the whole output to one reference. `wpack` is the packed block layout, read
// 16 channels at a time in its own order: one channel at a time strode 64 bytes per k through it, and
// Moonshine's 576 x 2016 kernel took 868 ms for six edge positions on the Pi Zero.
inline void conv1d_edges_host(const Conv1dShape & c, const float * x, size_t x_row_floats, const float * wpack,
                              float * dst, size_t dst_row_floats) {
    int lo, hi;
    conv1d_interior(c, lo, hi);
    const int K = c.IC * c.KW;
    auto one = [&](int t) {
        for (int b = 0; b < (c.OC + 15) / 16; b++) {
            const float * wb = wpack + size_t(b) * K * 16;
            float acc[16] = {};
            for (int ic = 0; ic < c.IC; ic++)
                for (int kx = 0; kx < c.KW; kx++) {
                    const int ti = t * c.s + kx * c.d - c.p;
                    const float v = ti >= 0 && ti < c.T_in ? x[size_t(ic) * x_row_floats + ti] : 0.f;
                    const float * w = wb + size_t(ic * c.KW + kx) * 16;
                    for (int r = 0; r < 16; r++) acc[r] = acc[r] + w[r] * v;
                }
            for (int r = 0; r < 16 && b * 16 + r < c.OC; r++) dst[size_t(b * 16 + r) * dst_row_floats + t] = acc[r];
        }
    };
    for (int t = 0; t < lo; t++) one(t);
    for (int t = hi; t < c.T_out; t++) one(t);
}

}  // namespace vc4
