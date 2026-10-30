// F32 GEMM for the VideoCore IV QPUs, in ggml's MUL_MAT layout:
//
//     A = src0 [K, M] (row m is K contiguous floats)  -- the weights
//     B = src1 [K, N] (row n is K contiguous floats)  -- the activations
//     C = dst  [M, N] with C[n*M + m] = sum_k A[m*K + k] * B[n*K + k]
//
// One QPU computes a 16(m) x 16(n) tile at a time, lane l <-> n0 + l, accumulator ra_r <-> m0 + r.
// Per k it needs B[n0+l, k] in every lane (a TMU gather, one word per lane) and the scalar
// A[m0+r, k] for r = 0..15 -- which is what the UNIFORM stream is: a broadcast scalar per read. So
// the weights are repacked once, k-major in 16-row blocks (Apack[mb][k][r]), and the uniform pointer
// is aimed at the tile's block. Every k is then 17 instructions:
//
//     16 x  fadd ra_(r-1) += r1  ;  fmul r1 = r4 * uniform      (dual issue: 32 flops each)
//      1 x  iadd r2 += 4         ;  tmu0_s = r2   + ldtmu0       (next B column; pop the oldest)
//
// i.e. 512 flops in 17 instructions = 94% of the ALU peak before loop overhead (U=4 k per loop
// iteration adds 5 instructions: 88%). TMU requests run TMU_AHEAD columns ahead of use.
//
// Output: the 16 accumulators are written VERTICALLY into VPM rows 0..15 (column r = ra_r), so VPM
// row l holds C[n0+l, m0..m0+15], contiguous in dst; one VDW stores the tile's valid rows with a
// row stride of Mp*4. M is padded to Mp = 16*ceil(M/16) in a scratch dst, so a partial m-block can
// never overwrite a neighbour; the host compacts. VPM use is serialised by the hardware mutex.
//
// Completion: execute_qpu returns early for >= 8 QPUs (measured 2026-10-28), so after its last tile
// each QPU DMAs one row holding `seq` to done[q]; the host spins on those words.
#pragma once

#include "qpu_isa.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace vc4 {

constexpr int GEMM_U = 4;          // k per loop iteration
constexpr int GEMM_TMU_AHEAD = 4;  // max outstanding TMU requests (the emulator's TMU_DEPTH); B padding covers it

// The order the weights-in-place kernels (qpu_gemm_wt.h, qpu_gemm_q8.h, qpu_gemm_qb.h) take their
// 16 x 16 tiles in, dealt round-robin to the QPUs, so neighbours in this list run at the same time.
// m-major (panel = 0): the tiles running at once share one 16-row block of W, which every QPU gathers
// through the TMU -- n-major (each QPU a different W block) ran 3.6x slower on the Pi Zero, four
// 16-row working sets per slice's TMU cache instead of one. With panel = P, the P column blocks of a
// panel go through every m-block before the next panel starts (P4.46): P = 2 measured 1.17-1.36x over
// m-major at every model shape (ggml-vc4.cpp's tile_panel has the numbers), P = 1 3.4x slower.
struct GemmTile { int p, m0, n0; };
inline void gemm_tiles(int M, int N, int panel, int p, std::vector<GemmTile> & out) {
    const int nb = (N + 15) / 16, P = panel > 0 ? panel : nb;
    for (int b0 = 0; b0 < nb; b0 += P)
        for (int m0 = 0; m0 < M; m0 += 16)
            for (int b = b0; b < std::min(nb, b0 + P); b++) out.push_back({p, m0, b * 16});
}

// The row stride a weight read through the TMU should have. Each k the 16 lanes gather one word from
// each of 16 rows, so the rows' cache lines must spread over the TMU cache's sets; rows a multiple of
// 512 bytes apart fall into one or two. Measured on the Pi Zero (P4.46, gemm_f32_wt_kernel, 12 QPUs,
// M = 128, N = 3656): K = 384 (1536-byte rows) 2.2 GFLOPS and K = 512 1.5, against 9.3-9.4 for K = 368,
// 400 and 416 -- and the same 2.2 at N = 457 or M = 192, so it was never the size of N that the
// P4.44 note blamed. 64 bytes more puts the 16 rows in 16 different sets.
inline size_t gemm_row_stride(size_t bytes) { return bytes % 512 == 0 ? bytes + 64 : bytes; }

// Where a tile's C lands: GEMM_OUT_N, dst row n (c_stride apart) holds C[n][m0..m0+15] -- a MUL_MAT;
// GEMM_OUT_M, dst row m holds C[n0..n0+15][m] -- a 1x1 convolution's [OC][T].
enum GemmOut { GEMM_OUT_N, GEMM_OUT_M };

// The VPM write setup and a tile's VDW stride / setup / address for either output layout: one VDW of
// nvalid rows x mvalid words (GEMM_OUT_N) or mvalid rows x nvalid words (GEMM_OUT_M, from the VPM rows
// the vertical writes filled). The stride is 13 bits: callers check gemm_out_ok.
inline uint32_t gemm_vpm_setup(GemmOut out) { return out == GEMM_OUT_N ? (1u << 12 | 1u << 11 | 2u << 8) : (1u << 12 | 2u << 8); }
inline bool gemm_out_ok(uint32_t c_stride) { return c_stride <= 8191 + 4; }
inline void gemm_tile_vdw(GemmOut out, uint32_t c, uint32_t c_stride, int m0, int n0, int mvalid, int nvalid,
                          uint32_t & stride_setup, uint32_t & vdw_setup, uint32_t & addr) {
    const int rows = out == GEMM_OUT_N ? nvalid : mvalid, words = out == GEMM_OUT_N ? mvalid : nvalid;
    stride_setup = 0xc0000000u | (c_stride - uint32_t(words) * 4);
    vdw_setup = 0x80000000u | uint32_t(rows) << 23 | uint32_t(words) << 16 | 1u << 14;
    addr = out == GEMM_OUT_N ? c + uint32_t(n0) * c_stride + uint32_t(m0) * 4 : c + uint32_t(m0) * c_stride + uint32_t(n0) * 4;
}

// The epilogue a convolution can absorb (P4.46): what loom's lowering puts after it and ggml's CPU
// backend fuses (ggml-0005/0007/0036) -- relu(conv + bias + residual) * mask, each optional. With the
// convolution on the QPUs and those nodes on the CPU, each was a full pass over the activation on the
// ARM1176: amy's ADDs and LEAKY_RELUs took 2.2 s of a 5.8 s synthesis (P4.45, profiled on the board).
//   bias      one per output channel
//   residual  [OC][T_out] F32, read through the TMU
//   relu      fmax(v, 0)
//   mask      one value per output position (NeMo's time mask)
// qpu_conv1d.h and qpu_gemm_q8.h (GEMM_OUT_M: a 1x1 convolution) each say what it costs them. A
// LEAKY_RELU before the convolution is the host's, as it stages x (see ggml-vc4.cpp).
enum Epilogue : unsigned { EPI_BIAS = 1, EPI_RESIDUAL = 2, EPI_RELU = 4, EPI_MASK = 8 };

// Register map.
//   ra0..ra15  accumulators         ra16 A block address   ra18 tiles left   ra19 seq
//   rb0 B row stride (bytes)        rb1 k-loop count        rb2 VPM vertical write setup
//   rb3 VDW stride setup (tiles)    rb5 constant 4          rb6 next tile params address
//   rb7 tile dst address            rb8 tile VDW setup      rb9 valid n rows - 1
//   rb11 done VPM setup  rb12 done VDW setup  rb13 done VDW stride (0)  rb14 done address
//   r1 product   r2 TMU address   r3 scratch / k counter   r4 TMU data   r0 scratch
// How the per-k TMU pop is scheduled. py-videocore's checker forbids an ldtmu signal in an
// instruction that also writes a TMU register; the first kernel did exactly that, and on hardware
// (2026-10-28) a pop then sometimes returned the column from `ahead` requests earlier -- whenever
// more than one QPU shared the slice's TMU, and more often the shallower the prefetch. Both
// variants below are exact on hardware at every depth 1..4 (8 launches x 12 QPUs each); the
// 17-instruction one, which pops on the last fmac (whose own r4 read sees the old value), is default.
enum GemmVariant { GEMM_SEPARATE_POP = 0, GEMM_LDTMU_ON_LAST_FMAC = 1 };

inline Program gemm_f32_kernel(int ahead = GEMM_TMU_AHEAD, int variant = GEMM_LDTMU_ON_LAST_FMAC) {
    if (ahead < 1 || ahead > GEMM_TMU_AHEAD) ahead = GEMM_TMU_AHEAD;
    Program p;
    auto unif_to_a = [&](uint32_t ra) { p.mov(ra, 0, MUX_A, R_UNIFORM); };
    auto unif_to_b = [&](uint32_t rb) { p.mov(rb, 1, MUX_A, R_UNIFORM); };

    // Globals.
    unif_to_a(18);  // tiles
    unif_to_b(0);   // B row stride
    unif_to_b(1);   // k-loop count = Kp / U
    unif_to_b(2);   // VPM vertical setup
    unif_to_b(3);   // VDW stride setup for tiles
    unif_to_b(11);  // done: VPM horizontal setup
    unif_to_b(13);  // done: VDW stride setup
    unif_to_b(12);  // done: VDW setup
    unif_to_b(14);  // done: address
    unif_to_a(19);  // seq
    p.raw(ldi(5, 1, 4));  // rb5 = 4

    const size_t tile_loop = p.here();
    // Tile params, in stream order: nvalid-1, B base, VDW setup, dst, A block, next params.
    unif_to_b(9);
    p.mov(W_R0, 0, MUX_A, R_UNIFORM);
    unif_to_b(8);
    unif_to_b(7);
    unif_to_a(16);
    unif_to_b(6);
    // r2 = B base + min(lane, nvalid-1) * row stride   (rb9 was written 5 instructions ago)
    { Alu a; a.op_add = A_MIN; a.waddr_add = W_R3; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_B;
      a.raddr_a = R_ELEMENT_NUMBER; a.raddr_b = 9; p.alu(a); }
    { Alu a; a.op_mul = M_MUL24; a.waddr_mul = W_R3; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R3; a.mul_b = MUX_B;
      a.raddr_b = 0; p.alu(a); }
    { Alu a; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R0; a.add_b = MUX_R3; p.alu(a); }
    p.mov(W_UNIFORMS_ADDR, 0, MUX_A, 16);  // the k loop reads the packed A block
    for (uint32_t r = 0; r < 16; r++) p.raw(ldi(r, 0, 0));
    p.raw(ldi(W_R1, 0, 0));
    // Prefetch: tmu0_s = r2 ; r2 += 4
    auto tmu_req = [&](uint32_t sig) {
        Alu a; a.sig = sig; a.op_add = A_ADD; a.waddr_add = W_R2; a.cond_add = C_ALWAYS; a.add_a = MUX_R2;
        a.add_b = MUX_B; a.raddr_b = 5;
        a.op_mul = M_V8MIN; a.waddr_mul = W_TMU0_S; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R2; a.mul_b = MUX_R2;
        p.alu(a);
    };
    for (int i = 0; i < ahead; i++) tmu_req(SIG_NONE);
    p.nop(SIG_LDTMU0);                       // r4 = B column 0
    p.mov(W_R3, 0, MUX_B, R_NOP, 1);         // r3 = k-loop count

    const size_t k_loop = p.here();
    auto fmac = [&](uint32_t r, uint32_t sig) {
        Alu a; a.sig = sig;
        a.op_add = A_FADD; a.waddr_add = (r + 15) & 15; a.cond_add = C_ALWAYS;
        a.add_a = MUX_A; a.raddr_a = (r + 15) & 15; a.add_b = MUX_R1;
        a.op_mul = M_FMUL; a.waddr_mul = W_R1; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R4; a.mul_b = MUX_B;
        a.raddr_b = R_UNIFORM;
        p.alu(a);
    };
    for (int u = 0; u < GEMM_U; u++) {
        if (variant == GEMM_LDTMU_ON_LAST_FMAC) {
            // Request column k+ahead first, so even ahead=1 has it in flight before the pop.
            tmu_req(SIG_NONE);
            for (uint32_t r = 0; r < 16; r++) fmac(r, r == 15 ? SIG_LDTMU0 : SIG_NONE);
        } else {
            for (uint32_t r = 0; r < 16; r++) fmac(r, SIG_NONE);
            tmu_req(SIG_NONE);     // request column k+ahead
            p.nop(SIG_LDTMU0);     // pop column k+1 -- never in the instruction that writes a TMU register
        }
    }
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = W_R3; a.cond_add = C_ALWAYS;
      a.add_a = MUX_R3; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, k_loop);
    p.nop(); p.nop(); p.nop();

    // acc15 += the last product; drain the requests that ran past the end.
    { Alu a; a.op_add = A_FADD; a.waddr_add = 15; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.raddr_a = 15;
      a.add_b = MUX_R1; p.alu(a); }
    for (int i = 0; i < ahead - 1; i++) p.nop(SIG_LDTMU0);

    // Store the tile.
    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 2);
    for (uint32_t r = 0; r < 16; r++) p.mov(W_VPM, 0, MUX_A, r);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 3);
    p.mov(W_VPMVCD_SETUP, 1, MUX_B, R_NOP, 8);
    p.mov(W_VPM_ADDR, 1, MUX_B, R_NOP, 7);
    p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    p.mutex_release();

    // Next tile.
    p.mov(W_UNIFORMS_ADDR, 0, MUX_B, R_NOP, 6);
    { Alu a; a.sig = SIG_SMALL_IMM; a.op_add = A_SUB; a.waddr_add = 18; a.cond_add = C_ALWAYS;
      a.add_a = MUX_A; a.raddr_a = 18; a.add_b = MUX_B; a.raddr_b = small_int(1); a.sf = 1; p.alu(a); }
    p.branch(B_ALL_ZC, tile_loop);
    p.nop(); p.nop(); p.nop();

    // Done marker: one VPM row of `seq`, DMA'd to done[q].
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

// Host side: where everything lives in GPU-visible memory, and the uniform streams.
// `Mem` must provide `uint32_t* cpu(uint32_t bus)`; all addresses below are bus addresses.
struct GemmLayout {
    int M, N, K, Mp, Kp, nq;
    uint32_t apack, b, c, done, unif;   // bus addresses
    std::vector<uint32_t> unif_start;  // per launched QPU
    size_t bytes_total;
};

// Bytes each region needs, so the caller can allocate one block. B gets TMU_AHEAD + U - 1 words
// of zero padding after its last row: the k loop reads up to Kp + TMU_AHEAD - 1 into a row.
inline size_t gemm_apack_bytes(int M, int K) {
    int Mp = (M + 15) / 16 * 16, Kp = (K + GEMM_U - 1) / GEMM_U * GEMM_U;
    return size_t(Mp) * Kp * 4;
}

// Repack A [K, M] into k-major 16-row blocks, zero-padded to Mp x Kp.
inline void gemm_pack_a(const float* A, int M, int K, float* out) {
    int Mp = (M + 15) / 16 * 16, Kp = (K + GEMM_U - 1) / GEMM_U * GEMM_U;
    memset(out, 0, size_t(Mp) * Kp * 4);
    for (int mb = 0; mb < Mp / 16; mb++)
        for (int k = 0; k < K; k++)
            for (int r = 0; r < 16; r++) {
                int m = mb * 16 + r;
                if (m < M) out[(size_t(mb) * Kp + k) * 16 + r] = A[size_t(m) * K + k];
            }
}

// Lay out B, scratch C, done words and the uniform streams starting at `base`; returns the layout.
// `cpu(bus)` maps a bus address to a host pointer. B is copied in, C is not cleared.
template <class CpuMap>
GemmLayout gemm_plan(CpuMap cpu, uint32_t base, uint32_t apack_bus, const float* B, int M, int N, int K,
                     int nq_max, uint32_t seq) {
    GemmLayout L{};
    L.M = M; L.N = N; L.K = K;
    L.Mp = (M + 15) / 16 * 16; L.Kp = (K + GEMM_U - 1) / GEMM_U * GEMM_U;
    const int mb = L.Mp / 16, nb = (N + 15) / 16, tiles = mb * nb;
    L.nq = std::min(nq_max, tiles);
    auto align = [](uint32_t v) { return (v + 63) & ~63u; };
    L.apack = apack_bus;
    L.b = align(base);
    const size_t b_words = size_t(N) * K + GEMM_TMU_AHEAD + GEMM_U;
    L.c = align(L.b + uint32_t(b_words * 4));
    L.done = align(L.c + uint32_t(size_t(N) * L.Mp * 4));
    L.unif = align(L.done + 12 * 64);
    memcpy(cpu(L.b), B, size_t(N) * K * 4);
    memset(cpu(L.b + uint32_t(size_t(N) * K * 4)), 0, (GEMM_TMU_AHEAD + GEMM_U) * 4);

    const uint32_t vpm_vsetup = 1u << 12 | 2u << 8;                  // vertical, 32-bit, stride 1, Y=0 X=0
    const uint32_t vdw_stride = 0xc0000000u | uint32_t(L.Mp * 4 - 64);
    const uint32_t done_vpm = 1u << 12 | 1u << 11 | 2u << 8;          // horizontal row 0
    const uint32_t done_vdw = 0x80000000u | 1u << 23 | 16u << 16 | 1u << 14;
    uint32_t at = L.unif;
    for (int q = 0; q < L.nq; q++) {
        std::vector<int> mine;
        for (int t = q; t < tiles; t += L.nq) mine.push_back(t);
        L.unif_start.push_back(at);
        uint32_t* u = cpu(at);
        int i = 0;
        u[i++] = uint32_t(mine.size());
        u[i++] = uint32_t(K * 4);
        u[i++] = uint32_t(L.Kp / GEMM_U);
        u[i++] = vpm_vsetup;
        u[i++] = vdw_stride;
        u[i++] = done_vpm;
        u[i++] = 0xc0000000u;
        u[i++] = done_vdw;
        u[i++] = L.done + q * 64;
        u[i++] = seq;
        for (size_t j = 0; j < mine.size(); j++) {
            const int t = mine[j], tm = t % mb, tn = t / mb;
            const int n0 = tn * 16, nvalid = std::min(16, N - n0);
            u[i++] = uint32_t(nvalid - 1);
            u[i++] = L.b + uint32_t(size_t(n0) * K * 4);
            u[i++] = 0x80000000u | uint32_t(nvalid) << 23 | 16u << 16 | 1u << 14;
            u[i++] = L.c + uint32_t((size_t(n0) * L.Mp + tm * 16) * 4);
            u[i++] = L.apack + uint32_t(size_t(tm) * L.Kp * 16 * 4);
            u[i] = at + uint32_t(i + 1) * 4;  // the next tile's params follow directly
            i++;
        }
        at = align(at + uint32_t(i) * 4);
    }
    L.bytes_total = at - base;
    return L;
}

// Bytes gemm_plan will use beyond `base` (upper bound), for allocation.
inline size_t gemm_plan_bytes(int M, int N, int K, int nq) {
    int Mp = (M + 15) / 16 * 16, Kp = (K + GEMM_U - 1) / GEMM_U * GEMM_U;
    (void) Kp;
    int tiles = Mp / 16 * ((N + 15) / 16);
    return 64 * 8 + (size_t(N) * K + GEMM_TMU_AHEAD + GEMM_U) * 4 + size_t(N) * Mp * 4 + 12 * 64 +
           size_t(nq) * 64 + size_t(tiles) * 6 * 4 + size_t(nq) * 10 * 4;
}

// Copy the padded scratch C into a dense dst [M, N].
inline void gemm_unpack_c(const float* c, int M, int N, int Mp, float* dst) {
    for (int n = 0; n < N; n++) memcpy(dst + size_t(n) * M, c + size_t(n) * Mp, size_t(M) * 4);
}

}  // namespace vc4
