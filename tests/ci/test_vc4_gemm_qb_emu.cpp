// gemm_qb_kernel in the QPU emulator, for ggml's Q8_0 and Q4_0 (one scale per 32 weights). Two
// references:
//   - bit-exact: the kernel's own arithmetic from the measured r4 unpack (u/255 truncated), the packed
//     255*X and each block's starting sum, in its order -- every output identical;
//   - meaning: an f64 product of the DEQUANTIZED weights and the activations -- every output within a
//     bound of the offset sums' rounding.
// Both output layouts (a MUL_MAT's rows of n, a 1x1 convolution's rows of m with X read transposed),
// panel tile orders, padding words untouched, done markers, and three legal-but-wrong kernels caught:
// an fsub in a block sum, byte b read as b + 1, and a block's scale applied to the next block.
//
// P4.46 (ADR-089's "Q4_0 weights in the int8 kernel"); ctest -R test_vc4_gemm_qb_emu
#include "qpu_emu.h"
#include "qpu_gemm_qb.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vc4;

struct BlockQ8 { uint16_t d; int8_t qs[32]; };
struct BlockQ4 { uint16_t d; uint8_t qs[16]; };

static float half_to_float(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) { float f = std::ldexp(float(m), -24); return s ? -f : f; }
    uint32_t b = s | (e + 112) << 23 | m << 13;
    float f; memcpy(&f, &b, 4); return f;
}
static uint16_t float_to_half(float f) {  // normal range only, round to nearest
    uint32_t b; memcpy(&b, &f, 4);
    const uint32_t s = b >> 31; int e = int((b >> 23) & 255) - 112; uint32_t m = (b >> 13) & 1023;
    if ((b >> 12) & 1) m++;
    if (m == 1024) { m = 0; e++; }
    return uint16_t(s << 15 | uint32_t(e) << 10 | m);
}

enum Sabotage { NONE, FSUB, BYTE, SCALE };

// q4: Q4_0 weights, else Q8_0. out: GEMM_OUT_M stores C transposed (dst row m) and packs X from a
// [K][N] source, a 1x1 convolution's [IC][T] input.
static int run_case(int M, int N, int K, int c_pad, int nq_max, bool q4, GemmOut out, int panel = 0,
                    Sabotage sabotage = NONE) {
    std::mt19937 rng(M * 7919 + N * 131 + K * 7 + q4);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int nb = K / 32;
    // dst rows: N rows of M (+pad) words, or M rows of N (+pad) words
    const int rows = out == GEMM_OUT_N ? N : M, cs = (out == GEMM_OUT_N ? M : N) + c_pad;
    std::vector<uint8_t> raw(size_t(M) * nb * (q4 ? sizeof(BlockQ4) : sizeof(BlockQ8)));
    for (int m = 0; m < M; m++)
        for (int j = 0; j < nb; j++) {
            const uint16_t dh = float_to_half(0.002f + 0.01f * std::fabs(d(rng)));  // a scale per block
            if (q4) {
                BlockQ4 & b = reinterpret_cast<BlockQ4 *>(raw.data())[size_t(m) * nb + j];
                b.d = dh;
                for (auto & q : b.qs) q = uint8_t(rng() & 0xff);
            } else {
                BlockQ8 & b = reinterpret_cast<BlockQ8 *>(raw.data())[size_t(m) * nb + j];
                b.d = dh;
                for (auto & q : b.qs) q = int8_t(int(rng() % 255) - 127);
            }
        }
    // X[n][k] lives at x[n*ns + k*ks]
    const size_t ns = out == GEMM_OUT_N ? K : 1, ks = out == GEMM_OUT_N ? 1 : N;
    std::vector<float> X(size_t(N) * K);
    for (auto & v : X) v = d(rng);

    EmuMemory mem;
    mem.bus_base = 0x97000000u;
    const size_t w_bytes = gemm_qb_weight_bytes(M, K), c_bytes = size_t(rows) * cs * 4, x_bytes = gemm_qb_xpack_bytes(N, K);
    const uint32_t w0 = mem.bus_base, c0 = (w0 + uint32_t(w_bytes) + 63) & ~63u, x0 = (c0 + uint32_t(c_bytes) + 63) & ~63u;
    const uint32_t done = (x0 + uint32_t(x_bytes) + 63) & ~63u, unif = done + 12 * 64;
    const GemmQbProblem P{M, N, K, w0, c0, uint32_t(cs * 4), x0};
    mem.words.assign((unif - mem.bus_base) / 4 + gemm_qb_uniform_words({P}, nq_max) + 1024, 0xEEEEEEEEu);
    auto cpu = [&](uint32_t bus) { return &mem.at(bus); };
    uint8_t * wq = reinterpret_cast<uint8_t *>(cpu(w0));
    if (q4) gemm_qb_repack(reinterpret_cast<const BlockQ4 *>(raw.data()), M, K, half_to_float, wq);
    else gemm_qb_repack(reinterpret_cast<const BlockQ8 *>(raw.data()), M, K, half_to_float, wq);
    float * xp = reinterpret_cast<float *>(cpu(x0));
    std::vector<float> xsrc(X.size());
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) xsrc[size_t(n) * ns + size_t(k) * ks] = X[size_t(n) * K + k];
    gemm_qb_pack_x(xsrc.data(), ns, ks, N, K, xp);

    Program prog = gemm_qb_kernel();
    if (sabotage != NONE) {
        int seen = 0;
        for (auto & w : prog.code) {
            const uint32_t op_add = (w >> 24) & 31, raddr_b = (w >> 12) & 63, unpack = (w >> 57) & 7, sig = uint32_t(w >> 60);
            if (sabotage == FSUB && op_add == A_FADD && raddr_b == R_UNIFORM && ++seen == 37) {
                w = (w & ~(uint64_t(31) << 24)) | uint64_t(A_FSUB) << 24;
                break;
            }
            if (sabotage == BYTE && op_add == A_FADD && unpack == 5 && ++seen == 3) {
                w = (w & ~(uint64_t(7) << 57)) | uint64_t(6) << 57;
                break;
            }
            // the flush's pop moved two instructions early: the scale is replaced before the last
            // column reads it, so T_15 takes the next block's first word as its scale (one early is
            // still right: an instruction that pops reads the old r4)
            if (sabotage == SCALE && sig == SIG_LDTMU0 && ((w >> 44) & 1)) {
                w = (w & ~(uint64_t(15) << 60)) | uint64_t(SIG_NONE) << 60;
                uint64_t & prev = (&w)[-2];
                prev = (prev & ~(uint64_t(15) << 60)) | uint64_t(SIG_LDTMU0) << 60;
                break;
            }
        }
    }
    const uint32_t seq = 0x5e900046u;
    std::vector<uint32_t> starts;
    const int nq = gemm_qb_uniforms(cpu, unif, {P}, nq_max, done, seq, starts, out, panel);
    EmuStats st;
    try {
        for (int q = 0; q < nq; q++) { QpuEmu emu(mem); emu.run(prog.code, starts[q], q, st); }
    } catch (const std::exception & e) {
        printf("M=%d N=%d K=%d: EMU ERROR %s\n", M, N, K, e.what());
        return 1;
    }
    const size_t rb = gemm_qb_row_bytes(K);
    size_t bad = 0, far = 0, touched = 0; double worst = 0;
    for (int n = 0; n < N; n++) {
        const float * xb = xp + size_t(n / 16) * nb * 33 * 16 + (n % 16);
        for (int m = 0; m < M; m++) {
            float T = 0.f;
            double exact = 0, mag = 0;
            for (int j = 0; j < nb; j++) {
                const uint8_t * row = wq + size_t(m) * rb + size_t(j) * 36;
                float dj; memcpy(&dj, row + 32, 4);
                const float * xj = xb + size_t(j) * 33 * 16;
                float p = xj[0];
                for (int i = 0; i < 32; i++) {
                    float c; const uint32_t cb = QpuEmu::colour8(row[i]); memcpy(&c, &cb, 4);
                    p = p + c * xj[size_t(i + 1) * 16];
                    const double xv = X[size_t(n) * K + j * 32 + i];
                    exact += double(dj) * (int(row[i]) - 128) * xv;
                    mag += std::fabs(double(dj) * row[i] * xv);
                }
                T = T + p * dj;
            }
            const size_t at = out == GEMM_OUT_N ? size_t(n) * cs + m : size_t(m) * cs + n;
            const uint32_t r = mem.at(c0 + uint32_t(at * 4));
            float got; memcpy(&got, &r, 4);
            bad += got != T;
            const double err = std::fabs(got - exact), bound = 4.0 * K * 1.1920929e-7 * mag;
            far += !(err <= bound);
            worst = std::max(worst, err / std::max(bound, 1e-30));
        }
    }
    for (int r = 0; r < rows; r++)
        for (int c = (out == GEMM_OUT_N ? M : N); c < cs; c++) touched += mem.at(c0 + uint32_t((size_t(r) * cs + c) * 4)) != 0xEEEEEEEEu;
    int done_ok = 0;
    for (int q = 0; q < nq; q++) done_ok += mem.at(done + q * 64) == seq;
    printf("%s %s M=%-4d N=%-4d K=%-4d panel=%d nq=%-2d: %zu not bit-exact, %zu outside the f64 bound (worst %.2f of it), "
           "%zu pad touched, done %d/%d, %llu instr, %llu TMU\n", q4 ? "Q4_0" : "Q8_0", out == GEMM_OUT_N ? "rows n" : "rows m",
           M, N, K, panel, nq, bad, far, worst, touched, done_ok, nq, (unsigned long long) st.instructions,
           (unsigned long long) st.tmu_requests);
    return (bad || far || touched || done_ok != nq) ? 1 : 0;
}

int main() {
    int fails = 0;
    for (bool q4 : {false, true}) {
        fails += run_case(16, 16, 32, 0, 12, q4, GEMM_OUT_N);
        fails += run_case(29, 37, 96, 3, 12, q4, GEMM_OUT_N);      // partial tiles both ways, 3 blocks
        fails += run_case(176, 76, 704, 0, 12, q4, GEMM_OUT_N);    // conformer-ctc's feed-forward
        fails += run_case(176, 76, 704, 0, 12, q4, GEMM_OUT_N, 2); // its tiles in panels of 2 column blocks
        fails += run_case(40, 23, 256, 1, 5, q4, GEMM_OUT_N, 1);   // panels of one: n-major per m sweep
        fails += run_case(48, 1, 64, 0, 1, q4, GEMM_OUT_N);        // N = 1, one QPU
        fails += run_case(20, 18, 4096, 0, 3, q4, GEMM_OUT_N);     // 4608-byte rows, padded (gemm_row_stride)
        fails += run_case(40, 70, 64, 2, 12, q4, GEMM_OUT_M);      // a 1x1 convolution: [IC 64][T 70] -> [OC 40][T]
        fails += run_case(33, 45, 96, 0, 7, q4, GEMM_OUT_M, 1);
    }
    printf("sabotage, fsub (must FAIL): ");
    const int s1 = run_case(29, 37, 96, 3, 12, false, GEMM_OUT_N, 0, FSUB);
    printf("sabotage, byte b -> b+1 (must FAIL): ");
    const int s2 = run_case(29, 37, 96, 3, 12, true, GEMM_OUT_N, 0, BYTE);
    printf("sabotage, scale popped early (must FAIL): ");
    const int s3 = run_case(40, 70, 64, 2, 12, false, GEMM_OUT_M, 0, SCALE);
    const bool ok = fails == 0 && s1 && s2 && s3;
    printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
