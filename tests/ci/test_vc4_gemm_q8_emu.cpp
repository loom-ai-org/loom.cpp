// gemm_q8_kernel in the QPU emulator. Two references:
//   - bit-exact: the kernel's own arithmetic from the measured r4 unpack (u/255 truncated), the
//     packed 255*X and the column sums, in its order -- every output identical;
//   - meaning: an f64 product of the DEQUANTIZED weights and the activations -- every output
//     within a bound of the offset sum's rounding (what the encoding costs, see qpu_gemm_q8.h).
// Plus: a row whose blocks differ in scale is refused by the repack, padding words untouched, done
// markers, and two legal-but-wrong kernels caught (an fsub in an accumulator; byte b read as b + 1).
// P4.47: the repack's inverse gives the blocks back byte for byte.
//
// P4.45 (ADR-088's "Q8_0 / Q4_0 weights in the kernel"); ctest -R test_vc4_gemm_q8_emu
#include "qpu_emu.h"
#include "qpu_gemm_q8.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vc4;

struct BlockQ8 { uint16_t d; int8_t qs[32]; };

static float half_to_float(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t b;
    if (e == 0) { float f = std::ldexp(float(m), -24); return s ? -f : f; }
    b = s | (e + 112) << 23 | m << 13;
    float f; memcpy(&f, &b, 4); return f;
}
static uint16_t float_to_half(float f) {  // normal range only, round to nearest
    uint32_t b; memcpy(&b, &f, 4);
    const uint32_t s = b >> 31; int e = int((b >> 23) & 255) - 112; uint32_t m = (b >> 13) & 1023;
    if ((b >> 12) & 1) m++;
    if (m == 1024) { m = 0; e++; }
    return uint16_t(s << 15 | uint32_t(e) << 10 | m);
}

enum Sabotage { NONE, FSUB, BYTE, RESIDUAL_ROW };

static int run_case(int M, int N, int K, int c_pad, int nq_max, Sabotage sabotage = NONE, GemmOut out = GEMM_OUT_N,
                    int panel = 0, unsigned epi = 0) {
    std::mt19937 rng(M * 7919 + N * 131 + K * 7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::uniform_int_distribution<int> qd(-127, 127);
    const int nb = K / 32, Kp = gemm_q8_kp(K);
    // dst rows: N rows of M (+pad) words, or (a 1x1 convolution) M rows of N (+pad)
    const int rows = out == GEMM_OUT_N ? N : M, cs = (out == GEMM_OUT_N ? M : N) + c_pad;
    std::vector<BlockQ8> W(size_t(M) * nb);
    for (int m = 0; m < M; m++) {
        const uint16_t dh = float_to_half(0.002f + 0.01f * std::fabs(d(rng)));
        for (int j = 0; j < nb; j++) {
            W[size_t(m) * nb + j].d = dh;
            for (auto & q : W[size_t(m) * nb + j].qs) q = int8_t(qd(rng));
        }
    }
    std::vector<float> X(size_t(N) * K);
    for (auto & v : X) v = d(rng);

    EmuMemory mem;
    mem.bus_base = 0x97000000u;
    const size_t w_bytes = gemm_q8_weight_bytes(M, K), c_bytes = size_t(rows) * cs * 4, x_bytes = gemm_q8_xpack_bytes(N, K);
    const uint32_t w0 = mem.bus_base, c0 = (w0 + uint32_t(w_bytes) + 63) & ~63u, x0 = (c0 + uint32_t(c_bytes) + 63) & ~63u;
    const uint32_t b0 = (x0 + uint32_t(x_bytes) + 63) & ~63u, r0 = (b0 + uint32_t(M) * 4 + 63) & ~63u;
    const uint32_t done = (r0 + uint32_t(c_bytes) + 63) & ~63u, unif = done + 12 * 64;
    const uint32_t ws = gemm_q8_stride(K);
    const GemmQ8Problem P{M, N, K, w0, ws, w0 + uint32_t(gemm_q8_scales_at(M, K)), c0, uint32_t(cs * 4), x0};
    mem.words.assign((unif - mem.bus_base) / 4 + gemm_q8_uniform_words({P}, nq_max) + 1024, 0xEEEEEEEEu);
    auto cpu = [&](uint32_t bus) { return &mem.at(bus); };
    uint8_t * wq = reinterpret_cast<uint8_t *>(cpu(w0));
    if (!gemm_q8_repack(W.data(), M, K, half_to_float, wq)) { printf("repack refused a per-channel weight\n"); return 1; }
    float * xp = reinterpret_cast<float *>(cpu(x0));
    if (out == GEMM_OUT_N) {
        gemm_q8_pack_x(X.data(), K, N, K, xp);
    } else {  // X as a convolution's input [K][N]
        std::vector<float> xt(X.size());
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) xt[size_t(k) * N + n] = X[size_t(n) * K + k];
        gemm_q8_pack_x(xt.data(), 1, N, N, K, xp);
    }

    // the epilogue's operands (GEMM_OUT_M): a bias per channel m, a residual laid out as dst, a mask per position n
    float * bias = reinterpret_cast<float *>(cpu(b0));
    float * res = reinterpret_cast<float *>(cpu(r0));
    for (int m = 0; m < M; m++) bias[m] = d(rng);
    for (size_t i = 0; i < c_bytes / 4; i++) res[i] = d(rng);
    std::vector<float> mask(N);
    for (int n = 0; n < N; n++) mask[n] = n % 5 == 0 ? 0.f : 1.f;
    GemmQ8Epi E;
    E.flags = epi; E.bias = b0; E.residual = r0; E.residual_row = uint32_t(cs * 4); E.mask = mask.data();
    Program prog = gemm_q8_kernel(epi);
    if (sabotage != NONE) {
        int seen = 0;
        for (auto & w : prog.code) {
            const uint32_t op_add = (w >> 24) & 31, raddr_b = (w >> 12) & 63, unpack = (w >> 57) & 7;
            if (sabotage == FSUB && op_add == A_FADD && raddr_b == R_UNIFORM && ++seen == 5) {
                w = (w & ~(uint64_t(31) << 24)) | uint64_t(A_FSUB) << 24;
                break;
            }
            if (sabotage == BYTE && op_add == A_FADD && unpack == 5 && ++seen == 3) {
                w = (w & ~(uint64_t(7) << 57)) | uint64_t(6) << 57;
                break;
            }
            // position 3's residual added to position 4's accumulator
            const uint32_t waddr_add = (w >> 38) & 63, add_b = (w >> 6) & 7;
            if (sabotage == RESIDUAL_ROW && op_add == A_FADD && add_b == MUX_R4 && waddr_add == 3) {
                w = (w & ~(uint64_t(63) << 38)) | uint64_t(4) << 38;
                w = (w & ~(uint64_t(63) << 18)) | uint64_t(4) << 18;
                break;
            }
        }
    }
    const uint32_t seq = 0x5e900004u;
    std::vector<uint32_t> starts;
    const int nq = gemm_q8_uniforms(cpu, unif, {P}, nq_max, done, seq, starts, out, panel, E);
    EmuStats st;
    try {
        for (int q = 0; q < nq; q++) { QpuEmu emu(mem); emu.run(prog.code, starts[q], q, st); }
    } catch (const std::exception & e) {
        printf("M=%d N=%d K=%d: EMU ERROR %s\n", M, N, K, e.what());
        return 1;
    }
    const float * scales = reinterpret_cast<const float *>(wq + gemm_q8_scales_at(M, K));
    size_t bad = 0, far = 0, touched = 0; double worst = 0;
    for (int n = 0; n < N; n++) {
        const float * xb = xp + size_t(n / 16) * (Kp + 1) * 16 + (n % 16);
        for (int m = 0; m < M; m++) {
            const uint32_t raw = mem.at(c0 + uint32_t((out == GEMM_OUT_N ? size_t(n) * cs + m : size_t(m) * cs + n) * 4));
            float acc = 0.f;
            double exact = 0, mag = 0;
            for (int k = 0; k < Kp; k++) {
                const uint8_t u = wq[size_t(m) * ws + k];
                float c; const uint32_t cb = QpuEmu::colour8(u); memcpy(&c, &cb, 4);
                acc = acc + c * xb[size_t(k) * 16];
                if (k < K) {
                    const double wv = double(scales[m]) * (int(u) - 128), xv = X[size_t(n) * K + k];
                    exact += wv * xv; mag += std::fabs(double(u) * scales[m] * xv);
                }
            }
            float want = (acc - xb[size_t(Kp) * 16]) * scales[m];
            if (epi & EPI_BIAS) { want = want + bias[m]; exact += bias[m]; }
            if (epi & EPI_RESIDUAL) { want = want + res[size_t(m) * cs + n]; exact += res[size_t(m) * cs + n]; }
            if (epi & EPI_RELU) { want = std::max(want, 0.f); exact = std::max(exact, 0.0); }
            if (epi & EPI_MASK) { want = want * mask[n]; exact *= mask[n]; }
            float got; memcpy(&got, &raw, 4);
            bad += got != want;
            const double err = std::fabs(got - exact), bound = 4.0 * K * 1.1920929e-7 * mag;
            far += !(err <= bound);
            worst = std::max(worst, err / std::max(bound, 1e-30));
        }
    }
    for (int r = 0; r < rows; r++)
        for (int c = (out == GEMM_OUT_N ? M : N); c < cs; c++) touched += mem.at(c0 + uint32_t((size_t(r) * cs + c) * 4)) != 0xEEEEEEEEu;
    int done_ok = 0;
    for (int q = 0; q < nq; q++) done_ok += mem.at(done + q * 64) == seq;
    printf("%s panel=%d epi=%x M=%-4d N=%-4d K=%-4d nq=%-2d: %zu not bit-exact, %zu outside the f64 bound (worst %.2f of it), %zu pad touched, "
           "done %d/%d, %llu instr, %llu TMU\n", out == GEMM_OUT_N ? "rows n" : "rows m", panel, epi, M, N, K, nq, bad, far, worst, touched, done_ok, nq,
           (unsigned long long) st.instructions, (unsigned long long) st.tmu_requests);
    return (bad || far || touched || done_ok != nq) ? 1 : 0;
}

int main() {
    int fails = 0;
    fails += run_case(16, 16, 32, 0, 12);
    fails += run_case(29, 37, 96, 3, 12);      // partial tiles both ways, K a multiple of 32 but not 64
    fails += run_case(288, 64, 288, 0, 12);    // Moonshine tiny's d_model, a shorter encoder
    fails += run_case(40, 23, 1152, 1, 5);     // its FFN K
    fails += run_case(48, 1, 64, 0, 1);        // N = 1, one QPU
    fails += run_case(40, 23, 512, 1, 5);      // 512-byte rows, padded to 576 (gemm_row_stride)
    fails += run_case(288, 64, 288, 0, 12, NONE, GEMM_OUT_N, 2);   // panels of two column blocks
    fails += run_case(256, 551, 256, 0, 12, NONE, GEMM_OUT_M);     // Citrinet-256's pointwise convolution
    fails += run_case(29, 37, 96, 3, 7, NONE, GEMM_OUT_M, 1);      // ... partial tiles, padded rows
    for (unsigned epi = 1; epi < 16; epi++) fails += run_case(29, 37, 96, 3, 7, NONE, GEMM_OUT_M, 0, epi);  // the epilogue
    fails += run_case(256, 551, 256, 0, 12, NONE, GEMM_OUT_M, 0, EPI_BIAS | EPI_RELU | EPI_MASK);         // Citrinet's
    // a row whose blocks have different scales is not per-channel: the repack must refuse it
    std::vector<BlockQ8> two(2);
    two[0].d = float_to_half(0.01f); two[1].d = float_to_half(0.02f);
    std::vector<uint8_t> scratch(gemm_q8_weight_bytes(1, 64));
    const bool refused = !gemm_q8_repack(two.data(), 1, 64, half_to_float, scratch.data());
    printf("mixed-scale row refused: %s\n", refused ? "yes" : "NO");
    // P4.47: gemm_q8_unpack gives back the blocks byte for byte (the CPU's fallback reads them), at a
    // row length the repack pads and one it does not
    bool round_trip = true;
    for (int K : {288, 512}) {
        const int M = 7, nb = K / 32;
        std::mt19937 rr(K);
        std::vector<BlockQ8> src(size_t(M) * nb), back(src.size());
        for (int m = 0; m < M; m++) {
            const uint16_t dh = float_to_half(0.001f * float(m + 1));
            for (int j = 0; j < nb; j++) {
                src[size_t(m) * nb + j].d = dh;
                for (auto & q : src[size_t(m) * nb + j].qs) q = int8_t(int(rr() % 255) - 127);
            }
        }
        std::vector<uint8_t> packed(gemm_q8_weight_bytes(M, K));
        round_trip = round_trip && gemm_q8_repack(src.data(), M, K, half_to_float, packed.data());
        gemm_q8_unpack(packed.data(), M, K, float_to_half, back.data());
        round_trip = round_trip && memcmp(src.data(), back.data(), src.size() * sizeof(BlockQ8)) == 0;
    }
    printf("repack -> unpack identical: %s\n", round_trip ? "yes" : "NO");
    printf("sabotage, fsub (must FAIL): ");
    const int s1 = run_case(288, 64, 288, 0, 12, FSUB);
    printf("sabotage, byte b -> b+1 (must FAIL): ");
    const int s2 = run_case(29, 37, 96, 3, 12, BYTE);
    printf("sabotage, residual position (must FAIL): ");
    const int s3 = run_case(29, 37, 96, 3, 7, RESIDUAL_ROW, GEMM_OUT_M, 0, EPI_RESIDUAL);
    const bool ok = fails == 0 && refused && round_trip && s1 && s2 && s3;
    printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
