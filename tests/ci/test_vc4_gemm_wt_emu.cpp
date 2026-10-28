// gemm_f32_wt_kernel in the QPU emulator: bit-identical to an fmul-then-fadd reference, every byte
// of dst outside the product untouched (it writes straight into dst), done markers, and a legal but
// wrong kernel caught.
//
// P4.44 (ADR-088); ctest -R test_vc4_gemm_wt_emu
#include "qpu_emu.h"
#include "qpu_gemm_wt.h"

#include <cstdio>
#include <random>

using namespace vc4;

// `batches` slices that share K; W rows padded by w_pad floats, dst rows by c_pad floats.
static int run_case(int M, int N, int K, int batches, int w_pad, int c_pad, int nq_max, bool sabotage = false) {
    std::mt19937 rng(M * 7919 + N * 131 + K * 7 + batches);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int ws = K + w_pad, cs = M + c_pad;
    EmuMemory mem;
    mem.bus_base = 0x97000000u;
    const size_t w_bytes = size_t(batches) * M * ws * 4 + gemm_wt_tail_bytes();
    const size_t c_bytes = size_t(batches) * N * cs * 4;
    const size_t x_bytes = size_t(batches) * gemm_wt_xpack_bytes(N, K);
    std::vector<GemmWtProblem> probs;
    const uint32_t w0 = mem.bus_base, c0 = (w0 + uint32_t(w_bytes) + 63) & ~63u, x0 = (c0 + uint32_t(c_bytes) + 63) & ~63u;
    const uint32_t done = (x0 + uint32_t(x_bytes) + 63) & ~63u, unif = done + 12 * 64;
    for (int b = 0; b < batches; b++)
        probs.push_back({M, N, K, w0 + uint32_t(b * M * ws * 4), uint32_t(ws * 4), c0 + uint32_t(b * N * cs * 4),
                         uint32_t(cs * 4), x0 + uint32_t(b * gemm_wt_xpack_bytes(N, K))});
    mem.words.assign((unif - mem.bus_base) / 4 + gemm_wt_uniform_words(probs, nq_max) + 1024, 0xEEEEEEEEu);
    auto cpu = [&](uint32_t bus) { return &mem.at(bus); };
    float* W = reinterpret_cast<float*>(cpu(w0));
    for (size_t i = 0; i < size_t(batches) * M * ws; i++) W[i] = d(rng);
    for (size_t i = 0; i < gemm_wt_tail_bytes() / 4; i++) W[size_t(batches) * M * ws + i] = 0.f;
    std::vector<float> X(size_t(batches) * N * K);
    for (auto& v : X) v = d(rng);
    for (int b = 0; b < batches; b++)
        gemm_wt_pack_x(&X[size_t(b) * N * K], K, N, K, reinterpret_cast<float*>(cpu(probs[b].xpack)));

    Program prog = gemm_f32_wt_kernel();
    if (sabotage) {
        int seen = 0;
        for (auto& w : prog.code)
            if (((w >> 24) & 31) == A_FADD && ((w >> 12) & 63) == R_UNIFORM && ++seen == 5) {
                w = (w & ~(uint64_t(31) << 24)) | uint64_t(A_FSUB) << 24;
                break;
            }
    }
    const uint32_t seq = 0x5e900002u;
    std::vector<uint32_t> starts;
    const int nq = gemm_wt_uniforms(cpu, unif, probs, nq_max, done, seq, starts);
    EmuStats st;
    try {
        for (int q = 0; q < nq; q++) { QpuEmu emu(mem); emu.run(prog.code, starts[q], q, st); }
    } catch (const std::exception& e) {
        printf("M=%d N=%d K=%d b=%d: EMU ERROR %s\n", M, N, K, batches, e.what());
        return 1;
    }
    size_t bad = 0, touched = 0;
    for (int b = 0; b < batches; b++)
        for (int n = 0; n < N; n++)
            for (int m = 0; m < cs; m++) {
                const uint32_t raw = mem.at(probs[b].c + uint32_t((n * cs + m) * 4));
                if (m >= M) { touched += raw != 0xEEEEEEEEu; continue; }
                float acc = 0.f;
                for (int k = 0; k < K; k++) acc = acc + W[size_t(b) * M * ws + size_t(m) * ws + k] * X[size_t(b) * N * K + size_t(n) * K + k];
                float got; memcpy(&got, &raw, 4);
                bad += got != acc;
            }
    int done_ok = 0;
    for (int q = 0; q < nq; q++) done_ok += mem.at(done + q * 64) == seq;
    printf("M=%-4d N=%-5d K=%-4d batches=%d wpad=%d cpad=%d nq=%-2d: %zu wrong, %zu padding words touched, done %d/%d, "
           "%llu instr, %llu TMU\n", M, N, K, batches, w_pad, c_pad, nq, bad, touched, done_ok, nq,
           (unsigned long long)st.instructions, (unsigned long long)st.tmu_requests);
    return (bad || touched || done_ok != nq) ? 1 : 0;
}

int main() {
    int fails = 0;
    fails += run_case(16, 16, 4, 1, 0, 0, 12);
    fails += run_case(1, 20, 7, 1, 0, 0, 12);
    fails += run_case(29, 37, 193, 1, 3, 5, 12);   // partial tiles both ways, padded strides
    fails += run_case(32, 101, 96, 1, 0, 0, 12);
    fails += run_case(192, 1, 192, 1, 0, 0, 12);    // a decode step: N = 1
    fails += run_case(384, 457, 192, 1, 0, 0, 12);
    fails += run_case(40, 23, 64, 3, 1, 2, 5);      // three batch slices in one launch
    fails += run_case(48, 16, 32, 1, 0, 0, 1);
    printf("sabotage (must FAIL): ");
    const int sab = run_case(32, 101, 96, 1, 0, 0, 12, true);
    printf("%s\n", sab ? "failed as it must" : "DID NOT FAIL");
    printf("%s\n", fails == 0 && sab ? "ALL OK" : "FAILURES");
    return fails == 0 && sab ? 0 : 1;
}
