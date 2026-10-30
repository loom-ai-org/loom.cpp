// Runs gemm_f32_kernel in the QPU emulator against a CPU reference that sums k in the same order
// (the QPU has no FMA: fmul then fadd), so the expected result is bit-identical.
//
// P4.44 (ADR-090); ctest -R test_vc4_gemm_emu
#include "qpu_emu.h"
#include "qpu_gemm.h"

#include <cstdio>
#include <random>

using namespace vc4;

static int run_case(int M, int N, int K, int nq_max, bool sabotage = false, int ahead = GEMM_TMU_AHEAD,
                    int variant = GEMM_LDTMU_ON_LAST_FMAC) {
    std::mt19937 rng(M * 1000003 + N * 101 + K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> A(size_t(M) * K), B(size_t(N) * K), ref(size_t(N) * M), out(size_t(N) * M);
    for (auto& v : A) v = d(rng);
    for (auto& v : B) v = d(rng);
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc = acc + A[size_t(m) * K + k] * B[size_t(n) * K + k];
            ref[size_t(n) * M + m] = acc;
        }

    EmuMemory mem;
    mem.bus_base = 0x97000000u;
    const size_t apack_bytes = gemm_apack_bytes(M, K);
    mem.words.assign((apack_bytes + gemm_plan_bytes(M, N, K, nq_max) + 4096) / 4, 0xEEEEEEEEu);
    auto cpu = [&](uint32_t bus) { return &mem.at(bus); };
    gemm_pack_a(A.data(), M, K, reinterpret_cast<float*>(cpu(mem.bus_base)));
    const uint32_t seq = 0x5e900001u;
    GemmLayout L = gemm_plan(cpu, mem.bus_base + uint32_t(apack_bytes), mem.bus_base, B.data(), M, N, K, nq_max, seq);

    Program prog = gemm_f32_kernel(ahead, variant);
    if (sabotage) {  // a LEGAL but wrong kernel: the 5th accumulate of the k loop becomes fsub
        int seen = 0;
        for (auto& w : prog.code)
            if ((w >> 60) == SIG_NONE && ((w >> 24) & 31) == A_FADD && ((w >> 12) & 63) == R_UNIFORM && ++seen == 5) {
                w = (w & ~(uint64_t(31) << 24)) | uint64_t(A_FSUB) << 24;
                break;
            }
    }
    EmuStats st;
    try {
        for (int q = 0; q < L.nq; q++) {
            QpuEmu emu(mem);
            emu.run(prog.code, L.unif_start[q], q, st);
        }
    } catch (const std::exception& e) {
        printf("M=%d N=%d K=%d nq=%d: EMU ERROR %s\n", M, N, K, L.nq, e.what());
        return 1;
    }
    int done_ok = 0;
    for (int q = 0; q < L.nq; q++) done_ok += mem.at(L.done + q * 64) == seq && mem.at(L.done + q * 64 + 60) == seq;
    gemm_unpack_c(reinterpret_cast<const float*>(cpu(L.c)), M, N, L.Mp, out.data());
    size_t bad = 0; double maxd = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (out[i] != ref[i]) bad++;
        maxd = std::max(maxd, double(std::fabs(out[i] - ref[i])));
    }
    // The padding rows of the scratch C (m in [M, Mp)) must hold exact zeros, and nothing past the
    // last valid n row may have been written.
    size_t pad_bad = 0;
    const float* c = reinterpret_cast<const float*>(cpu(L.c));
    for (int n = 0; n < N; n++)
        for (int m = M; m < L.Mp; m++) pad_bad += c[size_t(n) * L.Mp + m] != 0.f;
    const size_t instr_per_qpu_ideal = size_t(L.Kp) * 17 * ((L.Mp / 16) * ((N + 15) / 16)) / L.nq;
    printf("M=%-4d N=%-5d K=%-4d nq=%-2d tiles=%-4d: %zu/%zu wrong (max|d| %.3g), pad %zu, done %d/%d, "
           "%llu instr (k-loop ideal %zu/QPU), %llu TMU, %llu DMA rows, %llu ints\n",
           M, N, K, L.nq, (L.Mp / 16) * ((N + 15) / 16), bad, ref.size(), maxd, pad_bad, done_ok, L.nq,
           (unsigned long long)st.instructions, instr_per_qpu_ideal, (unsigned long long)st.tmu_requests,
           (unsigned long long)st.dma_rows, (unsigned long long)st.host_ints);
    return (bad || pad_bad || done_ok != L.nq || st.host_ints != (uint64_t)L.nq) ? 1 : 0;
}

int main() {
    int fails = 0;
    fails += run_case(16, 16, 4, 12);
    fails += run_case(1, 20, 7, 12);        // M=1 (the VITS Cout=1 conv), K not a multiple of U
    fails += run_case(29, 37, 193, 12);     // partial m-block and partial n-block
    fails += run_case(32, 101, 96, 12);
    fails += run_case(192, 101, 192, 12);   // VITS encoder conv shapes
    fails += run_case(384, 457, 192, 12);
    fails += run_case(64, 300, 448, 5);     // fewer QPUs than... and uneven tile split
    fails += run_case(48, 16, 32, 1);       // one QPU, several tiles
    fails += run_case(192, 101, 576, 12);   // the remaining shapes test_gemm_hw runs
    fails += run_case(128, 3656, 384, 12);
    for (int v = 0; v < 2; v++)
        for (int ahead = 1; ahead <= 4; ahead++) fails += run_case(29, 37, 193, 12, false, ahead, v);
    printf("sabotage (must FAIL): ");
    int sab = run_case(32, 101, 96, 12, true);
    printf("%s\n", sab ? "failed as it must" : "DID NOT FAIL -- the test cannot see a broken kernel");
    printf("%s\n", fails == 0 && sab ? "ALL OK" : "FAILURES");
    return fails == 0 && sab ? 0 : 1;
}
