// conv1d_f32_kernel in the QPU emulator, with the host's edge positions: the whole output
// bit-identical to an fmul-then-fadd reference over zero-padded taps, every dst word past T_out
// untouched (each row is its own DMA), done markers, and two legal-but-wrong kernels caught -- a
// subtraction in one accumulator, and the next-channel step swapped for the tap step. P4.46: the
// epilogue (bias, residual, relu, mask, every combination) against relu(bias + conv + residual) * mask
// in the kernel's order, and a sabotage that adds one channel's residual to the next.
//
// P4.45 (ADR-088's "convolutions as GEMM"), P4.46; ctest -R test_vc4_conv1d_emu
#include "qpu_conv1d.h"
#include "qpu_emu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vc4;

enum Sabotage { NONE, FSUB, STEP, RESIDUAL_ROW };

// `staged`: x goes in as ggml-vc4 stages it, [p zeros | x | pr zeros] with the QPUs given p = 0, so
// every position is theirs and the host computes none.
static int run_case(Conv1dShape c, int x_pad, int dst_pad, int nq_max, Conv1dOut out, Sabotage sabotage = NONE,
                    bool staged = false, unsigned epi = 0) {
    std::mt19937 rng(c.IC * 7919 + c.KW * 131 + c.T_in * 7 + c.OC + c.d);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int K = c.IC * c.KW, xr = c.T_in + x_pad, dr = c.T_out + dst_pad;
    EmuMemory mem;
    mem.bus_base = 0x97000000u;
    const uint32_t x0 = mem.bus_base;
    const uint32_t d0 = (x0 + uint32_t(c.IC) * (c.p + std::max(xr, c.T_in + conv1d_pad_right(c))) * 4 + 63) & ~63u;
    const uint32_t w0 = (d0 + uint32_t(c.OC) * dr * 4 + 63) & ~63u;
    const uint32_t discard = (w0 + uint32_t(conv1d_wpack_floats(c.OC, K)) * 4 + 63) & ~63u;
    const uint32_t res0 = discard + 64, mask0 = (res0 + uint32_t(c.OC) * dr * 4 + 63) & ~63u;
    const uint32_t done = (mask0 + uint32_t(dr) * 4 + 63) & ~63u, unif = done + 12 * 64;
    Conv1dShape cq = c;
    const int pr = conv1d_pad_right(c), Tp = c.p + c.T_in + pr;
    if (staged) { cq.T_in = Tp; cq.p = 0; }
    const int xs = staged ? Tp : xr;   // x row length as the QPUs see it
    const Conv1dProblem P{cq, x0, uint32_t(xs) * 4, d0, uint32_t(dr) * 4, w0, discard};
    mem.words.assign((unif - mem.bus_base) / 4 + conv1d_uniform_words(P, nq_max) + 1024, 0xEEEEEEEEu);
    auto cpu = [&](uint32_t bus) { return &mem.at(bus); };
    std::vector<float> X(size_t(c.IC) * xr);
    for (auto & v : X) v = dist(rng);
    float * XM = reinterpret_cast<float *>(cpu(x0));
    for (int ic = 0; ic < c.IC; ic++)
        for (int t = 0; t < xs; t++) {
            const int src = staged ? t - c.p : t;
            XM[size_t(ic) * xs + t] = staged ? (src >= 0 && src < c.T_in ? X[size_t(ic) * xr + src] : 0.f) : X[size_t(ic) * xr + t];
        }
    std::vector<float> W(size_t(c.OC) * K);
    for (auto & v : W) v = dist(rng);
    float * wp = reinterpret_cast<float *>(cpu(w0));
    conv1d_pack_w(W.data(), c.OC, K, wp);
    std::vector<float> bias(c.OC);
    for (auto & v : bias) v = dist(rng);
    float * R = reinterpret_cast<float *>(cpu(res0));
    float * MK = reinterpret_cast<float *>(cpu(mask0));
    for (size_t i = 0; i < size_t(c.OC) * dr; i++) R[i] = dist(rng);
    for (int t = 0; t < dr; t++) MK[t] = t % 3 == 0 ? 0.f : 1.f;   // a time mask, with zeros between
    Conv1dEpi E;
    E.flags = epi; E.bias = bias.data(); E.residual = res0; E.residual_row = uint32_t(dr) * 4; E.mask = mask0;

    const int tail = conv1d_tail(c.IC, c.KW);
    Program prog = conv1d_f32_kernel(c.KW, tail, out, epi);
    if (sabotage != NONE) {
        int seen = 0;
        for (auto & w : prog.code) {
            const uint32_t op_add = (w >> 24) & 31, raddr_b = (w >> 12) & 63, mul = (w >> 29) & 7;
            if (sabotage == FSUB && op_add == A_FADD && raddr_b == R_UNIFORM && ++seen == 5) {
                w = (w & ~(uint64_t(31) << 24)) | uint64_t(A_FSUB) << 24;
                break;
            }
            if (sabotage == STEP && op_add == A_ADD && mul == M_V8MIN && raddr_b == 4) {
                w = (w & ~(uint64_t(63) << 12)) | uint64_t(5) << 12;
                break;
            }
            // the residual added to the accumulator after its own: channel r takes row r + 1's
            const uint32_t waddr_add = (w >> 38) & 63, add_b = (w >> 6) & 7;
            if (sabotage == RESIDUAL_ROW && op_add == A_FADD && add_b == MUX_R4 && waddr_add == 3) {
                w = (w & ~(uint64_t(63) << 38)) | uint64_t(4) << 38;
                w = (w & ~(uint64_t(63) << 18)) | uint64_t(4) << 18;
                break;
            }
        }
    }
    const uint32_t seq = 0x5e900003u;
    std::vector<uint32_t> starts;
    const int nq = conv1d_uniforms(cpu, unif, P, nq_max, done, seq, starts, out, E);
    EmuStats st;
    try {
        for (int q = 0; q < nq; q++) { QpuEmu emu(mem); emu.run(prog.code, starts[q], q, st); }
    } catch (const std::exception & e) {
        printf("IC=%d KW=%d T=%d OC=%d: EMU ERROR %s\n", c.IC, c.KW, c.T_in, c.OC, e.what());
        return 1;
    }
    float * D = reinterpret_cast<float *>(cpu(d0));
    if (!staged) conv1d_edges_host(c, X.data(), xr, wp, D, dr);

    size_t bad = 0, touched = 0;
    for (int oc = 0; oc < c.OC; oc++)
        for (int t = 0; t < dr; t++) {
            const uint32_t raw = mem.at(d0 + uint32_t((size_t(oc) * dr + t) * 4));
            if (t >= c.T_out) { touched += raw != 0xEEEEEEEEu; continue; }
            float acc = epi & EPI_BIAS ? bias[oc] : 0.f;
            for (int ic = 0; ic < c.IC; ic++)
                for (int kx = 0; kx < c.KW; kx++) {
                    const int ti = t * c.s + kx * c.d - c.p;
                    const float v = ti >= 0 && ti < c.T_in ? X[size_t(ic) * xr + ti] : 0.f;
                    acc = acc + W[size_t(oc) * K + ic * c.KW + kx] * v;
                }
            if (epi & EPI_RESIDUAL) acc = acc + R[size_t(oc) * dr + t];
            if (epi & EPI_RELU) acc = std::max(acc, 0.f);
            if (epi & EPI_MASK) acc = acc * MK[t];
            float got; memcpy(&got, &raw, 4);
            bad += got != acc;
        }
    int done_ok = 0;
    for (int q = 0; q < nq; q++) done_ok += mem.at(done + q * 64) == seq;
    int lo, hi;
    conv1d_interior(cq, lo, hi);
    printf("%s%s epi=%x IC=%-3d KW=%-2d s=%d d=%d p=%-2d T=%-4d->%-4d OC=%-3d nq=%-2d [%d,%d): %zu wrong, %zu pad words touched, "
           "done %d/%d, %llu instr, %llu TMU\n", out == CONV1D_OUT_BLOCK ? "block" : "rows ", staged ? "+staged" : "", epi, c.IC, c.KW, c.s, c.d, c.p,
           c.T_in, c.T_out, c.OC, nq, lo, hi, bad,
           touched, done_ok, nq, (unsigned long long) st.instructions, (unsigned long long) st.tmu_requests);
    return (bad || touched || done_ok != nq) ? 1 : 0;
}

static Conv1dShape same(int IC, int KW, int d, int T, int OC) {
    const int p = d * (KW - 1) / 2;
    return {IC, KW, 1, d, p, T, T, OC};
}

int main() {
    int fails = 0;
    for (Conv1dOut out : {CONV1D_OUT_ROWS, CONV1D_OUT_BLOCK}) {
        fails += run_case(same(32, 7, 1, 200, 32), 0, 0, 12, out);    // a VITS vocoder convolution, shorter
        fails += run_case(same(16, 3, 5, 150, 24), 3, 5, 12, out);    // dilated, a partial channel block, padded rows
        fails += run_case(same(8, 11, 3, 120, 16), 0, 0, 7, out);     // KW 11, dilation 3: 15 padded positions a side
        fails += run_case(same(9, 1, 1, 70, 40), 0, 0, 12, out);      // pointwise: 4 channels a loop, peeled tail of 5
        fails += run_case(same(6, 2, 1, 40, 1), 0, 0, 3, out);        // even KW (p rounds down), one output channel
        fails += run_case({12, 4, 2, 1, 1, 101, 50, 20}, 2, 0, 12, out);  // stride 2
        fails += run_case({5, 3, 1, 1, 0, 33, 31, 17}, 0, 1, 12, out);    // no padding: all interior
        fails += run_case(same(16, 3, 5, 150, 24), 0, 5, 12, out, NONE, true);   // staged padded: no host edges
        fails += run_case(same(8, 11, 3, 120, 16), 0, 0, 7, out, NONE, true);
        fails += run_case({12, 4, 2, 1, 1, 101, 50, 20}, 0, 0, 12, out, NONE, true);
        fails += run_case(same(6, 2, 1, 40, 1), 0, 0, 3, out, NONE, true);   // right edge only
        // the epilogue: every combination on a VITS-like layer, then the shapes it meets
        for (unsigned epi = 1; epi < 16; epi++) fails += run_case(same(16, 3, 5, 150, 24), 0, 5, 12, out, NONE, true, epi);
        fails += run_case(same(32, 7, 1, 200, 32), 0, 0, 12, out, NONE, true, EPI_BIAS | EPI_RESIDUAL);  // HiFi-GAN
        fails += run_case(same(9, 1, 1, 70, 40), 0, 3, 12, out, NONE, true, EPI_BIAS | EPI_RELU | EPI_MASK);  // Citrinet
        fails += run_case({12, 4, 2, 1, 1, 101, 50, 20}, 0, 0, 12, out, NONE, true, EPI_BIAS | EPI_MASK);
    }
    printf("sabotage, fsub (must FAIL): ");
    const int s1 = run_case(same(32, 7, 1, 200, 32), 0, 0, 12, CONV1D_OUT_ROWS, FSUB);
    printf("sabotage, channel step (must FAIL): ");
    const int s2 = run_case(same(16, 3, 5, 150, 24), 3, 5, 12, CONV1D_OUT_BLOCK, STEP);
    printf("sabotage, residual row (must FAIL): ");
    const int s3 = run_case(same(32, 7, 1, 200, 32), 0, 0, 12, CONV1D_OUT_BLOCK, RESIDUAL_ROW, true, EPI_BIAS | EPI_RESIDUAL);
    const bool sab = s1 && s2 && s3;
    printf("%s\n", fails == 0 && sab ? "ALL OK" : "FAILURES");
    return fails == 0 && sab ? 0 : 1;
}
