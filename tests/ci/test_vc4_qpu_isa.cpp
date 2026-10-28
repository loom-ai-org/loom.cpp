// Golden test for qpu_isa.h (P4.44, ADR-088): rebuild qpu_probe.c's programs and compare word for
// word against tests/fixtures/vc4_qpu_golden.hex (qpu_probe --dump, whose encodings were diffed
// against py-videocore's assembler on 2026-10-28: equal except don't-care fields -- py sets sf on add
// ops by default, cond_mul on ldi to a nop mul address, writes nops as `nop r0`, and releases the
// mutex with 0), plus py-videocore's own sema encodings.
#include "qpu_isa.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

using namespace vc4;

static void prologue(Program& p) {
    p.mov(0, 0, MUX_A, R_UNIFORM);  // ra0 <- VPM write setup
    p.mov(2, 0, MUX_A, R_UNIFORM);  // ra2 <- VDW setup
    p.mov(3, 0, MUX_A, R_UNIFORM);  // ra3 <- output bus address
    p.mov(1, 0, MUX_A, R_UNIFORM);  // ra1 <- loop count
}
static void store_tail(Program& p, uint32_t result) {
    p.mutex_acquire();
    p.mov(W_VPMVCD_SETUP, 1, MUX_A, 0);
    p.mov(W_VPM, 0, result);
    p.mov(W_VPMVCD_SETUP, 1, MUX_A, 2);
    p.mov(W_VPM_ADDR, 1, MUX_A, 3);
    p.mov(W_NOP, 0, MUX_B, R_NOP, R_VPM_ST_WAIT);
    p.mutex_release();
    p.host_interrupt();
    p.thread_end();
}
static Program hello() {
    Program p; prologue(p);
    p.raw(ldi(W_R1, 0, 1000));
    Alu a; a.op_add = A_ADD; a.waddr_add = W_R0; a.cond_add = C_ALWAYS; a.add_a = MUX_A; a.add_b = MUX_R1;
    a.raddr_a = R_ELEMENT_NUMBER; p.alu(a);
    store_tail(p, MUX_R0);
    return p;
}
static Program empty() {
    Program p; prologue(p); p.host_interrupt(); p.thread_end(); return p;
}
static Program flops(int K) {
    Program p; prologue(p);
    p.raw(ldi(W_R0, 0, 0)); p.raw(ldi(W_R1, 0, 0x3f800000u));
    p.raw(ldi(W_R2, 0, 0x3f800000u)); p.raw(ldi(W_R3, 0, 0x3f800000u));
    size_t loop = p.here();
    for (int k = 0; k < K; k++) {
        Alu a; a.op_add = A_FADD; a.waddr_add = W_R0; a.cond_add = C_ALWAYS; a.add_a = MUX_R0; a.add_b = MUX_R1;
        a.op_mul = M_FMUL; a.waddr_mul = W_R2; a.cond_mul = C_ALWAYS; a.mul_a = MUX_R2; a.mul_b = MUX_R3;
        p.alu(a);
    }
    Alu d; d.sig = SIG_SMALL_IMM; d.op_add = A_SUB; d.waddr_add = 1; d.cond_add = C_ALWAYS; d.add_a = MUX_A;
    d.add_b = MUX_B; d.raddr_a = 1; d.raddr_b = small_int(1); d.sf = 1; p.alu(d);
    p.branch(B_ALL_ZC, loop);
    p.nop(); p.nop(); p.nop();
    store_tail(p, MUX_R0);
    return p;
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "vc4_qpu_golden.hex");
    std::map<std::string, std::vector<uint64_t>> golden;
    std::string line, cur;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string a; size_t n;
        if (line.size() != 16) { is >> cur >> n; continue; }
        golden[cur].push_back(std::stoull(line, nullptr, 16));
    }
    Program s; s.raw(sema(false, 3)); s.raw(sema(true, 5)); s.raw(sema(true, 15));
    std::map<std::string, Program> mine = {{"hello", hello()}, {"empty", empty()}, {"flops4", flops(4)}, {"sema", s}};
    auto compare = [&](const std::map<std::string, Program>& progs, bool quiet) {
        int bad = 0;
        for (auto& [name, p] : progs) {
            auto& g = golden[name];
            if (g.size() != p.code.size()) {
                if (!quiet) printf("%s: %zu words, golden %zu\n", name.c_str(), p.code.size(), g.size());
                bad++;
                continue;
            }
            int wrong = 0;
            for (size_t i = 0; i < g.size(); i++)
                if (g[i] != p.code[i]) {
                    if (!quiet) printf("%s[%zu]: %016llx golden %016llx\n", name.c_str(), i, (unsigned long long)p.code[i], (unsigned long long)g[i]);
                    wrong++;
                }
            bad += wrong;
            if (!quiet) printf("%s: %zu words %s\n", name.c_str(), g.size(), wrong ? "WRONG" : "OK");
        }
        return bad;
    };
    const int bad = compare(mine, false);
    // Sabotage: one bit the hardware does read (hello's add-ALU opcode, add -> sub) must be caught.
    std::map<std::string, Program> sab = mine;
    sab["hello"].code[5] ^= uint64_t(1) << 24;
    const bool sab_caught = compare(sab, true) != 0;
    printf("sabotage (must FAIL): %s\n", sab_caught ? "failed as it must" : "DID NOT FAIL");
    printf("%s\n", bad == 0 && sab_caught ? "ALL OK" : "FAILURES");
    return bad == 0 && sab_caught ? 0 : 1;
}
