// VideoCore IV QPU instruction encoder (Pi Zero / Pi 1-3 GPU).
//
// Field layout from the VideoCore IV 3D Architecture Reference Guide (Broadcom, 2013), section 3.
// Golden-tested against py-videocore's assembler (MIT): every program word matches it except in
// fields the hardware ignores (see test_qpu_isa.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vc4 {

// Signal (bits 60-63).
enum Sig : uint32_t {
    SIG_BKPT = 0, SIG_NONE = 1, SIG_THRSW = 2, SIG_END = 3, SIG_SBWAIT = 4, SIG_SBDONE = 5,
    SIG_LTHRSW = 6, SIG_LOADCV = 7, SIG_LOADC = 8, SIG_LDCEND = 9, SIG_LDTMU0 = 10, SIG_LDTMU1 = 11,
    SIG_LOADAM = 12, SIG_SMALL_IMM = 13, SIG_LOAD = 14, SIG_BRANCH = 15,
};

// Add-ALU opcodes (bits 24-28).
enum AddOp : uint32_t {
    A_NOP = 0, A_FADD = 1, A_FSUB = 2, A_FMIN = 3, A_FMAX = 4, A_FMINABS = 5, A_FMAXABS = 6,
    A_FTOI = 7, A_ITOF = 8, A_ADD = 12, A_SUB = 13, A_SHR = 14, A_ASR = 15, A_ROR = 16, A_SHL = 17,
    A_MIN = 18, A_MAX = 19, A_AND = 20, A_OR = 21, A_XOR = 22, A_NOT = 23, A_CLZ = 24, A_V8ADDS = 30,
    A_V8SUBS = 31,
};

// Mul-ALU opcodes (bits 29-31).
enum MulOp : uint32_t {
    M_NOP = 0, M_FMUL = 1, M_MUL24 = 2, M_V8MULD = 3, M_V8MIN = 4, M_V8MAX = 5, M_V8ADDS = 6,
    M_V8SUBS = 7,
};

// ALU input mux: an accumulator, or the value read from regfile A / B this instruction.
enum Mux : uint32_t { MUX_R0 = 0, MUX_R1, MUX_R2, MUX_R3, MUX_R4, MUX_R5, MUX_A, MUX_B };

// Condition codes (cond_add / cond_mul, and branch conditions are a separate set).
enum Cond : uint32_t { C_NEVER = 0, C_ALWAYS = 1, C_ZS = 2, C_ZC = 3, C_NS = 4, C_NC = 5, C_CS = 6, C_CC = 7 };
enum BranchCond : uint32_t {
    B_ALL_ZS = 0, B_ALL_ZC = 1, B_ANY_ZS = 2, B_ANY_ZC = 3, B_ALL_NS = 4, B_ALL_NC = 5,
    B_ANY_NS = 6, B_ANY_NC = 7, B_ALL_CS = 8, B_ALL_CC = 9, B_ANY_CS = 10, B_ANY_CC = 11, B_ALWAYS = 15,
};

// Write addresses 32-63 (I/O; which file A/B a write lands in is the add/mul `ws` swap).
// Regfile A: 0-31 = ra0..ra31; regfile B: 0-31 = rb0..rb31.
enum WAddr : uint32_t {
    W_R0 = 32, W_R1 = 33, W_R2 = 34, W_R3 = 35, W_TMU_NOSWAP = 36, W_R5 = 37, W_HOST_INT = 38,
    W_NOP = 39, W_UNIFORMS_ADDR = 40, W_QUAD_X = 41, W_MS_FLAGS = 42, W_TLB_STENCIL = 43,
    W_TLB_Z = 44, W_TLB_COLOUR_MS = 45, W_TLB_COLOUR_ALL = 46, W_TLB_ALPHA_MASK = 47,
    W_VPM = 48,
    W_VPMVCD_SETUP = 49,   // regfile A: vpmvcd_rd_setup; regfile B: vpmvcd_wr_setup
    W_VPM_ADDR = 50,       // regfile A: vpm_ld_addr;     regfile B: vpm_st_addr
    W_MUTEX = 51, W_SFU_RECIP = 52, W_SFU_RECIPSQRT = 53, W_SFU_EXP = 54, W_SFU_LOG = 55,
    W_TMU0_S = 56, W_TMU0_T = 57, W_TMU0_R = 58, W_TMU0_B = 59,
    W_TMU1_S = 60, W_TMU1_T = 61, W_TMU1_R = 62, W_TMU1_B = 63,
};

// Read addresses 32-63.
enum RAddr : uint32_t {
    R_UNIFORM = 32, R_VARYING = 35, R_ELEMENT_NUMBER = 38 /* A */, R_QPU_NUMBER = 38 /* B */,
    R_NOP = 39, R_X_PIXEL = 41, R_Y_PIXEL = 41, R_MS_FLAGS = 42, R_VPM = 48,
    R_VPM_LD_BUSY = 49 /* A */, R_VPM_ST_BUSY = 49 /* B */,
    R_VPM_LD_WAIT = 50 /* A */, R_VPM_ST_WAIT = 50 /* B */, R_MUTEX = 51,
};

// One ALU instruction: both pipes, explicit everything.
struct Alu {
    uint32_t sig = SIG_NONE;
    uint32_t op_add = A_NOP, waddr_add = W_NOP, add_a = MUX_R0, add_b = MUX_R0, cond_add = C_NEVER;
    uint32_t op_mul = M_NOP, waddr_mul = W_NOP, mul_a = MUX_R0, mul_b = MUX_R0, cond_mul = C_NEVER;
    uint32_t raddr_a = R_NOP, raddr_b = R_NOP;  // raddr_b is the small immediate under SIG_SMALL_IMM
    uint32_t ws = 0, sf = 0, pm = 0, unpack = 0, pack = 0;

    uint64_t encode() const {
        return uint64_t(mul_b) | uint64_t(mul_a) << 3 | uint64_t(add_b) << 6 | uint64_t(add_a) << 9 |
               uint64_t(raddr_b) << 12 | uint64_t(raddr_a) << 18 | uint64_t(op_add) << 24 |
               uint64_t(op_mul) << 29 | uint64_t(waddr_mul) << 32 | uint64_t(waddr_add) << 38 |
               uint64_t(ws) << 44 | uint64_t(sf) << 45 | uint64_t(cond_mul) << 46 |
               uint64_t(cond_add) << 49 | uint64_t(pack) << 52 | uint64_t(pm) << 56 |
               uint64_t(unpack) << 57 | uint64_t(sig) << 60;
    }
};

// Load immediate: a 32-bit value to the add and/or mul write address.
inline uint64_t ldi(uint32_t waddr_add, uint32_t ws, uint32_t imm, uint32_t waddr_mul = W_NOP, bool sf = false) {
    return uint64_t(imm) | uint64_t(waddr_mul) << 32 | uint64_t(waddr_add) << 38 | uint64_t(ws) << 44 |
           uint64_t(sf) << 45 | uint64_t(waddr_mul != W_NOP ? C_ALWAYS : C_NEVER) << 46 |
           uint64_t(C_ALWAYS) << 49 | uint64_t(SIG_LOAD) << 60;
}

// Relative branch. `offset_bytes` counts from the instruction after the three delay slots
// (i.e. from branch address + 32). `raddr_a`/`reg` add a register to the target (unused here).
inline uint64_t brr(uint32_t cond, int32_t offset_bytes, uint32_t waddr_link = W_NOP) {
    return uint64_t(uint32_t(offset_bytes)) | uint64_t(W_NOP) << 32 | uint64_t(waddr_link) << 38 |
           uint64_t(1) << 51 /* rel */ | uint64_t(cond) << 52 | uint64_t(SIG_BRANCH) << 60;
}

// Semaphore increment/decrement (load-immediate form with the semaphore bit set).
inline uint64_t sema(bool down, uint32_t id) {
    return uint64_t(uint32_t(down) << 4 | (id & 15)) | uint64_t(W_NOP) << 32 | uint64_t(W_NOP) << 38 |
           uint64_t(C_ALWAYS) << 46 | uint64_t(C_ALWAYS) << 49 | uint64_t(4) << 57 /* ldi type: semaphore */ | uint64_t(SIG_LOAD) << 60;
}

// Small immediates (raddr_b under SIG_SMALL_IMM): 0..15 -> 0..15, 16..31 -> -16..-1,
// 32..39 -> 1.0f * 2^(0..7), 40..47 -> 2^(-8..-1).
inline uint32_t small_int(int v) { return v >= 0 ? uint32_t(v) : uint32_t(32 + v); }

// Program builder with the conventions the backend's kernels use.
struct Program {
    std::vector<uint64_t> code;

    size_t here() const { return code.size(); }
    void raw(uint64_t w) { code.push_back(w); }
    void alu(const Alu& a) { code.push_back(a.encode()); }

    void nop(uint32_t sig = SIG_NONE) { Alu a; a.sig = sig; alu(a); }

    // dst <- src through the add pipe (`or dst, src, src`). `src` is an accumulator mux, or
    // MUX_A/MUX_B with the register in raddr_a/raddr_b. `ws` = 1 writes regfile B / B-side I/O.
    void mov(uint32_t waddr, uint32_t ws, uint32_t mux, uint32_t raddr_a = R_NOP, uint32_t raddr_b = R_NOP,
             uint32_t sig = SIG_NONE) {
        Alu a; a.sig = sig; a.op_add = A_OR; a.waddr_add = waddr; a.ws = ws; a.add_a = mux; a.add_b = mux;
        a.cond_add = C_ALWAYS; a.raddr_a = raddr_a; a.raddr_b = raddr_b;
        alu(a);
    }
    // Small-immediate value through the add pipe: or dst, imm, imm.
    void mov_imm(uint32_t waddr, uint32_t ws, uint32_t small_imm) {
        mov(waddr, ws, MUX_B, R_NOP, small_imm, SIG_SMALL_IMM);
    }

    void mutex_acquire() { mov(W_NOP, 0, MUX_A, R_MUTEX); }
    void mutex_release() { mov(W_MUTEX, 0, MUX_R0); }
    void host_interrupt() { mov_imm(W_HOST_INT, 0, small_int(1)); }
    // thread end + the two delay slots it requires.
    void thread_end() { nop(SIG_END); nop(); nop(); }

    // Branch to instruction index `target`; caller fills the three delay slots.
    void branch(uint32_t cond, size_t target) {
        size_t at = here();
        raw(brr(cond, int32_t((int64_t(target) - int64_t(at + 4)) * 8)));
    }
};

}  // namespace vc4
