// Host-side emulator for the subset of the VideoCore IV QPU ISA that ggml-vc4's kernels use.
//
// Why it exists: on the Pi Zero a QPU program that hangs (an unbalanced TMU queue, a bad address, a
// branch that never exits) leaves the QPUs stuck, and the only recovery found so far is a reboot.
// So every kernel runs here first, against a CPU reference, and the emulator also REFUSES the
// hazards the reference guide warns about rather than modelling them:
//   - reading a regfile A/B location in the instruction right after it was written,
//   - a uniform read less than 3 instructions after a write to uniforms_address,
//   - ldtmu with an empty queue, more than TMU_DEPTH outstanding requests, or thread end with any,
//   - a VPM / VPM DMA setup, VPM write or DMA store outside the mutex (the VPM and its DMA engine
//     are shared by every QPU, and the emulator runs them one at a time, so it cannot see a race;
//     it refuses what could be one instead), and a mutex acquired twice or released unheld,
//   - mul24 operands wider than 24 bits, any pack, any regfile-A unpack, conditions other than
//     always/never. The one unpack it models is r4's 8-bit one (pm = 1, unpack 4..7 = byte a..d),
//     as measured on the Pi Zero 2026-10-28 (probe_unpack): every op reading r4 gets the byte u as
//     a float u/255 TRUNCATED to 24 bits (exact for 0 and 255, one ulp under the nearest float for
//     the other 254), the integer ops too.
// Anything it does not implement is an error, never a silent no-op.
#pragma once

#include "qpu_isa.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

namespace vc4 {

struct EmuMemory {
    uint32_t bus_base = 0;           // bus address of words[0]
    std::vector<uint32_t> words;
    uint32_t& at(uint32_t bus) {
        if (bus % 4 || bus < bus_base || (bus - bus_base) / 4 >= words.size())
            throw std::runtime_error("emu: bad memory address 0x" + hex(bus));
        return words[(bus - bus_base) / 4];
    }
    static std::string hex(uint32_t v) { char b[16]; snprintf(b, sizeof b, "%08x", v); return b; }
};

struct EmuStats { uint64_t instructions = 0, tmu_requests = 0, dma_rows = 0, host_ints = 0; };

class QpuEmu {
public:
    static constexpr int TMU_DEPTH = 4;
    using Vec = std::array<uint32_t, 16>;

    QpuEmu(EmuMemory& mem) : mem_(mem) {}

    // Reading r4 in the instruction that signals ldtmu sees the OLD r4 (the pop lands at the end):
    // confirmed on the Pi Zero 2026-10-28 by the exact B=k probe at TMU depths 1..4.
    bool allow_r4_read_with_ldtmu = true;

    // Run one QPU program to thread end. `qpu` is the value qpu_number would read.
    void run(const std::vector<uint64_t>& code, uint32_t uniforms_bus, uint32_t qpu, EmuStats& st,
             uint64_t max_instructions = 2000000000ull) {
        ra_.fill({}); rb_.fill({}); for (auto& r : acc_) r.fill(0);
        tmu_.clear(); unif_ = uniforms_bus; qpu_ = qpu; mutex_ = false;
        zf_.fill(false); nf_.fill(false);
        written_a_prev_ = written_b_prev_ = -1;
        last_unif_write_ = -100;
        int64_t pc = 0, branch_target = -1, branch_delay = 0, end_delay = -1;
        for (uint64_t n = 0;; n++) {
            if (n >= max_instructions) throw err(pc, "instruction budget exhausted (infinite loop?)");
            if (pc < 0 || pc >= (int64_t)code.size()) throw err(pc, "pc out of program");
            step(code[pc], pc, n, st, branch_target, branch_delay, end_delay);
            st.instructions++;
            if (end_delay == 0) break;
            if (end_delay > 0) end_delay--;
            if (branch_delay > 0 && --branch_delay == 0) { pc = branch_target; continue; }
            pc++;
        }
        if (!tmu_.empty()) throw std::runtime_error("emu: thread end with outstanding TMU requests");
        if (mutex_) throw std::runtime_error("emu: thread end holding the mutex");
    }

    // VPM: 64 rows x 16 words is plenty for the kernels (the hardware has more).
    uint32_t vpm[64][16] = {};

private:
    EmuMemory& mem_;
    std::array<Vec, 32> ra_{}, rb_{};
    std::array<Vec, 6> acc_{};
    std::array<bool, 16> zf_{}, nf_{};
    std::deque<Vec> tmu_;
    bool mutex_ = false;
    void need_mutex(int64_t pc, const char* what) const {
        if (!mutex_) throw err(pc, std::string(what) + " outside the VPM mutex");
    }
    uint32_t unif_ = 0, qpu_ = 0;
    int written_a_prev_, written_b_prev_;  // regfile index written by the previous instruction
    int64_t last_unif_write_;
    // VPM write setup / VDW state
    uint32_t vpm_w_addr_ = 0, vpm_w_stride_ = 1; bool vpm_w_horiz_ = true;
    uint32_t vdw_units_ = 0, vdw_depth_ = 0, vdw_y_ = 0, vdw_x_ = 0, vdw_stride_ = 0; bool vdw_horiz_ = true;

    std::runtime_error err(int64_t pc, const std::string& m) const {
        return std::runtime_error("emu @" + std::to_string(pc) + ": " + m);
    }
    static float f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
    static uint32_t u(float x) { uint32_t v; memcpy(&v, &x, 4); return v; }

    Vec read_regfile(bool is_a, uint32_t addr, int64_t pc, int64_t n, bool& unif_read) {
        Vec v{};
        if (addr < 32) {
            if ((is_a ? written_a_prev_ : written_b_prev_) == (int)addr)
                throw err(pc, std::string("regfile ") + (is_a ? "A" : "B") + std::to_string(addr) +
                              " read right after its write");
            return is_a ? ra_[addr] : rb_[addr];
        }
        switch (addr) {
            case R_UNIFORM: {
                if (n - last_unif_write_ < 3) throw err(pc, "uniform read too soon after uniforms_address write");
                if (unif_read) throw err(pc, "two uniform reads in one instruction");
                unif_read = true;
                uint32_t w = mem_.at(unif_); unif_ += 4; v.fill(w); return v;
            }
            case 38: if (is_a) { for (int l = 0; l < 16; l++) v[l] = l; } else v.fill(qpu_); return v;
            case R_NOP: return v;
            case 50: return v;  // vpm_ld_wait / vpm_st_wait: DMA is synchronous here
            case R_MUTEX:
                if (mutex_) throw err(pc, "mutex acquired twice");
                mutex_ = true; return v;
            default: throw err(pc, "unsupported read address " + std::to_string(addr));
        }
    }

public:
    // r4's 8-bit colour unpack: u / 255 truncated toward zero to a float.
    static uint32_t colour8(uint32_t byte) {
        const double v = byte / 255.0;
        float x = float(v);
        if (double(x) > v) x = std::nextafter(x, 0.f);
        return u(x);
    }
private:
    static Vec small_imm(uint32_t s, int64_t pc, const QpuEmu* self) {
        Vec v{};
        if (s < 16) v.fill(s);
        else if (s < 32) v.fill(uint32_t(int32_t(s) - 32));
        else if (s < 40) v.fill(u(float(1u << (s - 32))));
        else if (s < 48) v.fill(u(1.0f / float(1u << (48 - s))));
        else throw self->err(pc, "vector-rotate small immediates not emulated");
        return v;
    }

    void write(uint32_t waddr, bool to_a, const Vec& v, int64_t pc, int64_t n, EmuStats& st, int& wa, int& wb) {
        if (waddr < 32) { (to_a ? ra_ : rb_)[waddr] = v; (to_a ? wa : wb) = (int)waddr; return; }
        switch (waddr) {
            case W_R0: case W_R1: case W_R2: case W_R3: acc_[waddr - 32] = v; return;
            case W_NOP: return;
            case W_HOST_INT: st.host_ints++; return;
            case W_UNIFORMS_ADDR:
                if (!to_a) throw err(pc, "uniforms_address is regfile-A only");
                unif_ = v[0]; last_unif_write_ = n; return;
            case W_VPM: need_mutex(pc, "VPM write"); vpm_write(v, pc); return;
            case W_VPMVCD_SETUP:
                if (to_a) throw err(pc, "vpmvcd_rd_setup (VPM reads/VDR) not emulated");
                need_mutex(pc, "VPM / VDW setup"); setup(v[0], pc); return;
            case W_VPM_ADDR:
                if (to_a) throw err(pc, "vpm_ld_addr (DMA load) not emulated");
                need_mutex(pc, "VDW store"); dma_store(v[0], pc, st); return;
            case W_MUTEX:
                if (!mutex_) throw err(pc, "mutex released without being held");
                mutex_ = false; return;
            case W_TMU0_S: {
                if ((int)tmu_.size() >= TMU_DEPTH) throw err(pc, "TMU queue overflow");
                Vec d{}; for (int l = 0; l < 16; l++) d[l] = mem_.at(v[l]);
                tmu_.push_back(d); st.tmu_requests++; return;
            }
            default: throw err(pc, "unsupported write address " + std::to_string(waddr));
        }
    }

    void vpm_write(const Vec& v, int64_t pc) {
        if (vpm_w_horiz_) {
            uint32_t y = vpm_w_addr_ & 0x3f;
            for (int l = 0; l < 16; l++) vpm[y][l] = v[l];
        } else {
            uint32_t x = vpm_w_addr_ & 15, y = (vpm_w_addr_ >> 4) * 16;
            if (y + 16 > 64) throw err(pc, "vertical VPM write out of emulated VPM");
            for (int l = 0; l < 16; l++) vpm[y + l][x] = v[l];
        }
        vpm_w_addr_ += vpm_w_stride_;
    }

    void setup(uint32_t w, int64_t pc) {
        switch (w >> 30) {
            case 0:  // VPM generic write setup
                if (((w >> 8) & 3) != 2) throw err(pc, "only 32-bit VPM writes emulated");
                if (w & (1u << 10)) throw err(pc, "laned VPM writes not emulated");
                vpm_w_stride_ = (w >> 12) & 0x3f; vpm_w_horiz_ = (w >> 11) & 1; vpm_w_addr_ = w & 0xff;
                return;
            case 2:  // VDW setup
                if (w & 7) throw err(pc, "only 32-bit VDW emulated");
                if (w & (1u << 15)) throw err(pc, "laned VDW not emulated");
                vdw_units_ = (w >> 23) & 0x7f; vdw_depth_ = (w >> 16) & 0x7f; vdw_horiz_ = (w >> 14) & 1;
                if (vdw_units_ == 0) vdw_units_ = 128;
                if (vdw_depth_ == 0) vdw_depth_ = 128;
                vdw_y_ = (w >> 7) & 0x7f; vdw_x_ = (w >> 3) & 15;
                return;
            case 3:  // VDW stride setup
                if (w & (1u << 16)) throw err(pc, "VDW block mode not emulated");
                vdw_stride_ = w & 0x1fff; return;
            default: throw err(pc, "VDR setup not emulated");
        }
    }

    void dma_store(uint32_t addr, int64_t pc, EmuStats& st) {
        if (!vdw_horiz_) throw err(pc, "vertical VDW not emulated");
        for (uint32_t r = 0; r < vdw_units_; r++) {
            if (vdw_y_ + r >= 64 || vdw_x_ + vdw_depth_ > 16) throw err(pc, "VDW outside emulated VPM");
            for (uint32_t c = 0; c < vdw_depth_; c++) mem_.at(addr + c * 4) = vpm[vdw_y_ + r][vdw_x_ + c];
            addr += vdw_depth_ * 4 + vdw_stride_;
            st.dma_rows++;
        }
    }

    Vec add_op(uint32_t op, const Vec& a, const Vec& b, int64_t pc) {
        Vec r{};
        for (int l = 0; l < 16; l++) {
            switch (op) {
                case A_FADD: r[l] = u(f(a[l]) + f(b[l])); break;
                case A_FSUB: r[l] = u(f(a[l]) - f(b[l])); break;
                case A_FMAX: r[l] = u(std::max(f(a[l]), f(b[l]))); break;
                case A_ADD:  r[l] = a[l] + b[l]; break;
                case A_SUB:  r[l] = a[l] - b[l]; break;
                case A_MIN:  r[l] = uint32_t(std::min(int32_t(a[l]), int32_t(b[l]))); break;
                case A_MAX:  r[l] = uint32_t(std::max(int32_t(a[l]), int32_t(b[l]))); break;
                case A_SHL:  r[l] = a[l] << (b[l] & 31); break;
                case A_SHR:  r[l] = a[l] >> (b[l] & 31); break;
                case A_AND:  r[l] = a[l] & b[l]; break;
                case A_OR:   r[l] = a[l] | b[l]; break;
                default: throw err(pc, "unsupported add op " + std::to_string(op));
            }
        }
        return r;
    }
    Vec mul_op(uint32_t op, const Vec& a, const Vec& b, int64_t pc) {
        Vec r{};
        for (int l = 0; l < 16; l++) {
            switch (op) {
                case M_FMUL: r[l] = u(f(a[l]) * f(b[l])); break;
                case M_MUL24:
                    if ((a[l] >> 24) || (b[l] >> 24)) throw err(pc, "mul24 operand wider than 24 bits");
                    r[l] = a[l] * b[l]; break;
                case M_V8MIN:
                    if (a[l] != b[l]) throw err(pc, "v8min used only as a move here");
                    r[l] = a[l]; break;
                default: throw err(pc, "unsupported mul op " + std::to_string(op));
            }
        }
        return r;
    }

    void step(uint64_t w, int64_t pc, int64_t n, EmuStats& st, int64_t& branch_target, int64_t& branch_delay,
              int64_t& end_delay) {
        const uint32_t sig = uint32_t(w >> 60);
        const uint32_t waddr_add = (w >> 38) & 63, waddr_mul = (w >> 32) & 63;
        const bool ws = (w >> 44) & 1, sf = (w >> 45) & 1;
        const uint32_t cond_mul = (w >> 46) & 7, cond_add = (w >> 49) & 7;
        int wa = -1, wb = -1;
        auto check_cond = [&](uint32_t c) {
            if (c != C_ALWAYS && c != C_NEVER) throw err(pc, "conditional writes not emulated");
        };

        if (sig == SIG_BRANCH) {
            if (branch_delay > 0) throw err(pc, "branch in a branch delay slot");
            const uint32_t cond = (w >> 52) & 15;
            if (!((w >> 51) & 1)) throw err(pc, "absolute branch not emulated");
            if ((w >> 50) & 1) throw err(pc, "register-relative branch not emulated");
            bool all_zc = true, all_zs = true;
            for (int l = 0; l < 16; l++) { all_zc &= !zf_[l]; all_zs &= zf_[l]; }
            bool take = cond == B_ALWAYS ? true : cond == B_ALL_ZC ? all_zc : cond == B_ALL_ZS ? all_zs :
                        throw err(pc, "branch condition not emulated");
            if (waddr_add != W_NOP || waddr_mul != W_NOP) throw err(pc, "branch link not emulated");
            if (take) { branch_target = pc + 4 + int32_t(uint32_t(w)) / 8; branch_delay = 4; }
            written_a_prev_ = written_b_prev_ = -1;
            return;
        }
        if (sig == SIG_LOAD) {
            if ((w >> 57) & 7) throw err(pc, "semaphore / per-element load immediates not emulated");
            check_cond(cond_add); check_cond(cond_mul);
            Vec v; v.fill(uint32_t(w));
            if (cond_add == C_ALWAYS) write(waddr_add, !ws, v, pc, n, st, wa, wb);
            if (cond_mul == C_ALWAYS) write(waddr_mul, ws, v, pc, n, st, wa, wb);
            if (sf) for (int l = 0; l < 16; l++) { zf_[l] = v[l] == 0; nf_[l] = int32_t(v[l]) < 0; }
            written_a_prev_ = wa; written_b_prev_ = wb;
            return;
        }
        const uint32_t pack = (w >> 52) & 15, pm = (w >> 56) & 1, unpack = (w >> 57) & 7;
        if (pack) throw err(pc, "pack not emulated");
        if (unpack && !pm) throw err(pc, "regfile-A unpack not emulated");
        if (pm && unpack && unpack < 4) throw err(pc, "r4 unpack modes 1..3 not emulated");
        if (sig != SIG_NONE && sig != SIG_END && sig != SIG_LDTMU0 && sig != SIG_SMALL_IMM)
            throw err(pc, "signal " + std::to_string(sig) + " not emulated");
        check_cond(cond_add); check_cond(cond_mul);

        const uint32_t op_add = (w >> 24) & 31, op_mul = (w >> 29) & 7;
        const uint32_t raddr_a = (w >> 18) & 63, raddr_b = (w >> 12) & 63;
        const uint32_t add_a = (w >> 9) & 7, add_b = (w >> 6) & 7, mul_a = (w >> 3) & 7, mul_b = w & 7;
        auto uses = [&](uint32_t mux) {
            return (op_add != A_NOP && (add_a == mux || add_b == mux)) ||
                   (op_mul != M_NOP && (mul_a == mux || mul_b == mux));
        };
        bool unif_read = false;
        Vec va{}, vb{};
        if (uses(MUX_A)) va = read_regfile(true, raddr_a, pc, n, unif_read);
        else if (raddr_a == R_UNIFORM || raddr_a == R_MUTEX || raddr_a == 50) va = read_regfile(true, raddr_a, pc, n, unif_read);
        if (uses(MUX_B)) vb = sig == SIG_SMALL_IMM ? small_imm(raddr_b, pc, this) : read_regfile(false, raddr_b, pc, n, unif_read);
        else if (sig != SIG_SMALL_IMM && (raddr_b == R_UNIFORM || raddr_b == R_MUTEX || raddr_b == 50))
            vb = read_regfile(false, raddr_b, pc, n, unif_read);
        if (uses(MUX_R5)) throw err(pc, "r5 not emulated");
        Vec r4u{};
        if (pm && unpack) {
            for (int l = 0; l < 16; l++) r4u[l] = colour8(acc_[4][l] >> (8 * (unpack - 4)) & 0xff);
        }
        auto mux = [&](uint32_t m) -> const Vec& {
            switch (m) { case MUX_A: return va; case MUX_B: return vb; case MUX_R4: return pm && unpack ? r4u : acc_[4];
                         default: return acc_[m]; }
        };
        Vec ra = op_add != A_NOP ? add_op(op_add, mux(add_a), mux(add_b), pc) : Vec{};
        Vec rm = op_mul != M_NOP ? mul_op(op_mul, mux(mul_a), mux(mul_b), pc) : Vec{};

        // Writes land after every read of this instruction, as on hardware.
        if (op_add != A_NOP && cond_add == C_ALWAYS) write(waddr_add, !ws, ra, pc, n, st, wa, wb);
        else if (op_add == A_NOP && waddr_add != W_NOP && cond_add == C_ALWAYS) throw err(pc, "write from a nop add");
        if (op_mul != M_NOP && cond_mul == C_ALWAYS) write(waddr_mul, ws, rm, pc, n, st, wa, wb);
        if (sf) {
            const Vec& r = op_add != A_NOP ? ra : rm;
            for (int l = 0; l < 16; l++) { zf_[l] = r[l] == 0; nf_[l] = int32_t(r[l]) < 0; }
        }
        if (sig == SIG_LDTMU0 && ((op_add != A_NOP && waddr_add >= W_TMU0_S) || (op_mul != M_NOP && waddr_mul >= W_TMU0_S)))
            throw err(pc, "ldtmu signal in an instruction that writes a TMU register (wrong data on hardware)");
        if (sig == SIG_LDTMU0) {
            if (uses(MUX_R4) && !allow_r4_read_with_ldtmu)
                throw err(pc, "r4 read in the instruction that signals ldtmu");
            if (tmu_.empty()) throw err(pc, "ldtmu0 with no outstanding request");
            acc_[4] = tmu_.front(); tmu_.pop_front();
        }
        if (sig == SIG_END) {
            if (end_delay >= 0) throw err(pc, "second thread end");
            end_delay = 2;  // two delay slots follow this instruction
        }
        written_a_prev_ = wa; written_b_prev_ = wb;
    }
};

}  // namespace vc4
