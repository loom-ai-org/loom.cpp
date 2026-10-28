// ggml-vc4: the Raspberry Pi Zero / 1 / 2 / 3 GPU (VideoCore IV, 12 QPUs) as an accelerator USED WITH
// the CPU backend -- ggml's GGML_BACKEND_DEVICE_TYPE_ACCEL role, the one ggml-blas plays.
//
// It is never a device on its own. It takes the MUL_MATs it can win (F32, large enough to amortise
// a ~150 us launch; P4.45: also per-channel Q8_0 held in its VC4_Q8 buffer type) and the 1-D
// convolutions (P4.45: any kernel type with a to_float, read in place by qpu_conv1d.h), and leaves
// everything else to the CPU, which computes on the same memory: the
// buffer type is host memory (CPU-mapped, write-back cached), allocated from the GPU's contiguous
// pool so the QPUs can reach it too. A device that ranks host-memory accelerators above the CPU
// (loom's rank 1) therefore gets "CPU + QPUs" with no copies between them.
//
// Coherence: the ARM1176's L1 is the only cache between the CPU and the QPUs (QPU accesses use the
// 0x8 bus alias, coherent with the VideoCore L2). Before a launch the weights (src0) and the packed
// activations are cleaned and dst is flushed; after it dst is invalidated.
//
// Launch: the firmware mailbox. Requires the legacy (non-KMS) firmware graphics: under
// vc4-kms-v3d the firmware refuses qpu_enable and execute_qpu wedges it until reboot, so the
// registry reports no device while the vc4 DRM driver is loaded. execute_qpu also returns before
// the QPUs finish for >= 8 of them, so completion is the kernel's own done marker.
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "qpu_conv1d.h"
#include "qpu_gemm_q8.h"
#include "qpu_gemm_wt.h"
#include "vc4_mbox.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// MUL_MATs below this many flops stay on the CPU: measured on the Pi Zero W, 16x16x4 (2 kflop)
// took 0.15 ms on the QPUs against 0.05 ms in ggml's CPU kernel, 29x37x193 (0.4 Mflop) 0.19 ms
// against 0.86 ms.
constexpr int64_t VC4_MIN_FLOPS = int64_t(1) << 18;

// GGML_VC4_STATS=1: at exit, what the QPUs took and what the buffers cost (stderr). Not what they
// declined: the scheduler asks supports_op several times per node, so this side cannot count it.
struct Stats {
    uint64_t mul_mats = 0, convs = 0, launches = 0; double flops = 0, qpu_s = 0, host_s = 0;
    // Every ggml buffer of this type is a CMA allocation (dma-heap + vcsm import), far slower than
    // malloc, and an engine that makes VC4 its primary allocates inputs and outputs here too.
    uint64_t buffers = 0; double buffer_bytes = 0, alloc_s = 0;
    ~Stats() {
        if (!getenv("GGML_VC4_STATS")) return;
        fprintf(stderr, "ggml-vc4: %llu MUL_MATs and %llu CONV_2Ds in %llu launches, %.3f GFLOP, QPU %.3f s, host side %.3f s; "
                        "%llu buffers, %.1f MB, allocated in %.3f s\n",
                (unsigned long long) mul_mats, (unsigned long long) convs, (unsigned long long) launches, flops * 1e-9,
                qpu_s, host_s,
                (unsigned long long) buffers, buffer_bytes * 1e-6, alloc_s);
    }
};
Stats g_stats;

// A convolution's weights in the uniform stream's layout (qpu_conv1d.h), packed on first use.
struct Packed {
    std::unique_ptr<vc4::GpuBuffer> mem;
    const ggml_backend_buffer_t owner;       // the vc4 buffer holding the weight it was packed from
};

struct Runtime {
    std::mutex mu;
    std::unique_ptr<vc4::Mailbox> mbox;
    std::unique_ptr<vc4::GpuBuffer> fixed;   // uncached: control list @0, done words @4096, code @8192
    std::unique_ptr<vc4::GpuBuffer> arena;   // cached: packed X / staged x + uniforms, grown on demand
    std::unique_ptr<vc4::GpuBuffer> codes;   // uncached: the generated kernels, bump-allocated
    size_t codes_used = 0;
    std::map<std::string, uint32_t> code_at;
    // Keyed by the weight's data pointer and dropped when its buffer is freed: a later model's weight
    // can come back at the same address.
    std::unordered_map<const void *, Packed> packed;
    uint32_t ctrl = 0, done = 0, code = 0;
    uint32_t seq = 0;
    std::string unavailable;                 // empty when the QPUs are usable
};

Runtime & runtime() {
    static Runtime rt;
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            if (access("/dev/vcio", F_OK) != 0) { rt.unavailable = "no /dev/vcio (not a VideoCore IV Raspberry Pi)"; return; }
            rt.mbox = std::make_unique<vc4::Mailbox>();  // refuses under KMS, stops if qpu_enable fails
            rt.fixed = std::make_unique<vc4::GpuBuffer>(64 * 1024);
            rt.ctrl = rt.fixed->bus();
            rt.done = rt.fixed->bus() + 4096;
            rt.code = rt.fixed->bus() + 8192;
            const vc4::Program prog = vc4::gemm_f32_wt_kernel();
            GGML_ASSERT(8192 + prog.code.size() * 8 <= rt.fixed->size());
            memcpy(rt.fixed->at(rt.code), prog.code.data(), prog.code.size() * 8);
        } catch (const std::exception & e) {
            rt.unavailable = e.what();
            rt.mbox.reset();
        }
    });
    return rt;
}

// ---- buffer: one vcsm-cma block per ggml buffer, CPU-cached ----

struct BufferCtx {
    std::unique_ptr<vc4::GpuBuffer> mem;
};

vc4::GpuBuffer & mem_of(ggml_backend_buffer_t buffer) { return *static_cast<BufferCtx *>(buffer->context)->mem; }

void buffer_free(ggml_backend_buffer_t buffer) {
    Runtime & rt = runtime();
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        for (auto it = rt.packed.begin(); it != rt.packed.end();) {
            it = it->second.owner == buffer ? rt.packed.erase(it) : std::next(it);
        }
    }
    delete static_cast<BufferCtx *>(buffer->context);
}
void * buffer_get_base(ggml_backend_buffer_t buffer) { return mem_of(buffer).base(); }

void buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
    GGML_UNUSED(buffer);
}
void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    GGML_UNUSED(buffer);
}
void buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
    GGML_UNUSED(buffer);
}
bool buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}
void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(mem_of(buffer).base(), value, mem_of(buffer).size());
}

const ggml_backend_buffer_i buffer_iface = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buffer_memset_tensor,
    /* .set_tensor    = */ buffer_set_tensor,
    /* .get_tensor    = */ buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ buffer_cpy_tensor,
    /* .clear         = */ buffer_clear,
    /* .reset         = */ nullptr,
};

ggml_backend_buffer_type_t vc4_q8_buffer_type();
extern const ggml_backend_buffer_i q8_buffer_iface;

const char * buft_get_name(ggml_backend_buffer_type_t buft) { return buft == vc4_q8_buffer_type() ? "VC4_Q8" : "VC4"; }

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto ctx = std::make_unique<BufferCtx>();
        ctx->mem = std::make_unique<vc4::GpuBuffer>(uint32_t((std::max<size_t>(size, 1) + 4095) & ~size_t(4095)), /*cached=*/true);
        g_stats.buffers++;
        g_stats.buffer_bytes += double(size);
        g_stats.alloc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ggml_backend_buffer_init(buft, buft == vc4_q8_buffer_type() ? q8_buffer_iface : buffer_iface, ctx.release(), size);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-vc4: cannot allocate %zu bytes of GPU memory: %s (raise the CMA pool, e.g. dtoverlay=cma,cma-256)\n",
                       size, e.what());
        return nullptr;
    }
}
size_t buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }
// The kernel's TMU reads run a few words past a weight matrix's last row, so leaf tensors (weights
// and inputs, the only ones a vc4 MUL_MAT takes as src0) get that much more, which the allocator
// keeps out of the next tensor's way and dma-heap hands out zeroed. Computed tensors keep their
// exact size: ggml asserts that only a short list of ops may grow (ggml_op_alloc_size_may_expand).
size_t buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (buft == vc4_q8_buffer_type()) return vc4::gemm_q8_weight_bytes(int(tensor->ne[1]), int(tensor->ne[0]));
    return ggml_nbytes(tensor) + (tensor->op == GGML_OP_NONE ? vc4::gemm_wt_tail_bytes() : 0);
}
bool buft_is_host(ggml_backend_buffer_type_t buft) { return buft != vc4_q8_buffer_type(); }

ggml_backend_buffer_type_t vc4_buffer_type();

bool is_vc4_buffer(const ggml_backend_buffer_t b) { return b != nullptr && b->buft == vc4_buffer_type(); }

// Where a weight lives. A view of one has no buffer of its own until the allocator initialises it,
// which is after the scheduler has asked supports_op -- so every reshaped kernel read as "not
// allocated" (P4.45: amy's F32 convolutions, all of them) -- but its root always has one.
ggml_backend_buffer_t weight_buffer(const ggml_tensor * w) { return w->view_src ? w->view_src->buffer : w->buffer; }

void cma_memory(size_t * free, size_t * total);

// ---- the VC4_Q8 buffer type (P4.45) ----
//
// Q8_0 weights whose blocks in a row share one scale -- loom's Q8_0_PC -- repacked on upload into
// qpu_gemm_q8.h's layout: offset-binary bytes, row scales after them. Not host memory, because the
// layout is not ggml's, so a weight here is read by the QPUs and by nothing else; an engine places a
// copy here only for MUL_MATs this backend says it takes, offered through ggml's extra buffer types.

ggml_backend_buffer_type_t vc4_q8_buffer_type();
bool is_vc4_q8_buffer(const ggml_backend_buffer_t b) { return b != nullptr && b->buft == vc4_q8_buffer_type(); }

struct BlockQ8_0 { ggml_fp16_t d; int8_t qs[32]; };   // ggml's block_q8_0
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 layout");

void q8_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q8_0 && offset == 0 && size == ggml_nbytes(tensor));
    const int K = int(tensor->ne[0]), M = int(tensor->ne[1]);
    if (!vc4::gemm_q8_repack(static_cast<const BlockQ8_0 *>(data), M, K, [](ggml_fp16_t h) { return ggml_fp16_to_fp32(h); },
                             static_cast<uint8_t *>(tensor->data))) {
        GGML_ABORT("ggml-vc4: '%s' is not per-channel (blocks of one row with different scales); VC4_Q8 holds only "
                   "weights with one scale per row", tensor->name);
    }
    mem_of(buffer).clean(tensor->data, vc4::gemm_q8_weight_bytes(M, K));
}
void q8_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void *, size_t, size_t) {
    GGML_ABORT("ggml-vc4: VC4_Q8 holds '%s' repacked; there is no ggml layout to read back", tensor->name);
}
bool q8_buffer_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) { return false; }

const ggml_backend_buffer_i q8_buffer_iface = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ q8_buffer_set_tensor,
    /* .get_tensor    = */ q8_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ q8_buffer_cpy_tensor,
    /* .clear         = */ buffer_clear,
    /* .reset         = */ nullptr,
};

// ---- launching ----

// The bus address of a generated kernel, generating and placing it on first use.
uint32_t program_at(Runtime & rt, const std::string & key, const std::function<vc4::Program()> & make) {
    auto it = rt.code_at.find(key);
    if (it != rt.code_at.end()) return it->second;
    const vc4::Program prog = make();
    const size_t bytes = prog.code.size() * 8;
    if (!rt.codes) rt.codes = std::make_unique<vc4::GpuBuffer>(256 * 1024);
    GGML_ASSERT(rt.codes_used + bytes <= rt.codes->size());
    const uint32_t at = rt.codes->bus() + uint32_t(rt.codes_used);
    memcpy(rt.codes->at(at), prog.code.data(), bytes);
    rt.codes_used += (bytes + 63) & ~size_t(63);
    rt.code_at[key] = at;
    return at;
}

// Start nq QPUs on `code` with the uniform streams in `starts` and wait for every done marker.
void launch_and_wait(Runtime & rt, int nq, const std::vector<uint32_t> & starts, uint32_t code, uint32_t seq,
                     const std::string & what) {
    for (int q = 0; q < nq; q++) {
        rt.fixed->at(rt.ctrl)[2 * q] = starts[q];
        rt.fixed->at(rt.ctrl)[2 * q + 1] = code;
    }
    __sync_synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    rt.mbox->execute(uint32_t(nq), rt.ctrl);
    for (;;) {
        int ok = 0;
        for (int q = 0; q < nq; q++) ok += *reinterpret_cast<volatile uint32_t *>(rt.fixed->at(rt.done + q * 64)) == seq;
        if (ok == nq) break;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            // A QPU program that did not finish leaves the QPUs stuck; disabling them then has wedged
            // the firmware, and so has a reboot, so stop here and say what is needed.
            GGML_ABORT("ggml-vc4: %d of %d QPUs did not finish %s in 2 s; the board needs a power cycle",
                       nq - ok, nq, what.c_str());
        }
    }
    g_stats.qpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    g_stats.launches++;
}

void ensure_arena(Runtime & rt, size_t need) {
    if (!rt.arena || rt.arena->size() < need) {
        rt.arena.reset();
        rt.arena = std::make_unique<vc4::GpuBuffer>(uint32_t((need + need / 4 + 65535) & ~size_t(65535)), /*cached=*/true);
    }
}

// ---- the MUL_MAT ----

// A per-channel Q8_0 weight the engine copied into VC4_Q8. Only from 16 columns: at N = 1 the kernel
// fills one accumulator of sixteen and measured even with the CPU's int8 GEMV (Moonshine's decode
// projections, 1.8 against 1.9 ms; its vocabulary, 23 against 25 ms), not worth a split.
bool mul_mat_q8_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w->type != GGML_TYPE_Q8_0 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || w->view_src != nullptr) return false;
    if (x->nb[0] != 4 || x->nb[1] % 4 || op->nb[0] != 4 || op->nb[1] % 4) return false;
    if (op->nb[1] > 8191 + 4) return false;  // VDW row stride is 13 bits
    if (x->ne[1] < 16) return false;
    return 2 * op->ne[0] * op->ne[1] * w->ne[0] * op->ne[2] * op->ne[3] >= VC4_MIN_FLOPS;
}

bool mul_mat_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (is_vc4_q8_buffer(weight_buffer(w))) return mul_mat_q8_supported(op);
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    // The weights must already live where the QPUs can read them, a vc4 buffer, and be a leaf (or a
    // view of one): only leaves carry the tail the TMU reads past the last row.
    if (!is_vc4_buffer(weight_buffer(w))) return false;
    if ((w->view_src ? w->view_src->op : w->op) != GGML_OP_NONE) return false;
    if (w->nb[0] != 4 || x->nb[0] != 4 || op->nb[0] != 4) return false;
    if (w->nb[1] % 4 || op->nb[1] % 4 || w->nb[1] >= (1u << 24)) return false;
    if (op->nb[1] > 8191 + 4) return false;  // VDW row stride is 13 bits (bytes after mvalid words)
    if (x->ne[2] % w->ne[2] || x->ne[3] % w->ne[3]) return false;
    const int64_t flops = 2 * op->ne[0] * op->ne[1] * w->ne[0] * op->ne[2] * op->ne[3];
    return flops >= VC4_MIN_FLOPS;
}

void mul_mat(ggml_tensor * dst) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    const auto t_host0 = std::chrono::steady_clock::now();
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    vc4::GpuBuffer & wmem = mem_of(weight_buffer(w));
    vc4::GpuBuffer & cmem = mem_of(dst->buffer);
    const int K = int(w->ne[0]), M = int(w->ne[1]), N = int(x->ne[1]);
    const int64_t r2 = x->ne[2] / w->ne[2], r3 = x->ne[3] / w->ne[3];
    const size_t xslice = vc4::gemm_wt_xpack_bytes(N, K);

    std::vector<vc4::GemmWtProblem> probs;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
            const uint8_t * wp = static_cast<const uint8_t *>(w->data) + (i2 / r2) * w->nb[2] + (i3 / r3) * w->nb[3];
            uint8_t * cp = static_cast<uint8_t *>(dst->data) + i2 * dst->nb[2] + i3 * dst->nb[3];
            probs.push_back({M, N, K, wmem.bus_of(wp), uint32_t(w->nb[1]), cmem.bus_of(cp), uint32_t(dst->nb[1]), 0});
        }
    const size_t need = probs.size() * xslice + vc4::gemm_wt_uniform_words(probs, 12) * 4 + 4096;
    ensure_arena(rt, need);
    uint8_t * xbase = rt.arena->base();
    size_t k = 0;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++, k++) {
            float * out = reinterpret_cast<float *>(xbase + k * xslice);
            const uint8_t * xs = static_cast<const uint8_t *>(x->data) + i2 * x->nb[2] + i3 * x->nb[3];
            if (x->nb[1] % 4 == 0) {
                vc4::gemm_wt_pack_x(reinterpret_cast<const float *>(xs), x->nb[1] / 4, N, K, out);
            } else {
                GGML_ABORT("ggml-vc4: activation rows not 4-byte aligned");
            }
            probs[k].xpack = rt.arena->bus_of(out);
        }
    const uint32_t unif = rt.arena->bus_of(xbase + ((probs.size() * xslice + 63) & ~size_t(63)));
    const uint32_t seq = ++rt.seq;
    std::vector<uint32_t> starts;
    auto cpu = [&](uint32_t bus) { return rt.arena->at(bus); };
    const int nq = vc4::gemm_wt_uniforms(cpu, unif, probs, 12, rt.done, seq, starts);
    // Coherence before the launch: the weights may have been written by the CPU (loading, or an
    // activation the CPU computed), the packed X and uniforms just were; dst must hold no dirty
    // lines that could be written back over what the QPUs store.
    rt.arena->clean(xbase, need);
    for (int64_t i3 = 0; i3 < w->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < w->ne[2]; i2++) {
            const uint8_t * wp = static_cast<const uint8_t *>(w->data) + i2 * w->nb[2] + i3 * w->nb[3];
            wmem.clean(wp, size_t(M - 1) * w->nb[1] + size_t(K) * 4 + vc4::gemm_wt_tail_bytes());
        }
    cmem.flush(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_host0).count();
    launch_and_wait(rt, nq, starts, rt.code, seq,
                    "a MUL_MAT [" + std::to_string(M) + " x " + std::to_string(N) + " x " + std::to_string(K) + "]");
    const auto t1 = std::chrono::steady_clock::now();
    cmem.invalidate(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    g_stats.mul_mats++;
    g_stats.flops += 2.0 * M * N * K * double(probs.size());
}

void mul_mat_q8(ggml_tensor * dst) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    const auto t_host0 = std::chrono::steady_clock::now();
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    vc4::GpuBuffer & wmem = mem_of(weight_buffer(w));
    vc4::GpuBuffer & cmem = mem_of(dst->buffer);
    const int K = int(w->ne[0]), M = int(w->ne[1]), N = int(x->ne[1]), Kp = vc4::gemm_q8_kp(K);
    const size_t xslice = vc4::gemm_q8_xpack_bytes(N, K);
    const uint32_t wbus = wmem.bus_of(w->data);
    std::vector<vc4::GemmQ8Problem> probs;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
            uint8_t * cp = static_cast<uint8_t *>(dst->data) + i2 * dst->nb[2] + i3 * dst->nb[3];
            probs.push_back({M, N, K, wbus, uint32_t(Kp), wbus + uint32_t(M) * Kp, cmem.bus_of(cp), uint32_t(dst->nb[1]), 0});
        }
    const size_t need = probs.size() * xslice + vc4::gemm_q8_uniform_words(probs, 12) * 4 + 4096;
    ensure_arena(rt, need);
    uint8_t * xbase = rt.arena->base();
    size_t k = 0;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++, k++) {
            float * out = reinterpret_cast<float *>(xbase + k * xslice);
            const uint8_t * xs = static_cast<const uint8_t *>(x->data) + i2 * x->nb[2] + i3 * x->nb[3];
            vc4::gemm_q8_pack_x(reinterpret_cast<const float *>(xs), x->nb[1] / 4, N, K, out);
            probs[k].xpack = rt.arena->bus_of(out);
        }
    const uint32_t unif = rt.arena->bus_of(xbase + ((probs.size() * xslice + 63) & ~size_t(63)));
    const uint32_t seq = ++rt.seq;
    std::vector<uint32_t> starts;
    auto cpu = [&](uint32_t bus) { return rt.arena->at(bus); };
    const int nq = vc4::gemm_q8_uniforms(cpu, unif, probs, 12, rt.done, seq, starts);
    const uint32_t code = program_at(rt, "gemm_q8", [] { return vc4::gemm_q8_kernel(); });
    rt.arena->clean(xbase, need);
    cmem.flush(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_host0).count();
    launch_and_wait(rt, nq, starts, code, seq,
                    "a Q8_0 MUL_MAT [" + std::to_string(M) + " x " + std::to_string(N) + " x " + std::to_string(K) + "]");
    const auto t1 = std::chrono::steady_clock::now();
    cmem.invalidate(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    g_stats.mul_mats++;
    g_stats.flops += 2.0 * M * N * K * double(probs.size());
}

// ---- the 1-D convolution (P4.45) ----

// GGML_VC4_DEBUG=1: why a CONV_2D was declined (once per reason and shape) and where each one's time
// went -- the scheduler asks supports_op many times per node, so a decline is printed once.
bool debug_on() {
    static const bool on = [] { const char * e = getenv("GGML_VC4_DEBUG"); return e && strcmp(e, "0") != 0; }();
    return on;
}
bool decline(const ggml_tensor * op, const char * why) {
    if (debug_on()) {
        static std::mutex mu;
        static std::unordered_map<std::string, int> seen;
        const ggml_tensor * w = op->src[0];
        const std::string key = std::string(why) + " " + ggml_type_name(w->type) + " w[" + std::to_string(w->ne[0]) + "," +
                                std::to_string(w->ne[1]) + "," + std::to_string(w->ne[2]) + "," + std::to_string(w->ne[3]) +
                                "] x[" + std::to_string(op->src[1]->ne[0]) + "," + std::to_string(op->src[1]->ne[1]) + "," +
                                std::to_string(op->src[1]->ne[2]) + "]";
        std::lock_guard<std::mutex> lock(mu);
        if (seen[key]++ == 0) fprintf(stderr, "ggml-vc4: CONV_2D declined: %s\n", key.c_str());
    }
    return false;
}

// GGML_VC4_CONV=0 leaves every convolution to the CPU (the A/B switch).
bool conv_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_VC4_CONV"); return !(e && strcmp(e, "0") == 0); }();
    return on;
}

// The 1-D geometry of a CONV_2D, declared [KW, 1, IC, OC] or folded [IC*KW, OC] (ggml-0013's
// op_params 6..8): H = 1 on both sides and no H padding, one batch.
bool conv_geom(const ggml_tensor * op, vc4::Conv1dShape & c) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    const int32_t * pp = reinterpret_cast<const int32_t *>(op->op_params);
    int64_t KW, KH, IC, OC;
    if (pp[6] > 0) { KW = pp[6]; KH = pp[7]; IC = pp[8]; OC = w->ne[1]; }
    else { KW = w->ne[0]; KH = w->ne[1]; IC = w->ne[2]; OC = w->ne[3]; }
    if (KH != 1 || pp[3] != 0 || x->ne[1] != 1 || x->ne[3] != 1 || op->ne[1] != 1 || op->ne[3] != 1) return false;
    if (x->ne[2] != IC || op->ne[2] != OC || pp[0] < 1 || pp[4] < 1 || pp[2] < 0) return false;
    c = {int(IC), int(KW), pp[0], pp[4], pp[2], int(x->ne[0]), int(op->ne[0]), int(OC)};
    return true;
}

constexpr size_t VC4_CONV_STAGE_BYTES = size_t(2) << 20;  // see conv()

bool conv_supported(const ggml_tensor * op) {
    if (!conv_enabled()) return false;
    vc4::Conv1dShape c;
    if (!conv_geom(op, c)) return decline(op, "not 1-D");
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    // An F32 kernel has no to_float -- it needs none -- which the first version read as "no way to
    // convert" and so declined every F32 convolution.
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr)) {
        return decline(op, "types");
    }
    // The weights are packed once from where they live, which must be a vc4 buffer: its free is
    // what drops the packed copy.
    if (!is_vc4_buffer(weight_buffer(w))) {
        return decline(op, weight_buffer(w) ? ggml_backend_buffer_name(weight_buffer(w)) : "weight not allocated");
    }
    if (!ggml_is_contiguous(w)) return decline(op, "weight not contiguous");
    if ((w->view_src ? w->view_src->op : w->op) != GGML_OP_NONE) return decline(op, "weight computed");
    if (x->nb[0] != 4 || x->nb[2] % 4 || op->nb[0] != 4 || op->nb[2] % 4) return decline(op, "strides");
    if (!vc4::conv1d_kernel_ok(c.IC, c.KW)) return decline(op, "IC too small for the kernel");
    int lo, hi;
    vc4::conv1d_interior(c, lo, hi);
    if (hi - lo < 16) return decline(op, "interior < 16");
    if (2 * int64_t(c.OC) * c.T_out * c.IC * c.KW < VC4_MIN_FLOPS) return decline(op, "below the flop threshold");
    // A weight not packed yet needs its F32 copy to fit in the GPU's pool beside what the arenas
    // grow to; declining here costs nothing, failing to allocate at compute time would.
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    // What the call needs from the pool if nothing is cached: x staged (padded) and the weights packed
    // into the arena, beside what the arena already holds.
    if (rt.packed.count(w->data) == 0) {
        size_t free_bytes = 0, total = 0;
        cma_memory(&free_bytes, &total);
        const size_t need = vc4::conv1d_wpack_floats(c.OC, c.IC * c.KW) * 4 + VC4_CONV_STAGE_BYTES * 2;
        const size_t have = free_bytes + (rt.arena ? rt.arena->size() : 0);
        if (have < need + (size_t(4) << 20)) return decline(op, "GPU pool full");
    }
    if (debug_on()) {
        static std::mutex mu;
        static std::unordered_map<std::string, int> seen;
        const std::string key = std::string(ggml_get_name(op)) + " " + ggml_type_name(w->type) + " T " + std::to_string(c.T_in);
        std::lock_guard<std::mutex> lk(mu);
        if (seen[key]++ == 0) fprintf(stderr, "ggml-vc4: CONV_2D accepted: %s\n", key.c_str());
    }
    return true;
}

// The weights in qpu_conv1d.h's block layout, written in output order (the ARM1176 does not
// allocate on a write, so scattered stores each go to the bus): 16 rows x KC k are dequantized into
// a 4-16 KB scratch that stays in L1, then copied out across. KC is a multiple of the block size,
// so to_float can start a row at k0; the last chunk may be short, still whole blocks.
void pack_conv_weights(const ggml_tensor * w, int OC, int K, float * out) {
    const auto * traits = ggml_get_type_traits(w->type);
    const int64_t blck = ggml_blck_size(w->type);
    const int KC = blck > 64 ? 256 : 64;
    const bool chunked = w->type == GGML_TYPE_F32 || (K % blck == 0 && KC % blck == 0);
    if (!chunked) {  // a row that is not a whole number of chunks: dequantize it whole
        std::vector<float> rows(size_t(OC) * K);
        for (int oc = 0; oc < OC; oc++)
            traits->to_float(static_cast<const uint8_t *>(w->data) + size_t(oc) * ggml_row_size(w->type, K), &rows[size_t(oc) * K], K);
        vc4::conv1d_pack_w(rows.data(), OC, K, out);
        return;
    }
    std::vector<float> scratch(size_t(16) * KC);
    const size_t row_bytes = ggml_row_size(w->type, K);
    for (int b = 0; b < (OC + 15) / 16; b++)
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kn = std::min(KC, K - k0);
            for (int r = 0; r < 16; r++) {
                const int oc = b * 16 + r;
                float * dst = &scratch[size_t(r) * KC];
                if (oc >= OC) { std::fill(dst, dst + kn, 0.f); continue; }
                const uint8_t * row = static_cast<const uint8_t *>(w->data) + size_t(oc) * row_bytes;
                if (w->type == GGML_TYPE_F32) memcpy(dst, row + size_t(k0) * 4, size_t(kn) * 4);
                else traits->to_float(row + size_t(k0 / blck) * ggml_type_size(w->type), dst, kn);
            }
            float * o = out + (size_t(b) * K + k0) * 16;
            for (int k = 0; k < kn; k++)
                for (int r = 0; r < 16; r++) *o++ = scratch[size_t(r) * KC + k];
        }
}

// Packed once and kept while the pool keeps a reserve for the arenas and the scheduler's buffers;
// past that, null, and the caller packs into the arena for this call alone. Kept unconditionally,
// VITS's text-encoder kernels (192 -> 768 channels, 1.5-1.7 MB each in F32) filled the 64 MB pool
// before the vocoder's were reached, and the synthesis failed to allocate.
constexpr size_t VC4_POOL_RESERVE = size_t(24) << 20;
Packed * packed_weights(Runtime & rt, const ggml_tensor * w, int OC, int K) {
    auto it = rt.packed.find(w->data);
    if (it != rt.packed.end()) return &it->second;
    const size_t bytes = vc4::conv1d_wpack_floats(OC, K) * 4;
    size_t free_bytes = 0, total = 0;
    cma_memory(&free_bytes, &total);
    if (free_bytes < bytes + VC4_POOL_RESERVE) return nullptr;
    auto mem = std::make_unique<vc4::GpuBuffer>(uint32_t((bytes + 4095) & ~size_t(4095)), /*cached=*/true);
    pack_conv_weights(w, OC, K, reinterpret_cast<float *>(mem->base()));
    mem->clean(mem->base(), bytes);
    return &rt.packed.emplace(w->data, Packed{std::move(mem), weight_buffer(w)}).first->second;
}

// A staged convolution runs as chunks along T of about VC4_CONV_STAGE_BYTES of input, so the arena
// stays small: the 64 MB pool also holds the model and the scheduler's buffers, and staging VITS's
// 75008 x 32 vocoder input whole (9.6 MB) made the synthesis fail to allocate.

void conv(ggml_tensor * dst) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t_host0 = clk::now();
    vc4::Conv1dShape c;
    GGML_ASSERT(conv_geom(dst, c));
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    GGML_ASSERT(is_vc4_buffer(dst->buffer));
    vc4::GpuBuffer & dmem = mem_of(dst->buffer);
    const int K = c.IC * c.KW;
    Packed * pk = packed_weights(rt, w, c.OC, K);

    // x where the TMU can read it. In place when it already is in a vc4 buffer and the host's share of
    // the padded edges is small; otherwise staged into the arena with its zero padding, which makes
    // every position the QPUs' -- the host's scalar edges cost more than the QPUs on a short
    // convolution (30 ms against 20 for VITS's 192 -> 384, k = 5, T = 293).
    int lo, hi;
    vc4::conv1d_interior(c, lo, hi);
    const double edge_macs = double(lo + c.T_out - hi) * c.OC * K;
    const bool in_place = is_vc4_buffer(x->buffer) && edge_macs < 2e5;
    int chunk = c.T_out;
    if (!in_place) {
        const int64_t per_out = int64_t(c.IC) * c.s * 4;
        chunk = std::min<int>(c.T_out, int(std::max<int64_t>(256, int64_t(VC4_CONV_STAGE_BYTES) / per_out)) / 16 * 16);
    }
    const int in_len = (chunk - 1) * c.s + (c.KW - 1) * c.d + 1;   // staged positions per chunk
    vc4::Conv1dShape cq = c;                                        // the shape one launch runs
    if (!in_place) { cq.T_in = in_len; cq.p = 0; cq.T_out = chunk; }
    const size_t staged = in_place ? 0 : (size_t(c.IC) * in_len * 4 + 63) & ~size_t(63);
    const size_t wbytes = pk ? 0 : (vc4::conv1d_wpack_floats(c.OC, K) * 4 + 63) & ~size_t(63);
    const vc4::Conv1dOut out = vc4::conv1d_block_ok(uint32_t(dst->nb[2])) ? vc4::CONV1D_OUT_BLOCK : vc4::CONV1D_OUT_ROWS;
    vc4::Conv1dProblem P{cq, 0, 0, 0, uint32_t(dst->nb[2]), 0, 0};
    const size_t uwords = vc4::conv1d_uniform_words(P, 12);
    ensure_arena(rt, wbytes + staged + 64 + uwords * 4 + 4096);
    uint8_t * abase = rt.arena->base();
    const float * wpack;
    if (pk) {
        P.wpack = pk->mem->bus();
        wpack = reinterpret_cast<const float *>(pk->mem->base());
    } else {
        pack_conv_weights(w, c.OC, K, reinterpret_cast<float *>(abase));
        P.wpack = rt.arena->bus();
        wpack = reinterpret_cast<const float *>(abase);
        rt.arena->clean(abase, wbytes);
    }
    uint8_t * sbase = abase + wbytes;
    P.discard = rt.arena->bus() + uint32_t(wbytes + staged);
    const uint32_t unif = P.discard + 64;
    const int tail = vc4::conv1d_tail(c.IC, c.KW);
    const uint32_t code = program_at(rt, "conv1d/" + std::to_string(c.KW) + "/" + std::to_string(tail) + "/" + std::to_string(int(out)),
                                     [&] { return vc4::conv1d_f32_kernel(c.KW, tail, out); });
    dmem.flush(dst->data, ggml_nbytes(dst));
    const auto t_prep = clk::now();
    g_stats.host_s += ms(t_host0, t_prep) * 1e-3;
    double stage_ms = 0, unif_ms = 0, qpu_ms = 0;
    int launches = 0;
    auto cpu = [&](uint32_t bus) { return rt.arena->at(bus); };
    std::vector<uint32_t> starts;
    for (int t0 = 0; t0 < c.T_out; t0 += chunk) {
        const auto ta = clk::now();
        if (in_place) {
            vc4::GpuBuffer & xmem = mem_of(x->buffer);
            P.x = xmem.bus_of(x->data);
            P.x_row = uint32_t(x->nb[2]);
            P.dst = dmem.bus_of(dst->data);
            xmem.clean(x->data, size_t(c.IC - 1) * x->nb[2] + size_t(c.T_in) * 4);
        } else {
            const int tn = std::min(chunk, c.T_out - t0);
            P.c.T_out = tn;
            P.c.T_in = (tn - 1) * c.s + (c.KW - 1) * c.d + 1;
            // staged position j holds input t0*s - p + j, zero outside [0, T_in)
            const int first = t0 * c.s - c.p, n = P.c.T_in;
            const int z0 = std::min(n, std::max(0, -first));
            const int end = std::max(z0, std::min(n, c.T_in - first));
            for (int ic = 0; ic < c.IC; ic++) {
                uint8_t * row = sbase + size_t(ic) * in_len * 4;
                const uint8_t * src = static_cast<const uint8_t *>(x->data) + size_t(ic) * x->nb[2];
                memset(row, 0, size_t(z0) * 4);
                memcpy(row + size_t(z0) * 4, src + size_t(first + z0) * 4, size_t(end - z0) * 4);
                memset(row + size_t(end) * 4, 0, size_t(n - end) * 4);
            }
            P.x = rt.arena->bus() + uint32_t(wbytes);
            P.x_row = uint32_t(in_len) * 4;
            P.dst = dmem.bus_of(dst->data) + uint32_t(t0) * 4;
        }
        const auto tb = clk::now();
        const uint32_t seq = ++rt.seq;
        const int nq = vc4::conv1d_uniforms(cpu, unif, P, 12, rt.done, seq, starts, out);
        GGML_ASSERT(nq > 0);
        rt.arena->clean(sbase, std::min<size_t>(rt.arena->size() - wbytes,
                                                staged + 64 + (rt.arena->at(starts.back()) - rt.arena->at(unif)) * 4 +
                                                    vc4::conv1d_uniform_words(P, 1) * 4));
        const auto tc = clk::now();
        launch_and_wait(rt, nq, starts, code, seq,
                        "a CONV_2D [IC " + std::to_string(c.IC) + " KW " + std::to_string(c.KW) + " T " + std::to_string(c.T_in) +
                        " OC " + std::to_string(c.OC) + "]");
        const auto td = clk::now();
        stage_ms += ms(ta, tb); unif_ms += ms(tb, tc); qpu_ms += ms(tc, td);
        g_stats.host_s += ms(ta, tc) * 1e-3;
        launches++;
    }
    const auto t1 = clk::now();
    dmem.invalidate(dst->data, ggml_nbytes(dst));
    if (in_place) vc4::conv1d_edges_host(c, static_cast<const float *>(x->data), x->nb[2] / 4, wpack,
                                         static_cast<float *>(dst->data), dst->nb[2] / 4);
    const auto t_end = clk::now();
    g_stats.host_s += ms(t1, t_end) * 1e-3;
    if (debug_on()) {
        fprintf(stderr, "ggml-vc4: CONV_2D IC %d KW %d s %d d %d T %d->%d OC %d %s%s%s x%d: pack+flush %.1f stage %.1f "
                        "uniforms+clean %.1f QPU %.1f inval+edges %.1f ms\n", c.IC, c.KW, c.s, c.d, c.T_in, c.T_out, c.OC,
                in_place ? "in place" : "staged", out == vc4::CONV1D_OUT_BLOCK ? " block" : " rows", pk ? " cached" : " per call",
                launches, ms(t_host0, t_prep), stage_ms, unif_ms, qpu_ms, ms(t1, t_end));
    }
    g_stats.convs++;
    g_stats.flops += 2.0 * c.OC * c.T_out * K;
}

// ---- backend ----

const char * backend_get_name(ggml_backend_t) { return "VC4"; }
void backend_free(ggml_backend_t backend) { delete backend; }

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (debug_on() && node->op != GGML_OP_NONE) fprintf(stderr, "ggml-vc4: computing %s %s\n", ggml_op_desc(node), ggml_get_name(node));
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (is_vc4_q8_buffer(weight_buffer(node->src[0]))) mul_mat_q8(node);
                else mul_mat(node);
                break;
            case GGML_OP_CONV_2D: conv(node); break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE: break;
            default: GGML_ABORT("ggml-vc4: unsupported op %s", ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

const ggml_backend_i backend_iface = {
    /* .get_name            = */ backend_get_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x76, 0x63, 0x34, 0x71, 0x70, 0x75, 0x2d, 0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x6c, 0x6f, 0x6f, 0x6d};
    return &guid;
}

// ---- device ----

ggml_backend_reg_t vc4_reg();

const char * dev_get_name(ggml_backend_dev_t) { return "VC4"; }
const char * dev_get_description(ggml_backend_dev_t) { return "VideoCore IV QPUs (12)"; }

// The GPU's contiguous pool (CMA) is what every vc4 buffer comes from.
void cma_memory(size_t * free, size_t * total) {
    *free = *total = 0;
    std::ifstream f("/proc/meminfo");
    std::string key; size_t kb; std::string unit;
    while (f >> key >> kb >> unit) {
        if (key == "CmaTotal:") *total = kb * 1024;
        if (key == "CmaFree:") *free = kb * 1024;
    }
}

// What an engine may put here: the pool less the room the QPUs work in -- the arena (staged input,
// packed activations, uniforms), packed convolution weights, the scheduler's buffers for the nodes
// this backend runs. VITS F32's 62.6 MB of weights "fitted" a 64.6 MB pool and left the QPUs room for
// nothing, so it ran every node on the CPU through the scheduler, 2.6% slower than the CPU alone.
constexpr size_t VC4_WORKING_RESERVE = size_t(16) << 20;
void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    cma_memory(free, total);
    *free = *free > VC4_WORKING_RESERVE ? *free - VC4_WORKING_RESERVE : 0;
}
enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = dev_get_name(dev);
    props->description = dev_get_description(dev);
    props->type = dev_get_type(dev);
    dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,  // weights must be copied into the GPU's pool
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char *) {
    return new ggml_backend{/* .guid = */ backend_guid(), /* .iface = */ backend_iface, /* .device = */ dev, /* .context = */ nullptr};
}

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t) { return vc4_buffer_type(); }

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (debug_on()) {
        static std::mutex mu;
        static bool seen[GGML_OP_COUNT] = {};
        std::lock_guard<std::mutex> lock(mu);
        if (!seen[op->op]) { seen[op->op] = true; fprintf(stderr, "ggml-vc4: asked about %s\n", ggml_op_desc(op)); }
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: return true;
        case GGML_OP_MUL_MAT: return mul_mat_supported(op);
        case GGML_OP_CONV_2D: return conv_supported(op);
        default: return false;
    }
}

bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || buft == vc4_q8_buffer_type();
}

const ggml_backend_device_i device_iface = {
    /* .get_name             = */ dev_get_name,
    /* .get_description      = */ dev_get_description,
    /* .get_memory           = */ dev_get_memory,
    /* .get_type             = */ dev_get_type,
    /* .get_props            = */ dev_get_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

ggml_backend_device & the_device() {
    static ggml_backend_device dev = {/* .iface = */ device_iface, /* .reg = */ vc4_reg(), /* .context = */ nullptr};
    return dev;
}

ggml_backend_buffer_type_t vc4_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ buft_get_name,
            /* .alloc_buffer     = */ buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ buft_get_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ buft_is_host,
        },
        /* .device  = */ &the_device(),
        /* .context = */ nullptr,
    };
    return &buft;
}

ggml_backend_buffer_type_t vc4_q8_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ buft_get_name,
            /* .alloc_buffer     = */ buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ buft_get_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ buft_is_host,
        },
        /* .device  = */ &the_device(),
        /* .context = */ nullptr,
    };
    return &buft;
}

// ---- registry ----

const char * reg_get_name(ggml_backend_reg_t) { return "VC4"; }

size_t reg_get_device_count(ggml_backend_reg_t) {
    Runtime & rt = runtime();
    static std::atomic<bool> said{false};
    if (!rt.unavailable.empty() && !said.exchange(true)) GGML_LOG_INFO("ggml-vc4: no device: %s\n", rt.unavailable.c_str());
    return rt.unavailable.empty() ? 1 : 0;
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return &the_device();
}

// VC4_Q8, through the name ggml's own extra buffer types are found by, so an engine can offer a weight
// to it without knowing this backend.
ggml_backend_buffer_type_t * dev_get_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t bufts[] = {vc4_q8_buffer_type(), nullptr};
    return bufts;
}

void * reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) return reinterpret_cast<void *>(dev_get_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

ggml_backend_reg_t vc4_reg() {
    static ggml_backend_reg reg = {/* .api_version = */ GGML_BACKEND_API_VERSION, /* .iface = */ reg_iface, /* .context = */ nullptr};
    return &reg;
}

}  // namespace

extern "C" {
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vc4_reg(void);
}
ggml_backend_reg_t ggml_backend_vc4_reg(void) { return vc4_reg(); }

GGML_BACKEND_DL_IMPL(ggml_backend_vc4_reg)
