// ggml-vc4: the Raspberry Pi Zero / 1 / 2 / 3 GPU (VideoCore IV, 12 QPUs) as an accelerator USED WITH
// the CPU backend -- ggml's GGML_BACKEND_DEVICE_TYPE_ACCEL role, the one ggml-blas plays.
//
// It is never a device on its own. It takes the MUL_MATs it can win (F32, large enough to amortise
// a ~150 us launch) and leaves everything else to the CPU, which computes on the same memory: the
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

#include "qpu_gemm_wt.h"
#include "vc4_mbox.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// MUL_MATs below this many flops stay on the CPU: measured on the Pi Zero W, 16x16x4 (2 kflop)
// took 0.15 ms on the QPUs against 0.05 ms in ggml's CPU kernel, 29x37x193 (0.4 Mflop) 0.19 ms
// against 0.86 ms.
constexpr int64_t VC4_MIN_FLOPS = int64_t(1) << 18;

// GGML_VC4_STATS=1: at exit, what the QPUs took and what the buffers cost (stderr). Not what they
// declined: the scheduler asks supports_op several times per node, so this side cannot count it.
struct Stats {
    uint64_t mul_mats = 0, launches = 0; double flops = 0, qpu_s = 0, host_s = 0;
    // Every ggml buffer of this type is a CMA allocation (dma-heap + vcsm import), far slower than
    // malloc, and an engine that makes VC4 its primary allocates inputs and outputs here too.
    uint64_t buffers = 0; double buffer_bytes = 0, alloc_s = 0;
    ~Stats() {
        if (!getenv("GGML_VC4_STATS")) return;
        fprintf(stderr, "ggml-vc4: %llu MUL_MATs in %llu launches, %.3f GFLOP, QPU %.3f s, host side %.3f s; "
                        "%llu buffers, %.1f MB, allocated in %.3f s\n",
                (unsigned long long) mul_mats, (unsigned long long) launches, flops * 1e-9, qpu_s, host_s,
                (unsigned long long) buffers, buffer_bytes * 1e-6, alloc_s);
    }
};
Stats g_stats;

struct Runtime {
    std::mutex mu;
    std::unique_ptr<vc4::Mailbox> mbox;
    std::unique_ptr<vc4::GpuBuffer> fixed;   // uncached: control list @0, done words @4096, code @8192
    std::unique_ptr<vc4::GpuBuffer> arena;   // cached: packed X + uniforms, grown on demand
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

void buffer_free(ggml_backend_buffer_t buffer) { delete static_cast<BufferCtx *>(buffer->context); }
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

const char * buft_get_name(ggml_backend_buffer_type_t) { return "VC4"; }

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto ctx = std::make_unique<BufferCtx>();
        ctx->mem = std::make_unique<vc4::GpuBuffer>(uint32_t((std::max<size_t>(size, 1) + 4095) & ~size_t(4095)), /*cached=*/true);
        g_stats.buffers++;
        g_stats.buffer_bytes += double(size);
        g_stats.alloc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ggml_backend_buffer_init(buft, buffer_iface, ctx.release(), size);
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
size_t buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor) + (tensor->op == GGML_OP_NONE ? vc4::gemm_wt_tail_bytes() : 0);
}
bool buft_is_host(ggml_backend_buffer_type_t) { return true; }

ggml_backend_buffer_type_t vc4_buffer_type();

bool is_vc4_buffer(const ggml_backend_buffer_t b) { return b != nullptr && b->buft == vc4_buffer_type(); }

// ---- the MUL_MAT ----

bool mul_mat_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    // The weights must already live where the QPUs can read them, a vc4 buffer, and be a leaf (or a
    // view of one): only leaves carry the tail the TMU reads past the last row.
    if (!is_vc4_buffer(w->buffer)) return false;
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
    vc4::GpuBuffer & wmem = mem_of(w->buffer);
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
    if (!rt.arena || rt.arena->size() < need) {
        rt.arena.reset();
        rt.arena = std::make_unique<vc4::GpuBuffer>(uint32_t((need + need / 4 + 65535) & ~size_t(65535)), /*cached=*/true);
    }
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
    for (int q = 0; q < nq; q++) {
        rt.fixed->at(rt.ctrl)[2 * q] = starts[q];
        rt.fixed->at(rt.ctrl)[2 * q + 1] = rt.code;
    }
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
    __sync_synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    g_stats.host_s += std::chrono::duration<double>(t0 - t_host0).count();
    rt.mbox->execute(uint32_t(nq), rt.ctrl);
    for (;;) {
        int ok = 0;
        for (int q = 0; q < nq; q++) ok += *reinterpret_cast<volatile uint32_t *>(rt.fixed->at(rt.done + q * 64)) == seq;
        if (ok == nq) break;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            // A QPU program that did not finish leaves the QPUs stuck; disabling them then has wedged
            // the firmware, so stop here and say what is needed.
            GGML_ABORT("ggml-vc4: %d of %d QPUs did not finish a MUL_MAT [%d x %d x %d] in 2 s; the board needs a reboot",
                       nq - ok, nq, M, N, K);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    cmem.invalidate(dst->data, ggml_nbytes(dst));
    g_stats.qpu_s += std::chrono::duration<double>(t1 - t0).count();
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    g_stats.mul_mats++;
    g_stats.launches++;
    g_stats.flops += 2.0 * M * N * K * double(probs.size());
}

// ---- backend ----

const char * backend_get_name(ggml_backend_t) { return "VC4"; }
void backend_free(ggml_backend_t backend) { delete backend; }

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT: mul_mat(node); break;
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
void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = *total = 0;
    std::ifstream f("/proc/meminfo");
    std::string key; size_t kb; std::string unit;
    while (f >> key >> kb >> unit) {
        if (key == "CmaTotal:") *total = kb * 1024;
        if (key == "CmaFree:") *free = kb * 1024;
    }
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
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: return true;
        case GGML_OP_MUL_MAT: return mul_mat_supported(op);
        default: return false;
    }
}

bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return ggml_backend_buft_is_host(buft); }

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

void * reg_get_proc_address(ggml_backend_reg_t, const char *) { return nullptr; }

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
