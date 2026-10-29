// ggml-vc4: the Raspberry Pi Zero / 1 / 2 / 3 GPU (VideoCore IV, 12 QPUs) as an accelerator USED WITH
// the CPU backend -- ggml's GGML_BACKEND_DEVICE_TYPE_ACCEL role, the one ggml-blas plays.
//
// It is never a device on its own. It takes the MUL_MATs it can win (F32, large enough to amortise
// a ~150 us launch; P4.45: also per-channel Q8_0 held in its VC4_Q8 buffer type; P4.46: Q8_0 and Q4_0
// with a scale per block) and the 1-D convolutions (P4.45: any kernel type with a to_float, read in
// place by qpu_conv1d.h; P4.46: 1x1 ones over a VC4_Q8 weight in int8, and the elementwise nodes
// around a convolution folded into it), and leaves everything else to the CPU, which computes on the
// same memory: the
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
#include "qpu_gemm_qb.h"
#include "qpu_gemm_wt.h"
#include "vc4_mbox.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// MUL_MATs below this many flops stay on the CPU: measured on the Pi Zero W, 16x16x4 (2 kflop)
// took 0.15 ms on the QPUs against 0.05 ms in ggml's CPU kernel, 29x37x193 (0.4 Mflop) 0.19 ms
// against 0.86 ms.
constexpr int64_t VC4_MIN_FLOPS = int64_t(1) << 18;

// GGML_VC4_STATS=1: at exit, what the QPUs took and what the buffers cost (stderr). Not what they
// declined: the scheduler asks supports_op several times per node, so this side cannot count it.
struct Stats {
    uint64_t mul_mats = 0, convs = 0, convts = 0, launches = 0, fused = 0, cpu_nodes = 0, refused = 0;
    double flops = 0, qpu_s = 0, host_s = 0;
    // Every ggml buffer of this type is a CMA allocation (dma-heap + vcsm import), far slower than
    // malloc, and an engine that makes VC4 its primary allocates inputs and outputs here too.
    uint64_t buffers = 0, host_buffers = 0; double buffer_bytes = 0, host_buffer_bytes = 0, alloc_s = 0;
    ~Stats() {
        if (!getenv("GGML_VC4_STATS")) return;
        fprintf(stderr, "ggml-vc4: %llu MUL_MATs, %llu CONV_2Ds and %llu CONV_TRANSPOSE_1Ds in %llu launches, %.3f GFLOP, QPU %.3f s, "
                        "host side %.3f s; %llu nodes folded into convolutions, %llu run by the CPU backend (%llu for want of GPU memory); "
                        "%llu buffers, %.1f MB, allocated in %.3f s; %llu VC4_HOST buffers, %.1f MB\n",
                (unsigned long long) mul_mats, (unsigned long long) convs, (unsigned long long) convts, (unsigned long long) launches,
                flops * 1e-9,
                qpu_s, host_s, (unsigned long long) fused, (unsigned long long) cpu_nodes, (unsigned long long) refused,
                (unsigned long long) buffers, buffer_bytes * 1e-6, alloc_s, (unsigned long long) host_buffers, host_buffer_bytes * 1e-6);
    }
};
Stats g_stats;

// A weight in a kernel's own layout, packed on first use: a convolution's in the uniform stream's
// (qpu_conv1d.h), a block-quantized MUL_MAT's in qpu_gemm_qb.h's.
// P4.47: in slabs (packed_weights), at `off` in one shared with other weights' copies.
struct Packed {
    std::shared_ptr<vc4::GpuBuffer> slab;
    uint32_t off;
    const ggml_backend_buffer_t owner;       // the vc4 buffer holding the weight it was packed from
    uint32_t bus() const { return slab->bus() + off; }
    uint8_t * base() const { return slab->base() + off; }
};
enum PackKind { PACK_CONV_F32, PACK_QB, PACK_WT_PAD, PACK_CONVT_F32 };

// The mailbox call that starts the QPUs blocks until they finish (and for 8 or more of them sometimes
// returns early, which is why completion is the done markers), so a launch that should overlap host
// work -- staging a convolution's next chunk (P4.46) -- makes that call from this thread. It sleeps in
// the kernel while the QPUs run, leaving the Pi Zero's one core to the caller.
class Launcher {
public:
    explicit Launcher(vc4::Mailbox & mbox) : mbox_(mbox), thread_([this] { loop(); }) {}
    ~Launcher() {
        { std::lock_guard<std::mutex> l(mu_); stop_ = true; }
        cv_.notify_all();
        thread_.join();
    }
    // Hands the call to the thread once the previous one has returned.
    void start(uint32_t nq, uint32_t ctrl) {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return !pending_; });
        nq_ = nq; ctrl_ = ctrl; pending_ = true;
        cv_.notify_all();
    }
    // Waits for the call to return; rethrows what it threw.
    void drain() {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return !pending_; });
        if (!error_.empty()) {
            const std::string e = error_;
            error_.clear();
            throw std::runtime_error(e);
        }
    }

private:
    void loop() {
        std::unique_lock<std::mutex> l(mu_);
        for (;;) {
            cv_.wait(l, [&] { return pending_ || stop_; });
            if (stop_) return;
            const uint32_t nq = nq_, ctrl = ctrl_;
            l.unlock();
            std::string err;
            try { mbox_.execute(nq, ctrl); } catch (const std::exception & e) { err = e.what(); }
            l.lock();
            if (!err.empty()) error_ = err;
            pending_ = false;
            cv_.notify_all();
        }
    }
    vc4::Mailbox & mbox_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool pending_ = false, stop_ = false;
    uint32_t nq_ = 0, ctrl_ = 0;
    std::string error_;
    std::thread thread_;
};

struct Runtime {
    std::mutex mu;
    std::unique_ptr<vc4::Mailbox> mbox;
    std::unique_ptr<Launcher> launcher;
    std::unique_ptr<vc4::GpuBuffer> fixed;   // uncached: control list @0, done words @4096, code @8192
    std::unique_ptr<vc4::GpuBuffer> arena;   // cached: packed X / staged x + uniforms, grown on demand
    std::unique_ptr<vc4::GpuBuffer> codes;   // uncached: the generated kernels, bump-allocated
    size_t codes_used = 0;
    std::map<std::string, uint32_t> code_at;
    // Keyed by the weight's data pointer, the layout and its size (a view of a weight at its start is
    // another weight), and dropped when its buffer is freed: a later model's weight can come back at the
    // same address.
    std::map<std::tuple<const void *, int, size_t>, Packed> packed;
    std::shared_ptr<vc4::GpuBuffer> slab;   // the slab packed_weights fills now, `slab_used` bytes of it
    size_t slab_used = 0;
    // P4.47: VC4_Q8 weights back in ggml's Q8_0 layout, in VC4_HOST, for the CPU to run a node over one
    // when the pool refuses its call (q8_cpu_copy); dropped with the VC4_Q8 buffer.
    struct CpuCopy {
        ggml_context * ctx;
        ggml_backend_buffer_t buf;
        ggml_tensor * t;
        ggml_backend_buffer_t owner;
    };
    std::map<std::tuple<const void *, int, size_t>, CpuCopy> cpu_copies;
    // The CPU backend the nodes this backend claims but cannot fold into a convolution run on (P4.46).
    ggml_backend_t cpu = nullptr;
    uint32_t ctrl = 0, done = 0, code = 0;
    uint32_t seq = 0;
    std::string unavailable;                 // empty when the QPUs are usable
};

std::tuple<const void *, int, size_t> pack_key(const ggml_tensor * w, PackKind kind) {
    return std::make_tuple(static_cast<const void *>(w->data), int(kind), ggml_nbytes(w));
}

Runtime & runtime() {
    static Runtime rt;
    static std::once_flag once;
    std::call_once(once, [] {
        try {
            if (access("/dev/vcio", F_OK) != 0) { rt.unavailable = "no /dev/vcio (not a VideoCore IV Raspberry Pi)"; return; }
            rt.mbox = std::make_unique<vc4::Mailbox>();  // refuses under KMS, stops if qpu_enable fails
            rt.launcher = std::make_unique<Launcher>(*rt.mbox);
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
    std::vector<Runtime::CpuCopy> gone;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        for (auto it = rt.packed.begin(); it != rt.packed.end();) {
            it = it->second.owner == buffer ? rt.packed.erase(it) : std::next(it);
        }
        rt.slab.reset();   // a slab goes with the last copy in it; the next copy starts another
        for (auto it = rt.cpu_copies.begin(); it != rt.cpu_copies.end();) {
            if (it->second.owner != buffer) { ++it; continue; }
            gone.push_back(it->second);
            it = rt.cpu_copies.erase(it);
        }
    }
    for (auto & c : gone) {   // outside the lock: freeing a VC4_HOST buffer takes it
        ggml_backend_buffer_free(c.buf);
        ggml_free(c.ctx);
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
ggml_backend_buffer_type_t vc4_host_buffer_type();
extern const ggml_backend_buffer_i q8_buffer_iface;
bool debug_on();

const char * buft_get_name(ggml_backend_buffer_type_t buft) {
    return buft == vc4_q8_buffer_type() ? "VC4_Q8" : buft == vc4_host_buffer_type() ? "VC4_HOST" : "VC4";
}

// A buffer ggml needs (weights, the scheduler's compute buffers) comes before what this backend keeps
// for speed: when the pool cannot provide one, the packed weights and the arena are let go and the
// allocation tried again. The packed copies grow while the pool keeps VC4_POOL_RESERVE free, and the
// scheduler reserves a graph's buffers when it first sees the graph -- after earlier graphs have packed
// theirs: VITS's vocoder asked for 16 MB once its text encoder's kernels held the rest (P4.46). Never
// during a computation (the scheduler allocates before it computes), so nothing in use is freed.
std::unique_ptr<vc4::GpuBuffer> alloc_or_release(uint32_t bytes) {
    try {
        return std::make_unique<vc4::GpuBuffer>(bytes, /*cached=*/true);
    } catch (const std::exception &) {
        Runtime & rt = runtime();
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            if (rt.packed.empty() && !rt.arena) throw;
            if (debug_on())
                fprintf(stderr, "ggml-vc4: %u bytes refused; releasing %zu packed weights and the arena\n", bytes, rt.packed.size());
            rt.packed.clear();
            rt.slab.reset();
            rt.arena.reset();
        }
        return std::make_unique<vc4::GpuBuffer>(bytes, /*cached=*/true);
    }
}
void note_refused();

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto ctx = std::make_unique<BufferCtx>();
        ctx->mem = alloc_or_release(uint32_t((std::max<size_t>(size, 1) + 4095) & ~size_t(4095)));
        g_stats.buffers++;
        g_stats.buffer_bytes += double(size);
        g_stats.alloc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ggml_backend_buffer_init(buft, buft == vc4_q8_buffer_type() ? q8_buffer_iface : buffer_iface, ctx.release(), size);
    } catch (const std::exception & e) {
        note_refused();
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
bool is_vc4_host_buffer(const ggml_backend_buffer_t b);
// A weight this backend packs from: in its pool (vc4) or in its RAM (VC4_HOST). Either way a buffer of
// this backend's, whose free drops the packed copies.
bool holds_weights(const ggml_backend_buffer_t b) { return is_vc4_buffer(b) || is_vc4_host_buffer(b); }

// Where a weight lives. A view of one has no buffer of its own until the allocator initialises it,
// which is after the scheduler has asked supports_op -- so every reshaped kernel read as "not
// allocated" (P4.45: amy's F32 convolutions, all of them) -- but its root always has one.
ggml_backend_buffer_t weight_buffer(const ggml_tensor * w) { return w->view_src ? w->view_src->buffer : w->buffer; }

void cma_memory(size_t * free, size_t * total);
void pool_memory(size_t * free, size_t * total);

// ---- the VC4_HOST buffer type (P4.47) ----
//
// Ordinary memory, for a model's weights: what the QPUs read of a weight is a copy in their own layout
// (packed_weights) -- a convolution's packed blocks, a Q4_0 / Q8_0 MUL_MAT's repacked rows, an F32
// MUL_MAT's (padded) rows -- so the weight itself need not be in the GPU's pool, where it held 64 MB's
// worth of space beside those copies: VITS Q8_0_PC's 29 MB of weights, of which the QPUs read none in
// place once its convolutions were packed. Offered as the device's host buffer type, which loom puts a
// model's weights in for a device whose own buffers are host memory (weights_buffer_type). A buffer of
// this type is the vc4 backend's so that freeing it drops what was packed from it.

struct HostBufferCtx {
    void * base = nullptr;
};
void host_buffer_free(ggml_backend_buffer_t buffer) {
    Runtime & rt = runtime();
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        for (auto it = rt.packed.begin(); it != rt.packed.end();) {
            it = it->second.owner == buffer ? rt.packed.erase(it) : std::next(it);
        }
        rt.slab.reset();
    }
    auto * ctx = static_cast<HostBufferCtx *>(buffer->context);
    free(ctx->base);
    delete ctx;
}
void * host_buffer_get_base(ggml_backend_buffer_t buffer) { return static_cast<HostBufferCtx *>(buffer->context)->base; }
void host_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) { memset(host_buffer_get_base(buffer), value, buffer->size); }

const ggml_backend_buffer_i host_buffer_iface = {
    /* .free_buffer   = */ host_buffer_free,
    /* .get_base      = */ host_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buffer_memset_tensor,
    /* .set_tensor    = */ buffer_set_tensor,
    /* .get_tensor    = */ buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ buffer_cpy_tensor,
    /* .clear         = */ host_buffer_clear,
    /* .reset         = */ nullptr,
};

ggml_backend_buffer_t host_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * base = nullptr;
    if (posix_memalign(&base, 64, std::max<size_t>(size, 64)) != 0) return nullptr;
    auto * ctx = new HostBufferCtx{base};
    g_stats.host_buffers++;
    g_stats.host_buffer_bytes += double(size);
    return ggml_backend_buffer_init(buft, host_buffer_iface, ctx, size);
}
bool is_vc4_host_buffer(const ggml_backend_buffer_t b) { return b != nullptr && b->buft == vc4_host_buffer_type(); }

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

// Start nq QPUs on `code` with the uniform streams in `starts`; launch_wait waits for every done
// marker. Between the two the caller may work on the host (the mailbox call is the launcher thread's),
// as long as it does not touch the control list -- the next launch writes it after this one's wait.
using clk = std::chrono::steady_clock;
clk::time_point launch_start(Runtime & rt, int nq, const std::vector<uint32_t> & starts, uint32_t code) {
    for (int q = 0; q < nq; q++) {
        rt.fixed->at(rt.ctrl)[2 * q] = starts[q];
        rt.fixed->at(rt.ctrl)[2 * q + 1] = code;
    }
    __sync_synchronize();
    const clk::time_point t0 = clk::now();
    rt.launcher->start(uint32_t(nq), rt.ctrl);
    return t0;
}
void launch_wait(Runtime & rt, int nq, uint32_t seq, const std::string & what, clk::time_point t0) {
    for (;;) {
        int ok = 0;
        for (int q = 0; q < nq; q++) ok += *reinterpret_cast<volatile uint32_t *>(rt.fixed->at(rt.done + q * 64)) == seq;
        if (ok == nq) break;
        if (clk::now() - t0 > std::chrono::seconds(2)) {
            // A QPU program that did not finish leaves the QPUs stuck; disabling them then has wedged
            // the firmware, and so has a reboot, so stop here and say what is needed.
            GGML_ABORT("ggml-vc4: %d of %d QPUs did not finish %s in 2 s; the board needs a power cycle",
                       nq - ok, nq, what.c_str());
        }
    }
    rt.launcher->drain();
    g_stats.qpu_s += std::chrono::duration<double>(clk::now() - t0).count();
    g_stats.launches++;
}
void launch_and_wait(Runtime & rt, int nq, const std::vector<uint32_t> & starts, uint32_t code, uint32_t seq,
                     const std::string & what) {
    launch_wait(rt, nq, seq, what, launch_start(rt, nq, starts, code));
}

// The arena for this call (rt.mu held). When the pool refuses it, the packed weights other than the
// one this call is using (`keep`) are let go and it is tried again at exactly the size needed.
// GGML_VC4_ARENA_MAX_KB caps it, so the board test can make the pool refuse a call and check that the
// node runs on the CPU instead (P4.47).
void ensure_arena(Runtime & rt, size_t need, const void * keep = nullptr) {
    static const size_t cap = [] { const char * e = getenv("GGML_VC4_ARENA_MAX_KB"); return e ? size_t(atoll(e)) << 10 : SIZE_MAX; }();
    if (need > cap) throw std::runtime_error("arena of " + std::to_string(need) + " bytes above GGML_VC4_ARENA_MAX_KB");
    if (rt.arena && rt.arena->size() >= need) return;
    rt.arena.reset();
    try {
        rt.arena = std::make_unique<vc4::GpuBuffer>(uint32_t((need + need / 4 + 65535) & ~size_t(65535)), /*cached=*/true);
    } catch (const std::exception &) {
        for (auto it = rt.packed.begin(); it != rt.packed.end();) it = std::get<0>(it->first) == keep ? std::next(it) : rt.packed.erase(it);
        rt.slab.reset();
        rt.arena = std::make_unique<vc4::GpuBuffer>(uint32_t((need + 65535) & ~size_t(65535)), /*cached=*/true);
    }
}

// ---- weights packed into a kernel's own layout ----

// Packed once and kept while the pool keeps a reserve for the arenas and the scheduler's buffers;
// past that, null, and the caller packs into the arena for this call alone. Kept unconditionally,
// VITS's text-encoder kernels (192 -> 768 channels, 1.5-1.7 MB each in F32) filled the 64 MB pool
// before the vocoder's were reached, and the synthesis failed to allocate.
//
// In slabs of VC4_SLAB_BYTES, not an allocation a weight: each GPU allocation is a dma-heap buffer
// imported into vcsm, ~2 ms on the Pi Zero, and with its weights in RAM (P4.47) conformer-ctc's 195 F32
// matrices each got a copy on first use -- 0.45 s more on its first inference than P4.46, which had
// read them in place (later inferences equal). A slab is freed with the last copy in it.
constexpr size_t VC4_POOL_RESERVE = size_t(24) << 20;
constexpr size_t VC4_SLAB_BYTES = size_t(2) << 20;
Packed * packed_weights(Runtime & rt, const ggml_tensor * w, PackKind kind, size_t bytes,
                        const std::function<void(uint8_t *)> & pack) {
    const auto key = pack_key(w, kind);
    auto it = rt.packed.find(key);
    if (it != rt.packed.end()) return &it->second;
    const size_t span = (bytes + 63) & ~size_t(63);
    if (!rt.slab || rt.slab_used + span > rt.slab->size()) {
        const size_t sbytes = (std::max(span, VC4_SLAB_BYTES) + 4095) & ~size_t(4095);
        size_t free_bytes = 0, total = 0;
        pool_memory(&free_bytes, &total);
        if (free_bytes < sbytes + VC4_POOL_RESERVE) return nullptr;
        try {
            rt.slab = std::make_shared<vc4::GpuBuffer>(uint32_t(sbytes), /*cached=*/true);
        } catch (const std::exception &) {
            rt.slab.reset();
            return nullptr;   // the estimate was optimistic (pool_memory): pack for this call instead
        }
        rt.slab_used = 0;
    }
    const uint32_t off = uint32_t(rt.slab_used);
    rt.slab_used += span;
    uint8_t * at = rt.slab->base() + off;
    pack(at);
    rt.slab->clean(at, bytes);
    return &rt.packed.emplace(key, Packed{rt.slab, off, weight_buffer(w)}).first->second;
}

// The tile order (qpu_gemm.h's gemm_tiles): panels of two column blocks. Measured on the Pi Zero
// (P4.46, gemm_f32_wt_kernel, 12 QPUs, rows padded where gemm_row_stride says): against m-major,
// 1.17-1.36x at every shape the models run -- 384x457x192 8.9 -> 11.0 GFLOPS, 192x3000x576 9.2 ->
// 12.5, 128x3656x384 9.1 -> 12.3, 768x156x768 10.0 -> 11.9 -- while panels of one ran 2.6 GFLOPS and
// panels of three or four fell between. GGML_VC4_PANEL=P overrides it (0: m-major) for measuring.
int tile_panel(int /*M*/, int /*N*/, int /*K*/) {
    static const int env = [] { const char * e = getenv("GGML_VC4_PANEL"); return e ? atoi(e) : -1; }();
    return env >= 0 ? env : 2;
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

struct BlockQ4_0 { ggml_fp16_t d; uint8_t qs[16]; };  // ggml's block_q4_0
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 layout");

// P4.46: Q8_0 and Q4_0 weights with a scale per block, where the model put them (a vc4 buffer, ggml's
// layout), repacked once into qpu_gemm_qb.h's rows -- or per call, into the arena, when the pool is
// short. From 16 columns, as the per-channel kernel.
bool mul_mat_qb_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if ((w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_Q4_0) || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!holds_weights(w->buffer) || w->view_src != nullptr || w->op != GGML_OP_NONE || !ggml_is_contiguous(w)) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] % vc4::QB_BLOCK) return false;
    if (x->nb[0] != 4 || x->nb[1] % 4 || op->nb[0] != 4 || op->nb[1] % 4) return false;
    if (op->nb[1] > 8191 + 4) return false;  // VDW row stride is 13 bits
    if (x->ne[1] < 16) return false;
    if (2 * op->ne[0] * op->ne[1] * w->ne[0] * op->ne[2] * op->ne[3] < VC4_MIN_FLOPS) return false;
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    if (rt.packed.count(pack_key(w, PACK_QB)) == 0) {
        size_t free_bytes = 0, total = 0;
        pool_memory(&free_bytes, &total);
        const size_t have = free_bytes + (rt.arena ? rt.arena->size() : 0);
        if (have < vc4::gemm_qb_weight_bytes(int(w->ne[1]), int(w->ne[0])) + (size_t(8) << 20)) return false;
    }
    return true;
}

bool mul_mat_supported(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (is_vc4_q8_buffer(weight_buffer(w))) return mul_mat_q8_supported(op);
    if (w->type == GGML_TYPE_Q8_0 || w->type == GGML_TYPE_Q4_0) return mul_mat_qb_supported(op);
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    // The weights must be a leaf (or a view of one) in a vc4 buffer, where the QPUs read them in place
    // -- only leaves carry the tail the TMU reads past the last row -- or in VC4_HOST, read through a
    // copy in the pool.
    if (!holds_weights(weight_buffer(w))) return false;
    if ((w->view_src ? w->view_src->op : w->op) != GGML_OP_NONE) return false;
    if (is_vc4_host_buffer(weight_buffer(w))) {
        if (w->nb[0] != 4 || w->nb[1] % 4) return false;
        Runtime & rt = runtime();
        std::lock_guard<std::mutex> lock(rt.mu);
        if (rt.packed.count(pack_key(w, PACK_WT_PAD)) == 0) {
            size_t free_bytes = 0, total = 0;
            pool_memory(&free_bytes, &total);
            const size_t have = free_bytes + (rt.arena ? rt.arena->size() : 0);
            if (have < size_t(w->ne[1] * w->ne[2] * w->ne[3]) * vc4::gemm_row_stride(w->nb[1]) + (size_t(8) << 20)) return false;
        }
    }
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
    // in the pool, read in place; or in VC4_HOST (P4.47), read through a copy (`copied` below)
    vc4::GpuBuffer * wmem = is_vc4_buffer(weight_buffer(w)) ? &mem_of(weight_buffer(w)) : nullptr;
    vc4::GpuBuffer & cmem = mem_of(dst->buffer);
    const int K = int(w->ne[0]), M = int(w->ne[1]), N = int(x->ne[1]);
    const int64_t r2 = x->ne[2] / w->ne[2], r3 = x->ne[3] / w->ne[3];
    const size_t xslice = vc4::gemm_wt_xpack_bytes(N, K);

    // Rows a multiple of 512 bytes apart thrash the TMU cache (gemm_row_stride): read a copy with
    // padded rows instead -- kept like a packed convolution weight, or made for this call when it is
    // small; a large weight the pool cannot keep a copy of is read in place, slow as before. A weight in
    // VC4_HOST is always read through such a copy, padded or not, made for this call when not kept,
    // every slice of a batched one (a slice after another, M rows apart).
    const size_t pstride = vc4::gemm_row_stride(w->nb[1]);
    const int64_t slices = w->ne[2] * w->ne[3];
    const size_t sbytes = size_t(M) * pstride, pbytes = size_t(slices) * sbytes + vc4::gemm_wt_tail_bytes();
    const bool pad = wmem == nullptr || (pstride != w->nb[1] && slices == 1);
    auto copy_padded = [&](uint8_t * o) {
        for (int64_t i3 = 0; i3 < w->ne[3]; i3++)
            for (int64_t i2 = 0; i2 < w->ne[2]; i2++) {
                const uint8_t * src = static_cast<const uint8_t *>(w->data) + i2 * w->nb[2] + i3 * w->nb[3];
                uint8_t * dsl = o + size_t(i3 * w->ne[2] + i2) * sbytes;
                for (int m = 0; m < M; m++) memcpy(dsl + size_t(m) * pstride, src + size_t(m) * w->nb[1], size_t(K) * 4);
            }
        memset(o + size_t(slices) * sbytes, 0, vc4::gemm_wt_tail_bytes());
    };
    Packed * pk = pad ? packed_weights(rt, w, PACK_WT_PAD, pbytes, copy_padded) : nullptr;
    const size_t wcall = pad && pk == nullptr && (pbytes <= (size_t(1) << 20) || wmem == nullptr) ? (pbytes + 63) & ~size_t(63) : 0;
    const bool copied = pk != nullptr || wcall != 0;

    std::vector<vc4::GemmWtProblem> probs;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
            const uint8_t * wp = static_cast<const uint8_t *>(w->data) + (i2 / r2) * w->nb[2] + (i3 / r3) * w->nb[3];
            uint8_t * cp = static_cast<uint8_t *>(dst->data) + i2 * dst->nb[2] + i3 * dst->nb[3];
            probs.push_back({M, N, K, copied ? 0u : wmem->bus_of(wp), uint32_t(w->nb[1]), cmem.bus_of(cp), uint32_t(dst->nb[1]), 0});
        }
    const size_t need = wcall + probs.size() * xslice + vc4::gemm_wt_uniform_words(probs, 12) * 4 + 4096;
    ensure_arena(rt, need, w->data);
    if (copied) {   // kept, or the arena's first wcall bytes (the packed X follows)
        if (pk == nullptr) copy_padded(rt.arena->base());
        const uint32_t base = pk != nullptr ? pk->bus() : rt.arena->bus();
        size_t k = 0;
        for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
            for (int64_t i2 = 0; i2 < x->ne[2]; i2++, k++) {
                probs[k].w = base + uint32_t(size_t((i3 / r3) * w->ne[2] + i2 / r2) * sbytes);
                probs[k].w_stride = uint32_t(pstride);
            }
    }
    uint8_t * xbase = rt.arena->base() + wcall;
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
    const int nq = vc4::gemm_wt_uniforms(cpu, unif, probs, 12, rt.done, seq, starts, vc4::GEMM_OUT_N, tile_panel(M, N, K));
    // Coherence before the launch: the weights may have been written by the CPU (loading, or an
    // activation the CPU computed), the packed X and uniforms just were; dst must hold no dirty
    // lines that could be written back over what the QPUs store.
    rt.arena->clean(rt.arena->base(), need);
    if (!copied)
        for (int64_t i3 = 0; i3 < w->ne[3]; i3++)
            for (int64_t i2 = 0; i2 < w->ne[2]; i2++) {
                const uint8_t * wp = static_cast<const uint8_t *>(w->data) + i2 * w->nb[2] + i3 * w->nb[3];
                wmem->clean(wp, size_t(M - 1) * w->nb[1] + size_t(K) * 4 + vc4::gemm_wt_tail_bytes());
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
    const int K = int(w->ne[0]), M = int(w->ne[1]), N = int(x->ne[1]);
    const size_t xslice = vc4::gemm_q8_xpack_bytes(N, K);
    const uint32_t wbus = wmem.bus_of(w->data), ws = vc4::gemm_q8_stride(K);
    std::vector<vc4::GemmQ8Problem> probs;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++) {
            uint8_t * cp = static_cast<uint8_t *>(dst->data) + i2 * dst->nb[2] + i3 * dst->nb[3];
            probs.push_back({M, N, K, wbus, ws, wbus + uint32_t(vc4::gemm_q8_scales_at(M, K)), cmem.bus_of(cp), uint32_t(dst->nb[1]), 0});
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
    const int nq = vc4::gemm_q8_uniforms(cpu, unif, probs, 12, rt.done, seq, starts, vc4::GEMM_OUT_N, tile_panel(M, N, K));
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

void repack_qb(const ggml_tensor * w, uint8_t * out) {
    const int K = int(w->ne[0]), M = int(w->ne[1]);
    auto d2f = [](ggml_fp16_t h) { return ggml_fp16_to_fp32(h); };
    if (w->type == GGML_TYPE_Q4_0) vc4::gemm_qb_repack(static_cast<const BlockQ4_0 *>(w->data), M, K, d2f, out);
    else vc4::gemm_qb_repack(static_cast<const BlockQ8_0 *>(w->data), M, K, d2f, out);
}

void mul_mat_qb(ggml_tensor * dst) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    const auto t_host0 = std::chrono::steady_clock::now();
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    vc4::GpuBuffer & cmem = mem_of(dst->buffer);
    const int K = int(w->ne[0]), M = int(w->ne[1]), N = int(x->ne[1]);
    const size_t wbytes = vc4::gemm_qb_weight_bytes(M, K), xslice = vc4::gemm_qb_xpack_bytes(N, K);
    Packed * pk = packed_weights(rt, w, PACK_QB, wbytes, [&](uint8_t * out) { repack_qb(w, out); });
    const size_t wcall = pk ? 0 : (wbytes + 63) & ~size_t(63);   // repacked into the arena for this call
    const size_t nslices = size_t(x->ne[2] * x->ne[3]);
    std::vector<vc4::GemmQbProblem> probs(nslices, vc4::GemmQbProblem{M, N, K, 0, 0, 0, 0});
    const size_t need = wcall + nslices * xslice + vc4::gemm_qb_uniform_words(probs, 12) * 4 + 4096;
    ensure_arena(rt, need, w->data);
    uint8_t * abase = rt.arena->base();
    if (!pk) repack_qb(w, abase);
    const uint32_t wbus = pk ? pk->bus() : rt.arena->bus();
    size_t k = 0;
    for (int64_t i3 = 0; i3 < x->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < x->ne[2]; i2++, k++) {
            uint8_t * cp = static_cast<uint8_t *>(dst->data) + i2 * dst->nb[2] + i3 * dst->nb[3];
            float * out = reinterpret_cast<float *>(abase + wcall + k * xslice);
            const uint8_t * xs = static_cast<const uint8_t *>(x->data) + i2 * x->nb[2] + i3 * x->nb[3];
            vc4::gemm_qb_pack_x(reinterpret_cast<const float *>(xs), x->nb[1] / 4, 1, N, K, out);
            probs[k] = {M, N, K, wbus, cmem.bus_of(cp), uint32_t(dst->nb[1]), rt.arena->bus_of(out)};
        }
    const uint32_t unif = rt.arena->bus_of(abase + ((wcall + nslices * xslice + 63) & ~size_t(63)));
    const uint32_t seq = ++rt.seq;
    std::vector<uint32_t> starts;
    auto cpu = [&](uint32_t bus) { return rt.arena->at(bus); };
    const int nq = vc4::gemm_qb_uniforms(cpu, unif, probs, 12, rt.done, seq, starts, vc4::GEMM_OUT_N, tile_panel(M, N, K));
    const uint32_t code = program_at(rt, "gemm_qb", [] { return vc4::gemm_qb_kernel(); });
    rt.arena->clean(abase, need);
    cmem.flush(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_host0).count();
    launch_and_wait(rt, nq, starts, code, seq,
                    std::string("a ") + ggml_type_name(w->type) + " MUL_MAT [" + std::to_string(M) + " x " + std::to_string(N) +
                    " x " + std::to_string(K) + "]");
    const auto t1 = std::chrono::steady_clock::now();
    cmem.invalidate(dst->data, ggml_nbytes(dst));
    g_stats.host_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    g_stats.mul_mats++;
    g_stats.flops += 2.0 * M * N * K * double(nslices);
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
        if (seen[key]++ == 0) fprintf(stderr, "ggml-vc4: %s declined: %s\n", ggml_op_desc(op), key.c_str());
    }
    return false;
}

// GGML_VC4_CONV=0 leaves every convolution to the CPU (the A/B switch); GGML_VC4_FUSE=0 leaves the
// elementwise nodes around them to it too (P4.46's A/B switch).
bool conv_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_VC4_CONV"); return !(e && strcmp(e, "0") == 0); }();
    return on;
}
// Off for the rest of the process once the pool has refused a buffer the scheduler asked for: what this
// backend claims to fold is then allocated there too, and the CPU split holds it in ordinary memory.
// loom re-splits once after a refused allocation (GraphBuilder::allocate_scheduled), and fewer claims is
// what lets that second plan fit: Citrinet's whole residual stream fitted the pool as the CPU's
// tensors and not as ours (36 MB refused, P4.46).
std::atomic<bool> g_fuse_refused{false};
bool fuse_enabled_env() {
    static const bool on = [] { const char * e = getenv("GGML_VC4_FUSE"); return conv_enabled() && !(e && strcmp(e, "0") == 0); }();
    return on;
}
bool fuse_enabled() { return fuse_enabled_env() && !g_fuse_refused.load(); }
// The nodes backend_graph_optimize made views of a convolution for the graph being planned now: the
// scheduler allocates right after the splits are optimized, and a computation starting means it got
// its buffers. When it did not, they are made ordinary nodes again before loom re-splits -- computed in
// place by the CPU they still kept the chain in the pool, and Citrinet F32 (39 MB of weights) was
// refused its buffers a second time. Only nodes still unallocated: that is a graph being planned.
std::mutex g_inplaced_mu;
std::vector<ggml_tensor *> g_inplaced;
void undo_inplace() {
    std::lock_guard<std::mutex> lock(g_inplaced_mu);
    for (ggml_tensor * n : g_inplaced)
        if (n->data == nullptr && n->buffer == nullptr) {
            n->view_src = nullptr;
            n->view_offs = 0;
        }
    g_inplaced.clear();
}

void note_refused() {
    undo_inplace();
    if (!g_fuse_refused.exchange(true) && fuse_enabled_env())
        GGML_LOG_WARN("ggml-vc4: the GPU pool refused a buffer; the elementwise nodes around convolutions stay on the CPU from here on\n");
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

// See conv_run(). GGML_VC4_STAGE_KB overrides it, for measuring.
size_t conv_stage_bytes() {
    static const size_t v = [] { const char * e = getenv("GGML_VC4_STAGE_KB"); return e ? size_t(atoi(e)) << 10 : size_t(1) << 20; }();
    return v;
}
constexpr int VC4_STAGE_SLOTS = 3;

// P4.46: a 1x1 convolution over a per-channel Q8_0 weight loom copied into VC4_Q8 (Citrinet's
// pointwise convolutions, which CPU_ARMV6_I8 holds) runs as gemm_q8_kernel with the output transposed:
// M = OC, N = T, K = IC, X packed from x's [IC][T] rows.
bool conv_q8_supported(const ggml_tensor * op, const vc4::Conv1dShape & c) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w->type != GGML_TYPE_Q8_0 || w->ne[2] != 1 || w->ne[3] != 1 || w->view_src != nullptr) return decline(op, "VC4_Q8 not 2-D");
    if (c.KW != 1 || c.s != 1 || c.p != 0 || c.T_in != c.T_out || w->ne[0] != c.IC) return decline(op, "VC4_Q8 not 1x1");
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || x->nb[0] != 4 || x->nb[2] % 4) return decline(op, "types");
    if (!ggml_is_contiguous(op) || !vc4::gemm_out_ok(uint32_t(op->nb[2]))) return decline(op, "VC4_Q8 output rows");
    if (c.T_out < 16 || 2 * int64_t(c.OC) * c.T_out * c.IC < VC4_MIN_FLOPS) return decline(op, "below the flop threshold");
    return true;
}

// A weight not packed yet needs its F32 copy to fit in the GPU's pool beside what the arenas grow to;
// declining here costs nothing, failing to allocate at compute time would. What the call needs from
// the pool if nothing is cached: x staged (padded) and the weights packed into the arena, beside what
// the arena already holds.
bool conv_pool_fits(const ggml_tensor * w, PackKind kind, const vc4::Conv1dShape & c) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    if (rt.packed.count(pack_key(w, kind)) != 0) return true;
    size_t free_bytes = 0, total = 0;
    pool_memory(&free_bytes, &total);
    const size_t need = vc4::conv1d_wpack_floats(c.OC, c.IC * c.KW) * 4 + conv_stage_bytes() * (VC4_STAGE_SLOTS + 1);
    const size_t have = free_bytes + (rt.arena ? rt.arena->size() : 0);
    return have >= need + (size_t(4) << 20);
}

bool conv_supported(const ggml_tensor * op) {
    if (!conv_enabled()) return false;
    vc4::Conv1dShape c;
    if (!conv_geom(op, c)) return decline(op, "not 1-D");
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (is_vc4_q8_buffer(weight_buffer(w))) return conv_q8_supported(op, c);
    // An F32 kernel has no to_float -- it needs none -- which the first version read as "no way to
    // convert" and so declined every F32 convolution.
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr)) {
        return decline(op, "types");
    }
    // The weights are packed once from where they live, which must be this backend's buffer: its free
    // is what drops the packed copy.
    if (!holds_weights(weight_buffer(w))) {
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
    if (!conv_pool_fits(w, PACK_CONV_F32, c)) return decline(op, "GPU pool full");
    if (debug_on()) {
        static std::mutex mu;
        static std::unordered_map<std::string, int> seen;
        const std::string key = std::string(ggml_get_name(op)) + " " + ggml_type_name(w->type) + " T " + std::to_string(c.T_in);
        std::lock_guard<std::mutex> lk(mu);
        if (seen[key]++ == 0) fprintf(stderr, "ggml-vc4: CONV_2D accepted: %s\n", key.c_str());
    }
    return true;
}

// P4.47: a CONV_TRANSPOSE_1D as qpu_conv1d.h's polyphase convolution, its OC*s virtual channels
// written straight into ggml's [OC][(T-1)*s + K] rows. Kernel [K, OC, IC] with K a multiple of the
// stride, data [T, IC], no padding or dilation (the only form ggml has). HiFi-GAN's upsampling stages
// -- VITS's four, piper's three -- were what was left of their vocoders on the CPU: ~1.5 s of amy's
// 5.4 s and of VITS's 7.6 s under P4.46's "auto".
bool convt_geom(const ggml_tensor * op, vc4::Conv1dShape & c, int & s) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    const int32_t * pp = reinterpret_cast<const int32_t *>(op->op_params);
    s = pp[0];
    const int64_t K = w->ne[0], OC = w->ne[1], IC = w->ne[2];
    if (pp[1] != 0 || pp[2] != 1 || !vc4::conv1d_phases_ok(s) || K % s != 0) return false;
    if (w->ne[3] != 1 || x->ne[1] != IC || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (op->ne[0] != (x->ne[0] - 1) * s + K || op->ne[1] != OC || op->ne[2] != 1 || op->ne[3] != 1) return false;
    c = vc4::conv1d_transposed(int(IC), int(OC), int(K), s, int(x->ne[0]));
    return true;
}

bool convt_supported(const ggml_tensor * op) {
    if (!conv_enabled()) return false;
    vc4::Conv1dShape c;
    int s = 1;
    if (!convt_geom(op, c, s)) return decline(op, "stride, padding or shape");
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr)) {
        return decline(op, "types");
    }
    if (!holds_weights(weight_buffer(w))) {
        return decline(op, weight_buffer(w) ? ggml_backend_buffer_name(weight_buffer(w)) : "weight not allocated");
    }
    if (!ggml_is_contiguous(w)) return decline(op, "weight not contiguous");
    if ((w->view_src ? w->view_src->op : w->op) != GGML_OP_NONE) return decline(op, "weight computed");
    if (x->nb[0] != 4 || x->nb[1] % 4 || !ggml_is_contiguous(op)) return decline(op, "strides");
    if (!vc4::conv1d_kernel_ok(c.IC, c.KW)) return decline(op, "IC too small for the kernel");
    if (c.T_out < 16) return decline(op, "T < 16");
    if (2 * int64_t(c.OC) * c.T_out * c.IC * c.KW < VC4_MIN_FLOPS) return decline(op, "below the flop threshold");
    if (!conv_pool_fits(w, PACK_CONVT_F32, c)) return decline(op, "GPU pool full");
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

// A transposed convolution's weights in conv1d_pack_wt's layout, dequantized whole first when not F32.
void pack_convt_weights(const ggml_tensor * w, int IC, int OC, int K, int s, float * out) {
    if (w->type == GGML_TYPE_F32) {
        vc4::conv1d_pack_wt(static_cast<const float *>(w->data), IC, OC, K, s, out);
        return;
    }
    std::vector<float> f(size_t(ggml_nelements(w)));
    ggml_get_type_traits(w->type)->to_float(w->data, f.data(), ggml_nelements(w));
    vc4::conv1d_pack_wt(f.data(), IC, OC, K, s, out);
}

// ---- the elementwise nodes around a convolution (P4.46) ----
//
// What ggml's CPU backend fuses into a convolution (ggml-0005/0007/0036): a LEAKY_RELU before it, the
// bias ADD, a residual ADD, a RELU and a time-mask MUL after it. With the convolution here and those
// nodes on the CPU, each was a whole pass over the activation on the ARM1176 -- amy under "auto",
// profiled on the Pi Zero: ADD 1.50 s and LEAKY_RELU 0.69 s of a 5.79 s synthesis, against 1.93 s for
// the convolutions themselves. So this backend claims them (dev_supports_op) and graph_compute folds
// what a convolution in its split can absorb into the QPU epilogue -- or into the staging copy, for
// the LEAKY_RELU. What it claimed and cannot fold runs on the CPU backend, the same kernels the CPU
// split would have run.

const ggml_tensor * view_root(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}
// The node that computed `t`: t itself, or what a reshape / view of it reads. Not view_root: once
// backend_graph_optimize has made a chain compute in place, its nodes are views of the convolution's
// tensor, and view_root would name the convolution for every one of them.
const ggml_tensor * producer(const ggml_tensor * t) {
    for (int depth = 0; depth < 8 && t != nullptr && ggml_op_is_empty(t->op) && t->src[0] != nullptr; depth++) t = t->src[0];
    return t;
}
bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->data == nullptr || b->data == nullptr) return false;
    const char * a0 = static_cast<const char *>(a->data);
    const char * b0 = static_cast<const char *>(b->data);
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}
bool same_memory(const ggml_tensor * a, const ggml_tensor * b) { return a->data == b->data && ggml_nbytes(a) == ggml_nbytes(b); }
bool is_f32_contiguous(const ggml_tensor * t) { return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

// The whole graph's use count of `t`, -1 when the graph does not hold it. A split's graph is a view
// (ggml_graph_view) that shares the full graph's counts, so a reader in another split counts too.
int32_t use_count(const ggml_cgraph * g, const ggml_tensor * t) {
    if (g->use_counts == nullptr) return -1;
    const size_t h = ggml_hash_find(&g->visited_hash_set, t);
    if (h == GGML_HASHSET_FULL || !ggml_bitset_get(g->visited_hash_set.used, h)) return -1;
    return g->use_counts[h];
}

// Whether `consumer` is the only reader of `t` -- directly or through views of it, which loom's
// lowering puts between a convolution and its bias -- and t is not a graph output.
bool private_to(const ggml_cgraph * g, const ggml_tensor * t, const ggml_tensor * consumer) {
    for (int depth = 0; depth < 4; depth++) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || use_count(g, t) != 1) return false;
        for (int s = 0; s < GGML_MAX_SRC; s++)
            if (consumer->src[s] == t) return true;
        const ggml_tensor * v = nullptr;
        for (int i = 0; i < g->n_nodes && v == nullptr; i++)
            if (g->nodes[i]->view_src != nullptr && g->nodes[i]->src[0] == t) v = g->nodes[i];
        if (v == nullptr) return false;
        t = v;
    }
    return false;
}

int next_compute(const ggml_cgraph * g, int from) {
    while (from < g->n_nodes && ggml_op_is_empty(g->nodes[from]->op)) from++;
    if (from >= g->n_nodes || (g->nodes[from]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return -1;
    return from;
}

// Shapes, relative to a convolution's result x seen as [T, ...]: a bias is one value per channel
// (ne0 = 1, the rest x's), a residual is x's shape, a mask one value per position (ne0 = T, the rest 1).
enum Operand { OPERAND_NONE, OPERAND_BIAS, OPERAND_RESIDUAL, OPERAND_MASK };
Operand operand_kind(const ggml_tensor * x, const ggml_tensor * o) {
    if (!is_f32_contiguous(o) || !is_f32_contiguous(x)) return OPERAND_NONE;
    if (ggml_are_same_shape(x, o)) return OPERAND_RESIDUAL;
    bool bias = o->ne[0] == 1, mask = o->ne[0] == x->ne[0];
    for (int i = 1; i < 4; i++) { bias = bias && o->ne[i] == x->ne[i]; mask = mask && o->ne[i] == 1; }
    return bias ? OPERAND_BIAS : mask ? OPERAND_MASK : OPERAND_NONE;
}

// For an ADD or MUL with one source a view of `prev`: the other source, and which kind it is.
const ggml_tensor * operand_of(const ggml_tensor * node, const ggml_tensor * prev, Operand & kind) {
    kind = OPERAND_NONE;
    if (!is_f32_contiguous(node)) return nullptr;
    for (int s = 0; s < 2; s++) {
        const ggml_tensor * x = node->src[s];
        const ggml_tensor * o = node->src[1 - s];
        if (x == nullptr || o == nullptr || producer(x) != prev || ggml_nelements(x) != ggml_nelements(node)) continue;
        kind = operand_kind(x, o);
        if (kind != OPERAND_NONE) return o;
    }
    return nullptr;
}

struct Fusion {
    const ggml_tensor * act = nullptr;     // the LEAKY_RELU whose input is staged in place of its output
    float slope = 0.f;
    ggml_tensor * conv = nullptr;
    const ggml_tensor * bias = nullptr, * residual = nullptr, * mask = nullptr;
    bool relu = false;
    ggml_tensor * out = nullptr;           // the last node covered: where the result goes
    int last = -1;                         // its index in the split
    unsigned epi() const {
        return (bias ? vc4::EPI_BIAS : 0u) | (residual ? vc4::EPI_RESIDUAL : 0u) | (relu ? vc4::EPI_RELU : 0u) |
               (mask ? vc4::EPI_MASK : 0u);
    }
};

// The chain that starts at node i -- a LEAKY_RELU whose only reader is the next node, a convolution,
// or that convolution -- as far as it goes in this split: bias ADD, then residual ADD (only behind a
// bias, as the CPU backend matches it), RELU, mask MUL, each optional, each the only reader of the
// previous result. Ok with nothing after the convolution: f.out is then the convolution itself.
bool fusion_at(const ggml_cgraph * g, int i, Fusion & f) {
    f = Fusion{};
    ggml_tensor * node = g->nodes[i];
    int idx = i;
    if (node->op == GGML_OP_LEAKY_RELU) {
        const ggml_tensor * in = node->src[0];
        if (!fuse_enabled() || !is_f32_contiguous(node) || !is_f32_contiguous(in) || ggml_nbytes(in) != ggml_nbytes(node)) return false;
        idx = next_compute(g, i + 1);
        if (idx < 0 || (g->nodes[idx]->op != GGML_OP_CONV_2D && g->nodes[idx]->op != GGML_OP_CONV_TRANSPOSE_1D)) return false;
        ggml_tensor * conv = g->nodes[idx];
        // Staging the unary's input in place of its output, pointer for pointer: the convolution's own
        // view of `node` still describes what it reads.
        if (conv->src[1] == nullptr || producer(conv->src[1]) != node || is_vc4_q8_buffer(weight_buffer(conv->src[0])) ||
            !private_to(g, node, conv)) {
            return false;
        }
        f.act = node;
        f.slope = ggml_get_op_params_f32(node, 0);
        node = conv;
    }
    if (node->op != GGML_OP_CONV_2D && node->op != GGML_OP_CONV_TRANSPOSE_1D) return false;
    f.conv = node;
    f.out = node;
    f.last = idx;
    // a result the chain cannot be written over: the convolution alone (and the LEAKY_RELU not here)
    if (!is_f32_contiguous(node) || (node->flags & GGML_TENSOR_FLAG_OUTPUT) || !fuse_enabled()) return f.act == nullptr;
    auto take = [&](ggml_op op, Operand want, const ggml_tensor *& slot) {
        const int j = next_compute(g, f.last + 1);
        if (j < 0 || g->nodes[j]->op != op) return false;
        Operand kind;
        const ggml_tensor * o = operand_of(g->nodes[j], f.out, kind);
        if (o == nullptr || kind != want || !private_to(g, f.out, g->nodes[j])) return false;
        slot = o;
        f.out = g->nodes[j];
        f.last = j;
        return true;
    };
    // a transposed convolution's tile holds s phases of 16/s channels, not one channel's positions: its
    // epilogue is the bias alone (all HiFi-GAN puts after one)
    if (node->op == GGML_OP_CONV_TRANSPOSE_1D) {
        take(GGML_OP_ADD, OPERAND_BIAS, f.bias);
        return true;
    }
    if (take(GGML_OP_ADD, OPERAND_BIAS, f.bias)) take(GGML_OP_ADD, OPERAND_RESIDUAL, f.residual);
    {
        const int j = next_compute(g, f.last + 1);
        ggml_tensor * n = j >= 0 ? g->nodes[j] : nullptr;
        if (n != nullptr && n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_RELU && is_f32_contiguous(n) &&
            n->src[0] != nullptr && producer(n->src[0]) == f.out && ggml_nelements(n) == ggml_nelements(f.out) &&
            private_to(g, f.out, n)) {
            f.relu = true;
            f.out = n;
            f.last = j;
        }
    }
    take(GGML_OP_MUL, OPERAND_MASK, f.mask);
    return true;
}

// The largest result of a claimed node that is not computed in place: a LEAKY_RELU (its result cannot
// take its input's memory -- the residual still reads x), and the residual ADD, RELU and mask MUL after
// a convolution, whose results carry the residual stream. A claimed node is allocated in the GPU's
// pool; claiming them at any size put VITS's vocoder residual stream there -- several 9.6 MB tensors at
// once -- and its compute buffer grew from 29 to 48 MB of a 64 MB pool, which it did not get (P4.46).
// The bias ADD is claimed at any size: ggml's allocator computes it in place over the convolution's
// result, so it costs the pool nothing. Defaults: 1/32 of the pool (2 MB of the default 64 MB);
// GGML_VC4_FUSE_ACT_MB / GGML_VC4_FUSE_RES_MB set them.
size_t claim_cap(const char * env) {
    if (const char * e = getenv(env)) return size_t(atof(e) * 1048576.0);
    size_t free_bytes = 0, total = 0;
    cma_memory(&free_bytes, &total);
    return total / 32;
}
size_t act_claim_bytes() { static const size_t v = claim_cap("GGML_VC4_FUSE_ACT_MB"); return v; }
size_t res_claim_bytes() { static const size_t v = claim_cap("GGML_VC4_FUSE_RES_MB"); return v; }

// Which step of the chain fusion_at matches a node can be: the order is conv, bias ADD, residual ADD,
// RELU, mask MUL, each after an earlier one. dev_supports_op claims an elementwise node only in its
// place in that chain -- not, say, HiFi-GAN's sum of three resblock outputs, an ADD of two residual
// results, which no convolution absorbs.
// A transposed convolution's chain is its own two steps, after the others so the range tests below
// never take them: its bias, and nothing after it.
enum Step { STEP_NONE, STEP_CONV, STEP_BIAS, STEP_RESIDUAL, STEP_RELU, STEP_MASK, STEP_TCONV, STEP_TBIAS };
Step step_of(const ggml_tensor * t, int depth = 0) {
    if (depth > 5 || !fuse_enabled()) return STEP_NONE;
    t = producer(t);
    switch (t->op) {
        case GGML_OP_CONV_2D: return conv_supported(t) ? STEP_CONV : STEP_NONE;
        case GGML_OP_CONV_TRANSPOSE_1D: return convt_supported(t) ? STEP_TCONV : STEP_NONE;
        case GGML_OP_LEAKY_RELU: return STEP_NONE;
        case GGML_OP_UNARY: {
            if (ggml_get_unary_op(t) != GGML_UNARY_OP_RELU || !is_f32_contiguous(t) || ggml_nbytes(t) > res_claim_bytes()) return STEP_NONE;
            const Step prev = step_of(t->src[0], depth + 1);
            return prev >= STEP_CONV && prev <= STEP_RESIDUAL ? STEP_RELU : STEP_NONE;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            if (!is_f32_contiguous(t)) return STEP_NONE;
            for (int s = 0; s < 2; s++) {
                const ggml_tensor * x = t->src[s];
                const ggml_tensor * o = t->src[1 - s];
                if (x == nullptr || o == nullptr || ggml_nelements(x) != ggml_nelements(t)) continue;
                const Operand kind = operand_kind(x, o);
                if (t->op == GGML_OP_ADD && kind == OPERAND_BIAS) {
                    const Step prev = step_of(x, depth + 1);
                    if (prev == STEP_CONV) return STEP_BIAS;
                    if (prev == STEP_TCONV) return STEP_TBIAS;
                }
                if (ggml_nbytes(t) > res_claim_bytes()) continue;
                if (t->op == GGML_OP_ADD && kind == OPERAND_RESIDUAL && step_of(x, depth + 1) == STEP_BIAS) return STEP_RESIDUAL;
                if (t->op == GGML_OP_MUL && kind == OPERAND_MASK) {
                    const Step prev = step_of(x, depth + 1);
                    if (prev >= STEP_CONV && prev <= STEP_RELU) return STEP_MASK;
                }
            }
            return STEP_NONE;
        default: return STEP_NONE;
    }
}
bool claims(const ggml_tensor * op) {
    if (!fuse_enabled()) return false;
    if (op->op == GGML_OP_LEAKY_RELU)
        // Its reader is not visible here, so any one: a LEAKY_RELU no convolution here absorbs runs on
        // the CPU backend, the kernel the CPU split would have run.
        return is_f32_contiguous(op) && is_f32_contiguous(op->src[0]) && ggml_nbytes(op->src[0]) == ggml_nbytes(op) &&
               ggml_nbytes(op) <= act_claim_bytes();
    return step_of(op) > STEP_CONV;
}

// The staged copy of x's rows [first, first + n) (zero outside [0, T_in)), through the fused
// LEAKY_RELU when there is one: max(v, slope * v) for a slope below 1, chosen on the sign bit -- the
// VFP compare-and-move the obvious form compiles to stalls the ARM1176's pipeline every element
// (ggml-0036's integer RELU, for the same reason).
void stage_rows(const Fusion & f, const vc4::Conv1dShape & c, const uint8_t * x, size_t x_row, int first, int n, uint8_t * out,
                size_t out_row) {
    const int z0 = std::min(n, std::max(0, -first));
    const int end = std::max(z0, std::min(n, c.T_in - first));
    for (int ic = 0; ic < c.IC; ic++) {
        uint8_t * row = out + size_t(ic) * out_row;
        const uint8_t * src = x + size_t(ic) * x_row;
        memset(row, 0, size_t(z0) * 4);
        if (f.act == nullptr) {
            memcpy(row + size_t(z0) * 4, src + size_t(first + z0) * 4, size_t(end - z0) * 4);
        } else {
            const uint32_t * s = reinterpret_cast<const uint32_t *>(src) + first;
            uint32_t * d = reinterpret_cast<uint32_t *>(row);
            const float slope = f.slope;
            for (int t = z0; t < end; t++) {
                const uint32_t u = s[t];
                float v, sv;
                memcpy(&v, &u, 4);
                sv = v * slope;
                uint32_t us;
                memcpy(&us, &sv, 4);
                d[t] = int32_t(u) < 0 ? us : u;
            }
        }
        memset(row + size_t(end) * 4, 0, size_t(n - end) * 4);
    }
}

// Where an operand the QPUs read through the TMU can be read from: in place when it is in a vc4
// buffer (cleaned first), else copied into the arena at `at`. Returns its bus address.
uint32_t gpu_view(Runtime & rt, const ggml_tensor * t, uint8_t * at) {
    const ggml_backend_buffer_t b = view_root(t)->buffer;
    if (is_vc4_buffer(b)) {
        mem_of(b).clean(t->data, ggml_nbytes(t));
        return mem_of(b).bus_of(t->data);
    }
    GGML_ASSERT(at != nullptr);
    memcpy(at, t->data, ggml_nbytes(t));
    return rt.arena->bus_of(at);
}
bool in_vc4(const ggml_tensor * t) { return is_vc4_buffer(view_root(t)->buffer); }

// The 1x1 int8 convolution (conv_q8_supported) with its epilogue.
void conv_q8_run(Runtime & rt, const Fusion & f, ggml_tensor * target) {
    using ms_t = std::chrono::duration<double, std::milli>;
    const auto t0 = clk::now();
    vc4::Conv1dShape c;
    GGML_ASSERT(conv_geom(f.conv, c));
    const ggml_tensor * w = f.conv->src[0];
    const ggml_tensor * x = f.conv->src[1];
    vc4::GpuBuffer & wmem = mem_of(weight_buffer(w));
    vc4::GpuBuffer & dmem = mem_of(view_root(target)->buffer);
    const int M = c.OC, N = c.T_out, K = c.IC;
    const size_t xbytes = (vc4::gemm_q8_xpack_bytes(N, K) + 63) & ~size_t(63);
    const size_t bbytes = f.bias && !in_vc4(f.bias) ? (ggml_nbytes(f.bias) + 63) & ~size_t(63) : 0;
    const size_t rbytes = f.residual && !in_vc4(f.residual) ? (ggml_nbytes(f.residual) + 63) & ~size_t(63) : 0;
    const uint32_t wbus = wmem.bus_of(w->data);
    GGML_ASSERT(ggml_is_contiguous(target) && ggml_nelements(target) == ggml_nelements(f.conv));
    std::vector<vc4::GemmQ8Problem> probs{{M, N, K, wbus, vc4::gemm_q8_stride(K), wbus + uint32_t(vc4::gemm_q8_scales_at(M, K)), dmem.bus_of(target->data),
                                           uint32_t(N) * 4, 0}};
    const size_t need = xbytes + bbytes + rbytes + vc4::gemm_q8_uniform_words(probs, 12) * 4 + 4096;
    ensure_arena(rt, need);
    uint8_t * abase = rt.arena->base();
    vc4::gemm_q8_pack_x(static_cast<const float *>(x->data), 1, x->nb[2] / 4, N, K, reinterpret_cast<float *>(abase));
    probs[0].xpack = rt.arena->bus();
    vc4::GemmQ8Epi E;
    E.flags = f.epi();
    if (f.bias) E.bias = gpu_view(rt, f.bias, abase + xbytes);
    if (f.residual) {
        E.residual = gpu_view(rt, f.residual, abase + xbytes + bbytes);
        E.residual_row = uint32_t(N) * 4;
    }
    std::vector<float> mask;
    if (f.mask) { mask.assign(static_cast<const float *>(f.mask->data), static_cast<const float *>(f.mask->data) + N); E.mask = mask.data(); }
    const uint32_t unif = rt.arena->bus() + uint32_t(xbytes + bbytes + rbytes);
    const uint32_t seq = ++rt.seq;
    std::vector<uint32_t> starts;
    auto cpu = [&](uint32_t bus) { return rt.arena->at(bus); };
    const int nq = vc4::gemm_q8_uniforms(cpu, unif, probs, 12, rt.done, seq, starts, vc4::GEMM_OUT_M, tile_panel(M, N, K), E);
    const unsigned epi = f.epi();
    const uint32_t code = program_at(rt, "gemm_q8/" + std::to_string(epi), [epi] { return vc4::gemm_q8_kernel(epi); });
    rt.arena->clean(abase, need);
    dmem.flush(target->data, ggml_nbytes(target));
    const auto t1 = clk::now();
    g_stats.host_s += std::chrono::duration<double>(t1 - t0).count();
    launch_and_wait(rt, nq, starts, code, seq,
                    "a Q8_0 1x1 CONV_2D [IC " + std::to_string(K) + " T " + std::to_string(N) + " OC " + std::to_string(M) + "]");
    const auto t2 = clk::now();
    dmem.invalidate(target->data, ggml_nbytes(target));
    if (debug_on())
        fprintf(stderr, "ggml-vc4: CONV_2D 1x1 Q8 IC %d T %d OC %d epi %x: pack %.1f QPU %.1f ms\n", K, N, M, epi,
                ms_t(t1 - t0).count(), ms_t(t2 - t1).count());
    g_stats.flops += 2.0 * M * N * K;
}

// The F32-packed convolution (any kernel type with a to_float) with its epilogue, staged or in place;
// a transposed one (P4.47) always staged, as conv1d_transposed's polyphase shape: c.OC is then its
// virtual channels, c.T_out the positions q, and each real channel's row s times as long.
void conv_f32_run(Runtime & rt, const Fusion & f, ggml_tensor * target) {
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t_host0 = clk::now();
    const ggml_tensor * conv = f.conv;
    const bool transposed = conv->op == GGML_OP_CONV_TRANSPOSE_1D;
    vc4::Conv1dShape c;
    int phases = 1;
    GGML_ASSERT(transposed ? convt_geom(conv, c, phases) : conv_geom(conv, c));
    const ggml_tensor * w = conv->src[0];
    const ggml_tensor * xv = conv->src[1];
    // x's bytes: with a fused LEAKY_RELU its input, through the convolution's own view of its output
    const uint8_t * xdata = static_cast<const uint8_t *>(xv->data);
    if (f.act) xdata = static_cast<const uint8_t *>(f.act->src[0]->data) + (xdata - static_cast<const uint8_t *>(f.act->data));
    const size_t x_row = transposed ? xv->nb[1] : xv->nb[2];
    GGML_ASSERT(ggml_is_contiguous(target) && ggml_nelements(target) == ggml_nelements(conv));
    vc4::GpuBuffer & dmem = mem_of(view_root(target)->buffer);
    const uint32_t dst_row = uint32_t(c.T_out) * uint32_t(phases) * 4;
    const int K = c.IC * c.KW;
    const size_t wfull = vc4::conv1d_wpack_floats(c.OC, K) * 4;
    const int OC_real = c.OC / phases, KT = int(w->ne[0]);
    auto pack = [&](float * o) {
        if (transposed) pack_convt_weights(w, c.IC, OC_real, KT, phases, o);
        else pack_conv_weights(w, c.OC, K, o);
    };
    Packed * pk = packed_weights(rt, w, transposed ? PACK_CONVT_F32 : PACK_CONV_F32, wfull,
                                 [&](uint8_t * o) { pack(reinterpret_cast<float *>(o)); });
    const unsigned epi = f.epi();

    // x where the TMU can read it. In place when it already is in a vc4 buffer and the host's share of
    // the padded edges is small; otherwise staged into the arena with its zero padding, which makes
    // every position the QPUs' -- the host's scalar edges cost more than the QPUs on a short
    // convolution (30 ms against 20 for VITS's 192 -> 384, k = 5, T = 293). Always staged with
    // anything fused: the host's edges know no epilogue, and the staging copy is where a LEAKY_RELU goes.
    int lo, hi;
    vc4::conv1d_interior(c, lo, hi);
    const double edge_macs = double(lo + c.T_out - hi) * c.OC * K;
    const bool in_place = !transposed && epi == 0 && f.act == nullptr && in_vc4(xv) && edge_macs < 2e5;
    int chunk = c.T_out;
    if (!in_place) {
        const int64_t per_out = int64_t(c.IC) * c.s * 4;
        chunk = std::min<int>(c.T_out, int(std::max<int64_t>(256, int64_t(conv_stage_bytes()) / per_out)) / 16 * 16);
    }
    GGML_ASSERT(in_place || chunk >= c.p);   // staging two ahead covers the left halo only then
    const int nchunks = (c.T_out + chunk - 1) / chunk, slots = in_place ? 1 : VC4_STAGE_SLOTS;
    const int in_len = (chunk - 1) * c.s + (c.KW - 1) * c.d + 1;   // staged positions per chunk
    vc4::Conv1dShape cq = c;                                        // the shape one launch runs
    if (!in_place) { cq.T_in = in_len; cq.p = 0; cq.T_out = chunk; }
    const bool res_staged = f.residual && !in_vc4(f.residual);    // else read in place
    const size_t xstage = in_place ? 0 : (size_t(c.IC) * in_len * 4 + 63) & ~size_t(63);
    const size_t rstage = res_staged ? (size_t(c.OC) * chunk * 4 + 63) & ~size_t(63) : 0;
    const size_t mbytes = f.mask && !in_vc4(f.mask) ? (size_t(c.T_out) * 4 + 63) & ~size_t(63) : 0;
    const size_t wbytes = pk ? 0 : (wfull + 63) & ~size_t(63);
    const vc4::Conv1dOut out = transposed ? vc4::CONV1D_OUT_PHASES
                               : vc4::conv1d_block_ok(dst_row) ? vc4::CONV1D_OUT_BLOCK : vc4::CONV1D_OUT_ROWS;
    vc4::Conv1dProblem P{cq, 0, 0, 0, dst_row, 0, 0};
    P.phases = phases;
    const size_t ubytes = (vc4::conv1d_uniform_words(P, 12) * 4 + 63) & ~size_t(63);
    // arena: [weights packed for this call] [slots x (x | residual)] [mask] [discard] [slots x uniforms]
    const size_t slot_bytes = xstage + rstage, slots_at = wbytes, mask_at = slots_at + slots * slot_bytes;
    const size_t discard_at = mask_at + mbytes, unif_at = discard_at + ((vc4::conv1d_discard_bytes(phases) + 63) & ~size_t(63));
    ensure_arena(rt, unif_at + slots * ubytes + 4096, w->data);
    uint8_t * abase = rt.arena->base();
    auto bus = [&](size_t at) { return rt.arena->bus() + uint32_t(at); };
    const float * wpack;
    if (pk) {
        P.wpack = pk->bus();
        wpack = reinterpret_cast<const float *>(pk->base());
    } else {
        pack(reinterpret_cast<float *>(abase));
        P.wpack = rt.arena->bus();
        wpack = reinterpret_cast<const float *>(abase);
        rt.arena->clean(abase, wbytes);
    }
    P.discard = bus(discard_at);
    vc4::Conv1dEpi E;
    E.flags = epi;
    std::vector<float> bias_v;   // transposed: one per virtual channel
    if (f.bias && transposed) {
        bias_v.resize(size_t(c.OC));
        for (int v = 0; v < c.OC; v++) bias_v[size_t(v)] = static_cast<const float *>(f.bias->data)[v / phases];
        E.bias = bias_v.data();
    } else if (f.bias) {
        E.bias = static_cast<const float *>(f.bias->data);
    }
    uint32_t mask_bus = 0, res_bus = 0;
    if (f.mask) {
        mask_bus = gpu_view(rt, f.mask, abase + mask_at);
        if (mbytes) rt.arena->clean(abase + mask_at, mbytes);
    }
    if (f.residual && !res_staged) res_bus = gpu_view(rt, f.residual, nullptr);
    const int tail = vc4::conv1d_tail(c.IC, c.KW);
    const uint32_t code = program_at(rt, "conv1d/" + std::to_string(c.KW) + "/" + std::to_string(tail) + "/" + std::to_string(int(out)) +
                                     "/" + std::to_string(epi) + "/" + std::to_string(phases),
                                     [&] { return vc4::conv1d_f32_kernel(c.KW, tail, out, epi, phases); });
    dmem.flush(target->data, ggml_nbytes(target));
    const auto t_prep = clk::now();
    g_stats.host_s += ms(t_host0, t_prep) * 1e-3;

    struct Prepared { int nq = 0; uint32_t seq = 0; std::vector<uint32_t> starts; };
    std::vector<Prepared> prep(slots);
    double stage_ms = 0;
    auto cpu = [&](uint32_t b) { return rt.arena->at(b); };
    auto prepare = [&](int k) {
        const auto ta = clk::now();
        const int slot = k % slots, t0 = k * chunk;
        Prepared & pr = prep[slot];
        vc4::Conv1dProblem Q = P;
        vc4::Conv1dEpi Ek = E;
        if (in_place) {
            vc4::GpuBuffer & xmem = mem_of(view_root(xv)->buffer);
            Q.x = xmem.bus_of(xv->data);
            Q.x_row = uint32_t(x_row);
            Q.dst = dmem.bus_of(target->data);
            xmem.clean(xv->data, size_t(c.IC - 1) * x_row + size_t(c.T_in) * 4);
        } else {
            const int tn = std::min(chunk, c.T_out - t0);
            Q.c.T_out = tn;
            Q.c.T_in = (tn - 1) * c.s + (c.KW - 1) * c.d + 1;
            // staged position j holds input t0*s - p + j, zero outside [0, T_in)
            uint8_t * xs = abase + slots_at + size_t(slot) * slot_bytes;
            stage_rows(f, c, xdata, x_row, t0 * c.s - c.p, Q.c.T_in, xs, size_t(in_len) * 4);
            Q.x = rt.arena->bus_of(xs);
            Q.x_row = uint32_t(in_len) * 4;
            Q.dst = dmem.bus_of(target->data) + uint32_t(t0) * uint32_t(phases) * 4;
            rt.arena->clean(xs, xstage);
            if (res_staged) {
                uint8_t * rs = xs + xstage;
                for (int oc = 0; oc < c.OC; oc++)
                    memcpy(rs + size_t(oc) * chunk * 4, static_cast<const uint8_t *>(f.residual->data) + (size_t(oc) * c.T_out + t0) * 4,
                           size_t(tn) * 4);
                rt.arena->clean(rs, rstage);
                Ek.residual = rt.arena->bus_of(rs);
                Ek.residual_row = uint32_t(chunk) * 4;
            }
        }
        if (f.residual && !res_staged) { Ek.residual = res_bus + uint32_t(t0) * 4; Ek.residual_row = dst_row; }
        if (f.mask) Ek.mask = mask_bus + uint32_t(t0) * 4;
        pr.seq = ++rt.seq;
        const size_t u_at = unif_at + size_t(slot) * ubytes;
        pr.nq = vc4::conv1d_uniforms(cpu, bus(u_at), Q, 12, rt.done, pr.seq, pr.starts, out, Ek);
        GGML_ASSERT(pr.nq > 0);
        rt.arena->clean(abase + u_at, ubytes);
        stage_ms += ms(ta, clk::now());
    };
    const std::string what = std::string(transposed ? "a CONV_TRANSPOSE_1D" : "a CONV_2D") + " [IC " + std::to_string(c.IC) + " KW " +
                             std::to_string(c.KW) + " T " + std::to_string(c.T_in) + " OC " + std::to_string(c.OC) +
                             (transposed ? " s " + std::to_string(phases) : std::string()) + "]";
    double wait_ms = 0;
    for (int k = 0; k < std::min(nchunks, 2); k++) prepare(k);
    for (int k = 0; k < nchunks; k++) {
        const Prepared & pr = prep[k % slots];
        const clk::time_point t0 = launch_start(rt, pr.nq, pr.starts, code);
        if (k + 2 < nchunks) prepare(k + 2);
        const auto tw = clk::now();
        launch_wait(rt, pr.nq, pr.seq, what, t0);
        wait_ms += ms(tw, clk::now());
    }
    g_stats.host_s += stage_ms * 1e-3;
    const auto t1 = clk::now();
    dmem.invalidate(target->data, ggml_nbytes(target));
    if (in_place) vc4::conv1d_edges_host(c, reinterpret_cast<const float *>(xdata), x_row / 4, wpack, static_cast<float *>(target->data),
                                         dst_row / 4);
    const auto t_end = clk::now();
    g_stats.host_s += ms(t1, t_end) * 1e-3;
    if (debug_on()) {
        fprintf(stderr, "ggml-vc4: %s IC %d KW %d s %d d %d T %d->%d OC %d %s%s%s epi %x%s x%d: pack+flush %.1f stage+uniforms %.1f "
                        "(overlapped) waiting %.1f inval+edges %.1f ms\n", transposed ? "CONV_TRANSPOSE_1D (polyphase)" : "CONV_2D", c.IC, c.KW,
                transposed ? phases : c.s, c.d, c.T_in, c.T_out * phases, c.OC / phases,
                in_place ? "in place" : "staged", out == vc4::CONV1D_OUT_BLOCK ? " block" : out == vc4::CONV1D_OUT_PHASES ? " phases" : " rows",
                pk ? " cached" : " per call",
                epi, f.act ? " +lrelu" : "", nchunks, ms(t_host0, t_prep), stage_ms, wait_ms, ms(t1, t_end));
    }
    g_stats.flops += 2.0 * c.OC * c.T_out * K;
}

// A staged convolution runs as chunks along T of about conv_stage_bytes() of input, so the arena
// stays small: the 64 MB pool also holds the model and the scheduler's buffers, and staging VITS's
// 75008 x 32 vocoder input whole (9.6 MB) made the synthesis fail to allocate. P4.46: the host stages
// chunk k + 2 while the QPUs run chunk k (the mailbox call is the launcher thread's), into one of three
// slots. Two ahead, not one, because the result may be written over x itself -- the allocator gives a
// HiFi-GAN layer's residual ADD the block of its input -- and the next chunk's left halo is staged from
// positions this one is about to overwrite: chunk k + 1 is always staged before chunk k starts.
void conv_run(const Fusion & f) {
    Runtime & rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    ggml_tensor * conv = f.conv;
    // The result goes straight to the last node covered, unless it overlaps something still read --
    // then into the convolution's own tensor, which the allocator never overlaps with its live inputs,
    // and across at the end. Exactly the input is fine (staged two chunks ahead), exactly the residual
    // too (each element read in the instruction stream before it is stored).
    const ggml_tensor * xsrc = f.act ? f.act->src[0] : conv->src[1];
    const ggml_tensor * out = f.out;
    const bool direct = out == conv ||
                        (in_vc4(out) && !overlaps(out, conv->src[0]) && (f.bias == nullptr || !overlaps(out, f.bias)) &&
                         (f.mask == nullptr || !overlaps(out, f.mask)) && (!overlaps(out, xsrc) || same_memory(out, xsrc)) &&
                         (f.residual == nullptr || !overlaps(out, f.residual) || same_memory(out, f.residual)));
    ggml_tensor * target = direct ? const_cast<ggml_tensor *>(out) : conv;
    if (debug_on())
        fprintf(stderr, "ggml-vc4: fused %s%s%s%s%s%s -> %s: %.2f MB, result %s the convolution's memory%s\n", f.act ? "LEAKY_RELU " : "",
                ggml_op_desc(conv), f.bias ? " +bias" : "", f.residual ? " +residual" : "", f.relu ? " RELU" : "", f.mask ? " *mask" : "",
                ggml_get_name(out), ggml_nbytes(out) / 1048576.0, out->data == conv->data ? "IN" : "NOT in", direct ? "" : " (via it)");
    GGML_ASSERT(is_vc4_buffer(view_root(target)->buffer));
    if (is_vc4_q8_buffer(weight_buffer(conv->src[0]))) {
        conv_q8_run(rt, f, target);
    } else {
        conv_f32_run(rt, f, target);
    }
    if (target != out) memmove(out->data, target->data, ggml_nbytes(out));
    (conv->op == GGML_OP_CONV_TRANSPOSE_1D ? g_stats.convts : g_stats.convs)++;
    g_stats.fused += (f.act != nullptr) + (f.bias != nullptr) + (f.residual != nullptr) + f.relu + (f.mask != nullptr);
}

// A VC4_Q8 weight back in ggml's layout (gemm_q8_unpack: exact), for the CPU backend; made once.
ggml_tensor * q8_cpu_copy(const ggml_tensor * w) {
    Runtime & rt = runtime();
    const auto key = pack_key(w, PACK_QB);
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.cpu_copies.find(key);
        if (it != rt.cpu_copies.end()) return it->second.t;
    }
    ggml_init_params ip = {ggml_tensor_overhead(), nullptr, /*no_alloc=*/true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_dup_tensor(ctx, w);
    ggml_set_name(t, ggml_get_name(w));
    ggml_backend_buffer_t buf = host_buft_alloc_buffer(vc4_host_buffer_type(), ggml_nbytes(t));
    if (buf == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("no memory for a CPU copy of a VC4_Q8 weight");
    }
    GGML_ASSERT(ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) == GGML_STATUS_SUCCESS);
    vc4::gemm_q8_unpack(static_cast<const uint8_t *>(w->data), int(w->ne[1]), int(w->ne[0]), [](float d) { return ggml_fp32_to_fp16(d); },
                        static_cast<BlockQ8_0 *>(t->data));
    std::lock_guard<std::mutex> lock(rt.mu);
    rt.cpu_copies.emplace(key, Runtime::CpuCopy{ctx, buf, t, w->buffer});
    return t;
}

// ---- backend ----

const char * backend_get_name(ggml_backend_t) { return "VC4"; }
void backend_free(ggml_backend_t backend) { delete backend; }

// The CPU backend for the nodes this backend claims and does not fold into a convolution, with as
// many threads as the board has cores (ggml's default of 4 on a Pi Zero's one). Found through the
// registry, which is libggml's, not libggml-base's: a backend module links only the latter, and a
// direct call left an undefined symbol that made ggml's loader drop the whole module (P4.46 -- the
// board test linked libggml itself and loaded it fine; under loom-py "auto" silently ran on the CPU).
ggml_backend_dev_t cpu_device() {
    using dev_by_type_t = ggml_backend_dev_t (*)(enum ggml_backend_dev_type);
    void * f = nullptr;
    if (void * h = dlopen("libggml.so", RTLD_NOW | RTLD_NOLOAD)) f = dlsym(h, "ggml_backend_dev_by_type");
    if (f == nullptr) f = dlsym(RTLD_DEFAULT, "ggml_backend_dev_by_type");   // a build that links ggml in
    return f ? reinterpret_cast<dev_by_type_t>(f)(GGML_BACKEND_DEVICE_TYPE_CPU) : nullptr;
}
ggml_backend_t cpu_backend(Runtime & rt) {
    if (rt.cpu == nullptr) {
        ggml_backend_dev_t dev = cpu_device();
        if (dev == nullptr) GGML_ABORT("ggml-vc4: no CPU backend registered for the nodes it hands back");
        rt.cpu = ggml_backend_dev_init(dev, nullptr);
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads"));
        if (set_threads != nullptr) set_threads(rt.cpu, int(std::max(1u, std::thread::hardware_concurrency())));
    }
    return rt.cpu;
}

// Before allocation: each private chain fusion_at finds is made to compute in place, the nodes after
// the convolution becoming views of its tensor. What a folded chain then needs from the GPU's pool is
// the convolution's result, as when the CPU ran those nodes; without it every node in the chain had a
// tensor of its own there, VITS's vocoder compute buffer grew from 29 to 48 MB, and the 64 MB pool
// could not hold it (P4.46). ggml's allocator would make the bias ADD in place itself but never does
// over a view -- loom's lowering reshapes a convolution's result before the ADD, and the allocator's
// reuse of a view's source is unreachable (ggml_gallocr_is_own is false for every view). It is right
// whether or not graph_compute folds the chain: each node is elementwise over the same positions, so
// computed on the CPU backend it is the same ADD, RELU or MUL done in place.
void backend_graph_optimize(ggml_backend_t, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params *) {
    if (!fuse_enabled()) return;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_op op = cgraph->nodes[i]->op;
        if (op != GGML_OP_CONV_2D && op != GGML_OP_CONV_TRANSPOSE_1D && op != GGML_OP_LEAKY_RELU) continue;
        Fusion f;
        if (!fusion_at(cgraph, i, f) || f.out == f.conv) continue;
        bool after = false;
        for (int j = i; j <= f.last; j++) {
            ggml_tensor * n = cgraph->nodes[j];
            if (n == f.conv) { after = true; continue; }
            if (!after || ggml_op_is_empty(n->op) || n->view_src != nullptr || n->data != nullptr || n->buffer != nullptr) continue;
            if (ggml_nbytes(n) != ggml_nbytes(f.conv) || !ggml_is_contiguous(n)) break;
            n->view_src = f.conv;
            n->view_offs = 0;
            std::lock_guard<std::mutex> lock(g_inplaced_mu);
            g_inplaced.push_back(n);
        }
        i = f.last;
    }
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph * cgraph) {
    {   // allocated: what graph_optimize made views of is theirs now
        std::lock_guard<std::mutex> lock(g_inplaced_mu);
        g_inplaced.clear();
    }
    // Nodes [run0, i) go to the CPU backend together, as one graph view: its own fusions see them
    // side by side, and the view shares the full graph's use counts.
    int run0 = -1;
    auto flush_cpu = [&](int end) {
        if (run0 < 0) return;
        ggml_cgraph view = ggml_graph_view(cgraph, run0, end);
        Runtime & rt = runtime();
        const ggml_status st = ggml_backend_graph_compute(cpu_backend(rt), &view);
        if (st != GGML_STATUS_SUCCESS) GGML_ABORT("ggml-vc4: the CPU backend failed on nodes this backend handed back");
        for (int j = run0; j < end; j++) g_stats.cpu_nodes += !ggml_op_is_empty(cgraph->nodes[j]->op);
        run0 = -1;
    };
    // P4.47: what the QPUs cannot get memory for at compute time -- the arena for staged input and
    // weights packed for one call, grown on demand -- runs on the CPU backend instead, the node and any
    // chain folded into it: VITS Q8_0_PC with its upsampling on the QPUs and its 29 MB of weights still
    // in the 64 MB pool failed the whole synthesis there ("dma_heap alloc: Cannot allocate memory").
    // Everything this backend allocates is before its first launch, so nothing was half written. A
    // VC4_Q8 weight, which only the QPUs can read, is given to the CPU as its Q8_0 copy (q8_cpu_copy).
    auto on_cpu = [&](int first, int last, const std::exception & e) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true) || debug_on())
            GGML_LOG_WARN("ggml-vc4: %s on the CPU: the GPU pool refused its memory (%s)\n", ggml_op_desc(cgraph->nodes[first]), e.what());
        // a VC4_Q8 weight is not in ggml's layout: the CPU reads its Q8_0 copy, for this run
        std::vector<std::pair<ggml_tensor **, ggml_tensor *>> swapped;
        for (int j = first; j <= last; j++)
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                ggml_tensor *& src = cgraph->nodes[j]->src[k];
                if (src != nullptr && is_vc4_q8_buffer(src->buffer)) {
                    swapped.emplace_back(&src, src);
                    src = q8_cpu_copy(src);
                }
            }
        run0 = first;
        flush_cpu(last + 1);
        for (auto & sw : swapped) *sw.first = sw.second;
        g_stats.refused++;
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (debug_on() && node->op != GGML_OP_NONE)
            fprintf(stderr, "ggml-vc4: computing %s %s %.2f MB at %p\n", ggml_op_desc(node), ggml_get_name(node), ggml_nbytes(node) / 1048576.0,
                    node->data);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_op_is_empty(node->op)) continue;
        Fusion f;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                flush_cpu(i);
                try {
                    if (is_vc4_q8_buffer(weight_buffer(node->src[0]))) mul_mat_q8(node);
                    else if (node->src[0]->type == GGML_TYPE_F32) mul_mat(node);
                    else mul_mat_qb(node);
                } catch (const std::exception & e) {
                    on_cpu(i, i, e);
                }
                break;
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_TRANSPOSE_1D:
                flush_cpu(i);
                if (!fusion_at(cgraph, i, f)) GGML_ABORT("ggml-vc4: a %s it cannot run", ggml_op_desc(node));
                try {
                    conv_run(f);
                } catch (const std::exception & e) {
                    on_cpu(i, f.last, e);
                }
                i = f.last;
                break;
            case GGML_OP_LEAKY_RELU:
                if (fusion_at(cgraph, i, f)) {
                    flush_cpu(i);
                    try {
                        conv_run(f);
                    } catch (const std::exception & e) {
                        on_cpu(i, f.last, e);
                    }
                    i = f.last;
                    break;
                }
                if (run0 < 0) run0 = i;
                break;
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_UNARY:
                if (run0 < 0) run0 = i;   // claimed, and not folded into a convolution
                break;
            default: GGML_ABORT("ggml-vc4: unsupported op %s", ggml_op_desc(node));
        }
    }
    flush_cpu(cgraph->n_nodes);
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
    /* .graph_optimize      = */ backend_graph_optimize,
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

// How much of the pool an allocation can still get. Not CmaFree, which the kernel lowers by lending
// idle pool pages to the page cache and which a freed buffer never raises again while the process
// lives (GpuBuffer::live_bytes): with the pool idle it read 44 MB of 64, and "auto" put Moonshine and
// conformer-ctc on the CPU in one run and on the QPUs in the next (P4.46). So: the pool less what this
// process holds and a slack for anything else using it -- an allocation the kernel cannot satisfy
// still fails, and says so -- or CmaFree when that is larger. No slack for other users: 8 MB put
// conformer-ctc (44-46 MB of weights) on the CPU where P4.45, reading CmaFree on a fresh pool, had run
// it on the QPUs.
constexpr size_t VC4_POOL_SLACK = 0;
void pool_memory(size_t * free, size_t * total) {
    size_t cma_free = 0;
    cma_memory(&cma_free, total);
    const size_t held = vc4::GpuBuffer::live_bytes();
    const size_t ours = *total > held + VC4_POOL_SLACK ? *total - held - VC4_POOL_SLACK : 0;
    *free = std::max(cma_free, ours);
}

// What an engine may put here: the pool less the room the QPUs work in -- the arena (staged input,
// packed activations, uniforms), packed convolution weights, the scheduler's buffers for the nodes
// this backend runs. VITS F32's 62.6 MB of weights "fitted" a 64.6 MB pool and left the QPUs room for
// nothing, so it ran every node on the CPU through the scheduler, 2.6% slower than the CPU alone.
constexpr size_t VC4_WORKING_RESERVE = size_t(16) << 20;
void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    pool_memory(free, total);
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
ggml_backend_buffer_type_t dev_get_host_buffer_type(ggml_backend_dev_t) { return vc4_host_buffer_type(); }

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
        case GGML_OP_CONV_TRANSPOSE_1D: return convt_supported(op);
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_UNARY:
        case GGML_OP_ADD:
        case GGML_OP_MUL: return claims(op);
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
    /* .get_host_buffer_type = */ dev_get_host_buffer_type,
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

size_t host_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) { return ggml_nbytes(tensor); }
bool host_buft_is_host(ggml_backend_buffer_type_t) { return true; }

ggml_backend_buffer_type_t vc4_host_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ buft_get_name,
            /* .alloc_buffer     = */ host_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ host_buft_get_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ host_buft_is_host,
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
