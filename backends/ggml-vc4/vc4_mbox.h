// Pi Zero / Pi 1-3 QPU access through the firmware mailbox (/dev/vcio) and GPU memory from
// /dev/vcsm-cma. Linux only, legacy (non-KMS) firmware graphics only.
//
// Two traps found on 2026-10-27/28 shape this file (see the ggml-vc4 notes):
//   - Under vc4-kms-v3d the firmware refuses qpu_enable, and execute_qpu then wedges the firmware
//     until reboot. open() refuses while the vc4 DRM driver is loaded and stops if qpu_enable fails.
//   - execute_qpu can return before the QPUs finish (>= 8 QPUs). Completion is the kernel's own
//     done marker; `execute` only starts the work.
//   - qpu_enable(0) after a hung program wedged the firmware too, so this file never disables.
#pragma once

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vc4 {

class Mailbox {
public:
    Mailbox() {
        if (access("/sys/module/vc4", F_OK) == 0)
            throw std::runtime_error("vc4 DRM driver loaded (vc4-kms-v3d): the mailbox QPU route would wedge the "
                                     "firmware; boot without the overlay");
        fd_ = ::open("/dev/vcio", 0);
        if (fd_ < 0) throw std::runtime_error(std::string("open /dev/vcio: ") + strerror(errno));
        uint32_t on = 1;
        uint32_t r = call(0x30012, &on, 1, 1);
        if (r != 0) throw std::runtime_error("qpu_enable refused (0x" + std::to_string(r) + "); not launching");
    }
    ~Mailbox() { if (fd_ >= 0) ::close(fd_); }
    Mailbox(const Mailbox&) = delete;
    Mailbox& operator=(const Mailbox&) = delete;

    // Start n_qpu programs from a control list at `ctrl_bus` ([uniforms, code] per QPU). The return
    // value is NOT a completion signal.
    uint32_t execute(uint32_t n_qpu, uint32_t ctrl_bus, uint32_t timeout_ms = 10000, uint32_t noflush = 0) {
        uint32_t in[4] = {n_qpu, ctrl_bus, noflush, timeout_ms};
        return call(0x30011, in, 4, 1);
    }

private:
    int fd_ = -1;
    uint32_t call(uint32_t tag, const uint32_t* in, int n_in, int n_out) {
        uint32_t buf[32] = {0};
        int nval = n_in > n_out ? n_in : n_out, i = 0;
        buf[i++] = 0; buf[i++] = 0; buf[i++] = tag; buf[i++] = nval * 4; buf[i++] = n_in * 4;
        for (int k = 0; k < nval; k++) buf[i++] = k < n_in ? in[k] : 0;
        buf[i++] = 0;
        buf[0] = i * 4;
        if (ioctl(fd_, _IOWR(100, 0, char*), buf) < 0)
            throw std::runtime_error("mailbox tag " + std::to_string(tag) + ": " + strerror(errno));
        return buf[5];
    }
};

// One contiguous GPU-visible block. `bus` is the address exactly as vcsm-cma returns it (the 0x8
// L2-cached alias); the 0xC direct alias hung TMU reads -- do not rewrite it.
class GpuBuffer {
public:
    // Uncached: a vcsm-cma allocation (its dmabuf maps uncached for the CPU whatever `cached` the
    // allocation asks for -- measured 2026-10-28: identical CPU timings either way). Cached: a
    // /dev/dma_heap/linux,cma buffer, which maps write-back cached, imported into vcsm-cma for its
    // VideoCore bus address. With `cached` the caller owns coherence: clean()/flush()/invalidate()
    // around every QPU access. The ARM1176's L1 is the only cache between the two (QPU accesses use
    // the 0x8 alias, coherent with the VideoCore L2).
    explicit GpuBuffer(uint32_t size, bool cached = false) : size_(size), cached_(cached) {
        // A failed constructor runs no destructor: close what was opened before saying why.
        auto fail = [&](const char* what) {
            const int e = errno;
            release();
            throw std::runtime_error(std::string(what) + ": " + strerror(e));
        };
        vcsm_ = ::open("/dev/vcsm-cma", O_RDWR);
        if (vcsm_ < 0) fail("open /dev/vcsm-cma");
        if (!cached) {
            struct {
                uint32_t size, num, cached, pad;
                uint8_t name[32];
                int32_t handle;
                uint32_t vc_handle;
                uint64_t dma_addr;
            } a;
            memset(&a, 0, sizeof a);
            a.size = size; a.num = 1;
            strcpy(reinterpret_cast<char*>(a.name), "ggml-vc4");
            if (ioctl(vcsm_, _IOR('J', 0x5A, decltype(a)), &a) < 0) fail("vcsm-cma alloc");
            dmabuf_ = a.handle;
            bus_ = uint32_t(a.dma_addr);
        } else {
            heap_ = ::open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
            if (heap_ < 0) fail("open /dev/dma_heap/linux,cma");
            struct { uint64_t len; uint32_t fd, fd_flags; uint64_t heap_flags; } h = {size, 0, O_RDWR | O_CLOEXEC, 0};
            // Twice: a large first request can fail while the kernel is still migrating the movable
            // pages it lent out of the pool -- 40 MB with 55 MB free failed with EBUSY on the Pi Zero,
            // and a 56 MB request right after it succeeded (P4.46).
            if (ioctl(heap_, _IOWR('H', 0x0, decltype(h)), &h) < 0 && ioctl(heap_, _IOWR('H', 0x0, decltype(h)), &h) < 0)
                fail("dma_heap alloc");
            dmabuf_ = int(h.fd);
            struct {
                int32_t dmabuf_fd;
                uint32_t cached;
                uint8_t name[32];
                int32_t handle;
                uint32_t vc_handle;
                uint32_t size, pad;
                uint64_t dma_addr;
            } im;
            memset(&im, 0, sizeof im);
            im.dmabuf_fd = dmabuf_; im.cached = 1;
            strcpy(reinterpret_cast<char*>(im.name), "ggml-vc4");
            if (ioctl(vcsm_, _IOR('J', 0x5B, decltype(im)), &im) < 0) fail("vcsm-cma import");
            import_ = im.handle;
            bus_ = uint32_t(im.dma_addr);
        }
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_, 0);
        if (p == MAP_FAILED) fail("mmap");
        cpu_ = static_cast<uint8_t*>(p);
        live_bytes() += size_;
    }
    ~GpuBuffer() {
        if (cpu_) live_bytes() -= size_;
        release();
    }

    // Bytes of the pool this process holds. The kernel's CmaFree cannot say how much of the pool is
    // allocatable: it lends free pool pages to the page cache (CmaFree 44 MB of 64 with the pool idle,
    // 56 after drop_caches) and a freed buffer does not show up in it again while the process lives,
    // though the next allocation reuses it (P4.46, Pi Zero).
    static std::atomic<size_t>& live_bytes() {
        static std::atomic<size_t> n{0};
        return n;
    }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Cache maintenance on [p, p + n) of this block's CPU mapping (no-ops when uncached).
    void clean(const void* p, size_t n) { sync(p, n, 2); }       // write dirty lines back: QPU will read
    void invalidate(const void* p, size_t n) { sync(p, n, 1); }  // drop lines: QPU wrote memory
    void flush(const void* p, size_t n) { sync(p, n, 3); }       // both: QPU is about to write here

    uint8_t* base() const { return cpu_; }
    uint32_t bus_of(const void* p) const { return bus_ + uint32_t(static_cast<const uint8_t*>(p) - cpu_); }
    bool contains(const void* p) const { return p >= cpu_ && p < cpu_ + size_; }
    bool cached() const { return cached_; }
    uint32_t bus() const { return bus_; }
    uint32_t size() const { return size_; }
    uint32_t* at(uint32_t bus) {
        if (bus % 4 || bus < bus_ || bus - bus_ >= size_) throw std::runtime_error("GpuBuffer: address out of block");
        return reinterpret_cast<uint32_t*>(cpu_ + (bus - bus_));
    }

private:
    int vcsm_ = -1, heap_ = -1, dmabuf_ = -1, import_ = -1;

    void release() {
        if (cpu_) munmap(cpu_, size_);
        if (import_ >= 0) ::close(import_);
        if (dmabuf_ >= 0) ::close(dmabuf_);
        if (heap_ >= 0) ::close(heap_);
        if (vcsm_ >= 0) ::close(vcsm_);
        cpu_ = nullptr;
        import_ = dmabuf_ = heap_ = vcsm_ = -1;
    }
    uint8_t* cpu_ = nullptr;
    uint32_t bus_ = 0, size_;
    bool cached_;

    // VC_SM_CMA_IOCTL_MEM_CLEAN_INVALID2 (vc_sm_cma_ioctl.h): one block, mode 1 invalidate, 2 clean,
    // 3 clean + invalidate, on a user virtual address range.
    void sync(const void* p, size_t n, uint32_t mode) {
        if (!cached_ || n == 0) return;
        struct {
            uint32_t op_count, pad;
            struct { uint32_t invalidate_mode, block_count; void* start_address; uint32_t block_size, inter_block_stride; } s;
        } op;
        memset(&op, 0, sizeof op);
        op.op_count = 1;
        op.s.invalidate_mode = mode;
        op.s.block_count = 1;
        op.s.start_address = const_cast<void*>(p);
        op.s.block_size = uint32_t(n);
        // The uapi struct ends in a zero-length block array, so the command encodes only its 8-byte
        // header -- not sizeof(op).
        struct header { uint32_t op_count, pad; };
        if (ioctl(vcsm_, _IOR('J', 0x5C, header), &op) < 0)
            throw std::runtime_error(std::string("vcsm-cma cache op: ") + strerror(errno));
    }
};

}  // namespace vc4
