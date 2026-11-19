#include "loom/core/gpu_reset_watch.h"

#include "loom/loom_errors.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/klog.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#endif

namespace loom::gpu_reset_watch {

namespace {

std::atomic<bool> g_armed{false};
std::mutex g_mutex;
std::function<int64_t()> g_source;  // guarded by g_mutex

#if defined(__linux__)
// include/uapi/drm/v3d_drm.h, by value: building against a system's kernel headers would tie the build
// to the box it ran on, and these values are ABI.
struct DrmV3dGetParam {
    uint32_t param;
    uint32_t pad;
    uint64_t value;
};
constexpr unsigned long kDrmIoctlV3dGetParam = _IOWR('d', 0x40 + 0x04, DrmV3dGetParam);
constexpr uint32_t kV3dParamGlobalResetCounter = 15;  // Linux 6.18

// The render node whose driver is v3d, or -1.
int open_v3d_render_node() {
    DIR* dir = opendir("/sys/class/drm");
    if (dir == nullptr) return -1;
    int fd = -1;
    while (dirent* e = readdir(dir)) {
        const std::string name = e->d_name;
        if (name.rfind("renderD", 0) != 0) continue;
        char target[PATH_MAX];
        const std::string link = "/sys/class/drm/" + name + "/device/driver";
        const ssize_t n = readlink(link.c_str(), target, sizeof(target) - 1);
        if (n <= 0) continue;
        target[n] = '\0';
        const std::string driver = target;
        if (driver.substr(driver.find_last_of('/') + 1) != "v3d") continue;
        fd = open(("/dev/dri/" + name).c_str(), O_RDWR | O_CLOEXEC);
        if (fd >= 0) break;
    }
    closedir(dir);
    return fd;
}

int64_t ioctl_count(int fd) {
    DrmV3dGetParam p{kV3dParamGlobalResetCounter, 0, 0};
    if (ioctl(fd, kDrmIoctlV3dGetParam, &p) != 0) return -1;
    return static_cast<int64_t>(p.value);
}

int64_t klog_token() {
    const int size = klogctl(10 /* SYSLOG_ACTION_SIZE_BUFFER */, nullptr, 0);
    if (size <= 0) return -1;
    std::vector<char> buf(static_cast<size_t>(size));
    const int n = klogctl(3 /* SYSLOG_ACTION_READ_ALL */, buf.data(), size);
    if (n < 0) return -1;
    return log_token(std::string(buf.data(), static_cast<size_t>(n)));
}

// The kernel's own count, if this kernel has one; else the kernel log; else nothing (with a warning).
std::function<int64_t()> v3d_source() {
    const int fd = open_v3d_render_node();
    if (fd >= 0 && ioctl_count(fd) >= 0) {
        return [fd] { return ioctl_count(fd); };
    }
    if (fd >= 0) close(fd);
    if (klog_token() >= 0) return [] { return klog_token(); };
    std::fprintf(stderr,
                 "loom: this V3D GPU's resets cannot be seen (no DRM_V3D_PARAM_GLOBAL_RESET_COUNTER before "
                 "Linux 6.18, and the kernel log is not readable: dmesg_restrict, or not in group adm). A "
                 "job the kernel kills for hanging is reported as finished by V3DV, and its output would "
                 "not be caught (loom.cpp#124).\n");
    return {};
}
#endif

} // namespace

int64_t log_token(const std::string& log) {
    const std::string marker = "Resetting GPU for hang";
    int64_t lines = 0;
    size_t last = std::string::npos;
    for (size_t at = log.find(marker); at != std::string::npos; at = log.find(marker, at + 1)) {
        ++lines;
        last = at;
    }
    if (last == std::string::npos) return 0;
    const size_t newline = log.rfind('\n', last);
    const size_t line_start = newline == std::string::npos ? 0 : newline + 1;
    const size_t open = log.find('[', line_start);
    if (open != std::string::npos && open < last) {
        char* end = nullptr;
        const double seconds = std::strtod(log.c_str() + open + 1, &end);
        // +1: a reset logged at time 0 must still differ from "no reset in the log".
        if (end != nullptr && *end == ']') return static_cast<int64_t>(seconds * 1e6) + 1;
    }
    return lines;
}

void arm(const std::string& device_description) {
    if (device_description.find("V3D") == std::string::npos) return;
#if defined(__linux__)
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_armed.load(std::memory_order_relaxed)) return;
    g_source = v3d_source();
    g_armed.store(static_cast<bool>(g_source), std::memory_order_relaxed);
#endif
}

bool armed() { return g_armed.load(std::memory_order_relaxed); }

int64_t count() {
    if (!armed()) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_source ? g_source() : -1;
}

void arm_for_test(std::function<int64_t()> source) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_source = std::move(source);
    g_armed.store(static_cast<bool>(g_source), std::memory_order_relaxed);
}

Check::Check() : before_(armed() ? count() : -1) {}

void Check::verify(const std::string& what) const {
    if (before_ < 0) return;
    const int64_t after = count();
    if (after >= 0 && after != before_) {
        throw DeviceLostError(
            "the GPU was reset while " + what + " ran (the kernel's hang check; reset count " +
            std::to_string(before_) + " -> " + std::to_string(after) + "). This driver reports a job the "
            "kernel killed as finished, so its output is not used (loom.cpp#124). A reset caused by another "
            "process on the same GPU trips this too; the call can be retried.");
    }
}

} // namespace loom::gpu_reset_watch
