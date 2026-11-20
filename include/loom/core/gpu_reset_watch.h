// A GPU reset is a failed graph, even when the driver says the graph finished (loom.cpp#124, ADR-140).
//
// On a Raspberry Pi 4 (V3D 4.2, Mesa v3dv) a compute job that never finishes is killed by the kernel's
// hang check ("Resetting GPU for hang"), and the fence the user waits on still signals SUCCESS:
// vkWaitForFences and vkDeviceWaitIdle both return VK_SUCCESS and the output buffers hold whatever was
// there. Measured on kernel 6.18.34 + Mesa 25.0.7 with a planted infinite loop: VK_SUCCESS after 1.04 s.
// The reporter traced it to the kernel (a later job is not failed when a job it depends on was) and Mesa
// (a signalled fence's error status is never read). ggml-vulkan already turns VK_ERROR_DEVICE_LOST into
// an exception; the driver just never returns it.
//
// So loom asks the kernel instead. `Device::open` arms the reset count when it selects a V3D device, and
// from then on every graph that RUNS ON a V3D device (a `GraphBuilder` whose backends include one) reads
// the count before and after it, and a count that moved throws DeviceLostError. Per device, not per
// process: a reset cannot touch a graph that ran on the CPU, so a CPU model in the same process is never
// refused for it. The count is the kernel's `DRM_V3D_PARAM_GLOBAL_RESET_COUNTER` (Linux 6.18+),
// read on the v3d render node, which needs only the `render` group. On an older kernel it falls back to
// the time of the last "Resetting GPU for hang" line in the kernel log (`klogctl`, which needs
// `dmesg_restrict=0` or the `adm` group/CAP_SYSLOG). With neither, `arm` warns once that a reset cannot
// be seen. The counter is GLOBAL: a reset caused by another process on the same GPU also fails the graph
// running here, which is the safe answer, because that reset may have taken this process's queued jobs
// with it.
//
// For every graph not on a V3D device (so almost every run) the check is one branch on a bool the
// builder computed once.
#pragma once

#include <ggml-backend.h>

#include <cstdint>
#include <functional>
#include <string>

namespace loom::gpu_reset_watch {

// Arms the watch when `device_description` names a V3D GPU ("V3D 4.2.14.0"); does nothing otherwise.
// Called by Device::open for the device it selected.
void arm(const std::string& device_description);

bool armed();

// Whether a graph running on `backend` is watched: its device's description (or name) says V3D. Asked once
// per GraphBuilder.
bool watches(ggml_backend_t backend);

// The current reset count, or -1 when the watch is unarmed or the source cannot be read.
int64_t count();

// The kernel-log fallback's value for `log` (the text `dmesg` prints): the time of the last "Resetting GPU
// for hang" line in microseconds since boot, plus one, or the number of such lines when the log has no
// timestamps; 0 when there is none. It moves exactly when a reset is logged. (If the log's ring buffer
// wraps past the last such line mid-graph, it moves without one: a false alarm, never a missed reset.)
int64_t log_token(const std::string& log);

// For tests: arm with `source` as the count (an empty function disarms and restores the kernel's).
void arm_for_test(std::function<int64_t()> source);

// For tests: the text a device's description or name must contain to be watched ("V3D"; "CPU" lets a CI test
// watch a CPU graph). Read when a GraphBuilder is made, so set it before building the bridge under test.
void watch_devices_for_test(const std::string& needle);

// A snapshot taken before a graph runs; `verify` throws DeviceLostError when the count has moved since.
// `watched` is the graph builder's `watches(...)`: an unwatched graph reads nothing and never throws.
class Check {
public:
    explicit Check(bool watched);
    void verify(const std::string& what) const;

private:
    int64_t before_;
};

} // namespace loom::gpu_reset_watch
