// Can the engine tell a discrete accelerator from a host-side one WITHOUT knowing any backend's name?
//
// And, since P4.8d answered that with "the type field will not tell you": can the KERNEL tell, when
// the backend will not? `device_id` is the device's PCI address or null -- null being a reliable
// reading rather than uninitialised memory, because ggml_backend_dev_get_props memsets the struct
// before the backend fills it -- and sysfs then says what class that address is. PCI base class 0x03
// is a display controller; 0x12 is a processing accelerator.
//
// Measured with this: CUDA0 reports 0000:02:00.0 / 0x030000, while OPENVINO0 driving an Intel NPU
// reports no address at all. That asymmetry is what `Device::open` breaks rank-0 ties on, and the
// reason it only ever promotes on a positive answer is visible here too -- ggml-metal, ggml-sycl,
// ggml-opencl, ggml-webgpu and ggml-cann are real GPU backends that also report nothing.
#include <ggml-backend.h>

#include <cstdio>
#include <string>

static const char* type_name(ggml_backend_dev_t d) {
    switch (ggml_backend_dev_type(d)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

// The kernel's own word on what a PCI address is, or "-" when there is nothing to ask about. Linux
// only; everywhere else this column is blank and the tie-break it feeds simply never fires.
static std::string pci_class(const char* device_id) {
    if (device_id == nullptr || *device_id == '\0') return "-";
    const std::string path = std::string("/sys/bus/pci/devices/") + device_id + "/class";
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return "(not in sysfs)";
    char buf[64] = {0};
    const char* got = std::fgets(buf, sizeof(buf), f);
    std::fclose(f);
    if (got == nullptr) return "(unreadable)";
    std::string cls(buf);
    while (!cls.empty() && (cls.back() == '\n' || cls.back() == '\r')) cls.pop_back();
    if (cls.compare(0, 4, "0x03") == 0) cls += "  <- display controller";
    if (cls.compare(0, 4, "0x12") == 0) cls += "  <- processing accelerator";
    return cls;
}

int main() {
    ggml_backend_load_all();
    std::printf("%-10s %-6s %-14s %-12s %-16s %s\n", "device", "type", "buft_is_host", "host_buffer",
                "device_id", "kernel says");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d, &props);
        const bool is_host = ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(d));
        std::printf("%-10s %-6s %-14s %-12s %-16s %s\n", ggml_backend_dev_name(d), type_name(d),
                    is_host ? "true" : "false", props.caps.host_buffer ? "true" : "false",
                    props.device_id ? props.device_id : "(null)", pci_class(props.device_id).c_str());
    }
    return 0;
}
