// Can the engine tell a discrete accelerator from a host-side one WITHOUT knowing any backend's name?
#include <ggml-backend.h>

#include <cstdio>

static const char* type_name(ggml_backend_dev_t d) {
    switch (ggml_backend_dev_type(d)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

int main() {
    ggml_backend_load_all();
    std::printf("%-10s %-6s %-14s %-12s\n", "device", "type", "buft_is_host", "host_buffer");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d, &props);
        const bool is_host = ggml_backend_buft_is_host(ggml_backend_dev_buffer_type(d));
        std::printf("%-10s %-6s %-14s %-12s\n", ggml_backend_dev_name(d), type_name(d),
                    is_host ? "true" : "false", props.caps.host_buffer ? "true" : "false");
    }
    return 0;
}
