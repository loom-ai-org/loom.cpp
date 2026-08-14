// What does ggml_backend_sched actually get handed, per device spec?
#include "loom/loom.h"

#include <cstdio>

int main(int argc, char** argv) {
    const char* spec = argc > 1 ? argv[1] : "auto";
    try {
        loom::Device device = loom::Device::open(spec);
        const loom::Backends backends = device.backends();
        std::printf("spec %-8s -> primary %-8s hybrid=%d assists=%zu | schedule order:", spec,
                    device.name().c_str(), static_cast<int>(backends.hybrid()),
                    backends.assists.size());
        for (ggml_backend_t b : backends.schedule_order()) {
            std::printf(" %s", ggml_backend_name(b));
        }
        std::printf("\n");
    } catch (const loom::Error& e) {
        std::printf("spec %-8s -> throws: %s\n", spec, e.what());
    }
    return 0;
}
