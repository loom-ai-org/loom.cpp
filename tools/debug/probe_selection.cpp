// Scratch probe: with more than one offload device registered, what do "auto" and "gpu" resolve to,
// and in what order does the registry report them? BACKLOG.md P4.8, item 2.
#include "loom/loom.h"

#include <cstdio>

int main() {
    std::printf("registry order:\n");
    int index = 0;
    for (const loom::DeviceInfo& d : loom::available_devices()) {
        std::printf("  [%d] %-10s %-45s %s\n", index++, d.name.c_str(), d.description.c_str(),
                    d.is_cpu ? "(cpu)" : "(offload)");
    }
    for (const char* spec : {"auto", "gpu", "cpu"}) {
        try {
            loom::Device dev = loom::Device::open(spec);
            std::printf("  \"%s\" -> %s\n", spec, dev.name().c_str());
        } catch (const loom::Error& e) {
            std::printf("  \"%s\" -> throws: %s\n", spec, e.what());
        }
    }
    return 0;
}
