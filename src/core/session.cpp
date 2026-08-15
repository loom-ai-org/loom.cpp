#include "loom/core/session.h"

#include "loom/core/graph_topology.h"
#include "loom/loom_errors.h"

namespace loom {

Session::Session(GgufModel& model, Backends backends)
    : model_(model), bridge_(backends) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("this GGUF carries no driver_script, so it can be inspected but not run. "
                        "Re-export it with a current loom-exporter.");
    }

    for (const std::string& name : model.topology_names()) {
        GraphTopology topo = GraphTopology::parse(model.topology_json(name));
        // A topology carrying ATTENTION nodes needs a KV cache to write into (KV-CACHE.md stage 2),
        // sized from the model's own declared geometry -- the host asks the file rather than knowing
        // anything per-model, which is the point of declaring it there.
        if (topo.uses_kv_cache() && kv_cache_ == nullptr) kv_cache_ = make_kv_cache(model, backends);
        // A hybrid's ShortConv blocks carry history the KV cache does not hold, from the file's own
        // loom.n_conv_* keys (BACKLOG.md P4.0.10). Both are allocated ONCE and shared by every module
        // that asks, which is what the drivers assume: a decoder's cache must survive its own re-entry.
        if (topo.uses_conv_state() && conv_state_ == nullptr) {
            conv_state_ = make_conv_state_cache(model, backends);
        }
        KvCache* kv = topo.uses_kv_cache() ? kv_cache_.get() : nullptr;
        ConvStateCache* conv = topo.uses_conv_state() ? conv_state_.get() : nullptr;
        bridge_.register_module(name, model, std::move(topo), kv, conv);
    }

    bridge_.load_script(model.kv_str("model.driver_script"));
}

} // namespace loom
