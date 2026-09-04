#include "loom/core/session.h"

#include "loom/core/graph_topology.h"
#include "loom/loom_errors.h"

namespace loom {

ModuleCaches register_topologies(GgufModel& model, Backends backends, LoomLuaBridge& bridge) {
    ModuleCaches caches;
    for (const std::string& name : model.topology_names()) {
        GraphTopology topo = GraphTopology::parse(model.topology_json(name));
        // A topology carrying ATTENTION nodes needs a KV cache to write into (KV-CACHE.md stage 2),
        // sized from the model's own declared geometry -- the host asks the file rather than knowing
        // anything per-model, which is the point of declaring it there.
        //
        // One shared cache, EXCEPT for a module that declares `kv_cache_scope: "private"`, which gets
        // its own. That declaration is how a driver says "this is a second stream, not a second
        // phase" -- family 10's classifier-free guidance runs the decoder twice per step over two
        // independent histories, and two streams sharing one cache read each other's cells and still
        // produce plausible tokens (ADR-023).
        if (topo.uses_kv_cache() && topo.private_kv_cache) {
            caches.private_kv.push_back(make_kv_cache(model, backends));
        } else if (topo.uses_kv_cache() && caches.kv == nullptr) {
            caches.kv = make_kv_cache(model, backends);
        }
        // A hybrid's ShortConv blocks carry history the KV cache does not hold, from the file's own
        // loom.n_conv_* keys (BACKLOG.md P4.0.10). Allocated ONCE and shared by every module that
        // asks, which is what the drivers assume: a decoder's state must survive its own re-entry.
        if (topo.uses_conv_state() && caches.conv == nullptr) {
            caches.conv = make_conv_state_cache(model, backends);
        }
        KvCache* kv = nullptr;
        if (topo.uses_kv_cache()) {
            kv = topo.private_kv_cache ? caches.private_kv.back().get() : caches.kv.get();
        }
        ConvStateCache* conv = topo.uses_conv_state() ? caches.conv.get() : nullptr;
        bridge.register_module(name, model, std::move(topo), kv, conv);
    }
    return caches;
}

Session::Session(GgufModel& model, Backends backends)
    : model_(model), bridge_(backends) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("this GGUF carries no driver_script, so it can be inspected but not run. "
                        "Re-export it with a current loom-exporter.");
    }
    caches_ = register_topologies(model, backends, bridge_);
    bridge_.load_script(model.kv_str("model.driver_script"));
}

} // namespace loom
