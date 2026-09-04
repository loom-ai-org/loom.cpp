#pragma once

// A loaded model, ready to run: every topology the file declares registered on a Lua bridge, with the
// caches those topologies ask for, and the driver script loaded.
//
// WHY THIS IS IN THE ENGINE. The sequence below was written three times -- `run_asr` and the generation
// path in `tools/loom_cli/main.cpp`, and again in loom-py's `src/binding.cpp` -- and the copies had
// already diverged: `run_asr`'s attaches a `KvCache` but no `ConvStateCache`, so a speech model with
// ShortConv blocks would have thrown inside the driver on its first SHORT_CONV node, exactly the
// failure BACKLOG.md P4.0.10 fixed for the umbrella header. Nothing in that difference was a decision.
//
// It is host BOILERPLATE, not host policy: which caches a file needs is declared by its own topologies
// (`uses_kv_cache()` / `uses_conv_state()`) and sized by its own hparams, so there is nothing here for a
// host to have an opinion about. A fourth front end should inherit it rather than rediscover it.
//
// LIFETIME, which is the reason this is a class and not a function: `LoomLuaBridge::register_module`
// takes the caches by NON-OWNING pointer and keeps them for the bridge's whole life. Returning a bridge
// alone would leave the caller holding two objects it must not drop, in an order it must get right.
// Here the caches are members declared BEFORE the bridge, so the bridge is destroyed first and the
// pointers it holds are valid for exactly as long as it is.

#include "loom/core/backend.h"
#include "loom/core/conv_state_cache.h"
#include "loom/core/gguf_model.h"
#include "loom/core/kv_cache.h"
#include "loom/core/lua_bridge.h"

#include <memory>
#include <vector>

namespace loom {

// The caches a model's topologies asked for, owned by whoever registered them.
//
// **A struct rather than three members, because the ORDER matters and a caller should not have to
// know that.** `LoomLuaBridge::register_module` keeps these by non-owning pointer for the bridge's
// whole life, so they must outlive it; handing back one object makes "declare this before the bridge"
// a single, checkable thing rather than three.
struct ModuleCaches {
    // One shared cache for every module that asked and did not ask for its own.
    std::unique_ptr<KvCache> kv;
    // One per module declaring `kv_cache_scope: "private"` -- a second decode STREAM, which is what
    // classifier-free guidance runs (ADR-023). Sharing here is a wrong answer, not a slow one.
    std::vector<std::unique_ptr<KvCache>> private_kv;
    std::unique_ptr<ConvStateCache> conv;
};

// Parses every topology `model` declares, allocates the caches they ask for, and registers each as a
// module on `bridge`. Returns the caches, which the caller must keep alive for the bridge's lifetime.
//
// **This exists because the sequence was written four times and the copies diverged** -- see the note
// above `Session`, which is the third caller. The fourth is loom-py's binding, which cannot use
// `Session` itself only because it supports a GGUF with no driver script, and which had silently
// re-acquired the exact bug ADR-023 fixed. A rule that decides correctness must have one
// implementation, and this is it.
ModuleCaches register_topologies(GgufModel& model, Backends backends, LoomLuaBridge& bridge);

class Session {
public:
    // Registers every declared topology, allocates the caches any of them need, and loads the embedded
    // driver script. Throws loom::LoadError when the file carries no driver -- such a file can be
    // inspected but not run, and a Session is the running of it.
    //
    // `model` is NOT owned and must outlive this object, the same non-owning convention the bridge and
    // GraphBuilder already use.
    Session(GgufModel& model, Backends backends);

    LoomLuaBridge& bridge() { return bridge_; }
    const LoomLuaBridge& bridge() const { return bridge_; }
    GgufModel& model() { return model_; }
    const GgufModel& model() const { return model_; }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

private:
    GgufModel& model_;
    // Declared before `bridge_` on purpose -- see the lifetime note above. Every member is null or
    // empty when no topology in the file asks for that kind of state, which is the common case.
    ModuleCaches caches_;
    LoomLuaBridge bridge_;
};

} // namespace loom
