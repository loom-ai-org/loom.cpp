// `LoomLuaBridge::has_function`, and the selection it exists to serve.
//
// `loom::text::generate` prefers a driver's `infer_with_past` over its `infer` when both are exported,
// because the first runs the decode loop against the KV cache and the second makes the HOST re-feed a
// growing prompt -- 2.83x on Qwen3-0.6B, and O(n^2) rather than O(n) in tokens. That preference is
// invisible from outside: both entry points return a plausible answer, so a wrong choice is a
// performance bug that no correctness assertion catches. This is the assertion that catches it.
//
// Bridge-level rather than end-to-end on purpose: the choice is made from what the SCRIPT defines, so
// a script is the whole input, and this needs no checkpoint and no fixture.
#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    {
        // A driver exporting both, which is what every generated causal-LM export looks like.
        loom::LoomLuaBridge bridge(loom::Backends{backend.get(), nullptr});
        bridge.load_script("function infer(i) return 1 end\n"
                           "function infer_with_past(i) return {1, 2} end\n");
        LOOM_CHECK(bridge.has_function("infer"));
        LOOM_CHECK(bridge.has_function("infer_with_past"));
        LOOM_CHECK(!bridge.has_function("infer_with_future"));
    }

    {
        // LFM2's shape: ShortConv state no KV cache holds, so only `infer` is exported and the re-fed
        // loop is REQUIRED rather than merely tolerated. The preference must degrade to it silently.
        loom::LoomLuaBridge bridge(loom::Backends{backend.get(), nullptr});
        bridge.load_script("function infer(i) return 7 end\n");
        LOOM_CHECK(bridge.has_function("infer"));
        LOOM_CHECK(!bridge.has_function("infer_with_past"));
    }

    {
        // A global that exists but is NOT callable must not be selected -- `lua_getglobal` succeeding
        // is not the question, `lua_isfunction` is.
        loom::LoomLuaBridge bridge(loom::Backends{backend.get(), nullptr});
        bridge.load_script("infer_with_past = 42\nfunction infer(i) return 3 end\n");
        LOOM_CHECK(!bridge.has_function("infer_with_past"));
        LOOM_CHECK(bridge.has_function("infer"));
    }

    std::printf("ok\n");
    return 0;
}
