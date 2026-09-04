// `GgufModel::load_metadata`: the same file's KV table, without its weights.
//
// **The point is a cost, so the test is about EQUIVALENCE plus a guard.** A caller that only wants to
// know what a file IS -- its task, its modality pair, its declared hparams -- should not pay for
// every tensor being allocated on a backend and streamed off disk. For the largest model in this
// project that is 6.4 GB and several seconds to read a handful of strings sitting in the header,
// ahead of the tensor data. loom-py's model-card gate is what found it: six of its rows opened a
// whole model to read one `interface` string and then threw it away, which on a 6.4 GB card was half
// the row's memory and got the run OOM-killed.
//
// Two things have to hold, and the second is what makes the first safe to use:
//
//   1. **every metadata answer is identical** to the fully-loaded model's -- otherwise this is a
//      second, cheaper way to be told something different, which is the failure this project keeps
//      removing;
//   2. **asking a metadata-only model for a weight throws**, naming the loader. Its tensors exist as
//      correctly-shaped structs with null `data`, so the alternative to an error is a segfault deep
//      inside a graph build with nothing pointing back to here.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/contract_declared.gguf";
    auto full = loom::GgufModel::load(path, backend.get());
    auto meta = loom::GgufModel::load_metadata(path);
    LOOM_CHECK(full != nullptr && meta != nullptr);

    // --- 1. Which one is which, asked of the models themselves. ---
    LOOM_CHECK(full->has_weights());
    LOOM_CHECK(!meta->has_weights());

    // --- 2. Every metadata answer agrees. The contract is the reason this loader exists, so it is
    //        compared field by field rather than through one derived string. ---
    const loom::ModelContract a = loom::ModelContract::read(*full);
    const loom::ModelContract b = loom::ModelContract::read(*meta);
    LOOM_CHECK(a.task == b.task);
    LOOM_CHECK(a.input_kind == b.input_kind);
    LOOM_CHECK(a.output_kind == b.output_kind);
    LOOM_CHECK(a.sample_rate == b.sample_rate);
    LOOM_CHECK(a.clip_samples == b.clip_samples);
    LOOM_CHECK(a.text_frontend == b.text_frontend);
    LOOM_CHECK(a.languages == b.languages);
    LOOM_CHECK(a.labels == b.labels);
    LOOM_CHECK(a.interface_name() == b.interface_name());
    // Not vacuous: this fixture declares a real contract, so the comparison above is between two
    // populated values rather than between two defaults.
    LOOM_CHECK(!a.interface_name().empty());

    // ...and so do the raw accessors the contract is built out of, plus the topology JSON, which is a
    // KV like any other and is what a host needs to build a graph it has not yet loaded weights for.
    LOOM_CHECK(full->architecture() == meta->architecture());
    LOOM_CHECK(full->topology_names() == meta->topology_names());
    for (const std::string& name : full->topology_names()) {
        LOOM_CHECK(full->topology_json(name) == meta->topology_json(name));
    }

    // --- 3. The guard. Both weight accessors throw, and the message names the loader so the fix is
    //        readable from the error alone. ---
    const std::string any_weight = "token_embd.weight";
    bool threw = false;
    try {
        (void)meta->weight(any_weight);
    } catch (const loom::Error& e) {
        threw = std::string(e.what()).find("load_metadata") != std::string::npos;
    }
    LOOM_CHECK(threw);

    threw = false;
    try {
        (void)meta->has_weight(any_weight);
    } catch (const loom::Error& e) {
        threw = std::string(e.what()).find("load_metadata") != std::string::npos;
    }
    LOOM_CHECK(threw);

    // The same call on the fully-loaded model does NOT throw -- otherwise the guard above would be
    // satisfied by a weight accessor that was simply broken.
    LOOM_CHECK(full->has_weight(any_weight) || !full->has_weight(any_weight));

    LOOM_TEST_REPORT_AND_RETURN();
}
