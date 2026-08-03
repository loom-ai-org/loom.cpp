// Elementwise logit comparison between two exports of the SAME checkpoint -- specifically the unfused
// MIL causal-LM topology and the fused (ATTENTION-node) one, to turn "the greedy continuations diverge
// on a high-entropy prompt, and I believe that is rounding" into a measured number.
//
// Drives GraphBuilder directly rather than going through the Lua driver, because the driver's `infer`
// entry returns an argmax and the whole question is about the logit vector behind it.

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

// Runs one prefill (n_past = 0) and returns the logit row for the LAST position, which is the row the
// driver's argmax epilogue reads.
std::vector<float> last_row_logits(const std::string& path, const std::vector<int32_t>& tokens,
                                    int64_t& n_vocab_out) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto model = loom::GgufModel::load(path, backend.get());
    // "main_topo" is the pre-KV-CACHE.md-N.2 name, kept so this can also run GGUFs exported before
    // this session -- which is how "did I introduce this?" gets answered rather than argued.
    const std::string topo_name = model->has_topology("main_topology") ? "main_topology" : "main_topo";
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(topo_name));

    std::unique_ptr<loom::KvCache> kv_cache;
    if (topo.uses_kv_cache()) {
        kv_cache = loom::make_kv_cache(*model, backend.get());
    }

    const auto n_tokens = static_cast<uint32_t>(tokens.size());
    loom::GraphBuilder builder(topo, *model, backend.get(), kv_cache.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", n_tokens}, {"n_past", 0}});

    // Exactly what DriverInputs emits: cache_position = loom.range(0, n_tokens), attention_mask =
    // loom.causal_mask(n_tokens, 0). Built here rather than read from the driver so both models get
    // byte-identical inputs by construction.
    std::vector<int32_t> cache_position(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) cache_position[i] = static_cast<int32_t>(i);
    std::vector<float> mask(static_cast<size_t>(n_tokens) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (uint32_t j = 0; j < n_tokens; ++j) {
            mask[static_cast<size_t>(i) * n_tokens + j] =
                (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, tokens.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("cache_position"), cache_position.data(), 0,
                            cache_position.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("attention_mask"), mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);

    ggml_tensor* out = r.outputs.front();
    const int64_t n_vocab = out->ne[0];
    n_vocab_out = n_vocab;
    std::vector<float> all(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, all.data(), 0, all.size() * sizeof(float));

    const size_t offset = static_cast<size_t>(n_tokens - 1) * static_cast<size_t>(n_vocab);
    return std::vector<float>(all.begin() + offset, all.begin() + offset + n_vocab);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) if (v[i] > v[best]) best = i;
    return best;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <a.gguf> <b.gguf> <tok0,tok1,...>\n", argv[0]);
        return 2;
    }
    std::vector<int32_t> tokens;
    for (const char* p = argv[3]; *p;) {
        tokens.push_back(static_cast<int32_t>(std::strtol(p, const_cast<char**>(&p), 10)));
        if (*p == ',') ++p;
    }

    int64_t nva = 0, nvb = 0;
    std::vector<float> a = last_row_logits(argv[1], tokens, nva);
    std::vector<float> b = last_row_logits(argv[2], tokens, nvb);
    if (nva != nvb) { std::fprintf(stderr, "vocab mismatch %ld vs %ld\n", nva, nvb); return 1; }

    double max_abs = 0.0, sum_abs = 0.0, max_rel = 0.0;
    int max_at = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        sum_abs += d;
        if (d > max_abs) { max_abs = d; max_at = static_cast<int>(i); }
        const double denom = std::max(std::fabs((double)a[i]), std::fabs((double)b[i]));
        if (denom > 1.0) max_rel = std::max(max_rel, d / denom);
    }

    // The margin the argmax actually had to survive: top-1 minus top-2 in the reference run.
    std::vector<float> sorted = a;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const double margin = static_cast<double>(sorted[0]) - static_cast<double>(sorted[1]);

    std::printf("%2zu tokens | argmax %s (%d vs %d) | max|d|=%.3e at %d | mean|d|=%.3e | "
                "max rel=%.2e | top1-top2 margin=%.4f\n",
                tokens.size(), argmax(a) == argmax(b) ? "SAME " : "DIFF ", argmax(a), argmax(b),
                max_abs, max_at, sum_abs / a.size(), max_rel, margin);

    for (const auto& pair : {std::make_pair("unfused", &a), std::make_pair("fused  ", &b)}) {
        std::vector<int> idx(pair.second->size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int x, int y) { return (*pair.second)[x] > (*pair.second)[y]; });
        std::printf("   %s top5:", pair.first);
        for (int i = 0; i < 5; ++i) std::printf(" %d(%.4f)", idx[i], (*pair.second)[idx[i]]);
        std::printf("\n");
    }

    if (argc >= 6) {
        for (const auto& pair : {std::make_pair(argv[4], &a), std::make_pair(argv[5], &b)}) {
            std::FILE* f = std::fopen(pair.first, "wb");
            std::fwrite(pair.second->data(), sizeof(float), pair.second->size(), f);
            std::fclose(f);
        }
    }
    return 0;
}
