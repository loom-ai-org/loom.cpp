#include "codes_file.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace loom_cli {

namespace {

constexpr const char* kArchitecture = "loom-codes";

struct GgufFree {
    void operator()(gguf_context* g) const { gguf_free(g); }
};
struct GgmlFree {
    void operator()(ggml_context* c) const { ggml_free(c); }
};

} // namespace

void write_codes_gguf(const std::string& path, const std::vector<int32_t>& codes, uint32_t n_codebooks,
                      const std::string& source) {
    if (n_codebooks == 0 || codes.size() % n_codebooks != 0) {
        throw std::runtime_error("codes are not a whole number of " + std::to_string(n_codebooks) +
                                 "-wide frames");
    }
    const int64_t n_frames = static_cast<int64_t>(codes.size() / n_codebooks);
    ggml_init_params params{ggml_tensor_overhead() + codes.size() * sizeof(int32_t) + 64, nullptr, false};
    std::unique_ptr<ggml_context, GgmlFree> ctx(ggml_init(params));
    // ggml's ne[0] is the fastest-varying axis, so [n_codebooks, n_frames] here is the row-major
    // [n_frames, n_codebooks] a reader (gguf-py, numpy) sees.
    ggml_tensor* t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_codebooks, n_frames);
    ggml_set_name(t, "codes");
    if (!codes.empty()) std::memcpy(t->data, codes.data(), codes.size() * sizeof(int32_t));

    std::unique_ptr<gguf_context, GgufFree> g(gguf_init_empty());
    gguf_set_val_str(g.get(), "general.architecture", kArchitecture);
    gguf_set_val_u32(g.get(), "loom.codes.n_codebooks", n_codebooks);
    gguf_set_val_u32(g.get(), "loom.codes.n_frames", static_cast<uint32_t>(n_frames));
    if (!source.empty()) gguf_set_val_str(g.get(), "loom.codes.source", source.c_str());
    gguf_add_tensor(g.get(), t);
    if (!gguf_write_to_file(g.get(), path.c_str(), /*only_meta=*/false)) {
        throw std::runtime_error("could not write '" + path + "'");
    }
}

bool is_gguf(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    return in.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

CodesFile read_codes_gguf(const std::string& path) {
    ggml_context* raw_ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, &raw_ctx};
    std::unique_ptr<gguf_context, GgufFree> g(gguf_init_from_file(path.c_str(), params));
    std::unique_ptr<ggml_context, GgmlFree> ctx(raw_ctx);
    if (!g) throw std::runtime_error("could not read '" + path + "' as a GGUF");

    const int64_t arch = gguf_find_key(g.get(), "general.architecture");
    if (arch < 0 || gguf_get_kv_type(g.get(), arch) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(g.get(), arch)) != kArchitecture) {
        throw std::runtime_error("'" + path + "' is a GGUF but not a codes file (general.architecture is not '" +
                                 kArchitecture + "')");
    }
    CodesFile out;
    const int64_t width = gguf_find_key(g.get(), "loom.codes.n_codebooks");
    if (width < 0) throw std::runtime_error("'" + path + "' declares no loom.codes.n_codebooks");
    out.n_codebooks = gguf_get_val_u32(g.get(), width);
    const int64_t source = gguf_find_key(g.get(), "loom.codes.source");
    if (source >= 0) out.source = gguf_get_val_str(g.get(), source);

    const ggml_tensor* t = ggml_get_tensor(ctx.get(), "codes");
    if (t == nullptr || t->type != GGML_TYPE_I32 || ggml_n_dims(t) > 2 || t->ne[0] != out.n_codebooks) {
        throw std::runtime_error("'" + path + "' has no I32 `codes` tensor " + std::to_string(out.n_codebooks) +
                                 " codebooks wide");
    }
    const auto* data = static_cast<const int32_t*>(t->data);
    out.codes.assign(data, data + ggml_nelements(t));
    return out;
}

} // namespace loom_cli
