#include "loom/core/voice_file.h"

#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <ggml.h>
#include <gguf.h>

#include <memory>

namespace loom {
namespace {

struct GgufFree {
    void operator()(gguf_context* g) const { gguf_free(g); }
};
struct GgmlFree {
    void operator()(ggml_context* c) const { ggml_free(c); }
};

std::string voice_str(const gguf_context* g, const std::string& path, const char* key, bool required) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
        if (!required) return {};
        throw LoadError("load_voice: " + path + " has no string '" + key + "', so it is not a loom voice file");
    }
    return gguf_get_val_str(g, id);
}

} // namespace

VoiceFile load_voice(const GgufModel& model, const std::string& path) {
    if (!model.has_kv("loom.voice.compat")) {
        throw SchemaError("load_voice: this model (" + model.architecture() + ") declares no "
                          "'loom.voice.compat', so it takes no voice files -- its voices, if any, are "
                          "chosen some other way");
    }
    ggml_context* raw_ctx = nullptr;
    gguf_init_params params{/*no_alloc=*/false, &raw_ctx};
    std::unique_ptr<gguf_context, GgufFree> g(gguf_init_from_file(path.c_str(), params));
    std::unique_ptr<ggml_context, GgmlFree> ctx(raw_ctx);
    if (!g) {
        throw LoadError("load_voice: could not read '" + path + "' as a GGUF");
    }
    const std::string architecture = voice_str(g.get(), path, "loom.voice.architecture", true);
    const std::string compat = voice_str(g.get(), path, "loom.voice.compat", true);
    if (architecture != model.architecture()) {
        throw SchemaError("load_voice: " + path + " is a voice for '" + architecture + "', and this model is '" +
                          model.architecture() + "'");
    }
    const std::string expected = model.kv_str("loom.voice.compat");
    if (compat != expected) {
        throw SchemaError("load_voice: " + path + " was made for other weights (fingerprint " + compat +
                          ", this model's is " + expected + "). A voice state only fits the weights that "
                          "produced it; convert it from this checkpoint's own voices instead");
    }

    VoiceFile voice;
    voice.name = voice_str(g.get(), path, "loom.voice.name", false);
    voice.license = voice_str(g.get(), path, "loom.voice.license", false);
    voice.origin = voice_str(g.get(), path, "loom.voice.origin", false);
    const int64_t n = gguf_get_n_tensors(g.get());
    if (n == 0) throw LoadError("load_voice: " + path + " holds no tensors, so it sets no input");
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(g.get(), i);
        const ggml_tensor* t = ggml_get_tensor(ctx.get(), name);
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            throw LoadError("load_voice: " + path + "'s tensor '" + name + "' is not F32; a voice is the "
                            "state that was saved, and a converted one is not");
        }
        const auto* data = static_cast<const float*>(t->data);
        voice.inputs[name].assign(data, data + ggml_nelements(t));
    }
    return voice;
}

} // namespace loom
