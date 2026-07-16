#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <fstream>
#include <vector>

namespace loom {

std::unique_ptr<GgufModel> GgufModel::load(const std::string& path, ggml_backend_t backend) {
    auto model = std::unique_ptr<GgufModel>(new GgufModel());
    model->backend_ = backend;

    // Parse the GGUF container with no_alloc=true: this populates meta_ctx_ with correctly-shaped
    // ggml_tensor structs but leaves their `data` pointers null -- we bind real storage ourselves below
    // via ggml_backend_alloc_ctx_tensors, so the exact same loader works when `backend` is a GPU backend.
    ggml_context* raw_meta_ctx = nullptr;
    gguf_init_params gguf_params{/*no_alloc=*/true, /*ctx=*/&raw_meta_ctx};
    gguf_context* raw_gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!raw_gguf_ctx) {
        throw LoadError("GgufModel::load: failed to parse GGUF file '" + path + "'");
    }
    model->gguf_ctx_.reset(raw_gguf_ctx);
    model->meta_ctx_.reset(raw_meta_ctx);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(model->meta_ctx_.get(), backend);
    if (!buf) {
        throw LoadError("GgufModel::load: failed to allocate a backend buffer for weights in '" + path + "'");
    }
    model->weights_buf_.reset(buf);

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw LoadError("GgufModel::load: failed to reopen '" + path + "' to read tensor data");
    }

    const size_t data_offset = gguf_get_data_offset(model->gguf_ctx_.get());
    const int64_t n_tensors = gguf_get_n_tensors(model->gguf_ctx_.get());
    std::vector<uint8_t> staging;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(model->gguf_ctx_.get(), i);
        ggml_tensor* t = ggml_get_tensor(model->meta_ctx_.get(), name);
        if (!t) {
            throw LoadError("GgufModel::load: tensor '" + std::string(name) +
                             "' listed in GGUF KV table but missing from the ggml context");
        }

        const size_t offset = data_offset + gguf_get_tensor_offset(model->gguf_ctx_.get(), i);
        const size_t nbytes = ggml_nbytes(t);
        staging.resize(nbytes);
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char*>(staging.data()), static_cast<std::streamsize>(nbytes));
        if (!file) {
            throw LoadError("GgufModel::load: failed to read " + std::to_string(nbytes) +
                             " bytes for tensor '" + std::string(name) + "' from '" + path + "'");
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        model->symbols_[name] = t;
    }

    const int64_t topo_kv = gguf_find_key(model->gguf_ctx_.get(), "model.graph_topology");
    if (topo_kv < 0) {
        throw LoadError("GgufModel::load: '" + path + "' is missing the required 'model.graph_topology' KV");
    }
    if (gguf_get_kv_type(model->gguf_ctx_.get(), topo_kv) != GGUF_TYPE_STRING) {
        throw LoadError("GgufModel::load: 'model.graph_topology' KV in '" + path + "' is not a string");
    }
    model->topology_json_ = gguf_get_val_str(model->gguf_ctx_.get(), topo_kv);

    return model;
}

ggml_tensor* GgufModel::weight(const std::string& name) const {
    auto it = symbols_.find(name);
    if (it == symbols_.end()) {
        throw LoadError("GgufModel::weight: no such weight '" + name + "'");
    }
    return it->second;
}

bool GgufModel::has_weight(const std::string& name) const {
    return symbols_.find(name) != symbols_.end();
}

uint32_t GgufModel::hparam_u32(const std::string& key) const {
    const std::string full = "loom." + key;
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full.c_str());
    if (kv < 0) {
        throw LoadError("GgufModel::hparam_u32: missing key '" + full + "'");
    }
    const gguf_type t = gguf_get_kv_type(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_UINT32) return gguf_get_val_u32(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_INT32)  return static_cast<uint32_t>(gguf_get_val_i32(gguf_ctx_.get(), kv));
    throw LoadError("GgufModel::hparam_u32: key '" + full + "' is not a u32/i32 KV");
}

float GgufModel::hparam_f32(const std::string& key) const {
    const std::string full = "loom." + key;
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full.c_str());
    if (kv < 0) {
        throw LoadError("GgufModel::hparam_f32: missing key '" + full + "'");
    }
    const gguf_type t = gguf_get_kv_type(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_FLOAT64) return static_cast<float>(gguf_get_val_f64(gguf_ctx_.get(), kv));
    throw LoadError("GgufModel::hparam_f32: key '" + full + "' is not a f32/f64 KV");
}

std::string GgufModel::hparam_str(const std::string& key) const {
    const std::string full = "loom." + key;
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full.c_str());
    if (kv < 0) {
        throw LoadError("GgufModel::hparam_str: missing key '" + full + "'");
    }
    if (gguf_get_kv_type(gguf_ctx_.get(), kv) != GGUF_TYPE_STRING) {
        throw LoadError("GgufModel::hparam_str: key '" + full + "' is not a string KV");
    }
    return gguf_get_val_str(gguf_ctx_.get(), kv);
}

bool GgufModel::has_kv(const std::string& full_key) const {
    return gguf_find_key(gguf_ctx_.get(), full_key.c_str()) >= 0;
}

std::string GgufModel::kv_str(const std::string& full_key) const {
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full_key.c_str());
    if (kv < 0) {
        throw LoadError("GgufModel::kv_str: missing key '" + full_key + "'");
    }
    if (gguf_get_kv_type(gguf_ctx_.get(), kv) != GGUF_TYPE_STRING) {
        throw LoadError("GgufModel::kv_str: key '" + full_key + "' is not a string KV");
    }
    return gguf_get_val_str(gguf_ctx_.get(), kv);
}

bool GgufModel::kv_bool(const std::string& full_key, bool default_value) const {
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full_key.c_str());
    if (kv < 0) return default_value;
    if (gguf_get_kv_type(gguf_ctx_.get(), kv) != GGUF_TYPE_BOOL) {
        throw LoadError("GgufModel::kv_bool: key '" + full_key + "' is not a bool KV");
    }
    return gguf_get_val_bool(gguf_ctx_.get(), kv);
}

int32_t GgufModel::kv_i32(const std::string& full_key, int32_t default_value) const {
    const int64_t kv = gguf_find_key(gguf_ctx_.get(), full_key.c_str());
    if (kv < 0) return default_value;
    const gguf_type t = gguf_get_kv_type(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_INT32)  return gguf_get_val_i32(gguf_ctx_.get(), kv);
    if (t == GGUF_TYPE_UINT32) return static_cast<int32_t>(gguf_get_val_u32(gguf_ctx_.get(), kv));
    throw LoadError("GgufModel::kv_i32: key '" + full_key + "' is not an i32/u32 KV");
}

namespace {
int64_t find_array_kv(const gguf_context* ctx, const std::string& full_key, gguf_type expect_elem_type) {
    const int64_t kv = gguf_find_key(ctx, full_key.c_str());
    if (kv < 0) {
        throw LoadError("GgufModel: missing array key '" + full_key + "'");
    }
    if (gguf_get_kv_type(ctx, kv) != GGUF_TYPE_ARRAY) {
        throw LoadError("GgufModel: key '" + full_key + "' is not an array KV");
    }
    if (gguf_get_arr_type(ctx, kv) != expect_elem_type) {
        throw LoadError("GgufModel: array key '" + full_key + "' has an unexpected element type");
    }
    return kv;
}
} // namespace

std::vector<std::string> GgufModel::kv_arr_str(const std::string& full_key) const {
    const int64_t kv = find_array_kv(gguf_ctx_.get(), full_key, GGUF_TYPE_STRING);
    const size_t n = gguf_get_arr_n(gguf_ctx_.get(), kv);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(gguf_ctx_.get(), kv, i));
    }
    return out;
}

std::vector<float> GgufModel::kv_arr_f32(const std::string& full_key) const {
    const int64_t kv = find_array_kv(gguf_ctx_.get(), full_key, GGUF_TYPE_FLOAT32);
    const size_t n = gguf_get_arr_n(gguf_ctx_.get(), kv);
    const auto* data = static_cast<const float*>(gguf_get_arr_data(gguf_ctx_.get(), kv));
    return std::vector<float>(data, data + n);
}

std::vector<int32_t> GgufModel::kv_arr_i32(const std::string& full_key) const {
    const int64_t kv = find_array_kv(gguf_ctx_.get(), full_key, GGUF_TYPE_INT32);
    const size_t n = gguf_get_arr_n(gguf_ctx_.get(), kv);
    const auto* data = static_cast<const int32_t*>(gguf_get_arr_data(gguf_ctx_.get(), kv));
    return std::vector<int32_t>(data, data + n);
}

std::vector<uint8_t> GgufModel::kv_arr_u8(const std::string& full_key) const {
    const int64_t kv = find_array_kv(gguf_ctx_.get(), full_key, GGUF_TYPE_UINT8);
    const size_t n = gguf_get_arr_n(gguf_ctx_.get(), kv);
    const auto* data = static_cast<const uint8_t*>(gguf_get_arr_data(gguf_ctx_.get(), kv));
    return std::vector<uint8_t>(data, data + n);
}

SymbolEnv GgufModel::hparam_env() const {
    SymbolEnv env;
    const std::string prefix = "loom.";
    const int64_t n_kv = gguf_get_n_kv(gguf_ctx_.get());
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string key(gguf_get_key(gguf_ctx_.get(), i));
        if (key.rfind(prefix, 0) != 0) continue; // doesn't start with "loom."
        const std::string bare = key.substr(prefix.size());

        double value = 0.0;
        switch (gguf_get_kv_type(gguf_ctx_.get(), i)) {
            case GGUF_TYPE_UINT8:   value = gguf_get_val_u8(gguf_ctx_.get(), i);  break;
            case GGUF_TYPE_INT8:    value = gguf_get_val_i8(gguf_ctx_.get(), i);  break;
            case GGUF_TYPE_UINT16:  value = gguf_get_val_u16(gguf_ctx_.get(), i); break;
            case GGUF_TYPE_INT16:   value = gguf_get_val_i16(gguf_ctx_.get(), i); break;
            case GGUF_TYPE_UINT32:  value = gguf_get_val_u32(gguf_ctx_.get(), i); break;
            case GGUF_TYPE_INT32:   value = gguf_get_val_i32(gguf_ctx_.get(), i); break;
            case GGUF_TYPE_FLOAT32: value = gguf_get_val_f32(gguf_ctx_.get(), i); break;
            case GGUF_TYPE_UINT64:  value = static_cast<double>(gguf_get_val_u64(gguf_ctx_.get(), i)); break;
            case GGUF_TYPE_INT64:   value = static_cast<double>(gguf_get_val_i64(gguf_ctx_.get(), i)); break;
            case GGUF_TYPE_FLOAT64: value = gguf_get_val_f64(gguf_ctx_.get(), i); break;
            default: continue; // string/bool/array hparams aren't symbol-evaluable; skip
        }
        env.set(bare, value);
    }
    return env;
}

} // namespace loom
