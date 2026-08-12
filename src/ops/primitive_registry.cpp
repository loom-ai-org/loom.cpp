#include "loom/ops/primitive_registry.h"
#include "loom/loom_errors.h"

#include <ggml-backend.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>

namespace loom {

bool is_materialized(const ggml_tensor* t) {
    return t != nullptr && t->buffer != nullptr && t->data != nullptr;
}

void read_tensor_prefix(const ggml_tensor* t, void* dst, size_t nbytes) {
    if (!is_materialized(t)) {
        throw Error("read_tensor_prefix: tensor '" + std::string(t != nullptr ? ggml_get_name(t) : "(null)") +
                    "' has no data to read at graph-build time");
    }
    if (nbytes > ggml_nbytes(t)) {
        throw Error("read_tensor_prefix: asked for " + std::to_string(nbytes) + " bytes of tensor '" +
                    std::string(ggml_get_name(t)) + "', which holds " + std::to_string(ggml_nbytes(t)));
    }
    // Not a memcpy: on a device backend this is the download that makes the value host-readable at all.
    ggml_backend_tensor_get(const_cast<ggml_tensor*>(t), dst, 0, nbytes);
}

float scalar_value_or(const ggml_tensor* t, float fallback) {
    if (!is_materialized(t) || ggml_nelements(t) < 1) return fallback;
    switch (t->type) {
        case GGML_TYPE_F32: { float v = 0.0f;   read_tensor_prefix(t, &v, sizeof(v)); return v; }
        case GGML_TYPE_I32: { int32_t v = 0;    read_tensor_prefix(t, &v, sizeof(v)); return static_cast<float>(v); }
        case GGML_TYPE_I16: { int16_t v = 0;    read_tensor_prefix(t, &v, sizeof(v)); return static_cast<float>(v); }
        case GGML_TYPE_I8:  { int8_t  v = 0;    read_tensor_prefix(t, &v, sizeof(v)); return static_cast<float>(v); }
        default: return fallback;
    }
}

PrimitiveRegistry& PrimitiveRegistry::instance() {
    static PrimitiveRegistry registry;
    return registry;
}

void PrimitiveRegistry::register_op(const std::string& name, PrimitiveFn fn) {
    table_[name] = std::move(fn);
}

bool PrimitiveRegistry::has(const std::string& name) const {
    return table_.find(name) != table_.end();
}

const PrimitiveFn& PrimitiveRegistry::get(const std::string& name) const {
    auto it = table_.find(name);
    if (it == table_.end()) {
        throw UnknownOpError("PrimitiveRegistry: unknown op '" + name + "'");
    }
    return it->second;
}

double resolve_attr_number(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env) {
    if (!attrs.contains(key)) {
        throw SchemaError("resolve_attr_number: missing attribute '" + key + "'");
    }
    const nlohmann::json& v = attrs.at(key);
    if (v.is_string()) return env.eval(v.get<std::string>());
    if (v.is_number()) return v.get<double>();
    throw SchemaError("resolve_attr_number: attribute '" + key + "' must be a number or a symbol expression string");
}

int64_t resolve_attr_int(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env) {
    return static_cast<int64_t>(std::llround(resolve_attr_number(attrs, key, env)));
}

std::vector<int64_t> resolve_attr_int_array(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env) {
    if (!attrs.contains(key)) {
        throw SchemaError("resolve_attr_int_array: missing attribute '" + key + "'");
    }
    const nlohmann::json& arr = attrs.at(key);
    if (!arr.is_array()) {
        throw SchemaError("resolve_attr_int_array: attribute '" + key + "' must be an array");
    }
    std::vector<int64_t> out;
    out.reserve(arr.size());
    for (const nlohmann::json& v : arr) {
        if (v.is_string()) out.push_back(static_cast<int64_t>(std::llround(env.eval(v.get<std::string>()))));
        else if (v.is_number()) out.push_back(static_cast<int64_t>(std::llround(v.get<double>())));
        else throw SchemaError("resolve_attr_int_array: element of '" + key + "' must be a number or a symbol expression string");
    }
    return out;
}

} // namespace loom
