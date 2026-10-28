#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"
#include "loom/loom_errors.h"

#include <ggml-alloc.h>

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {

namespace {

// The CPU backend's extra buffer type called `name`, or null if `backend` is not the CPU or this ggml
// build does not register one.
ggml_backend_buffer_type_t find_extra_buffer_type(ggml_backend_t backend, const char* name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) return nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    if (get_extra == nullptr) return nullptr;
    for (ggml_backend_buffer_type_t* p = get_extra(dev); p != nullptr && *p != nullptr; ++p) {
        if (std::strcmp(ggml_backend_buft_name(*p), name) == 0) return *p;
    }
    return nullptr;
}

// Which of the CPU backend's extra buffer types LOOM_CPU_REPACK names: "1" / "repack" is ggml's own
// CPU_REPACK, "kleidiai" is CPU_KLEIDIAI (registered only by a GGML_CPU_KLEIDIAI=ON build). Null when
// unset, "0", or the named type is not registered -- the plain buffer then holds everything.
ggml_backend_buffer_type_t requested_extra_buffer_type(ggml_backend_t backend) {
    const char* e = std::getenv("LOOM_CPU_REPACK");
    if (e == nullptr || e[0] == '\0' || std::strcmp(e, "0") == 0) return nullptr;
    const char* want = std::strcmp(e, "kleidiai") == 0 ? "CPU_KLEIDIAI" : "CPU_REPACK";
    if (ggml_backend_buffer_type_t buft = find_extra_buffer_type(backend, want)) return buft;
    std::fprintf(stderr, "loom: LOOM_CPU_REPACK=%s, but this ggml build registers no %s buffer type; "
                         "weights stay in the plain buffer\n", e, want);
    return nullptr;
}

// ADR-079 P4.33: the convolutions that read each name, from the topologies' own text. CPU_ARMV6_I8 serves
// mul_mat and the pointwise convolutions, and a convolution it does not serve could not take the direct
// path with its kernel there (the CPU device declines a node whose source is in a non-host buffer), so
// the engine lowered it through im2col and a mul_mat: on the Pi Zero, with every Q8_0_PC weight placed,
// Citrinet-256-ls ran 0.86x and WakeHuBERT tiny 0.89x of their F32 cards. So a weight a convolution reads
// is placed only if the buffer type accepts that convolution, asked with a probe of the node the
// lowering builds (op_conv_1d's ggml_conv_2d_direct_packed). A repeat_for block's "{i}" is matched as any
// index, and an alias as its target.
struct ConvReader {
    std::string op;
    nlohmann::json attrs;
};

struct ConvReaders {
    std::unordered_map<std::string, std::vector<ConvReader>> names;
    std::vector<std::pair<std::regex, ConvReader>> patterns;

    void collect(const std::string& name, std::vector<ConvReader>& out) const {
        if (auto it = names.find(name); it != names.end()) out.insert(out.end(), it->second.begin(), it->second.end());
        for (const auto& [re, reader] : patterns) {
            if (std::regex_match(name, re)) out.push_back(reader);
        }
    }
};

ConvReaders conv_readers(gguf_context* gguf) {
    ConvReaders out;
    auto add = [&](const TopologyNode& node, const std::string& index_var) {
        if (node.op.find("CONV") == std::string::npos) return;
        const ConvReader reader{node.op, node.attrs};
        for (const std::string& in : node.inputs) {
            const std::string token = "{" + index_var + "}";
            if (index_var.empty() || in.find(token) == std::string::npos) {
                out.names[in].push_back(reader);
                continue;
            }
            std::string re;
            for (size_t i = 0; i < in.size();) {
                if (in.compare(i, token.size(), token) == 0) {
                    re += "[0-9]+";
                    i += token.size();
                } else {
                    if (std::strchr(".^$|()[]{}*+?\\", in[i])) re += '\\';
                    re += in[i++];
                }
            }
            out.patterns.emplace_back(std::regex(re), reader);
        }
    };
    const int64_t n_kv = gguf_get_n_kv(gguf);
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string key(gguf_get_key(gguf, i));
        if (key.rfind("model.graph_topology", 0) != 0 || gguf_get_kv_type(gguf, i) != GGUF_TYPE_STRING) continue;
        const GraphTopology topo = GraphTopology::parse(gguf_get_val_str(gguf, i));
        for (const TopologyItem& item : topo.items) {
            if (!item.is_repeat) {
                add(item.node, "");
            } else {
                for (const TopologyNode& node : item.repeat.nodes) add(node, item.repeat.index_var);
            }
        }
    }
    return out;
}

// The alias names that resolve to `target` (loom.tensor_alias.*), so a convolution reading a weight
// under another name is still seen.
std::vector<std::string> aliases_of(gguf_context* gguf, const std::string& target) {
    std::vector<std::string> out;
    const int64_t kn = gguf_find_key(gguf, "loom.tensor_alias.names");
    const int64_t kt = gguf_find_key(gguf, "loom.tensor_alias.targets");
    if (kn < 0 || kt < 0) return out;
    for (size_t i = 0; i < gguf_get_arr_n(gguf, kt) && i < gguf_get_arr_n(gguf, kn); ++i) {
        if (target == gguf_get_arr_str(gguf, kt, i)) out.emplace_back(gguf_get_arr_str(gguf, kn, i));
    }
    return out;
}

// Whether `buft` would serve every convolution in `readers` with `t` as its folded kernel: the node
// op_conv_1d builds for a folded kernel, probed. Any other convolution, or one whose geometry the
// topology does not state, answers no.
bool serves_convs(ggml_backend_t backend, ggml_backend_buffer_type_t buft, const ggml_tensor* t,
                  const std::vector<ConvReader>& readers) {
    if (readers.empty()) return true;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_ptr probe(ggml_backend_buft_alloc_buffer(buft, 64));
    if (!probe) return false;
    ggml_context_ptr ctx(ggml_init(ggml_init_params{8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true}));
    ggml_tensor* w = ggml_dup_tensor(ctx.get(), t);
    w->buffer = probe.get();
    ggml_backend_buffer_init_tensor(probe.get(), w);
    for (const ConvReader& r : readers) {
        bool known = true;   // an attr that is present but not an integer (a symbol) makes the geometry unknown
        auto num = [&](const char* key, int def) {
            const auto it = r.attrs.find(key);
            if (it == r.attrs.end()) return def;
            if (!it->is_number_integer()) known = false;
            return known ? it->get<int>() : def;
        };
        const int k = num("kernel_k", 0), s0 = num("s0", 1), p0 = num("p0", 0), d0 = num("d0", 1), groups = num("groups", 1);
        if (!known || r.op != "CONV_1D" || groups != 1 || k <= 0 || t->ne[0] % k != 0 || s0 < 1 || d0 < 1) return false;
        const int64_t ic = t->ne[0] / k;
        ggml_tensor* x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 16 * s0 + (k - 1) * d0, 1, ic, 1);
        ggml_tensor* op = ggml_conv_2d_direct_packed(ctx.get(), w, x, s0, 1, p0, 0, d0, 1, k, 1, static_cast<int>(ic));
        if (!ggml_backend_dev_supports_op(dev, op)) return false;
    }
    return true;
}

// ADR-079 P4.33: the weights an export wrote as per-channel int8 (ADR-076's Q8_0_PC, listed under
// `loom.quantize.per_channel`) go to a per-channel int8 buffer type by default, the only extra buffer
// types placed without being asked for: CPU_ARMV6_I8 (ggml registers it on ARMv6 only, and not with
// GGML_CPU_ARMV6_I8=0) or CPU_NEON_I8 (P4.37: AArch64 without DOTPROD, the Pi 4's A72; not with
// GGML_CPU_NEON_I8=0). At most one of them exists on a core. An explicit LOOM_CPU_REPACK (any value, "0"
// included) takes precedence, so the A/B switches keep their meaning.
ggml_backend_buffer_type_t per_channel_buffer_type(ggml_backend_t backend, gguf_context* gguf, ggml_context* meta,
                                                   std::unordered_set<std::string>& names) {
    const char* e = std::getenv("LOOM_CPU_REPACK");
    if (e != nullptr && e[0] != '\0') return nullptr;
    const int64_t kv = gguf_find_key(gguf, "loom.quantize.per_channel");
    if (kv < 0 || gguf_get_kv_type(gguf, kv) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf, kv) != GGUF_TYPE_STRING) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = find_extra_buffer_type(backend, "CPU_ARMV6_I8");
    if (buft == nullptr) buft = find_extra_buffer_type(backend, "CPU_NEON_I8");
    if (buft == nullptr) return nullptr;
    const ConvReaders convs = conv_readers(gguf);
    for (size_t i = 0; i < gguf_get_arr_n(gguf, kv); ++i) {
        const std::string name = gguf_get_arr_str(gguf, kv, i);
        const ggml_tensor* t = ggml_get_tensor(meta, name.c_str());
        if (t == nullptr) continue;
        std::vector<ConvReader> readers;
        convs.collect(name, readers);
        for (const std::string& alias : aliases_of(gguf, name)) convs.collect(alias, readers);
        if (serves_convs(backend, buft, t, readers)) names.insert(name);
    }
    return names.empty() ? nullptr : buft;
}

// The Q4_0 / Q8_0 matrices `buft` would actually serve, asked the way llama.cpp asks it: a probe
// `mul_mat` over the weight as if it lived in that buffer, put to the CPU device's supports_op, which
// consults the extra buffer type itself. That answers per tensor and per CPU (a type with no kernel on
// this core, a row count that is not a multiple of the interleave) without restating ggml's rules here.
// CPU_REPACK also needs `extra` set by init_tensor, because its set_tensor dereferences it unchecked.
std::vector<ggml_tensor*> extra_buffer_candidates(ggml_context* ctx, ggml_backend_t backend,
                                                  ggml_backend_buffer_type_t buft) {
    std::vector<ggml_tensor*> out;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_ptr probe(ggml_backend_buft_alloc_buffer(buft, 64));
    if (!probe) return out;
    int n = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) ++n;
    ggml_context_ptr pctx(ggml_init(ggml_init_params{(3 * static_cast<size_t>(n) + 8) * ggml_tensor_overhead(),
                                                     nullptr, /*no_alloc=*/true}));
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if ((t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q8_0) || ggml_n_dims(t) != 2) continue;
        ggml_tensor* w = ggml_dup_tensor(pctx.get(), t);
        w->buffer = probe.get();
        ggml_backend_buffer_init_tensor(probe.get(), w);
        ggml_tensor* x = ggml_new_tensor_2d(pctx.get(), GGML_TYPE_F32, t->ne[0], 8);
        ggml_tensor* op = ggml_mul_mat(pctx.get(), w, x);
        if (w->extra != nullptr && ggml_backend_dev_supports_op(dev, op)) out.push_back(t);
    }
    return out;
}

// Places `tensors` in a fresh buffer of `buft`, each at its padded offset, the way
// ggml_backend_alloc_ctx_tensors does for a whole context.
ggml_backend_buffer_t alloc_tensors(ggml_backend_buffer_type_t buft, const std::vector<ggml_tensor*>& tensors) {
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    for (ggml_tensor* t : tensors) size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, std::max<size_t>(size, align));
    if (buf == nullptr) return nullptr;
    ggml_tallocr talloc = ggml_tallocr_new(buf);
    for (ggml_tensor* t : tensors) {
        if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
    }
    return buf;
}

} // namespace

std::unique_ptr<GgufModel> GgufModel::load(const std::string& path, Backends backends) {
    // The weights go on the PRIMARY backend and nowhere else. When a scheduler later runs part of the
    // graph on the CPU, it is the scheduler that makes the host-side copies of whatever weights that
    // part reads -- an arrangement this loader neither knows nor needs to know about (BACKLOG.md P4.7).
    ggml_backend_t backend = backends.primary;
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

    // Refuse a device that says it cannot hold the weights, before asking it to. Exactly what
    // ggml_backend_alloc_ctx_tensors will allocate: each tensor's size as the buffer type pads it.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
        !ggml_backend_buft_is_host(buft)) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        size_t needed = 0;
        for (ggml_tensor* t = ggml_get_first_tensor(model->meta_ctx_.get()); t != nullptr;
             t = ggml_get_next_tensor(model->meta_ctx_.get(), t)) {
            needed += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
        }
        if (total != 0 && needed > free) {
            char numbers[160];
            std::snprintf(numbers, sizeof(numbers), "needs %.2f GB for its weights and %s has %.2f GB free of %.2f GB",
                          needed / 1e9, ggml_backend_dev_name(dev), free / 1e9, total / 1e9);
            throw LoadError("GgufModel::load: '" + path + "' " + numbers + " (" +
                            ggml_backend_dev_description(dev) + "). Free the models already loaded "
                            "there, or load this one with device 'cpu' -- or 'auto', which skips a "
                            "device that cannot hold the weights.");
        }
    }

    // Extra-buffer placement (route_repacked_uses has the why). These are the CPU's buffer types, so
    // only where the CPU is what reads those weights: a plain CPU primary, or the CPU fallback beside a
    // primary whose tensors are host memory (P4.44: ggml-vc4 on a Pi Zero, where "auto" made every
    // Q8_0_PC model lose CPU_ARMV6_I8). Such a primary cannot claim a repacked weight -- the buffer type
    // is not host memory, so its supports_buft says no -- and the scheduler gives those nodes to the CPU.
    // A primary with its own memory gets none: the scheduler plans its copies from each weight's
    // buffer and would have to read a repacked one back.
    model->path_ = path;
    ggml_backend_t repack_backend = !backends.hybrid() ? backend
                                    : ggml_backend_buft_is_host(buft) ? backends.fallback : nullptr;
    model->repack_backend_ = repack_backend;
    std::vector<ggml_tensor*> to_repack;
    ggml_backend_buffer_type_t repack_buft = repack_backend == nullptr ? nullptr : requested_extra_buffer_type(repack_backend);
    std::unordered_set<std::string> per_channel;
    if (repack_buft == nullptr && repack_backend != nullptr) {
        repack_buft = per_channel_buffer_type(repack_backend, model->gguf_ctx_.get(), model->meta_ctx_.get(), per_channel);
    }
    if (repack_buft != nullptr) {
        to_repack = extra_buffer_candidates(model->meta_ctx_.get(), repack_backend, repack_buft);
        if (!per_channel.empty()) {
            to_repack.erase(std::remove_if(to_repack.begin(), to_repack.end(),
                                           [&](ggml_tensor* t) { return per_channel.count(ggml_get_name(t)) == 0; }),
                            to_repack.end());
        }
    }
    ggml_backend_buffer_t buf = nullptr;
    if (to_repack.empty()) {
        buf = ggml_backend_alloc_ctx_tensors(model->meta_ctx_.get(), backend);
    } else {
        std::vector<ggml_tensor*> plain;
        for (ggml_tensor* t = ggml_get_first_tensor(model->meta_ctx_.get()); t != nullptr;
             t = ggml_get_next_tensor(model->meta_ctx_.get(), t)) {
            if (std::find(to_repack.begin(), to_repack.end(), t) == to_repack.end()) plain.push_back(t);
        }
        model->repack_buf_.reset(alloc_tensors(repack_buft, to_repack));
        if (!model->repack_buf_) {
            throw LoadError("GgufModel::load: failed to allocate the " + std::string(ggml_backend_buft_name(repack_buft)) +
                            " buffer for '" + path + "'");
        }
        size_t bytes = 0;
        for (ggml_tensor* t : to_repack) {
            model->repacked_[t] = Repacked{};
            bytes += ggml_nbytes(t);
        }
        std::fprintf(stderr, "loom: %zu weights (%.1f MB) in %s\n", to_repack.size(), bytes / 1e6,
                     ggml_backend_buft_name(repack_buft));
        buf = alloc_tensors(buft, plain);
    }
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
        // Into a CPU_REPACK buffer this IS the repack: its set_tensor re-lays the blocks as it copies.
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        if (auto rp = model->repacked_.find(t); rp != model->repacked_.end()) rp->second.file_offset = offset;
        model->symbols_[name] = t;
    }

    // Scans every KV for the bare "model.graph_topology" key (stored under "") and any
    // "model.graph_topology.<name>" keys (stored under "<name>") -- same "loop gguf_get_n_kv/
    // gguf_get_key, check a prefix" pattern as hparam_env() below. Supports both the single-topology
    // convention every model before Whisper's own multi-topology GGUF uses, and the new named-topology
    // convention (LOOM_PROCEDURAL_GENERALIZATION.md) in the SAME file.
    static const std::string kBareKey = "model.graph_topology";
    static const std::string kNamedPrefix = "model.graph_topology.";
    const int64_t n_kv = gguf_get_n_kv(model->gguf_ctx_.get());
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string key(gguf_get_key(model->gguf_ctx_.get(), i));
        std::string dest_name;
        if (key == kBareKey) {
            dest_name = "";
        } else if (key.rfind(kNamedPrefix, 0) == 0) {
            dest_name = key.substr(kNamedPrefix.size());
        } else {
            continue;
        }
        if (gguf_get_kv_type(model->gguf_ctx_.get(), i) != GGUF_TYPE_STRING) {
            throw LoadError("GgufModel::load: '" + key + "' KV in '" + path + "' is not a string");
        }
        model->topologies_[dest_name] = gguf_get_val_str(model->gguf_ctx_.get(), i);
    }
    if (model->topologies_.empty()) {
        throw LoadError("GgufModel::load: '" + path + "' is missing the required 'model.graph_topology' "
                         "(or any 'model.graph_topology.<name>') KV");
    }

    // Content-addressed weight payloads (BACKLOG.md P0.2): the writer hashes each tensor's on-disk
    // bytes and, when two declared weight names would be byte-identical, stores the payload once under
    // whichever name it saw first and records every later name as an alias in these two parallel
    // arrays instead of writing (and allocating backend storage for) a second copy. Resolving aliases
    // straight into `symbols_` -- a flat name->tensor map -- means no other code path needs to know
    // aliasing happened at all: `weight("suffix_1.module_weight")` returns the exact same ggml_tensor*
    // as `weight("prefix.module_weight")` for LFM2's tied embedding, same as if both had been written
    // out separately. Absent entirely for every model with no duplicate payloads (add_array() on the
    // Python side skips writing an empty array), so this is a no-op for the common case.
    if (model->has_kv("loom.tensor_alias.names")) {
        const std::vector<std::string> alias_names = model->kv_arr_str("loom.tensor_alias.names");
        const std::vector<std::string> alias_targets = model->kv_arr_str("loom.tensor_alias.targets");
        if (alias_names.size() != alias_targets.size()) {
            throw LoadError("GgufModel::load: '" + path + "' has mismatched 'loom.tensor_alias.names'/"
                             "'loom.tensor_alias.targets' array lengths");
        }
        for (size_t i = 0; i < alias_names.size(); ++i) {
            const auto target_it = model->symbols_.find(alias_targets[i]);
            if (target_it == model->symbols_.end()) {
                throw LoadError("GgufModel::load: '" + path + "' declares alias '" + alias_names[i] +
                                 "' -> '" + alias_targets[i] + "', but the target tensor doesn't exist");
            }
            model->symbols_[alias_names[i]] = target_it->second;
        }
    }

    return model;
}

const std::string& GgufModel::topology_json() const {
    auto it = topologies_.find("");
    if (it == topologies_.end()) {
        throw LoadError("GgufModel::topology_json: this file has no bare 'model.graph_topology' KV (it may "
                         "be a multi-topology file -- use topology_json(name) instead)");
    }
    return it->second;
}

const std::string& GgufModel::topology_json(const std::string& name) const {
    auto it = topologies_.find(name);
    if (it == topologies_.end()) {
        throw LoadError("GgufModel::topology_json: no 'model.graph_topology." + name + "' KV in this file");
    }
    return it->second;
}

bool GgufModel::has_topology(const std::string& name) const { return topologies_.find(name) != topologies_.end(); }

std::vector<std::string> GgufModel::topology_names() const {
    std::vector<std::string> names;
    for (const auto& pair : topologies_) {
        if (!pair.first.empty()) {
            names.push_back(pair.first);
        }
    }
    return names;
}

size_t GgufModel::weight_bytes() const {
    size_t total = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(meta_ctx_.get()); t != nullptr;
         t = ggml_get_next_tensor(meta_ctx_.get(), t)) {
        total += ggml_nbytes(t);
    }
    return total;
}

std::unique_ptr<GgufModel> GgufModel::load_metadata(const std::string& path) {
    auto model = std::unique_ptr<GgufModel>(new GgufModel());

    // The identical parse `load` opens with, and then it stops. `no_alloc=true` gives correctly-shaped
    // `ggml_tensor` structs with null `data`, which is all the KV accessors and `ModelContract::read`
    // ever look at -- they read the gguf context, not the tensors. What is skipped is the backend
    // buffer and the streaming loop, which is the entire cost.
    ggml_context* raw_meta_ctx = nullptr;
    gguf_init_params gguf_params{/*no_alloc=*/true, /*ctx=*/&raw_meta_ctx};
    gguf_context* raw_gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!raw_gguf_ctx) {
        throw LoadError("GgufModel::load_metadata: failed to parse GGUF file '" + path + "'");
    }
    model->gguf_ctx_.reset(raw_gguf_ctx);
    model->meta_ctx_.reset(raw_meta_ctx);
    // `symbols_` is deliberately left EMPTY and `weights_buf_` null: `has_weights()` reads the latter,
    // and the two accessors that would hand out a data-less tensor throw on it.
    return model;
}

ggml_tensor* GgufModel::weight(const std::string& name) const {
    // The metadata-only guard, and it is a guard rather than a comment because the failure it replaces
    // is a null `tensor->data` dereferenced deep inside a graph build -- a segfault with no name on it.
    if (!has_weights()) {
        throw LoadError("GgufModel::weight: '" + name + "' was requested from a model opened with "
                        "load_metadata(), which reads the KV table and no tensor data. Open it with "
                        "load() to run it.");
    }
    auto it = symbols_.find(name);
    if (it == symbols_.end()) {
        throw LoadError("GgufModel::weight: no such weight '" + name + "'");
    }
    return it->second;
}

bool GgufModel::has_weight(const std::string& name) const {
    if (!has_weights()) {
        throw LoadError("GgufModel::has_weight: asked of a model opened with load_metadata(), which "
                        "reads no tensor data. Its answer would be about a tensor that is not there.");
    }
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

// Every consumer of a repacked weight in `gf` that ggml's repack path cannot serve is pointed at a plain
// copy instead. What the repack serves, from repack.cpp's supports_op / forward_mul_mat: a MUL_MAT that
// reads the weight DIRECTLY as src0 (not through a view -- ggml dispatches on src0's buffer, and a view
// shares it), with an F32 activation whose rows are contiguous and which is at most 3-D. Anything else
// -- GET_ROWS, a conv's dequantize, a reshape or view, an F16 activation -- would read interleaved
// blocks as plain ones or trip an assert.
//
// Views are fixed in place: a view of a pre-allocated weight got its `data` when it was created (and
// `view_src` always names the root, however deep the chain), so it is re-pointed into the copy.
void GgufModel::route_repacked_uses(ggml_cgraph* gf) {
    if (repacked_.empty()) return;
    std::lock_guard<std::mutex> lock(repack_mutex_);
    auto fix = [&](ggml_tensor* t) {
        if (t->view_src != nullptr && repacked_.count(t->view_src)) {
            if (repacked_.at(t->view_src).plain == nullptr) {
                std::fprintf(stderr, "loom: plain copy of '%s' for a view (%s)\n", ggml_get_name(t->view_src),
                             ggml_op_desc(t));
            }
            ggml_tensor* plain = plain_copy(t->view_src);
            t->view_src = plain;
            t->data = static_cast<char*>(plain->data) + t->view_offs;
            t->buffer = plain->buffer;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor* s = t->src[j];
            if (s == nullptr || !repacked_.count(s)) continue;
            // The extra buffer type's own supports_op decides (CPU_REPACK: MUL_MAT only; CPU_KLEIDIAI:
            // MUL_MAT and GET_ROWS), plus the two things repack.cpp's forward asserts rather than
            // checks: a 3-D-at-most activation with contiguous rows. CPU_ARMV6_I8's supports_op checks
            // every layout its kernels rely on, including the pointwise CONV_2D it serves (ADR-079 P4.33),
            // so for it supports_op is the whole test: with these rules applied to it, every one of
            // Citrinet's pointwise kernels was routed to a plain copy and the int8 path never ran.
            // CPU_NEON_I8 (P4.37) checks its own layouts the same way.
            const ggml_tensor* act = t->src[1];
            const char* buft_name = s->buffer != nullptr ? ggml_backend_buft_name(ggml_backend_buffer_get_type(s->buffer)) : "";
            const bool own_checks = std::strcmp(buft_name, "CPU_ARMV6_I8") == 0 || std::strcmp(buft_name, "CPU_NEON_I8") == 0;
            const bool layout_ok = own_checks ||
                                   ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_GET_ROWS) && act != nullptr &&
                                    act->ne[3] == 1 && (t->op != GGML_OP_MUL_MAT || act->nb[0] == sizeof(float)));
            const bool served = j == 0 && layout_ok &&
                                ggml_backend_dev_supports_op(ggml_backend_get_device(repack_backend_), t);
            if (!served) {
                if (repacked_.at(s).plain == nullptr) {
                    std::fprintf(stderr, "loom: plain copy of '%s' for %s\n", ggml_get_name(s), ggml_op_desc(t));
                }
                t->src[j] = plain_copy(s);
            }
        }
    };
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) fix(ggml_graph_node(gf, i));
}

ggml_tensor* GgufModel::plain_copy(ggml_tensor* w) {
    Repacked& r = repacked_.at(w);
    if (r.plain != nullptr) return r.plain;
    ggml_context_ptr ctx(ggml_init(ggml_init_params{ggml_tensor_overhead() + 256, nullptr, /*no_alloc=*/true}));
    ggml_tensor* copy = ggml_dup_tensor(ctx.get(), w);
    ggml_set_name(copy, ggml_get_name(w));
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend_));
    if (!buf) {
        throw LoadError("GgufModel: failed to allocate a plain copy of repacked weight '" +
                        std::string(ggml_get_name(w)) + "'");
    }
    std::vector<uint8_t> bytes(ggml_nbytes(w));
    std::ifstream file(path_, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(r.file_offset));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw LoadError("GgufModel: failed to re-read weight '" + std::string(ggml_get_name(w)) + "' from '" +
                        path_ + "' for its plain copy");
    }
    ggml_backend_tensor_set(copy, bytes.data(), 0, bytes.size());
    plain_ctxs_.push_back(std::move(ctx));
    plain_bufs_.push_back(std::move(buf));
    r.plain = copy;
    return copy;
}

size_t GgufModel::plain_copy_count() const {
    size_t n = 0;
    for (const auto& kv : repacked_) n += kv.second.plain != nullptr;
    return n;
}

} // namespace loom
