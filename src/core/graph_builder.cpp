#include "loom/core/graph_builder.h"
#include "loom/core/output_store.h"
#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>

namespace loom {
namespace {

ggml_type parse_dtype(const std::string& dtype) {
    if (dtype == "f32") return GGML_TYPE_F32;
    if (dtype == "i32") return GGML_TYPE_I32;
    if (dtype == "f16") return GGML_TYPE_F16;
    throw SchemaError("GraphBuilder: unsupported input dtype '" + dtype + "'");
}

std::string substitute_placeholder(const std::string& s, const std::string& placeholder, const std::string& replacement) {
    std::string out = s;
    size_t pos = 0;
    while ((pos = out.find(placeholder, pos)) != std::string::npos) {
        out.replace(pos, placeholder.size(), replacement);
        pos += replacement.size();
    }
    return out;
}

void substitute_in_json(nlohmann::json& j, const std::string& placeholder, const std::string& replacement) {
    if (j.is_string()) {
        j = substitute_placeholder(j.get<std::string>(), placeholder, replacement);
    } else if (j.is_array() || j.is_object()) {
        for (auto& elem : j) substitute_in_json(elem, placeholder, replacement);
    }
}

// Instantiates one iteration of a repeat_for block: substitutes every "{index_var}" occurrence in the
// node's inputs/outputs/attrs with the concrete loop index `i` (e.g. "blk.{i}.attn_q.weight" ->
// "blk.3.attn_q.weight"), mirroring GGUF's own "blk.N.xxx" weight-naming convention.
TopologyNode instantiate_node(const TopologyNode& tmpl, const std::string& index_var, int64_t i) {
    TopologyNode node = tmpl;
    const std::string placeholder = "{" + index_var + "}";
    const std::string replacement = std::to_string(i);
    for (std::string& s : node.inputs) s = substitute_placeholder(s, placeholder, replacement);
    for (std::string& s : node.outputs) s = substitute_placeholder(s, placeholder, replacement);
    substitute_in_json(node.attrs, placeholder, replacement);
    return node;
}

void build_node(const TopologyNode& node, PrimitiveContext& pc, SymbolTable& symtab) {
    std::vector<ggml_tensor*> inputs;
    inputs.reserve(node.inputs.size());
    for (const std::string& name : node.inputs) {
        auto it = symtab.find(name);
        if (it == symtab.end()) {
            throw SchemaError("GraphBuilder: node '" + (node.name.empty() ? node.op : node.name) +
                               "' references unresolved input '" + name + "'");
        }
        if (it->second == nullptr) {
            throw SchemaError("GraphBuilder: node '" + (node.name.empty() ? node.op : node.name) +
                               "' input '" + name + "' is nullptr (missing weight from GGUF?)");
        }
        inputs.push_back(it->second);
    }

    const PrimitiveFn& fn = PrimitiveRegistry::instance().get(node.op);
    std::vector<ggml_tensor*> outputs;
    try {
        outputs = fn(pc, inputs, node.attrs);
    } catch (const Error& e) {
        std::string ins;
        for (const auto& in : node.inputs) ins += in + ",";
        throw SchemaError("node '" + (node.name.empty() ? node.op : node.name) + "' (op=" + node.op +
                           ", inputs=[" + ins + "], outputs=[" +
                           (node.outputs.empty() ? "" : node.outputs[0]) + "]): " + e.what());
    }

    if (outputs.size() != node.outputs.size()) {
        throw SchemaError("GraphBuilder: op '" + node.op + "' produced " + std::to_string(outputs.size()) +
                           " output(s) but the topology declares " + std::to_string(node.outputs.size()));
    }
    for (size_t k = 0; k < outputs.size(); ++k) {
        if (!node.name.empty() && outputs.size() == 1) {
            ggml_set_name(outputs[k], node.name.c_str());
        }
        symtab[node.outputs[k]] = outputs[k];
    }
}

// Generous upper bound on how many ggml ops the topology will expand into, used to size the
// ggml_cgraph's node-array capacity. Most topology nodes map to exactly one ggml call, but composite
// primitives (e.g. Phase 3's ATTENTION) expand into several -- the *8 factor covers that with headroom
// rather than requiring a separate counting pass through the primitive registry.
size_t estimate_graph_size(const GraphTopology& topo, const SymbolEnv& env) {
    size_t count = 0;
    for (const TopologyItem& item : topo.items) {
        if (item.is_repeat) {
            const int64_t n = static_cast<int64_t>(std::llround(env.eval(item.repeat.count_symbol)));
            count += static_cast<size_t>(std::max<int64_t>(n, 0)) * item.repeat.nodes.size();
        } else {
            count += 1;
        }
    }
    return std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, count * 8 + 64);
}

} // namespace

GraphBuilder::GraphBuilder(const GraphTopology& topo, GgufModel& model, ggml_backend_t backend,
                            KvCache* kv_cache, size_t compute_meta_bytes, ConvStateCache* conv_state)
    : topo_(topo), model_(model), backend_(backend), kv_cache_(kv_cache), conv_state_(conv_state),
      compute_meta_bytes_(compute_meta_bytes) {}

const GraphBuilder::BuildResult& GraphBuilder::build(const DynamicAxes& axes, OutputStore* out_store) {
    // BACKLOG.md P4.0.13. The retained graph is served back verbatim: its gallocr buffer was never
    // freed, so every tensor in it still points where ggml_gallocr_alloc_graph put it, and its declared
    // inputs still hold whatever was last written into them. Exactly one graph is retained -- see the
    // header for why an LRU keyed by shape would not be safe against a reshaped OutputStore.
    if (has_cached_ && cached_store_ == out_store && cached_axes_ == axes) {
        ++reuses_;
        return cached_;
    }
    // Drop the retained graph BEFORE anything below can throw, so a failed build leaves the builder
    // with no graph rather than one whose context has already been replaced underneath it. Order
    // matters: the compute graph reads the input tensors, so it goes first.
    has_cached_ = false;
    cached_ = BuildResult{};
    inputs_buf_.reset();
    inputs_ctx_.reset();

    ggml_init_params params{compute_meta_bytes_, nullptr, /*no_alloc=*/true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        throw Error("GraphBuilder::build: ggml_init failed (compute_meta_bytes=" + std::to_string(compute_meta_bytes_) + ")");
    }

    SymbolEnv env = model_.hparam_env();
    for (const auto& [name, value] : axes) {
        env.set(name, value);
    }
    // n_kv is the one derived axis a primitive itself reads directly from SymbolEnv (see this
    // function's own header comment) rather than only ever appearing in a JSON shape string -- auto-
    // derive it so a caller binding "n_tokens"/"n_past" doesn't also have to compute their sum.
    if (!axes.count("n_kv") && axes.count("n_tokens") && axes.count("n_past")) {
        env.set("n_kv", axes.at("n_tokens") + axes.at("n_past"));
    }

    SymbolTable symtab = model_.weights();

    BuildResult result;

    // The declared inputs live in their OWN context and backend buffer, never in the gallocr pool
    // (BACKLOG.md P4.0.13 -- same seam as KvCache/ConvStateCache/OutputStore). gallocr skips any tensor
    // whose data is already set, so it can neither place an intermediate on top of an input nor move an
    // input between builds; that is what makes handing the same graph back on the next call safe rather
    // than a silent-corruption hazard (tests/test_graph_reuse_safety.cpp documents the raw-ggml
    // behaviour this sidesteps). They are still ggml_set_input()-flagged: the flag states what the
    // tensor IS, and nothing about it depends on who allocated it.
    inputs_ctx_.reset(ggml_init(ggml_init_params{
        topo_.inputs.size() * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true}));
    if (!inputs_ctx_) {
        throw Error("GraphBuilder::build: ggml_init failed for the declared-input context");
    }
    for (const TensorSpec& spec : topo_.inputs) {
        std::vector<int64_t> ne;
        ne.reserve(spec.shape.size());
        for (const std::string& dim : spec.shape) {
            ne.push_back(static_cast<int64_t>(std::llround(env.eval(dim))));
        }
        ggml_tensor* t = ggml_new_tensor(inputs_ctx_.get(), parse_dtype(spec.dtype), static_cast<int>(ne.size()), ne.data());
        ggml_set_name(t, spec.name.c_str());
        ggml_set_input(t);
        symtab[spec.name] = t;
        result.input_tensors[spec.name] = t;
    }
    if (!topo_.inputs.empty()) {
        // A null return means "there was nothing to allocate", which is not an error for a topology
        // whose declared inputs are all zero-sized at these axes -- so the check is on the tensors
        // themselves rather than on the return value. Anything still unallocated and non-empty here
        // would fall through to gallocr, which is precisely what this arrangement exists to prevent.
        inputs_buf_.reset(ggml_backend_alloc_ctx_tensors(inputs_ctx_.get(), backend_));
        for (const auto& [name, t] : result.input_tensors) {
            if (t->data == nullptr && ggml_nbytes(t) > 0) {
                throw Error("GraphBuilder::build: failed to allocate the backend buffer for declared "
                            "input '" + name + "'");
            }
        }
    }

    std::vector<ggml_tensor*> side_effect_roots;
    PrimitiveContext pc{ctx.get(), env, kv_cache_, conv_state_, &side_effect_roots};
    for (const TopologyItem& item : topo_.items) {
        if (!item.is_repeat) {
            build_node(item.node, pc, symtab);
            continue;
        }
        const int64_t count = static_cast<int64_t>(std::llround(env.eval(item.repeat.count_symbol)));
        for (int64_t i = 0; i < count; ++i) {
            for (const TopologyNode& tmpl : item.repeat.nodes) {
                TopologyNode inst = instantiate_node(tmpl, item.repeat.index_var, i);
                build_node(inst, pc, symtab);
            }
        }
    }

    result.outputs.reserve(topo_.outputs.size());
    for (const std::string& out_name : topo_.outputs) {
        auto out_it = symtab.find(out_name);
        if (out_it == symtab.end()) {
            throw SchemaError("GraphBuilder::build: declared output '" + out_name + "' was never produced by any node");
        }
        // Every declared output needs its own ggml_set_output(), not just the first: without it,
        // gallocr's liveness analysis sees an "output" tensor with no reader as soon as the last node
        // that consumes it computes, and frees its buffer for reuse by something later in the same
        // graph -- silently corrupting it (see test_primitive_registry.cpp's own hand-built
        // multi-output precedent for this exact failure mode).
        ggml_set_output(out_it->second);
        result.outputs.push_back(out_it->second);
    }
    result.output = result.outputs.front();

    ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), estimate_graph_size(topo_, env), /*grads=*/false);
    // KV-cache writes (if any) have no data-dependency edge to any declared output, so they must be
    // expanded into the graph explicitly, and *before* the outputs -- ggml_cgraph nodes execute strictly
    // in the array order they were appended in, so this ordering is what guarantees a write lands in the
    // cache before the read that depends on it being there (see PrimitiveContext::side_effects).
    for (ggml_tensor* root : side_effect_roots) {
        ggml_build_forward_expand(gf, root);
    }
    // All outputs are expanded before the single ggml_gallocr_alloc_graph call below (same "build
    // forward from every co-equal output first, allocate once" shape as the hand-built multi-output
    // test in test_primitive_registry.cpp) -- for a single-output topology this is exactly the one
    // ggml_build_forward_expand call the pre-P2 code made, so behavior/byte-output is unchanged.
    for (ggml_tensor* out : result.outputs) {
        ggml_build_forward_expand(gf, out);
    }
    // Retained outputs (BACKLOG.md P4.0.12), expanded LAST and deliberately so: unlike a cache write,
    // this cpy has a real data dependency on the output it copies, so it must come after -- and
    // ggml_build_forward_expand appends only the cpy itself, every node it reads already being in the
    // graph. The destination is the store's own persistent tensor, whose data pointer is set outside
    // this graph, so gallocr treats it as pre-allocated exactly as it does a KvCache view.
    if (out_store != nullptr) {
        const std::vector<ggml_tensor*>& slots = out_store->reshape(result.outputs);
        for (size_t i = 0; i < slots.size(); ++i) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx.get(), result.outputs[i], slots[i]));
        }
    }
    result.graph = gf;

    if (!galloc_) {
        galloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    }
    if (!ggml_gallocr_alloc_graph(galloc_.get(), gf)) {
        throw Error("GraphBuilder::build: ggml_gallocr_alloc_graph failed");
    }

    result.ctx = std::move(ctx);

    cached_ = std::move(result);
    cached_axes_ = axes;
    cached_store_ = out_store;
    has_cached_ = true;
    ++builds_;
    return cached_;
}

size_t GraphBuilder::buffer_size() const {
    return galloc_ ? ggml_gallocr_get_buffer_size(galloc_.get(), 0) : 0;
}

void GraphBuilder::reserve(uint32_t n_ctx_max) {
    if (n_ctx_max == 0) return;
    // Not dead weight on the Lua path any more (BACKLOG.md P4.0.13's own complaint): the builder that
    // reserves is now the builder that serves every later call, so the allocator it sizes here is the
    // one those calls use. The decode build below also leaves its graph retained, so a decode-shaped
    // first call after reserve() is served straight out of the cache.
    // build() always allocates via gallocr internally, so simply building the worst-case prefill and
    // decode shapes once (and discarding the results) is enough to size the allocator's buffer for
    // every smaller/equal shape a real generation loop will request afterwards. Named "n_tokens"/
    // "n_past" directly rather than taking a caller-supplied DynamicAxes: this reservation strategy
    // (worst-case prefill vs. worst-case decode) is inherently an autoregressive-LLM-shaped concept,
    // and every such topology on this roadmap keeps that exact axis name (EXPORT-ROADMAP.md R1 only
    // renames the non-token-sequence families -- Conformer-CTC/Parakeet/Kokoro -- none of which call
    // reserve()).
    build({{"n_tokens", static_cast<double>(n_ctx_max)}, {"n_past", 0.0}});
    if (n_ctx_max > 1) {
        build({{"n_tokens", 1.0}, {"n_past", static_cast<double>(n_ctx_max - 1)}});
    }
}

} // namespace loom
