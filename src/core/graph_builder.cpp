#include "loom/core/graph_builder.h"
#include "loom/core/kv_cache.h"
#include "loom/core/output_store.h"
#include "loom/core/profile.h"
#include "loom/loom_errors.h"
#include "loom/ops/primitive_registry.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace loom {
namespace {

// **An F32 x F32 product is computed in F32 on every backend.** ggml's default precision for MUL_MAT
// lets a GPU backend stage its tiles in half precision: Vulkan's matmul_f32_f32 keeps both operands in
// float16_t shared memory on any fp16-capable device (and its coopmat paths convert them to f16
// outright), and Metal's kernel_mul_mm_f32_f32 loads both into `half` threadgroup tiles. Any |value| >
// 65504 becomes inf there, and inf * 0 is NaN. CPU and CUDA (cuBLAS SGEMM) never do this, so a model can
// be exact on both and still produce nothing on Vulkan or Metal: SenseVoice's kaldi power spectrum peaks
// near 1e10, its mel projection came back 99.5% NaN, and the transcript was ''. GGML_PREC_F32 is ggml's
// own way of asking for full precision; cmake/patches/ggml-0023 and ggml-0024 make those two backends
// honour it for F32 x F32 (stock ggml ignores it there). It is set with ggml_prec_set_acc, which since
// ggml v0.26 (llama/26675) is what ggml_mul_mat_set_prec became: the same op_params[0], so the patches
// read it unchanged. Tagged here, on the finished graph, rather than at each ggml_mul_mat call site, so
// that a new primitive cannot forget it. A non-F32 `a` (F16, a quantized weight) is left alone: that is
// the precision the model was exported at, and the CPU converts `b` to the weight's vec_dot_type on that
// path too.
//
// **An F32 x F32 CONV_2D gets the same tag**, for the same reason in a different shape: Vulkan's conv2d
// coopmat2 and KHR-coopmat variants stage both operands in float16_t AND accumulate in float16_t, so an
// NVIDIA GPU computes every F32 convolution at half precision (the plain shader, which a Vega 3 runs, is
// float). F5-TTS's reference STFT is a 1024-tap DFT-basis CONV_2D followed by log(clamp(x, 1e-5)): the
// half-precision floor moved the mean log-mel from -2.23 to -1.50 and the generated speech came out
// twice as loud and clipped (peak 1.21 against the CPU's 0.76). cmake/patches/ggml-0028 makes Vulkan
// honour the tag with the scalar shader; the CPU, CUDA and Metal compute conv_2d in F32 already.
void request_f32_precision(ggml_cgraph* gf) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor* node = ggml_graph_node(gf, i);
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_CONV_2D) &&
            node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) {
            ggml_prec_set_acc(node, GGML_PREC_F32);
        }
    }
}

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

// The one dimension name that means "how much of the KV cache this graph reads". A declared input whose
// LEADING dim is spelled this way is a mask spanning the cache, and is the only kind of input the
// bucket padding is visible to (BACKLOG.md P4.0.15). Both spellings, since SymbolEnv's '$' sigil is
// optional and the bespoke converters write it while the MIL compiler does not.
bool is_n_kv_dim(const std::string& dim) { return dim == "n_kv" || dim == "$n_kv"; }

// The KV length this call actually asks for, before any bucketing: the caller's own "n_kv" if it bound
// one, else the sum a cached topology's two axes imply, else 0 for a topology that has no such axis at
// all. Split out from effective_n_kv() because the difference between the two IS the padding, and both
// halves are needed at once.
int64_t requested_n_kv(const DynamicAxes& axes) {
    if (axes.count("n_kv")) {
        return static_cast<int64_t>(std::llround(axes.at("n_kv")));
    }
    if (axes.count("n_tokens") && axes.count("n_past")) {
        return static_cast<int64_t>(std::llround(axes.at("n_tokens") + axes.at("n_past")));
    }
    return 0;
}

// An upper bound on `n_nodes + n_leafs` for `gf`, which is what ggml_backend_sched_alloc_graph asserts
// its hash set covers -- and the one number a scheduler has to be sized by.
//
// It is counted rather than estimated because the alternatives are a crash or a waste. `struct
// ggml_cgraph` is opaque in ggml's public headers, so `n_leafs` cannot simply be read; the only bound
// available without counting is the graph's own declared capacity, and since estimate_graph_size() is a
// deliberately generous 8x that would size the scheduler's fixed overhead (capacity *
// GGML_SCHED_MAX_SPLIT_INPUTS * 2 ggml_tensor structs of context buffer) at hundreds of megabytes for a
// model whose graph needs tens -- on an engine whose target is edge devices.
//
// Every leaf ggml can add is some node's `src` or `view_src` (that is how ggml_visit_parents reaches
// one), so counting the DISTINCT tensors reachable through those two edges bounds the leaf count from
// above without having to reproduce ggml's own node/leaf classification -- which would be the fragile
// part, since it turns on op codes and tensor flags this file has no business knowing.
size_t scheduler_capacity_for(ggml_cgraph* gf) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    std::unordered_set<const ggml_tensor*> reachable;
    reachable.reserve(static_cast<size_t>(n_nodes) * 2);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] != nullptr) reachable.insert(node->src[s]);
        }
        if (node->view_src != nullptr) reachable.insert(node->view_src);
    }
    return static_cast<size_t>(n_nodes) + reachable.size();
}

} // namespace

GraphBuilder::GraphBuilder(const GraphTopology& topo, GgufModel& model, Backends backends,
                            KvCache* kv_cache, size_t compute_meta_bytes, ConvStateCache* conv_state)
    : topo_(topo), model_(model), backends_(backends), backend_(backends.primary),
      kv_cache_(kv_cache), conv_state_(conv_state), compute_meta_bytes_(compute_meta_bytes) {
    if (backend_ == nullptr) {
        throw Error("GraphBuilder: no primary backend -- construct a loom::Device, or pass a "
                    "ggml_backend_t directly");
    }
    // Asked once, here, rather than per build: uses_kv_cache() walks every node of the topology, and
    // for a flattened LM that is thousands of them. A cache the caller did not supply counts as "not
    // bucketing" -- op_attention will raise on such a topology anyway, and raising there says more.
    buckets_kv_ = kv_cache_ != nullptr && topo_.uses_kv_cache();
    // Dropping n_past from the retained graph's key is sound only if nothing in the topology can still
    // read it. The cache write no longer does; a shape expression or an attr string might, and this is
    // what makes that a checked claim rather than an assumption. A topology that does mention it simply
    // keeps rebuilding per step, exactly as everything did before this item.
    key_ignores_n_past_ = buckets_kv_ && !topo_.mentions_symbol("n_past");
}

uint32_t GraphBuilder::ring_size() const {
    return kv_cache_ != nullptr && kv_cache_->ring() ? kv_cache_->kv_size() : 0;
}

int64_t GraphBuilder::real_n_kv(const DynamicAxes& axes) const {
    const int64_t n_kv = requested_n_kv(axes);
    if (kv_cache_ == nullptr || !kv_cache_->ring() || n_kv == 0) return n_kv;
    // A RING holds the last `kv_size` positions (ADR-066), so a step past its capacity reads all of it.
    // A call of several tokens that WRAPS cannot be served: its later rows would overwrite cells its
    // earlier queries still attend to, since every row is written before any is read.
    const auto capacity = static_cast<int64_t>(kv_cache_->kv_size());
    if (n_kv > capacity && axes.count("n_tokens") && axes.at("n_tokens") > 1.0) {
        throw SchemaError("GraphBuilder::build: " + std::to_string(std::llround(axes.at("n_tokens"))) +
                           " tokens at n_past " + std::to_string(std::llround(axes.at("n_past"))) +
                           " would wrap this model's " + std::to_string(capacity) +
                           "-cell ring KV cache; past its capacity a ring takes one token per call");
    }
    return std::min(n_kv, capacity);
}

int64_t GraphBuilder::effective_n_kv(const DynamicAxes& axes) const {
    const int64_t n_kv = real_n_kv(axes);
    if (!buckets_kv_ || n_kv == 0) return n_kv;

    const auto capacity = static_cast<int64_t>(kv_cache_->kv_size());
    if (n_kv > capacity) {
        throw SchemaError("GraphBuilder::build: this step needs " + std::to_string(n_kv) +
                           " KV cells but the cache holds " + std::to_string(capacity) +
                           " -- raise the model's declared 'loom.kv_cache_size' (or the host's n_ctx_max)");
    }
    const auto bucket = static_cast<int64_t>(kKvBucket);
    // Capped at the capacity rather than allowed past it: the read view below spans [0, n_kv), so a
    // bucket boundary beyond the last allocated cell would read off the end of the store. Capping makes
    // the final, ragged stretch of a full cache a single bucket, which is if anything the better shape.
    return std::min((n_kv + bucket - 1) / bucket * bucket, capacity);
}

// A retained graph ends in one `cpy` per declared output, whose DESTINATION is the store's own
// tensor. `OutputStore::reshape` keeps those tensors only while the geometry it is handed is the same,
// and the recurrent and expansion bindings hand it a different one -- a sequence, not a build's
// output -- so a module whose graph is cached can have its slots replaced between two builds. Serving
// the cached graph then copies into freed memory.
//
// The store's own epoch answers it and the slot POINTERS do not -- see `cached_store_epoch_` for the
// bug that taught us the difference.
bool GraphBuilder::store_slots_unmoved(const OutputStore* out_store) const {
    return cached_store_epoch_ == out_store->layout_epoch();
}

const GraphBuilder::BuildResult& GraphBuilder::build(const DynamicAxes& axes, OutputStore* out_store) {
    // First, before anything can write an input: the previous call's aliases end here (ADR-073).
    restore_aliased_inputs();
    // Checked before anything else, including the cache lookup: a cached topology's cell indices come
    // from n_past and are rewritten on a REUSE too, so an axes map without it has no answer to give at
    // either end. (A caller could otherwise reach a retained decode graph by binding n_kv directly.)
    if (buckets_kv_ && !axes.count("n_past")) {
        throw SchemaError("GraphBuilder::build: this topology has a cached ATTENTION but the axes bind "
                           "no 'n_past', so there is no cell for its K/V to be written to");
    }
    const int64_t n_kv_eff = effective_n_kv(axes);

    // The axes reduced to what the graph's STRUCTURE depends on (BACKLOG.md P4.0.15). For a bucketed
    // cached topology that means: n_past leaves (it reaches the graph only as cell-index DATA now), and
    // n_kv is the padded length rather than the caller's own. Consecutive decode steps therefore key to
    // the same entry, which is the whole point of the item.
    DynamicAxes key = axes;
    if (key_ignores_n_past_) {
        key.erase("n_past");
        key["n_kv"] = static_cast<double>(n_kv_eff);
    }

    // BACKLOG.md P4.0.13. The retained graph is served back verbatim: its gallocr buffer was never
    // freed, so every tensor in it still points where ggml_gallocr_alloc_graph put it, and its declared
    // inputs still hold whatever was last written into them. Exactly one graph is retained -- see the
    // header for why an LRU keyed by shape would not be safe against a reshaped OutputStore.
    //
    // A reuse is also a CALL for the shrink policy (ADR-072), and its need is the retained graph's own.
    // When that call completes a window of under-use, the hit is turned into a rebuild: the retained
    // graph's tensors point into the oversized buffer, so the only way onto a smaller one is a fresh
    // allocation, and a rebuild is how this builder makes one. That costs one graph build once per
    // regime change, which is what the old shrink-on-the-next-build path cost too.
    bool shrink_due = false;
    if (has_cached_ && cached_store_ == out_store && cached_key_ == key &&
        (out_store == nullptr || store_slots_unmoved(out_store)) &&
        !(shrink_policy_active() && (shrink_due = note_call_need(cached_need_)))) {
        // ...with the two things that describe the STEP rather than the graph, and so must move even
        // when the graph does not. The cell indices are precisely what `n_past` turned into -- a value
        // the host rewrites between steps -- and `n_kv_real` is how much of the bucket this particular
        // step's mask actually covers. Serving a previous step's `n_kv_real` back is a real bug, not a
        // stale statistic: it is what a caller sizes its mask placement by.
        if (kv_cells_ != nullptr) {
            KvCache::fill_cell_index(kv_cells_, static_cast<uint32_t>(std::llround(axes.at("n_past"))),
                                     ring_size());
        }
        cached_.n_kv_real = real_n_kv(axes);
        ++reuses_;
        return cached_;
    }
    // Drop the retained graph BEFORE anything below can throw, so a failed build leaves the builder
    // with no graph rather than one whose context has already been replaced underneath it.
    drop_retained_graph();

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
    // derive it so a caller binding "n_tokens"/"n_past" doesn't also have to compute their sum. Since
    // P4.0.15 it is also the BUCKETED length for a cached topology, and the same value reaches the mask
    // input's declared shape and op_attention's read view because both resolve it from right here.
    if (n_kv_eff > 0) {
        env.set("n_kv", static_cast<double>(n_kv_eff));
    }

    SymbolTable symtab = model_.weights();

    BuildResult result;
    // What the caller asked for, against what the graph was built at -- the gap between them is the
    // padding a mask has to be placed into. Recorded on the result rather than on the builder because
    // it describes THIS build, and a caller holds the result.
    result.n_kv_real = real_n_kv(axes);

    // The declared inputs live in their OWN context and backend buffer, never in the gallocr pool
    // (BACKLOG.md P4.0.13 -- same seam as KvCache/ConvStateCache/OutputStore). gallocr skips any tensor
    // whose data is already set, so it can neither place an intermediate on top of an input nor move an
    // input between builds; that is what makes handing the same graph back on the next call safe rather
    // than a silent-corruption hazard (tests/test_graph_reuse_safety.cpp documents the raw-ggml
    // behaviour this sidesteps). They are still ggml_set_input()-flagged: the flag states what the
    // tensor IS, and nothing about it depends on who allocated it.
    //
    // The KV cell-index tensor shares that buffer (BACKLOG.md P4.0.15) and is the reason "outside the
    // gallocr pool" is now load-bearing for CORRECTNESS and not only for reuse safety: it is the one
    // tensor whose contents change while the graph does not, and a pool the allocator is free to move
    // or alias could not offer that.
    inputs_ctx_.reset(ggml_init(ggml_init_params{
        (topo_.inputs.size() + 1) * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true}));
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
        // A property of the TOPOLOGY and of whether this builder buckets at all -- deliberately not of
        // how much padding this particular build happens to need, because the graph outlives the build
        // that made it and the answer must stay true for every step served from it. Whether there is
        // anything to pad on a given call is then a question about two widths, asked at the write.
        // Empty for an unbucketed builder, so a caller that never asks behaves exactly as before.
        if (buckets_kv_ && !spec.shape.empty() && is_n_kv_dim(spec.shape.front())) {
            result.kv_padded_inputs.push_back(spec.name);
        }
    }
    if (buckets_kv_) {
        // `n_past` is guaranteed present -- checked at the top of build(), before the cache lookup.
        kv_cells_ = KvCache::new_cell_index(
            inputs_ctx_.get(), static_cast<uint32_t>(std::llround(env.get("n_tokens"))));
    }
    if (!topo_.inputs.empty() || kv_cells_ != nullptr) {
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
        if (kv_cells_ != nullptr) {
            if (kv_cells_->data == nullptr && ggml_nbytes(kv_cells_) > 0) {
                throw Error("GraphBuilder::build: failed to allocate the backend buffer for the KV "
                            "cell-index tensor");
            }
            KvCache::fill_cell_index(kv_cells_, static_cast<uint32_t>(std::llround(axes.at("n_past"))),
                                     ring_size());
        }
    }

    std::vector<ggml_tensor*> side_effect_roots;
    PrimitiveContext pc{ctx.get(), env, kv_cache_, conv_state_, kv_cells_, &side_effect_roots, backends_};
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
    request_f32_precision(gf);
    result.graph = gf;

    if (backends_.hybrid()) {
        allocate_scheduled(gf);
    } else {
        // The shrink policy (ADR-072), on the rebuild path. A rebuild into a live buffer is the one call
        // whose need is not already known, so it is planned on the probe first -- BEFORE the live
        // allocation, because once this graph's tensors point into the old buffer, dropping that buffer
        // would leave them dangling. A builder with no live buffer (the first build, or the one after a
        // shrink or a release) skips the probe: a fresh gallocr is sized to exactly this graph, so its
        // size after the allocation IS the need.
        bool probed = false;
        size_t need = 0;
        if (!shrink_due && shrink_policy_active() && galloc_ &&
            ggml_gallocr_get_buffer_size(galloc_.get(), 0) > 0) {
            need = measure_need(gf);
            probed = true;
            shrink_due = note_call_need(need);
        }
        if (shrink_due) {
            galloc_.reset();
            underused_calls_ = 0;
            ++shrinks_;
        }
        if (!galloc_) {
            galloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        }
        if (!ggml_gallocr_alloc_graph(galloc_.get(), gf)) {
            throw Error("GraphBuilder::build: ggml_gallocr_alloc_graph failed");
        }
        cached_need_ = probed ? need : ggml_gallocr_get_buffer_size(galloc_.get(), 0);
    }

    result.ctx = std::move(ctx);

    cached_ = std::move(result);
    cached_key_ = std::move(key);
    cached_store_ = out_store;
    cached_store_epoch_ = out_store == nullptr ? 0 : out_store->layout_epoch();
    has_cached_ = true;
    ++builds_;
    return cached_;
}

// **gallocr grows and never shrinks**, by design: `ggml_gallocr_reserve_n_impl` reallocates a chunk only
// when `new_chunk_size > cur_chunk_size` (ggml-alloc.c). That is the right default for a caller who
// reserves a worst case, and the wrong one for the shape this engine actually runs -- a prefill followed
// by a decode loop. Measured on gemma-3-270m-it at a 512-token prefill: the compute buffer reaches
// 513.2 MiB, a decode step needs 1.0 MiB, and before P4.0.16 the builder held all 513.2 MiB for the
// entire generation. So the builder gives the buffer back itself -- but only after a WINDOW of calls has
// shown it is not needed (ADR-072).
//
// **Why a window and not the first small call.** P4.0.16 shrank on the first call needing less than
// half, armed by a preceding growth of more than 2x. That is exactly right for a prefill->decode
// transition, which happens once, and exactly wrong for a TTS module serving sentences of mixed length:
// short -> long grows past 2x and arms, long -> short frees the buffer, the next long faults it all back
// in. On VITS (en-GB miro) alternating a 50- and a 198-token sentence, every long call took 38 740
// minor faults against 1 372 for the same call repeated -- the vocoder's 111 MB compute buffer, at
// 1.6 us a page on a Ryzen 3 3250U, ~46-60 ms of a ~0.95 s synthesis, and the short call paid its
// own share. Requiring kShrinkWindow under-used calls IN A ROW means an alternating workload never
// shrinks (its window always contains a long call), a regime change still shrinks once, and a long
// burst followed by short traffic gives the memory back after kShrinkWindow calls rather than never.
//
// **Every call counts, reuses included.** A decode loop reuses its graph for up to kKvBucket steps, so
// counting only rebuilds would hold a prefill's buffer for up to kShrinkWindow * kKvBucket tokens.
// Counting calls holds it for kShrinkWindow.
//
// **What it costs: one planning pass per REBUILD into a live buffer, never per reuse.** P4.0.16 avoided
// that pass on every build because, before P4.0.15, a decode step WAS a rebuild, and arming the probe
// on any growth cost +1.3 ms/step on gemma-3-270m-it. Since P4.0.15 a decode loop rebuilds once per
// kKvBucket steps, a reuse knows its need from the build that made it, and a fresh allocator needs no
// probe at all -- so the pass lands on the calls that already pay a whole graph build. On VITS's
// modules that build is 1-2.5 ms (Ryzen 3 3250U), and planning is the cheaper part of it.
//
// The size is measured on a SEPARATE gallocr, never on the live one: `ggml_gallocr_reserve_n_size` runs
// the real planner with `no_alloc=true`, which frees the live buffers in the *growing* case -- exactly
// the case where they are about to be needed. The probe never owns a buffer, so it has none to lose.
//
// **The factor is the one P4.0.16 chose, for the same reason.** Two is the smallest number that
// separates "a different regime is running now" from "n_kv grew by one bucket"; a call needing more
// than half the buffer is a call the buffer is FOR, and resets the window.
bool GraphBuilder::shrink_policy_active() const {
    return !backends_.hybrid() && !reserved_ && !held_;
}

bool GraphBuilder::note_call_need(size_t need) {
    const size_t current = galloc_ ? ggml_gallocr_get_buffer_size(galloc_.get(), 0) : 0;
    if (current > 0 && need * kShrinkFactor <= current) {
        ++underused_calls_;
    } else {
        underused_calls_ = 0;
    }
    return underused_calls_ >= kShrinkWindow;
}

size_t GraphBuilder::measure_need(ggml_cgraph* gf) {
    if (!probe_) {
        probe_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    }
    size_t need = 0;
    ggml_gallocr_reserve_n_size(probe_.get(), gf, nullptr, nullptr, &need);
    return need;
}

void GraphBuilder::drop_retained_graph() {
    // Order matters: the compute graph reads the input tensors, so it goes first.
    has_cached_ = false;
    cached_ = BuildResult{};
    inputs_buf_.reset();
    inputs_ctx_.reset();
    kv_cells_ = nullptr; // lives in inputs_ctx_, so it dies with it
    // The inputs these name died with inputs_ctx_; nothing is left to restore or to be stale.
    aliases_.clear();
    stale_inputs_.clear();
    input_views_.clear();
    input_views_scanned_ = false;
}

// ADR-073. Every condition below is one under which reading `src` in place could differ from reading a
// copy of it, so a `false` here is never an error -- the caller copies, exactly as before.
//
// * **CPU-only.** A hybrid builder's scheduler decides which backend each tensor lives on, and plans the
//   copies between them, from `tensor->buffer` when it allocates the graph. Repointing a buffer after
//   that is a question about the scheduler's internals this does not need to answer to be worth having:
//   on a device the copy it replaces is device-to-device.
// * **Same buffer type.** The input's own storage and the source must be the same kind of memory, so
//   that `data` means the same thing to every op that reads it.
// * **Contiguous, same type, and long enough.** The copy this replaces is a byte copy of a contiguous
//   range, and the input's own `nb` describes how the graph will read it.
//
// **And every view of the input moves with it.** ggml fixes a view's `data` when the graph is allocated,
// as its base's `data` plus `view_offs` (`ggml_backend_view_init`), and does not look at the base again.
// Repointing only the input left Whisper's decoder reading its K/V heads, through a reshape, out of the
// input's own storage -- which the alias had stopped filling -- and it transcribed jfk as "[ [ [". A
// consumer that read its input directly, which is all the first test did, could not see it.
// `ggml_new_tensor_impl` flattens a view of a view to the ultimate base, so `view_src == input` finds
// every one of them.
//
// The graph never writes a declared input -- nothing in this engine emits an in-place op, and a
// retained output is written only by its own module's final `cpy` -- so an alias cannot change the
// source. test_input_alias.cpp checks that by reading the source back.
bool GraphBuilder::alias_input(ggml_tensor* input, ggml_tensor* src, size_t offset) {
    if (backends_.hybrid() || input == nullptr || src == nullptr) return false;
    if (input->buffer == nullptr || src->buffer == nullptr || src->data == nullptr) return false;
    if (ggml_backend_buffer_get_type(input->buffer) != ggml_backend_buffer_get_type(src->buffer)) return false;
    if (input->type != src->type || !ggml_is_contiguous(input) || !ggml_is_contiguous(src)) return false;
    if (offset + ggml_nbytes(input) > ggml_nbytes(src)) return false;

    // An input aliased twice in one call keeps its FIRST saved storage, which is its own.
    const bool already = std::any_of(aliases_.begin(), aliases_.end(),
                                     [&](const InputAlias& a) { return a.input == input; });
    if (!already) aliases_.push_back({input, input->buffer, input->data});
    input->buffer = src->buffer;
    input->data = static_cast<char*>(src->data) + offset;
    repoint_views(input);
    return true;
}

void GraphBuilder::repoint_views(ggml_tensor* input) {
    if (!input_views_scanned_ && cached_.graph != nullptr) {
        // Nodes only: a view carries a view op, never GGML_OP_NONE, so it is never a leaf.
        std::unordered_set<const ggml_tensor*> inputs;
        for (const auto& [name, t] : cached_.input_tensors) inputs.insert(t);
        for (int i = 0; i < ggml_graph_n_nodes(cached_.graph); ++i) {
            ggml_tensor* t = ggml_graph_node(cached_.graph, i);
            if (t->view_src != nullptr && inputs.count(t->view_src)) input_views_[t->view_src].push_back(t);
        }
        input_views_scanned_ = true;
    }
    const auto it = input_views_.find(input);
    if (it == input_views_.end()) return;
    for (ggml_tensor* view : it->second) {
        view->buffer = input->buffer;
        view->data = static_cast<char*>(input->data) + view->view_offs;
    }
}

void GraphBuilder::mark_input_written(ggml_tensor* input) {
    stale_inputs_.erase(std::remove(stale_inputs_.begin(), stale_inputs_.end(), input), stale_inputs_.end());
}

void GraphBuilder::restore_aliased_inputs() {
    for (const InputAlias& a : aliases_) {
        a.input->buffer = a.buffer;
        a.input->data = a.data;
        repoint_views(a.input);
        if (std::find(stale_inputs_.begin(), stale_inputs_.end(), a.input) == stale_inputs_.end()) {
            stale_inputs_.push_back(a.input);
        }
    }
    aliases_.clear();
}

// Releasing a hold also forgets the retained graph (not the buffer). A held builder skips the probe, so
// the need it recorded for that graph is the whole buffer rather than what the graph uses -- and every
// reuse of it would then reset the window instead of advancing it, so the shrink would never come. The
// next call rebuilds into the same buffer and is measured like any other. One rebuild per toggle.
void GraphBuilder::hold_compute_buffer(bool hold) {
    if (held_ && !hold) drop_retained_graph();
    held_ = hold;
    underused_calls_ = 0;
}

// Everything the retained graph owns goes, in the order its pointers depend on each other: the graph
// (whose tensors point into both buffers) first, then the compute buffer. On a hybrid builder that buffer
// belongs to the scheduler, so the scheduler goes -- allocate_scheduled() recreates one on the next build
// exactly as it does for the first. The probe owns no buffer and is kept.
void GraphBuilder::release_compute_buffer() {
    drop_retained_graph();
    galloc_.reset();
    sched_.reset();
    sched_capacity_ = 0;
    cached_need_ = 0;
    underused_calls_ = 0;
}

// The hybrid counterpart of the gallocr block in build(), and deliberately a much shorter one: a
// scheduler does its own liveness analysis, its own per-backend buffers and its own growth policy, so
// the only decisions left here are how big to make it and when to let go of the previous allocation.
//
// **Sized from the built graph, not from estimate_graph_size().** ggml_backend_sched_alloc_graph asserts
// its hash set covers `n_nodes + n_leafs`, and the leafs are the WEIGHTS -- for a flattened LM, thousands
// of them, a count the 8x-per-topology-node estimate says nothing useful about in either direction.
// Sizing it from the real graph also keeps the scheduler's own fixed overhead honest: it allocates
// `capacity * GGML_SCHED_MAX_SPLIT_INPUTS * 2` ggml_tensor structs of context buffer, which at the
// estimate's bound would be hundreds of megabytes for a model whose graph needs tens.
//
// **Recreated, never shrunk.** A scheduler that is big enough stays; one that is not is replaced
// wholesale. There is no ggml_backend_sched equivalent of the gallocr shrink and none is faked here --
// see the shrink policy's comment above for what the CPU path's policy is actually buying, and
// note that none of that reasoning transfers: the scheduler's buffers are per backend and its planner
// already reuses them across allocations of different sizes.
void GraphBuilder::allocate_scheduled(ggml_cgraph* gf) {
    const size_t needed = scheduler_capacity_for(gf);
    if (!sched_ || sched_capacity_ < needed) {
        // A quarter of headroom so that the ordinary creep of a decode loop's graph (one KV bucket at a
        // time) does not rebuild the scheduler every few dozen steps, which would also throw away the
        // split plan the retained graph exists to keep.
        const size_t capacity = std::max<size_t>(needed + needed / 4, GGML_DEFAULT_GRAPH_SIZE);
        // The primary, then any host-memory assists, then the CPU. The CPU MUST be last:
        // ggml_backend_sched_new asserts it, because its split planner treats the final backend as the
        // one that can run anything. Backends::schedule_order() is what guarantees that ordering, and
        // it is the only place that knows it.
        // Non-const because ggml_backend_sched_new takes `ggml_backend_t *`, not a pointer to const.
        std::vector<ggml_backend_t> order = backends_.schedule_order();
        sched_.reset(ggml_backend_sched_new(order.data(), /*bufts=*/nullptr,
                                            /*n_backends=*/static_cast<int>(order.size()), capacity,
                                            /*parallel=*/false, /*op_offload=*/true));
        if (!sched_) {
            throw Error("GraphBuilder::build: ggml_backend_sched_new failed");
        }
        sched_capacity_ = capacity;
    } else {
        // Releases the PREVIOUS graph's allocation and split plan. Required before another
        // alloc_graph -- which asserts it is not already allocated -- and safe here for exactly the
        // reason the retained graph was already dropped above: nothing is going to run that graph again.
        ggml_backend_sched_reset(sched_.get());
    }
    if (!ggml_backend_sched_alloc_graph(sched_.get(), gf)) {
        throw Error("GraphBuilder::build: ggml_backend_sched_alloc_graph failed");
    }
}

// A REUSED graph is not re-allocated and not re-split: ggml_backend_sched keeps `is_alloc` set until the
// next reset, so repeated computes of the same graph skip straight to running the splits it already
// planned. That is what makes P4.0.13/P4.0.15's graph reuse worth as much on a device as on a CPU -- a
// decode loop that reuses its graph also reuses its split plan, and the per-step cost is the copies the
// splits name and nothing else.
void GraphBuilder::compute() {
    if (!has_cached_) {
        throw Error("GraphBuilder::compute: no graph has been built yet");
    }
    // The profiled variants are node-by-node and otherwise identical -- same graph, same order, same
    // buffers (loom/core/profile.h). The branch is a load of a cached bool, and it is here rather than
    // inside profile::compute so that a non-profiled run reaches the ggml call it always did.
    const bool profiled = profile::enabled();
    const ggml_status status = backends_.hybrid()
        ? (profiled ? profile::compute(sched_.get(), cached_.graph)
                    : ggml_backend_sched_graph_compute(sched_.get(), cached_.graph))
        : (profiled ? profile::compute(backend_, cached_.graph)
                    : ggml_backend_graph_compute(backend_, cached_.graph));
    if (status != GGML_STATUS_SUCCESS) {
        throw Error("GraphBuilder::compute: the graph failed to run on " +
                    std::string(ggml_backend_name(backend_)) + " (ggml_status " +
                    std::to_string(static_cast<int>(status)) + ")");
    }
}

std::vector<std::string> GraphBuilder::node_backends() const {
    std::vector<std::string> out;
    if (!sched_ || !has_cached_ || cached_.graph == nullptr) return out;
    const int n = ggml_graph_n_nodes(cached_.graph);
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched_.get(), ggml_graph_node(cached_.graph, i));
        out.emplace_back(b ? ggml_backend_name(b) : "");
    }
    return out;
}

int GraphBuilder::splits() const {
    return sched_ ? ggml_backend_sched_get_n_splits(sched_.get()) : 0;
}

size_t GraphBuilder::buffer_size() const {
    if (sched_) {
        size_t total = 0;
        // Every backend the scheduler was given, not just the pair -- an assist allocates its own
        // buffers like any other, and omitting it would under-report the graph's real footprint.
        for (ggml_backend_t b : backends_.schedule_order()) {
            total += ggml_backend_sched_get_buffer_size(sched_.get(), b);
        }
        return total;
    }
    return galloc_ ? ggml_gallocr_get_buffer_size(galloc_.get(), 0) : 0;
}

void GraphBuilder::reserve(uint32_t n_ctx_max) {
    if (n_ctx_max == 0) return;
    // Before the builds below, not after: the decode-shaped one needs less than the prefill-shaped one,
    // so an un-suppressed shrink would hand back the very buffer this call exists to hold.
    reserved_ = true;
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
