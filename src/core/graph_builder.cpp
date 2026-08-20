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

// The one factor the compute-buffer shrink is tuned by, used at BOTH ends so they cannot disagree: a
// growth arms the check only if the buffer more than doubled, and the check gives memory back only if
// less than half of it is needed. Two is not a tuned optimum -- it is the smallest number that
// separates "a different regime is running now" from "n_kv grew by one token", which is the only
// distinction this has to make. See shrink_allocator_if_oversized.
constexpr size_t kShrinkArmingGrowth = 2;

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

int64_t GraphBuilder::effective_n_kv(const DynamicAxes& axes) const {
    const int64_t n_kv = requested_n_kv(axes);
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

const GraphBuilder::BuildResult& GraphBuilder::build(const DynamicAxes& axes, OutputStore* out_store) {
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
    if (has_cached_ && cached_store_ == out_store && cached_key_ == key) {
        // ...with the two things that describe the STEP rather than the graph, and so must move even
        // when the graph does not. The cell indices are precisely what `n_past` turned into -- a value
        // the host rewrites between steps -- and `n_kv_real` is how much of the bucket this particular
        // step's mask actually covers. Serving a previous step's `n_kv_real` back is a real bug, not a
        // stale statistic: it is what a caller sizes its mask placement by.
        if (kv_cells_ != nullptr) {
            KvCache::fill_cell_index(kv_cells_, static_cast<uint32_t>(std::llround(axes.at("n_past"))));
        }
        cached_.n_kv_real = requested_n_kv(axes);
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
    kv_cells_ = nullptr; // lives in inputs_ctx_, so it dies with it

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
    result.n_kv_real = requested_n_kv(axes);

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
            KvCache::fill_cell_index(kv_cells_, static_cast<uint32_t>(std::llround(axes.at("n_past"))));
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
    result.graph = gf;

    if (backends_.hybrid()) {
        allocate_scheduled(gf);
    } else {
        shrink_allocator_if_oversized(gf);
        if (!galloc_) {
            galloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        }
        const size_t buffer_before = ggml_gallocr_get_buffer_size(galloc_.get(), 0);
        if (!ggml_gallocr_alloc_graph(galloc_.get(), gf)) {
            throw Error("GraphBuilder::build: ggml_gallocr_alloc_graph failed");
        }
        // Arm the shrink for the NEXT build, and only on a growth big enough to be a change of REGIME
        // rather than the ordinary creep of a decode loop. A cached causal LM's `n_kv` still creeps --
        // one bucket every kKvBucket steps since P4.0.15, one token per step before it -- so its buffer
        // still grows a little on rebuilds that are not regime changes; arming on any growth at all made
        // the probe run on roughly every other step and cost +1.3 ms/step on gemma-3-270m-it, which is
        // the measurement that put this factor here. A prefill->decode transition is 500x, not 1.001x.
        if (ggml_gallocr_get_buffer_size(galloc_.get(), 0) > buffer_before * kShrinkArmingGrowth) {
            may_shrink_ = true;
        }
    }

    result.ctx = std::move(ctx);

    cached_ = std::move(result);
    cached_key_ = std::move(key);
    cached_store_ = out_store;
    has_cached_ = true;
    ++builds_;
    return cached_;
}

// **gallocr grows and never shrinks**, by design: `ggml_gallocr_reserve_n_impl` reallocates a chunk only
// when `new_chunk_size > cur_chunk_size` (ggml-alloc.c). That is the right default for a caller who
// reserves a worst case, and the wrong one for the shape this engine actually runs -- a prefill followed
// by a decode loop. Measured on gemma-3-270m-it at a 512-token prefill: the compute buffer reaches
// 513.2 MiB, a decode step needs 1.0 MiB, and before this the builder held all 513.2 MiB for the entire
// generation. It runs only on the REBUILD path, after the retained graph has already been released --
// which is what keeps it compatible with P4.0.15's decode-loop reuse. That reuse changed how often this
// is reached, not whether it is safe: a prefill and its following decode steps still differ in
// `n_tokens`, so the transition this exists for is still a rebuild, and the ~31 reused steps after it
// never enter here at all.
//
// The size is measured on a SCRATCH gallocr, never on the live one: `ggml_gallocr_reserve_n_size` runs
// the real planner with `no_alloc=true`, which frees the live buffers in the *growing* case -- exactly
// the case where they are about to be needed. A scratch allocator has no buffers to lose.
//
// **Armed by a preceding growth (`may_shrink_`), not run on every build.** The scratch plan is a second
// full pass over the graph on top of the one `ggml_gallocr_alloc_graph` already does, and running it per
// build measured consistently slower on gemma-3-270m-it's 1742-node graph -- paid on every decode step
// to reclaim memory exactly once. (The wall-clock deltas on this machine were within its own run-to-run
// variance of about 1 ms/step, so no figure is quoted here; the argument the design rests on is the
// counted one, not the timed one.) Since the allocator only ever grows, the buffer can only be
// oversized if some earlier build inflated it, so arming on growth makes the check run **once per
// regime change** rather than once per build -- `shrinks()`/`builds()` report 1 and 101 for a
// 100-step generation, which is the property the test asserts. Clearing the flag even when nothing
// shrank is what stops a graph that genuinely needs its whole buffer from re-probing forever.
void GraphBuilder::shrink_allocator_if_oversized(ggml_cgraph* gf) {
    if (reserved_ || !may_shrink_ || !galloc_) return;
    may_shrink_ = false;
    const size_t current = ggml_gallocr_get_buffer_size(galloc_.get(), 0);
    if (current == 0) return;

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    ggml_gallocr_ptr probe(ggml_gallocr_new(buft));
    if (!probe) return;
    size_t needed = 0;
    ggml_gallocr_reserve_n_size(probe.get(), gf, nullptr, nullptr, &needed);
    // And only give it back when there is materially something to give. Shrinking to the exact current
    // need would hand back a buffer the next step grows straight back -- the same `n_kv` creep the
    // arming factor is about, seen from the other end.
    if (needed * kShrinkArmingGrowth > current) return;

    galloc_.reset(ggml_gallocr_new(buft));
    ++shrinks_;
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
// see shrink_allocator_if_oversized's comment for what the CPU path's policy is actually buying, and
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
