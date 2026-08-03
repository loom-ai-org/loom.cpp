#pragma once

// Umbrella header for loom-engine: the lean runtime surface.
//
// "Lean" is the architecture's stated goal, not a description of this file's length -- the engine
// targets edge devices and all model-specific complexity belongs in the exporter, because adding a
// family is far cheaper in a Python library than in specialized C++ (EXPORT-PREPARATION.md 1.3). The
// per-model C++ drivers that predate that decision are NOT included here; they live in
// "loom/loom_legacy.h", which carries their retirement policy. See P4.0.8/E.2 for why the split
// exists: while they were re-exported from this file, every consumer depended on them transitively
// and a grep for consumers reported none.

#include "loom/loom_errors.h"

// --- The contract a MIL-exported GGUF actually executes against. A model is a GGUF carrying its own
//     topologies and its own driver script; these headers are what runs it. ---
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"
#include "loom/core/graph_builder.h"
#include "loom/core/symbol_table.h"
#include "loom/core/kv_cache.h"
#include "loom/core/lua_bridge.h"
#include "loom/ops/primitive_registry.h"

// --- Host-side primitives the bridge exposes to a driver script. `lua_bridge.cpp` includes exactly
//     these two beyond the graph/cache pair above; both are generic tensor ops with a data-dependent
//     output length, which is the criterion lua_bridge.h records for any binding. ---
#include "loom/core/duration_aligner.h"
#include "loom/core/relative_position.h"

// --- Task-level helpers a host uses around a driver, not inside one: tokenization on the way in,
//     decoding on the way out. These are per-TASK, not per-model -- one CTC decoder covers every CTC
//     model -- which is why they stay in C++ (EXPORT-PREPARATION.md 1.3's "tokenizers and vocoders"
//     exception, which measurement narrowed to just tokenizers). ---
#include "loom/core/vocab.h"
#include "loom/core/bpe_vocab.h"
#include "loom/core/wordpiece_vocab.h"
#include "loom/core/byte_vocab.h"
#include "loom/core/unicode.h"
#include "loom/core/supertonic_text_vectorizer.h"
#include "loom/core/ctc_decode.h"
#include "loom/core/tdt_decoder.h"
#include "loom/core/generation.h"

// --- Standalone C++ components from the pre-MIL era. Each is model-agnostic and keeps its own unit
//     test, so none is legacy in the sense loom_legacy.h means; but each also has a Lua counterpart
//     that the MIL path uses instead (bilstm_stepper -> loom.run_recurrent + RecurrentPhase;
//     cfm_euler_sampler/ode_stepper -> the FlowMatchingSampler component; style_diffusion_sampler ->
//     StyleTTS2's ADPM2 driver fragment). They are here rather than in loom_legacy.h because their
//     remaining consumers are tests of the components themselves, not of any driver. ---
#include "loom/core/bilstm_stepper.h"
#include "loom/core/cfm_euler_sampler.h"
#include "loom/core/ode_stepper.h"
#include "loom/core/style_diffusion_sampler.h"
