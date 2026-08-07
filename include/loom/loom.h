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
#include "loom/core/output_store.h"
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
#include "loom/core/generation.h"

// --- One standalone C++ component from the pre-MIL era. It was four; P4.0.8's follow-up retired
//     `cfm_euler_sampler.h`, `ode_stepper.h` and `style_diffusion_sampler.h`, each of which had a Lua
//     counterpart the MIL path uses instead (the first two -> the `FlowMatchingSampler` component,
//     the third -> StyleTTS2's ADPM2 driver fragment) and no consumer but its own test.
//
//     `bilstm_stepper.h` did NOT follow them, and the reason is a measurement rather than a judgement:
//     unlike the other three its consumers are not tests OF it. test_e2e_kokoro_{text_encoder,
//     duration_predictor,f0n}.cpp construct a BiLstmStepper to drive the bespoke per-topology
//     checks they exist for, so deleting it deletes those checks. Its own MIL counterpart
//     (loom.run_recurrent + RecurrentPhase) has replaced it in every DRIVER; what keeps it alive is
//     the bespoke conversion path, and it retires with that in P6. ---
#include "loom/core/bilstm_stepper.h"
