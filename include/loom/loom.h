#pragma once

// Umbrella header for loom-engine: the lean runtime surface.
//
// "Lean" is the architecture's stated goal, not a description of this file's length -- the engine
// targets edge devices and all model-specific complexity belongs in the exporter, because adding a
// family is far cheaper in a Python library than in specialized C++ (EXPORT-PREPARATION.md 1.3). The
// The per-model C++ drivers that predate that decision used to live beside this file in
// "loom/loom_legacy.h", which carried their retirement policy. All six are gone -- five in P4.0.8's
// stage E, and WhisperDriver in P4.1, the commit that gave Whisper a MIL export to be replaced by --
// so that header is gone with them and this IS the surface.

#include "loom/loom_errors.h"

// --- The contract a MIL-exported GGUF actually executes against. A model is a GGUF carrying its own
//     topologies and its own driver script; these headers are what runs it. ---
// Device selection: which backend(s) a graph runs on, and the CPU fallback behind a device one. Every
// header below takes the `Backends` this one defines wherever it used to take a bare `ggml_backend_t`.
#include "loom/core/backend.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"
#include "loom/core/graph_builder.h"
#include "loom/core/symbol_table.h"
#include "loom/core/kv_cache.h"
// A hybrid's ShortConv blocks carry history the KV cache does not hold, so a model with any of them
// needs this too -- and needing it is not exotic: LFM2 has ten. It was missing here while `kv_cache.h`
// was present, which meant a host including only this umbrella could load such a model, tokenize for
// it, and fail inside the driver on the first SHORT_CONV node (BACKLOG.md P4.0.10).
#include "loom/core/conv_state_cache.h"
#include "loom/core/output_store.h"
#include "loom/core/profile.h"
#include "loom/core/lua_bridge.h"
// What a file declares about ITSELF -- the task it performs and the modality pair it maps between --
// and the ready-to-run bundle a host builds from it. Both exist so a host dispatches on what the file
// says rather than on which architecture it recognises (docs/HIGH-LEVEL-API.md).
#include "loom/core/model_contract.h"
#include "loom/core/session.h"
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
#include "loom/core/phoneme_vocab.h"
#include "loom/core/unicode.h"
#include "loom/core/supertonic_text_vectorizer.h"
#include "loom/core/chat_template.h"
#include "loom/core/ctc_decode.h"
#include "loom/core/generation.h"
// The end-to-end task doors. Per-TASK like the decoders above, and here for the same reason the
// CTC decoder is: every host needs them, and the copies hosts wrote had already drifted apart.
#include "loom/core/text_generate.h"
#include "loom/core/text_classify.h"
#include "loom/core/transcribe.h"

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
