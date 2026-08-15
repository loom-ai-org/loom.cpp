#pragma once

// The causal-LM decode loop, in the one place both front ends can reach it.
//
// WHY THIS IS IN THE ENGINE. It was written twice and the two copies disagree, which is not a
// hypothetical: `tools/loom_cli/main.cpp` ran exactly `n_predict` steps with **no end-of-sequence stop
// at all**, took `vec[0]` when a driver returned a list, and clamped ids to `< 65536`; loom-py's
// `generate_ids` stopped on the file's own `tokenizer.ggml.eos_token_id`, took `vec[-1]`, and stripped
// the stop token before returning. Same model, same driver, two different transcripts depending on
// which host you asked. Nothing in that difference was a decision -- it is what happens when a per-TASK
// loop is left to hosts, and it is the same defect `loom::audio::transcribe` was created to remove one
// modality over (docs/HIGH-LEVEL-API.md §1).
//
// TWO DRIVER SHAPES, told apart by what the driver RETURNS rather than by knowing the model. A driver
// whose cross-step state is entirely the KV cache generates internally and hands back the whole
// sequence; one whose state is not -- LFM2's ten ShortConv blocks -- exports a single forward pass and
// returns ONE next token, leaving the loop to the host. A list means the first, a number means the
// second. Both hosts already branched this way; only the details differed.
//
// It is also the reuse point for the speech models with a causal backbone. Their public door is
// `transcribe`, not `generate` -- a Qwen3-ASR prompt requires audio, so a bare text `generate` on one
// would be a method that cannot be called -- but the decode loop underneath is this one, and calling it
// is how that stays true (docs/HIGH-LEVEL-API.md §4).

#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace text {

struct GenerateOptions {
    uint32_t max_new_tokens = 64;
    // Stop when this id is generated; negative disables the check. Defaults to the file's own
    // `tokenizer.ggml.eos_token_id` in `generate` when left at this sentinel, so the loop stops where
    // the checkpoint says it should rather than where a host's default happened to land.
    int32_t eos_token = kEosFromFile;
    // Drop the stop token from the returned ids. It is a control token rather than text, and a caller
    // detokenizing the result would otherwise get its literal spelling on the end.
    bool strip_eos = true;
    // Passed through to the driver verbatim, for a model whose `infer` takes more than tokens.
    std::unordered_map<std::string, LoomLuaBridge::Value> extra_inputs;

    static constexpr int32_t kEosFromFile = -2;
};

// Runs the model's driver until it stops or `max_new_tokens` are produced. Returns the GENERATED ids
// only, never the prompt -- both driver shapes are normalised to that here, since a caller who has the
// prompt already does not want it back and cannot tell which shape ran.
std::vector<int32_t> generate(LoomLuaBridge& bridge, const GgufModel& model,
                              const std::vector<int32_t>& prompt_tokens,
                              const GenerateOptions& options = {});

} // namespace text
} // namespace loom
