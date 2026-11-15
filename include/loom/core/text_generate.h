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
#include "loom/core/stream_state.h"

#include <cstdint>
#include <optional>
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
    // detokenizing the result would otherwise get its literal spelling on the end. Applies to ANY id in
    // the checkpoint's stop set, not only `eos_token`: an instruction-tuned turn ends on
    // `<end_of_turn>`, and stripping only `<eos>` leaves that marker's spelling in the answer.
    bool strip_eos = true;

    // The decode rule (P4.24), unset by default -- which means "use what the file declared", and a file
    // declaring nothing means GREEDY. That default is not conservatism: a sampled default would move
    // every gate baseline that compares tokens or audio against a reference, and none of those
    // movements would mean anything.
    //
    // These reach the DRIVER, as `inputs.temperature` and friends, because the decode loop for a
    // KV-cached model is the driver's own -- a sampler applied in the loop below would reach only the
    // one-token driver shape and would silently do nothing for every model that matters here.
    std::optional<float> temperature;
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    // Seeds the bridge's RNG before the run, so a sampled generation is reproducible. Unset leaves the
    // stream where it was, which is what lets two calls in one session differ.
    std::optional<uint32_t> seed;

    // Passed through to the driver verbatim, for a model whose `infer` takes more than tokens.
    std::unordered_map<std::string, LoomLuaBridge::Value> extra_inputs;

    static constexpr int32_t kEosFromFile = -2;
};

// Every id that ends generation for `model`: `tokenizer.ggml.eos_token_ids` when the file carries the
// checkpoint's full set (P4.23), and its single `tokenizer.ggml.eos_token_id` otherwise.
//
// Exists as its own function because three callers need the same answer -- `generate` for its stop and
// its strip, and a host deciding whether a returned id is text. gemma-3-270m-it declares `[1, 106]`:
// `<eos>`, which its base model emits, and `<end_of_turn>`, which every chat turn ends on.
std::vector<int32_t> eos_token_ids(const GgufModel& model);

// Runs the model's driver until it stops or `max_new_tokens` are produced. Returns the GENERATED ids
// only, never the prompt -- both driver shapes are normalised to that here, since a caller who has the
// prompt already does not want it back and cannot tell which shape ran.
std::vector<int32_t> generate(LoomLuaBridge& bridge, const GgufModel& model,
                              const std::vector<int32_t>& prompt_tokens,
                              const GenerateOptions& options = {});

// The tokens-out door (ADR-099): the same generation as `generate`, handed back as it is drawn.
//
// A driver that loops over the KV cache inside one call exports `stream_open` (prefill, first token),
// `stream_feed{max_tokens}` (up to that many more) and `stream_finish` beside `infer_with_past`, keeps
// its cursor in the stream's `StreamState`, and sets the state value `done` when its own loop would
// have stopped. A one-token driver (`infer` only) is streamed by this class's own loop, so every such
// file already streams. A looping driver without the three functions is refused; `generate` still runs
// it.
//
// Greedy, the ids equal `generate`'s with the same options; seeded, too, because a seeded stream draws
// from a generator of its own (stream_state.h). Streams on one model are independent: the KV and conv
// caches are parked or per-stream as `LoomLuaBridge::set_stream_caches` says.
class GenerationStream {
public:
    GenerationStream(LoomLuaBridge& bridge, const GgufModel& model, std::vector<int32_t> prompt_tokens,
                     GenerateOptions options = {});

    // Whether this file's driver can be streamed at all.
    static bool supported(const LoomLuaBridge& bridge);

    // Up to `max_tokens` more generated ids; empty once finished. The stop token never comes back when
    // `strip_eos` is set, as with `generate`.
    std::vector<int32_t> next(uint32_t max_tokens = 1);

    // Back to the prompt, nothing generated.
    void reset();

    bool finished() const { return finished_; }
    // Everything handed back so far.
    const std::vector<int32_t>& tokens() const { return tokens_; }
    // Whether the stream ended on a stop token that was dropped.
    bool stopped_on_eos() const { return stripped_; }
    const StreamState& state() const { return state_; }

private:
    bool driver_done() const;
    static std::vector<double> as_ids(const LoomLuaBridge::Value& v);

    LoomLuaBridge& bridge_;
    std::vector<int32_t> prompt_;
    GenerateOptions options_;
    int32_t eos_ = -1;
    std::vector<int32_t> strip_;
    bool driver_loop_ = false;

    StreamState state_;
    std::vector<int32_t> tokens_;
    std::vector<double> running_;
    bool opened_ = false;
    bool finished_ = false;
    bool stripped_ = false;
};

} // namespace text
} // namespace loom
