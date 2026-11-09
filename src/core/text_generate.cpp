#include "loom/core/text_generate.h"

#include "loom/loom_errors.h"

#include <algorithm>
#include <string>

namespace loom {
namespace text {
namespace {

// The new token out of one driver call, whichever shape the driver has.
//
// **The last element, not the first**, and this is where the two host loops disagreed. A driver that
// returns one token returns a one-element list or a bare number, for which first and last are the same;
// a driver that hands back the whole RUNNING sequence differs, and there the first element is a prompt
// token -- `loom_cli` took it, and would have echoed the prompt back one token at a time had any driver
// had that shape. Taking the last is correct for all three and wrong for none.
int32_t new_token(const LoomLuaBridge::Value& result) {
    if (std::holds_alternative<double>(result)) {
        return static_cast<int32_t>(std::get<double>(result));
    }
    const auto& ids = std::get<std::vector<double>>(result);
    if (ids.empty()) {
        throw Error("generate: the driver returned an empty sequence, which is neither a token nor a "
                    "completed generation. Check `model.driver_source`'s own `infer` contract.");
    }
    return static_cast<int32_t>(ids.back());
}

// `eos` is what the driver's own loop is told to stop on; `strip` is every id that, ending the output,
// is a control token rather than text.
struct Stops {
    int32_t eos;
    std::vector<int32_t> strip;
};

Stops resolve_stops(const GgufModel& model, const GenerateOptions& options) {
    // The file's own stop tokens unless the caller named one, so the loop stops where the checkpoint
    // says it should rather than where a host's default happened to land. Negative means no early stop,
    // which the generated drivers document as the meaning of a negative `eos_token`.
    //
    // The whole SET, not the one the tokenizer config happened to name (P4.23). `eos` stays the first
    // of them because it is what the driver's own loop is told to stop on and what a caller may
    // override; the rest are what the driver breaks on from `extra_eos_tokens`, and what the strip
    // below has to know about -- a chat turn ends on `<end_of_turn>`, so stripping only `<eos>` leaves
    // a control token's spelling on the end of the answer.
    const std::vector<int32_t> stop_ids = eos_token_ids(model);
    const int32_t eos = options.eos_token == GenerateOptions::kEosFromFile
        ? (stop_ids.empty() ? -1 : stop_ids.front())
        : options.eos_token;
    // A caller-named stop token replaces the file's set rather than joining it: "stop here" is an
    // instruction, and a loop that also stopped somewhere else would not be following it.
    const std::vector<int32_t> strip_ids =
        options.eos_token == GenerateOptions::kEosFromFile ? stop_ids : std::vector<int32_t>{eos};
    return {eos, strip_ids};
}

std::unordered_map<std::string, LoomLuaBridge::Value> driver_args(const GenerateOptions& options, int32_t eos) {
    std::unordered_map<std::string, LoomLuaBridge::Value> args = options.extra_inputs;
    args["max_new_tokens"] = static_cast<double>(options.max_new_tokens);
    args["eos_token"] = static_cast<double>(eos);
    // Only what the caller SET (P4.24). An entry that is absent leaves the driver's own
    // `inputs.<knob> or <default>` fallback intact, which is the checkpoint's declared value --
    // passing a host default here instead would silently overrule the file for every caller who
    // named nothing.
    if (options.temperature) args["temperature"] = static_cast<double>(*options.temperature);
    if (options.top_k) args["top_k"] = static_cast<double>(*options.top_k);
    if (options.top_p) args["top_p"] = static_cast<double>(*options.top_p);
    return args;
}

} // namespace

std::vector<int32_t> eos_token_ids(const GgufModel& model) {
    if (model.has_kv("tokenizer.ggml.eos_token_ids")) {
        std::vector<int32_t> ids = model.kv_arr_i32("tokenizer.ggml.eos_token_ids");
        if (!ids.empty()) return ids;
    }
    const int32_t single = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
    return single >= 0 ? std::vector<int32_t>{single} : std::vector<int32_t>{};
}

std::vector<int32_t> generate(LoomLuaBridge& bridge, const GgufModel& model,
                              const std::vector<int32_t>& prompt_tokens,
                              const GenerateOptions& options) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("generate: model carries no driver_script; it can be inspected but not run.");
    }
    if (prompt_tokens.empty()) {
        throw Error("generate: the prompt is empty. A driver is called with its primary input, and a "
                    "model asked to continue nothing has nothing to continue.");
    }

    const Stops stops = resolve_stops(model, options);
    const int32_t eos = stops.eos;
    const std::vector<int32_t>& strip_ids = stops.strip;

    if (options.seed) bridge.seed_rng(*options.seed);

    std::vector<double> running(prompt_tokens.begin(), prompt_tokens.end());

    // `tokens` is a CONVENTION the exporter guarantees, not a guess: every generated driver accepts it
    // as an alias for its primary input whatever the traced graph called it (`input_ids`, `token_ids`,
    // `audio_signal`) -- `driver_components.GENERIC_PRIMARY_INPUT` is where that is written down.
    // WHICH ENTRY POINT, and it is worth 2.83x. Every generated causal-LM driver exports BOTH `infer`
    // -- one forward over whatever it is handed, returning one token -- and `infer_with_past`, which
    // runs the decode loop itself against the KV cache. Calling `infer` unconditionally meant the
    // branch below re-fed a growing prompt, so every step recomputed the WHOLE sequence: 112 MUL_MATs
    // per step at `ne1` = the current length rather than 1, and a decode that is O(n^2) in tokens.
    //
    // Measured on Qwen3-0.6B, 65 tokens, 24-core x86: 8.43 s through `infer`, 2.98 s through
    // `infer_with_past` -- 7.71 vs 21.78 tok/s. onnxruntime does 19.2-21.4 on the same box, so this
    // one call was the entire gap to it (Epic-05 §2).
    //
    // Not every driver has one. lfm2-350m-monolithic does (its ShortConv history is a ConvStateCache),
    // but the MODULAR export -- one topology per layer -- carries no caches, has only `infer`, and
    // keeps the re-fed loop. Hence prefer-if-present rather than require.
    const std::string entry = bridge.has_function("infer_with_past") ? "infer_with_past" : "infer";

    const auto call = [&]() {
        std::unordered_map<std::string, LoomLuaBridge::Value> args = driver_args(options, eos);
        args["tokens"] = running;
        return bridge.call(entry, args);
    };

    std::vector<int32_t> generated;
    const LoomLuaBridge::Value first = call();

    if (std::holds_alternative<std::vector<double>>(first)) {
        // The driver's cross-step state is entirely the KV cache, so it ran the whole loop itself and
        // returned what it generated. Its own stop condition already applied; there is nothing to
        // iterate here.
        for (double id : std::get<std::vector<double>>(first)) {
            generated.push_back(static_cast<int32_t>(id));
        }
        // The header promises the GENERATED ids only, and until P4.23 nothing enforced it on this
        // branch: the two shapes are told apart by what the driver RETURNS (list vs number), never by
        // whether the return contains the prompt, so a driver that handed back prompt+generated would
        // be echoed and nothing would catch it. Every generated `infer_with_past` starts its `_gen`
        // empty (`driver_components.py`), so this has never fired -- a hand-written or future driver
        // is what it exists for.
        //
        // **The COUNT is the check, deliberately, and not "does it start with the prompt".** That
        // second test cannot tell an echo from a model repeating its own prompt back, which is exactly
        // what an un-templated instruction model does (P4.23's reported symptom was that, generated) --
        // so it would throw on correct output. The count cannot be wrong about anything: the loop
        // breaks at `#_gen >= max_new_tokens`, so a conforming driver never returns more, whatever it
        // generated. `max(1)` because that break is checked AFTER the first token is appended, so one
        // token comes back even from `max_new_tokens = 0`.
        const size_t ceiling = std::max<size_t>(options.max_new_tokens, 1);
        if (generated.size() > ceiling) {
            throw Error("generate: the driver returned " + std::to_string(generated.size()) +
                        " ids for max_new_tokens=" + std::to_string(options.max_new_tokens) +
                        ". The list shape is defined as the GENERATED ids alone, so this driver is "
                        "returning something else -- the prompt in front of them, most likely. Fix its "
                        "own loop; normalising it here would hide which of the two contracts it thinks "
                        "it is meeting.");
        }
    } else {
        // One token per call: the prompt grows and is re-fed, which is what a driver without cross-step
        // state requires (lfm2-350m-modular's per-layer export carries no caches).
        generated.push_back(new_token(first));
        running.push_back(static_cast<double>(generated.back()));
        while (generated.size() < options.max_new_tokens && generated.back() != eos) {
            generated.push_back(new_token(call()));
            running.push_back(static_cast<double>(generated.back()));
        }
    }

    if (options.strip_eos && !generated.empty() &&
        std::find(strip_ids.begin(), strip_ids.end(), generated.back()) != strip_ids.end()) {
        generated.pop_back();
    }
    return generated;
}

GenerationStream::GenerationStream(LoomLuaBridge& bridge, const GgufModel& model,
                                   std::vector<int32_t> prompt_tokens, GenerateOptions options)
    : bridge_(bridge), prompt_(std::move(prompt_tokens)), options_(std::move(options)) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("GenerationStream: model carries no driver_script; it can be inspected but not run.");
    }
    if (!supported(bridge)) {
        throw LoadError("GenerationStream: this file's driver runs its decode loop inside one call "
                        "(`infer_with_past`) and exports no `stream_open` / `stream_feed` / "
                        "`stream_finish`, so it cannot hand tokens back as they are drawn. A re-export "
                        "with this runtime's exporter adds them; `generate` still runs it one-pass.");
    }
    if (prompt_.empty()) {
        throw Error("GenerationStream: the prompt is empty. A model asked to continue nothing has "
                    "nothing to continue.");
    }
    const Stops stops = resolve_stops(model, options_);
    eos_ = stops.eos;
    strip_ = stops.strip;
    driver_loop_ = bridge.has_function("stream_open");
    reset();
}

bool GenerationStream::supported(const LoomLuaBridge& bridge) {
    // The one-token shape keeps nothing across calls but the tokens, which this class holds.
    return bridge.has_function("stream_open") || !bridge.has_function("infer_with_past");
}

void GenerationStream::reset() {
    state_.clear();
    if (options_.seed) state_.seed_rng(*options_.seed);
    tokens_.clear();
    running_.assign(prompt_.begin(), prompt_.end());
    opened_ = false;
    finished_ = false;
    stripped_ = false;
}

std::vector<int32_t> GenerationStream::next(uint32_t max_tokens) {
    std::vector<int32_t> out;
    if (finished_ || max_tokens == 0) return out;

    if (driver_loop_) {
        if (!opened_) {
            std::unordered_map<std::string, LoomLuaBridge::Value> args = driver_args(options_, eos_);
            args["tokens"] = std::vector<double>(prompt_.begin(), prompt_.end());
            opened_ = true;
            for (double id : as_ids(bridge_.call("stream_open", args, state_))) out.push_back(static_cast<int32_t>(id));
        }
        // The driver's own loop decides when it is done -- an eos it was told about, one of the
        // file's extra stop ids it carries, its ceiling -- and says so in its state, so the host never
        // has to re-derive the rule `infer_with_past` applies.
        while (!driver_done() && out.size() < max_tokens) {
            const uint32_t want = max_tokens - static_cast<uint32_t>(out.size());
            const std::vector<double> ids =
                as_ids(bridge_.call("stream_feed", {{"max_tokens", static_cast<double>(want)}}, state_));
            for (double id : ids) out.push_back(static_cast<int32_t>(id));
            if (ids.empty() && !driver_done()) {
                throw Error("GenerationStream: the driver's stream_feed drew nothing and did not say it "
                            "was done, so another call would not draw anything either.");
            }
        }
        if (driver_done()) {
            finished_ = true;
            bridge_.call("stream_finish", {}, state_);
        }
    } else {
        // `generate`'s one-token loop: the first call always draws, then it runs until the ceiling or
        // `eos` -- `eos` alone, as there.
        while (!finished_ && out.size() < max_tokens) {
            std::unordered_map<std::string, LoomLuaBridge::Value> args = driver_args(options_, eos_);
            args["tokens"] = running_;
            opened_ = true;
            const int32_t id = new_token(bridge_.call("infer", args, state_));
            running_.push_back(static_cast<double>(id));
            out.push_back(id);
            if (tokens_.size() + out.size() >= std::max<size_t>(options_.max_new_tokens, 1) || id == eos_) {
                finished_ = true;
            }
        }
    }
    // As `generate`: only the LAST id is a stop token's spelling to drop, and only the stream's end has
    // a last id.
    if (finished_ && options_.strip_eos && !out.empty() &&
        std::find(strip_.begin(), strip_.end(), out.back()) != strip_.end()) {
        out.pop_back();
        stripped_ = true;
    }
    tokens_.insert(tokens_.end(), out.begin(), out.end());
    return out;
}

bool GenerationStream::driver_done() const {
    const StreamState::Value* done = state_.value("done");
    return done != nullptr && !done->data.empty() && done->data.front() != 0.0;
}

std::vector<double> GenerationStream::as_ids(const LoomLuaBridge::Value& v) {
    if (std::holds_alternative<double>(v)) return {std::get<double>(v)};
    return std::get<std::vector<double>>(v);
}

} // namespace text
} // namespace loom
