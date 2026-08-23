#include "loom/core/text_generate.h"

#include "loom/loom_errors.h"

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

} // namespace

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

    // The file's own stop token unless the caller named one, so the loop stops where the checkpoint
    // says it should rather than where a host's default happened to land. Negative means no early stop,
    // which the generated drivers document as the meaning of a negative `eos_token`.
    const int32_t eos = options.eos_token == GenerateOptions::kEosFromFile
        ? model.kv_i32("tokenizer.ggml.eos_token_id", -1)
        : options.eos_token;

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
    // Not every driver has one: LFM2's ShortConv blocks carry history no KV cache holds, so its export
    // has only `infer` and must keep the re-fed loop. Hence prefer-if-present rather than require.
    const std::string entry = bridge.has_function("infer_with_past") ? "infer_with_past" : "infer";

    const auto call = [&]() {
        std::unordered_map<std::string, LoomLuaBridge::Value> args = options.extra_inputs;
        args["tokens"] = running;
        args["max_new_tokens"] = static_cast<double>(options.max_new_tokens);
        args["eos_token"] = static_cast<double>(eos);
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
    } else {
        // One token per call: the prompt grows and is re-fed, which is what a driver without cross-step
        // state requires (LFM2's ShortConv blocks carry history no cache holds).
        generated.push_back(new_token(first));
        running.push_back(static_cast<double>(generated.back()));
        while (generated.size() < options.max_new_tokens && generated.back() != eos) {
            generated.push_back(new_token(call()));
            running.push_back(static_cast<double>(generated.back()));
        }
    }

    if (options.strip_eos && !generated.empty() && generated.back() == eos) generated.pop_back();
    return generated;
}

} // namespace text
} // namespace loom
