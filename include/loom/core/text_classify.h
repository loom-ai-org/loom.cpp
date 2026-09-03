#pragma once

// The token-classification door: text in, one declared class per token out (P5, family 12).
//
// WHY THIS IS IN THE ENGINE rather than in each host. The loop is trivial -- one driver call, no
// iteration -- so the argument `text_generate.h` makes ("it was written twice and the two copies
// disagree") does not apply yet. What does apply is the half of that argument that is about POLICY
// rather than about loops: a WordPiece encode wraps every sentence in the checkpoint's own framing
// tokens, the model labels those rows like any other, and whether they come back is a decision. Two
// hosts making it independently is how `loom_cli` and loom-py ended up with two different transcripts
// for the same model, one task over. Made once, here, on DECLARED ids (`tokenizer.ggml.bos_token_id`
// and `.seperator_token_id`) rather than on a spelling -- so it is right for a family whose framing
// tokens are not `[CLS]`/`[SEP]` without this file learning their names.
//
// WHAT IS NOT HERE. Anything that would make a second family cost C++. The label NAMES are read off
// the file (`ModelContract::labels`), the class count is whatever the head has, and the driver does
// the reduction -- so a punctuation model, a truecaser and a NER tagger reach this same function with
// nothing to distinguish them but their own metadata. See docs/HIGH-LEVEL-API.md §2 for the rule.

#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"
#include "loom/core/model_contract.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace text {

// One input token and the class the model gave it.
struct TokenLabel {
    // The token id as it was fed to the model -- kept so a caller can detokenize the labelled span
    // without re-encoding, which is the same reason `Transcription` keeps its times.
    int32_t token = 0;
    // The class index the driver returned.
    int32_t label_id = 0;
    // Its declared name, or empty when the file names its classes nothing.
    std::string label;
};

struct ClassifyOptions {
    // Drop the rows belonging to the tokenizer's own framing tokens. They are added by the encode
    // rather than written by the caller, and a caller who never asked for them has no way to line the
    // remaining labels up with their own text if they come back.
    bool strip_special = true;
    // Passed through to the driver verbatim, for a model whose `infer` takes more than tokens.
    std::unordered_map<std::string, LoomLuaBridge::Value> extra_inputs;
};

// Runs `model`'s driver once over `tokens` and pairs each row's class with the token that produced it.
//
// The driver returns one id per row of its own output, so the count is a real check rather than a
// formality: a graph whose output is not row-per-token (a pooled sequence classifier, say) reaches this
// with a length mismatch and gets told so, instead of silently having its single row zipped against the
// first token.
std::vector<TokenLabel> classify(LoomLuaBridge& bridge, const GgufModel& model,
                                 const std::vector<int32_t>& tokens,
                                 const ClassifyOptions& options = {});

} // namespace text
} // namespace loom
