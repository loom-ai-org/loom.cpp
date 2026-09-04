#include "loom/core/text_classify.h"

#include "loom/loom_errors.h"

#include <algorithm>
#include <string>

namespace loom {
namespace text {
namespace {

// The ids an encode adds around the caller's own text, from what the file DECLARES. `bos` is where a
// WordPiece export writes CLS (llama.cpp's own convention, which `wordpiece_tokenizer_export.py`
// follows and `wordpiece_vocab.h` documents) and `sep` is SEP; a file naming neither strips nothing,
// which is the right answer for a model whose encode adds nothing.
//
// `eos` is here because the FRAMING IS PER-TOKENIZER, NOT PER-TASK. A WordPiece classifier wraps its
// sentence in CLS/SEP; a SentencePiece one wraps it in BOS/EOS (XLM-R's own post-processor is
// `<s> $A </s>`), and reading only the first list left a trailing `</s>` in the answer wearing whatever
// label the head gave it -- one extra entry, at the end, where a caller comparing lengths against its
// own word count would find it last. Every id is looked up by KV and skipped when the file names none,
// so a model that frames with two of these four still strips exactly two.
std::vector<int32_t> framing_ids(const GgufModel& model) {
    std::vector<int32_t> ids;
    for (const char* key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                            "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.padding_token_id"}) {
        const int32_t id = model.kv_i32(key, -1);
        if (id >= 0) ids.push_back(id);
    }
    return ids;
}

} // namespace

std::vector<TokenLabel> classify(LoomLuaBridge& bridge, const GgufModel& model,
                                 const std::vector<int32_t>& tokens,
                                 const ClassifyOptions& options) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("classify: model carries no driver_script; it can be inspected but not run.");
    }
    if (tokens.empty()) {
        throw Error("classify: no tokens. A classifier labels what it is given, and there is no "
                    "label for an empty sentence.");
    }

    const ModelContract contract = ModelContract::read(model);

    std::unordered_map<std::string, LoomLuaBridge::Value> args = options.extra_inputs;
    // `tokens` is the exporter's guaranteed alias for a driver's primary input whatever the traced
    // graph called it -- `driver_components.GENERIC_PRIMARY_INPUT` is where that is written down.
    args["tokens"] = std::vector<double>(tokens.begin(), tokens.end());
    const LoomLuaBridge::Value result = bridge.call("infer", args);

    if (!std::holds_alternative<std::vector<double>>(result)) {
        throw Error("classify: the driver returned a single value where one class per token was "
                    "expected. `token_labels_epilogue` returns `loom.argmax_rows`, an array; a driver "
                    "that returns a number is a causal LM's, and `loom::text::generate` is its door.");
    }
    const std::vector<double>& label_ids = std::get<std::vector<double>>(result);
    if (label_ids.size() != tokens.size()) {
        throw Error("classify: the driver returned " + std::to_string(label_ids.size()) +
                    " classes for " + std::to_string(tokens.size()) + " tokens. This door pairs row i "
                    "with token i, so a model whose output is not one row per input token does not "
                    "answer it -- a pooled sequence classifier is a different contract, not a "
                    "degenerate case of this one.");
    }

    const std::vector<int32_t> framing = options.strip_special ? framing_ids(model)
                                                               : std::vector<int32_t>{};
    std::vector<TokenLabel> labelled;
    labelled.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (std::find(framing.begin(), framing.end(), tokens[i]) != framing.end()) continue;
        TokenLabel entry;
        entry.token = tokens[i];
        entry.label_id = static_cast<int32_t>(label_ids[i]);
        // An id past the declared table is not an error: the file may name fewer classes than the head
        // has, and reporting the id with no name is more useful than refusing to report it.
        if (entry.label_id >= 0 && static_cast<size_t>(entry.label_id) < contract.labels.size()) {
            entry.label = contract.labels[entry.label_id];
        }
        labelled.push_back(entry);
    }
    return labelled;
}

} // namespace text
} // namespace loom
