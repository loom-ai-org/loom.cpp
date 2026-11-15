#include "loom/core/transcribe_stream.h"

#include "loom/core/model_contract.h"
#include "loom/loom_errors.h"

#include <unordered_map>
#include <utility>

namespace loom {
namespace audio {

namespace {

const char* const kEntryPoints[] = {"stream_open", "stream_feed", "stream_finish"};

AsrText load_text(const GgufModel& model) {
    AsrText text(model);
    // The same dropped ids `transcribe` drops, from the same declared table.
    const AsrDecodeTable table = AsrDecodeTable::read(model, text.bpe(), ModelContract::read(model));
    text.add_control_ids(table.control_ids);
    return text;
}

} // namespace

bool TranscriptionStream::supported(const LoomLuaBridge& bridge) {
    for (const char* fn : kEntryPoints) {
        if (!bridge.has_function(fn)) return false;
    }
    return true;
}

double TranscriptionStream::language_input(const GgufModel& model, const std::string& name) {
    AsrText text(model);
    const AsrDecodeTable table = AsrDecodeTable::read(model, text.bpe(), ModelContract::read(model));
    const int32_t id = table.language(name);
    if (id < 0) {
        std::string have;
        for (const auto& l : table.language_names) have += (have.empty() ? "" : " ") + l;
        throw LoadError("this model's stream has no language named \"" + name + "\" (" +
                        (have.empty() ? std::string("it declares none") : "it declares: " + have) + ")");
    }
    return static_cast<double>(id);
}

TranscriptionStream::TranscriptionStream(LoomLuaBridge& bridge, const GgufModel& model, Inputs inputs)
    : bridge_(bridge), inputs_(std::move(inputs)), asr_text_(load_text(model)) {
    if (!supported(bridge)) {
        throw LoadError("this model's driver has no live path (stream_open / stream_feed / stream_finish): "
                        "it was exported before streaming, or its family does not stream yet. Its one-pass "
                        "transcription is unaffected.");
    }
    reset();
}

void TranscriptionStream::reset() {
    state_.clear();
    tokens_.clear();
    text_.clear();
    samples_ = 0;
    finished_ = false;
    bridge_.call("stream_open", inputs_, state_);
}

TranscriptUpdate TranscriptionStream::push(const float* samples, size_t n) {
    if (finished_) throw Error("TranscriptionStream::push: the stream has finished; reset() reopens it");
    std::vector<double> pcm(samples, samples + n);
    samples_ += n;
    return absorb(bridge_.call("stream_feed", {{"waveform", std::move(pcm)}}, state_));
}

TranscriptUpdate TranscriptionStream::finish() {
    if (finished_) throw Error("TranscriptionStream::finish: the stream has already finished");
    finished_ = true;
    return absorb(bridge_.call("stream_finish", {}, state_));
}

TranscriptUpdate TranscriptionStream::absorb(const LoomLuaBridge::Value& result) {
    TranscriptUpdate update;
    if (const StreamState::Value* hypothesis = state_.value("hypothesis")) {
        // A re-decoding driver's whole transcript so far (ADR-106): it replaces, it does not append.
        for (double v : hypothesis->data) update.tokens.push_back(static_cast<int32_t>(v));
        tokens_ = update.tokens;
    } else {
        if (std::holds_alternative<std::vector<double>>(result)) {
            for (double v : std::get<std::vector<double>>(result)) update.tokens.push_back(static_cast<int32_t>(v));
        }
        tokens_.insert(tokens_.end(), update.tokens.begin(), update.tokens.end());
    }
    std::string text = asr_text_.detokenize(tokens_);
    // A detokenizer may rewrite earlier text when more ids arrive (a byte-level BPE completing a
    // multi-byte character); the delta is only a suffix when the old text is a prefix of the new.
    if (text.compare(0, text_.size(), text_) == 0) {
        update.delta = text.substr(text_.size());
    } else {
        update.delta = text;
        update.revised = true;
    }
    text_ = std::move(text);
    update.text = text_;
    update.samples = samples_;
    return update;
}

} // namespace audio
} // namespace loom
