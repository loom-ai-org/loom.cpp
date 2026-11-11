#include "loom/core/frame_stream.h"

#include "loom/core/model_contract.h"
#include "loom/core/transcribe_stream.h"
#include "loom/loom_errors.h"

#include <string>

namespace loom {
namespace audio {

bool FrameStream::supported(const LoomLuaBridge& bridge) {
    // The same three functions the transcript door looks for.
    return TranscriptionStream::supported(bridge);
}

FrameStream::FrameStream(LoomLuaBridge& bridge, const GgufModel& model) : bridge_(bridge) {
    const ModelContract contract = ModelContract::read(model);
    if (contract.input_kind != modality::AUDIO ||
        (contract.output_kind != modality::CLASS && contract.output_kind != modality::EMBEDDINGS)) {
        throw Error("FrameStream: this model declares " +
                    (contract.declared() ? contract.interface_name() : std::string("no contract")) +
                    ", not audio in and `class` or `embeddings` out.");
    }
    if (contract.output_granularity != granularity::FRAME) {
        throw Error("FrameStream: this model answers per `" + contract.output_granularity + "`, and only a "
                    "frame answer streams: a clip's one row needs the whole clip. Its door is the one-pass "
                    "classify / embed.");
    }
    output_kind_ = contract.output_kind;
    if (output_kind_ == modality::CLASS) {
        labels_ = contract.labels;
        width_ = static_cast<uint32_t>(labels_.size());
    } else {
        width_ = contract.embedding_dim;
    }
    if (width_ == 0) {
        // `classify`'s and `embed`'s refusal, for their reason: without the width a flat answer cannot be
        // cut into rows, and guessing it from the size is the inference the contract exists to remove.
        throw Error(std::string("FrameStream: this model declares no row width (") +
                    (output_kind_ == modality::CLASS ? "`loom.labels`" : "`loom.output.embedding_dim`") +
                    "), so its answer cannot be split into one row per frame.");
    }
    if (!(contract.frame_rate > 0.0)) {
        throw Error("FrameStream: this model declares no frame rate (`loom.output.frame_rate`), so a row "
                    "has no time.");
    }
    frame_rate_ = contract.frame_rate;
    frame_offset_ = contract.frame_offset;
    if (!supported(bridge)) {
        throw LoadError("this model's driver has no live path (stream_open / stream_feed / stream_finish): "
                        "it was exported before streaming, or its family does not stream yet. Its one-pass "
                        "classify / embed is unaffected.");
    }
    reset();
}

void FrameStream::reset() {
    state_.clear();
    rows_ = 0;
    samples_ = 0;
    finished_ = false;
    bridge_.call("stream_open", {}, state_);
}

FrameUpdate FrameStream::push(const float* samples, size_t n) {
    if (finished_) throw Error("FrameStream::push: the stream has finished; reset() reopens it");
    std::vector<double> pcm(samples, samples + n);
    samples_ += n;
    return absorb(bridge_.call("stream_feed", {{"waveform", std::move(pcm)}}, state_), "stream_feed");
}

FrameUpdate FrameStream::finish() {
    if (finished_) throw Error("FrameStream::finish: the stream has already finished");
    finished_ = true;
    return absorb(bridge_.call("stream_finish", {}, state_), "stream_finish");
}

FrameUpdate FrameStream::absorb(const LoomLuaBridge::Value& result, const char* fn) {
    FrameUpdate update;
    update.first_row = rows_;
    update.samples = samples_;
    if (std::holds_alternative<std::vector<double>>(result)) {
        const auto& raw = std::get<std::vector<double>>(result);
        if (raw.size() % width_ != 0) {
            throw Error(std::string("FrameStream: the driver's ") + fn + " returned " +
                        std::to_string(raw.size()) + " numbers, which is not a whole number of rows of " +
                        std::to_string(width_) + ".");
        }
        update.values.assign(raw.begin(), raw.end());
        update.n_rows = static_cast<uint32_t>(raw.size() / width_);
    } else {
        // A single number is one value, never a row of a model wider than one -- and a width-1 model's
        // driver returns its rows as an array like any other.
        throw Error(std::string("FrameStream: the driver's ") + fn + " returned a single number where an "
                    "array of rows was expected.");
    }
    rows_ += update.n_rows;
    return update;
}

} // namespace audio
} // namespace loom
