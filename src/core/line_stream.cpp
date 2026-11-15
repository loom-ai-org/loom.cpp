#include "loom/core/line_stream.h"

#include "loom/core/model_contract.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace loom {
namespace audio {

LineStream::LineStream(LoomLuaBridge& asr_bridge, const GgufModel& asr, LoomLuaBridge& vad_bridge,
                       const GgufModel& vad, LineOptions options)
    : asr_bridge_(asr_bridge), asr_(asr), options_(std::move(options)) {
    const ModelContract asr_contract = ModelContract::read(asr);
    const ModelContract vad_contract = ModelContract::read(vad);
    vad_ = std::make_unique<FrameStream>(vad_bridge, vad);
    if (vad_->output_kind() != modality::CLASS) {
        throw Error("LineStream: the VAD answers embeddings, not a speech probability per frame.");
    }
    const auto& labels = vad_->labels();
    const auto speech = std::find(labels.begin(), labels.end(), "speech");
    if (speech == labels.end()) {
        throw Error("LineStream: the VAD declares no `speech` label, so it has no speech probability to "
                    "threshold.");
    }
    speech_column_ = static_cast<uint32_t>(speech - labels.begin());
    // A file exported before the contract carried a rate is read at 16 kHz, as `transcribe` reads it.
    rate_ = vad_contract.sample_rate ? vad_contract.sample_rate : 16000;
    const uint32_t asr_rate = asr_contract.sample_rate ? asr_contract.sample_rate : 16000;
    if (asr_rate != rate_) {
        throw Error("LineStream: the recogniser takes " + std::to_string(asr_rate) +
                    " Hz and the VAD " + std::to_string(rate_) + " Hz; a line is cut from the VAD's samples "
                    "and handed to the recogniser as they are.");
    }
    // One VAD row per hop: the hop is the VAD's own frame, or the rows are not the hops' probabilities.
    const double frame = static_cast<double>(rate_) / vad_->frame_rate();
    if (options_.hop == 0 || std::fabs(frame - options_.hop) > 1e-9 || vad_->frame_offset() != 0.0) {
        throw Error("LineStream: the VAD answers one row per " + std::to_string(frame) + " samples from " +
                    std::to_string(vad_->frame_offset()) + " s, and the hop is " + std::to_string(options_.hop) +
                    " samples from 0.");
    }
    if (!(options_.update_seconds >= 0.0) || !(options_.window_seconds > 0.0f) ||
        !(options_.max_line_seconds >= 0.0f) || !(options_.max_tokens_per_second >= 0.0f)) {
        throw Error("LineStream: update_seconds, max_line_seconds and max_tokens_per_second are at least 0, "
                    "window_seconds more than 0.");
    }
    // `vad_window_size_from_duration` and `vad_sample_count_from_duration`, in their float arithmetic.
    window_ = static_cast<uint32_t>(
        std::ceil((options_.window_seconds * static_cast<float>(rate_)) / static_cast<float>(options_.hop)));
    max_segment_ = static_cast<size_t>(std::round(options_.max_line_seconds * static_cast<float>(rate_)));
    reset();
}

void LineStream::reset() {
    vad_->reset();
    pending_.clear();
    remainder_.clear();
    look_behind_.assign(options_.look_behind, 0.0f);
    segment_.clear();
    window_probs_.assign(window_, 0.0f);
    window_index_ = 0;
    previous_voice_ = false;
    processed_ = 0;
    lines_.clear();
    line_audio_.clear();
    touched_.clear();
    probs_.clear();
    smoothed_.clear();
    samples_ = 0;
    finished_ = false;
}

std::string LineStream::text() const {
    std::string out;
    for (const Line& line : lines_) {
        if (line.text.empty()) continue;
        if (!out.empty()) out += ' ';
        out += line.text;
    }
    return out;
}

LineUpdate LineStream::push(const float* samples, size_t n) {
    if (finished_) throw Error("LineStream::push: the stream has finished; reset() reopens it");
    pending_.insert(pending_.end(), samples, samples + n);
    samples_ += n;
    // `new_audio_duration >= transcription_interval`, both float.
    const float gathered = static_cast<float>(pending_.size()) / static_cast<float>(rate_);
    if (pending_.empty() || !(gathered >= static_cast<float>(options_.update_seconds))) {
        LineUpdate update;
        update.samples = samples_;
        return update;
    }
    return run(false);
}

LineUpdate LineStream::finish() {
    if (finished_) throw Error("LineStream::finish: the stream has already finished");
    finished_ = true;
    if (pending_.empty()) {
        // Upstream's stop with nothing new: the open line closes on the text it has, nothing re-runs.
        if (previous_voice_) {
            previous_voice_ = false;
            lines_.back().complete = true;
            line_audio_.back().clear();
        }
        LineUpdate update;
        update.samples = samples_;
        return update;
    }
    return run(true);
}

LineUpdate LineStream::run(bool stopping) {
    touched_.assign(lines_.size(), false);
    std::vector<float> buffer = std::move(remainder_);
    buffer.insert(buffer.end(), pending_.begin(), pending_.end());
    pending_.clear();
    const size_t hops = buffer.size() / options_.hop;
    if (hops > 0) {
        // The VAD's rows for these hops, one push: its driver answers each whole frame as it arrives.
        const FrameUpdate rows = vad_->push(buffer.data(), hops * options_.hop);
        if (rows.n_rows != hops) {
            throw Error("LineStream: the VAD answered " + std::to_string(rows.n_rows) + " rows for " +
                        std::to_string(hops) + " hops.");
        }
        for (size_t h = 0; h < hops; ++h) {
            probs_.push_back(rows.values[h * vad_->width() + speech_column_]);
            hop(buffer.data() + h * options_.hop);
        }
    }
    remainder_.assign(buffer.begin() + static_cast<long>(hops * options_.hop), buffer.end());
    if (stopping) {
        // `stop()`: an open line closes on the audio it has; the remainder short of a hop is dropped.
        if (previous_voice_) {
            previous_voice_ = false;
            lines_.back().complete = true;
            touched_.back() = true;
        }
        remainder_.clear();
    }
    LineUpdate update;
    update.ran = true;
    update.samples = samples_;
    transcribe_touched(update);
    return update;
}

void LineStream::hop(const float* samples) {
    const size_t n = options_.hop;
    processed_ += n;
    std::move(look_behind_.begin() + static_cast<long>(n), look_behind_.end(), look_behind_.begin());
    std::copy(samples, samples + n, look_behind_.end() - static_cast<long>(n));

    window_probs_[window_index_] = probs_.back();
    window_index_ = (window_index_ + 1) % window_;
    // `std::accumulate(..., 0.0f) / size()`: a float sum in ring order.
    float smoothed = 0.0f;
    for (float p : window_probs_) smoothed += p;
    smoothed = smoothed / static_cast<float>(window_probs_.size());
    // Faded to 0 over the last third of a line, so a line always ends.
    const size_t fade = (max_segment_ * 2) / 3;
    if (max_segment_ && segment_.size() > fade) {
        const float factor = static_cast<float>(segment_.size() - fade) / static_cast<float>(fade);
        smoothed = smoothed * factor;
    }
    smoothed_.push_back(smoothed);
    const bool voice = smoothed > options_.threshold;
    if (voice && !previous_voice_) {
        const size_t back = std::min(options_.look_behind, processed_);
        segment_.assign(look_behind_.end() - static_cast<long>(back), look_behind_.end());
        Line line;
        line.start = processed_ - segment_.size();
        line.samples = segment_.size();
        lines_.push_back(line);
        line_audio_.push_back(segment_);
        touched_.push_back(true);
    } else if (!voice && previous_voice_) {
        segment_.insert(segment_.end(), samples, samples + n);
        lines_.back().samples = segment_.size();
        lines_.back().complete = true;
        line_audio_.back() = segment_;
        touched_.back() = true;
        segment_.clear();
    } else if (voice && previous_voice_) {
        segment_.insert(segment_.end(), samples, samples + n);
        lines_.back().samples = segment_.size();
        line_audio_.back() = segment_;
        touched_.back() = true;
    }
    previous_voice_ = voice;
}

void LineStream::transcribe_touched(LineUpdate& update) {
    for (size_t k = 0; k < lines_.size(); ++k) {
        if (!touched_[k]) continue;
        update.touched.push_back(k);
        Line& line = lines_[k];
        if (!line.complete && !options_.decode_incomplete) {
            line.text.clear();
            continue;
        }
        TranscribeOptions opts = options_.transcribe;
        if (options_.max_tokens_per_second > 0.0f) {
            // `ceil(audio_duration * max_tokens_per_second)`, the duration in float.
            const float seconds = static_cast<float>(line_audio_[k].size()) / static_cast<float>(rate_);
            opts.max_new_tokens = static_cast<uint32_t>(std::ceil(seconds * options_.max_tokens_per_second));
        }
        line.text = transcribe(asr_bridge_, asr_, line_audio_[k], opts).text;
        // A closed line is final: its audio is not needed again.
        if (line.complete) std::vector<float>().swap(line_audio_[k]);
    }
    touched_.assign(lines_.size(), false);
}

} // namespace audio
} // namespace loom
