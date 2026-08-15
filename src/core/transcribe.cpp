#include "loom/core/transcribe.h"

#include "loom/core/audio_window.h"
#include "loom/core/bpe_vocab.h"
#include "loom/core/vocab.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>

namespace loom {
namespace audio {
namespace {

// Splits one window's token ids into timestamped segments, appending them to `out`.
//
// Whisper emits `<|t0|> text <|t1|> <|t1|> more text <|t2|> ...`: every timestamp token both closes the
// span before it and opens the one after, which is why this is a two-state walk rather than a scan for
// pairs. Anything after the final timestamp is text the model had not finished when the window ran out
// -- kept (it is real transcript) but marked `closed = false`, because it must not be trusted as a seek
// boundary: its end time is the window edge, not something the model chose.
//
// `last_complete_end` returns the end of the last CLOSED segment, in whole-file seconds, or stays
// negative when the model closed none.
template <typename Detokenize>
void split_into_segments(const std::vector<int32_t>& ids, int32_t ts_base, double ts_step,
                          double window_start, double window_end, const Detokenize& detokenize,
                          std::vector<Segment>& out, double& last_complete_end) {
    std::vector<int32_t> pending;
    bool open = false;
    double start = window_start;

    const auto flush = [&](double end, bool closed) {
        if (pending.empty()) return;
        std::string text = detokenize(pending);
        pending.clear();
        // A segment whose text is only the special tokens we drop is not a segment.
        if (text.find_first_not_of(" \t\r\n") == std::string::npos) return;
        out.push_back({start, end, std::move(text), closed});
        if (closed) last_complete_end = end;
    };

    for (int32_t id : ids) {
        if (id >= ts_base) {
            const double t = window_start + static_cast<double>(id - ts_base) * ts_step;
            if (open) flush(t, /*closed=*/true);
            start = t;
            open = true;
            continue;
        }
        pending.push_back(id);
    }
    // Whatever is left ran past the window: end it at the window edge and mark it unclosed.
    flush(window_end, /*closed=*/false);
}

// A decode of one clip's worth of audio: the driver's own `infer`, with whatever optional arguments the
// caller supplied. Returns the token ids it generated.
//
// `language` is passed through only when the caller named one. Omitting the key is not the same as
// passing a default: a driver that can detect the language does so when it is absent, and one that
// cannot falls back to its own default -- the resolution order lives in the model's driver, which is the
// only place that knows whether detection is possible at all.
std::vector<int32_t> run_driver(LoomLuaBridge& bridge, const std::vector<double>& waveform,
                                 int32_t language, int32_t task, bool timestamps,
                                 const std::vector<double>& prev_tokens,
                                 uint32_t max_new_tokens, int32_t eos_token) {
    std::unordered_map<std::string, LoomLuaBridge::Value> args = {
        {"waveform", waveform},
        {"max_new_tokens", static_cast<double>(max_new_tokens)},
        {"eos_token", static_cast<double>(eos_token)},
    };
    // The previous window's tokens, raw: which of them count as TEXT is the driver's question, since
    // the ids that do not (timestamps, <|notimestamps|>, eos) are the ones it has constants for.
    if (!prev_tokens.empty()) args["prev_tokens"] = prev_tokens;
    // Each optional argument is OMITTED rather than defaulted when the caller did not name it, which is
    // what lets the driver apply its own resolution order -- detect, or fall back to what this
    // checkpoint can actually do.
    if (language >= 0) args["language"] = static_cast<double>(language);
    if (task >= 0) args["task"] = static_cast<double>(task);
    if (timestamps) args["timestamps"] = 1.0;

    // Held in a named local before unpacking: `call` returns by value, so a reference bound straight
    // into `std::get<...>(call(...))` outlives the variant holding the vector.
    const LoomLuaBridge::Value result = bridge.call("infer", args);
    std::vector<int32_t> ids;
    if (std::holds_alternative<std::vector<double>>(result)) {
        const auto& out = std::get<std::vector<double>>(result);
        ids.reserve(out.size());
        for (double v : out) ids.push_back(static_cast<int32_t>(v));
    }
    return ids;
}


} // namespace

Transcription transcribe(LoomLuaBridge& bridge, const GgufModel& model,
                         const std::vector<float>& waveform, const TranscribeOptions& options) {
    // Two vocab schemas reach this path, so both are tried: NeMo's checkpoints carry SentencePiece
    // ("llama"/"t5" -> loom::Vocab) and Whisper's carries GPT-2 byte-level BPE ("gpt2" -> BpeVocab).
    // Only the BPE one can resolve a token by TEXT, which is what timestamps need.
    //
    // **BPE first, and the order is load-bearing**: `BpeVocab::load` returns nullptr for a schema that
    // is not its own, while `Vocab::load` THROWS on one -- so asking the SentencePiece loader about a
    // gpt2 file kills the run before the BPE loader is ever reached.
    auto bpe_vocab = BpeVocab::load(model);
    auto spm_vocab = bpe_vocab ? nullptr : Vocab::load(model);
    if (!spm_vocab && !bpe_vocab) {
        throw LoadError("transcribe: model has no tokenizer vocab (tokenizer.ggml.model KV missing)");
    }
    if (!model.has_kv("model.driver_script")) {
        throw LoadError("transcribe: model carries no driver_script; re-export it with `loom-export "
                        "<checkpoint> --task automatic-speech-recognition`");
    }

    // Control tokens a transcript must not contain, dropped before detokenizing because each is a real
    // vocabulary piece that otherwise prints as its literal spelling: the end-of-sequence token, which
    // the driver returns deliberately, and `<|notimestamps|>`, which is the model stating something
    // about the decode rather than a word that was spoken. Timestamp markers are NOT dropped.
    const int32_t eos_id = default_eos_token(model);
    std::vector<int32_t> control_ids{eos_id};
    if (bpe_vocab) {
        const int32_t no_ts = bpe_vocab->piece_to_id("<|notimestamps|>");
        if (no_ts >= 0) control_ids.push_back(no_ts);
    }
    const auto detokenize = [&](const std::vector<int32_t>& ids) {
        std::vector<int32_t> text_ids;
        text_ids.reserve(ids.size());
        for (int32_t id : ids) {
            if (std::find(control_ids.begin(), control_ids.end(), id) == control_ids.end()) {
                text_ids.push_back(id);
            }
        }
        return spm_vocab ? spm_vocab->decode(text_ids) : bpe_vocab->decode(text_ids);
    };

    // Names to token ids, here rather than in a host: only the vocabulary can answer, and every host
    // would otherwise reimplement it or push the lookup onto a caller who cannot do it.
    const auto resolve = [&](const std::string& name, const char* what) -> int32_t {
        if (name.empty()) return -1;
        if (!bpe_vocab) {
            throw LoadError("transcribe: this model's vocabulary cannot resolve a " + std::string(what) +
                            " by name (only a byte-level BPE vocab carries `<|" + name + "|>` tokens); "
                            "omit it and let the driver decide.");
        }
        const int32_t id = bpe_vocab->piece_to_id("<|" + name + "|>");
        if (id < 0) {
            throw LoadError("transcribe: this model has no " + std::string(what) + " token `<|" + name +
                            "|>`. An English-only checkpoint has no language tokens at all, and no "
                            "translate task; omit the argument to let the driver decide.");
        }
        return id;
    };
    const int32_t language_id = resolve(options.language, "language");
    const int32_t task_id = resolve(options.task, "task");

    Transcription out;
    const uint32_t clip = fixed_clip_samples(model);
    const uint32_t max_new_tokens = default_max_new_tokens(model);

    if (clip == 0) {
        // Dynamic length: the whole file in one call, which is the NeMo families' shape.
        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<double> length_d{static_cast<double>(waveform.size())};
        const LoomLuaBridge::Value result =
            bridge.call("infer", {{"waveform", waveform_d}, {"length", length_d}});
        std::vector<int32_t> ids;
        if (std::holds_alternative<std::vector<double>>(result)) {
            for (double id : std::get<std::vector<double>>(result)) {
                ids.push_back(static_cast<int32_t>(id));
            }
        }
        out.text = detokenize(ids);
        out.segments.push_back({0.0, 0.0, out.text, false});
        out.windows = 1;
        return out;
    }

    const uint32_t sample_rate = model.has_kv("loom.sample_rate") ? model.hparam_u32("sample_rate") : 0;
    const uint32_t n_audio_ctx = model.has_kv("loom.n_audio_ctx") ? model.hparam_u32("n_audio_ctx") : 0;
    const int32_t ts_base = bpe_vocab ? bpe_vocab->piece_to_id("<|0.00|>") : -1;
    // Seconds per timestamp token: one encoder frame, i.e. the clip's duration over the number of
    // frames it becomes. Derived from the file's own three numbers rather than hardcoded as 0.02.
    const double ts_step = (ts_base >= 0 && sample_rate > 0 && n_audio_ctx > 0)
        ? (static_cast<double>(clip) / sample_rate) / n_audio_ctx : 0.0;
    const bool can_timestamp = ts_step > 0.0;
    const double rate = sample_rate ? sample_rate : 16000;
    const size_t n_clips = (waveform.size() + clip - 1) / clip;
    out.timestamped = can_timestamp;

    // Timestamps are REQUESTED whenever they are usable and there is more than one window, even if the
    // caller did not ask to see them: they are what the seek advances on. A single-window file without
    // them keeps decoding in no-timestamps mode, so short audio behaves exactly as it did.
    const bool want_timestamps = options.timestamps || (can_timestamp && n_clips > 1);

    const size_t prev_cap = model.has_kv("loom.n_text_ctx") ? model.hparam_u32("n_text_ctx") : 448;
    std::vector<double> prev_tokens;
    size_t seek = 0;
    while (seek < waveform.size()) {
        const size_t avail = std::min(static_cast<size_t>(clip), waveform.size() - seek);
        const std::vector<double> window = window_at(waveform, seek, clip);
        const std::vector<int32_t> ids = run_driver(bridge, window, language_id, task_id,
                                                    want_timestamps, prev_tokens, max_new_tokens,
                                                    eos_id);
        const double window_start = static_cast<double>(seek) / rate;
        const double window_end = window_start + static_cast<double>(avail) / rate;

        double last_complete_end = -1.0;
        if (can_timestamp && want_timestamps) {
            split_into_segments(ids, ts_base, ts_step, window_start, window_end, detokenize,
                                out.segments, last_complete_end);
        } else {
            out.segments.push_back({window_start, window_end, detokenize(ids), false});
        }
        ++out.windows;

        // Advance to where the last complete segment ended rather than by a fixed `clip`, so the next
        // window begins on a boundary the model itself chose. The fallback is the window edge, for the
        // case where it closed no segment at all. Floored at one second as a guarantee of progress: a
        // model closing a 0.02 s segment on a mostly-silent window would otherwise take ~1500
        // iterations to cross it.
        size_t advance = clip;
        if (last_complete_end > window_start) {
            auto candidate = static_cast<size_t>((last_complete_end - window_start) * rate);
            candidate = std::max(candidate, static_cast<size_t>(rate));
            if (candidate <= clip) advance = candidate;
        }

        if (options.condition_on_previous) {
            for (int32_t id : ids) prev_tokens.push_back(static_cast<double>(id));
            if (prev_tokens.size() > prev_cap) {
                prev_tokens.erase(prev_tokens.begin(), prev_tokens.end() - prev_cap);
            }
        }
        seek += advance;
    }

    for (const Segment& s : out.segments) {
        if (!out.text.empty() && !s.text.empty() && out.text.back() != ' ' && s.text.front() != ' ') {
            out.text += ' ';
        }
        out.text += s.text;
    }
    return out;
}

} // namespace audio
} // namespace loom
