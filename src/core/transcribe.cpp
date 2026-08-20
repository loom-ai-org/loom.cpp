#include "loom/core/transcribe.h"

#include "loom/core/audio_window.h"
#include "loom/core/bpe_vocab.h"
#include "loom/core/model_contract.h"
#include "loom/core/vocab.h"
#include "loom/loom_errors.h"

#include <algorithm>
#include <cmath>
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

    // What this checkpoint declares about its own decode -- timestamp ids, control ids, the language and
    // task tables. Everything below reads from here rather than spelling a token, which is what keeps
    // this loop per-TASK: Canary, Qwen3-ASR and Granite-Speech each spell timestamps and languages
    // differently, and under the old arrangement the second of them would have cost engine code (see
    // model_contract.h, and docs/HIGH-LEVEL-API.md §3). Files exported before the table existed get the
    // Whisper spellings back as a documented fallback inside `read`.
    const ModelContract contract = ModelContract::read(model);
    const AsrDecodeTable table = AsrDecodeTable::read(model, bpe_vocab.get(), contract);

    // Control tokens a transcript must not contain, dropped before detokenizing because each is a real
    // vocabulary piece that otherwise prints as its literal spelling: the end-of-sequence token, which
    // the driver returns deliberately, and `<|notimestamps|>`, which is the model stating something
    // about the decode rather than a word that was spoken. Timestamp markers are NOT dropped.
    //
    // EOS is added here rather than read from the table because every vocabulary family names it the
    // same way, in `tokenizer.ggml.eos_token_id` -- it needs no per-task declaration to be found.
    const int32_t eos_id = default_eos_token(model);
    std::vector<int32_t> control_ids = table.control_ids;
    control_ids.push_back(eos_id);
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

    Transcription out;

    const uint32_t clip = fixed_clip_samples(model);
    const uint32_t max_new_tokens = default_max_new_tokens(model);

    // Whether this file lets a caller CHOOSE at all, which is a different question from whether it has
    // the particular name asked for -- and telling the two apart is what decides between a warning and
    // a refusal. A declared table answers directly. A file with no table falls back to Whisper's
    // `<|xx|>` spelling, and there the probe is `<|en|>` / `<|transcribe|>`: Whisper's multilingual
    // vocabulary carries every language token including English, and its English-only vocabulary
    // carries none of them, so the presence of the probe separates "wrong name" from "no choice here".
    //
    // The FIRST condition is the decode path, and it is the one that generalises. A dynamic-length
    // family (`clip == 0` -- the NeMo CTC and transducer models) calls `infer` with the waveform and
    // its length and NOTHING else: there is no prompt to put a language token in, so no argument here
    // can reach the model whatever its vocabulary happens to contain. Deciding on the vocabulary alone
    // got this wrong in a way worth recording -- parakeet-tdt carries a literal `<|en|>` PIECE in its
    // SentencePiece vocabulary, so a spelled probe reports it as language-selectable, and a caller
    // passing `language="en"` would have been given neither a warning nor an effect.
    const auto offers = [&](const std::vector<std::string>& declared, const char* probe) {
        if (clip == 0) return false;
        if (!declared.empty()) return true;
        return table.legacy_spelling && bpe_vocab && bpe_vocab->piece_to_id(probe) >= 0;
    };
    const bool selectable_language = offers(table.language_names, "<|en|>");
    const bool selectable_task = offers(table.task_names, "<|transcribe|>");

    // Names to token ids, here rather than in a host: only the file can answer, and every host would
    // otherwise reimplement it or push the lookup onto a caller who cannot do it. The declared table is
    // asked first; a file that carries none falls back to Whisper's `<|xx|>` spelling, which is the one
    // place that spelling survives and is flagged as such by `legacy_spelling`.
    //
    // AN ARGUMENT THAT SELECTS NOTHING IS IGNORED, NOT REFUSED. A monolingual checkpoint has no
    // language tokens, so `language="en"` on one is redundant rather than wrong -- it names exactly
    // what the model was always going to do. Throwing there sent callers to look for a defect in a
    // pipeline that had none, and made the obvious spelling of a correct call an error. It warns and
    // proceeds now.
    //
    // A request the model cannot SERVE still throws, and the two are not the same:
    //   * a multilingual file asked for a language it does not have -- the caller wants that language;
    //   * any file asked to `translate` when it has no task tokens -- transcribing instead would
    //     return fluent output in the wrong language and look like a success.
    // `transcribe` is the exception among tasks: it names the default, so on a file with no task
    // tokens it is redundant in exactly the way `language` is.
    const auto resolve = [&](const std::string& name, const char* what, bool is_task,
                             bool selectable) -> int32_t {
        if (name.empty()) return -1;
        int32_t id = is_task ? table.task(name) : table.language(name);
        if (id < 0 && table.legacy_spelling && bpe_vocab) id = bpe_vocab->piece_to_id("<|" + name + "|>");
        if (id < 0) {
            // A file that DECLARES which languages it speaks overrules the "no tokens, so harmless"
            // reading: an English-only checkpoint asked for French has no way to comply, and ignoring
            // the argument would hand back fluent English as though it were the translation. Inert
            // today -- no exporter writes `loom.text.languages` yet (its own gap) -- and correct the
            // day one does, which is why it is a check rather than a comment.
            const bool contradicted = !is_task && !contract.languages.empty() &&
                std::find(contract.languages.begin(), contract.languages.end(), name) ==
                    contract.languages.end();
            const bool redundant = !selectable && !contradicted && (!is_task || name == "transcribe");
            if (redundant) {
                out.warnings.push_back(
                    "transcribe: this model has no " + std::string(what) + " tokens, so " +
                    std::string(what) + "=\"" + name + "\" selects nothing and was ignored. It "
                    "decodes in the one " + std::string(what) + " it was trained for; omit the "
                    "argument to say so explicitly.");
                return -1;
            }
            throw LoadError("transcribe: this model has no " + std::string(what) + " named \"" + name +
                            "\". " + (selectable
                                ? "It does select a " + std::string(what) + ", so this name is wrong "
                                  "rather than unnecessary -- `model.contract` lists what it declares."
                                : "It has no " + std::string(what) + " tokens at all, so this cannot "
                                  "be honoured by ignoring it: the result would not be what was asked "
                                  "for."));
        }
        return id;
    };
    const int32_t language_id = resolve(options.language, "language", /*is_task=*/false, selectable_language);
    const int32_t task_id = resolve(options.task, "task", /*is_task=*/true, selectable_task);

    // 16 kHz is what every ASR family exported so far takes, and is the only rate at which a file that
    // declares none can be interpreted at all -- the alternative is refusing to transcribe a model that
    // worked before the contract existed.
    const double rate = contract.sample_rate ? contract.sample_rate : 16000;

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
        // The ONE segment spans the whole clip, and says so. It used to be `{0.0, 0.0}`, which reads as
        // a zero-length segment at the start of the file -- a measurement, and a wrong one. These
        // families (CTC and transducer) emit no timestamp tokens, so there is no boundary the model
        // chose to report; what IS true is the extent this transcript covers, which is all of it.
        //
        // `closed` stays false and `out.timestamped` stays false, which is where a caller reads that
        // these are not model-chosen boundaries. An end of 0.0 said that too, but only to a reader who
        // already knew -- and it broke anything that sorts, seeks, or sums durations.
        out.segments.push_back({0.0, static_cast<double>(waveform.size()) / rate, out.text, false});
        out.windows = 1;
        return out;
    }

    const int32_t ts_base = table.timestamp_first_id;
    const double ts_step = table.timestamp_step_sec;
    const bool can_timestamp = table.timestamped();
    const size_t n_clips = (waveform.size() + clip - 1) / clip;
    out.timestamped = can_timestamp;

    // Timestamps are REQUESTED whenever they are usable and there is more than one window, even if the
    // caller did not ask to see them: they are what the seek advances on. A single-window file without
    // them keeps decoding in no-timestamps mode, so short audio behaves exactly as it did.
    const bool want_timestamps = options.timestamps || (can_timestamp && n_clips > 1);

    const size_t prev_cap = table.prev_context ? table.prev_context : 448;
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
            // ROUNDED, not truncated, and the difference is a whole extra decode. A segment end is a
            // float number of seconds, so the sample index it maps to is almost never exact: with the
            // timestamp step read from the file as f32 (`loom.asr.timestamp_step_sec`), 550 steps of
            // 0.02 s come to 10.99999975 s rather than 11, and 11 s of audio at 16 kHz truncates to
            // 175999 -- one sample short of the end. The loop then ran a SECOND window over four
            // samples of real audio and 30 s of zero padding, and Whisper transcribed the silence as
            // "[BLANK_AUDIO]", which was appended to the transcript.
            //
            // Truncating a sample index derived from a float time was arbitrary anyway; the value is a
            // measurement with error either side of it, and rounding is what that deserves. Found by
            // re-exporting the gate fixtures with the declared table and comparing against the same
            // audio through the old derived-from-hparams path, which had absorbed the error by luck.
            auto candidate = static_cast<size_t>(std::llround((last_complete_end - window_start) * rate));
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
