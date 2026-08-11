// loom_cli: demo/inspection binary for loom-engine.
//
// Without --prompt/--wav, just loads a .gguf model and reports what GgufModel parsed out of it. With
// --prompt, additionally runs greedy autoregressive generation via Generator and prints the sampled
// tokens. If the model has a "tokenizer.ggml.model"="gpt2" vocab (e.g. a real Qwen3 conversion --
// see tools/convert_qwen3/), --prompt is real text, encoded/decoded via loom::BpeVocab; otherwise it
// falls back to the original whitespace-separated integer token ids (the only option for a model with no
// tokenizer at all, e.g. the Milestone-1 toy LLM). With --wav, runs a real audio-to-text Conformer-CTC demo: loads a 16kHz PCM16 WAV file of ANY length
// (sequence length is genuinely dynamic -- see SPECIFICATION.md §4), runs the full
// waveform -> mel-frontend -> encoder -> CTC-decoder graph sized exactly to that length,
// greedy-CTC-decodes the logits, and detokenizes with the model's real SentencePiece vocab.

#include "loom/loom.h"
#include "loom/core/conv_state_cache.h"
#include "wav_file.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                  "usage: %s --model <path.gguf> --prompt \"<text or token ids>\" [--n-predict N]\n"
                  "       %s --model <asr.gguf> --wav <path.wav> [--language en] "
                  "[--task transcribe|translate] [--timestamps] "
                  "[--no-condition-on-previous]\n",
                  argv0, argv0);
}

std::vector<int32_t> parse_token_ids(const std::string& text) {
    std::vector<int32_t> tokens;
    std::istringstream iss(text);
    int32_t tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}


// One timestamped span of transcript. `closed` records whether the model ended it with a timestamp of
// its own or whether it simply ran out of window -- the distinction the seek below depends on.
struct Segment {
    double start = 0.0;
    double end = 0.0;
    std::string text;
    bool closed = false;
};

std::string format_time(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const auto total_ms = static_cast<long long>(seconds * 1000.0 + 0.5);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld.%03lld", total_ms / 3600000,
                  (total_ms / 60000) % 60, (total_ms / 1000) % 60, total_ms % 1000);
    return buf;
}

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
std::vector<int32_t> run_driver(loom::LoomLuaBridge& bridge, const std::vector<double>& waveform,
                                 int32_t language, int32_t task, bool timestamps,
                                 const std::vector<double>& prev_tokens,
                                 uint32_t max_new_tokens, int32_t eos_token) {
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
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
    const loom::LoomLuaBridge::Value result = bridge.call("infer", args);
    std::vector<int32_t> ids;
    if (std::holds_alternative<std::vector<double>>(result)) {
        const auto& out = std::get<std::vector<double>>(result);
        ids.reserve(out.size());
        for (double v : out) ids.push_back(static_cast<int32_t>(v));
    }
    return ids;
}

void run_asr(loom::GgufModel& model, ggml_backend_t backend, const std::string& wav_path,
             const std::string& language_name, const std::string& task_name, bool timestamps,
             bool condition_on_previous) {
    // Model-agnostic: register whatever topologies the file declares, call the driver it ships, and
    // detokenize with the vocab it embeds. Conformer-CTC, Parakeet-TDT and Parakeet-RNN-T all work
    // through this one path -- the driver is the thing that differs between them, and it travels with
    // the model (BACKLOG.md P4.0.17).
    //
    // It replaces a Conformer-specific routine that read the BARE `model.graph_topology`, computed the
    // relative-position table host-side and drove `loom::ctc_greedy_decode` from C++. All three of
    // those were properties of the bespoke converter's artifact: the MIL export names its topologies,
    // traces the mel frontend and rel-pos attention, and carries its own decode.
    //
    // Two vocab schemas reach this path, so both are tried: NeMo's checkpoints carry SentencePiece
    // ("llama"/"t5" -> loom::Vocab) and Whisper's carries GPT-2 byte-level BPE ("gpt2" -> BpeVocab).
    // Only the BPE one can resolve a token by TEXT, which is what `--language` needs.
    //
    // **BPE first, and the order is load-bearing rather than arbitrary**: `BpeVocab::load` returns
    // nullptr for a schema that is not its own (its header says so, and says callers should try both),
    // while `Vocab::load` THROWS on one -- so asking the SentencePiece loader about a gpt2 file kills
    // the run before the BPE loader is ever reached.
    auto bpe_vocab = loom::BpeVocab::load(model);
    auto spm_vocab = bpe_vocab ? nullptr : loom::Vocab::load(model);
    if (!spm_vocab && !bpe_vocab) {
        throw loom::LoadError("--wav: model has no tokenizer vocab (tokenizer.ggml.model KV missing)");
    }
    if (!model.has_kv("model.driver_script")) {
        throw loom::LoadError("--wav: model carries no driver_script; re-export it with `loom-export "
                              "<checkpoint> --task automatic-speech-recognition`");
    }
    // Control tokens a transcript must not contain, dropped before detokenizing because each is a real
    // vocabulary piece that otherwise prints as its literal spelling.
    //
    //   * the end-of-sequence token, which the driver returns deliberately -- a generator's caller may
    //     want to know whether it stopped or ran out of budget. A transcript is not that caller.
    //   * `<|notimestamps|>`, which the MODEL emits for itself when the prompt did not force it, i.e.
    //     exactly under `--timestamps`: it is Whisper deciding not to timestamp, which is a statement
    //     about the decode rather than a word that was spoken.
    //
    // Timestamp markers themselves are NOT dropped: asking for them is the entire point of the flag.
    const int32_t eos_id = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
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

    // `--language xx` becomes this model's own `<|xx|>` token id, by TEXT rather than by a number this
    // CLI would otherwise have to carry per checkpoint. -1 means "not named", which is what asks the
    // driver to detect it (or to use its default when it cannot).
    // `--language xx` / `--task transcribe|translate` become this model's own `<|xx|>` / `<|task|>`
    // token ids, by TEXT rather than by numbers this CLI would otherwise carry per checkpoint.
    const auto special_id = [&](const std::string& piece, const std::string& flag,
                                 const std::string& hint) {
        const int32_t id = bpe_vocab ? bpe_vocab->piece_to_id(piece) : -1;
        if (id < 0) {
            throw loom::LoadError(flag + ": this model's vocabulary has no '" + piece + "' token. " + hint);
        }
        return id;
    };
    int32_t language = -1;
    if (!language_name.empty()) {
        language = special_id("<|" + language_name + "|>", "--language " + language_name,
                               "An English-only checkpoint has no language tokens at all; a "
                               "multilingual one names them by ISO code (en, de, fr, ...).");
    }
    int32_t task = -1;
    if (!task_name.empty()) {
        task = special_id("<|" + task_name + "|>", "--task " + task_name,
                           "Whisper names two: transcribe (same language out) and translate "
                           "(into English). An English-only checkpoint has neither.");
    }

    const std::vector<float> waveform = loom_cli::load_wav_pcm16_mono_16k(wav_path);

    loom::LoomLuaBridge bridge(backend);
    // A topology carrying ATTENTION nodes needs a KV cache to write into, sized from the model's own
    // declared geometry -- the same registration the generation path above already does, and missing
    // here until an ASR family turned up with a cached phase (Whisper's decoder, BACKLOG.md P4.1).
    // Without it `op_attention` throws and the file simply cannot be run through the ASR path.
    std::unique_ptr<loom::KvCache> kv_cache;
    for (const std::string& name : model.topology_names()) {
        loom::GraphTopology topo = loom::GraphTopology::parse(model.topology_json(name));
        if (topo.uses_kv_cache() && kv_cache == nullptr) {
            kv_cache = loom::make_kv_cache(model, backend);
        }
        loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
        bridge.register_module(name, model, std::move(topo), cache_for_module);
    }
    bridge.load_script(model.kv_str("model.driver_script"));

    // A model whose graph is built at ONE fixed clip length says so with `loom.n_samples` (Whisper: 30 s,
    // which is what it was trained on and what its encoder's every shape is a constant of). Absent, the
    // sequence length is genuinely dynamic and the whole file goes through in one call, which is the
    // NeMo families' shape.
    const uint32_t clip = model.has_kv("loom.n_samples") ? model.hparam_u32("n_samples") : 0;
    constexpr uint32_t kMaxNewTokensPerClip = 224; // half of Whisper's 448-token context, its own convention

    if (clip == 0) {
        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<double> length_d{static_cast<double>(waveform.size())};
        const loom::LoomLuaBridge::Value result = bridge.call(
            "infer", {{"waveform", waveform_d}, {"length", length_d}});
        const auto& ids_d = std::get<std::vector<double>>(result);
        std::vector<int32_t> token_ids;
        token_ids.reserve(ids_d.size());
        for (double id : ids_d) token_ids.push_back(static_cast<int32_t>(id));
        std::printf("transcript: %s\n", detokenize(token_ids).c_str());
        return;
    }

    // Fixed-clip models walk the file one `clip`-sample window at a time. WHERE the next window starts
    // is the whole question, and the answer is the model's own timestamps -- see the loop below.
    const uint32_t sample_rate = model.has_kv("loom.sample_rate") ? model.hparam_u32("sample_rate") : 0;
    const uint32_t n_audio_ctx = model.has_kv("loom.n_audio_ctx") ? model.hparam_u32("n_audio_ctx") : 0;
    const int32_t ts_base = bpe_vocab ? bpe_vocab->piece_to_id("<|0.00|>") : -1;
    // Seconds per timestamp token: one encoder frame, i.e. the clip's duration over the number of frames
    // it becomes. Derived from the file's own three numbers rather than hardcoded as Whisper's 0.02.
    const double ts_step = (ts_base >= 0 && sample_rate > 0 && n_audio_ctx > 0)
        ? (static_cast<double>(clip) / sample_rate) / n_audio_ctx : 0.0;
    const bool can_timestamp = ts_step > 0.0;
    const size_t n_clips = (waveform.size() + clip - 1) / clip;

    // Timestamps are REQUESTED whenever they are usable and there is more than one window, even if the
    // caller did not ask to see them: they are what the seek below advances on. A single-window file
    // without `--timestamps` keeps decoding in no-timestamps mode, so short audio behaves exactly as it
    // did -- forcing them would change its transcript for no benefit, since there is nothing to seek to.
    const bool want_timestamps = timestamps || (can_timestamp && n_clips > 1);
    if (n_clips > 1 && !can_timestamp) {
        std::fprintf(stderr, "note: this model exposes no timestamp tokens, so the %zu windows are cut "
                              "at a fixed %.0f s and each is decoded independently\n",
                     n_clips, static_cast<double>(clip) / (sample_rate ? sample_rate : 16000));
    }

    // Everything generated so far, carried into the next window as context (`<|startofprev|>`). The
    // driver takes the tail it has room for and filters it down to text; this just has to not grow
    // without bound, so it is capped at the text context the file declares -- comfortably more than the
    // half of it the driver will use.
    const size_t prev_cap = model.has_kv("loom.n_text_ctx") ? model.hparam_u32("n_text_ctx") : 448;
    std::vector<double> prev_tokens;

    std::vector<Segment> segments;
    size_t seek = 0;
    while (seek < waveform.size()) {
        const size_t avail = std::min(static_cast<size_t>(clip), waveform.size() - seek);
        std::vector<double> window(clip, 0.0); // zero-padded: Whisper's own pad_or_trim convention
        for (size_t i = 0; i < avail; ++i) window[i] = waveform[seek + i];

        const std::vector<int32_t> ids =
            run_driver(bridge, window, language, task, want_timestamps, prev_tokens,
                       kMaxNewTokensPerClip, eos_id);
        const double window_start = static_cast<double>(seek) / (sample_rate ? sample_rate : 16000);

        // How far into THIS window the model actually got. `last_complete_end` is the end of the last
        // segment it closed with a timestamp; text after that has no closing timestamp, which is the
        // model saying "this segment runs past the window edge".
        double last_complete_end = -1.0;
        const size_t before = segments.size();
        const double rate = sample_rate ? sample_rate : 16000;
        const double window_end = window_start + static_cast<double>(avail) / rate;
        if (can_timestamp && want_timestamps) {
            split_into_segments(ids, ts_base, ts_step, window_start, window_end, detokenize, segments,
                                 last_complete_end);
        } else {
            segments.push_back({window_start, window_end, detokenize(ids), /*closed=*/false});
        }

        // **This is the timestamp-aware part.** Advance to where the last complete segment ended rather
        // than by a fixed `clip`, so the next window begins on a boundary the model itself chose -- an
        // utterance cut in half by the window edge is re-decoded whole instead of being transcribed as
        // two fragments. Whisper's own long-form loop does exactly this.
        //
        // The fallback is the window edge, for the case where the model closed no segment at all -- then
        // there is no boundary it chose and a full stride is the only honest guess.
        //
        // **The final window is NOT special-cased, and an earlier version of this got that wrong.** It
        // looked reasonable to advance by the full stride once `avail < clip`, on the grounds that the
        // rest is padding -- but `avail < clip` only means the window is not FULL, and the audio inside
        // it is real. A model that closes at 23 s of a 20..45 s window has transcribed three seconds and
        // stopped; seeking to 23 gives the remaining twenty-two another decode, which is exactly what
        // re-seeking is for. Ending the loop there instead silently dropped them.
        size_t advance = clip;
        if (last_complete_end > window_start) {
            // Relative to THIS window: `last_complete_end` is a whole-file time, the seek is a sample
            // offset from the window's own start.
            //
            // Floored at one second, which is a guarantee of progress rather than a tuning knob: a model
            // that closes a zero-length or 0.02 s segment on a mostly-silent window would otherwise take
            // ~1500 iterations to cross it. Real segments are seconds long, so the floor is unreachable
            // in the case it is not protecting against.
            auto candidate = static_cast<size_t>((last_complete_end - window_start) * rate);
            candidate = std::max(candidate, static_cast<size_t>(rate));
            if (candidate <= clip) advance = candidate;
        }

        if (n_clips > 1) {
            std::fprintf(stderr, "  window at %.2fs: %zu tokens, %zu segment(s), advancing %.2fs\n",
                         window_start, ids.size(), segments.size() - before,
                         static_cast<double>(advance) / rate);
        }
        // Carry this window's output forward. Off (`--no-condition-on-previous`) it stays empty, which
        // is how the driver is told not to condition -- the same "omit the argument" convention the
        // optional inputs use everywhere else here.
        if (condition_on_previous) {
            for (int32_t id : ids) prev_tokens.push_back(static_cast<double>(id));
            if (prev_tokens.size() > prev_cap) {
                prev_tokens.erase(prev_tokens.begin(),
                                   prev_tokens.end() - static_cast<std::ptrdiff_t>(prev_cap));
            }
        }
        seek += advance;
    }

    if (timestamps) {
        for (const Segment& s : segments) {
            std::printf("[%s --> %s] %s\n", format_time(s.start).c_str(), format_time(s.end).c_str(),
                         s.text.c_str());
        }
        return;
    }
    std::string transcript;
    for (const Segment& s : segments) transcript += s.text;
    std::printf("transcript: %s\n", transcript.c_str());
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt_text;
    std::string wav_path;
    std::string language_name;
    std::string task_name;
    bool timestamps = false;
    bool condition_on_previous = true;
    bool has_prompt = false;
    bool has_wav = false;
    uint32_t n_predict = 16;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt_text = argv[++i];
            has_prompt = true;
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i];
            has_wav = true;
        } else if (arg == "--language" && i + 1 < argc) {
            // Optional by design. Omitted, a driver that can detect the language does; one that cannot
            // uses its own default. See run_asr.
            language_name = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            task_name = argv[++i];
        } else if (arg == "--timestamps") {
            timestamps = true;
        } else if (arg == "--no-condition-on-previous") {
            // On by default, as in Whisper's own CLI, and switchable for the same reason it is there:
            // carried context is what makes a sentence survive a window boundary, and it is also what
            // lets a repetition loop persist across one. With greedy decoding and no temperature
            // fallback to break out of such a loop, an off switch is the only recovery.
            condition_on_previous = false;
        } else if (arg == "--n-predict" && i + 1 < argc) {
            n_predict = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    if (!backend) {
        std::fprintf(stderr, "error: failed to initialize CPU backend\n");
        return 1;
    }

    try {
        auto model = loom::GgufModel::load(model_path, backend.get());
        std::printf("loaded '%s'\n", model_path.c_str());
        std::printf("  architecture: %s\n", model->architecture().c_str());

        bool is_multi_topology = model->has_kv("model.driver_script");
        if (is_multi_topology) {
            std::printf("  graph_topology: Multi-topology file (Lua driven), %zu sub-modules\n", 
                        model->topology_names().size());
        } else {
            std::printf("  graph_topology: %zu bytes of JSON\n", model->topology_json().size());
        }
        std::printf("  weights: %zu tensors\n", model->weights().size());

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "bert") {
            // WordPiece (BERT-family) models aren't causal LMs -- no generation loop applies, this is
            // inspection-only so loom_cli doesn't dead-end on a "bert" GGUF (see EXPORT-BACKLOG.md item 4).
            auto wp_vocab = loom::WordPieceVocab::load(*model);
            std::printf("  tokenizer: WordPiece (bert), %zu tokens\n", wp_vocab->size());
            if (has_prompt) {
                const auto ids = wp_vocab->encode(prompt_text);
                std::printf("  encode(\"%s\") -> [", prompt_text.c_str());
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
                std::printf("]\n");
            }
            return 0;
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "byt5") {
            // ByT5-family byte-level models: inspection-only, same reasoning as the "bert" branch above --
            // no generation loop is wired up for this model shape here.
            auto byte_vocab = loom::ByteVocab::load(*model);
            std::printf("  tokenizer: byte-level (byt5), %zu tokens\n", byte_vocab->size());
            if (has_prompt) {
                const auto ids = byte_vocab->encode(prompt_text);
                std::printf("  encode(\"%s\") -> [", prompt_text.c_str());
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
                std::printf("]\n");
            }
            return 0;
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "supertonic") {
            // SupertonicTTS's grapheme text front-end: inspection-only, same reasoning as the two branches
            // above. Falling THROUGH to the generation path below would have been the bug this branch
            // exists to prevent -- that path's `bpe_vocab` stays null for any non-"gpt2" tag, so a
            // supertonic GGUF's `--prompt` would have been parsed as literal token ids rather than
            // encoded, and the model's own vocabulary silently ignored.
            auto text_vec = loom::SupertonicTextVectorizer::load(*model);
            std::printf("  tokenizer: grapheme codepoints (supertonic), %zu tokens, default lang \"%s\"\n",
                        text_vec->n_tokens(), text_vec->default_lang().c_str());
            if (has_prompt) {
                const auto ids = text_vec->tokenize(prompt_text);
                std::printf("  encode(\"%s\") -> [", prompt_text.c_str());
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
                std::printf("]\n");
                // The one number that decides whether those ids are usable: every text-touching topology
                // in this export was traced at a FIXED length, so a caller has to match it exactly.
                if (model->has_kv("loom.txt_len")) {
                    const uint32_t txt_len = model->hparam_u32("txt_len");
                    std::printf("  loom.txt_len = %u%s\n", txt_len,
                                ids.size() == txt_len ? "" : "  <-- encoded length differs; pad or shorten");
                }
            }
            return 0;
        }

        if (has_prompt) {
            std::unique_ptr<loom::BpeVocab> bpe_vocab;
            if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "gpt2") {
                bpe_vocab = loom::BpeVocab::load(*model);
            }

            const std::vector<int32_t> prompt_tokens =
                bpe_vocab ? bpe_vocab->encode(prompt_text) : parse_token_ids(prompt_text);
            if (prompt_tokens.empty()) {
                std::fprintf(stderr, "error: --prompt produced no token ids\n");
                return 1;
            }

            if (is_multi_topology) {
                // Initialize the Lua JIT dynamic driver bridge
                loom::LoomLuaBridge bridge(backend.get());
                
                // Dynamically discover and register all sub-graph modules present in GGUF. A topology
                // carrying ATTENTION nodes needs a KV cache to write into (KV-CACHE.md stage 2), sized
                // from the model's own declared geometry -- the CLI asks the file rather than knowing
                // anything per-model, which is the point of declaring it there.
                std::unique_ptr<loom::KvCache> kv_cache;
                std::unique_ptr<loom::ConvStateCache> conv_state;
                const std::vector<std::string> sub_modules = model->topology_names();
                for (const std::string& mod_name : sub_modules) {
                    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(mod_name));
                    if (topo.uses_kv_cache() && kv_cache == nullptr) {
                        kv_cache = loom::make_kv_cache(*model, backend.get());
                    }
                    loom::KvCache* cache_for_module = topo.uses_kv_cache() ? kv_cache.get() : nullptr;
                    // A hybrid's ShortConv blocks carry their own history, which the KV cache does not
                    // hold -- allocated from the file's own loom.n_conv_* keys, same as above
                    // (BACKLOG.md P4.0.10).
                    if (topo.uses_conv_state() && conv_state == nullptr) {
                        conv_state = loom::make_conv_state_cache(*model, backend.get());
                    }
                    loom::ConvStateCache* conv_for_module = topo.uses_conv_state() ? conv_state.get() : nullptr;
                    bridge.register_module(mod_name, *model, std::move(topo), cache_for_module, conv_for_module);
                }
                
                // Load the master driver script
                bridge.load_script(model->kv_str("model.driver_script"));
                
                // Set up autoregressive prompt and loop
                std::vector<double> current_prompt;
                current_prompt.reserve(prompt_tokens.size() + n_predict);
                for (int32_t tok : prompt_tokens) {
                    current_prompt.push_back(static_cast<double>(tok));
                }
                
                std::printf("Running dynamic GGUF generation for %d tokens...\n", n_predict);
                std::vector<int32_t> generated;
                generated.reserve(n_predict);
                
                for (uint32_t step = 0; step < n_predict; ++step) {
                    loom::LoomLuaBridge::Value result = bridge.call("infer", {
                        {"tokens", current_prompt}
                    });
                    double next_tok_val = 0.0;
                    if (std::holds_alternative<double>(result)) {
                        next_tok_val = std::get<double>(result);
                    } else if (std::holds_alternative<std::vector<double>>(result)) {
                        const auto& vec = std::get<std::vector<double>>(result);
                        if (!vec.empty()) {
                            next_tok_val = vec[0];
                        }
                    }
                    int32_t next_tok = static_cast<int32_t>(next_tok_val);
                    if (next_tok < 0 || next_tok >= 65536) {
                        next_tok = 0; // Guard out-of-range token predictions
                    }
                    generated.push_back(next_tok);
                    current_prompt.push_back(static_cast<double>(next_tok));
                }
                
                if (bpe_vocab) {
                    std::printf("generated %zu tokens -> \"%s\"\n", generated.size(),
                                bpe_vocab->decode(generated).c_str());
                } else {
                    std::printf("generated %zu tokens:", generated.size());
                    for (int32_t tok : generated) std::printf(" %d", tok);
                    std::printf("\n");
                }
            } else {
                loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
                loom::GenerationConfig cfg;
                cfg.max_new_tokens = n_predict;
                cfg.n_ctx_max = static_cast<uint32_t>(prompt_tokens.size()) + n_predict;

                loom::Generator generator(*model, topo, cfg, backend.get());
                const std::vector<int32_t> generated = generator.generate(prompt_tokens);

                if (bpe_vocab) {
                    std::printf("generated %zu tokens -> \"%s\"\n", generated.size(),
                                bpe_vocab->decode(generated).c_str());
                } else {
                    std::printf("generated %zu tokens:", generated.size());
                    for (int32_t tok : generated) std::printf(" %d", tok);
                    std::printf("\n");
                }
            }
        }

        if (has_wav) {
            run_asr(*model, backend.get(), wav_path, language_name, task_name, timestamps,
                    condition_on_previous);
        }
    } catch (const loom::Error& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    } catch (const std::runtime_error& e) { // load_wav_pcm16_mono_16k throws this, not loom::Error
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
