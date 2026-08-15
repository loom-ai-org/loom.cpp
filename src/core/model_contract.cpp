#include "loom/core/model_contract.h"

#include "loom/core/bpe_vocab.h"

#include <algorithm>

namespace loom {
namespace {

// Every reader below is absence-tolerant, because a GGUF exported before the contract existed declares
// none of these and must keep working (model_contract.h's header comment).
std::string opt_str(const GgufModel& model, const char* bare_key) {
    return model.has_kv(std::string("loom.") + bare_key) ? model.hparam_str(bare_key) : std::string{};
}

uint32_t opt_u32(const GgufModel& model, const char* bare_key) {
    return model.has_kv(std::string("loom.") + bare_key) ? model.hparam_u32(bare_key) : 0u;
}

double opt_f32(const GgufModel& model, const char* bare_key) {
    return model.has_kv(std::string("loom.") + bare_key)
        ? static_cast<double>(model.hparam_f32(bare_key)) : 0.0;
}

std::vector<std::string> opt_arr_str(const GgufModel& model, const char* full_key) {
    return model.has_kv(full_key) ? model.kv_arr_str(full_key) : std::vector<std::string>{};
}

std::vector<int32_t> opt_arr_i32(const GgufModel& model, const char* full_key) {
    return model.has_kv(full_key) ? model.kv_arr_i32(full_key) : std::vector<int32_t>{};
}

// The modality a name folds onto for interface naming. `token_ids` and `phoneme_ids` are both TEXT in:
// they are how a caller supplies text to a model that cannot encode it, not a different contract, and
// a host that offered `Tokens2Speech` alongside `Text2Speech` would be naming one door twice.
std::string interface_side(const std::string& kind) {
    if (kind == modality::AUDIO) return "speech";
    if (kind == modality::TEXT || kind == modality::TOKEN_IDS || kind == modality::PHONEME_IDS) {
        return "text";
    }
    return kind; // image, class, embeddings, and whatever a newer exporter declares
}

} // namespace

ModelContract ModelContract::read(const GgufModel& model) {
    ModelContract c;
    c.task = opt_str(model, "task");
    c.input_kind = opt_str(model, "input.kind");
    c.output_kind = opt_str(model, "output.kind");

    // Pre-existing keys, read under their own names -- renaming a declared key costs a re-export of
    // every model and buys nothing.
    c.sample_rate = opt_u32(model, "sample_rate");
    c.clip_samples = opt_u32(model, "n_samples");
    c.max_input_tokens = opt_u32(model, "txt_len");

    c.text_frontend = opt_str(model, "text.frontend");
    c.phoneme_alphabet = opt_str(model, "text.phoneme_alphabet");
    c.phonemizer_ruleset = opt_str(model, "phonemizer.ruleset");
    c.languages = opt_arr_str(model, "loom.text.languages");
    c.entry_points = opt_arr_str(model, "loom.entry_points");

    c.default_steps = opt_u32(model, "tts.default_steps");
    c.voices = opt_arr_str(model, "loom.tts.voices");

    // A file that declares no front end but embeds a vocabulary can encode text, and saying so here is
    // not a guess -- the vocabulary either is in the file or is not. This is the ONE inference kept,
    // because it is about the file's own contents rather than about which architecture it is.
    if (c.text_frontend.empty() && model.has_kv("tokenizer.ggml.model")) c.text_frontend = "vocab";
    return c;
}

std::string ModelContract::interface_name() const {
    if (input_kind.empty() || output_kind.empty()) return {};
    return interface_side(input_kind) + "2" + interface_side(output_kind);
}

namespace audio {

int32_t AsrDecodeTable::language(const std::string& name) const {
    for (size_t i = 0; i < language_names.size() && i < language_ids.size(); ++i) {
        if (language_names[i] == name) return language_ids[i];
    }
    return -1;
}

int32_t AsrDecodeTable::task(const std::string& name) const {
    for (size_t i = 0; i < task_names.size() && i < task_ids.size(); ++i) {
        if (task_names[i] == name) return task_ids[i];
    }
    return -1;
}

AsrDecodeTable AsrDecodeTable::read(const GgufModel& model, const loom::BpeVocab* vocab,
                                    const ModelContract& contract) {
    AsrDecodeTable t;
    t.prev_context = model.has_kv("loom.asr.prev_context") ? model.hparam_u32("asr.prev_context")
                                                           : opt_u32(model, "n_text_ctx");
    t.control_ids = opt_arr_i32(model, "loom.asr.control_ids");
    t.language_names = opt_arr_str(model, "loom.asr.language_names");
    t.language_ids = opt_arr_i32(model, "loom.asr.language_ids");
    t.task_names = opt_arr_str(model, "loom.asr.task_names");
    t.task_ids = opt_arr_i32(model, "loom.asr.task_ids");

    const bool declares_timestamps = model.has_kv("loom.asr.timestamp_first_id");
    if (declares_timestamps) {
        t.timestamp_first_id = model.kv_i32("loom.asr.timestamp_first_id", -1);
        t.timestamp_step_sec = opt_f32(model, "asr.timestamp_step_sec");
    }

    // ---- LEGACY FALLBACK, Whisper-only, for files exported before any of the above existed. ----
    //
    // Every branch here is keyed on a token SPELLING, which is the thing the declared table exists to
    // stop -- so each is guarded on the corresponding key being absent, and the whole block becomes
    // dead once the fleet is re-exported. Deleting it should be its own commit, not a side effect.
    if (vocab == nullptr) return t;

    if (!declares_timestamps) {
        t.timestamp_first_id = vocab->piece_to_id("<|0.00|>");
        // Seconds per timestamp token: one encoder frame, i.e. the clip's duration over the number of
        // frames it becomes. Derived from the file's own three numbers rather than hardcoded as 0.02.
        const uint32_t n_audio_ctx = model.has_kv("loom.n_audio_ctx") ? model.hparam_u32("n_audio_ctx") : 0;
        if (t.timestamp_first_id >= 0 && contract.sample_rate > 0 && n_audio_ctx > 0 &&
            contract.clip_samples > 0) {
            t.timestamp_step_sec =
                (static_cast<double>(contract.clip_samples) / contract.sample_rate) / n_audio_ctx;
        }
    }
    // Language and task names are NOT resolved here, because the fallback cannot be: enumerating a
    // hundred `<|xx|>` pieces to build a table is exactly the Whisper archaeology this replaces. The
    // flag says "ask the vocabulary directly", and `transcribe.cpp` is the single place that does.
    t.legacy_spelling = t.language_names.empty();
    if (t.control_ids.empty()) {
        // `<|notimestamps|>` is the model stating something about the decode rather than a word that
        // was spoken, and it prints as its literal spelling if it survives to the detokenizer. The
        // end-of-sequence id is added by the caller, which knows it from `tokenizer.ggml.eos_token_id`
        // for every vocabulary family rather than just this one.
        const int32_t no_ts = vocab->piece_to_id("<|notimestamps|>");
        if (no_ts >= 0) t.control_ids.push_back(no_ts);
    }
    return t;
}

} // namespace audio
} // namespace loom
