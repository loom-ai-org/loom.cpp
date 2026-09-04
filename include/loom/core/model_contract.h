#pragma once

// What a GGUF declares about ITSELF: the task it performs and the I/O contract it performs it against.
//
// WHY THIS EXISTS. Until now a file said `loom.architecture` -- a per-MODEL name -- and nothing about
// what it does, so every host worked out what a model was by inference: `loom_cli` branched on
// `tokenizer.ggml.model` and dead-ended TTS files as "inspection-only", and loom-py could offer no
// end-to-end door at all without a table mapping architecture names to behaviour. Both are the
// per-architecture host code this project's three CLAUDE.md files forbid, arrived at because the file
// left no alternative (docs/HIGH-LEVEL-API.md §1).
//
// A declared contract removes the inference. `loom.task` names what the model is for, and
// `loom.input.kind` / `loom.output.kind` name the modality pair it maps between -- which IS the
// contract, and is what a host dispatches on. A host that reads them needs to know no architecture,
// and a model this engine has never heard of gets the same doors as one it has.
//
// EVERY FIELD IS OPTIONAL, AND ABSENCE IS NOT AN ERROR. Nothing exported before this existed declares
// any of it, and those files must keep working exactly as they did -- so each reader below falls back
// to what the engine already inferred, and `declared()` is how a host tells a file that states its
// contract from one the caller has to know about. That fallback is a migration measure with an end:
// when the fleet is re-exported it becomes dead weight, and removing it should be a deliberate commit
// rather than an accident.
//
// NAMING: the keys that already existed are NOT renamed. `loom.n_samples`, `loom.sample_rate`,
// `loom.txt_len`, `loom.n_audio_ctx` and `loom.n_text_ctx` are read here under their own names, because
// renaming a declared key costs a re-export of every model to buy nothing. `loom.sample_rate` needs no
// input/output qualifier for the same reason it never did: the modality pair says which side the audio
// is on. A model with audio on BOTH sides would need two, and none exists yet -- when one does, add
// `loom.output.sample_rate` and read it here rather than re-spelling this one.

#include "loom/core/gguf_model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

// The canonical `loom.task` names, mirroring `loom_exporter/tasks.py`'s vocabulary. Kept as strings
// rather than an enum on purpose: the engine branches on the modality PAIR, not on the task name, so an
// unknown task from a newer exporter must flow through here untouched instead of failing to parse.
namespace task_names {
inline constexpr const char* TEXT_GENERATION = "text-generation";
inline constexpr const char* ASR = "automatic-speech-recognition";
inline constexpr const char* TTS = "text-to-speech";
inline constexpr const char* AUDIO_CODEC = "audio-codec";
inline constexpr const char* TOKEN_CLASSIFICATION = "token-classification";
} // namespace task_names

// The modality names `loom.input.kind` / `loom.output.kind` take. Same reasoning: strings, open set.
// `token_ids` and `phoneme_ids` are inputs a caller supplies already-encoded -- the distinction from
// `text` is whether this file can do the encoding itself, which is exactly what a host needs to know
// before offering a text door.
namespace modality {
inline constexpr const char* TEXT = "text";
inline constexpr const char* TOKEN_IDS = "token_ids";
inline constexpr const char* PHONEME_IDS = "phoneme_ids";
inline constexpr const char* AUDIO = "audio";
inline constexpr const char* IMAGE = "image";
// Codec tokens, and NOT a flavour of `token_ids` (ADR-020). The fold below sends `token_ids` and
// `phoneme_ids` to "text" because text is what they encode -- a caller holding the string could
// produce them. A codec token encodes AUDIO: it is a compressed acoustic frame with no string behind
// it, and a file declaring `token_ids` here would advertise `text2speech` and be handed a sentence.
inline constexpr const char* AUDIO_CODES = "audio_codes";
inline constexpr const char* CLASS = "class";
inline constexpr const char* EMBEDDINGS = "embeddings";
} // namespace modality

struct ModelContract {
    // Empty when the file declares none, which is every GGUF exported before this existed.
    std::string task;
    // The modality pair. Empty when undeclared; see `infer_*` in the .cpp for what a legacy file gets.
    std::string input_kind;
    std::string output_kind;

    // Audio sample rate, whichever side the audio is on. 0 when the file names none.
    uint32_t sample_rate = 0;
    // The clip length this model's graph is built at, in samples; 0 means the length is dynamic, which
    // is the common case (every ASR family but Whisper). Reads `loom.n_samples`.
    uint32_t clip_samples = 0;
    // The fixed text axis a text-input graph was traced at; 0 when the axis is dynamic. Reads
    // `loom.txt_len`, which Supertonic declares and pads against.
    uint32_t max_input_tokens = 0;

    // "vocab" when this file can encode text with a vocabulary it embeds, "phonemes" when it takes
    // phoneme ids a G2P step produces outside the engine. Empty when undeclared -- in which case the
    // presence of `tokenizer.ggml.model` is the older, weaker signal a host had to use.
    std::string text_frontend;
    // "ipa", "arpabet", ... -- meaningful only for a phoneme front end.
    std::string phoneme_alphabet;
    // The phonemizer rule-set version this export was validated against. Phonemization rules live in
    // the engine and improve under a model by design (docs/HIGH-LEVEL-API.md §5), so a mismatch here is
    // the difference between an output change a user can attribute and one that is a mystery. It is a
    // WARNING, never a failure: the rules are a superset of what these checkpoints were trained on and
    // a newer set is expected to be an improvement.
    std::string phonemizer_ruleset;
    // Language codes this model handles, when it says.
    std::vector<std::string> languages;
    // The Lua functions the driver defines, when the export lists them. Empty means "ask the driver",
    // which in practice means `infer` and nothing else.
    std::vector<std::string> entry_points;

    // Sampler steps to use when a caller names none, for the flow-matching/diffusion TTS families.
    uint32_t default_steps = 0;
    // Named voices the file carries.
    std::vector<std::string> voices;

    // The class names a `class`-output model chooses between, INDEXED BY the id its driver returns --
    // `loom.labels`. Empty for every model that is not a classifier, and legitimately empty for one
    // that names its classes nothing, in which case the id is the only answer there is.
    //
    // It is in the FILE rather than here by the tier-0 admission test (docs/HIGH-LEVEL-API.md §2):
    // only the checkpoint knows that class 3 is `B-PER`, and a host has to branch on it to say anything
    // a person can read. Written as one id-indexed string array rather than parallel name/id arrays --
    // unlike the ASR language table, which is a sparse map into a vocabulary -- because a classifier's
    // classes are 0..n-1 by construction.
    std::vector<std::string> labels;

    // Reads whatever `model` declares. Never throws for an absent key -- see the header comment.
    static ModelContract read(const GgufModel& model);

    // Whether this file states its own contract, as opposed to having one worked out for it. A host
    // should offer a task-shaped door only when this is true; guessing is what this file exists to stop.
    bool declared() const { return !task.empty() && !input_kind.empty() && !output_kind.empty(); }

    // "speech2text", "text2speech", ... -- the modality pair as one name, which is the interface a host
    // exposes. Empty when the pair is not declared. `token_ids`/`phoneme_ids` fold onto their natural
    // modality here (both are "text" in), because they are how you SUPPLY text to a model that cannot
    // encode it, not a different contract.
    std::string interface_name() const;
};

// Forward-declared at `loom` scope rather than inside `loom::audio` below, where `class BpeVocab*` in a
// parameter list would have declared a DIFFERENT, never-defined `loom::audio::BpeVocab`.
class BpeVocab;

namespace audio {

// The vocabulary facts a transcription loop needs, which are per-CHECKPOINT and so belong in the file.
//
// This is what stops `transcribe` being Whisper-shaped. Its constants used to be spelled in C++ --
// `piece_to_id("<|0.00|>")`, `"<|" + language + "|>"`, `<|notimestamps|>` -- which is Whisper's
// convention rather than ASR's, so the second timestamped family (Canary, Qwen3-ASR, Granite-Speech,
// each of which spells all three differently) would have cost engine code in an engine whose rule is
// that a family costs Python in the exporter.
//
// `read` falls back to those spellings when the file declares nothing, because no GGUF declares this
// yet and the ones on disk must keep transcribing. The fallback is Whisper-only and says so.
struct AsrDecodeTable {
    // First timestamp token id, and the seconds one timestamp step covers. `timestamp_first_id < 0`
    // means this model emits no timestamps at all, which is a real answer: the CTC and transducer
    // families do not, and their transcripts are window slices rather than boundaries a model chose.
    int32_t timestamp_first_id = -1;
    double timestamp_step_sec = 0.0;

    // Ids to drop before detokenizing -- end-of-sequence, `<|notimestamps|>`, anything else the model
    // emits that is a statement about the decode rather than a word that was spoken.
    std::vector<int32_t> control_ids;

    // Parallel arrays rather than a map, because that is what GGUF stores: `loom.asr.language_names`
    // and `loom.asr.language_ids` are one str[] and one i32[] of the same length.
    std::vector<std::string> language_names;
    std::vector<int32_t> language_ids;
    std::vector<std::string> task_names;
    std::vector<int32_t> task_ids;

    // How many of the previous window's tokens may be carried forward as `prev_tokens`.
    uint32_t prev_context = 0;

    // True when this file declared no language/task table and a caller must fall back to resolving a
    // name against the vocabulary by Whisper's `<|xx|>` spelling. Set only when a vocabulary was
    // available to fall back to. It exists so the one remaining spelled lookup is visible at its call
    // site instead of being a silent second path inside `language()`.
    bool legacy_spelling = false;

    bool timestamped() const { return timestamp_first_id >= 0 && timestamp_step_sec > 0.0; }

    // The id for a language/task NAME, or -1 when this model has none by that name. Resolution belongs
    // to the file for the reason the whole of this header exists: `<|en|>` is a vocabulary entry, and a
    // caller has no way to look up its id.
    int32_t language(const std::string& name) const;
    int32_t task(const std::string& name) const;

    // `vocab` is used only by the legacy fallback and may be null; a file that declares its table needs
    // no vocabulary to be read at all.
    static AsrDecodeTable read(const GgufModel& model, const loom::BpeVocab* vocab,
                               const ModelContract& contract);
};

} // namespace audio
} // namespace loom
