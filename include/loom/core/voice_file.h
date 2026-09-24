#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// A VOICE FILE: a small GGUF whose tensors are driver inputs by name, for a model that takes its voice
// as data rather than as a reference clip (ADR-045). Pocket-TTS's are its flow LM's saved KV cache
// (`voice_kv`, which the driver seeds with `loom.seed_kv`); the format says nothing about that, which is
// what lets `loom_cli --voice` and loom-py's `voice=` load any model's voices with no per-model code.
//
// **A voice only fits the weights it was made with**, so a file declares `loom.voice.architecture` and
// `loom.voice.compat` (a fingerprint the exporter takes of those weights), and the MODEL declares the
// same `loom.voice.compat`. `load_voice` refuses any mismatch by name: a voice state fed to other
// weights still produces audio, and the reference's own note on it is that it "typically never emits
// EOS".
struct VoiceFile {
    std::string name;
    std::string license;
    std::string origin;
    // Tensor name -> its values, row-major: pass each as the driver input of that name.
    std::unordered_map<std::string, std::vector<double>> inputs;
};

// Throws `SchemaError` when `model` takes no voice files or the file was made for other weights, and
// `LoadError` when the file is not a voice file or holds a tensor that is not F32.
VoiceFile load_voice(const GgufModel& model, const std::string& path);

} // namespace loom
