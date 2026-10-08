#include "loom/core/audio_classify.h"

#include "loom/loom_errors.h"

#include <string>
#include <unordered_map>

namespace loom {
namespace audio {
namespace {

// The one driver call both doors make. `length` beside the waveform is the family-1 encoders' second
// input (TitaNet's and MarbleNet's); a driver that takes the waveform alone (ECAPA, pyannote) never
// reads it, and a clip handed over whole is all real audio, so it is the sample count.
std::vector<double> run(LoomLuaBridge& bridge, const GgufModel& model, const std::vector<float>& samples,
                        const char* door) {
    if (!model.has_kv("model.driver_script")) {
        throw LoadError(std::string(door) + ": model carries no driver_script; it can be inspected but "
                        "not run.");
    }
    if (samples.empty()) {
        throw Error(std::string(door) + ": no audio. There is nothing to classify in an empty clip.");
    }
    std::unordered_map<std::string, LoomLuaBridge::Value> args;
    args["waveform"] = std::vector<double>(samples.begin(), samples.end());
    args["length"] = std::vector<double>{static_cast<double>(samples.size())};
    const LoomLuaBridge::Value result = bridge.call("infer", args);
    if (!std::holds_alternative<std::vector<double>>(result)) {
        throw Error(std::string(door) + ": the driver returned a single number where an array was "
                    "expected.");
    }
    return std::get<std::vector<double>>(result);
}

void require(const ModelContract& contract, const char* kind, const char* door) {
    if (contract.input_kind != modality::AUDIO || contract.output_kind != kind) {
        throw Error(std::string(door) + ": this model declares " +
                    (contract.declared() ? contract.interface_name() : std::string("no contract")) +
                    ", not audio in and `" + kind + "` out.");
    }
}

} // namespace

ClassProbabilities classify(LoomLuaBridge& bridge, const GgufModel& model, const std::vector<float>& samples) {
    const ModelContract contract = ModelContract::read(model);
    require(contract, modality::CLASS, "classify");
    const std::string& granularity = contract.output_granularity;
    if (granularity != granularity::FRAME && granularity != granularity::CLIP) {
        throw Error("classify: this model's classes are per `" + granularity + "`; an audio door "
                    "answers per frame or per clip.");
    }
    if (contract.labels.empty()) {
        // Without the label count a flat answer cannot be cut into rows, and guessing it from the size
        // is exactly the inference the contract exists to remove.
        throw Error("classify: this model names no labels (`loom.labels`), so its answer cannot be "
                    "split into one row per frame.");
    }

    const std::vector<double> raw = run(bridge, model, samples, "classify");
    const size_t width = contract.labels.size();
    if (raw.empty() || raw.size() % width != 0) {
        throw Error("classify: the driver returned " + std::to_string(raw.size()) + " numbers, which is "
                    "not a whole number of rows of the " + std::to_string(width) + " declared labels.");
    }
    ClassProbabilities out;
    out.granularity = granularity;
    out.labels = contract.labels;
    out.n_rows = static_cast<uint32_t>(raw.size() / width);
    if (granularity == granularity::CLIP && out.n_rows != 1) {
        throw Error("classify: a clip classifier returned " + std::to_string(out.n_rows) + " rows.");
    }
    if (granularity == granularity::FRAME) {
        out.frame_rate = contract.frame_rate;
        out.frame_offset = contract.frame_offset;
    }
    out.probabilities.assign(raw.begin(), raw.end());
    return out;
}

Embeddings embed(LoomLuaBridge& bridge, const GgufModel& model, const std::vector<float>& samples) {
    const ModelContract contract = ModelContract::read(model);
    require(contract, modality::EMBEDDINGS, "embed");
    const std::string& granularity = contract.output_granularity;
    if (granularity != granularity::FRAME && granularity != granularity::CLIP) {
        throw Error("embed: this model's embeddings are per `" + granularity + "`; an audio door "
                    "answers per frame or per clip.");
    }
    if (granularity == granularity::FRAME && contract.embedding_dim == 0) {
        // The same refusal as `classify`'s missing labels, for the same reason: the frame count is not
        // the host's to derive, so without the width there is no cut that is not a guess. Every frame
        // file from an exporter older than the key is one of these; re-exporting it adds the key.
        throw Error("embed: this model's embeddings are per frame but it declares no width "
                    "(`loom.output.embedding_dim`), so its answer cannot be split into one row per "
                    "frame. Re-export it with a current loom-exporter, or call `infer` for the flat "
                    "answer.");
    }

    const std::vector<double> raw = run(bridge, model, samples, "embed");
    Embeddings out;
    out.granularity = granularity;
    // A clip file with no declared width is one row of whatever came back -- TitaNet's and ECAPA's
    // published files predate the key, and a clip answer has nothing to cut.
    out.dim = contract.embedding_dim != 0 ? contract.embedding_dim : static_cast<uint32_t>(raw.size());
    if (out.dim == 0 || raw.size() % out.dim != 0) {
        throw Error("embed: the driver returned " + std::to_string(raw.size()) + " numbers, which is "
                    "not a whole number of rows of the declared width " + std::to_string(out.dim) + ".");
    }
    out.n_rows = static_cast<uint32_t>(raw.size() / out.dim);
    if (granularity == granularity::CLIP && out.n_rows != 1) {
        throw Error("embed: a clip embedder returned " + std::to_string(out.n_rows) + " rows of the "
                    "declared width " + std::to_string(out.dim) + ".");
    }
    if (granularity == granularity::FRAME) {
        out.frame_rate = contract.frame_rate;
        out.frame_offset = contract.frame_offset;
    }
    out.values.assign(raw.begin(), raw.end());
    return out;
}

} // namespace audio
} // namespace loom
