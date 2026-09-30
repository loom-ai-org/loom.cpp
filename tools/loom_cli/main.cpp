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
#include "loom/core/transcribe.h"
#include "loom/core/conv_state_cache.h"
#include "codes_file.h"
#include "wav_file.h"


#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <memory>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                  "usage: %s --model <path.gguf> --prompt \"<text or token ids>\" [--n-predict N]\n"
                  "       %s --model <asr.gguf> --wav <path.wav> [--language en] "
                  "[--task transcribe|translate] [--timestamps] "
                  "[--no-condition-on-previous]\n"
                  "       %s --model <tts-or-codec.gguf> --prompt \"<phonemes|text|codes>\" "
                  "--out out.wav\n"
                  "       %s --model <voice-cloning-tts.gguf> --wav <ref.wav> "
                  "--ref-text \"<what it says>\" --prompt \"<text>\" --out out.wav\n"
                  "       %s --model <text-to-codes.gguf> --prompt \"<text>\" --codec <codec.gguf> "
                  "--out out.wav\n"
                  "\n"
                  "  --chat                        wrap --prompt in the model's own chat template\n"
                  "  --system <text>               a system turn ahead of it (implies --chat)\n"
                  "  --temperature F --top-k N --top-p F   sample instead of taking the argmax;\n"
                  "                                omitted, each falls back to what the checkpoint\n"
                  "                                declared, which for most models is greedy\n"
                  "  --seed N                      make a sampled generation reproducible\n"
                  "\n"
                  "  --out <path.wav>              for a model whose answer is AUDIO: synthesise and\n"
                  "                                write it. --prompt is the text, the IPA phonemes or\n"
                  "                                the codes, depending on what the file declares\n"
                  "  --voice <voice.gguf>          a voice file for a model that takes one (pocket-tts's,\n"
                  "                                cosyvoice3's and voxtral-tts's `voices/*.gguf`); refused if made for\n"
                  "                                other weights\n"
                  "  --codec <codec.gguf>          for a model whose answer is CODEC TOKENS (dia,\n"
                  "                                qwen3-tts, moss-tts): the codec that decodes them\n"
                  "                                to --out. --n-predict caps FRAMES; --language picks\n"
                  "                                one the file declares; --wav (and --ref-text) is\n"
                  "                                the clip a cloning model takes its voice from\n"
                  "  --codes-out <codes.gguf>      ... and/or write the codes; a codec decodes the file\n"
                  "                                later with --prompt @codes.gguf\n"
                  "  --input <name=1,2,3|@file>    an extra driver input: kokoro's `ref_s`, matcha's\n"
                  "                                `n_steps`, styletts2's `diffusion_steps`, or\n"
                  "                                `sample_rate=N` for a model that declares none\n"
                  "  --device <auto|cpu|gpu|NAME>  where to run (default: auto, or $LOOM_DEVICE)\n"
                  "  --list-devices                print the devices this build can reach, and exit\n"
                  "\n"
                  "  $LOOM_PROFILE=1               time every graph node and print a per-op breakdown\n"
                  "  $LOOM_PROFILE=<path>          ... to a file instead of stderr\n"
                  "  $LOOM_PROFILE_NODES=1         ... and a second table keyed on the NODE name, which\n"
                  "                                is the only thing that says which graph a bucket is in\n"
                  "                                (profile with ONE thread; see include/loom/core/profile.h)\n",
                  argv0, argv0, argv0, argv0, argv0);
}

// What ran where, after a device run. The number that matters is the split count: each split is a point
// at which execution crossed between the device and the CPU fallback, and every crossing is a copy in
// each direction. A module reported as 1 split ran entirely on one backend.
// `1,2,3` or `@path` (whitespace- or comma-separated numbers in a file). The file form is what makes
// Kokoro reachable at all: its `ref_s` is 256 floats chosen per voice, which belongs in a file rather
// than on a command line.
std::vector<double> read_number_spec(const std::string& spec) {
    std::string text = spec;
    if (!spec.empty() && spec[0] == '@') {
        std::ifstream in(spec.substr(1));
        if (!in) throw std::runtime_error("could not read '" + spec.substr(1) + "'");
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    for (char& c : text) {
        if (c == ',' || c == ';' || c == '\n' || c == '\t' || c == '\r') c = ' ';
    }
    std::vector<double> out;
    std::istringstream stream(text);
    double value = 0.0;
    while (stream >> value) out.push_back(value);
    return out;
}

void print_device_report(const loom::LoomLuaBridge& bridge);

// One driver input as the command line gathered it. `array` is what the bridge must be handed, and it
// cannot be read off the length: `--input n_steps=32` is ONE number and means a number, while a voice
// file's `reference_frames` for a single reference is a one-element TENSOR the driver takes the length
// of. Collapsing every one-element list to a scalar handed MOSS-TTS's driver the number 137 where it
// expected `{137}`, and `#_frames` failed on the first one-reference voice run through this CLI.
struct DriverInput {
    std::string name;
    std::vector<double> values;
    bool array = false;
};
using DriverInputs = std::vector<DriverInput>;

void set_inputs(std::unordered_map<std::string, loom::LoomLuaBridge::Value>& inputs, const DriverInputs& extra) {
    for (const auto& [name, values, array] : extra) {
        if (name == "sample_rate") continue;               // ours, not the driver's
        if (values.size() == 1 && !array) {
            inputs[name] = values[0];
        } else {
            inputs[name] = values;
        }
    }
}

// `--input sample_rate=N`, for the three models that declare none of their own.
uint32_t rate_override(const DriverInputs& extra) {
    for (const auto& [name, values, array] : extra) {
        if (name == "sample_rate" && !values.empty()) return static_cast<uint32_t>(values[0]);
    }
    return 0;
}

// Runs a model whose OUTPUT KIND is audio and writes the waveform.
//
// **This is the half of the CLI that did not exist**, and its absence was not a small gap: with two
// output modes -- a transcript and tokens -- every TTS and codec family in the zoo was unreachable
// from here, on any device. `scripts/tts_synth.cpp` has done it for measurement since P4.13, per
// family, with the input names hard-coded; what is different here is that the driver's own declared
// primary input is used (`tokens`, which every synthesized driver aliases) and anything else the
// family needs arrives through `--input`.
//
// The sample rate is the FILE's when it declares one. VITS and the two LJSpeech models do not, so a
// rate is required from the caller rather than assumed -- a wrong rate does not fail, it plays the
// audio at the wrong speed, which is the failure this refuses to make silently.
int synthesize(loom::GgufModel& model, const loom::Backends& backends,
                const std::vector<int32_t>& ids, const std::string& input_name,
                const DriverInputs& extra,
                const std::string& out_path, uint32_t rate_override, uint32_t seed) {
    uint32_t rate = model.has_kv("loom.sample_rate") ? model.hparam_u32("sample_rate") : 0;
    if (rate == 0) rate = rate_override;
    if (rate == 0) {
        std::fprintf(stderr,
                     "this model declares no `loom.sample_rate`, so the rate has to come from you: "
                     "pass --input sample_rate=22050 (vits/matcha) or 24000 (kokoro/styletts2). A "
                     "wrong rate plays the audio at the wrong speed rather than failing.\n");
        return 1;
    }

    loom::Session session(model, backends);
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    std::vector<double> as_doubles(ids.begin(), ids.end());
    inputs[input_name] = as_doubles;
    // Every phoneme-input TTS family draws noise and reads `inputs.seed` for it -- VITS's z and its
    // stochastic duration predictor, StyleTTS2's diffusion, the flow-matching z0 -- and a driver that
    // does not name it simply ignores the entry. Defaulted rather than required, because a CLI that
    // refuses to speak until you pick a random number is asking the wrong question; `--seed` sets it.
    inputs["seed"] = static_cast<double>(seed);
    set_inputs(inputs, extra);

    const auto result = session.bridge().call("infer", inputs);
    const auto* samples = std::get_if<std::vector<double>>(&result);
    if (samples == nullptr) {
        std::fprintf(stderr, "this model's driver returned a single number, not a waveform\n");
        return 1;
    }
    std::vector<float> audio(samples->begin(), samples->end());
    double peak = 0.0, sum_sq = 0.0;
    for (float v : audio) { peak = std::max(peak, std::abs(static_cast<double>(v))); sum_sq += v * v; }
    // Peak and rms, because they have caught something: real speech lands near +-0.3, and audio that
    // leaves [-1, 1] means the conditioning is wrong rather than the vocoder (Retro-006).
    // Interleaved channels, read off the file: a stereo codec returns `L R L R ...`, and dividing its
    // length by the rate alone would report twice the duration and write a mono file at half speed.
    const uint32_t channels = model.has_kv("loom.channels")
                                  ? std::max<uint32_t>(model.hparam_u32("channels"), 1) : 1;
    std::printf("  %zu samples x %u channel(s) at %u Hz = %.2f s, peak %.4f, rms %.4f\n",
                audio.size() / channels, channels, rate,
                static_cast<double>(audio.size()) / channels / rate, peak,
                std::sqrt(sum_sq / std::max<size_t>(audio.size(), 1)));
    loom_cli::write_wav_pcm16(out_path, audio, rate, channels);
    std::printf("  wrote %s\n", out_path.c_str());
    print_device_report(session.bridge());
    return 0;
}

// Whether the file's own driver reads `inputs.<name>`. A driver ignores an input it does not name, so
// without this a `--wav` handed to a model that takes no clip would be dropped and the audio would
// come back in the default voice with nothing said -- the silent substitution Retro-006 is about. The
// drivers are generated, and every one spells an input read as `inputs.<name>`.
bool driver_reads(const loom::GgufModel& model, const std::string& name) {
    if (!model.has_kv("model.driver_script")) return false;
    const std::string script = model.kv_str("model.driver_script");
    const std::string dotted = "inputs." + name;
    for (size_t at = script.find(dotted); at != std::string::npos; at = script.find(dotted, at + 1)) {
        const size_t end = at + dotted.size();
        if (end == script.size() || !(std::isalnum(static_cast<unsigned char>(script[end])) || script[end] == '_')) {
            return true;
        }
    }
    return false;
}

// Text to ids through whichever vocabulary a text-to-codes file embeds. The three that ship use two
// tags -- Dia's byte-level `byt5`, Qwen3-TTS's and MOSS-TTS's Qwen2 `gpt2` -- and the SentencePiece
// pair is here because it costs one line and is the next thing an AR codec LM would carry. These are
// the same vocabularies loom-py's `tokenize` dispatches to, so the ids are the ones its
// `text2codes.infer(text)` sends.
std::vector<int32_t> encode_for_codes(const loom::GgufModel& model, const std::string& text,
                                      std::string* described) {
    const std::string tag = model.has_kv("tokenizer.ggml.model") ? model.kv_str("tokenizer.ggml.model") : "";
    if (tag == "byt5") {
        auto vocab = loom::ByteVocab::load(model);
        if (described) *described = "byte-level (byt5), " + std::to_string(vocab->size()) + " tokens";
        return vocab->encode(text);
    }
    if (tag == "gpt2") {
        auto vocab = loom::BpeVocab::load(model);
        if (described) *described = "byte-level BPE, " + std::to_string(vocab->size()) + " tokens";
        return vocab->encode(text);
    }
    if (tag == "t5" || tag == "llama") {
        auto vocab = loom::Vocab::load(model);
        if (described) *described = "SentencePiece (" + tag + "), " + std::to_string(vocab->size()) + " pieces";
        return vocab->encode(text);
    }
    throw loom::SchemaError("this text-to-codes model's vocabulary is tagged '" + tag +
                            "', which loom_cli cannot encode text with; pass token ids through loom-py's "
                            "text2codes.infer(tokens=[...])");
}

// Why rows `width` wide cannot go to a codec `codec_width` wide, or "" when they can: no wider, and
// narrower only when the codec declares the id that means "codebook absent" -- a residual quantizer
// decodes its first k codebooks as a prefix of the full sum (ADR-050). One rule for both doors that
// feed a codec here, the pair and a codes file.
std::string pairing_error(uint32_t width, uint32_t codec_width, const std::optional<uint32_t>& absent) {
    if (codec_width == 0) return "declares no `loom.codec.n_codebooks`";
    if (width > codec_width) return "decodes only " + std::to_string(codec_width) + ". They are not a pair.";
    if (width < codec_width && !absent) {
        return "decodes " + std::to_string(codec_width) +
               " and declares no `codec.absent_code` to pad the rest with. They are not a pair.";
    }
    return {};
}

// Frame-major rows `width` wide, each filled out to `codec_width` with the absent id. Call only after
// `pairing_error` said yes.
std::vector<int32_t> widen_rows(const std::vector<int32_t>& codes, uint32_t width, uint32_t codec_width,
                                const std::optional<uint32_t>& absent) {
    if (width == codec_width) return codes;
    std::vector<int32_t> rows;
    rows.reserve(codes.size() / width * codec_width);
    for (size_t f = 0; f < codes.size() / width; ++f) {
        for (uint32_t c = 0; c < codec_width; ++c) {
            rows.push_back(c < width ? codes[f * width + c] : static_cast<int32_t>(*absent));
        }
    }
    return rows;
}

// What a `text2codes` run needs beyond the model: the flags, gathered so the function below does not
// take fifteen arguments.
struct Text2CodesRequest {
    std::string prompt;
    bool has_prompt = false;
    std::string wav_path;               // a reference clip, for a driver that reads `waveform`
    std::string ref_text;               // what that clip says, for a driver that replays it (ICL)
    bool has_ref_text = false;
    std::string language;
    std::optional<uint32_t> max_frames; // `--n-predict`, only when given
    loom::text::GenerateOptions sampling;
    uint32_t seed = 1234;
    DriverInputs extra;
    std::string codec_path;
    std::string codes_out;
    std::string out_wav;
};

// **Text in, codec tokens out, and the codec that turns them into audio.** Dia, Qwen3-TTS and MOSS-TTS
// are two files each (ADR-022): an AR LM that emits frames of codes and a codec that decodes them.
// Until this existed they ran only through loom-py, because the LM fell through to the text generator
// below (Qwen3-TTS and MOSS-TTS are `gpt2` files) or to the byt5 inspection branch (Dia).
//
// The shape is loom-py's `text2codes.infer` then `codes2speech.infer`, in one process:
//
//   * the codec's WIDTH is checked from its header before a single frame is generated. MOSS-TTS emits
//     12 of MOSS-Audio-Tokenizer's 32 codebooks, and a codec that declares `codec.absent_code`
//     decodes a narrower row as a prefix of its residual sum (ADR-050); one that does not cannot, and
//     finding that out after minutes of generation would be the wrong order;
//   * the LM is RELEASED before the codec loads. MOSS-TTS is 16 GB at F32 and its codec 4 GB, and the
//     two are never needed at once;
//   * `--codes-out` writes the frames as a codes GGUF (codes_file.h), because the codes are the
//     return value rather than an implementation detail -- a caller may cache them or decode them
//     elsewhere, and the codec branch below reads the file back with its width.
int text_to_codes(std::unique_ptr<loom::GgufModel>& model, const loom::ModelContract& contract,
                  const loom::Backends& backends, const Text2CodesRequest& req) {
    const uint32_t width = model->has_kv("loom.codec.n_codebooks") ? model->hparam_u32("codec.n_codebooks") : 0;
    if (width == 0) {
        std::fprintf(stderr, "this model declares no `loom.codec.n_codebooks`, so its output cannot be cut "
                             "into frames; re-export it with a current loom-exporter\n");
        return 1;
    }
    std::string vocab_desc;
    std::printf("  task: %s (%s), %u codebook(s) per frame\n", contract.task.c_str(),
                contract.interface_name().c_str(), width);
    if (!contract.languages.empty()) std::printf("  languages: %zu declared\n", contract.languages.size());
    if (!req.has_prompt) {
        std::printf("  pass --prompt \"<text>\" --codec <codec.gguf> --out out.wav to speak it, or "
                    "--codes-out codes.gguf for the codes alone\n");
        return 0;
    }
    const std::vector<int32_t> ids = encode_for_codes(*model, req.prompt, &vocab_desc);
    std::printf("  tokenizer: %s\n  %zu id(s)\n", vocab_desc.c_str(), ids.size());
    if (ids.empty()) {
        std::fprintf(stderr, "error: --prompt produced no token ids\n");
        return 1;
    }

    // The codec, from its header only.
    uint32_t codec_width = 0;
    std::optional<uint32_t> absent;
    if (!req.codec_path.empty()) {
        const auto header = loom::GgufModel::load_metadata(req.codec_path);
        const loom::ModelContract codec_contract = loom::ModelContract::read(*header);
        if (codec_contract.interface_name() != "codes2speech") {
            std::fprintf(stderr, "error: --codec '%s' is a %s file, not a codec (codes2speech)\n",
                         req.codec_path.c_str(),
                         codec_contract.interface_name().empty() ? "undeclared" : codec_contract.interface_name().c_str());
            return 1;
        }
        codec_width = header->has_kv("loom.codec.n_codebooks") ? header->hparam_u32("codec.n_codebooks") : 0;
        if (header->has_kv("loom.codec.absent_code")) absent = header->hparam_u32("codec.absent_code");
        const std::string refused = pairing_error(width, codec_width, absent);
        if (!refused.empty()) {
            std::fprintf(stderr, "error: this model emits %u codebook(s) per frame and --codec '%s' %s\n",
                         width, req.codec_path.c_str(), refused.c_str());
            return 1;
        }
        std::printf("  codec: %s, %u codebook(s)", req.codec_path.c_str(), codec_width);
        if (width < codec_width) std::printf(", the last %u padded with id %u", codec_width - width, *absent);
        std::printf("\n");
    }
    if (!req.out_wav.empty() && req.codec_path.empty()) {
        std::fprintf(stderr, "error: this model's answer is codec tokens, not audio. Pass --codec "
                             "<codec.gguf> to decode them to --out, or --codes-out to keep the codes\n");
        return 1;
    }
    if (req.out_wav.empty() && req.codes_out.empty()) return 0;

    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    set_inputs(inputs, req.extra);
    inputs["tokens"] = std::vector<double>(ids.begin(), ids.end());
    // Every one of the three drivers seeds its sampler from this and falls back to its checkpoint's
    // declared temperature/top-k/top-p; the flags override those, as they do for a text LM.
    inputs["seed"] = static_cast<double>(req.seed);
    if (req.sampling.temperature) inputs["temperature"] = static_cast<double>(*req.sampling.temperature);
    if (req.sampling.top_k) inputs["top_k"] = static_cast<double>(*req.sampling.top_k);
    if (req.sampling.top_p) inputs["top_p"] = static_cast<double>(*req.sampling.top_p);
    // FRAMES, not rows -- the drivers undo their own delay pattern -- and only when asked, so the
    // model's own ceiling applies by default rather than `--n-predict`'s text-LM default of 16.
    if (req.max_frames) inputs["max_new_tokens"] = static_cast<double>(*req.max_frames);
    if (!req.language.empty()) {
        // The driver's `language` is a 1-based position in what the FILE declares, 0 meaning none
        // (loom-py's `_language_index`). A code it does not declare is refused rather than dropped.
        const auto it = std::find(contract.languages.begin(), contract.languages.end(), req.language);
        if (it == contract.languages.end() || !driver_reads(*model, "language")) {
            std::string have;
            for (const auto& l : contract.languages) have += (have.empty() ? "" : " ") + l;
            std::fprintf(stderr, "error: --language %s is not one this model declares (%s)\n",
                         req.language.c_str(), have.empty() ? "it declares none" : have.c_str());
            return 1;
        }
        inputs["language"] = static_cast<double>(it - contract.languages.begin() + 1);
    }
    if (!req.wav_path.empty()) {
        // A reference clip is the voice for a model that clones from audio (Qwen3-TTS: an x-vector
        // drawn from it; with --ref-text, also the clip's own codes replayed as the prompt, ICL).
        if (!driver_reads(*model, "waveform")) {
            std::fprintf(stderr, "error: this model's driver reads no reference clip, so --wav would be "
                                 "ignored. A voice for it is a --voice file, if it takes one.\n");
            return 1;
        }
        if (contract.sample_rate == 0) {
            std::fprintf(stderr, "error: this model declares no `loom.sample_rate` for its reference clip\n");
            return 1;
        }
        const std::vector<float> clip = loom_cli::load_wav_pcm16_mono(req.wav_path, contract.sample_rate);
        std::printf("  reference: %zu samples = %.2f s\n", clip.size(),
                    static_cast<double>(clip.size()) / contract.sample_rate);
        const std::vector<double> as_doubles(clip.begin(), clip.end());
        inputs["waveform"] = as_doubles;
        if (req.has_ref_text) {
            if (!driver_reads(*model, "ref_audio") || !driver_reads(*model, "ref_tokens")) {
                std::fprintf(stderr, "error: this model's driver does not replay a reference, so "
                                     "--ref-text would be ignored\n");
                return 1;
            }
            // The same vocabulary as the text, un-templated: the driver wraps both.
            const std::vector<int32_t> ref_ids = encode_for_codes(*model, req.ref_text, nullptr);
            inputs["ref_audio"] = as_doubles;
            inputs["ref_tokens"] = std::vector<double>(ref_ids.begin(), ref_ids.end());
            std::printf("  in-context: the clip's codes and its %zu transcript id(s) replayed ahead of "
                        "the text\n", ref_ids.size());
        }
    } else if (req.has_ref_text) {
        std::fprintf(stderr, "error: --ref-text is what a reference clip says; pass the clip as --wav\n");
        return 1;
    } else if (driver_reads(*model, "waveform") && !inputs.count("x_vector")) {
        // Qwen3-TTS-Base has no built-in speaker, so cloning is its only mode.
        std::fprintf(stderr, "error: this model clones a voice and has none of its own: pass --wav "
                             "<ref.wav> (%u Hz mono)\n", contract.sample_rate);
        return 1;
    }

    std::vector<double> flat;
    {
        loom::Session session(*model, backends);
        const auto result = session.bridge().call("infer", inputs);
        const auto* codes = std::get_if<std::vector<double>>(&result);
        if (codes == nullptr) {
            std::fprintf(stderr, "this model's driver returned a single number, not a run of codes\n");
            return 1;
        }
        flat = *codes;
        print_device_report(session.bridge());
    }
    if (flat.size() % width != 0) {
        std::fprintf(stderr, "error: the driver returned %zu codes, not a whole number of %u-wide frames; "
                             "the export and the driver disagree\n", flat.size(), width);
        return 1;
    }
    const size_t n_frames = flat.size() / width;
    std::printf("  %zu frame(s) x %u codebook(s)\n", n_frames, width);

    const std::vector<int32_t> codes(flat.begin(), flat.end());
    if (!req.codes_out.empty()) {
        loom_cli::write_codes_gguf(req.codes_out, codes, width, model->architecture());
        std::printf("  wrote %s\n", req.codes_out.c_str());
    }
    if (req.out_wav.empty()) return 0;

    model.reset();
    auto codec = loom::GgufModel::load(req.codec_path, backends);
    std::printf("loaded '%s'\n", req.codec_path.c_str());
    return synthesize(*codec, backends, widen_rows(codes, width, codec_width, absent), "codes", {},
                      req.out_wav, rate_override(req.extra), req.seed);
}

void print_device_report(const loom::LoomLuaBridge& bridge) {
    const auto report = bridge.device_report();
    if (report.empty()) return;
    std::printf("device report (module: splits, device nodes / cpu-fallback nodes)\n");
    for (const auto& m : report) {
        std::printf("  %-28s %3d   %6zu / %zu\n", m.module.c_str(), m.splits, m.device_nodes,
                     m.fallback_nodes);
    }
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

void run_asr(loom::GgufModel& model, loom::Backends backends, const std::string& wav_path,
             const std::string& language_name, const std::string& task_name, bool timestamps,
             bool condition_on_previous) {
    // EVERYTHING BELOW THE ARGUMENT PARSING IS THE ENGINE'S NOW (loom/core/transcribe.h). This function
    // used to hold the whole long-form loop -- windowing, segment splitting, the timestamp-aware seek,
    // prev_tokens conditioning -- and loom-py could not reach any of it, so its users got fixed cuts
    // and a worse transcript for no reason but where the code sat. What is left here is what a CLI
    // actually owns: turning `--language en` into an id, and printing.
    const std::vector<float> waveform = loom_cli::load_wav_pcm16_mono_16k(wav_path);

    // Registering the topologies and attaching the caches they declare is the engine's now too
    // (loom/core/session.h). The copy that used to be here attached a KvCache and no ConvStateCache,
    // which would have thrown inside the driver for any speech model carrying ShortConv blocks.
    loom::Session session(model, backends);

    loom::audio::TranscribeOptions options;
    options.timestamps = timestamps;
    options.condition_on_previous = condition_on_previous;
    // Straight through as NAMES. Resolving them to `<|en|>`-style token ids used to happen here, which
    // is why loom-py callers had to pass an integer they could not look up; the engine holds the vocab
    // and does it for both front ends now.
    options.language = language_name;
    options.task = task_name;

    const loom::audio::Transcription result =
        loom::audio::transcribe(session.bridge(), model, waveform, options);

    if (timestamps && result.timestamped) {
        for (const loom::audio::Segment& seg : result.segments) {
            std::printf("[%s --> %s] %s\n", format_time(seg.start).c_str(),
                        format_time(seg.end).c_str(), seg.text.c_str());
        }
    } else {
        std::printf("transcript: %s\n", result.text.c_str());
    }
    if (result.windows > 1) {
        std::fprintf(stderr, "(%zu windows%s)\n", result.windows,
                     result.timestamped ? ", seeking on the model's own timestamps"
                                        : ", cut at fixed boundaries -- this model exposes no timestamps");
    }
    print_device_report(session.bridge());
}


} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt_text;
    std::string ref_text;
    std::string wav_path;
    std::string language_name;
    std::string task_name;
    bool timestamps = false;
    bool condition_on_previous = true;
    std::string system_text;
    bool has_prompt = false;
    bool has_ref_text = false;
    bool has_wav = false;
    bool chat = false;
    bool has_system = false;
    uint32_t n_predict = 16;
    bool has_n_predict = false;
    std::string codec_path;
    std::string codes_out;
    loom::text::GenerateOptions gen_opts;
    std::string device_spec;
    bool list_devices = false;
    // Where a model whose answer is AUDIO writes it. Empty means the old behaviour -- print what the
    // file declares and stop -- so no existing invocation changes.
    std::string out_wav;
    // Extra driver inputs, `name=1,2,3` or `name=@file` (whitespace-separated numbers). The four TTS
    // families do not all take the same ones: Kokoro needs a 256-float `ref_s` voice embedding that is
    // not in the GGUF, StyleTTS2 takes `diffusion_steps`, Matcha `n_steps`. Rather than a flag per
    // family, the driver's own declared input names are the interface -- a wrong one is an error from
    // the engine naming the module and the input.
    DriverInputs extra_inputs;
    // A voice file's tensors become driver inputs by name (ADR-045), loaded once the model is, because
    // whether the file FITS the model is a question only the model can answer.
    std::string voice_path;

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
        } else if (arg == "--ref-text" && i + 1 < argc) {
            // The voice-cloning TTS families take a reference CLIP and what that clip SAYS. F5-TTS
            // conditions by in-filling: the transcript is prepended to the text to speak and the
            // model continues one spectrogram, so the two are not separable inputs.
            ref_text = argv[++i];
            has_ref_text = true;
        } else if (arg == "--voice" && i + 1 < argc) {
            voice_path = argv[++i];
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
        } else if (arg == "--chat") {
            // Wraps --prompt in the checkpoint's own chat template (P4.23). A flag rather than the
            // default because a base model has no template and an un-templated prompt is a legitimate
            // thing to ask an instruction-tuned one for -- but a flag that could only produce the
            // wrong answer would be worse than no flag, which is why this shipped with the tokenizer
            // fix and not before it.
            chat = true;
        } else if (arg == "--system" && i + 1 < argc) {
            system_text = argv[++i];
            has_system = true;
            chat = true;
        } else if (arg == "--temperature" && i + 1 < argc) {
            gen_opts.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            gen_opts.top_k = static_cast<int32_t>(std::stol(argv[++i]));
        } else if (arg == "--top-p" && i + 1 < argc) {
            gen_opts.top_p = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            gen_opts.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--n-predict" && i + 1 < argc) {
            n_predict = static_cast<uint32_t>(std::stoul(argv[++i]));
            has_n_predict = true;
        } else if (arg == "--codec" && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (arg == "--codes-out" && i + 1 < argc) {
            codes_out = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_wav = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "--input takes name=1,2,3 or name=@file, got '%s'\n", spec.c_str());
                return 2;
            }
            extra_inputs.push_back({spec.substr(0, eq), read_number_spec(spec.substr(eq + 1)), false});
        } else if (arg == "--device" && i + 1 < argc) {
            device_spec = argv[++i];
        } else if (arg == "--list-devices") {
            list_devices = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Before the --model check, so it answers on its own -- "what can this build reach" is a question
    // about the BUILD, and having to name a GGUF to ask it would be absurd.
    if (list_devices) {
        for (const loom::DeviceInfo& d : loom::available_devices()) {
            std::printf("%-12s %s", d.name.c_str(), d.description.c_str());
            if (d.memory_total > 0) {
                std::printf("  [%zu / %zu MiB free]", d.memory_free >> 20, d.memory_total >> 20);
            }
            std::printf("%s\n", d.is_cpu ? "  (cpu)" : "");
        }
        return 0;
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::unique_ptr<loom::Device> device;
    try {
        device = std::make_unique<loom::Device>(loom::Device::open(device_spec));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const loom::Backends backends = device->backends();
    // What a stochastic TTS driver seeds with. `--seed` is the same flag that makes a sampled
    // generation reproducible, which is the same question asked of a different kind of model.
    const uint32_t synth_seed = gen_opts.seed ? *gen_opts.seed : 1234u;
    if (!device->is_cpu()) {
        std::printf("device: %s (%s)\n", device->name().c_str(), device->description().c_str());
    }

    try {
        auto model = loom::GgufModel::load(model_path, backends);
        std::printf("loaded '%s'\n", model_path.c_str());
        if (!voice_path.empty()) {
            const loom::VoiceFile voice = loom::load_voice(*model, voice_path);
            std::printf("  voice: %s (%s)\n", voice.name.c_str(), voice.license.c_str());
            // Ahead of the --input ones, so a caller's explicit input still wins.
            for (const auto& [name, values] : voice.inputs) extra_inputs.insert(extra_inputs.begin(), {name, values, true});
        }
        std::printf("  architecture: %s\n", model->architecture().c_str());

        bool is_multi_topology = model->has_kv("model.driver_script");
        if (is_multi_topology) {
            std::printf("  graph_topology: Multi-topology file (Lua driven), %zu sub-modules\n", 
                        model->topology_names().size());
        } else {
            std::printf("  graph_topology: %zu bytes of JSON\n", model->topology_json().size());
        }
        std::printf("  weights: %zu tensors\n", model->weights().size());

        // The first branch here that asks the FILE what it is instead of guessing from its vocabulary
        // tag, which is what docs/HIGH-LEVEL-API.md §3 said the tag branches below would become. A
        // token classifier is a WordPiece file like the `"bert"` dead-end underneath it and is not
        // inspection-only: it has a driver, a declared task and a declared label set, so it gets run.
        const loom::ModelContract contract = loom::ModelContract::read(*model);
        if (contract.task == loom::task_names::TOKEN_CLASSIFICATION) {
            auto wp_vocab = loom::WordPieceVocab::load(*model);
            std::printf("  task: %s (%s), %zu labels\n", contract.task.c_str(),
                        contract.interface_name().c_str(), contract.labels.size());
            if (!has_prompt) {
                std::printf("  tokenizer: WordPiece (bert), %zu tokens\n", wp_vocab->size());
                std::printf("  pass --prompt \"<text>\" to label it\n");
                return 0;
            }
            const std::vector<int32_t> ids = wp_vocab->encode(prompt_text);
            if (ids.empty()) {
                std::fprintf(stderr, "error: --prompt produced no token ids\n");
                return 1;
            }
            loom::Session session(*model, backends);
            const auto labelled = loom::text::classify(session.bridge(), *model, ids);
            for (const auto& entry : labelled) {
                // The piece, not the sentence: a token classifier's answer IS per token, and joining
                // the labelled pieces back into text is a presentation choice this host has no basis
                // for making -- a "##continuation" piece belongs to the word before it, and which of
                // the two labels then wins is the caller's rule, not the model's.
                std::printf("  %-16s %s\n", wp_vocab->decode({entry.token}).c_str(),
                            entry.label.empty() ? std::to_string(entry.label_id).c_str()
                                                : entry.label.c_str());
            }
            print_device_report(session.bridge());
            return 0;
        }

        // Ahead of the vocabulary-tag branches below, because two of the three files it answers for
        // carry tags those branches claim: Dia's `byt5` is inspection-only there and Qwen3-TTS's and
        // MOSS-TTS's `gpt2` is the text generator.
        if (contract.interface_name() == "text2codes") {
            Text2CodesRequest req;
            req.prompt = prompt_text;
            req.has_prompt = has_prompt;
            if (has_wav) req.wav_path = wav_path;
            req.ref_text = ref_text;
            req.has_ref_text = has_ref_text;
            req.language = language_name;
            if (has_n_predict) req.max_frames = n_predict;
            req.sampling = gen_opts;
            req.seed = synth_seed;
            req.extra = extra_inputs;
            req.codec_path = codec_path;
            req.codes_out = codes_out;
            req.out_wav = out_wav;
            return text_to_codes(model, contract, backends, req);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "bert") {
            // A WordPiece file that declares no task -- a plain encoder, or an export older than
            // `loom.task`. No generation loop applies, so this stays inspection-only rather than
            // dead-ending loom_cli on a "bert" GGUF (see EXPORT-BACKLOG.md item 4).
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

        // A codec decoder: `audio_codes` in, audio out, and no vocabulary anywhere in the file. Its
        // `--prompt` is the codes themselves -- frame-major, `codec.n_codebooks` wide -- because that
        // is what the caller has: they came out of an AR model or an encoder, never off a keyboard.
        if (contract.interface_name() == "codes2speech") {
            const uint32_t width = model->has_kv("loom.codec.n_codebooks")
                                       ? model->hparam_u32("codec.n_codebooks") : 0;
            std::printf("  codec: %u stream(s) per frame", width);
            if (model->has_kv("loom.codec.frame_rate")) {
                std::printf(", %.4g frame(s) per second", model->hparam_f32("codec.frame_rate"));
            }
            std::printf("\n");
            if (!has_prompt || out_wav.empty()) {
                std::printf("  pass --prompt @codes.gguf (a text2codes run's --codes-out) or --prompt "
                            "\"<codes>\" --out out.wav to decode; codes are frame-major "
                            "(all %u for frame 0, then frame 1, ...)\n", width);
                return 0;
            }
            if (prompt_text.size() > 1 && prompt_text[0] == '@' && loom_cli::is_gguf(prompt_text.substr(1))) {
                // A codes file states its own width, so rows narrower than this codec's -- MOSS-TTS's 12
                // into MOSS-Audio-Tokenizer's 32 -- are widened by the same rule the pair uses, where a
                // flat list below can only be taken at this codec's width.
                const loom_cli::CodesFile file = loom_cli::read_codes_gguf(prompt_text.substr(1));
                std::optional<uint32_t> absent;
                if (model->has_kv("loom.codec.absent_code")) absent = model->hparam_u32("codec.absent_code");
                const std::string refused = pairing_error(file.n_codebooks, width, absent);
                if (!refused.empty()) {
                    std::fprintf(stderr, "error: '%s' holds %u codebook(s) per frame and this codec %s\n",
                                 prompt_text.c_str() + 1, file.n_codebooks, refused.c_str());
                    return 1;
                }
                std::printf("  %zu frame(s) x %u codebook(s)%s%s", file.n_frames(), file.n_codebooks,
                            file.source.empty() ? "" : " from ", file.source.c_str());
                if (file.n_codebooks < width) {
                    std::printf(", the last %u padded with id %u", width - file.n_codebooks, *absent);
                }
                std::printf("\n");
                return synthesize(*model, backends, widen_rows(file.codes, file.n_codebooks, width, absent),
                                  "codes", extra_inputs, out_wav, rate_override(extra_inputs), synth_seed);
            }
            const std::vector<double> codes = read_number_spec(prompt_text);
            if (width > 0 && codes.size() % width != 0) {
                std::fprintf(stderr, "  %zu code(s) is not a whole number of %u-wide frames\n",
                              codes.size(), width);
                return 1;
            }
            std::printf("  %zu code(s) = %zu frame(s)\n", codes.size(),
                        width > 0 ? codes.size() / width : codes.size());
            const std::vector<int32_t> ids(codes.begin(), codes.end());
            return synthesize(*model, backends, ids, "codes", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
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
                // in this export was traced at a FIXED length, so it is a CEILING on what a caller may
                // send. It was an exact requirement until the driver started padding (BACKLOG.md P4.6),
                // and printing "pad or shorten" at anything under it now would be telling a user to fix
                // something that already works.
                if (model->has_kv("loom.txt_len")) {
                    const uint32_t txt_len = model->hparam_u32("txt_len");
                    std::printf("  loom.txt_len = %u (max; the driver pads)%s\n", txt_len,
                                ids.size() <= txt_len ? "" : "  <-- encoded length EXCEEDS it; shorten");
                }
                if (!out_wav.empty()) {
                    return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                                      rate_override(extra_inputs), synth_seed);
                }
            }
            return 0;
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "chatterbox") {
            // Chatterbox: text in, the model's own built-in voice, audio out. The vocabulary runs the
            // reference's whole text path (`punc_norm`, `[SPACE]`, BPE), so what is printed as
            // "normalized" is the sentence the model is actually asked to say.
            auto vocab = loom::ChatterboxVocab::load(*model);
            std::printf("  tokenizer: character BPE (chatterbox), %zu tokens\n", vocab->size());
            if (!has_prompt) return 0;
            size_t unknown = 0;
            const auto ids = vocab->encode(prompt_text, &unknown);
            std::printf("  normalized: \"%s\"\n  %zu id(s)\n", vocab->normalize(prompt_text).c_str(),
                        ids.size());
            if (unknown > 0) {
                // The reference's own fallback, and a silent substitution a caller should see
                // (Retro-006).
                std::printf("  %zu character%s not in this model's table became [UNK]\n", unknown,
                            unknown == 1 ? "" : "s");
            }
            if (out_wav.empty()) return 0;
            return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "pocket_tts") {
            // Pocket-TTS: text in, the file's built-in voice, audio out. The vocabulary runs the
            // reference's whole text path, including its split into sentence chunks, so each chunk is
            // printed as the model will be asked to say it.
            auto vocab = loom::PocketTtsVocab::load(*model);
            std::printf("  tokenizer: SentencePiece unigram (pocket_tts), %zu pieces\n", vocab->size());
            if (!has_prompt) return 0;
            const auto chunks = vocab->chunks(prompt_text);
            for (size_t i = 0; i < chunks.size(); ++i) {
                size_t n_words = 0;
                const std::string prepared = vocab->prepare(chunks[i], &n_words);
                std::printf("  chunk %zu (%zu word%s): \"%s\"\n", i + 1, n_words, n_words == 1 ? "" : "s",
                            prepared.c_str());
            }
            const auto ids = vocab->encode(prompt_text);
            std::printf("  %zu id(s)\n", ids.size());
            if (out_wav.empty()) return 0;
            return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "cosyvoice3") {
            // CosyVoice3: text in, the file's default voice (or `--voice`, a cloned one), audio out. The
            // vocabulary runs the reference's text normalisation -- numbers spelled out, the paragraph
            // split into pieces the model says one at a time -- so each chunk is printed as it will be
            // said, and the driver generates them in turn.
            auto vocab = loom::CosyVoice3Vocab::load(*model);
            std::printf("  tokenizer: byte-level BPE (cosyvoice3), %zu tokens\n", vocab->size());
            if (!has_prompt) return 0;
            const auto chunks = vocab->chunks(prompt_text);
            for (size_t i = 0; i < chunks.size(); ++i) {
                std::printf("  chunk %zu: \"%s\"\n", i + 1, chunks[i].c_str());
            }
            const auto ids = vocab->encode(prompt_text);
            std::printf("  %zu id(s)\n", ids.size());
            if (out_wav.empty()) return 0;
            return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "voxcpm2") {
            // VoxCPM2: text in, a zero-shot voice (or one DESIGNED in the text, "(a calm older man)..."),
            // audio out. The vocabulary runs the reference's text path, so the printed text is what the
            // model is asked to say.
            auto vocab = loom::VoxCpmVocab::load(*model);
            std::printf("  tokenizer: character BPE with byte fallback (voxcpm2), %zu pieces\n", vocab->size());
            if (!has_prompt) return 0;
            size_t fallback = 0;
            const auto ids = vocab->encode(prompt_text, &fallback);
            std::printf("  normalized: \"%s\"\n  %zu id(s)\n", vocab->normalize(prompt_text).c_str(), ids.size());
            if (fallback > 0) {
                std::printf("  %zu character%s not in this model's table fell back to bytes\n", fallback,
                            fallback == 1 ? "" : "s");
            }
            if (out_wav.empty()) return 0;
            return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (contract.task == loom::task_names::TTS && model->has_kv("tokenizer.ggml.model") &&
            model->kv_str("tokenizer.ggml.model") == "gpt2") {
            // A TTS whose text front end is a plain byte-level BPE (Voxtral-4B-TTS's Tekken): the
            // vocabulary encodes the text and the driver builds the rest of the prompt, so nothing here
            // is per-model. Without this branch the file would fall through to the LM path below and be
            // run as a text generator.
            auto vocab = loom::BpeVocab::load(*model);
            std::printf("  tokenizer: byte-level BPE (%s), %zu tokens\n",
                        model->has_kv("tokenizer.ggml.pre") ? model->kv_str("tokenizer.ggml.pre").c_str() : "qwen2",
                        vocab->size());
            if (!has_prompt) return 0;
            const auto ids = vocab->encode(prompt_text);
            std::printf("  %zu id(s)\n", ids.size());
            if (out_wav.empty()) return 0;
            return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "f5") {
            // F5-TTS: the one TTS family here that takes a reference CLIP as well as text. It clones
            // the voice in that clip by IN-FILLING -- the reference's transcript is prepended to the
            // text to speak and the model continues one spectrogram -- so `--wav` is the voice,
            // `--ref-text` is what it says and `--prompt` is what to say in it.
            auto vocab = loom::F5Vocab::load(*model);
            std::printf("  tokenizer: characters (f5), %zu rows, graph ids are table ids + %u\n",
                        vocab->size(), vocab->filler_offset());
            if (!has_prompt) return 0;
            if (!has_wav || !has_ref_text) {
                std::fprintf(stderr,
                             "  this model clones a voice, so it needs one: pass --wav <ref.wav> "
                             "(%u Hz mono) and --ref-text \"<what that clip says>\" beside --prompt.\n",
                             contract.sample_rate);
                return 1;
            }
            // The reference's own join: `text_list = [ref_text + gen_text]`, with a space appended to
            // the transcript when it does not end in one (`infer_batch_process` does this by byte
            // length). Joining without it runs the last word of the prompt into the first word of the
            // text, which the model reads as one word and says as one.
            std::string joined = ref_text;
            if (!joined.empty() && joined.back() != ' ') joined += ' ';
            size_t unknown_ref = 0, unknown_all = 0;
            const auto ref_ids = vocab->encode(joined, &unknown_ref);
            const auto all_ids = vocab->encode(joined + prompt_text, &unknown_all);
            std::printf("  %zu reference id(s) + %zu to speak = %zu\n", ref_ids.size(),
                        all_ids.size() - ref_ids.size(), all_ids.size());
            if (unknown_all > 0) {
                // Not exceptional -- the table's fallback is a real row -- but a SILENT substitution
                // is how a caller gets audio that is not the sentence (Retro-006).
                std::printf("  %zu character%s not in this model's table, mapped to id %d; it will "
                            "say \"%s\"\n", unknown_all, unknown_all == 1 ? "" : "s", vocab->unk_id(),
                            vocab->decode(all_ids).c_str());
            }
            if (out_wav.empty()) return 0;

            const std::vector<float> reference =
                loom_cli::load_wav_pcm16_mono(wav_path, contract.sample_rate ? contract.sample_rate
                                                                             : 24000);
            std::printf("  reference: %zu samples = %.2f s\n", reference.size(),
                        static_cast<double>(reference.size()) /
                            std::max<uint32_t>(contract.sample_rate, 1));
            auto extra = extra_inputs;
            extra.push_back({"waveform", std::vector<double>(reference.begin(), reference.end()), true});
            extra.push_back({"n_ref_text", std::vector<double>{static_cast<double>(ref_ids.size())}, false});
            return synthesize(*model, backends, all_ids, "text_ids", extra, out_wav,
                              rate_override(extra_inputs), synth_seed);
        }

        if (model->has_kv("tokenizer.ggml.model") && model->kv_str("tokenizer.ggml.model") == "phonemes") {
            // The four phoneme-input TTS families (VITS, Kokoro, StyleTTS2, Matcha). Inspection-only for
            // the same reason as the three branches above -- no generation loop applies to a vocoder --
            // but here the fall-through was not a hypothetical: `bpe_vocab` stays null for any non-"gpt2"
            // tag, so an IPA `--prompt` reached `parse_token_ids`, which found no integers in it and
            // died on "produced no token ids" while the model's own symbol table sat unread in the same
            // file. loom-py has encoded through that table since it was exported and
            // this host never learned to; that is the identical half-wiring the Supertonic branch above
            // was written to correct.
            auto phoneme_vocab = loom::PhonemeVocab::load(*model);
            std::printf("  tokenizer: phoneme symbols (phonemes), %zu tokens\n", phoneme_vocab->size());
            // The assembly is declared per checkpoint and is not part of the table, so it is printed
            // beside it: piper builds [BOS, p1, blank, ..., pn, blank, EOS] where StyleTTS2 wraps with
            // neither. A -1 is this model saying it HAS no such token -- printing the sentinel as a
            // number invites a caller to send it, which reaches the engine as an out-of-range GET_ROWS.
            const auto assembly = [](int32_t id) {
                return id < 0 ? std::string("none") : std::to_string(id);
            };
            std::printf("  assembly: bos=%s eos=%s blank=%s%s\n", assembly(phoneme_vocab->bos_id()).c_str(),
                        assembly(phoneme_vocab->eos_id()).c_str(), assembly(phoneme_vocab->blank_id()).c_str(),
                        phoneme_vocab->interleave_blank() ? ", interleaved between every phoneme" : "");
            if (has_prompt) {
                // --prompt is PHONEMES here, not text: grapheme-to-phoneme is a property of the language
                // rather than of any checkpoint and is in no GGUF, so this host has nothing to run it
                // with (docs/HIGH-LEVEL-API.md §5). What it can do is the half that IS in the file,
                // which is exactly the half that was unreachable from here.
                size_t unknown = 0;
                const auto ids = phoneme_vocab->encode(prompt_text, &unknown);
                std::printf("  encode(\"%s\") -> [", prompt_text.c_str());
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
                std::printf("]\n");
                // A dropped symbol is expected rather than exceptional -- a rule-based G2P emits a
                // superset of what any one checkpoint was trained on, which is why the vocabulary skips
                // instead of raising -- but a SILENT drop is how a caller gets audio that is subtly not
                // the sentence, so say how many went and what is left to say (Retro-006).
                if (unknown > 0) {
                    std::printf("  %zu symbol%s not in this model's table, dropped; it will say \"%s\"\n",
                                unknown, unknown == 1 ? "" : "s", phoneme_vocab->decode(ids).c_str());
                }
                if (!out_wav.empty()) {
                    return synthesize(*model, backends, ids, "tokens", extra_inputs, out_wav,
                                      rate_override(extra_inputs), synth_seed);
                }
            }
            return 0;
        }

        if (has_prompt) {
            std::unique_ptr<loom::BpeVocab> bpe_vocab;
            // The SentencePiece half of the same job, and it is here because family 6 arrived without
            // it: `loom::Vocab` has encoded and decoded UGM/SentencePiece-BPE vocabularies since the
            // ASR families needed them, and this branch only ever asked for `"gpt2"` -- so a T5 file
            // reached `parse_token_ids`, which found no integers in the sentence and reported
            // "produced no token ids" while the model's own 32,100-piece vocabulary sat unread in it.
            // Exactly the half-wiring the phoneme branch above was written to correct, one modality
            // over.
            std::unique_ptr<loom::Vocab> spm_vocab;
            const std::string vocab_tag =
                model->has_kv("tokenizer.ggml.model") ? model->kv_str("tokenizer.ggml.model") : "";
            if (vocab_tag == "gpt2") {
                bpe_vocab = loom::BpeVocab::load(*model);
            } else if (vocab_tag == "t5" || vocab_tag == "llama") {
                spm_vocab = loom::Vocab::load(*model);
            }
            // One name for "this file can turn text into ids", so the three call sites below cannot
            // disagree about which vocabulary answered.
            const auto encode_prompt = [&](const std::string& text) {
                if (bpe_vocab) return bpe_vocab->encode(text);
                if (spm_vocab) return spm_vocab->encode(text);
                return parse_token_ids(text);
            };
            const auto decode_ids = [&](const std::vector<int32_t>& ids) {
                if (bpe_vocab) return bpe_vocab->decode(ids);
                if (spm_vocab) return spm_vocab->decode(ids);
                return std::string();
            };
            const bool has_text_vocab = bpe_vocab != nullptr || spm_vocab != nullptr;

            // The chat template and the tokenizer are one feature, not two: the template's markers are
            // ADDED tokens, and a vocabulary that cannot emit those atomically turns each of them into
            // seven literal ids (P4.23). So this path exists only where `bpe_vocab` does.
            std::string effective_prompt = prompt_text;
            if (chat) {
                auto tmpl = loom::ChatTemplate::load(*model);
                if (!tmpl) {
                    std::fprintf(stderr, "error: --chat, but this model carries no chat template. A "
                                          "base model has none, and one whose template could not be "
                                          "reduced to role tags exports without it -- see the export's "
                                          "own \"no chat template\" line.\n");
                    return 1;
                }
                std::vector<loom::ChatMessage> messages;
                if (has_system) messages.push_back({"system", system_text});
                messages.push_back({"user", prompt_text});
                effective_prompt = tmpl->apply(messages, /*add_generation_prompt=*/true);
            }

            const std::vector<int32_t> prompt_tokens = encode_prompt(effective_prompt);
            if (prompt_tokens.empty()) {
                std::fprintf(stderr, "error: --prompt produced no token ids\n");
                return 1;
            }

            // A TTS file whose text front end is a PLAIN vocabulary (SpeechT5's SentencePiece
            // characters) has no family branch above, and without this one it reached the token loop,
            // which ran the whole synthesis and then refused the waveform as "68096 ids for
            // max_new_tokens=16". The declared output kind is what says which door this is.
            const bool speaks = model->has_kv("loom.output.kind") && model->kv_str("loom.output.kind") == "audio";
            if (is_multi_topology && speaks && has_text_vocab) {
                std::printf("  %zu id(s)\n", prompt_tokens.size());
                if (out_wav.empty()) {
                    std::printf("  pass --out out.wav to synthesise it\n");
                    return 0;
                }
                return synthesize(*model, backends, prompt_tokens, "tokens", extra_inputs, out_wav,
                                  rate_override(extra_inputs), synth_seed);
            }

            if (is_multi_topology) {
                // THE LOOP IS THE ENGINE'S NOW (loom/core/text_generate.h), and unifying it changed this
                // CLI's behaviour in three ways that were all bugs rather than choices: it ran the full
                // `--n-predict` regardless of the model's own end-of-sequence token, it took the FIRST
                // element of a list return where the new token is the last, and it silently rewrote any
                // id >= 65536 to 0 -- a guard that would corrupt output for any vocabulary larger than
                // that rather than reporting anything. loom-py's copy of this loop did none of the three,
                // which is how the divergence was found (docs/HIGH-LEVEL-API.md §1).
                loom::Session session(*model, backends);

                std::printf("Running dynamic GGUF generation for %d tokens...\n", n_predict);
                loom::text::GenerateOptions gen = gen_opts;
                gen.max_new_tokens = n_predict;
                const std::vector<int32_t> generated =
                    loom::text::generate(session.bridge(), *model, prompt_tokens, gen);

                if (has_text_vocab) {
                    std::printf("generated %zu tokens -> \"%s\"\n", generated.size(),
                                decode_ids(generated).c_str());
                } else {
                    std::printf("generated %zu tokens:", generated.size());
                    for (int32_t tok : generated) std::printf(" %d", tok);
                    std::printf("\n");
                }
                print_device_report(session.bridge());
            } else {
                loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
                loom::GenerationConfig cfg;
                cfg.max_new_tokens = n_predict;
                cfg.n_ctx_max = static_cast<uint32_t>(prompt_tokens.size()) + n_predict;

                loom::Generator generator(*model, topo, cfg, backends);
                const std::vector<int32_t> generated = generator.generate(prompt_tokens);

                if (has_text_vocab) {
                    std::printf("generated %zu tokens -> \"%s\"\n", generated.size(),
                                decode_ids(generated).c_str());
                } else {
                    std::printf("generated %zu tokens:", generated.size());
                    for (int32_t tok : generated) std::printf(" %d", tok);
                    std::printf("\n");
                }
            }
        }

        if (has_wav) {
            run_asr(*model, backends, wav_path, language_name, task_name, timestamps,
                    condition_on_previous);
        }
    } catch (const loom::Error& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    } catch (const std::runtime_error& e) { // load_wav_pcm16_mono_16k throws this, not loom::Error
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    // Explicit rather than left to the atexit handler profile.cpp registers, purely for ORDERING: this
    // process writes its results to a block-buffered stdout, and a report emitted at exit to unbuffered
    // stderr lands ahead of them in a pipe. No-op unless $LOOM_PROFILE asked for one.
    loom::profile::write_report();
    return 0;
}
