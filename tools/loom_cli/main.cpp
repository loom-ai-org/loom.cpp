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
#include "wav_file.h"


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
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
                  argv0, argv0, argv0);
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

// `--input sample_rate=N`, for the three models that declare none of their own.
uint32_t rate_override(const std::vector<std::pair<std::string, std::vector<double>>>& extra) {
    for (const auto& [name, values] : extra) {
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
                const std::vector<std::pair<std::string, std::vector<double>>>& extra,
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
    for (const auto& [name, values] : extra) {
        if (name == "sample_rate") continue;               // ours, not the driver's
        if (values.size() == 1) {
            inputs[name] = values[0];
        } else {
            inputs[name] = values;
        }
    }

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
    std::printf("  %zu samples at %u Hz = %.2f s, peak %.4f, rms %.4f\n", audio.size(), rate,
                static_cast<double>(audio.size()) / rate, peak,
                std::sqrt(sum_sq / std::max<size_t>(audio.size(), 1)));
    loom_cli::write_wav_pcm16_mono(out_path, audio, rate);
    std::printf("  wrote %s\n", out_path.c_str());
    print_device_report(session.bridge());
    return 0;
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
    std::string wav_path;
    std::string language_name;
    std::string task_name;
    bool timestamps = false;
    bool condition_on_previous = true;
    std::string system_text;
    bool has_prompt = false;
    bool has_wav = false;
    bool chat = false;
    bool has_system = false;
    uint32_t n_predict = 16;
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
    std::vector<std::pair<std::string, std::vector<double>>> extra_inputs;

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
        } else if (arg == "--out" && i + 1 < argc) {
            out_wav = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "--input takes name=1,2,3 or name=@file, got '%s'\n", spec.c_str());
                return 2;
            }
            extra_inputs.emplace_back(spec.substr(0, eq), read_number_spec(spec.substr(eq + 1)));
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
                std::printf("  pass --prompt \"<codes>\" --out out.wav to decode; codes are frame-major "
                            "(all %u for frame 0, then frame 1, ...)\n", width);
                return 0;
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
