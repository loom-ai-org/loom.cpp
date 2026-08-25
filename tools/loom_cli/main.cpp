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
                  "\n"
                  "  --device <auto|cpu|gpu|NAME>  where to run (default: auto, or $LOOM_DEVICE)\n"
                  "  --list-devices                print the devices this build can reach, and exit\n"
                  "\n"
                  "  $LOOM_PROFILE=1               time every graph node and print a per-op breakdown\n"
                  "  $LOOM_PROFILE=<path>          ... to a file instead of stderr\n"
                  "  $LOOM_PROFILE_NODES=1         ... and a second table keyed on the NODE name, which\n"
                  "                                is the only thing that says which graph a bucket is in\n"
                  "                                (profile with ONE thread; see include/loom/core/profile.h)\n",
                  argv0, argv0);
}

// What ran where, after a device run. The number that matters is the split count: each split is a point
// at which execution crossed between the device and the CPU fallback, and every crossing is a copy in
// each direction. A module reported as 1 split ran entirely on one backend.
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
    bool has_prompt = false;
    bool has_wav = false;
    uint32_t n_predict = 16;
    std::string device_spec;
    bool list_devices = false;

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
                // in this export was traced at a FIXED length, so it is a CEILING on what a caller may
                // send. It was an exact requirement until the driver started padding (BACKLOG.md P4.6),
                // and printing "pad or shorten" at anything under it now would be telling a user to fix
                // something that already works.
                if (model->has_kv("loom.txt_len")) {
                    const uint32_t txt_len = model->hparam_u32("txt_len");
                    std::printf("  loom.txt_len = %u (max; the driver pads)%s\n", txt_len,
                                ids.size() <= txt_len ? "" : "  <-- encoded length EXCEEDS it; shorten");
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
                // THE LOOP IS THE ENGINE'S NOW (loom/core/text_generate.h), and unifying it changed this
                // CLI's behaviour in three ways that were all bugs rather than choices: it ran the full
                // `--n-predict` regardless of the model's own end-of-sequence token, it took the FIRST
                // element of a list return where the new token is the last, and it silently rewrote any
                // id >= 65536 to 0 -- a guard that would corrupt output for any vocabulary larger than
                // that rather than reporting anything. loom-py's copy of this loop did none of the three,
                // which is how the divergence was found (docs/HIGH-LEVEL-API.md §1).
                loom::Session session(*model, backends);

                std::printf("Running dynamic GGUF generation for %d tokens...\n", n_predict);
                loom::text::GenerateOptions gen;
                gen.max_new_tokens = n_predict;
                const std::vector<int32_t> generated =
                    loom::text::generate(session.bridge(), *model, prompt_tokens, gen);

                if (bpe_vocab) {
                    std::printf("generated %zu tokens -> \"%s\"\n", generated.size(),
                                bpe_vocab->decode(generated).c_str());
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
