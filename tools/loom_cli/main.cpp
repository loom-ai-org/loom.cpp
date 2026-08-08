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
                  "       %s --model <asr.gguf> --wav <path.wav> [--language en]\n",
                  argv0, argv0);
}

std::vector<int32_t> parse_token_ids(const std::string& text) {
    std::vector<int32_t> tokens;
    std::istringstream iss(text);
    int32_t tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}


// A decode of one clip's worth of audio: the driver's own `infer`, with whatever optional arguments the
// caller supplied. Returns the token ids it generated.
//
// `language` is passed through only when the caller named one. Omitting the key is not the same as
// passing a default: a driver that can detect the language does so when it is absent, and one that
// cannot falls back to its own default -- the resolution order lives in the model's driver, which is the
// only place that knows whether detection is possible at all.
std::vector<int32_t> run_driver(loom::LoomLuaBridge& bridge, const std::vector<double>& waveform,
                                 int32_t language, uint32_t max_new_tokens, int32_t eos_token) {
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
        {"waveform", waveform},
        {"max_new_tokens", static_cast<double>(max_new_tokens)},
        {"eos_token", static_cast<double>(eos_token)},
    };
    if (language >= 0) args["language"] = static_cast<double>(language);

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
             const std::string& language_name) {
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
    // The driver returns the end-of-sequence token it stopped on -- deliberately, since a generator's
    // caller may want to know whether it stopped or ran out of budget. A transcript is not that caller:
    // decoded literally, `<|endoftext|>` is a real vocabulary piece and prints as those nine characters.
    const int32_t eos_id = model.kv_i32("tokenizer.ggml.eos_token_id", -1);
    const auto detokenize = [&](const std::vector<int32_t>& ids) {
        std::vector<int32_t> text_ids;
        text_ids.reserve(ids.size());
        for (int32_t id : ids) {
            if (id != eos_id) text_ids.push_back(id);
        }
        return spm_vocab ? spm_vocab->decode(text_ids) : bpe_vocab->decode(text_ids);
    };

    // `--language xx` becomes this model's own `<|xx|>` token id, by TEXT rather than by a number this
    // CLI would otherwise have to carry per checkpoint. -1 means "not named", which is what asks the
    // driver to detect it (or to use its default when it cannot).
    int32_t language = -1;
    if (!language_name.empty()) {
        const std::string piece = "<|" + language_name + "|>";
        language = bpe_vocab ? bpe_vocab->piece_to_id(piece) : -1;
        if (language < 0) {
            throw loom::LoadError("--language " + language_name + ": this model's vocabulary has no '" +
                                  piece + "' token. An English-only checkpoint has no language tokens "
                                  "at all; a multilingual one names them by ISO code (en, de, fr, ...).");
        }
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

    // Fixed-clip models: walk the file in back-to-back windows of exactly `clip` samples, zero-padding
    // the last one, and concatenate what each window transcribes.
    //
    // **Naive on purpose, and the seam is real.** Each window is decoded independently, so a word
    // straddling a boundary is split and no window sees the previous one's text. Whisper's own long-form
    // algorithm avoids that by conditioning each window on the previous window's tokens and ending
    // windows at emitted timestamps rather than at a fixed 30 s -- a substantially larger feature than
    // this, and one that needs the timestamp tokens this CLI currently suppresses.
    const size_t n_windows = (waveform.size() + clip - 1) / clip;
    std::string transcript;
    for (size_t w = 0; w < n_windows; ++w) {
        const size_t begin = w * clip;
        const size_t avail = std::min(static_cast<size_t>(clip), waveform.size() - begin);
        std::vector<double> window(clip, 0.0); // zero-padded: Whisper's own pad_or_trim convention
        for (size_t i = 0; i < avail; ++i) window[i] = waveform[begin + i];

        const std::vector<int32_t> ids = run_driver(bridge, window, language, kMaxNewTokensPerClip, eos_id);
        const std::string piece = detokenize(ids);
        if (n_windows > 1) {
            std::fprintf(stderr, "  window %zu/%zu [%.1fs..%.1fs]: %zu tokens\n", w + 1, n_windows,
                         static_cast<double>(begin) / 16000.0,
                         static_cast<double>(begin + avail) / 16000.0, ids.size());
        }
        transcript += piece;
    }
    std::printf("transcript: %s\n", transcript.c_str());
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt_text;
    std::string wav_path;
    std::string language_name;
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
            run_asr(*model, backend.get(), wav_path, language_name);
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
