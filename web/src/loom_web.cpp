// The WebAssembly binding (ADR-139): the engine's doors, for JavaScript, through embind.
//
// The same sequence loom-py's `binding.cpp` performs: open a device sized for the weights, load the GGUF,
// register every topology it declares with the caches they ask for, load the driver it carries, and
// read its tokenizer (`loom::Tokenizer`) and chat template. What a host does on top (the door
// arguments, voices by name, sentence pieces) is `web/js/loom.mjs`, as it is `loom/_interfaces.py` for
// Python. This file only marshals.
//
// **Every call that may run a graph is async under JSPI** (`LOOM_WEB_JSPI`, the WebGPU build): ggml's
// WebGPU backend waits on the GPU, and JSPI suspends the WebAssembly stack while the browser resolves
// the wait. From JavaScript each of those calls returns a Promise. The CPU build waits on nothing, so
// its calls return plain values, and the JS layer awaits both alike.
//
// **A C++ exception becomes a JavaScript Error carrying its message.** Without this, embind hands
// JavaScript an opaque WebAssembly.Exception and the engine's messages are lost. Those messages are
// how a caller learns that a file has no chat template, or which languages it knows.

#include "loom/loom.h"
#include "loom/core/chat_template.h"
#include "loom/core/model_contract.h"
#include "loom/core/text_generate.h"
#include "loom/core/tokenizer.h"
#include "loom/core/transcribe.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using emscripten::val;
using json = nlohmann::json;
using Value = loom::LoomLuaBridge::Value;
using Inputs = std::unordered_map<std::string, Value>;

namespace {

[[noreturn]] void throw_js(const std::string& message) { val::global("Error").new_(message).throw_(); }

// Runs `f`, turning any C++ exception into a JavaScript Error with its message.
template <class F>
auto guarded(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        throw_js(e.what());
    }
}

template <class T>
val typed_array(const char* type, const std::vector<T>& v) {
    // `new Float32Array(view)` copies, so the result outlives the vector (and any memory growth).
    return val::global(type).new_(val(emscripten::typed_memory_view(v.size(), v.data())));
}

std::vector<int32_t> ids_of(const val& v) { return emscripten::convertJSArrayToNumberVector<int32_t>(v); }

// One driver input: a number, or an array-like of numbers (Array or any TypedArray). The bridge's
// own Value is exactly those two.
Value to_value(const std::string& key, const val& v) {
    const std::string type = v.typeOf().as<std::string>();
    if (type == "number") return v.as<double>();
    if (type == "boolean" || type == "string" || v.isNull() || v.isUndefined() || !v["length"].isNumber()) {
        throw std::invalid_argument("input '" + key + "' is a " + type +
                                    "; a driver's inputs are numbers and arrays of numbers");
    }
    return emscripten::convertJSArrayToNumberVector<double>(v);
}

Inputs to_inputs(const val& obj) {
    Inputs out;
    if (obj.isUndefined() || obj.isNull()) return out;
    const val keys = val::global("Object").call<val>("keys", obj);
    const int n = keys["length"].as<int>();
    for (int i = 0; i < n; ++i) {
        const std::string key = keys[i].as<std::string>();
        out.emplace(key, to_value(key, obj[key]));
    }
    return out;
}

val from_value(const Value& v) {
    if (std::holds_alternative<double>(v)) return val(std::get<double>(v));
    return typed_array("Float64Array", std::get<std::vector<double>>(v));
}

template <class T>
std::optional<T> opt(const val& o, const char* key) {
    if (o.isUndefined() || o.isNull()) return std::nullopt;
    const val v = o[key];
    if (v.isUndefined() || v.isNull()) return std::nullopt;
    return v.as<T>();
}

json contract_json(const loom::GgufModel& model) {
    const loom::ModelContract c = loom::ModelContract::read(model);
    json out = {
        {"declared", c.declared()}, {"task", c.task}, {"input_kind", c.input_kind},
        {"output_kind", c.output_kind}, {"interface", c.interface_name()}, {"sample_rate", c.sample_rate},
        {"channels", c.channels}, {"clip_samples", c.clip_samples}, {"max_input_tokens", c.max_input_tokens},
        {"text_frontend", c.text_frontend}, {"phoneme_alphabet", c.phoneme_alphabet},
        {"phonemizer_ruleset", c.phonemizer_ruleset}, {"languages", c.languages},
        {"entry_points", c.entry_points}, {"default_steps", c.default_steps}, {"voices", c.voices},
        {"voice_languages", c.voice_languages}, {"labels", c.labels},
        {"output_granularity", c.output_granularity}, {"frame_rate", c.frame_rate},
        {"frame_offset", c.frame_offset}, {"embedding_dim", c.embedding_dim},
    };
    if (c.task == loom::task_names::ASR) {
        const loom::audio::AsrDecodeTable t = loom::audio::AsrDecodeTable::read(model, nullptr, c);
        out["asr_languages"] = t.language_names;
        out["asr_tasks"] = t.task_names;
        out["asr_target_languages"] = t.target_language_names;
    }
    return out;
}

class TokenStream {
public:
    TokenStream(loom::LoomLuaBridge& bridge, const loom::GgufModel& model, std::vector<int32_t> prompt,
                loom::text::GenerateOptions options)
        : stream_(bridge, model, std::move(prompt), std::move(options)) {}
    val next(int max_tokens) {
        return guarded([&] { return typed_array("Int32Array", stream_.next(static_cast<uint32_t>(max_tokens))); });
    }
    val tokens() const { return typed_array("Int32Array", stream_.tokens()); }
    bool finished() const { return stream_.finished(); }
    bool stopped_on_eos() const { return stream_.stopped_on_eos(); }
    void reset() { guarded([&] { stream_.reset(); }); }

private:
    loom::text::GenerationStream stream_;
};

class SpeechStream {
public:
    SpeechStream(loom::LoomLuaBridge& bridge, Inputs inputs) : stream_(bridge, std::move(inputs)) {}
    val next() { return guarded([&] { return typed_array("Float32Array", stream_.next()); }); }
    bool finished() const { return stream_.finished(); }
    int pieces_done() const { return static_cast<int>(stream_.pieces_done()); }
    int pieces_total() const { return static_cast<int>(stream_.pieces_total()); }
    void reset() { guarded([&] { stream_.reset(); }); }

private:
    loom::audio::SpeechStream stream_;
};

class Model {
public:
    Model(const std::string& path, const std::string& device)
        : device_(std::make_unique<loom::Device>(
              loom::Device::open(device, loom::GgufModel::load_metadata(path)->weight_bytes()))) {
        const loom::Backends backends = device_->backends();
        model_ = loom::GgufModel::load(path, backends);
        if (model_ == nullptr) throw std::runtime_error("could not load a loom GGUF from " + path);
        bridge_ = std::make_unique<loom::LoomLuaBridge>(backends);
        caches_ = loom::register_topologies(*model_, backends, *bridge_);
        if (model_->has_kv("model.driver_script")) driver_ = model_->kv_str("model.driver_script");
        if (!driver_.empty()) bridge_->load_script(driver_);
        tokenizer_ = loom::Tokenizer::load(*model_);
        chat_ = loom::ChatTemplate::load(*model_);
    }

    std::string contract() const { return contract_json(*model_).dump(); }
    std::string architecture() const { return model_->architecture(); }
    std::string device_name() const { return device_->name(); }
    std::string device_description() const { return device_->description(); }
    std::string driver_source() const { return driver_; }
    bool takes_voice_files() const { return model_->has_kv("loom.voice.compat"); }
    std::string hparam_str(const std::string& key) const {
        return guarded([&] { return model_->hparam_str(key); });
    }
    // Any string key by its full name ("tokenizer.chat_template...", a fixture's "test.*").
    std::string kv_str(const std::string& key) const { return guarded([&] { return model_->kv_str(key); }); }

    // -- text ---------------------------------------------------------------------------------------
    std::string tokenizer_kind() const { return tokenizer_ ? tokenizer_->kind() : std::string{}; }
    val encode(const std::string& text, const std::string& lang) const {
        return guarded([&] { return typed_array("Int32Array", tokenizer().encode(text, lang)); });
    }
    // Bytes, not a string: a byte-level BPE token can end inside a UTF-8 character, and the JS side
    // decodes a stream incrementally (TextDecoder with `stream: true`).
    val decode_bytes(const val& ids) const {
        return guarded([&] {
            const std::string bytes = tokenizer().decode(ids_of(ids));
            return val::global("Uint8Array").new_(
                val(emscripten::typed_memory_view(bytes.size(), reinterpret_cast<const uint8_t*>(bytes.data()))));
        });
    }
    val chat_roles() const {
        val out = val::array();
        if (chat_) for (const std::string& r : chat_->roles()) out.call<void>("push", r);
        return out;
    }
    std::string apply_chat_template(const val& messages, bool add_generation_prompt) const {
        return guarded([&] {
            if (chat_ == nullptr) {
                throw std::runtime_error("this GGUF carries no chat template (a base model); generate() "
                                         "continues a raw prompt.");
            }
            std::vector<loom::ChatMessage> converted;
            const int n = messages["length"].as<int>();
            for (int i = 0; i < n; ++i) {
                converted.push_back({messages[i]["role"].as<std::string>(), messages[i]["content"].as<std::string>()});
            }
            return chat_->apply(converted, add_generation_prompt);
        });
    }
    val eos_token_ids() const { return typed_array("Int32Array", loom::text::eos_token_ids(*model_)); }

    val generate(const val& tokens, const val& options) {
        return guarded([&] {
            require_driver("generate with");
            return typed_array("Int32Array",
                               loom::text::generate(*bridge_, *model_, ids_of(tokens), generate_options(options)));
        });
    }
    bool can_stream_tokens() const {
        return !driver_.empty() && loom::text::GenerationStream::supported(*bridge_, *model_);
    }
    TokenStream* open_token_stream(const val& tokens, const val& options) {
        return guarded([&] {
            require_driver("stream");
            return new TokenStream(*bridge_, *model_, ids_of(tokens), generate_options(options));
        });
    }

    // -- audio --------------------------------------------------------------------------------------
    std::string transcribe(const val& waveform, const val& options) {
        return guarded([&] {
            require_driver("transcribe");
            loom::audio::TranscribeOptions o;
            if (auto v = opt<std::string>(options, "language")) o.language = *v;
            if (auto v = opt<std::string>(options, "task")) o.task = *v;
            if (auto v = opt<std::string>(options, "target_language")) o.target_language = *v;
            if (auto v = opt<bool>(options, "timestamps")) o.timestamps = *v;
            if (auto v = opt<bool>(options, "condition_on_previous")) o.condition_on_previous = *v;
            const auto samples = emscripten::convertJSArrayToNumberVector<float>(waveform);
            const loom::audio::Transcription r = loom::audio::transcribe(*bridge_, *model_, samples, o);
            json segments = json::array();
            for (const auto& s : r.segments) {
                segments.push_back({{"start", s.start}, {"end", s.end}, {"text", s.text}, {"closed", s.closed}});
            }
            return json{{"text", r.text}, {"segments", segments}, {"windows", r.windows},
                        {"timestamped", r.timestamped}, {"warnings", r.warnings}}.dump();
        });
    }
    bool can_stream_speech() const {
        return !driver_.empty() && loom::audio::SpeechStream::supported(*bridge_);
    }
    SpeechStream* open_speech_stream(const val& inputs) {
        return guarded([&] {
            require_driver("stream");
            return new SpeechStream(*bridge_, to_inputs(inputs));
        });
    }
    // A voice file (ADR-045) already written to the virtual filesystem: its driver inputs by name.
    val load_voice(const std::string& path) const {
        return guarded([&] {
            const loom::VoiceFile voice = loom::load_voice(*model_, path);
            val inputs = val::object();
            for (const auto& [name, values] : voice.inputs) inputs.set(name, typed_array("Float64Array", values));
            val out = val::object();
            out.set("name", voice.name);
            out.set("inputs", inputs);
            return out;
        });
    }

    // The raw driver call: `inputs` are numbers and arrays of numbers by name.
    val call(const std::string& fn, const val& inputs) {
        return guarded([&] {
            require_driver("call");
            const Value result = bridge_->call(fn, to_inputs(inputs));
            return from_value(result);
        });
    }

private:
    void require_driver(const char* what) const {
        if (driver_.empty()) throw std::runtime_error(std::string("this GGUF carries no driver script, so there is nothing to ") + what + ".");
    }
    const loom::Tokenizer& tokenizer() const {
        if (tokenizer_ == nullptr) {
            throw std::runtime_error("this GGUF carries no tokenizer vocabulary; its driver takes ids directly.");
        }
        return *tokenizer_;
    }
    loom::text::GenerateOptions generate_options(const val& o) const {
        loom::text::GenerateOptions options;
        options.max_new_tokens = opt<uint32_t>(o, "max_new_tokens").value_or(64);
        options.eos_token = opt<int32_t>(o, "eos_token").value_or(loom::text::GenerateOptions::kEosFromFile);
        options.temperature = opt<float>(o, "temperature");
        options.top_k = opt<int32_t>(o, "top_k");
        options.top_p = opt<float>(o, "top_p");
        options.seed = opt<uint32_t>(o, "seed");
        if (!o.isUndefined() && !o.isNull() && !o["inputs"].isUndefined()) options.extra_inputs = to_inputs(o["inputs"]);
        return options;
    }

    // Declared in the order loom-py's binding gives and for its reason: each outlives what holds it.
    std::unique_ptr<loom::Device> device_;
    std::unique_ptr<loom::GgufModel> model_;
    loom::ModuleCaches caches_;
    std::unique_ptr<loom::LoomLuaBridge> bridge_;
    std::unique_ptr<loom::Tokenizer> tokenizer_;
    std::unique_ptr<loom::ChatTemplate> chat_;
    std::string driver_;
};

Model* load_model(const std::string& path, const std::string& device) {
    return guarded([&] { return new Model(path, device); });
}

std::string devices() {
    json out = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        out.push_back({{"name", ggml_backend_dev_name(d)}, {"description", ggml_backend_dev_description(d)},
                       {"type", static_cast<int>(ggml_backend_dev_type(d))}});
    }
    return out.dump();
}

} // namespace

#ifdef LOOM_WEB_JSPI
#define LOOM_ASYNC , emscripten::async()
#else
#define LOOM_ASYNC
#endif

EMSCRIPTEN_BINDINGS(loom) {
    using namespace emscripten;
    function("loadModel", &load_model, allow_raw_pointers() LOOM_ASYNC);
    // Async too: the first call into ggml's registry registers the WebGPU backend, which waits on the adapter.
    function("devices", &devices LOOM_ASYNC);
    function("jspi", optional_override([] {
#ifdef LOOM_WEB_JSPI
        return true;
#else
        return false;
#endif
    }));

    class_<TokenStream>("TokenStream")
        .function("next", &TokenStream::next LOOM_ASYNC)
        .function("tokens", &TokenStream::tokens)
        .function("finished", &TokenStream::finished)
        .function("stoppedOnEos", &TokenStream::stopped_on_eos)
        .function("reset", &TokenStream::reset LOOM_ASYNC);

    class_<SpeechStream>("SpeechStream")
        .function("next", &SpeechStream::next LOOM_ASYNC)
        .function("finished", &SpeechStream::finished)
        .function("piecesDone", &SpeechStream::pieces_done)
        .function("piecesTotal", &SpeechStream::pieces_total)
        .function("reset", &SpeechStream::reset LOOM_ASYNC);

    class_<Model>("Model")
        .function("contract", &Model::contract)
        .function("architecture", &Model::architecture)
        .function("deviceName", &Model::device_name)
        .function("deviceDescription", &Model::device_description)
        .function("driverSource", &Model::driver_source)
        .function("takesVoiceFiles", &Model::takes_voice_files)
        .function("hparamStr", &Model::hparam_str)
        .function("kvStr", &Model::kv_str)
        .function("tokenizerKind", &Model::tokenizer_kind)
        .function("encode", &Model::encode)
        .function("decodeBytes", &Model::decode_bytes)
        .function("chatRoles", &Model::chat_roles)
        .function("applyChatTemplate", &Model::apply_chat_template)
        .function("eosTokenIds", &Model::eos_token_ids)
        .function("generate", &Model::generate LOOM_ASYNC)
        .function("canStreamTokens", &Model::can_stream_tokens)
        .function("openTokenStream", &Model::open_token_stream, allow_raw_pointers() LOOM_ASYNC)
        .function("transcribe", &Model::transcribe LOOM_ASYNC)
        .function("canStreamSpeech", &Model::can_stream_speech)
        .function("openSpeechStream", &Model::open_speech_stream, allow_raw_pointers() LOOM_ASYNC)
        .function("loadVoice", &Model::load_voice)
        .function("call", &Model::call LOOM_ASYNC);
}
