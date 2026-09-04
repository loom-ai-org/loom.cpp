// Synthesise one utterance from any of loom's four TTS families and write it as a WAV, on any device.
//
// NOT part of the build -- a standalone measurement, kept because it is the FRONT HALF OF THE ASR
// ORACLE and that oracle is the only trustworthy verdict on a TTS change (Retro-006: cosine 0.996
// shipped noise; P4.13/P4.28: correlation 0.025 shipped speech). Without it, "does this still say the
// sentence" has to be re-implemented from scratch every time, which is how it stops being asked.
//
//   g++ -O2 -std=c++17 -I include -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/tts_synth.cpp -o tts_synth \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//
//   scripts/tts_ids.py "Hey, can you shut down the computer, my friend?" <model.gguf> ids.txt
//   ./tts_synth <model.gguf> <vits|matcha|kokoro|styletts2|dia> ids.txt <rate> out.wav [--device gpu] \
//               [--ref-s kokoro.ref_s.txt] [--codec dac_44khz.gguf] [--frames N]
//
// `text:<sentence>` in the ids slot encodes with the MODEL's own embedded vocabulary instead of
// reading a file -- for the families that carry one (Dia's bytes, Supertonic's codepoints), which is
// the only sane way to drive them since no external phonemizer is involved.
//   # then the BACK half of the oracle, which resamples and transcribes:
//   ~/.venvs/piper/bin/python scripts/asr_oracle.py out.wav --expect "hey can you shut down"
//
// The families differ only in what `infer` is called with, which is what the gate tests show and all
// this file really encodes: vits {token_ids, seed, noise_scale, length_scale, noise_scale_w},
// matcha {tokens, n_steps, seed}, kokoro {input_ids, ref_s, speed, seed},
// styletts2 {input_ids, diffusion_steps, seed}, dia {tokens, seed, max_new_tokens}. Sample rates:
// vits/matcha 22050, kokoro/styletts2 24000, dia 44100 (kokoro and dia's codec declare
// `loom.sample_rate`; the others do not, hence the argument).
//
// **`dia` is TWO FILES and so needs `--codec`.** It emits codec tokens, not audio (ADR-020/ADR-022),
// so this runs the LM, then feeds its frame-major codes into a DAC GGUF. It is also the one family
// here that SAMPLES by default -- the file declares the checkpoint's own temperature/top-k/top-p and
// classifier-free guidance -- which is exactly why an ASR oracle is the only verdict on it, and why
// `--seed` matters here in a way it does not for a greedy family.
//
// TWO THINGS IT PRINTS BESIDES THE AUDIO, both of which have caught something:
//
//   * peak and rms. Real speech lands near +-0.3; anything leaving [-1, 1] means the conditioning is
//     wrong rather than the vocoder (see feedback on the Kokoro noise-voice bug).
//   * `device_report()`, on a device build. A scheduler that hands every node back to the CPU produces
//     exactly the same correct audio, so a GPU claim without this line is vacuous -- it is what turned
//     "a folded quantized kernel runs on Vulkan" into a real result in P4.13 (0 fallback nodes).
//
// `synthetic:N` instead of an ids file builds a BOS/blank/EOS sequence of N phonemes with no phonemizer
// involved. That is not speech and must never be transcribed; it exists to sweep LENGTH, which is how
// P4.28 checked that removing VITS's static relative-position pad kept both of the real code's branches
// (n=2 and 4 are the crop branch, 2202 and 5002 are past the bound the old export threw at).
#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static void w32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void w16(FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

static std::vector<double> read_numbers(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<double> v; double x;
    while (in >> x) v.push_back(x);
    return v;
}

// The BOS/blank/EOS shape a piper-style phoneme sequence has, with arbitrary ids in the middle. See the
// header: for sweeping length only.
static std::vector<double> synthetic_ids(int n) {
    std::vector<double> ids{1};
    for (int i = 0; i < n; ++i) { ids.push_back(20 + (i % 90)); ids.push_back(0); }
    ids.push_back(2);
    return ids;
}

// `loom.codec.n_codebooks`, or 0 for a file that declares none. Both halves of the family-10 pair
// write it, which is what lets a host check that a pair fits before running a generation into a shape
// error -- the same check `Codes2Speech` makes in loom-py.
static uint32_t codebooks(const loom::GgufModel& model) {
    try {
        return model.hparam_u32("codec.n_codebooks");
    } catch (const std::exception&) {
        return 0;
    }
}

// The model's own embedded vocabulary, for a family that carries one.
//
// Dispatched on `tokenizer.ggml.model` the way `loom_cli` and loom-py's binding both do, rather than
// through one base class, because there isn't one: `loom::Vocab` covers the SentencePiece families and
// `loom::ByteVocab` is its own type. Two kinds are enough for every model this tool drives -- Dia's
// bytes and Supertonic's codepoints -- and a third raises here instead of encoding wrongly.
static std::vector<double> encode_with_model(loom::GgufModel& model, const std::string& text) {
    std::vector<int32_t> encoded;
    const std::string kind = model.has_kv("tokenizer.ggml.model")
        ? model.kv_str("tokenizer.ggml.model") : std::string{};
    if (kind == "byt5") {
        encoded = loom::ByteVocab::load(model)->encode(text);
    } else if (auto vocab = loom::Vocab::load(model)) {
        encoded = vocab->encode(text);
    } else {
        std::fprintf(stderr, "this GGUF embeds no vocabulary this tool can drive ('%s'), so `text:` "
                              "cannot encode -- pass an ids file\n", kind.c_str());
        std::exit(2);
    }
    std::vector<double> ids;
    for (int32_t id : encoded) ids.push_back(static_cast<double>(id));
    return ids;
}

static void write_wav(const std::string& path, const std::vector<double>& audio, uint32_t rate) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); std::exit(1); }
    const uint32_t n = static_cast<uint32_t>(audio.size());
    std::fwrite("RIFF", 1, 4, f); w32(f, 36 + n * 2); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(f, 16); w16(f, 1); w16(f, 1);
    w32(f, rate); w32(f, rate * 2); w16(f, 2); w16(f, 16);
    std::fwrite("data", 1, 4, f); w32(f, n * 2);
    for (double s : audio) {
        const double c = std::max(-1.0, std::min(1.0, s));
        w16(f, static_cast<uint16_t>(static_cast<int16_t>(std::lround(c * 32767.0))));
    }
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <gguf> <vits|matcha|kokoro|styletts2> <ids.txt|synthetic:N> <rate> <out.wav|->\n"
            "          [--device NAME] [--ref-s FILE] [--seed N]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1], family = argv[2], ids_arg = argv[3], out = argv[5];
    const uint32_t rate = static_cast<uint32_t>(std::atoi(argv[4]));
    std::string device_name = "cpu", ref_s_path, codec_path, codes_out;
    std::unordered_map<std::string, double> knobs;
    double seed = 42.0;
    int max_frames = 0;
    for (int i = 6; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--device") == 0) device_name = argv[++i];
        else if (std::strcmp(argv[i], "--ref-s") == 0) ref_s_path = argv[++i];
        else if (std::strcmp(argv[i], "--seed") == 0) seed = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--codec") == 0) codec_path = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0) max_frames = std::atoi(argv[++i]);
        // The decoding knobs, for a family that declares its own. Named ONLY when given, so an
        // unnamed one stays the file's -- and each is here because bisecting a sampled family means
        // being able to move exactly one of them. `--temperature 0` is the greedy decode a reference
        // comparison uses; a very small one instead exercises the SAMPLING path and must agree with
        // it, which is how the draw itself gets tested apart from the distribution.
        else if (std::strcmp(argv[i], "--temperature") == 0) knobs["temperature"] = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--top-k") == 0) knobs["top_k"] = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--top-p") == 0) knobs["top_p"] = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--guidance") == 0) knobs["guidance_scale"] = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dump-codes") == 0) codes_out = argv[++i];
    }

    try {
        loom::Device device = loom::Device::open(device_name);
        loom::Backends backends = device.backends();
        auto model = loom::GgufModel::load(gguf, backends.primary);
        if (!model) { std::fprintf(stderr, "load failed: %s\n", gguf.c_str()); return 1; }

        // Resolved after the model is open, because `text:` needs the file's own vocabulary. The other
        // two forms do not, but keeping all three here means there is one place ids come from.
        std::vector<double> ids;
        if (ids_arg.rfind("synthetic:", 0) == 0) {
            ids = synthetic_ids(std::atoi(ids_arg.c_str() + 10));
        } else if (ids_arg.rfind("text:", 0) == 0) {
            ids = encode_with_model(*model, ids_arg.substr(5));
        } else {
            ids = read_numbers(ids_arg);
        }
        if (ids.empty()) { std::fprintf(stderr, "no ids\n"); return 2; }

        // `Session` registers every topology the file declares, whatever the family -- kokoro has 27,
        // vits has 2, Dia under guidance has 5 -- and allocates the caches they ask for. Written out
        // here as a loop until family 10 arrived, which needs a KV cache and needs one module to have
        // its OWN (ADR-023); that rule decides correctness, so it has one implementation.
        loom::Session session(*model, backends);
        loom::LoomLuaBridge& bridge = session.bridge();

        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args;
        if (family == "vits") {
            // The three scales PINNED, as bench_vits_loom.cpp pins them and for the same reason: VITS's
            // duration predictor is stochastic, so an unpinned pair of runs synthesises two different
            // utterances and nothing downstream is comparable.
            args = {{"token_ids", ids}, {"seed", seed},
                    {"noise_scale", 0.0}, {"length_scale", 1.0}, {"noise_scale_w", 0.0}};
        } else if (family == "matcha") {
            args = {{"tokens", ids}, {"n_steps", 10.0}, {"seed", seed}};
        } else if (family == "styletts2") {
            args = {{"input_ids", ids}, {"diffusion_steps", 5.0}, {"seed", seed}};
        } else if (family == "kokoro") {
            if (ref_s_path.empty()) {
                std::fprintf(stderr, "kokoro needs --ref-s (scripts/tts_ids.py writes one from the "
                                     "file's own loom.default_style.ref_s)\n");
                return 2;
            }
            args = {{"input_ids", ids}, {"ref_s", read_numbers(ref_s_path)},
                    {"speed", 1.0}, {"seed", seed}};
        } else if (family == "dia") {
            // Everything else is the FILE's: it declares this checkpoint's own temperature, top-k,
            // top-p and guidance scale, and naming any of them here would be this tool overriding the
            // model with a number nobody chose. The seed is the exception, because reproducing a
            // sampled utterance is the whole point of an oracle run.
            args = {{"tokens", ids}, {"seed", seed}};
            if (max_frames > 0) args["max_new_tokens"] = static_cast<double>(max_frames);
            for (const auto& [name, value] : knobs) args[name] = value;
            if (codec_path.empty()) {
                std::fprintf(stderr, "dia emits codec tokens, not audio -- pass --codec <dac.gguf>\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown family '%s'\n", family.c_str());
            return 2;
        }

        // BY VALUE, and the `const auto&` this used to be was a real bug that only clang reports.
        // `bridge.call` returns a `Value` by value; binding a reference to the vector INSIDE that
        // temporary leaves the reference dangling at the end of the full expression. gcc happened to
        // leave the storage readable and every Linux run looked right; on macOS the same line printed
        // `samples=0 peak=0.0000` for all four families, which reads exactly like a broken export.
        // -Wdangling-gsl names it; the lesson is that a silent zero is what a dangling read looks like.
        std::vector<double> audio = std::get<std::vector<double>>(bridge.call("infer", args));

        // The second half of the pair. Dia's `infer` returned CODES -- frame-major, `n_codebooks`
        // wide -- and the codec is a separate GGUF the host chains, which is what ADR-022 decided and
        // what makes this two `Session`s rather than one. Loaded AFTER the LM has run, so the two
        // 6.4 GB and 217 MB files are never resident together for longer than the handover.
        if (family == "dia") {
            std::printf("dia        codes=%zu frames=%zu\n", audio.size(),
                        audio.size() / std::max<size_t>(1, codebooks(*model)));
            // The codes themselves, when asked for: they are the intermediate the pair is joined on,
            // and a difference between two decodes is attributable there and nowhere downstream.
            if (!codes_out.empty()) {
                std::ofstream f(codes_out);
                for (double c : audio) f << static_cast<int64_t>(c) << "\n";
            }
            auto codec = loom::GgufModel::load(codec_path, backends.primary);
            if (!codec) { std::fprintf(stderr, "load failed: %s\n", codec_path.c_str()); return 1; }
            if (codebooks(*codec) != codebooks(*model)) {
                std::fprintf(stderr, "these two files do not fit: the LM emits %u codebooks per frame "
                                      "and the codec takes %u\n", codebooks(*model), codebooks(*codec));
                return 1;
            }
            loom::Session codec_session(*codec, backends);
            audio = std::get<std::vector<double>>(
                codec_session.bridge().call("infer", {{"codes", audio}}));
        }
        double peak = 0.0, energy = 0.0;
        for (double s : audio) { peak = std::max(peak, std::fabs(s)); energy += s * s; }
        std::printf("%-10s n_ids=%zu samples=%zu peak=%.4f rms=%.5f\n", family.c_str(), ids.size(),
                    audio.size(), peak, std::sqrt(energy / std::max<size_t>(1, audio.size())));
        for (const auto& m : bridge.device_report()) {
            std::printf("  module=%-20s splits=%d device_nodes=%zu fallback_nodes=%zu\n",
                        m.module.c_str(), m.splits, m.device_nodes, m.fallback_nodes);
        }
        if (out != "-") write_wav(out, audio, rate);
    } catch (const std::exception& e) {
        // Printed rather than thrown, because a LENGTH sweep wants the message: the engine's VIEW
        // bounds check is what reported VITS's old ~2053-token ceiling, naming the tensor and offset.
        std::printf("THREW: %s\n", e.what());
        return 1;
    }
    return 0;
}
