// Times a GGUF's own driver -- any model, any device -- by calling `infer` directly.
//
//   g++ -O2 -std=c++17 -I include -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/bench_driver.cpp \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread \
//       -o bench_driver
//   ./bench_driver <model.gguf> <device> <nrun> name=1,2,3 [name=num:10] [name=rand:800:2047] ...
//
// **Why this exists rather than `loom_cli`.** The CLI has two output modes -- transcription and token
// generation -- so it cannot run a model whose answer is audio at all: every TTS and codec family in
// the zoo is unreachable from it, on any device. That is fine for a CLI (nobody wants a WAV on stdout)
// and useless for measuring, because those are exactly the families whose drivers do the most work
// per call. This calls the driver's entry point with whatever inputs it declares and times the call.
//
// MODEL LOAD IS OUTSIDE THE TIMER, for `bench_asr_loom.cpp`'s reason: a load is a large and unstable
// fraction of a short call, and it is the same term in both arms of any comparison.
//
// Two arms are run INTERLEAVED by the caller (A B B A), not by this program: it times one model, and
// comparing two is a shell loop. What it does guarantee is that the answer is printed -- a checksum of
// the returned array -- because two builds that return different values are not doing equal work, and
// a ratio over unequal work measures nothing (Retro-010).
#include "loom/loom.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<double> parse_list(const std::string& spec) {
    std::vector<double> out;
    size_t start = 0;
    while (start <= spec.size()) {
        const size_t comma = spec.find(',', start);
        const std::string piece = spec.substr(start, comma == std::string::npos ? std::string::npos
                                                                                : comma - start);
        if (!piece.empty()) out.push_back(std::strtod(piece.c_str(), nullptr));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

// `rand:N:MAX` -- N deterministic ids in [0, MAX]. Deterministic because a benchmark that draws
// different inputs per arm is comparing two different amounts of work.
std::vector<double> parse_rand(const std::string& spec) {
    const size_t colon = spec.find(':');
    const long n = std::strtol(spec.c_str(), nullptr, 10);
    const long max = colon == std::string::npos ? 255 : std::strtol(spec.c_str() + colon + 1, nullptr, 10);
    std::vector<double> out(static_cast<size_t>(n));
    unsigned long state = 12345;
    for (double& v : out) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<double>((state >> 33) % static_cast<unsigned long>(max + 1));
    }
    return out;
}

double checksum(const loom::LoomLuaBridge::Value& v) {
    if (const auto* arr = std::get_if<std::vector<double>>(&v)) {
        double sum = 0.0;
        for (double x : *arr) sum += std::abs(x);
        return sum;
    }
    return std::get<double>(v);
}

size_t value_size(const loom::LoomLuaBridge::Value& v) {
    if (const auto* arr = std::get_if<std::vector<double>>(&v)) return arr->size();
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <device> <nrun> name=1,2,3 [name=num:10] "
                     "[name=rand:N:MAX] ...\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string device_spec = argv[2];
    const int nrun = std::atoi(argv[3]);

    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        const size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "bench_driver: '%s' is not name=value\n", arg.c_str());
            return 2;
        }
        const std::string name = arg.substr(0, eq), spec = arg.substr(eq + 1);
        if (spec.rfind("num:", 0) == 0) {
            inputs[name] = std::strtod(spec.c_str() + 4, nullptr);
        } else if (spec.rfind("rand:", 0) == 0) {
            inputs[name] = parse_rand(spec.substr(5));
        } else {
            inputs[name] = parse_list(spec);
        }
    }

    loom::Device device = loom::Device::open(device_spec);
    std::printf("device: %s\n", device.name().c_str());

    auto model = loom::GgufModel::load(path, device.backends().primary);
    loom::Session session(*model, device.backends());

    // One untimed call: the first build allocates every graph and compute buffer, which is a
    // per-process cost and not per-call work.
    const auto warm = session.bridge().call("infer", inputs);
    std::printf("returned %zu value(s), checksum %.6f\n", value_size(warm), checksum(warm));

    double best = 1e30, total = 0.0;
    for (int run = 0; run < nrun; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto out = session.bridge().call("infer", inputs);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (std::abs(checksum(out) - checksum(warm)) > 1e-6 * std::abs(checksum(warm))) {
            std::fprintf(stderr, "bench_driver: run %d returned a different answer -- not equal work\n",
                          run);
            return 1;
        }
        best = std::min(best, ms);
        total += ms;
        std::printf("  run %2d %8.1f ms\n", run, ms);
    }
    std::printf("RESULT %s best %.1f ms mean %.1f ms over %d run(s)\n",
                path.c_str(), best, total / nrun, nrun);
    return 0;
}
