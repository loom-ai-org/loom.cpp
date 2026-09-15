// dump_topology: run ONE of a GGUF's topologies directly and write its output as raw f32.
//
//   ./dump_topology <model.gguf> <device> <topology> <out.f32> \
//       --input <name>=@<file.f32> | --input <name>=<v1,v2,...> ... \
//       --axis  <name>=<value> ...
//
// **Why this exists beside `dump_driver`.** `dump_driver` runs a model's own `infer` and writes what
// it RETURNS, which for a classifier is the driver's answer rather than the tensor behind it -- a CTC
// driver returns collapsed token ids, so grading an export against a PyTorch oracle through it is a
// token oracle, and a wrong encoder decodes most tokens right (loom.cpp's own standing rule; family 3
// measured 71 of 80). This one goes one level down: it loads an extra Lua function into the same VM
// the driver lives in and calls `loom.run_subgraph` on a named topology, so what comes out is the
// graph's own output tensor. That is the oracle.
//
// Nothing here is family-specific: the inputs and the axis values are named on the command line, and
// the Lua is generated from those names.
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -I include -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/dump_topology.cpp \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread \
//       -o dump_topology
//   LD_LIBRARY_PATH=$PWD/build:$PWD/build/_deps/ggml-build/src ./dump_topology ...
//
#include "loom/loom.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<double> read_f32(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<double> out;
    float v = 0.0f;
    while (std::fread(&v, sizeof(float), 1, f) == 1) out.push_back(static_cast<double>(v));
    std::fclose(f);
    return out;
}

std::vector<double> parse_csv(const std::string& text) {
    std::vector<double> out;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t comma = text.find(',', pos);
        const std::string piece = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!piece.empty()) out.push_back(std::strtod(piece.c_str(), nullptr));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

// `<name>=<rest>`, or a false return when there is no `=`.
bool split_pair(const char* arg, std::string& name, std::string& rest) {
    const char* eq = std::strchr(arg, '=');
    if (eq == nullptr) return false;
    name.assign(arg, eq);
    rest.assign(eq + 1);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <device> <topology> <out.f32> "
            "[--input name=@file.f32|name=v1,v2,..] [--axis name=value]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], device_name = argv[2];
    const std::string topology = argv[3], out_path = argv[4];

    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    std::vector<std::string> input_names;
    std::vector<std::pair<std::string, std::string>> axes;
    for (int i = 5; i < argc; ++i) {
        const bool is_input = std::strcmp(argv[i], "--input") == 0;
        const bool is_axis = std::strcmp(argv[i], "--axis") == 0;
        if ((!is_input && !is_axis) || i + 1 >= argc) {
            std::fprintf(stderr, "unexpected argument %s\n", argv[i]);
            return 2;
        }
        std::string name, rest;
        if (!split_pair(argv[++i], name, rest)) {
            std::fprintf(stderr, "expected name=value, got %s\n", argv[i]);
            return 2;
        }
        if (is_axis) { axes.emplace_back(name, rest); continue; }
        inputs[name] = rest.rfind('@', 0) == 0 ? read_f32(rest.substr(1)) : parse_csv(rest);
        input_names.push_back(name);
        std::fprintf(stderr, "input %s: %zu values\n", name.c_str(),
                     std::get<std::vector<double>>(inputs[name]).size());
    }

    // The generated entry point. Every axis whose value was not given on the command line would be a
    // silent zero in the axis table, so they are all named explicitly and nothing is defaulted here.
    std::string lua = "function dump_topology_entry(inputs)\n  return loom.run_subgraph('" + topology + "', {";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i != 0) lua += ", ";
        lua += axes[i].first + " = " + axes[i].second;
    }
    lua += "}, {";
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (i != 0) lua += ", ";
        lua += input_names[i] + " = inputs." + input_names[i];
    }
    lua += "})\nend\n";
    std::fprintf(stderr, "%s", lua.c_str());

    loom::Device device = loom::Device::open(device_name);
    auto model = loom::GgufModel::load(model_path, device.backends().primary);
    loom::Session session(*model, device.backends());
    session.bridge().load_script(lua);

    const auto out = session.bridge().call("dump_topology_entry", inputs);
    const auto* values = std::get_if<std::vector<double>>(&out);
    if (values == nullptr) { std::fprintf(stderr, "topology did not return an array\n"); return 1; }
    std::fprintf(stderr, "%zu output values\n", values->size());

    FILE* g = std::fopen(out_path.c_str(), "wb");
    if (g == nullptr) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    for (const double d : *values) { const float f = static_cast<float>(d); std::fwrite(&f, sizeof(float), 1, g); }
    std::fclose(g);
    return 0;
}
