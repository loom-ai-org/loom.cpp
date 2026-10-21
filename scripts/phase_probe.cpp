// phase_probe: run ONE topology (or any Lua sequence) of an exported GGUF through its own Session, with
// inputs from files, via an ad-hoc Lua function loaded beside the driver. The fastest engine-side
// bisect tool, and ADR-078's tensor oracle (scripts/armv6_runfast_gate.sh).
//
//   ./phase_probe model.gguf out.f32 "<lua body>" name=@input.f32 name=3 ...
//   e.g. body: loom.run_subgraph_and_retain('encoder', {n_samples = #inputs.waveform, n_past = 0},
//              {waveform = inputs.waveform}); local o = loom.get_output('encoder', 1); return o
//
// Link against libloom_engine, libggml and libggml-base. Against a GGML_BACKEND_DL build (a wheel),
// the libggml-cpu*.so files must sit beside the executable.
#include "loom/loom.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
int main(int argc, char** argv) {
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i]; auto eq = a.find('=');
        std::string name = a.substr(0, eq), rest = a.substr(eq + 1);
        if (!rest.empty() && rest[0] == '@') {
            std::vector<double> v; std::ifstream f(rest.substr(1), std::ios::binary); float s;
            while (f.read(reinterpret_cast<char*>(&s), sizeof(float))) v.push_back(s);
            inputs[name] = v;
        } else inputs[name] = std::atof(rest.c_str());
    }
    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(argv[1], device.backends().primary);
    loom::Session session(*model, device.backends());
    session.bridge().load_script(std::string("function probe(inputs)\n") + argv[3] + "\nend\n");
    const auto out = session.bridge().call("probe", inputs);
    const auto* arr = std::get_if<std::vector<double>>(&out);
    FILE* g = std::fopen(argv[2], "wb");
    for (double d : *arr) { float f = static_cast<float>(d); std::fwrite(&f, sizeof f, 1, g); }
    std::fclose(g);
    std::fprintf(stderr, "probe returned %zu values\n", arr->size());
}
