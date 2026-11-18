// PLaMo 3 translate (ADR-132) against the checkpoint's own tokenizer, chat template and greedy generate.
//
//   1. Tokenizer: every text in `tokenizer_cases.json` encodes to the reference's ids (`Plamo3Tokenizer`),
//      and decodes back to itself.
//   2. Prompt: each case's `translation_prompt`, called with the case's pieces encoded by the file's own
//      vocabulary, is `apply_chat_template`'s ids exactly.
//   3. Greedy translation: the engine's loop over the prompt makes the reference's ids, through the stop
//      token; and the token stream (ADR-099) makes the same ids three at a time.
//
// LOOM_PLAMO_GGUF      the export
// LOOM_PLAMO_REF_DIR   tokenizer_cases.json and one directory per case (`scripts/plamo_translate_reference.py`)

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"
#include "loom/core/plamo_vocab.h"
#include "loom/core/text_generate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

std::vector<double> npy(const std::string& path) {
    std::vector<int64_t> shape;
    const auto v = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(v.begin(), v.end());
}

std::vector<double> ids_of(const loom::PlamoVocab& vocab, const std::string& s) {
    const auto ids = vocab.encode(s);
    return std::vector<double>(ids.begin(), ids.end());
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_PLAMO_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_PLAMO_REF_DIR");
    if (gguf == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_PLAMO_GGUF and LOOM_PLAMO_REF_DIR\n");
        return 77;
    }
    const std::string dir = ref_env;
    loom::Device device = loom::Device::open(loom_test::test_device_name());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    auto vocab = loom::PlamoVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    // 1. The tokenizer.
    {
        const auto cases = nlohmann::json::parse(std::ifstream(dir + "/tokenizer_cases.json"));
        size_t equal = 0;
        for (const auto& c : cases) {
            const std::string text = c["text"].get<std::string>();
            const auto want = c["ids"].get<std::vector<int32_t>>();
            const auto got = vocab->encode(text);
            const bool ok = got == want;
            equal += ok;
            LOOM_CHECK(ok);
            LOOM_CHECK(vocab->decode(got) == text);
        }
        std::fprintf(stderr, "  tokenizer: %zu/%zu texts encode to the reference's ids\n", equal, cases.size());
    }

    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    LOOM_CHECK(bridge.has_function("translation_prompt"));

    std::vector<std::string> case_dirs;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_directory()) case_dirs.push_back(entry.path().string());
    }
    std::sort(case_dirs.begin(), case_dirs.end());
    LOOM_CHECK(!case_dirs.empty());

    for (const std::string& d : case_dirs) {
        const auto c = nlohmann::json::parse(std::ifstream(d + "/case.json"));
        const std::string name = std::filesystem::path(d).filename().string();

        // 2. The prompt, from the pieces as a host hands them over.
        Inputs in;
        in["text"] = ids_of(*vocab, c["text"].get<std::string>());
        in["target_lang"] = ids_of(*vocab, c["target_lang"].get<std::string>());
        for (const char* key : {"source_lang", "context", "form"}) {
            if (c.contains(key) && !c[key].is_null()) in[key] = ids_of(*vocab, c[key].get<std::string>());
        }
        for (const char* key : {"glossary", "history"}) {
            const std::string sep = std::string(key) == "glossary" ? "=" : "|||";
            if (!c.contains(key) || c[key].empty()) continue;
            std::vector<double> ids, offsets{0.0};
            for (const auto& entry : c[key]) {
                const std::string s = entry.get<std::string>();
                const size_t at = s.find(sep);
                for (const std::string& part : {s.substr(0, at), s.substr(at + sep.size())}) {
                    const auto p = ids_of(*vocab, part);
                    ids.insert(ids.end(), p.begin(), p.end());
                    offsets.push_back(static_cast<double>(ids.size()));
                }
            }
            in[key] = ids;
            in[std::string(key) + "_offsets"] = offsets;
        }
        const auto prompt = std::get<std::vector<double>>(bridge.call("translation_prompt", in));
        const auto want_prompt = npy(d + "/prompt_ids.npy");
        LOOM_CHECK(prompt == want_prompt);
        std::fprintf(stderr, "  %s: prompt of %zu ids %s the template's\n", name.c_str(), prompt.size(),
                     prompt == want_prompt ? "==" : "!=");
        std::vector<int32_t> prompt_ids(want_prompt.begin(), want_prompt.end());

        // 3. Greedy, through the stop token.
        const auto want = npy(d + "/output_ids.npy");
        loom::text::GenerateOptions opts;
        // Room past the reference's length, so a stop rule that never fires shows as a longer answer.
        opts.max_new_tokens = static_cast<uint32_t>(want.size()) + 4;
        const auto got = loom::text::generate(bridge, *model, prompt_ids, opts);
        size_t same = 0;
        while (same < std::min(got.size(), want.size()) && got[same] == static_cast<int32_t>(want[same])) ++same;
        std::fprintf(stderr, "  %s: %zu of %zu generated ids the reference's: \"%s\"\n", name.c_str(), same,
                     want.size(), vocab->decode(got, true).c_str());
        // The reference's ids run through its stop token; the engine stops ON it and may or may not return it.
        LOOM_CHECK(same + 1 >= want.size());
        LOOM_CHECK(got.size() <= want.size());

        loom::text::GenerationStream stream(bridge, *model, prompt_ids, opts);
        std::vector<int32_t> streamed;
        while (!stream.finished()) {
            for (int32_t id : stream.next(3)) streamed.push_back(id);
        }
        LOOM_CHECK(streamed == got);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
