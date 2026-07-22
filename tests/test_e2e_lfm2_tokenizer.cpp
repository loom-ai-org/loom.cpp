// Validates the MIL-exporter's new tokenizer-export path (EXPORT-BACKLOG.md item 4) end to end: loads
// the real LFM2-350M GGUF produced by export_lfm2_monolithic.py (which now also writes
// "tokenizer.ggml.*" KVs via tools/loom_mil_compiler/bpe_tokenizer_export.py) and checks
// loom::BpeVocab::encode/decode against the real HF `AutoTokenizer` loaded on the SAME checkpoint --
// same "no meaningful toy substitute" reasoning as test_vocab_spm_bpe.cpp, since correctness here is
// entirely about faithfully reproducing this specific tokenizer's real behavior: its "llama3"-style
// grouped-up-to-3-digit pretokenizer regex (distinct from Qwen2/Qwen3's single-digit regex, see
// bpe_vocab.h's own doc comment) and its BOS-token auto-prepending (LFM2, unlike Qwen3, sets
// tokenizer_config.json's add_bos_token=true).
//
// Not generated at ctest time (needs the real LFM2-350M checkpoint + coremltools) -- skips cleanly if
// the fixture isn't present, same convention as test_e2e_lfm2_mil_export.cpp. To (re)generate:
// `~/.venvs/piper/bin/python3 export_lfm2_monolithic.py` from the repo root (writes
// lfm2_350m_monolithic.gguf there), or point LOOM_LFM2_MONOLITHIC_GGUF at an existing copy.
//
// Expected id sequences below were generated via:
//   python3 -c "from transformers import AutoTokenizer; \
//       tok = AutoTokenizer.from_pretrained('/home/flavio/Dev/models/lfm2-350m'); \
//       print(tok.encode('<text>'))"
// against the real checkpoint directory.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdlib>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

} // namespace

int main() {
    const char* mono_env = std::getenv("LOOM_LFM2_MONOLITHIC_GGUF");
    const std::string gguf_path = mono_env != nullptr ? mono_env : "lfm2_350m_monolithic.gguf";
    if (!path_exists(gguf_path)) {
        std::fprintf(stderr,
                      "skipping: real LFM2-350M fixture not found at '%s' (set LOOM_LFM2_MONOLITHIC_GGUF, "
                      "or run export_lfm2_monolithic.py from the repo root to produce one)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 64400);
    LOOM_CHECK(vocab->bos_id() == 1);
    LOOM_CHECK(vocab->eos_id() == 7);

    // Every case includes the auto-prepended BOS (id 1) -- confirms tokenizer.ggml.add_bos_token is both
    // written by the exporter and actually honored by BpeVocab::encode, not just carried through as an
    // inert KV.
    {
        const auto ids = vocab->encode("hello world");
        LOOM_CHECK((ids == std::vector<int32_t>{1, 52572, 2031}));
        LOOM_CHECK(vocab->decode(std::vector<int32_t>(ids.begin() + 1, ids.end())) == "hello world");
    }
    {
        // "2024"/"365": exercises the "llama3"-style grouped-up-to-3-digit pretokenizer alternative
        // (\p{N}{1,3}) -- LFM2's real regex, distinct from Qwen2/Qwen3's single-digit \p{N} (see
        // test_bpe_vocab.cpp's own "12" case, which deliberately splits digit-by-digit for THAT family).
        const auto ids = vocab->encode("The year 2024 has 365 days.");
        LOOM_CHECK((ids == std::vector<int32_t>{1, 1098, 1423, 730, 1718, 529, 1178, 730, 19869, 3378, 523}));
        LOOM_CHECK(vocab->decode(std::vector<int32_t>(ids.begin() + 1, ids.end())) == "The year 2024 has 365 days.");
    }
    {
        const auto ids = vocab->encode("don't");
        LOOM_CHECK((ids == std::vector<int32_t>{1, 16203, 1901}));
        LOOM_CHECK(vocab->decode(std::vector<int32_t>(ids.begin() + 1, ids.end())) == "don't");
    }
    {
        // Non-Latin script (CJK), same "raw byte-level fallback round-trips" signal test_bpe_vocab.cpp's
        // own CJK case checks, here against the real 64400-piece vocab instead of a tiny fixture.
        const std::string cjk = "\xE4\xB8\xAD\xE6\x96\x87"; // "中文"
        const auto ids = vocab->encode(cjk);
        LOOM_CHECK((ids == std::vector<int32_t>{1, 2377, 5467}));
        LOOM_CHECK(vocab->decode(std::vector<int32_t>(ids.begin() + 1, ids.end())) == cjk);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
