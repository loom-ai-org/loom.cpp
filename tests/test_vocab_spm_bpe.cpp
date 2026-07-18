// Vocab correctness test for the new SentencePiece BPE path ("tokenizer.ggml.model" == "llama", see
// Vocab::encode_bpe): loads the real nvidia/parakeet-tdt-0.6b-v3 checkpoint's real tokenizer.model
// (converted by tools/convert_nemo/convert_parakeet_tdt.py) and checks Vocab::encode/decode against the
// actual `sentencepiece` Python library loaded against the SAME .model file -- same "no meaningful toy
// substitute" reasoning as test_vocab.cpp's own UGM test, since correctness here is entirely about
// faithfully reproducing this specific model's real greedy merge-by-score behavior.
//
// Expected id sequences below were generated via:
//   python3 -c "import sentencepiece as spm; sp = spm.SentencePieceProcessor(); \
//       sp.Load('<tokenizer.model>'); print(sp.EncodeAsIds('<text>'))"
// against the real tokenizer.model extracted from parakeet-tdt-0.6b-v3.nemo. Unlike test_vocab.cpp's
// small Conformer-CTC vocab, none of these test strings hit <unk> (id 0) -- this real 8192-piece BPE
// vocabulary has broad enough Unicode/punctuation coverage that decode can be checked on every case here.

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
    const char* dir_env = std::getenv("LOOM_PARAKEET_TDT_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/parakeet_tdt_model";
    const std::string gguf_path = dir + "/gguf/parakeet_encoder.gguf";
    if (!path_exists(gguf_path)) {
        std::fprintf(stderr,
                      "skipping: real Parakeet-TDT fixture not found at '%s' (set LOOM_PARAKEET_TDT_DIR "
                      "or see tools/convert_nemo/ to produce one)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 8192);
    LOOM_CHECK(vocab->unk_id() == 0);

    {
        const auto ids = vocab->encode("the cat sat");
        LOOM_CHECK((ids == std::vector<int32_t>{506, 6911, 3704}));
        LOOM_CHECK(vocab->decode(ids) == "the cat sat");
    }
    {
        const auto ids = vocab->encode("hello world");
        LOOM_CHECK((ids == std::vector<int32_t>{303, 3164, 2088, 2493}));
        LOOM_CHECK(vocab->decode(ids) == "hello world");
    }
    {
        // Exercises add_space_prefix's lowercasing-via-charsmap-normalization AND real merge chains
        // (e.g. "Hello" -> ['He','llo'] rather than single-character pieces).
        const auto ids = vocab->encode("Hello World!");
        LOOM_CHECK((ids == std::vector<int32_t>{425, 3164, 499, 294, 2493, 8020}));
        LOOM_CHECK(vocab->decode(ids) == "Hello World!");
    }
    {
        // Accented character (non-ASCII) genuinely exercises the XCDA charsmap walk, not just the
        // identity path.
        const auto ids = vocab->encode("café");
        LOOM_CHECK((ids == std::vector<int32_t>{298, 1389, 7906}));
        LOOM_CHECK(vocab->decode(ids) == "café");
    }
    {
        // This checkpoint's real remove_extra_whitespaces is FALSE (confirmed against the real
        // ModelProto, unlike Conformer-CTC-small's UGM vocab, which has it TRUE) -- runs of spaces are
        // preserved, not collapsed. Exercises that this Vocab instance picked up its own real KV value
        // rather than assuming the UGM-typical default.
        const auto ids = vocab->encode("  multiple   spaces  ");
        LOOM_CHECK((ids == std::vector<int32_t>{7863, 7863, 2310, 321, 3725, 7863, 7863, 485, 566, 283, 7863, 7863}));
        LOOM_CHECK(vocab->decode(ids) == "  multiple   spaces  ");
    }
    {
        const auto ids = vocab->encode("a");
        LOOM_CHECK((ids == std::vector<int32_t>{279}));
        LOOM_CHECK(vocab->decode(ids) == "a");
    }
    {
        const auto ids = vocab->encode("");
        LOOM_CHECK(ids.empty());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
