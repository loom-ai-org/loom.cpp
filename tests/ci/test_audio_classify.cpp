// The audio classifier and embedder doors (family 13, ADR-062): how a flat driver answer becomes rows,
// and what is refused -- per frame and per clip, for classes and for embeddings.
//
// Every fixture's driver answers from the sample count alone, so nothing here runs a graph. What is
// under test is the CUT -- rows of the declared labels or embedding width, the time each frame row
// covers, and the refusals -- which is exactly the part two hosts reshaping on their own would get to
// disagree about.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>

namespace {

std::unique_ptr<loom::GgufModel> load(const char* name, ggml_backend_t backend) {
    return loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/" + name, backend);
}

template <typename F>
bool throws(F&& f) {
    try {
        f();
    } catch (const loom::Error&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    const std::vector<float> ten(10, 0.5f);

    // ---- Per frame: rows of the labels, with the time each one covers ------------------------------
    {
        auto model = load("audio_frame_classifier.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const loom::ModelContract contract = loom::ModelContract::read(*model);
        LOOM_CHECK(contract.interface_name() == "speech2class");
        LOOM_CHECK(contract.output_granularity == loom::granularity::FRAME);
        LOOM_CHECK(contract.sample_rate == 16000);  // written as i32, like every exported contract int
        loom::Session session(*model, backend.get());

        // Ten samples, one row per four: two rows, and the remainder is not a row.
        const auto result = loom::audio::classify(session.bridge(), *model, ten);
        LOOM_CHECK(result.n_rows == 2);
        LOOM_CHECK(result.labels.size() == 2 && result.labels[1] == "speech");
        // Row 0 is [0.25, 0.75], row 1 the other way round -- row-major, label-minor.
        LOOM_CHECK(result.at(0, 1) == 0.75f && result.at(0, 0) == 0.25f);
        LOOM_CHECK(result.at(1, 1) == 0.25f && result.at(1, 0) == 0.75f);
        // 4 frames/s from 0.5 s: frame i starts at 0.5 + i / 4.
        LOOM_CHECK(std::fabs(result.row_start(0) - 0.5) < 1e-9);
        LOOM_CHECK(std::fabs(result.row_start(1) - 0.75) < 1e-9);

        // The wrong door for this file refuses rather than handing back a "vector".
        LOOM_CHECK(throws([&] { loom::audio::embed(session.bridge(), *model, ten); }));
        // And no audio is no answer.
        LOOM_CHECK(throws([&] { loom::audio::classify(session.bridge(), *model, {}); }));
    }

    // ---- Per clip: one row, and no frame times -----------------------------------------------------
    {
        auto model = load("audio_clip_classifier.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::Session session(*model, backend.get());
        const auto result = loom::audio::classify(session.bridge(), *model, ten);
        LOOM_CHECK(result.granularity == loom::granularity::CLIP);
        LOOM_CHECK(result.n_rows == 1 && result.labels[1] == "de");
        LOOM_CHECK(std::fabs(result.at(0, 1) - 0.7f) < 1e-6f);
        LOOM_CHECK(result.frame_rate == 0.0 && result.row_start(0) == 0.0);
    }

    // ---- An embedding is the vector, and the door hands the driver the length beside the audio -----
    {
        auto model = load("audio_embedder.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const loom::ModelContract contract = loom::ModelContract::read(*model);
        LOOM_CHECK(contract.interface_name() == "speech2embeddings");
        // Exported before `loom.output.embedding_dim`, like TitaNet's and ECAPA's published files: a
        // clip answer is one row of whatever came back.
        LOOM_CHECK(contract.embedding_dim == 0);
        loom::Session session(*model, backend.get());
        const loom::audio::Embeddings embedding = loom::audio::embed(session.bridge(), *model, ten);
        LOOM_CHECK(embedding.granularity == loom::granularity::CLIP);
        LOOM_CHECK(embedding.n_rows == 1 && embedding.dim == 3);
        LOOM_CHECK(embedding.row(0)[0] == 10.0f && embedding.row(0)[1] == 10.0f && embedding.row(0)[2] == 0.5f);
        LOOM_CHECK(embedding.frame_rate == 0.0 && embedding.row_start(0) == 0.0);
        LOOM_CHECK(throws([&] { loom::audio::classify(session.bridge(), *model, ten); }));
    }

    // ---- A clip embedder that declares its width: the same one row, and a wrong width is refused ---
    {
        auto model = load("audio_embedder_declared.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        LOOM_CHECK(loom::ModelContract::read(*model).embedding_dim == 3);  // written as i32
        loom::Session session(*model, backend.get());
        const loom::audio::Embeddings embedding = loom::audio::embed(session.bridge(), *model, ten);
        LOOM_CHECK(embedding.n_rows == 1 && embedding.dim == 3 && embedding.values.size() == 3);

        // Three numbers at a declared width of one would be three rows -- which a clip answer is not.
        auto wrong = load("audio_embedder_wrong_width.gguf", backend.get());
        LOOM_CHECK(wrong != nullptr);
        loom::Session wrong_session(*wrong, backend.get());
        LOOM_CHECK(throws([&] { loom::audio::embed(wrong_session.bridge(), *wrong, ten); }));
    }

    // ---- Per-frame embeddings: rows of the declared width, with the time each one covers -----------
    {
        auto model = load("audio_frame_embedder.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        const loom::ModelContract contract = loom::ModelContract::read(*model);
        // The door answers both granularities, so the interface a host lists is the one that works.
        LOOM_CHECK(contract.interface_name() == "speech2embeddings");
        LOOM_CHECK(contract.output_granularity == loom::granularity::FRAME);
        LOOM_CHECK(contract.embedding_dim == 3);
        loom::Session session(*model, backend.get());

        // Ten samples, one row per four: two rows of three, and the remainder is not a row.
        const loom::audio::Embeddings result = loom::audio::embed(session.bridge(), *model, ten);
        LOOM_CHECK(result.granularity == loom::granularity::FRAME);
        LOOM_CHECK(result.n_rows == 2 && result.dim == 3 && result.values.size() == 6);
        // Row r is [r, r + 0.25, r + 0.5]: a cut at the wrong width or out of order reads otherwise.
        for (uint32_t r = 0; r < result.n_rows; ++r) {
            LOOM_CHECK(result.row(r)[0] == static_cast<float>(r));
            LOOM_CHECK(result.row(r)[1] == static_cast<float>(r) + 0.25f);
            LOOM_CHECK(result.row(r)[2] == static_cast<float>(r) + 0.5f);
        }
        // 4 frames/s from 0.5 s: frame i starts at 0.5 + i / 4.
        LOOM_CHECK(std::fabs(result.frame_rate - 4.0) < 1e-9);
        LOOM_CHECK(std::fabs(result.row_start(0) - 0.5) < 1e-9);
        LOOM_CHECK(std::fabs(result.row_start(1) - 0.75) < 1e-9);

        LOOM_CHECK(throws([&] { loom::audio::classify(session.bridge(), *model, ten); }));
        LOOM_CHECK(throws([&] { loom::audio::embed(session.bridge(), *model, {}); }));
    }

    // ---- A frame embedder with no declared width is refused, not cut by a guess --------------------
    {
        auto model = load("audio_frame_embedder_undeclared.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        LOOM_CHECK(loom::ModelContract::read(*model).embedding_dim == 0);
        loom::Session session(*model, backend.get());
        LOOM_CHECK(throws([&] { loom::audio::embed(session.bridge(), *model, ten); }));
    }

    // ---- A frame answer that is not whole rows of the declared width is refused ---------------------
    {
        auto model = load("audio_frame_embedder_skewed.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::Session session(*model, backend.get());
        // Twelve samples: three rows of three is nine numbers, not a whole number of rows of two.
        const std::vector<float> twelve(12, 0.5f);
        LOOM_CHECK(throws([&] { loom::audio::embed(session.bridge(), *model, twelve); }));
    }

    // ---- An answer that is not whole rows is refused, not skewed -----------------------------------
    {
        auto model = load("audio_skewed_classifier.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::Session session(*model, backend.get());
        LOOM_CHECK(throws([&] { loom::audio::classify(session.bridge(), *model, ten); }));
    }

    // ---- A token classifier declares no granularity, and is per token ------------------------------
    {
        // Family 12's files predate `loom.output.granularity`; an absent key on a `class` output means
        // what it always meant, so no published file moves -- and the audio door refuses it.
        auto model = load("classify_driver.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        LOOM_CHECK(loom::ModelContract::read(*model).output_granularity == loom::granularity::TOKEN);
        loom::Session session(*model, backend.get());
        LOOM_CHECK(throws([&] { loom::audio::classify(session.bridge(), *model, ten); }));
    }

    std::printf("test_audio_classify: OK\n");
    LOOM_TEST_REPORT_AND_RETURN();
}
