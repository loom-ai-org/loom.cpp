// The long-form seek: that it advances to where the model closed its segment, and STOPS there.
//
// This exists because of an off-by-one that no hermetic test could catch. A segment end is a float
// number of seconds, so the sample index it maps to is almost never exact: with the timestamp step read
// from the file as f32, 550 steps of 0.02 s come to 10.99999975 rather than 11, and 11 s of audio at
// 16 kHz TRUNCATED to 175999 -- one sample short of the end. The loop ran a second window over four
// samples of real audio plus thirty seconds of zero padding, and Whisper transcribed the silence as
// "[BLANK_AUDIO]", which went into the transcript.
//
// It was invisible until the gate fixtures were re-exported with a declared decode table, because the
// old path derived the step from three hparams in double and absorbed the error by luck. Before this
// file the only ASR fixture that emits timestamps at all was a real 970 MB Whisper export, so "the seek
// stops at the end of the audio" was verified by listening rather than by anything that runs in CI.
//
// The fixture's driver returns one closed span covering the clip and ignores its input, which is what
// makes a second window unambiguous: it can only mean the seek did not reach the end.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/timestamped_asr.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());

    // The declared table is what the loop reads, and it must survive the f32 round trip as timestamped.
    const loom::ModelContract contract = loom::ModelContract::read(*model);
    LOOM_CHECK(contract.clip_samples == 480000);
    LOOM_CHECK(contract.sample_rate == 16000);

    // Exactly 11 s, which is what the driver's closing timestamp names. The audio ends where the model
    // says its segment ends, so a correct seek has nothing left to decode.
    const std::vector<float> waveform(176000, 0.25f);

    loom::audio::TranscribeOptions options;
    options.timestamps = true;
    const loom::audio::Transcription result =
        loom::audio::transcribe(session.bridge(), *model, waveform, options);

    // ONE window. Two means the seek landed short of the end -- the bug -- and the extra window is a
    // decode of zero padding whose text lands in the transcript.
    LOOM_CHECK(result.windows == 1);
    LOOM_CHECK(result.timestamped);
    LOOM_CHECK(result.segments.size() == 1);
    LOOM_CHECK(result.segments[0].closed);
    LOOM_CHECK(result.segments[0].start == 0.0);
    // Not exactly 11.0, and that is the point rather than a tolerance: 550 f32 steps come to
    // 10.99999975, the value the seek has to handle correctly rather than one it gets to round away.
    LOOM_CHECK(result.segments[0].end > 10.999 && result.segments[0].end < 11.001);
    LOOM_CHECK(result.text.find("hello") != std::string::npos);

    // Audio that genuinely runs past the closed segment still gets its second window: the fix must not
    // have bought "stops at the end" by capping the loop at one iteration.
    const std::vector<float> longer(176000 * 3, 0.25f);
    const loom::audio::Transcription multi =
        loom::audio::transcribe(session.bridge(), *model, longer, options);
    LOOM_CHECK(multi.windows > 1);

    LOOM_TEST_REPORT_AND_RETURN();
}
