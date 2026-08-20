// What `transcribe` does with an argument the model has nothing to select with, and what it reports as
// a segment when the model chose no boundaries. Both against a DYNAMIC-LENGTH fixture, which is the
// shape of every ASR export here except Whisper.
//
// Reported together, because they were met together: an English-only ASR model refused
// `language="en"` outright, and the transcript it produced after the argument was dropped carried one
// segment reading `0.00 -> 0.00`. The first is a correct call being called an error; the second is a
// measurement that was never taken being reported as though it had been.
//
// The rule the first half encodes is NOT "ignore what you cannot do". An argument is ignored only when
// nothing in the decode could act on it -- here the driver is handed the waveform and its length and
// no prompt at all, so no language token can reach it. A request the model cannot SERVE still throws:
// `translate` on a file with no task tokens would otherwise transcribe and return fluent output in the
// wrong language, which looks exactly like success.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/dynamic_asr.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());

    // 2 s at 16 kHz, so the expected extent is a number no default could produce by accident.
    const std::vector<float> waveform(32000, 0.25f);

    // --- the segment covers the clip, and says it was not a chosen boundary ---
    {
        loom::audio::TranscribeOptions options;
        const loom::audio::Transcription r =
            loom::audio::transcribe(session.bridge(), *model, waveform, options);
        LOOM_CHECK(r.segments.size() == 1);
        LOOM_CHECK(r.segments[0].start == 0.0);
        LOOM_CHECK(r.segments[0].end == 2.0);   // was 0.0, which read as an empty span at the start
        LOOM_CHECK(!r.segments[0].closed);      // not a boundary the model chose
        LOOM_CHECK(!r.timestamped);             // and this is where a caller reads that
        LOOM_CHECK(r.warnings.empty());         // nothing was asked for, so nothing to say
    }

    // --- a language argument is ignored, with a warning, not refused ---
    {
        loom::audio::TranscribeOptions options;
        options.language = "en";
        const loom::audio::Transcription r =
            loom::audio::transcribe(session.bridge(), *model, waveform, options);
        LOOM_CHECK(r.warnings.size() == 1);
        LOOM_CHECK(r.warnings[0].find("language") != std::string::npos);
        LOOM_CHECK(r.warnings[0].find("ignored") != std::string::npos);
        LOOM_CHECK(r.text.find("hello") != std::string::npos);   // and it still transcribed
    }

    // --- `transcribe` names the default, so it is redundant in the same way ---
    {
        loom::audio::TranscribeOptions options;
        options.task = "transcribe";
        const loom::audio::Transcription r =
            loom::audio::transcribe(session.bridge(), *model, waveform, options);
        LOOM_CHECK(r.warnings.size() == 1);
        LOOM_CHECK(r.text.find("hello") != std::string::npos);
    }

    // --- `translate` is a capability this file does not have, and still throws ---
    {
        loom::audio::TranscribeOptions options;
        options.task = "translate";
        bool threw = false;
        try {
            loom::audio::transcribe(session.bridge(), *model, waveform, options);
        } catch (const loom::LoadError&) {
            threw = true;
        }
        LOOM_CHECK(threw);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
