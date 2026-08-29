// loom's side of the ASR comparison: wall time for one transcription of the SAME clip, with the model
// already loaded. NOT part of the build -- a standalone measurement, kept for the same reason as
// bench_vits_loom.cpp: "what did a change actually buy" is a per-machine question that has to stay
// re-runnable on the next machine.
//
//   g++ -O3 -std=c++17 -I include -I tools/loom_cli -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include \
//       scripts/bench_asr_loom.cpp tools/loom_cli/wav_file.cpp -o bench_asr_loom \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//   ./bench_asr_loom <asr.gguf> <clip.wav> [nrun] [language]
//
// MODEL LOAD IS OUTSIDE THE TIMER, which is the whole reason this exists rather than timing
// `loom_cli`. Whisper-small's load is a large fraction of a short clip's transcription, and on the Pi
// -- 3.8 GB of RAM, weights evicted from page cache between runs -- it is not even a stable fraction.
// Timing the loop instead of the process removes that term from both arms rather than estimating it.
//
// The transcript is printed for every run and they must MATCH: two engines, or two builds, that
// transcribe differently are not doing equal work, and a ratio over unequal work measures nothing
// (Retro-010, and the same rule as bench_vits_loom.cpp's sample count).
#include "loom/loom.h"
#include "wav_file.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <asr.gguf> <clip.wav> [nrun] [language]\n", argv[0]); return 2; }
    const std::string gguf = argv[1];
    const std::string wav  = argv[2];
    const int nrun = argc > 3 ? std::atoi(argv[3]) : 5;
    const std::string language = argc > 4 ? argv[4] : "en";

    // Device::open rather than a bare CPU backend, because that is what applies $LOOM_N_THREADS --
    // ggml's own default is 4 whatever the machine has, so a bare backend cannot be asked for 24.
    loom::Device device = loom::Device::open("cpu");
    loom::Backends backends = device.backends();

    auto model = loom::GgufModel::load(gguf, backends.primary);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    const std::vector<float> waveform = loom_cli::load_wav_pcm16_mono_16k(wav);

    // A Session, not a bare bridge: it allocates the caches the topology declares, and it is the same
    // object loom_cli builds -- so this times the engine's real configuration rather than a cut-down one.
    loom::Session session(*model, backends);

    loom::audio::TranscribeOptions options;
    options.language = language;
    options.task = "transcribe";

    // ONE WARM-UP, DISCARDED, because the other arm has one. bench_onnx_tasks.py's ASR task warms
    // before timing (`text = run()`), and bench_{vits,lm}_loom.cpp both do too -- this harness was the
    // only one that did not, so every ratio taken from it timed loom's COLD run against onnxruntime's
    // warm one (Retro-012's rule: equal work, equal thermals, equal ESTIMATOR).
    //
    // Measured 2026-08-28, cold/warm on whisper-small: 1.25-1.7x at 24 threads on the 285K, 1.02x at
    // four on the same box, and BELOW 1.0 on the 2-core Ryzen, where the box heats up faster than the
    // first run pays for itself. The penalty is a thread-count effect, so it hit exactly the cell the
    // README called loom's ASR win.
    //
    // AND USE nrun >= 3. `times[times.size() / 2]` on two samples is the LARGER of the two, not a
    // median, so nrun=2 reported the max of a cold run and a warm one. The two faults together are
    // 1.43x at 24 threads on the 285K: nine launches at nrun=2 with no warm-up median 1.650 s, nine at
    // nrun=5 with one median 1.157 s, same binary, same clip, same box.
    const auto w0 = std::chrono::steady_clock::now();
    const loom::audio::Transcription warm_r =
        loom::audio::transcribe(session.bridge(), *model, waveform, options);
    const double warm = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();

    std::vector<double> times;
    std::string first_text;
    for (int i = 0; i < nrun; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const loom::audio::Transcription r =
            loom::audio::transcribe(session.bridge(), *model, waveform, options);
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (i == 0) first_text = r.text;
        if (r.text != warm_r.text) {
            std::fprintf(stderr, "TRANSCRIPT CHANGED between runs -- not equal work, ratio meaningless\n");
            return 1;
        }
    }
    std::sort(times.begin(), times.end());
    std::printf("loom   asr   audio=%.2fs  median %.4f s  min %.4f s  (n=%d, warm-up %.4f s "
                "discarded)\n",
                waveform.size() / 16000.0, times[times.size() / 2], times.front(), nrun, warm);
    std::printf("  text: %s\n", first_text.c_str());
    return 0;
}
