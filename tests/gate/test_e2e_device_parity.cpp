// Does a real exported model compute the same thing on a GPU as it does on the CPU, and did the GPU
// actually do the work? (BACKLOG.md P4.7.)
//
// Skips -- exit 77, which ctest reports as Skipped -- on two independent conditions, and the second one
// is the point: this test skips on a CPU-ONLY BUILD as readily as on a missing fixture, because there is
// no device for it to compare against. A developer on a machine with no GPU, or a build configured
// without -DGGML_VULKAN=ON, still gets a green suite.
//
// The model is the MIL-exported NeMo Conformer-CTC-small (the same GGUF
// test_e2e_conformer_ctc_mil_export.cpp uses), chosen because it is a real encoder rather than a toy:
// convolutional subsampling, relative-position attention, layer norms and a CTC head, ~1700 ggml nodes,
// which is enough surface for a backend's op coverage to be genuinely tested. It needs no reference
// directory here -- the oracle is the CPU, not PyTorch, so the input is a deterministic synthetic
// waveform and the comparison is against this same engine's own CPU answer. What the reference fixture
// pins (that the CPU answer is RIGHT) is that other test's job; what this one pins is that the device
// answer is the same.
//
// **Tensor oracle, not token oracle.** The whole output tensor is compared elementwise, not the argmax
// or the decoded transcript: a backend can get an encoder materially wrong and still hand back a
// plausible-looking CTC path (this project has been bitten by exactly that -- see BACKLOG.md).
//
// **Why a tolerance at all, and why it is this size.** A GPU reduces a dot product in a different order
// than a CPU does, so bit-identity is not available and demanding it would be a test of the hardware
// rather than of this engine. How big the resulting gap gets was BISECTED rather than guessed, by
// truncating this topology's node list and comparing each prefix (BACKLOG.md P4.7): the graph is exact
// through node 19, the STFT's `CONV_1D` pair introduces 8.3e-5, and node 33 -- the log-mel's `LOG` --
// turns that into 1e-2, because d(log x) = dx/x and a near-silent mel bin has a tiny x. Everything after
// it inherits that. So the number below is a property of this MODEL's front end, not a measure of how
// wrong a backend is allowed to be.
//
// Two things keep it from being a rubber stamp. The waveform below carries a noise floor, which keeps
// every mel bin away from the zero that does the amplifying (measured: 2.5e-2 without it, 3.4e-3 with);
// and the frame-wise argmax -- what a CTC model's output is actually FOR -- is compared exactly, with no
// tolerance at all. A backend that got an op wrong does not land at 3e-3 and keep every argmax: it lands
// at 1e-1, or NaN, or a different token.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// Must match the length reference_forward_conformer.py traces at, since the export's own declared shapes
// follow it -- see test_e2e_conformer_ctc_mil_export.cpp.
constexpr uint32_t kNSamples = 10240;

// Relative to the output tensor's own peak magnitude. Measured at 3.4e-3 on RADV/GFX9 (see the header
// for the bisect that explains it); this is that with enough headroom for a different device's
// reduction order, and still 30x below what a wrong op produces.
constexpr float kMaxRelativeError = 1e-2f;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool has_device() {
    for (const loom::DeviceInfo& d : loom::available_devices()) {
        if (!d.is_cpu) return true;
    }
    return false;
}

// A deterministic waveform, so the two runs are fed identical bytes and a rerun compares the same thing.
// Two decaying sinusoids keep the encoder's activations in the range real audio puts them in, and the
// noise floor on top is not decoration: it is what stops any mel bin from reaching the near-zero where
// the log-mel amplifies a 1e-4 backend difference into a 1e-2 one (see the header). An LCG rather than
// <random>, so the bytes are the same on every standard library.
std::vector<float> synthetic_waveform() {
    std::vector<float> w(kNSamples);
    uint32_t state = 12345;
    for (uint32_t i = 0; i < kNSamples; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        state = state * 1664525u + 1013904223u;
        const double dither = (static_cast<double>(state >> 8) / 8388608.0 - 1.0) * 1e-3;
        w[i] = static_cast<float>(0.35 * std::sin(2.0 * M_PI * 220.0 * t) * std::exp(-1.5 * t) +
                                   0.15 * std::sin(2.0 * M_PI * 933.0 * t) + dither);
    }
    return w;
}

struct Run {
    std::vector<float> output;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int splits = 0;
    size_t device_nodes = 0;
    size_t fallback_nodes = 0;
};

Run run_on(const std::string& gguf_path, loom::Backends backends, const std::vector<float>& waveform) {
    auto model = loom::GgufModel::load(gguf_path, backends);
    // By whatever the file calls its one topology, rather than by a name spelled here: this export has
    // been through more than one naming convention ("main_topology", "main_topo"), and the fixture a
    // developer points this at is whichever they last built. A single-topology file has exactly one
    // answer, and this test is only ever pointed at one.
    const std::vector<std::string> names = model->topology_names();
    LOOM_CHECK(names.size() == 1);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(names.front()));
    loom::GraphBuilder builder(topo, *model, backends, /*kv_cache=*/nullptr);

    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_samples", kNSamples}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("waveform"), waveform.data(), 0,
                             waveform.size() * sizeof(float));
    const int32_t length_val = static_cast<int32_t>(kNSamples);
    ggml_backend_tensor_set(r.input_tensors.at("length"), &length_val, 0, sizeof(int32_t));
    builder.compute();

    Run run;
    run.ne0 = r.output->ne[0];
    run.ne1 = r.output->ne[1];
    run.output.resize(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, run.output.data(), 0, run.output.size() * sizeof(float));
    run.splits = builder.splits();
    if (backends.hybrid()) {
        const std::string device_name = ggml_backend_name(backends.primary);
        for (const std::string& node_backend : builder.node_backends()) {
            (node_backend == device_name ? run.device_nodes : run.fallback_nodes)++;
        }
    }
    return run;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CONFORMER_CTC_MIL_GGUF");
    const std::string gguf_path =
        gguf_env != nullptr ? gguf_env : "conformer_ctc_small_mil_monolithic.gguf";
    if (!path_exists(gguf_path)) {
        std::fprintf(stderr, "skipping: MIL-exported Conformer-CTC GGUF ('%s') not found\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }
    if (!has_device()) {
#ifdef LOOM_TEST_EXPECTS_DEVICE
        // A device backend WAS compiled in, so an empty registry is a broken deployment rather than a
        // machine without a GPU -- fail, do not skip (BACKLOG.md P4.8f). This exact case shipped: under
        // GGML_BACKEND_DL nothing had registered the backend directory, the registry held only the CPU,
        // and this gate exited 77 into a green ctest run on a build with Vulkan compiled in.
        std::fprintf(stderr,
                      "FAIL: a device backend is compiled into this build, but the registry reports no "
                      "device other than the CPU. In a GGML_BACKEND_DL build that means nothing swept "
                      "for backend .so files -- see tests/support/cpu_backend.h.\n");
        return 1;
#else
        std::fprintf(stderr,
                      "skipping: this build reaches no GPU/accelerator device, so there is nothing to "
                      "compare the CPU against (configure with -DGGML_VULKAN=ON, -DGGML_CUDA=ON or "
                      "-DGGML_METAL=ON on a machine that has one)\n");
        return kSkipReturnCode;
#endif
    }

    loom::Device cpu = loom::Device::open("cpu");
    loom::Device gpu = loom::Device::open("gpu");
    LOOM_CHECK(cpu.is_cpu());
    LOOM_CHECK(!gpu.is_cpu());
    std::fprintf(stderr, "comparing %s (%s) against %s\n", gpu.name().c_str(),
                 gpu.description().c_str(), cpu.name().c_str());

    const std::vector<float> waveform = synthetic_waveform();
    const Run reference = run_on(gguf_path, cpu.backends(), waveform);
    const Run actual = run_on(gguf_path, gpu.backends(), waveform);

    LOOM_CHECK(actual.ne0 == reference.ne0);
    LOOM_CHECK(actual.ne1 == reference.ne1);
    LOOM_CHECK(actual.output.size() == reference.output.size());
    LOOM_CHECK(!reference.output.empty());

    // --- The device actually ran the model ------------------------------------------------------------
    // Without this the test would pass just as happily if every node had fallen back to the CPU, which is
    // precisely the failure mode "it runs on the GPU now" most easily hides.
    std::fprintf(stderr, "device ran %zu node(s), CPU fallback ran %zu, in %d split(s)\n",
                 actual.device_nodes, actual.fallback_nodes, actual.splits);
    LOOM_CHECK(actual.device_nodes > 0);
    LOOM_CHECK(actual.device_nodes > actual.fallback_nodes);
    // The CPU run has no scheduler at all, which is the arrangement it is supposed to keep.
    LOOM_CHECK(reference.splits == 0);

    // --- ...and got the same answer ---------------------------------------------------------------------
    float peak = 0.0f;
    for (float v : reference.output) peak = std::max(peak, std::fabs(v));
    LOOM_CHECK(peak > 0.0f);

    double sum_sq_error = 0.0;
    float worst = 0.0f;
    size_t worst_index = 0;
    for (size_t i = 0; i < reference.output.size(); ++i) {
        // A NaN never compares greater than anything, so it would slip past the max below -- checked
        // explicitly, because a NaN is exactly what an unsupported op silently returning garbage looks
        // like.
        LOOM_CHECK(std::isfinite(actual.output[i]));
        const float diff = std::fabs(actual.output[i] - reference.output[i]);
        sum_sq_error += static_cast<double>(diff) * diff;
        if (diff > worst) {
            worst = diff;
            worst_index = i;
        }
    }
    const double rms_error = std::sqrt(sum_sq_error / static_cast<double>(reference.output.size()));
    std::fprintf(stderr, "max |device - cpu| = %.3e at [%zu] (peak %.3e, relative %.3e); rms %.3e\n",
                 static_cast<double>(worst), worst_index, static_cast<double>(peak),
                 static_cast<double>(worst / peak), rms_error);
    LOOM_CHECK(worst / peak < kMaxRelativeError);

    // --- ...including the part of the answer the model exists to produce ----------------------------
    // The tolerance above is a statement about float arithmetic; this is a statement about the model.
    // A CTC head's output is one class per frame, and that decision is EXACT or the device is wrong --
    // there is no tolerance in an argmax. This is the assertion that survives however the numerics
    // drift, and the reason the loose bound above is not a rubber stamp.
    const int64_t n_classes = reference.ne0;
    const int64_t n_frames = reference.ne1;
    LOOM_CHECK(n_classes > 1 && n_frames > 0);
    size_t disagreements = 0;
    for (int64_t frame = 0; frame < n_frames; ++frame) {
        const size_t base = static_cast<size_t>(frame) * static_cast<size_t>(n_classes);
        int64_t best_ref = 0;
        int64_t best_dev = 0;
        for (int64_t c = 1; c < n_classes; ++c) {
            if (reference.output[base + c] > reference.output[base + best_ref]) best_ref = c;
            if (actual.output[base + c] > actual.output[base + best_dev]) best_dev = c;
        }
        if (best_ref != best_dev) ++disagreements;
    }
    std::fprintf(stderr, "frame-wise argmax: %zu disagreement(s) over %lld frame(s)\n", disagreements,
                 static_cast<long long>(n_frames));
    LOOM_CHECK(disagreements == 0);

    LOOM_TEST_REPORT_AND_RETURN();
}
