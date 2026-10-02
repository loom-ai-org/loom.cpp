// Tests the RING KV cache (ADR-066) end to end, through a driver: one ATTENTION layer with a 4-cell
// ring, every K zero so attention is uniform, and V = p + 1 written at position p. A step's output is
// then the mean of exactly the values it attended to, which names its window: the last min(p + 1, 4)
// positions. Fixture: tests/fixtures/make_ring_kv_gguf.py.

#include "test_util.h"

#include "loom/loom.h"

#include <string>
#include <variant>
#include <vector>

int main() {
    loom::Device device = loom::Device::open("cpu");
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/ring_kv_test.gguf";
    auto model = loom::GgufModel::load(path, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // Ten single-token steps through a 4-cell cache: a linear cache would refuse step 4.
    {
        loom::Session session(*model, device.backends());
        const auto out = std::get<std::vector<double>>(session.bridge().call("stepwise", {{"steps", 10.0}}));
        LOOM_CHECK(out.size() == 10);
        for (int p = 0; p < 10 && p < static_cast<int>(out.size()); ++p) {
            const int lo = p < 3 ? 0 : p - 3;
            double want = 0.0;
            for (int j = lo; j <= p; ++j) want += j + 1;
            want /= (p - lo + 1);
            LOOM_CHECK_NEAR(out[static_cast<size_t>(p)], want, 1e-5);
        }
    }
    // Several tokens in one call while the ring has room: causal over the cells they fill.
    {
        loom::Session session(*model, device.backends());
        const auto out = std::get<std::vector<double>>(session.bridge().call("prefill", {{"n", 3.0}, {"at", 0.0}}));
        LOOM_CHECK(out.size() == 1);
        LOOM_CHECK_NEAR(out[0], 2.0, 1e-5);     // the last row attends to 1, 2, 3
    }
    // ... and refused once they would wrap, where a later row would overwrite a cell an earlier one reads.
    {
        loom::Session session(*model, device.backends());
        bool refused = false;
        try {
            session.bridge().call("prefill", {{"n", 3.0}, {"at", 2.0}});
        } catch (const loom::Error&) {
            refused = true;
        }
        LOOM_CHECK(refused);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
