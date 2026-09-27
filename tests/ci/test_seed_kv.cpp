// `loom.seed_kv`: writing a saved attention state into a module's KV cache (ADR-043).
//
// **The failure this guards is silent.** A seeded cache in the wrong layout -- K and V swapped, heads
// transposed within a row, layers out of order -- still attends over something, and a TTS voice read
// that way still produces audio. So the fixture (tests/fixtures/make_seed_kv_gguf.py) makes every
// head of every layer pick exactly ONE seeded V row, each holding distinct values, and this checks the
// numbers it picked.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

using Args = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;

// Head 0 of layer 0 is hot on seeded row 1, head 1 on row 2; layer 1's on rows 0 and 1. V[l][r][h][d]
// is 100l + 10r + 2h + d + 1.
const std::vector<double> kExpected = {11, 12, 23, 24, 101, 102, 113, 114};

bool close_to(const std::vector<double>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::abs(got[i] - want[i]) > 1e-4) return false;
    }
    return true;
}

// The fixture's seed, rebuilt here as a Lua-side array: [layer][K then V][row][head * dim].
std::vector<double> seed_array(double v_offset) {
    std::vector<double> kv(2 * 2 * 3 * 4, 0.0);
    const auto at = [](int layer, int which, int row, int head, int d) {
        return ((((layer * 2 + which) * 3 + row) * 2 + head) * 2) + d;
    };
    const int hot[2][2] = {{1, 2}, {0, 1}};
    for (int layer = 0; layer < 2; ++layer) {
        for (int head = 0; head < 2; ++head) kv[static_cast<size_t>(at(layer, 0, hot[layer][head], head, head))] = 50.0;
        for (int row = 0; row < 3; ++row) {
            for (int head = 0; head < 2; ++head) {
                for (int d = 0; d < 2; ++d) {
                    kv[static_cast<size_t>(at(layer, 1, row, head, d))] = v_offset + 100 * layer + 10 * row + 2 * head + d + 1;
                }
            }
        }
    }
    return kv;
}

bool throws(loom::Session& session, const std::string& fn, const Args& args) {
    try {
        session.bridge().call(fn, args);
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/seed_kv.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    loom::Session session(*model, backend.get());

    // 1. From the weight, tensor to tensor -- the form a shipped voice takes.
    const auto from_weight = std::get<std::vector<double>>(session.bridge().call("seeded", {}));
    LOOM_CHECK(close_to(from_weight, kExpected));

    // 2. From a Lua array holding the same values -- a caller's own saved state.
    const auto from_array = std::get<std::vector<double>>(session.bridge().call("seeded", {{"kv", seed_array(0.0)}}));
    LOOM_CHECK(close_to(from_array, kExpected));

    // 3. Re-seeding replaces what the last seed wrote: a second state is read, not the first.
    const auto reseeded =
        std::get<std::vector<double>>(session.bridge().call("seeded", {{"kv", seed_array(1000.0)}}));
    std::vector<double> shifted = kExpected;
    for (double& v : shifted) v += 1000.0;
    LOOM_CHECK(close_to(reseeded, shifted));

    // 4. The row count is derived from the size, and an explicit one must agree with it.
    LOOM_CHECK(!throws(session, "seeded", {{"kv", seed_array(0.0)}, {"n_rows", 3.0}}));
    LOOM_CHECK(throws(session, "seeded", {{"kv", seed_array(0.0)}, {"n_rows", 2.0}}));

    // 5. A size that is not a whole number of positions is refused, not truncated.
    std::vector<double> ragged = seed_array(0.0);
    ragged.pop_back();
    LOOM_CHECK(throws(session, "seeded", {{"kv", ragged}}));

    // 6. More positions than the cache holds (8) is refused.
    LOOM_CHECK(throws(session, "seeded", {{"kv", std::vector<double>(2 * 2 * 9 * 4, 0.0)}}));

    // 7. A module with no ATTENTION node has no cache to seed.
    LOOM_CHECK(throws(session, "no_cache", {}));

    LOOM_TEST_REPORT_AND_RETURN();
}
