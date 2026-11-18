// Exercises LoomLuaBridge's host-math bindings (loom.range/causal_mask/zero_mask/argmax_row/seed_rng/
// gaussian_array/uniform_array/expand_by_duration) and the
// load_script()/call() plumbing
// in isolation, bypassing loom.run_subgraph entirely (no GGUF/GraphBuilder needed) -- same "bypass the
// full pipeline, test each piece against a hand-computed expectation" discipline as
// test_primitive_registry.cpp. loom.run_subgraph itself is exercised for real only by each model's own
// Lua-driver e2e test, which needs actual registered modules.

#include <algorithm>
#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <limits>
#include <random>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    loom::LoomLuaBridge bridge(backend.get());

    bridge.load_script(R"lua(
        function test_range(inputs)
            return loom.range(inputs.start, inputs.count)
        end

        function test_causal_mask(inputs)
            return loom.causal_mask(inputs.n_tokens, inputs.n_past)
        end

        function test_windowed_causal_mask(inputs)
            return loom.causal_mask(inputs.n_tokens, inputs.n_past, inputs.window)
        end

        function test_zero_mask(inputs)
            return loom.zero_mask(inputs.rows, inputs.cols)
        end

        function test_argmax(inputs)
            return loom.argmax_row(inputs.flat, inputs.n_vocab, inputs.row_index)
        end

        function test_scalar_roundtrip(inputs)
            return inputs.x * 2
        end

        function test_aten_randn(inputs)
            return loom.aten_randn(inputs.seed, inputs.n)
        end

        function test_gaussian(inputs)
            loom.seed_rng(inputs.seed)
            return loom.gaussian_array(inputs.n)
        end

        function test_uniform(inputs)
            loom.seed_rng(inputs.seed)
            return loom.uniform_array(inputs.n)
        end

        function test_uniform_then_gaussian(inputs)
            loom.seed_rng(inputs.seed)
            local u = loom.uniform_array(inputs.n_u)
            local g = loom.gaussian_array(inputs.n_g)
            local out = {}
            for i = 1, #u do out[#out + 1] = u[i] end
            for i = 1, #g do out[#out + 1] = g[i] end
            return out
        end

        function test_expand_by_duration(inputs)
            return loom.expand_by_duration(inputs.rows_flat, inputs.T, inputs.C, inputs.durations)
        end

    )lua");

    // --- loom.range ---
    {
        auto result = bridge.call("test_range", {{"start", 3.0}, {"count", 4.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK((arr == std::vector<double>{3, 4, 5, 6}));
    }

    // --- loom.causal_mask: n_tokens=2, n_past=1 -> n_kv=3, query positions {1,2} ---
    {
        auto result = bridge.call("test_causal_mask", {{"n_tokens", 2.0}, {"n_past", 1.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK(arr.size() == 6);
        const double inf = std::numeric_limits<double>::infinity();
        // row 0 (query_pos=1): j<=1 -> [0,0,-inf]; row 1 (query_pos=2): j<=2 -> [0,0,0]
        LOOM_CHECK(arr[0] == 0.0 && arr[1] == 0.0 && arr[2] == -inf);
        LOOM_CHECK(arr[3] == 0.0 && arr[4] == 0.0 && arr[5] == 0.0);
    }

    // --- loom.causal_mask, sliding window (BACKLOG.md P4.0.11a) ---
    // A query at absolute position p sees keys in (p - window, p] rather than [0, p]. The banding is
    // the whole of what makes a windowed model correct; the cache is untouched and still holds
    // everything, which is the half of P4.0.11 deliberately left for later.
    {
        const double inf = std::numeric_limits<double>::infinity();
        // n_tokens=2, n_past=1, window=2 -> n_kv=3.
        // row 0 (query_pos=1): keys 0 and 1 are within 2 of it -> [0, 0, -inf]
        // row 1 (query_pos=2): key 0 is 2 back, i.e. OUTSIDE (p-window, p] -> [-inf, 0, 0]
        auto result = bridge.call("test_windowed_causal_mask",
                                   {{"n_tokens", 2.0}, {"n_past", 1.0}, {"window", 2.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK(arr.size() == 6);
        LOOM_CHECK(arr[0] == 0.0 && arr[1] == 0.0 && arr[2] == -inf);
        LOOM_CHECK(arr[3] == -inf && arr[4] == 0.0 && arr[5] == 0.0);
    }

    // A window at least as wide as the whole KV extent must reproduce the full-causal mask exactly --
    // the property that makes this argument additive rather than a second mask implementation.
    {
        auto windowed = bridge.call("test_windowed_causal_mask",
                                     {{"n_tokens", 3.0}, {"n_past", 2.0}, {"window", 99.0}});
        auto plain = bridge.call("test_causal_mask", {{"n_tokens", 3.0}, {"n_past", 2.0}});
        LOOM_CHECK(std::get<std::vector<double>>(windowed) == std::get<std::vector<double>>(plain));
    }

    // window <= 0 means "no window", which is how every existing two-argument call site keeps its
    // exact previous behaviour.
    {
        auto zero_window = bridge.call("test_windowed_causal_mask",
                                        {{"n_tokens", 2.0}, {"n_past", 1.0}, {"window", 0.0}});
        auto plain = bridge.call("test_causal_mask", {{"n_tokens", 2.0}, {"n_past", 1.0}});
        LOOM_CHECK(std::get<std::vector<double>>(zero_window) == std::get<std::vector<double>>(plain));
    }

    // --- loom.zero_mask ---
    {
        auto result = bridge.call("test_zero_mask", {{"rows", 2.0}, {"cols", 3.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK((arr == std::vector<double>{0, 0, 0, 0, 0, 0}));
    }

    // --- loom.argmax_row: flat=[1,5,2, 9,0,3], n_vocab=3, row_index=1 -> argmax([9,0,3])=0 ---
    {
        auto result = bridge.call(
            "test_argmax",
            {{"flat", std::vector<double>{1, 5, 2, 9, 0, 3}}, {"n_vocab", 3.0}, {"row_index", 1.0}});
        LOOM_CHECK(std::get<double>(result) == 0.0);
    }

    // --- scalar in/out round trip (not a host binding, just plumbing) ---
    {
        auto result = bridge.call("test_scalar_roundtrip", {{"x", 21.0}});
        LOOM_CHECK(std::get<double>(result) == 42.0);
    }

    // --- loom.seed_rng/loom.gaussian_array: same std::mt19937+std::normal_distribution<float>(0,1)
    //     engine every hand-written driver's own RNG uses -- compare directly against an independent
    //     instance of that exact engine/distribution computed here, not hand-transcribed magic numbers. ---
    {
        auto result = bridge.call("test_gaussian", {{"seed", 42.0}, {"n", 5.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK(arr.size() == 5);
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        bool all_match = true;
        for (double v : arr) {
            if (std::fabs(v - static_cast<double>(normal(rng))) > 1e-9) all_match = false;
        }
        LOOM_CHECK(all_match);

        // Re-seeding with the SAME seed reproduces the SAME sequence (persistent-until-reseeded state).
        auto result2 = bridge.call("test_gaussian", {{"seed", 42.0}, {"n", 5.0}});
        LOOM_CHECK(std::get<std::vector<double>>(result2) == arr);
    }

    // --- loom.uniform_array: same std::mt19937+std::uniform_real_distribution<float>(0,1) engine as
    //     KokoroDriver/StyleTTS2Driver's own uniform01(rng_) draws -- compare against an independent
    //     instance of that exact engine/distribution. ---
    {
        auto result = bridge.call("test_uniform", {{"seed", 7.0}, {"n", 5.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK(arr.size() == 5);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        bool all_match = true;
        for (double v : arr) {
            if (std::fabs(v - static_cast<double>(uniform(rng))) > 1e-9) all_match = false;
            if (v < 0.0 || v >= 1.0) all_match = false;
        }
        LOOM_CHECK(all_match);

        // Re-seeding with the SAME seed reproduces the SAME sequence.
        auto result2 = bridge.call("test_uniform", {{"seed", 7.0}, {"n", 5.0}});
        LOOM_CHECK(std::get<std::vector<double>>(result2) == arr);
    }

    // --- loom.uniform_array THEN loom.gaussian_array against the SAME seeded rng_ stream, matching
    //     KokoroDriver/StyleTTS2Driver's own draw order (uniform01 first, then normal) -- both
    //     distributions must share the one engine so this reproduces bit-exactly against an independent
    //     pair of distribution objects drawing from one shared std::mt19937 in the same order. ---
    {
        auto result = bridge.call("test_uniform_then_gaussian", {{"seed", 99.0}, {"n_u", 3.0}, {"n_g", 3.0}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK(arr.size() == 6);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<double> expected;
        for (int i = 0; i < 3; ++i) expected.push_back(static_cast<double>(uniform(rng)));
        for (int i = 0; i < 3; ++i) expected.push_back(static_cast<double>(normal(rng)));
        bool all_match = true;
        for (size_t i = 0; i < arr.size(); ++i) {
            if (std::fabs(arr[i] - expected[i]) > 1e-9) all_match = false;
        }
        LOOM_CHECK(all_match);
    }

    // --- loom.aten_randn: torch.randn(n, generator=manual_seed(seed)), values from torch 2.8 under
    //     ATEN_CPU_CAPABILITY=default (ATen's scalar path, which the builtin reproduces). Seed 0 at 17
    //     exercises the redrawn tail block; 2^32-1 the seed's full 32 bits. The tolerance is for libm:
    //     on glibc every value is identical, another platform's logf/sinf/cosf may differ by an ulp.
    {
        const std::vector<std::pair<std::pair<double, double>, std::vector<double>>> cases = {
            {{0.0, 17.0}, {-1.1258398294448853, -1.695931315422058, 0.5666505098342896, 0.7935084104537964,
                           0.5988394618034363, -1.5550950765609741, -0.34136030077934265, 1.85300612449646,
                           0.46809640526771545, -0.5854971408843994, -0.1733970195055008, 0.18347792327404022,
                           1.3893661499023438, 1.5863343477249146, 0.946298360824585, -0.8436768054962158,
                           0.931826651096344}},
            {{4294967295.0, 33.0}, {-2.664297580718994, -1.0910485982894897, 0.45144736766815186, 0.22836072742938995,
                                    1.8986179828643799, 1.5642088651657104, -1.2553997039794922, 0.7936442494392395,
                                    -1.7514946460723877, -0.7096747756004333, 1.218266248703003, 0.22224649786949158,
                                    -2.5158841609954834, 0.48375222086906433, 0.8417865037918091, 0.1409095972776413,
                                    1.7933999300003052, 1.7220548391342163, 0.7700403332710266, -1.4820665121078491,
                                    -0.863675057888031, -0.10301630944013596, -1.2180538177490234, -0.9539583325386047,
                                    0.7631782293319702, -0.9984314441680908, 0.6818176507949829, -1.1006033420562744,
                                    1.562240719795227, -1.885819673538208, -1.0488550662994385, -0.7110393047332764,
                                    0.04037103429436684}},
        };
        for (const auto& [args, want] : cases) {
            const auto result = bridge.call("test_aten_randn", {{"seed", args.first}, {"n", args.second}});
            const auto& got = std::get<std::vector<double>>(result);
            LOOM_CHECK(got.size() == want.size());
            double worst = 0.0;
            for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
            LOOM_CHECK(worst <= 1e-6);
        }
        bool refused = false;
        try {
            bridge.call("test_aten_randn", {{"seed", 0.0}, {"n", 15.0}});
        } catch (const std::exception&) {
            refused = true;
        }
        LOOM_CHECK(refused);
    }

    // --- loom.expand_by_duration: rows=[[1,2],[3,4]] (T=2,C=2), durations=[2,1] ->
    //     [[1,2],[1,2],[3,4]] ---
    {
        auto result = bridge.call("test_expand_by_duration",
                                   {{"rows_flat", std::vector<double>{1, 2, 3, 4}},
                                    {"T", 2.0},
                                    {"C", 2.0},
                                    {"durations", std::vector<double>{2, 1}}});
        const auto& arr = std::get<std::vector<double>>(result);
        LOOM_CHECK((arr == std::vector<double>{1, 2, 1, 2, 3, 4}));
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
