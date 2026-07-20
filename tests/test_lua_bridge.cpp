// Exercises LoomLuaBridge's host-math bindings (loom.range/causal_mask/zero_mask/argmax_row/seed_rng/
// gaussian_array/uniform_array/expand_by_duration/pad_crop_relative_embeddings) and the
// load_script()/call() plumbing
// in isolation, bypassing loom.run_subgraph entirely (no GGUF/GraphBuilder needed) -- same "bypass the
// full pipeline, test each piece against a hand-computed expectation" discipline as
// test_primitive_registry.cpp. loom.run_subgraph itself is exercised for real only by each model's own
// Lua-driver e2e test, which needs actual registered modules.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <limits>
#include <random>

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    loom::LoomLuaBridge bridge(backend.get());

    bridge.load_script(R"lua(
        function test_range(inputs)
            return loom.range(inputs.start, inputs.count)
        end

        function test_causal_mask(inputs)
            return loom.causal_mask(inputs.n_tokens, inputs.n_past)
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

        function test_pad_crop(inputs)
            return loom.pad_crop_relative_embeddings(inputs.raw, inputs.window_size, inputs.k_channels, inputs.length)
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

    // --- loom.pad_crop_relative_embeddings: window_size=1,k_channels=1,raw=[10,20,30] ---
    {
        // length=1 (< window_size+1=2): crops down to just the middle row.
        auto result_crop = bridge.call("test_pad_crop", {{"raw", std::vector<double>{10, 20, 30}},
                                                           {"window_size", 1.0},
                                                           {"k_channels", 1.0},
                                                           {"length", 1.0}});
        LOOM_CHECK((std::get<std::vector<double>>(result_crop) == std::vector<double>{20}));

        // length=3 (> window_size+1=2): zero-pads by 1 on each side.
        auto result_pad = bridge.call("test_pad_crop", {{"raw", std::vector<double>{10, 20, 30}},
                                                          {"window_size", 1.0},
                                                          {"k_channels", 1.0},
                                                          {"length", 3.0}});
        LOOM_CHECK((std::get<std::vector<double>>(result_pad) == std::vector<double>{0, 10, 20, 30, 0}));
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
