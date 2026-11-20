// Seeded draws that are the same numbers on every standard library (ADR-141).
//
// std::mt19937's output is fixed by the standard; std::normal_distribution's is not. libstdc++ and libc++
// both run Marsaglia's polar method, but libstdc++ returns the pair's `y` first and keeps `x`, and libc++
// returns `x` first: seed 0 gives 2.21220636, 1.16307867, ... on Linux and 1.16307867, 2.21220636, ... on
// macOS and under emscripten (web/). Every model that draws Gaussian noise through the bridge
// (`loom.gaussian_array`, `seed_rng`, a stream's own generator) therefore made different audio from the
// same seed on a Mac or in a browser than on Linux, and a gate could only catch it there.
//
// These classes are libstdc++'s algorithms written out -- `generate_canonical<float, 24>` over one 32-bit
// draw, then the polar method with its pair order -- so Linux's numbers do not move and every other
// platform now has them. The one thing left to the platform is `logf`/`sqrtf`, whose last bit may differ
// between libms; that is rounding, not a different sequence.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

namespace loom {

// Uniform on [0, 1), as libstdc++'s uniform_real_distribution<float>(0, 1) draws it.
class UniformFloat {
public:
    float operator()(std::mt19937& g) const {
        // generate_canonical<float, 24>: one draw (24 bits fit in 32), summed in float, divided by 2^32,
        // and a result that rounded up to 1 is pulled back below it.
        float r = static_cast<float>(g() - std::mt19937::min()) / 4294967296.0f;
        if (r >= 1.0f) r = std::nextafter(1.0f, 0.0f);
        return r;
    }
    void reset() {}
};

// Standard normal, as libstdc++'s normal_distribution<float>(0, 1) draws it.
class NormalFloat {
public:
    float operator()(std::mt19937& g) {
        if (saved_available_) {
            saved_available_ = false;
            return saved_;
        }
        const UniformFloat uniform;
        float x, y, r2;
        do {
            // As libstdc++ writes it: a float times the draw, minus the double 1.0, stored as float.
            x = static_cast<float>(2.0f * uniform(g) - 1.0);
            y = static_cast<float>(2.0f * uniform(g) - 1.0);
            r2 = x * x + y * y;
        } while (r2 > 1.0f || r2 == 0.0f);
        const float mult = std::sqrt(-2.0f * std::log(r2) / r2);
        saved_ = x * mult;
        saved_available_ = true;
        return y * mult;
    }
    void reset() { saved_available_ = false; }

private:
    bool saved_available_ = false;
    float saved_ = 0.0f;
};

} // namespace loom
