#include "loom/core/cfm_euler_sampler.h"

namespace loom {

std::vector<float> cfm_euler_sample(const std::vector<float>& z0, const VelocityFn& velocity_fn,
                                     int n_steps) {
    std::vector<float> z = z0;
    const float dt = 1.0f / static_cast<float>(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n_steps);
        const std::vector<float> v = velocity_fn(z, t);
        for (size_t j = 0; j < z.size(); ++j) z[j] += v[j] * dt;
    }
    return z;
}

} // namespace loom
