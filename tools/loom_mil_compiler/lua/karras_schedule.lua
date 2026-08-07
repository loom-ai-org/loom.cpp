-- --- Style-diffusion sampler: the ADPM2/Karras host-math orchestration, which is NOT part of what is
--     MIL-traced -- only the denoiser NETWORK that "diffusion" calls into is. It began as a port of
--     the C++ `style_diffusion_sampler.cpp`, verified against it; that file is retired (P4.0.8's
--     follow-up) and this is the only implementation now. See BACKLOG.md for why this loop is
--     expected to stay hand-written rather than generated. ---
local function karras_schedule(num_steps, sigma_min, sigma_max, rho)
    local rho_inv = 1.0 / rho
    local smin_r = sigma_min ^ rho_inv
    local smax_r = sigma_max ^ rho_inv
    local denom = (num_steps > 1) and (num_steps - 1) or 1
    local sigmas = {}
    for i = 0, num_steps - 1 do
        local t = i / denom
        local v = smax_r + t * (smin_r - smax_r)
        sigmas[i + 1] = v ^ rho
    end
    sigmas[num_steps + 1] = 0.0
    return sigmas
end
