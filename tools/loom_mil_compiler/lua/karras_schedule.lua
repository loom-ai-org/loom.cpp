-- --- Style-diffusion sampler: matches style_diffusion_sampler.cpp exactly (unchanged from
--     styletts2_driver.lua's own copy -- the ADPM2/Karras host-math orchestration is NOT part of what's
--     MIL-traced here, only the denoiser NETWORK "diffusion" calls into is; see BACKLOG.md's own
--     reasoning for why this sampler loop is expected to stay bespoke regardless). ---
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
