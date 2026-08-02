local function adpm2_step(x, sigma, sigma_next, denoise_fn)
    local sigma_up = math.sqrt(sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next) / (sigma * sigma))
    local sigma_down = math.sqrt(sigma_next * sigma_next - sigma_up * sigma_up)
    local sigma_mid = (sigma + sigma_down) / 2.0

    local denoised = denoise_fn(x, sigma)
    local x_mid = {}
    for i = 1, #x do
        local d = (x[i] - denoised[i]) / sigma
        x_mid[i] = x[i] + d * (sigma_mid - sigma)
    end

    local denoised_mid = denoise_fn(x_mid, sigma_mid)
    local x_next = {}
    for i = 1, #x do
        local d_mid = (x_mid[i] - denoised_mid[i]) / sigma_mid
        x_next[i] = x[i] + d_mid * (sigma_down - sigma)
    end

    local noise = loom.gaussian_array(#x)
    for i = 1, #x do x_next[i] = x_next[i] + noise[i] * sigma_up end
    return x_next
end
