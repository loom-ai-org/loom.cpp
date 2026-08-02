local function adpm2_sample(noise, denoise_fn, sigmas, num_steps)
    local x = {}
    for i = 1, #noise do x[i] = sigmas[1] * noise[i] end
    for i = 0, num_steps - 2 do
        x = adpm2_step(x, sigmas[i + 1], sigmas[i + 2], denoise_fn)
    end
    return x
end
