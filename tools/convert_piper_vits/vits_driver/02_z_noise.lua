
    local z_noise = loom.gaussian_array(T * 2)
    for i = 1, #z_noise do z_noise[i] = z_noise[i] * inputs.noise_scale_w end
