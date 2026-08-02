
    -- --- Style-diffusion sampler: ADPM2 over the MIL-traced Transformer1d, conditioned on RAW bert_dur
    --     (not bert_encoder's projection). No attn_mask input anymore -- see module docstring. ---
    local function denoise_fn(x, sigma)
        local sigma_data = inputs.sigma_data
        local c_skip = (sigma_data * sigma_data) / (sigma * sigma + sigma_data * sigma_data)
        local c_out = sigma * sigma_data / math.sqrt(sigma_data * sigma_data + sigma * sigma)
        local c_in = 1.0 / math.sqrt(sigma * sigma + sigma_data * sigma_data)
        local c_noise = math.log(sigma) * 0.25

        local x_scaled = {}
        for i = 1, #x do x_scaled[i] = x[i] * c_in end

        local model_out = loom.run_subgraph("diffusion", {n_tokens = T_text, n_past = 0},
                                             {x_in = x_scaled, time = {c_noise}, embedding = bert_out})

        local x_denoised = {}
        for i = 1, #x do x_denoised[i] = c_skip * x[i] + c_out * model_out[i] end
        return x_denoised
    end

    local style_vec_dim = 2 * style_dim
    local noise0 = loom.gaussian_array(style_vec_dim)
    local sigmas = karras_schedule(inputs.diffusion_steps, inputs.sigma_min, inputs.sigma_max, inputs.rho)
    local s_pred = adpm2_sample(noise0, denoise_fn, sigmas, inputs.diffusion_steps)
    local s_decoder, s_predictor = {}, {}
    for i = 1, style_dim do s_decoder[i] = s_pred[i] end
    for i = 1, style_dim do s_predictor[i] = s_pred[style_dim + i] end
