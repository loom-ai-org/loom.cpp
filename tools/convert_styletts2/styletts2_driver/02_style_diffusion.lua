
    -- --- Style-diffusion sampler: ADPM2 over the MIL-traced Transformer1d, conditioned on RAW bert_dur
    --     (not bert_encoder's projection). No attn_mask input anymore -- see module docstring. ---
    local function denoise_fn(x, sigma)
        local c_skip = (SIGMA_DATA * SIGMA_DATA) / (sigma * sigma + SIGMA_DATA * SIGMA_DATA)
        local c_out = sigma * SIGMA_DATA / math.sqrt(SIGMA_DATA * SIGMA_DATA + sigma * sigma)
        local c_in = 1.0 / math.sqrt(sigma * sigma + SIGMA_DATA * SIGMA_DATA)
        local c_noise = math.log(sigma) * 0.25

        local x_scaled = {}
        for i = 1, #x do x_scaled[i] = x[i] * c_in end

        local model_out = loom.run_subgraph("diffusion", {n_tokens = T_text, n_past = 0},
                                             {x_in = x_scaled, time = {c_noise}, embedding = bert_out})

        local x_denoised = {}
        for i = 1, #x do x_denoised[i] = c_skip * x[i] + c_out * model_out[i] end
        return x_denoised
    end

    local style_vec_dim = 2 * STYLE_DIM
    local noise0 = loom.gaussian_array(style_vec_dim)
    local sigmas = karras_schedule(inputs.diffusion_steps, SIGMA_MIN, SIGMA_MAX, RHO)
    local s_pred = adpm2_sample(noise0, denoise_fn, sigmas, inputs.diffusion_steps)
    local s_decoder = array_slice(s_pred, 1, STYLE_DIM)
    local s_predictor = array_slice(s_pred, STYLE_DIM + 1, STYLE_DIM)
