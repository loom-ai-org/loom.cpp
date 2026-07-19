-- Lua port of loom::MatchaDriver::synthesize (src/core/matcha_driver.cpp), validating the
-- procedural-generalization architecture (LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md)
-- against a second deterministic Euler-ODE (CFM) sampling loop -- same control-flow shape already
-- proven for SupertonicTTS, this time combined with loom.expand_by_duration for real per-token
-- duration expansion (SupertonicTTS's own duration is a single scalar, not per-token).
--
-- Expects four modules pre-registered by the host: "encoder_mu", "encoder_logw", "decoder", "vocoder"
-- -- none use a KvCache.
--
-- inputs: tokens (int array, real n_vocab=178 Matcha-TTS vocabulary), n_steps (int, Euler sampler step
-- count), seed (int, seeds loom.gaussian_array -- the ONLY stochastic point in this pipeline), plus the
-- real model constants n_feats, mel_mean, mel_std (MatchaConfig's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as MatchaDriver::synthesize's own return.
function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local n_feats = inputs.n_feats
    local t_text = #inputs.tokens
    local positions = loom.range(0, t_text)
    local attn_mask_text = loom.zero_mask(t_text, t_text)

    -- --- TextEncoder: mu_x (channel-first [n_feats,T_text] -- flat layout is ALREADY "T_text rows of
    --     n_feats contiguous floats each", exactly loom.expand_by_duration's own rows_flat convention,
    --     no transpose needed here) ---
    local mu_x_ct = loom.run_subgraph("encoder_mu", t_text, 0, { tokens = inputs.tokens, positions = positions, attn_mask = attn_mask_text })

    -- --- TextEncoder: logw (per-token log duration, ne=[1,T_text] -- flat index = t directly, no
    --     reindexing: t+c*T and c*T+t are the same formula when the channel count is 1). ---
    local logw = loom.run_subgraph("encoder_logw", t_text, 0, { tokens = inputs.tokens, positions = positions, attn_mask = attn_mask_text })

    -- --- Real w_ceil = ceil(exp(logw)) (length_scale=1.0) -- per-token integer durations. ---
    local durations = {}
    local t_mel = 0
    for t = 1, t_text do
        local d = math.ceil(math.exp(logw[t]))
        if d < 1 then d = 1 end
        durations[t] = d
        t_mel = t_mel + d
    end
    -- Extend the LAST token's duration so t_mel is an exact multiple of 4 -- the Decoder topology
    -- drops all padding-mask handling (see MatchaConfig's own docstring).
    local remainder = t_mel % 4
    if remainder ~= 0 then
        local extra = 4 - remainder
        durations[t_text] = durations[t_text] + extra
        t_mel = t_mel + extra
    end

    -- --- Row-repeat duration expansion (degenerate generate_path) -> mu_y, then transpose to the
    --     Decoder's own [T_mel,n_feats] (T=ne[0] fastest) convention. ---
    local mu_y_rows = loom.expand_by_duration(mu_x_ct, t_text, n_feats, durations)
    local mu_y = {}
    for t = 0, t_mel - 1 do
        for c = 0, n_feats - 1 do
            mu_y[t + c * t_mel + 1] = mu_y_rows[t * n_feats + c + 1]
        end
    end

    -- --- Deterministic Euler CFM sampling loop over the Decoder U-Net estimator. z0's own flat order
    --     (sequential loom.gaussian_array fill) already matches the Decoder's [T_mel,n_feats]
    --     convention directly -- Gaussian noise has no structure to transpose. ---
    local z = loom.gaussian_array(t_mel * n_feats)
    local attn_mask_full = loom.zero_mask(t_mel, t_mel)
    local t_half = t_mel / 2
    local attn_mask_half = loom.zero_mask(t_half, t_half)

    local dt = 1.0 / inputs.n_steps
    for step = 0, inputs.n_steps - 1 do
        local t = step / inputs.n_steps
        local v = loom.run_subgraph("decoder", t_mel, 0, {
            z = z,
            mu = mu_y,
            t = { t },
            attn_mask_full = attn_mask_full,
            attn_mask_half = attn_mask_half,
        })
        for i = 1, #z do
            z[i] = z[i] + v[i] * dt
        end
    end

    -- --- Denormalize (real denormalize(decoder_outputs, mel_mean, mel_std)) ---
    local mel = z
    for i = 1, #mel do
        mel[i] = mel[i] * inputs.mel_std + inputs.mel_mean
    end

    -- --- HiFi-GAN v1 vocoder: mel [T_mel,n_feats] -> waveform ---
    local waveform = loom.run_subgraph("vocoder", t_mel, 0, { mel = mel })
    return waveform
end
