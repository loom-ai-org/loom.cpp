-- Lua port of loom::SupertonicDriver::synthesize (src/core/supertonic_driver.cpp), validating the
-- procedural-generalization architecture (LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md)
-- against a deterministic Euler-ODE (CFM) sampling loop -- the second control-flow shape this
-- architecture needs to prove out, after Whisper's autoregressive/KV-cache case.
--
-- Expects four modules pre-registered by the host: "dp" (DurationPredictor), "ttl_text"
-- (TTLTextEncoder), "vfe" (VectorFieldEstimator), "decoder" (SpeechDecoder) -- none use a KvCache.
--
-- inputs: txt_ids (int array, length T_TEXT -- the real, documented fixed-length scope limitation
-- carried forward unchanged from SupertonicConfig, see convert_supertonic_lua_all.py), style_ttl (flat
-- f32 array, n_style_ttl*style_dim_ttl), style_dp (flat f32 array, n_style_dp*style_dim_dp), n_steps
-- (int, the Euler sampler's own step count), seed (int, seeds BOTH loom.gaussian_array calls below --
-- the ONLY stochastic point in this whole pipeline, matching the real driver's own single `rng_`), plus
-- the real model constants t_text, lat_dim, txt_dim, sample_rate, base_chunk_size, compression_factor
-- (all fixed architecture hyperparameters, not per-call data -- passed in rather than hardcoded so this
-- script stays checkpoint-agnostic).
--
-- Returns: the raw waveform (flat f32 array), same convention as SupertonicDriver::synthesize's own
-- return value.
function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local t_text = inputs.t_text
    local txt_dim = inputs.txt_dim
    local lat_dim = inputs.lat_dim

    -- --- DurationPredictor: DPTextEncoder + MLP head -> scalar duration (seconds) ---
    local dur_arr = loom.run_subgraph("dp", t_text, 0, { txt_ids = inputs.txt_ids, stl_emb = inputs.style_dp })
    local duration = dur_arr[1]

    -- --- Real get_latent_mask: wav_length=duration*sample_rate; latent_size=base_chunk_size*
    --     compression_factor; T_lat=ceil(wav_length/latent_size) -- matches
    --     src/core/supertonic_driver.cpp's own compute_t_lat exactly (math.floor for the real code's
    --     truncating uint32_t cast, since duration/wav_length are always positive here). ---
    local wav_length = math.floor(duration * inputs.sample_rate)
    local latent_size = inputs.base_chunk_size * inputs.compression_factor
    local t_lat = math.floor((wav_length + latent_size - 1) / latent_size)

    -- --- TTLTextEncoder -> txt_emb (Layout A [t_text,txt_dim]), crossed to Layout B [txt_dim,t_text]
    --     for VFE -- same host transpose as SupertonicDriver's own layout_a_to_b. ---
    local txt_emb_ta = loom.run_subgraph("ttl_text", t_text, 0, { txt_ids = inputs.txt_ids, stl_emb_ttl_cb = inputs.style_ttl })
    local txt_emb_cb = {}
    for t = 0, t_text - 1 do
        for c = 0, txt_dim - 1 do
            txt_emb_cb[t * txt_dim + c + 1] = txt_emb_ta[c * t_text + t + 1]
        end
    end

    -- --- Deterministic Euler CFM sampling loop over VectorFieldEstimator (loom::cfm_euler_sample's
    --     own shape: z += v(z,t)*dt, uniform dt=1/n_steps). ---
    local z = loom.gaussian_array(t_lat * lat_dim)

    local lat_frac = {}
    for i = 0, t_lat - 1 do lat_frac[i + 1] = i / t_lat end
    local txt_frac = {}
    for i = 0, t_text - 1 do txt_frac[i + 1] = i / t_text end

    local dt = 1.0 / inputs.n_steps
    for step = 0, inputs.n_steps - 1 do
        local t = step / inputs.n_steps
        local v = loom.run_subgraph("vfe", t_lat, 0, {
            z_t = z,
            txt_emb_cb = txt_emb_cb,
            stl_emb_ttl_cb = inputs.style_ttl,
            t = { t },
            lat_frac = lat_frac,
            txt_frac = txt_frac,
        })
        for i = 1, #z do
            z[i] = z[i] + v[i] * dt
        end
    end

    -- --- SpeechDecoder: z (Layout A [t_lat,lat_dim]) -> raw waveform ---
    local waveform = loom.run_subgraph("decoder", t_lat, 0, { latent = z })
    return waveform
end
