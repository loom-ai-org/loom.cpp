-- Lua port of loom::VitsDriver::synthesize (src/core/vits_driver.cpp), validating the
-- procedural-generalization architecture (LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md)
-- against the most involved of the three CFM/flow-style drivers ported so far: dynamic-length
-- relative-position table cropping (loom.pad_crop_relative_embeddings + loom.get_weight, since the raw
-- learned tables are read directly off the GGUF weight table, not through any topology's own
-- declared inputs/outputs) combined with Gaussian-noise-interleaved duration expansion.
--
-- Expects three modules pre-registered by the host, ALL sharing the SAME underlying GgufModel (one
-- combined vits.gguf): "stats", "logw", "flow_vocoder" -- none use a KvCache.
--
-- inputs: token_ids (int array, TextEncoder's own vocabulary), seed (int, seeds loom.gaussian_array --
-- SDP's own z_noise AND z_p's own noise, the only two stochastic points in this pipeline, matching the
-- real driver's own single `rng_`), plus the real model constants n_text_layers, window_size,
-- k_channels, inter_channels, noise_scale, noise_scale_w, length_scale (VitsConfig's own real
-- defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as VitsDriver::synthesize's own return.
function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local T = #inputs.token_ids
    local attn_mask = loom.zero_mask(T, T)

    -- Shared per-call fill logic: tokens, attn_mask, and every TextEncoder layer's dynamic-T
    -- relative-position tables -- identical for both the "stats" and "logw" calls (matches
    -- VitsDriver::synthesize's own fill_text_encoder_inputs lambda).
    local function text_encoder_inputs()
        local tbl = { tokens = inputs.token_ids, attn_mask = attn_mask }
        for i = 0, inputs.n_text_layers - 1 do
            local raw_k = loom.get_weight("stats", "enc_p.encoder.attn_layers." .. i .. ".emb_rel_k_raw")
            local raw_v = loom.get_weight("stats", "enc_p.encoder.attn_layers." .. i .. ".emb_rel_v_raw")
            tbl["emb_rel_k_" .. i] = loom.pad_crop_relative_embeddings(raw_k, inputs.window_size, inputs.k_channels, T)
            tbl["emb_rel_v_" .. i] = loom.pad_crop_relative_embeddings(raw_v, inputs.window_size, inputs.k_channels, T)
        end
        return tbl
    end

    -- --- Phase 1a: stats = TextEncoder -> [m_p;logs_p] (channel-first [2*inter_channels,T] -- flat
    --     layout is ALREADY "T rows of 2*inter_channels contiguous floats", matching the C++ driver's
    --     own stats[t*2*inter_channels+c] indexing directly, no transpose needed). ---
    local stats = loom.run_subgraph("stats", T, 0, text_encoder_inputs())

    -- --- Phase 1b: logw = TextEncoder + StochasticDurationPredictor(reverse) -> [T] duration logits ---
    local z_noise = loom.gaussian_array(T * 2)
    for i = 1, #z_noise do z_noise[i] = z_noise[i] * inputs.noise_scale_w end
    local logw_inputs = text_encoder_inputs()
    logw_inputs.z_noise = z_noise
    local logw = loom.run_subgraph("logw", T, 0, logw_inputs)

    -- --- Host generate_path: w_ceil[t]=ceil(exp(logw[t])*length_scale); y_length=max(sum(w_ceil),1).
    --     Degenerates to a plain "replicate column t of m_p/logs_p for w_ceil[t] frames" expansion once
    --     x_mask/y_mask are dropped (single unpadded utterance) -- computed directly below, same as the
    --     C++ driver, no explicit attn matrix ever materialized. ---
    local w_ceil = {}
    local y_length = 0
    for t = 1, T do
        w_ceil[t] = math.ceil(math.exp(logw[t]) * inputs.length_scale)
        y_length = y_length + w_ceil[t]
    end
    if y_length < 1 then y_length = 1 end

    -- --- z_p: [y_length,inter_channels] (T-major [T,C]) -- m_p[c,t]=stats[t*2C+c],
    --     logs_p[c,t]=stats[t*2C+C+c]. ---
    local inter_channels = inputs.inter_channels
    local z_p = {}
    local out_frame = 0
    local noise_idx = 0
    local gaussian_pool = loom.gaussian_array(y_length * inter_channels)
    for t = 0, T - 1 do
        if out_frame >= y_length then break end
        local m_p_base = t * 2 * inter_channels
        local logs_p_base = m_p_base + inter_channels
        for rep = 1, w_ceil[t + 1] do
            if out_frame >= y_length then break end
            for c = 0, inter_channels - 1 do
                noise_idx = noise_idx + 1
                z_p[out_frame * inter_channels + c + 1] =
                    stats[m_p_base + c + 1] + gaussian_pool[noise_idx] * math.exp(stats[logs_p_base + c + 1]) * inputs.noise_scale
            end
            out_frame = out_frame + 1
        end
    end

    -- --- Phase 2: coupling flow (reverse) + HiFi-GAN vocoder -> waveform ---
    local waveform = loom.run_subgraph("flow_vocoder", y_length, 0, { z_p = z_p })
    return waveform
end
