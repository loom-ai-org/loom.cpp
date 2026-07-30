-- Lua orchestration for the MIL-traced SupertonicTTS export (export_supertonic_mil.py), mirroring the
-- bespoke tools/convert_supertonic/supertonic_driver.lua's own control flow (DurationPredictor ->
-- get_latent_mask (host) -> TTLTextEncoder -> Euler CFM loop over VectorFieldEstimator -> SpeechDecoder)
-- with one simplification: the traced "vfe" topology computes VFTextCrossAttention's fractional-RoPE
-- positions INTERNALLY from real (all-ones) masks (see export_supertonic_mil.py's `_ones_mask_from_*`),
-- so this driver -- unlike the bespoke one -- never needs to host-compute `lat_frac`/`txt_frac` arrays
-- itself.
--
-- Expects four modules pre-registered by the host: "dp", "ttl_text", "vfe", "decoder" (none use a
-- KvCache) -- SAME names/roles as the bespoke driver.
--
-- Text-length scope note (see export_supertonic_mil.py's own module docstring): "vfe"'s own `txt_emb`
-- input has a FIXED shape (T_TEXT_FIXED, baked in at export time) -- `inputs.txt_ids` MUST be exactly
-- that length for this driver to run correctly, same real constraint the bespoke driver already carries.
--
-- inputs: txt_ids (int array, length t_text), style_ttl (flat f32 array, n_style_ttl*style_dim_ttl),
-- style_dp (flat f32 array, n_style_dp*style_dim_dp), n_steps (int), seed (int, seeds
-- loom.gaussian_array), plus the real model constants t_text, lat_dim, sample_rate, base_chunk_size,
-- compression_factor.
--
-- Returns: the raw waveform (flat f32 array).

--@loom:samplers

function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local t_text = inputs.t_text
    local lat_dim = inputs.lat_dim

    -- --- DurationPredictor: DPTextEncoder + MLP head -> scalar duration (seconds) ---
    local dur_arr = loom.run_subgraph("dp", {n_tokens = t_text, n_past = 0}, { txt_ids = inputs.txt_ids, stl_emb = inputs.style_dp })
    local duration = dur_arr[1]

    -- --- Real get_latent_mask: wav_length=duration*sample_rate; latent_size=base_chunk_size*
    --     compression_factor; T_lat=ceil(wav_length/latent_size) -- matches
    --     src/core/supertonic_driver.cpp's own compute_t_lat exactly. ---
    local wav_length = math.floor(duration * inputs.sample_rate)
    local latent_size = inputs.base_chunk_size * inputs.compression_factor
    local t_lat = math.floor((wav_length + latent_size - 1) / latent_size)

    -- --- TTLTextEncoder -> txt_emb, ne=[t_text,txt_dim] (T-fast, the traced module's own native torch
    --     layout -- no host-side layout crossing needed, unlike the bespoke driver's own Layout A/B
    --     bridging, since "vfe" was traced expecting exactly this same layout for its own txt_emb input). ---
    local txt_emb = loom.run_subgraph("ttl_text", {n_tokens = t_text, n_past = 0}, { txt_ids = inputs.txt_ids, stl_emb = inputs.style_ttl })

    -- --- Deterministic Euler CFM sampling loop over VectorFieldEstimator -- see sample_vfe above,
    --     generated from export_supertonic_mil.py's own IterativeRefinementSpec. ---
    local z = sample_vfe(t_lat, t_lat * lat_dim, inputs.n_steps,
                          { txt_emb = txt_emb, stl_emb = inputs.style_ttl })

    -- --- SpeechDecoder: z (ne=[t_lat,lat_dim]) -> raw waveform ---
    local waveform = loom.run_subgraph("decoder", {n_tokens = t_lat, n_past = 0}, { latent = z })
    return waveform
end
