-- Lua port of loom::MatchaDriver::synthesize, wired to the MIL-traced topologies of matcha_mil.gguf
-- (export_matcha_mil.py) instead of the bespoke hand-built ones (convert_matcha_*.py /
-- convert_matcha_lua_all.py's own matcha_driver.lua). Same control-flow shape (per-token duration
-- expansion + deterministic Euler CFM sampling loop) and the same host/inputs/outputs contract as
-- matcha_driver.lua -- only the internal tensor-layout bridging differs, see below.
--
-- Layout note (differs from matcha_driver.lua): the bespoke "encoder_mu" topology emits `mu` C-fast
-- (ne=[n_feats,T], "rows_flat" convention -- T rows of n_feats contiguous floats each) because it's
-- built via an explicit MUL_MAT against a [C,T]-convention `x`. Tracing the REAL TextEncoder module
-- instead preserves its own native torch (1,n_feats,T) layout untouched, which is T-FAST (ne=[T,n_feats],
-- flat[c*T+t]) -- the SAME convention the Decoder's own z/mu/dphi_dt and the vocoder's own mel already
-- use (see export_matcha_mil.py's own module docstring for why no corrective transpose was added: doing
-- so would make the topology's declared output a bare GGML PERMUTE, a live non-contiguous view that
-- silently corrupts the read-out once compiled). Net effect: unlike matcha_driver.lua, NO transpose is
-- needed bridging TextEncoder's mu into the Decoder's own mu input here -- but the duration-expansion
-- (per-token repeat) step itself must operate directly in this T-fast layout instead of reusing
-- loom.expand_by_duration (which expects the opposite, C-fast "rows_flat" convention) -- done with a
-- direct nested-loop repeat below instead.
--
-- Expects four modules pre-registered by the host: "encoder_mu", "encoder_logw", "decoder", "vocoder"
-- -- none use a KvCache. The MIL-traced "encoder_mu"/"encoder_logw" topologies take ONLY `tokens` (no
-- separate `positions`/`attn_mask` inputs -- RoPE positions and the all-ones attention mask are both
-- derived internally from the real traced module, not supplied by the host, unlike the bespoke
-- topologies' own explicit inputs).
--
-- inputs: tokens (int array, real n_vocab=178 Matcha-TTS vocabulary), n_steps (int, Euler sampler step
-- count), seed (int, seeds loom.gaussian_array -- the ONLY stochastic point in this pipeline), plus the
-- real model constants n_feats, mel_mean, mel_std (MatchaConfig's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as MatchaDriver::synthesize's own return.

--@loom:samplers

function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local n_feats = inputs.n_feats
    local t_text = #inputs.tokens

    -- --- TextEncoder: mu_x, T-fast (ne=[T_text,n_feats], flat[c*t_text+t]) ---
    local mu_x = loom.run_subgraph("encoder_mu", {n_tokens = t_text, n_past = 0}, { tokens = inputs.tokens })

    -- --- TextEncoder: logw (per-token log duration; channel count 1 makes T-fast and C-fast layouts
    --     coincide, flat index = t directly) ---
    local logw = loom.run_subgraph("encoder_logw", {n_tokens = t_text, n_past = 0}, { tokens = inputs.tokens })

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

    -- --- Row-repeat duration expansion, directly in T-fast layout (no loom.expand_by_duration reuse --
    --     see module docstring): mu_y[c*t_mel + t_out] = mu_x[c*t_text + t_src], repeated
    --     durations[t_src+1] times per source token. ---
    local mu_y = {}
    for c = 0, n_feats - 1 do
        local t_out = 0
        for t = 0, t_text - 1 do
            local v = mu_x[c * t_text + t + 1]
            for _ = 1, durations[t + 1] do
                mu_y[c * t_mel + t_out + 1] = v
                t_out = t_out + 1
            end
        end
    end

    -- --- Deterministic Euler CFM sampling loop over the Decoder U-Net estimator -- see sample_decoder
    --     above, generated from export_matcha_mil.py's own IterativeRefinementSpec. z0 is fresh Gaussian
    --     noise with no inherent structure, so loom.gaussian_array's own sequential flat fill is usable
    --     directly as the Decoder's T-fast `z` input with no reindexing needed. ---
    local z = sample_decoder(t_mel, t_mel * n_feats, inputs.n_steps, { mu = mu_y })

    -- --- Denormalize (real denormalize(decoder_outputs, mel_mean, mel_std)) ---
    local mel = z
    for i = 1, #mel do
        mel[i] = mel[i] * inputs.mel_std + inputs.mel_mean
    end

    -- --- HiFi-GAN v1 vocoder: mel (T-fast, matches its own native torch (1,n_feats,T) layout) ->
    --     waveform ---
    local waveform = loom.run_subgraph("vocoder", {n_tokens = t_mel, n_past = 0}, { mel = mel })
    return waveform
end
