-- Lua orchestration for the MIL-traced VITS export (export_vits_mil.py), analogous to vits_driver.lua
-- (the hand-built-topology driver) but for the three MACHINE-TRACED topologies export_vits_mil.py
-- produces. The cross-phase host logic itself (generate_path frame expansion, Gaussian noise sampling)
-- is IDENTICAL math to vits_driver.lua -- that part was never something MIL tracing could produce either
-- way, it's genuine host control flow bridging three independent graphs plus a data-dependent frame
-- count. What's different, and why this isn't just vits_driver.lua verbatim:
--   - No `attn_mask`/`emb_rel_k_i`/`emb_rel_v_i` declared inputs at all: the MIL-traced TextEncoder
--     computes masking (always all-ones, single-utterance convention) and the dynamic relative-position
--     table (via a static-pad-then-dynamic-slice trick, see export_vits_mil.py's own
--     `_get_relative_embeddings_traceable`) IN-GRAPH now, not host-side -- `loom.pad_crop_relative_
--     embeddings`/`loom.get_weight` are gone entirely.
--   - `stats` is T-fast (`stats[c*T+t]`), not channel-fast (`stats[t*2C+c]`) -- a bare PERMUTE as a
--     traced graph's own declared OUTPUT is a live, non-contiguous view that a raw contiguous byte copy
--     silently reads in PRE-permute order, so export_vits_mil.py's StatsWrapper deliberately returns the
--     untransposed (T-fast) layout rather than fighting that. See its own docstring.
--   - `z_p` is likewise T-fast (`z_p[c*y_length+out_frame]`), matching the natural (untransposed)
--     torch (1,inter_channels,T) trace convention FlowVocoderWrapper's real Conv1d-based submodules use.
--
-- Expects three modules pre-registered by the host, ALL sharing the SAME underlying GgufModel (one
-- combined vits_mil.gguf): "stats", "logw", "flow_vocoder" -- none use a KvCache.
--
-- inputs: token_ids (int array, TextEncoder's own vocabulary), seed (int, seeds loom.gaussian_array --
-- SDP's own z_noise AND z_p's own noise, the only two stochastic points in this pipeline), plus the real
-- model constants inter_channels, noise_scale, noise_scale_w, length_scale (VitsConfig's own real
-- defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as vits_driver.lua's own return.
function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local T = #inputs.token_ids

    -- --- Phase 1a: stats = TextEncoder -> [m_p;logs_p], T-fast: stats[c*T+t] (0-based c/t). ---
    local stats = loom.run_subgraph("stats", {n_tokens = T, n_past = 0}, { tokens = inputs.token_ids })

    -- --- Phase 1b: logw = TextEncoder + StochasticDurationPredictor(reverse) -> [T] duration logits.
    --     z_noise is host-sampled and ALREADY scaled by noise_scale_w (matches export_vits_mil.py's own
    --     LogwWrapper convention: the graph itself applies no further noise_scale multiply). ---
    local z_noise = loom.gaussian_array(T * 2)
    for i = 1, #z_noise do z_noise[i] = z_noise[i] * inputs.noise_scale_w end
    local logw = loom.run_subgraph("logw", {n_tokens = T, n_past = 0}, { tokens = inputs.token_ids, z_noise = z_noise })

    -- --- Host generate_path: w_ceil[t]=ceil(exp(logw[t])*length_scale); y_length=max(sum(w_ceil),1).
    --     Degenerates to a plain "replicate column t of m_p/logs_p for w_ceil[t] frames" expansion once
    --     x_mask/y_mask are dropped (single unpadded utterance) -- same as vits_driver.lua, no explicit
    --     attn matrix ever materialized. ---
    local w_ceil = {}
    local y_length = 0
    for t = 1, T do
        w_ceil[t] = math.ceil(math.exp(logw[t]) * inputs.length_scale)
        y_length = y_length + w_ceil[t]
    end
    if y_length < 1 then y_length = 1 end

    -- --- z_p: [inter_channels, y_length], T-fast (z_p[c*y_length+out_frame]) -- m_p[c,t]=stats[c*T+t],
    --     logs_p[c,t]=stats[(c+inter_channels)*T+t]. ---
    local inter_channels = inputs.inter_channels
    local z_p = {}
    local out_frame = 0
    local noise_idx = 0
    local gaussian_pool = loom.gaussian_array(y_length * inter_channels)
    for t = 0, T - 1 do
        if out_frame >= y_length then break end
        for rep = 1, w_ceil[t + 1] do
            if out_frame >= y_length then break end
            for c = 0, inter_channels - 1 do
                noise_idx = noise_idx + 1
                local m = stats[c * T + t + 1]
                local logs = stats[(c + inter_channels) * T + t + 1]
                z_p[c * y_length + out_frame + 1] = m + gaussian_pool[noise_idx] * math.exp(logs) * inputs.noise_scale
            end
            out_frame = out_frame + 1
        end
    end

    -- --- Phase 2: coupling flow (reverse) + HiFi-GAN vocoder -> waveform ---
    local waveform = loom.run_subgraph("flow_vocoder", {n_tokens = y_length, n_past = 0}, { z_p = z_p })
    return waveform
end
