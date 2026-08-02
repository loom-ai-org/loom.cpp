-- Lua port of loom::StyleTTS2Driver::synthesize (src/core/styletts2_driver.cpp). Every piece EXCEPT the
-- style-diffusion sampler is architecturally identical to (and reuses the SAME per-topology conventions
-- as) tools/convert_kokoro/kokoro_driver.lua -- see that file's own comments for the shared BiLSTM/
-- AdainResBlk1d/SineGen/STFT/Generator mechanics, duplicated here per this project's usual per-driver
-- script convention rather than factored into a shared library (matching styletts2_driver.cpp's own
-- "duplicated, not shared" precedent for run_block_layout_a/layout_a_to_tc).
--
-- The genuinely NEW piece is the style-diffusion sampler: ADPM2 (Karras et al. 2022 "ancestral" DPM-2)
-- over the real Transformer1d denoiser, KDiffusion preconditioning done in plain Lua host math (mirrors
-- style_diffusion_sampler.cpp exactly), noise drawn from loom.gaussian_array (the SAME shared rng_
-- stream loom.seed_rng resets, matching StyleTTS2Driver::synthesize's own single rng_ feeding both the
-- diffusion sampler's noise AND SineGen's rand_ini/noise draws, in that draw order).
--
-- Expects every topology pre-registered by the host, ALL sharing the SAME underlying GgufModel (one
-- combined styletts2.gguf): "albert", "diffusion", "bert_encoder", "text_encoder_cnn",
-- "text_encoder_lstm_{h,c}_{fwd,bwd}", "duration_lstm_{0,1,2}_{h,c}_{fwd,bwd}", "duration_adaln_{0,1,2}",
-- "top_lstm_{h,c}_{fwd,bwd}", "duration_proj", "f0n_shared_lstm_{h,c}_{fwd,bwd}",
-- "f0n_f0_block{0,1,2}", "f0n_n_block{0,1,2}", "f0n_f0_proj", "f0n_n_proj", "decoder_core", "sinegen",
-- "stft_forward", "generator" -- none use a KvCache. The Kokoro-equivalent "stft_inverse" topology is
-- intentionally NOT registered: confirmed dead weight in the real C++ driver too.
--
-- inputs: input_ids (int array, real StyleTTS2 wraps with a SINGLE LEADING 0 token only -- caller's
-- responsibility), diffusion_steps (int, ADPM2Sampler's own num_steps), seed (int, seeds
-- loom.seed_rng -- BOTH the diffusion sampler's own noise draws AND SineGen's rand_ini/noise draws are
-- drawn from this one shared stream, in that order), plus the real model constants style_dim, d_model,
-- hidden_per_dir, harmonic_num, upsample_scale, gen_istft_n_fft, gen_istft_hop, sigma_min, sigma_max,
-- rho, sigma_data (StyleTTS2Config's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as StyleTTS2Driver::synthesize's own
-- return.
--
-- PRECISION NOTE (unique to this driver among the ported models): the final waveform matches the C++
-- oracle to only ~1e-3 to ~1e-2, not the ~1e-6 achieved by every other ported model. Diagnosed in depth
-- (see BACKLOG.md's dated entry) before accepting this: every individual piece checks out --
-- albert/text_encoder_cnn/decoder_core/sinegen/generator all match EXACTLY given identical inputs, every
-- weight tensor is byte-identical between the old and new conversion paths, and the diffusion sampler's
-- OWN 256-float sample matches the C++ oracle to ~1e-6/1e-7 in isolation (Lua host math run in double
-- precision vs C++'s float32 -- an unavoidable, tiny difference; explicitly truncating every
-- intermediate to float32 via LuaJIT's FFI was tried and did NOT close the gap, ruling out "fixable
-- host-math imprecision" as the cause). What's actually happening: this is the ONLY ported driver whose
-- style vector comes out of an ITERATIVE numerical process (5 ADPM2 steps, each feeding its own output
-- back into the next network call) rather than a plain passthrough (Kokoro's ref_s) or a single affine
-- combination (VITS's z_p) -- and that vector then conditions ~50+ sequential AdaIN/conv layers across
-- DurationEncoder/F0Ntrain/Decoder/Generator, the last of which is an adversarially-trained (GAN-style)
-- istftnet vocoder, a network class documented to have high sensitivity to small input perturbations.
-- A ~1e-6 relative difference in the sampler's own output, independently reproduced in a different
-- language/precision, compounding through that whole cascade, is a completely ordinary amount of
-- amplification for this kind of network -- not a logic bug.

-- --- Layout helpers (identical to kokoro_driver.lua's own -- see that file's header comment for the
--     three distinct flat conventions this pipeline uses). ---
local function to_row_major(rows, C)
    local T = #rows
    local flat = {}
    for t = 0, T - 1 do
        for c = 0, C - 1 do flat[t * C + c + 1] = rows[t + 1][c + 1] end
    end
    return flat
end

local function from_row_major(flat, T, C)
    local rows = {}
    for t = 0, T - 1 do
        local row = {}
        for c = 0, C - 1 do row[c + 1] = flat[t * C + c + 1] end
        rows[t + 1] = row
    end
    return rows
end

local function to_layout_a(rows, T, C)
    local flat = {}
    for t = 0, T - 1 do
        for c = 0, C - 1 do flat[c * T + t + 1] = rows[t + 1][c + 1] end
    end
    return flat
end

local function from_layout_a(flat, T, C)
    local rows = {}
    for t = 0, T - 1 do
        local row = {}
        for c = 0, C - 1 do row[c + 1] = flat[c * T + t + 1] end
        rows[t + 1] = row
    end
    return rows
end

local function bilstm_run(namespace_, seq, hidden_dim)
    local T = #seq
    local out = {}
    for t = 1, T do out[t] = {} end

    local h_fwd, c_fwd = {}, {}
    for i = 1, hidden_dim do h_fwd[i], c_fwd[i] = 0.0, 0.0 end
    for t = 1, T do
        local h_new = loom.run_subgraph(namespace_ .. "_h_fwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_fwd, c_prev = c_fwd})
        local c_new = loom.run_subgraph(namespace_ .. "_c_fwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_fwd, c_prev = c_fwd})
        h_fwd, c_fwd = h_new, c_new
        for i = 1, hidden_dim do out[t][i] = h_new[i] end
    end

    local h_bwd, c_bwd = {}, {}
    for i = 1, hidden_dim do h_bwd[i], c_bwd[i] = 0.0, 0.0 end
    for i = 0, T - 1 do
        local t = T - i
        local h_new = loom.run_subgraph(namespace_ .. "_h_bwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_bwd, c_prev = c_bwd})
        local c_new = loom.run_subgraph(namespace_ .. "_c_bwd", {n_tokens = 0, n_past = 0}, {layer_input = seq[t], h_prev = h_bwd, c_prev = c_bwd})
        h_bwd, c_bwd = h_new, c_new
        for j = 1, hidden_dim do out[t][hidden_dim + j] = h_new[j] end
    end
    return out
end

local function run_resblk_stack(name_prefix, x_rows, style)
    local cur = x_rows
    for i = 0, 2 do
        local T_in = #cur
        local dim_in = #cur[1]
        local flat, shape = loom.run_subgraph(name_prefix .. "_block" .. i, {n_tokens = T_in, n_past = 0},
                                               {x = to_layout_a(cur, T_in, dim_in), style = style})
        cur = from_layout_a(flat, shape[1], shape[2])
    end
    return cur
end

local function run_proj1x1(name, feat)
    local T = #feat
    local C = #feat[1]
    return loom.run_subgraph(name, {n_tokens = T, n_past = 0}, {x = to_layout_a(feat, T, C)})
end

local function sigmoid(v) return 1.0 / (1.0 + math.exp(-v)) end

local function round_half_to_even(x)
    local f = math.floor(x)
    local diff = x - f
    if diff < 0.5 then return f
    elseif diff > 0.5 then return f + 1
    elseif f % 2 == 0 then return f
    else return f + 1 end
end

-- Matches loom::predict_durations exactly (see kokoro_driver.lua's own copy for the full rationale).
local function predict_durations(duration_logits, speed)
    local pred_dur = {}
    for t = 1, #duration_logits do
        local sum = 0.0
        for _, v in ipairs(duration_logits[t]) do sum = sum + sigmoid(v) end
        local rounded = round_half_to_even(sum / speed)
        pred_dur[t] = math.max(rounded, 1)
    end
    return pred_dur
end

-- --- Style-diffusion sampler: matches style_diffusion_sampler.cpp exactly. ---
local function karras_schedule(num_steps, sigma_min, sigma_max, rho)
    local rho_inv = 1.0 / rho
    local smin_r = sigma_min ^ rho_inv
    local smax_r = sigma_max ^ rho_inv
    local denom = (num_steps > 1) and (num_steps - 1) or 1
    local sigmas = {}
    for i = 0, num_steps - 1 do
        local t = i / denom
        local v = smax_r + t * (smin_r - smax_r)
        sigmas[i + 1] = v ^ rho
    end
    sigmas[num_steps + 1] = 0.0
    return sigmas
end

local function adpm2_step(x, sigma, sigma_next, denoise_fn)
    local sigma_up = math.sqrt(sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next) / (sigma * sigma))
    local sigma_down = math.sqrt(sigma_next * sigma_next - sigma_up * sigma_up)
    local sigma_mid = (sigma + sigma_down) / 2.0

    local denoised = denoise_fn(x, sigma)
    local x_mid = {}
    for i = 1, #x do
        local d = (x[i] - denoised[i]) / sigma
        x_mid[i] = x[i] + d * (sigma_mid - sigma)
    end

    local denoised_mid = denoise_fn(x_mid, sigma_mid)
    local x_next = {}
    for i = 1, #x do
        local d_mid = (x_mid[i] - denoised_mid[i]) / sigma_mid
        x_next[i] = x[i] + d_mid * (sigma_down - sigma)
    end

    local noise = loom.gaussian_array(#x)
    for i = 1, #x do x_next[i] = x_next[i] + noise[i] * sigma_up end
    return x_next
end

local function adpm2_sample(noise, denoise_fn, sigmas, num_steps)
    local x = {}
    for i = 1, #noise do x[i] = sigmas[1] * noise[i] end
    for i = 0, num_steps - 2 do
        x = adpm2_step(x, sigmas[i + 1], sigmas[i + 2], denoise_fn)
    end
    return x
end

function infer(inputs)
    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir

    -- --- CustomAlbert: raw bert_dur (last_hidden_state), Layout B [768,T] ---
    local positions = loom.range(0, T_text)
    local attn_mask = loom.zero_mask(T_text, T_text)
    local bert_out = loom.run_subgraph("albert", {n_tokens = T_text, n_past = 0},
                                        {tokens = inputs.input_ids, positions = positions, attn_mask = attn_mask})

    -- --- Style-diffusion sampler: ADPM2 over the real Transformer1d, conditioned on RAW bert_dur (not
    --     bert_encoder's projection). ---
    local function denoise_fn(x, sigma)
        local sigma_data = inputs.sigma_data
        local c_skip = (sigma_data * sigma_data) / (sigma * sigma + sigma_data * sigma_data)
        local c_out = sigma * sigma_data / math.sqrt(sigma_data * sigma_data + sigma * sigma)
        local c_in = 1.0 / math.sqrt(sigma * sigma + sigma_data * sigma_data)
        local c_noise = math.log(sigma) * 0.25

        local x_scaled = {}
        for i = 1, #x do x_scaled[i] = x[i] * c_in end

        local model_out = loom.run_subgraph("diffusion", {n_tokens = T_text, n_past = 0},
                                             {x_in = x_scaled, time = {c_noise}, embedding = bert_out, attn_mask = attn_mask})

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

    -- --- bert_encoder ---
    local d_en_flat = loom.run_subgraph("bert_encoder", {n_tokens = T_text, n_past = 0}, {x = bert_out})  -- Layout A [T,512]

    -- --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating style ---
    local x = {}
    for t = 0, T_text - 1 do
        local row = {}
        for c = 0, d_model - 1 do row[c + 1] = d_en_flat[c * T_text + t + 1] end
        for s = 1, style_dim do row[d_model + s] = s_predictor[s] end
        x[t + 1] = row
    end
    for i = 0, 2 do
        local lstm_out = bilstm_run("duration_lstm_" .. i, x, hidden_per_dir)  -- T_text x 512
        -- adaln's own "x" input is declared [channels,T] (flat[t*channels+c]).
        local seq_ct = {}
        for t = 0, T_text - 1 do
            for c = 0, d_model - 1 do seq_ct[t * d_model + c + 1] = lstm_out[t + 1][c + 1] end
        end
        local ada_out = loom.run_subgraph("duration_adaln_" .. i, {n_tokens = T_text, n_past = 0}, {x = seq_ct, style = s_predictor})
        local new_x = {}
        for t = 0, T_text - 1 do
            local row = {}
            for c = 0, d_model - 1 do row[c + 1] = ada_out[t * d_model + c + 1] end
            for s = 1, style_dim do row[d_model + s] = s_predictor[s] end
            new_x[t + 1] = row
        end
        x = new_x
    end
    local d = x  -- (T_text, 640)

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations ---
    local top_out = bilstm_run("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", {n_tokens = 0, n_past = 0}, {x = top_out[t]})
    end
    -- Real quirk (no /speed at all -- the real demo's own inference() has no such parameter):
    -- pred_dur[-1] += 5, padding the last token's duration.
    local pred_dur = predict_durations(duration_logits, 1.0)
    pred_dur[#pred_dur] = pred_dur[#pred_dur] + 5

    -- --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder) ---
    local T_frames = 0
    for t = 1, T_text do T_frames = T_frames + pred_dur[t] end
    local d_channels = d_model + style_dim
    local en = from_row_major(loom.expand_by_duration(to_row_major(d, d_channels), T_text, d_channels, pred_dur),
                               T_frames, d_channels)

    local cnn_flat, cnn_shape = loom.run_subgraph("text_encoder_cnn", {n_tokens = T_text, n_past = 0}, {tokens = inputs.input_ids})
    local te_channels = cnn_shape[2]
    local cnn_rows = from_layout_a(cnn_flat, T_text, te_channels)
    local t_en = bilstm_run("text_encoder_lstm", cnn_rows, hidden_per_dir)  -- T_text x 512
    local asr = from_row_major(loom.expand_by_duration(to_row_major(t_en, 512), T_text, 512, pred_dur),
                                T_frames, 512)

    -- --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections ---
    local shared_out = bilstm_run("f0n_shared_lstm", en, hidden_per_dir)  -- T_frames x 512
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)
    local T_f0 = #f0_feat  -- 2*T_frames

    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

    -- --- Decoder core: F0_conv/N_conv + encode/decode AdainResBlk1d stack -> x (512ch, T_f0 long) ---
    local decoder_x_flat = loom.run_subgraph("decoder_core", {n_tokens = T_frames, n_past = 0}, {
        asr = to_layout_a(asr, T_frames, 512),
        f0_curve = F0_curve,
        n_curve = N_curve,
        style = s_decoder,
    })

    -- --- SineGen: harmonic source from F0_curve (uniform THEN gaussian draws, matching the C++'s own
    --     uniform01(rng_)-then-normal(rng_) order against the ONE shared rng_ stream) ---
    local dim = inputs.harmonic_num + 1
    local L = T_f0 * inputs.upsample_scale
    local rand_ini = {0.0}
    local u = loom.uniform_array(dim - 1)
    for i = 1, dim - 1 do rand_ini[i + 1] = u[i] end
    local noise = loom.gaussian_array(dim * L)
    local har_source = loom.run_subgraph("sinegen", {n_tokens = T_f0, n_past = 0}, {f0_curve = F0_curve, rand_ini = rand_ini, noise = noise})

    -- --- forward STFT: har_source (host reflect-padded) -> har (mag+phase concat) ---
    local n_fft = inputs.gen_istft_n_fft
    local hop = inputs.gen_istft_hop
    local pad = n_fft / 2
    local waveform_padded = {}
    for i = 0, pad - 1 do waveform_padded[i + 1] = har_source[pad - i + 1] end
    for i = 0, L - 1 do waveform_padded[pad + i + 1] = har_source[i + 1] end
    for i = 0, pad - 1 do waveform_padded[pad + L + i + 1] = har_source[L - 2 - i + 1] end
    local har_flat = loom.run_subgraph("stft_forward", {n_tokens = #waveform_padded, n_past = 0}, {waveform_padded = waveform_padded})
    local T_har = (#waveform_padded - n_fft) / hop + 1

    -- --- Generator core: x + har + host-precomputed wsum -> waveform ---
    local out_len_full = (T_har - 1) * hop + n_fft
    local window = {}
    for i = 0, n_fft - 1 do window[i + 1] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / n_fft) end
    local wsum = {}
    for i = 1, out_len_full do wsum[i] = 0.0 end
    for t = 0, T_har - 1 do
        for i = 0, n_fft - 1 do
            local idx = t * hop + i + 1
            wsum[idx] = wsum[idx] + window[i + 1] * window[i + 1]
        end
    end
    local waveform = loom.run_subgraph("generator", {n_tokens = T_f0, n_past = 0},
                                        {x = decoder_x_flat, style = s_decoder, har = har_flat, wsum = wsum})
    return waveform
end
