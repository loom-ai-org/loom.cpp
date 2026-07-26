-- Lua orchestration for the MIL-traced StyleTTS2 export (export_styletts2_mil.py), analogous to
-- styletts2_driver.lua (the hand-built-topology driver) but wiring in the THREE machine-traced
-- topologies export_styletts2_mil.py produces ("albert", "decoder_vocoder", "diffusion") in place of
-- FIVE of styletts2_driver.lua's own bespoke topology calls ("albert" alone, and
-- "decoder_core"+"sinegen"+"stft_forward"+"generator" respectively -- "diffusion" keeps its own name but
-- is now a MIL trace of the REAL Transformer1d.run() instead of a hand-derived topology). Everything else
-- (bert_encoder, DurationEncoder/predictor.lstm/duration_proj, TextEncoder's BiLSTM, F0Ntrain) stays on
-- the EXISTING bespoke, hand-built LSTM-bound topologies -- ggml has no native LSTM op, same deliberate
-- scoping exclusion Kokoro's own MIL export already established (see kokoro_driver_mil.lua's own header).
--
-- What's different from styletts2_driver.lua, and why this isn't just that file with three calls swapped:
--   - No `positions`/`attn_mask` inputs to "albert" at all: the MIL-traced CustomAlbert computes position
--     ids (from a registered buffer, sliced dynamically) and the additive attention mask (always
--     all-zeros -- real usage is always a single, unpadded utterance) IN-GRAPH now, not host-side --
--     identical reasoning to kokoro_driver_mil.lua's own "albert_bert_encoder" phase.
--   - "albert" returns TIME-MAJOR (T,768) (`flat[t*768+c]`, this file's own "row_major" convention) instead
--     of styletts2_driver.lua's own Layout-B convention -- a deliberate choice in export_styletts2_mil.py's
--     own AlbertWrapper to avoid returning a bare `.transpose()`/permute as a traced graph's own output
--     (the exact bug export_vits_mil.py's own StatsWrapper already found and worked around). Byte-identical
--     to ne=[768,T] either way (row-major IS Layout B, see convert_styletts2_diffusion.py's own axis-
--     convention comment) -- `bert_out` feeds straight into both "bert_encoder" (unchanged, Layout-A-typed
--     input `x`) and "diffusion"'s own `embedding` input with NO reordering needed.
--   - "diffusion" no longer takes an `attn_mask` input: the REAL Transformer1d has no masking at all
--     (plain, un-masked self-attention -- confirmed reading Modules/diffusion/modules.py directly, no
--     `attn_mask`/`rel_pos` argument anywhere in AttentionBase.forward). The OLD bespoke topology declared
--     one anyway only because loom's hand-authored ATTENTION op needed some mask tensor as an API
--     formality (always zero-filled) -- the traced version's manual einsum-free (see export_styletts2_mil
--     .py's own AttentionBase.forward patch) softmax attention has no such requirement.
--   - "decoder_vocoder" replaces FOUR bespoke calls (decoder_core, sinegen, stft_forward, generator) with
--     ONE: it takes asr/f0_curve/n_curve/s/rand_ini/noise_in/wsum directly and returns the finished
--     WAVEFORM -- no host-side har (STFT mag/phase) assembly or Generator-input wiring needed at all, only
--     the SAME rand_ini/noise/wsum host-precomputation styletts2_driver.lua's own generator call already
--     required (F0Ntrain and everything upstream of the decoder is IDENTICAL to styletts2_driver.lua --
--     still bespoke/LSTM-bound, unchanged). Matches kokoro_driver_mil.lua's own identical simplification,
--     input-for-input (StyleTTS2's decoder/generator/sinegen ARE Kokoro's own istftnet classes, just
--     StyleTTS2's own checkpoint weights -- see export_styletts2_mil.py's own module docstring).
--
-- Expects topologies pre-registered by the host, from TWO GgufModel instances sharing one bridge:
--   styletts2_mil.gguf: "albert", "decoder_vocoder", "diffusion"
--   the existing bespoke output of convert_styletts2_reused.py (kokoro_bert_encoder.gguf,
--     kokoro_text_encoder_cnn.gguf, kokoro_text_encoder_lstm_{h,c}_{fwd,bwd}.gguf,
--     kokoro_duration_lstm_{0,1,2}_{h,c}_{fwd,bwd}.gguf, kokoro_duration_adaln_{0,1,2}.gguf,
--     kokoro_duration_top_lstm_{h,c}_{fwd,bwd}.gguf, kokoro_duration_proj.gguf,
--     kokoro_f0n_shared_lstm_{h,c}_{fwd,bwd}.gguf, kokoro_f0n_f0_block{0,1,2}.gguf,
--     kokoro_f0n_n_block{0,1,2}.gguf, kokoro_f0n_f0_proj.gguf, kokoro_f0n_n_proj.gguf) -- none use a
--     KvCache. register_module doesn't care which GgufModel a topology's weights live in, only that all
--     of them share one LoomLuaBridge instance.
--
-- inputs: input_ids (int array, real StyleTTS2 wraps with a SINGLE LEADING 0 token only -- caller's
-- responsibility), diffusion_steps (int, ADPM2Sampler's own num_steps), seed (int, seeds loom.seed_rng --
-- BOTH the diffusion sampler's own noise draws AND SineGen's rand_ini/noise draws are drawn from this one
-- shared stream, in that order, matching styletts2_driver.lua's own draw order EXACTLY so the two drivers
-- stay bit-reproducible against each other given the same seed), plus the real model constants style_dim,
-- d_model, hidden_per_dir, harmonic_num, upsample_scale, gen_istft_n_fft, gen_istft_hop, sigma_min,
-- sigma_max, rho, sigma_data (StyleTTS2Config's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as styletts2_driver.lua's own return.

-- --- Layout helpers, same conventions styletts2_driver.lua's own header comment already documents. ---
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
        local h_new = loom.run_subgraph(namespace_ .. "_h_fwd", 0, 0, {layer_input = seq[t], h_prev = h_fwd, c_prev = c_fwd})
        local c_new = loom.run_subgraph(namespace_ .. "_c_fwd", 0, 0, {layer_input = seq[t], h_prev = h_fwd, c_prev = c_fwd})
        h_fwd, c_fwd = h_new, c_new
        for i = 1, hidden_dim do out[t][i] = h_new[i] end
    end

    local h_bwd, c_bwd = {}, {}
    for i = 1, hidden_dim do h_bwd[i], c_bwd[i] = 0.0, 0.0 end
    for i = 0, T - 1 do
        local t = T - i
        local h_new = loom.run_subgraph(namespace_ .. "_h_bwd", 0, 0, {layer_input = seq[t], h_prev = h_bwd, c_prev = c_bwd})
        local c_new = loom.run_subgraph(namespace_ .. "_c_bwd", 0, 0, {layer_input = seq[t], h_prev = h_bwd, c_prev = c_bwd})
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
        local flat, shape = loom.run_subgraph(name_prefix .. "_block" .. i, T_in, 0,
                                               {x = to_layout_a(cur, T_in, dim_in), style = style})
        cur = from_layout_a(flat, shape[1], shape[2])
    end
    return cur
end

local function run_proj1x1(name, feat)
    local T = #feat
    local C = #feat[1]
    return loom.run_subgraph(name, T, 0, {x = to_layout_a(feat, T, C)})
end

local function sigmoid(v) return 1.0 / (1.0 + math.exp(-v)) end

-- Matches loom::predict_durations exactly (see kokoro_driver.lua's own copy for the full rationale).
local function round_half_to_even(x)
    local f = math.floor(x)
    local diff = x - f
    if diff < 0.5 then return f
    elseif diff > 0.5 then return f + 1
    elseif f % 2 == 0 then return f
    else return f + 1 end
end

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

-- --- Style-diffusion sampler: matches style_diffusion_sampler.cpp exactly (unchanged from
--     styletts2_driver.lua's own copy -- the ADPM2/Karras host-math orchestration is NOT part of what's
--     MIL-traced here, only the denoiser NETWORK "diffusion" calls into is; see BACKLOG.md's own
--     reasoning for why this sampler loop is expected to stay bespoke regardless). ---
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

-- Matches export_kokoro_mil.py's own `compute_wsum_np` (== tools/convert_kokoro/kokoro_stft_common.py's
-- `compute_wsum`), driven by T_frames directly -- same "host-precomputed real-valued denominator"
-- convention as kokoro_driver_mil.lua's own copy.
local function compute_wsum(t_frames, n_fft, hop, upsample_scale)
    local t_f0 = 2 * t_frames
    local length = t_f0 * upsample_scale
    local pad = n_fft / 2
    local padded_len = length + 2 * pad
    local t_har = math.floor((padded_len - n_fft) / hop) + 1
    local out_len_full = (t_har - 1) * hop + n_fft
    local window = {}
    for i = 0, n_fft - 1 do window[i + 1] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / n_fft) end
    local wsum = {}
    for i = 1, out_len_full do wsum[i] = 0.0 end
    for t = 0, t_har - 1 do
        for i = 0, n_fft - 1 do
            local idx = t * hop + i + 1
            wsum[idx] = wsum[idx] + window[i + 1] * window[i + 1]
        end
    end
    return wsum
end

function synthesize(inputs)
    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir

    -- --- CustomAlbert, ONE MIL-traced call -> bert_out, time-major (T,768) (== ne=[768,T] Layout B, see
    --     module docstring for why this convention, not styletts2_driver.lua's own explicit-transpose one). ---
    local bert_out = loom.run_subgraph("albert", T_text, 0, {tokens = inputs.input_ids})

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

        local model_out = loom.run_subgraph("diffusion", T_text, 0,
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

    -- --- bert_encoder (existing bespoke topology, unchanged) ---
    local d_en_flat = loom.run_subgraph("bert_encoder", T_text, 0, {x = bert_out})  -- Layout A [T,512]

    -- --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating style (bespoke, unchanged) ---
    local x = {}
    for t = 0, T_text - 1 do
        local row = {}
        for c = 0, d_model - 1 do row[c + 1] = d_en_flat[c * T_text + t + 1] end
        for s = 1, style_dim do row[d_model + s] = s_predictor[s] end
        x[t + 1] = row
    end
    for i = 0, 2 do
        local lstm_out = bilstm_run("duration_lstm_" .. i, x, hidden_per_dir)  -- T_text x 512
        local seq_ct = {}
        for t = 0, T_text - 1 do
            for c = 0, d_model - 1 do seq_ct[t * d_model + c + 1] = lstm_out[t + 1][c + 1] end
        end
        local ada_out = loom.run_subgraph("duration_adaln_" .. i, T_text, 0, {x = seq_ct, style = s_predictor})
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

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations (bespoke, unchanged) ---
    local top_out = bilstm_run("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", 0, 0, {x = top_out[t]})
    end
    -- Real quirk (no /speed at all -- the real demo's own inference() has no such parameter):
    -- pred_dur[-1] += 5, padding the last token's duration.
    local pred_dur = predict_durations(duration_logits, 1.0)
    pred_dur[#pred_dur] = pred_dur[#pred_dur] + 5

    -- --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder,
    --     bespoke, unchanged) ---
    local T_frames = 0
    for t = 1, T_text do T_frames = T_frames + pred_dur[t] end
    local d_channels = d_model + style_dim
    local en = from_row_major(loom.expand_by_duration(to_row_major(d, d_channels), T_text, d_channels, pred_dur),
                               T_frames, d_channels)

    local cnn_flat, cnn_shape = loom.run_subgraph("text_encoder_cnn", T_text, 0, {tokens = inputs.input_ids})
    local te_channels = cnn_shape[2]
    local cnn_rows = from_layout_a(cnn_flat, T_text, te_channels)
    local t_en = bilstm_run("text_encoder_lstm", cnn_rows, hidden_per_dir)  -- T_text x 512
    local asr = from_row_major(loom.expand_by_duration(to_row_major(t_en, 512), T_text, 512, pred_dur),
                                T_frames, 512)

    -- --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections (bespoke, unchanged) ---
    local shared_out = bilstm_run("f0n_shared_lstm", en, hidden_per_dir)  -- T_frames x 512
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)

    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

    -- --- decoder_vocoder: ONE MIL-traced call replaces decoder_core+sinegen+stft_forward+generator,
    --     input-for-input identical to kokoro_driver_mil.lua's own (StyleTTS2's decoder/generator/sinegen
    --     ARE Kokoro's own istftnet classes, just this checkpoint's own weights). rand_ini/noise draws in
    --     the SAME order as styletts2_driver.lua (uniform then gaussian, against the ONE shared rng_
    --     stream, AFTER the diffusion sampler's own draws) -- index 0 of rand_ini is a placeholder, zeroed
    --     BY THE GRAPH ITSELF, kept here only to preserve the exact draw-count/order the C++ oracle
    --     established. ---
    local dim = inputs.harmonic_num + 1
    local T_f0 = 2 * T_frames
    local L = T_f0 * inputs.upsample_scale
    local rand_ini = {0.0}
    local u = loom.uniform_array(dim - 1)
    for i = 1, dim - 1 do rand_ini[i + 1] = u[i] end
    local noise_in = loom.gaussian_array(dim * L)
    local wsum = compute_wsum(T_frames, inputs.gen_istft_n_fft, inputs.gen_istft_hop, inputs.upsample_scale)

    local waveform = loom.run_subgraph("decoder_vocoder", T_frames, 0, {
        asr = to_layout_a(asr, T_frames, 512),
        f0_curve = F0_curve,
        n_curve = N_curve,
        s = s_decoder,
        rand_ini = rand_ini,
        noise_in = noise_in,
        wsum = wsum,
    })
    return waveform
end
