-- Lua port of loom::KokoroDriver::synthesize (src/core/kokoro_driver.cpp), validating the
-- procedural-generalization architecture on the largest driver ported so far: 43 topologies (6 BiLSTM
-- instances host-stepped via loom.run_subgraph in a Lua for-loop, matching loom::BiLstmStepper's own
-- per-timestep 4-graph-calls-per-direction mechanics), plus loom.uniform_array/loom.gaussian_array for
-- SineGen's rand_ini/noise draws (drawn from the SAME shared rng_ stream, uniform first then gaussian,
-- matching KokoroDriver::synthesize's own draw order exactly).
--
-- Expects every topology pre-registered by the host, ALL sharing the SAME underlying GgufModel (one
-- combined kokoro.gguf): "albert", "bert_encoder", "text_encoder_cnn",
-- "text_encoder_lstm_{h,c}_{fwd,bwd}", "duration_lstm_{0,1,2}_{h,c}_{fwd,bwd}",
-- "duration_adaln_{0,1,2}", "top_lstm_{h,c}_{fwd,bwd}", "duration_proj",
-- "f0n_shared_lstm_{h,c}_{fwd,bwd}", "f0n_f0_block{0,1,2}", "f0n_n_block{0,1,2}", "f0n_f0_proj",
-- "f0n_n_proj", "decoder_core", "sinegen", "stft_forward", "generator" -- none use a KvCache.
-- "kokoro_stft_inverse" is intentionally NOT registered: confirmed dead weight in the real C++ driver
-- too (loaded by its constructor, never called from synthesize()).
--
-- inputs: input_ids (int array, CustomAlbert's own vocabulary, caller wraps with leading/trailing 0 per
-- real KModel.forward's own convention), ref_s (256 floats: [1..style_dim]=decoder style,
-- [style_dim+1..2*style_dim]=predictor style), speed (float), seed (int, seeds loom.seed_rng --
-- SineGen's rand_ini/noise draws are the only stochastic step in this whole pipeline), plus the real
-- model constants style_dim, d_model, hidden_per_dir, harmonic_num, upsample_scale, gen_istft_n_fft,
-- gen_istft_hop (KokoroConfig's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as KokoroDriver::synthesize's own return.

-- --- Layout helpers (three DISTINCT flat conventions used across this pipeline, kept as named
--     functions rather than inlined arithmetic to avoid conflating them -- see kokoro_driver.cpp's own
--     run_block_layout_a/layout_a_to_tc for the C++ precedent this mirrors):
--       row_major:  flat[t*C+c] (T-slow, C-fast)  -- loom.expand_by_duration's own host convention.
--       layout_a:   flat[c*T+t] (T-fast, C-slow)  -- ggml's [T,C] CONV_1D-family convention (CNN,
--                   AdainResBlk1d blocks, bert_encoder/adaln outputs, decoder_core's own "asr" input).
--     BiLSTM cell inputs/outputs need NEITHER -- each timestep's row is already a plain 1-D vector. ---
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

-- Host-steps one BiLSTM instance over a T x input_dim sequence, matching loom::BiLstmStepper's own
-- lstm_cell_step exactly: 2 subgraph calls (h, c) per timestep per direction, h/c threaded explicitly
-- between steps. `namespace_` selects the 4 registered cell topologies ("<namespace_>_h_fwd" etc).
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

-- Runs a 3-block AdainResBlk1d stack (F0Ntrain's own F0/N branches), each block a single Layout-A
-- graph call -- no recurrence at all, unlike the BiLSTM pieces.
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

-- Matches loom::predict_durations (duration_aligner.cpp) exactly: sigmoid-sum/speed, round-half-to-even
-- (std::nearbyint's own default FP rounding mode -- NOT round-half-away-from-zero), clamp >= 1. Real
-- float32 duration sums essentially never land on an exact .5 tie (per duration_aligner.cpp's own
-- comment), but the tie-break is implemented correctly rather than assumed away.
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

function infer(inputs)
    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir

    local s_decoder, s_predictor = {}, {}
    for i = 1, style_dim do s_decoder[i] = inputs.ref_s[i] end
    for i = 1, style_dim do s_predictor[i] = inputs.ref_s[style_dim + i] end

    -- --- CustomAlbert -> bert_encoder ---
    local positions = loom.range(0, T_text)
    local attn_mask = loom.zero_mask(T_text, T_text)
    local bert_out = loom.run_subgraph("albert", {n_tokens = T_text, n_past = 0},
                                        {tokens = inputs.input_ids, positions = positions, attn_mask = attn_mask})
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
        -- adaln's own "x" input is declared [channels,T] (flat[t*channels+c]) -- a DIFFERENT convention
        -- from to_layout_a's [T,channels] (flat[c*T+t]), so built directly here rather than reusing it.
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
    local d = x  -- (T_text, 640) -- DurationEncoder's real "d"

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations ---
    local top_out = bilstm_run("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", {n_tokens = 0, n_past = 0}, {x = top_out[t]})
    end
    local pred_dur = predict_durations(duration_logits, inputs.speed)

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
