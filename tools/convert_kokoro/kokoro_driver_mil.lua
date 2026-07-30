-- Lua orchestration for the MIL-traced Kokoro export (export_kokoro_mil.py), analogous to
-- kokoro_driver.lua (the hand-built-topology driver) but wiring in the TWO machine-traced combined
-- topologies export_kokoro_mil.py produces ("albert_bert_encoder", "decoder_vocoder") in place of FOUR
-- of kokoro_driver.lua's own bespoke topology calls each ("albert"+"bert_encoder", and
-- "decoder_core"+"sinegen"+"stft_forward"+"generator" respectively). Kokoro leans heavily on
-- `torch.nn.LSTM` (ggml has no native LSTM op) -- deliberate scoping decision (see BACKLOG.md): the
-- LSTM-bound pieces (TextEncoder's BiLSTM, DurationEncoder, predictor.lstm, F0Ntrain's shared LSTM)
-- REMAIN the existing bespoke, hand-built topologies (unchanged from kokoro_driver.lua), loaded from the
-- separate bespoke kokoro.gguf (tools/convert_kokoro/convert_kokoro_lua_all.py) alongside this driver's
-- own kokoro_mil.gguf -- register_module doesn't care which GgufModel a topology's weights live in, only
-- that all of them share one LoomLuaBridge instance.
--
-- What's different from kokoro_driver.lua, and why this isn't just that file with two calls swapped:
--   - No `positions`/`attn_mask` inputs to "albert_bert_encoder" at all: the MIL-traced CustomAlbert
--     computes position ids (from a registered buffer, sliced dynamically) and the additive attention
--     mask (always all-zeros -- real usage is always a single, unpadded utterance, same convention
--     convert_kokoro_albert.py's own bespoke topology already established) IN-GRAPH now, not host-side.
--   - "albert_bert_encoder" returns TIME-MAJOR (T,512) (`flat[t*512+c]`, this file's own "row_major"
--     convention below) instead of kokoro_driver.lua's "d_en_flat" Layout-A convention (`flat[c*T+t]`) --
--     a deliberate choice in export_kokoro_mil.py's own AlbertBertEncoderWrapper to avoid returning a
--     bare `.transpose()` as a traced graph's own output (a live non-contiguous view read in PRE-permute
--     order by this project's raw contiguous-byte-copy GGUF/weight extraction -- the exact bug
--     export_vits_mil.py's own StatsWrapper already found and worked around for VITS's `stats` output).
--     `from_row_major` converts straight to the per-timestep rows DurationEncoder's own "x" construction
--     needs, actually SIMPLER than kokoro_driver.lua's own manual c*T+t indexing loop.
--   - "decoder_vocoder" replaces FOUR bespoke calls (decoder_core, sinegen, stft_forward, generator) with
--     ONE: it takes asr/F0_curve/N_curve/style/rand_ini/noise_in/wsum directly and returns the finished
--     WAVEFORM -- no host-side har (STFT mag/phase) assembly or Generator-input wiring needed at all,
--     only the SAME rand_ini/noise/wsum host-precomputation kokoro_driver.lua's own generator call
--     already required (F0Ntrain and everything upstream of the decoder is IDENTICAL to kokoro_driver.lua
--     -- still bespoke/LSTM-bound, unchanged).
--
-- Expects topologies pre-registered by the host, from TWO GgufModel instances sharing one bridge:
--   kokoro_mil.gguf: "albert_bert_encoder", "decoder_vocoder"
--   kokoro.gguf (bespoke): "text_encoder_cnn", "text_encoder_lstm_{h,c}_{fwd,bwd}",
--     "duration_lstm_{0,1,2}_{h,c}_{fwd,bwd}", "duration_adaln_{0,1,2}", "top_lstm_{h,c}_{fwd,bwd}",
--     "duration_proj", "f0n_shared_lstm_{h,c}_{fwd,bwd}", "f0n_f0_block{0,1,2}", "f0n_n_block{0,1,2}",
--     "f0n_f0_proj", "f0n_n_proj" -- none use a KvCache.
--
-- inputs: input_ids (int array, CustomAlbert's own vocabulary, caller wraps with leading/trailing 0 per
-- real KModel.forward's own convention), ref_s (256 floats: [1..style_dim]=decoder style,
-- [style_dim+1..2*style_dim]=predictor style), speed (float), seed (int, seeds loom.seed_rng --
-- SineGen's rand_ini/noise draws are the only stochastic step in this whole pipeline, SAME draw order as
-- kokoro_driver.lua: uniform first then gaussian, against the ONE shared rng_ stream), plus the real
-- model constants style_dim, d_model, hidden_per_dir, harmonic_num, upsample_scale, gen_istft_n_fft,
-- gen_istft_hop (KokoroConfig's own real defaults).
--
-- Returns: the raw waveform (flat f32 array), same convention as kokoro_driver.lua's own return.

-- --- Layout helpers, same two conventions kokoro_driver.lua's own header comment already documents:
--       row_major:  flat[t*C+c] (T-slow, C-fast)  -- loom.expand_by_duration's own host convention, AND
--                   this file's own "albert_bert_encoder" output convention (see module docstring).
--       layout_a:   flat[c*T+t] (T-fast, C-slow)  -- ggml's [T,C] CONV_1D-family convention (CNN,
--                   AdainResBlk1d blocks, bert_encoder/adaln outputs). ---
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

-- Host-steps one BiLSTM instance over a T x input_dim sequence -- identical to kokoro_driver.lua's own
-- `bilstm_run` (the LSTM-bound topologies it drives are byte-for-byte the same bespoke ones).
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

-- Runs a 3-block AdainResBlk1d stack (F0Ntrain's own F0/N branches) -- identical to kokoro_driver.lua's
-- own `run_resblk_stack`.
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

-- Matches loom::predict_durations (duration_aligner.cpp) exactly -- identical to kokoro_driver.lua's own.
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

-- Matches export_kokoro_mil.py's own `compute_wsum_np` (== tools/convert_kokoro/kokoro_stft_common.py's
-- `compute_wsum`, driven by T_frames directly rather than a real waveform's own length) -- "decoder_
-- vocoder" needs this as a genuine declared LEAF input (a fixed function of T_frames only, not of any
-- real data, the same "host-precomputed real-valued denominator" convention wsum already used in
-- kokoro_driver.lua, just computed from T_frames up front here instead of from the (no-longer-host-
-- visible, now fully in-graph) SineGen output's own length).
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

    local s_decoder, s_predictor = {}, {}
    for i = 1, style_dim do s_decoder[i] = inputs.ref_s[i] end
    for i = 1, style_dim do s_predictor[i] = inputs.ref_s[style_dim + i] end

    -- --- CustomAlbert + bert_encoder, ONE MIL-traced call -> d_en, time-major (T,512) (see module
    --     docstring for why this convention, not kokoro_driver.lua's own Layout-A one). ---
    local d_en_flat = loom.run_subgraph("albert_bert_encoder", {n_tokens = T_text, n_past = 0}, {tokens = inputs.input_ids})
    local d_en_rows = from_row_major(d_en_flat, T_text, d_model)

    -- --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating style (bespoke, unchanged) ---
    local x = {}
    for t = 1, T_text do
        local row = {}
        for c = 1, d_model do row[c] = d_en_rows[t][c] end
        for s = 1, style_dim do row[d_model + s] = s_predictor[s] end
        x[t] = row
    end
    for i = 0, 2 do
        local lstm_out = bilstm_run("duration_lstm_" .. i, x, hidden_per_dir)  -- T_text x 512
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

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations (bespoke, unchanged) ---
    local top_out = bilstm_run("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", {n_tokens = 0, n_past = 0}, {x = top_out[t]})
    end
    local pred_dur = predict_durations(duration_logits, inputs.speed)

    -- --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder,
    --     bespoke, unchanged) ---
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

    -- --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections (bespoke, unchanged) ---
    local shared_out = bilstm_run("f0n_shared_lstm", en, hidden_per_dir)  -- T_frames x 512
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)

    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)

    -- --- decoder_vocoder: ONE MIL-traced call replaces decoder_core+sinegen+stft_forward+generator.
    --     rand_ini/noise draws in the SAME order as kokoro_driver.lua (uniform then gaussian, against the
    --     ONE shared rng_ stream) -- index 0 of rand_ini is a placeholder, zeroed BY THE GRAPH ITSELF
    --     (`AlbertBertEncoderWrapper`'s sibling `_f02sine_traceable` builds `rand_ini_full` by replacing
    --     index 0 with a real zero unconditionally), kept here only to preserve the exact draw-count/order
    --     kokoro_driver.lua's own C++ oracle established. ---
    local dim = inputs.harmonic_num + 1
    local T_f0 = 2 * T_frames
    local L = T_f0 * inputs.upsample_scale
    local rand_ini = {0.0}
    local u = loom.uniform_array(dim - 1)
    for i = 1, dim - 1 do rand_ini[i + 1] = u[i] end
    local noise_in = loom.gaussian_array(dim * L)
    local wsum = compute_wsum(T_frames, inputs.gen_istft_n_fft, inputs.gen_istft_hop, inputs.upsample_scale)

    local waveform = loom.run_subgraph("decoder_vocoder", {n_enc_frames = T_frames, n_past = 0}, {
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
