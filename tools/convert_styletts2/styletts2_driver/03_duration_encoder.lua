
    -- --- bert_encoder (existing bespoke topology, unchanged) ---
    local d_en_flat = loom.run_subgraph("bert_encoder", {n_tokens = T_text, n_past = 0}, {x = bert_out})  -- Layout A [T,512]

    -- --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating style (bespoke, unchanged) ---
    local x = {}
    for t = 0, T_text - 1 do
        local row = {}
        for c = 0, d_model - 1 do row[c + 1] = d_en_flat[c * T_text + t + 1] end
        for s = 1, style_dim do row[d_model + s] = s_predictor[s] end
        x[t + 1] = row
    end
    for i = 0, 2 do
        local lstm_out = run_bi_lstm("duration_lstm_" .. i, x, hidden_per_dir)  -- T_text x 512
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

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations (bespoke, unchanged) ---
    local top_out = run_bi_lstm("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", {n_tokens = 0, n_past = 0}, {x = top_out[t]})
    end
    -- Real quirk (no /speed at all -- the real demo's own inference() has no such parameter):
    -- pred_dur[-1] += 5, padding the last token's duration.
    local pred_dur = predict_durations(duration_logits, 1.0)
    pred_dur[#pred_dur] = pred_dur[#pred_dur] + 5
