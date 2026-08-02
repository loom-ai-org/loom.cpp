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
    local d = x  -- (T_text, 640) -- DurationEncoder's real "d"

    -- --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations (bespoke, unchanged) ---
    local top_out = run_bi_lstm("top_lstm", d, hidden_per_dir)  -- T_text x 512
    local duration_logits = {}
    for t = 1, T_text do
        duration_logits[t] = loom.run_subgraph("duration_proj", {n_tokens = 0, n_past = 0}, {x = top_out[t]})
    end
    local pred_dur = predict_durations(duration_logits, inputs.speed)
