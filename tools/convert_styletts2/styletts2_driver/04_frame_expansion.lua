
    -- --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder,
    --     bespoke, unchanged) ---
    local T_frames = array_sum(pred_dur)
    local d_channels = D_MODEL + STYLE_DIM
    local en = from_row_major(loom.expand_by_duration(to_row_major(d, d_channels), T_text, d_channels, pred_dur),
                               T_frames, d_channels)

    local cnn_flat, cnn_shape = loom.run_subgraph("text_encoder_cnn", {n_tokens = T_text, n_past = 0}, {tokens = inputs.input_ids})
    local te_channels = cnn_shape[2]
    local cnn_rows = from_layout_a(cnn_flat, T_text, te_channels)
    local t_en = run_bi_lstm("text_encoder_lstm", cnn_rows, HIDDEN_PER_DIR)  -- T_text x 512
    local asr = from_row_major(loom.expand_by_duration(to_row_major(t_en, 512), T_text, 512, pred_dur),
                                T_frames, 512)
