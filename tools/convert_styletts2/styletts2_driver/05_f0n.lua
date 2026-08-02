
    -- --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections (bespoke, unchanged) ---
    local shared_out = run_bi_lstm("f0n_shared_lstm", en, hidden_per_dir)  -- T_frames x 512
    local f0_feat = run_resblk_stack("f0n_f0", shared_out, s_predictor)
    local n_feat = run_resblk_stack("f0n_n", shared_out, s_predictor)

    local F0_curve = run_proj1x1("f0n_f0_proj", f0_feat)
    local N_curve = run_proj1x1("f0n_n_proj", n_feat)
