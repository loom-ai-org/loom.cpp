
    -- --- Row-repeat duration expansion, directly in T-fast layout (no loom.expand_by_duration reuse
    --     -- see module docstring): mu_y[c*t_mel + t_out] = mu_x[c*t_text + t_src], repeated
    --     durations[t_src+1] times per source token. ---
    local mu_y = repeat_by_duration_tfast(mu_x, n_feats, t_text, t_mel, durations)
