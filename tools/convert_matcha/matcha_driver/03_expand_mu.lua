
    -- --- Row-repeat duration expansion, directly in T-fast layout (no loom.expand_by_duration reuse --
    --     see module docstring): mu_y[c*t_mel + t_out] = mu_x[c*t_text + t_src], repeated
    --     durations[t_src+1] times per source token. ---
    local mu_y = {}
    for c = 0, n_feats - 1 do
        local t_out = 0
        for t = 0, t_text - 1 do
            local v = mu_x[c * t_text + t + 1]
            for _ = 1, durations[t + 1] do
                mu_y[c * t_mel + t_out + 1] = v
                t_out = t_out + 1
            end
        end
    end
