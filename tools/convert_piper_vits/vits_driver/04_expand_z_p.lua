
    -- --- z_p: [inter_channels, y_length], T-fast (z_p[c*y_length+out_frame]) -- m_p[c,t]=stats[c*T+t],
    --     logs_p[c,t]=stats[(c+inter_channels)*T+t]. ---
    local inter_channels = inputs.inter_channels
    local z_p = {}
    local out_frame = 0
    local noise_idx = 0
    local gaussian_pool = loom.gaussian_array(y_length * inter_channels)
    for t = 0, T - 1 do
        if out_frame >= y_length then break end
        for rep = 1, w_ceil[t + 1] do
            if out_frame >= y_length then break end
            for c = 0, inter_channels - 1 do
                noise_idx = noise_idx + 1
                local m = stats[c * T + t + 1]
                local logs = stats[(c + inter_channels) * T + t + 1]
                z_p[c * y_length + out_frame + 1] = m + gaussian_pool[noise_idx] * math.exp(logs) * inputs.noise_scale
            end
            out_frame = out_frame + 1
        end
    end
