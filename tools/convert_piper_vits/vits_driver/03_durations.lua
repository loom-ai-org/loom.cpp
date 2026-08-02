
    -- --- Host generate_path: w_ceil[t]=ceil(exp(logw[t])*length_scale); y_length=max(sum(w_ceil),1).
    --     Degenerates to a plain "replicate column t of m_p/logs_p for w_ceil[t] frames" expansion once
    --     x_mask/y_mask are dropped (single unpadded utterance) -- same as vits_driver.lua, no explicit
    --     attn matrix ever materialized. ---
    local w_ceil = {}
    local y_length = 0
    for t = 1, T do
        w_ceil[t] = math.ceil(math.exp(logw[t]) * inputs.length_scale)
        y_length = y_length + w_ceil[t]
    end
    if y_length < 1 then y_length = 1 end
