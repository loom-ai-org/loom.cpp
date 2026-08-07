
    -- --- Host generate_path: w_ceil[t]=ceil(exp(logw[t])*length_scale); y_length=max(sum(w_ceil),1).
    --     Degenerates to a plain "replicate column t of m_p/logs_p for w_ceil[t] frames" expansion once
    --     x_mask/y_mask are dropped (single unpadded utterance) -- same as vits_driver.lua, no explicit
    --     attn matrix ever materialized. ---
    local w_ceil = durations_from_logw(logw, T, inputs.length_scale or LENGTH_SCALE, 0)
    local y_length = array_sum(w_ceil)
    if y_length < 1 then y_length = 1 end
