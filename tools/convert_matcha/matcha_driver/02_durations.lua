
    -- --- Real w_ceil = ceil(exp(logw)) (length_scale=1.0) -- per-token integer durations, then
    --     extend the LAST one so t_mel is an exact multiple of 4: the Decoder topology drops all
    --     padding-mask handling (see MatchaConfig's own docstring). ---
    local durations = durations_from_logw(logw, t_text, 1.0, 1)
    local t_mel = pad_last_to_multiple(durations, array_sum(durations), 4)
