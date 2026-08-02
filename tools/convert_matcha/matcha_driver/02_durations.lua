
    -- --- Real w_ceil = ceil(exp(logw)) (length_scale=1.0) -- per-token integer durations. ---
    local durations = {}
    local t_mel = 0
    for t = 1, t_text do
        local d = math.ceil(math.exp(logw[t]))
        if d < 1 then d = 1 end
        durations[t] = d
        t_mel = t_mel + d
    end
    -- Extend the LAST token's duration so t_mel is an exact multiple of 4 -- the Decoder topology
    -- drops all padding-mask handling (see MatchaConfig's own docstring).
    local remainder = t_mel % 4
    if remainder ~= 0 then
        local extra = 4 - remainder
        durations[t_text] = durations[t_text] + extra
        t_mel = t_mel + extra
    end
