-- Per-token integer durations from a log-duration prediction: `ceil(exp(logw[t]) * length_scale)`,
-- floored at `min_each`. Matcha and VITS both predict durations this way and differ only in the two
-- parameters -- Matcha floors each duration at 1 with no scale, VITS scales and floors the TOTAL
-- instead (which is the caller's business, not this function's).
local function durations_from_logw(logw, n, length_scale, min_each)
    local durations = {}
    for t = 1, n do
        local d = math.ceil(math.exp(logw[t]) * length_scale)
        if d < min_each then d = min_each end
        durations[t] = d
    end
    return durations
end
