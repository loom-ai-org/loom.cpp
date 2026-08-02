-- Duration expansion in T-FAST layout (`flat[c*T + t]`): repeats source frame `t` of every channel
-- `durations[t+1]` times. The host-side counterpart of the `loom.expand_by_duration` C++ binding, which
-- expects the opposite C-fast "rows_flat" convention -- a traced module that keeps its native torch
-- (1,C,T) layout (Matcha's TextEncoder) produces T-fast and cannot use the binding.
local function repeat_by_duration_tfast(src, channels, t_src, t_dst, durations)
    local out = {}
    for c = 0, channels - 1 do
        local t_out = 0
        for t = 0, t_src - 1 do
            local v = src[c * t_src + t + 1]
            for _ = 1, durations[t + 1] do
                out[c * t_dst + t_out + 1] = v
                t_out = t_out + 1
            end
        end
    end
    return out
end
