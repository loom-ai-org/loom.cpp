-- Round-half-to-even, matching `loom::predict_durations`'s own rounding (duration_aligner.cpp)
-- exactly -- which is why it is this and not `math.floor(x + 0.5)`.
local function round_half_to_even(x)
    local f = math.floor(x)
    local diff = x - f
    if diff < 0.5 then return f
    elseif diff > 0.5 then return f + 1
    elseif f % 2 == 0 then return f
    else return f + 1 end
end
