-- Extends the LAST duration so the total is an exact multiple of `k`, returning the new total. Matcha's
-- Decoder topology drops all padding-mask handling and so requires a multiple of 4; a model whose
-- decoder has the same constraint needs exactly this and nothing more.
local function pad_last_to_multiple(durations, total, k)
    local remainder = total % k
    if remainder ~= 0 then
        local extra = k - remainder
        durations[#durations] = durations[#durations] + extra
        total = total + extra
    end
    return total
end
