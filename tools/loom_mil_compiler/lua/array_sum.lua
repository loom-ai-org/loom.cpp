-- Sum of a flat array. Three drivers compute a total frame count this way (Matcha's `t_mel`, VITS's
-- `y_length`, Kokoro/StyleTTS2's `T_frames`), each with its own inline loop before this existed.
local function array_sum(a)
    local total = 0
    for i = 1, #a do total = total + a[i] end
    return total
end
