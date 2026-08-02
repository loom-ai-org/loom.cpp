-- `count` elements of a flat array starting at 1-based `from`. Kokoro and StyleTTS2 both split one
-- packed style vector into its decoder and predictor halves with two copies of this loop.
local function array_slice(a, from, count)
    local out = {}
    for i = 1, count do out[i] = a[from + i - 1] end
    return out
end
