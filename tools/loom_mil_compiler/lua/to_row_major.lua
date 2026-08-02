-- --- Layout helpers, same two conventions kokoro_driver.lua's own header comment already documents:
--       row_major:  flat[t*C+c] (T-slow, C-fast)  -- loom.expand_by_duration's own host convention, AND
--                   this file's own "albert_bert_encoder" output convention (see module docstring).
--       layout_a:   flat[c*T+t] (T-fast, C-slow)  -- ggml's [T,C] CONV_1D-family convention (CNN,
--                   AdainResBlk1d blocks, bert_encoder/adaln outputs). ---
local function to_row_major(rows, C)
    local T = #rows
    local flat = {}
    for t = 0, T - 1 do
        for c = 0, C - 1 do flat[t * C + c + 1] = rows[t + 1][c + 1] end
    end
    return flat
end
