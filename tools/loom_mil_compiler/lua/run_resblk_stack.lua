-- Runs a 3-block AdainResBlk1d stack (F0Ntrain's F0/N branches), threading style through each block
-- and crossing between row and Layout-A representations around each call.
local function run_resblk_stack(name_prefix, x_rows, style)
    local cur = x_rows
    for i = 0, 2 do
        local T_in = #cur
        local dim_in = #cur[1]
        local flat, shape = loom.run_subgraph(name_prefix .. "_block" .. i, {n_tokens = T_in, n_past = 0},
                                               {x = to_layout_a(cur, T_in, dim_in), style = style})
        cur = from_layout_a(flat, shape[1], shape[2])
    end
    return cur
end
