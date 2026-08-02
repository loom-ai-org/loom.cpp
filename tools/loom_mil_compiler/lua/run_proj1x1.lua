local function run_proj1x1(name, feat)
    local T = #feat
    local C = #feat[1]
    return loom.run_subgraph(name, {n_tokens = T, n_past = 0}, {x = to_layout_a(feat, T, C)})
end
