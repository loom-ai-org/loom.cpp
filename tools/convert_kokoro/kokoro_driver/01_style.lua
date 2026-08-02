    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir

    local s_decoder = array_slice(inputs.ref_s, 1, style_dim)
    local s_predictor = array_slice(inputs.ref_s, style_dim + 1, style_dim)
