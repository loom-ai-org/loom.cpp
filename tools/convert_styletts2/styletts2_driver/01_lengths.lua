    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir
