    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids
    local style_dim = inputs.style_dim
    local d_model = inputs.d_model
    local hidden_per_dir = inputs.hidden_per_dir

    local s_decoder, s_predictor = {}, {}
    for i = 1, style_dim do s_decoder[i] = inputs.ref_s[i] end
    for i = 1, style_dim do s_predictor[i] = inputs.ref_s[style_dim + i] end
