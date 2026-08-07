    loom.seed_rng(inputs.seed)

    local T_text = #inputs.input_ids

    local s_decoder = array_slice(inputs.ref_s, 1, STYLE_DIM)
    local s_predictor = array_slice(inputs.ref_s, STYLE_DIM + 1, STYLE_DIM)
