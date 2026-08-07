    -- The encoder output is the one tensor this driver genuinely indexes host-side: the joint consumes
    -- ONE frame at a time and a retained reference copies a whole tensor, so the frames come back as a
    -- flat array and are sliced here. [n_embd, n_frames], n_embd fastest.
    local n_embd, n_frames = _enc_shape[1], _enc_shape[2]

    local tokens = {}
    local last_label = BLANK_ID
    local h, c = {}, {}
    for l = 1, N_PRED_LAYERS do
        h[l], c[l] = {}, {}
        for i = 1, PRED_HIDDEN do h[l][i], c[l][i] = 0.0, 0.0 end
    end

    -- The prediction network runs once per EMITTED TOKEN, not once per frame: its output is a pure
    -- function of (last_label, h, c), and all three change only on emission. `top_h == nil` means "must
    -- recompute", so a blank costs one joint call and nothing else -- and most frames of real audio are
    -- blank. Same restructuring the C++ decoder got before it was retired; the discarded recompute it
    -- replaces could not have differed, which is why this is equivalence and not an approximation.
    local top_h = nil
    local t = 0
    while t < n_frames do
        local frame = {}
        local base = t * n_embd
        for i = 1, n_embd do frame[i] = _enc[base + i] end

        local symbols = 0
        local advanced = false
        while symbols < MAX_SYMBOLS_PER_STEP do
            if top_h == nil then
                local layer_input = loom.run_subgraph('embed', {n_tokens = 0, n_past = 0},
                                                       {last_label = {last_label}})
                for l = 1, N_PRED_LAYERS do
                    local h_new, c_new = loom.run_subgraph('pred_lstm_l' .. (l - 1) .. '_fwd',
                                                            {n_tokens = 0, n_past = 0},
                                                            {layer_input = layer_input,
                                                             h_prev = h[l], c_prev = c[l]})
                    h[l], c[l] = h_new, c_new
                    layer_input = h_new
                end
                top_h = h[N_PRED_LAYERS]
            end

            -- Retained, so the 8198-wide joint output never becomes a Lua table: the token head is
            -- reduced engine-side and only the handful of duration logits are marshalled.
            loom.run_subgraph_and_retain('joint', {n_tokens = 0, n_past = 0},
                                          {encoder_frame = frame, decoder_out = top_h})
            local k = loom.argmax_row('joint', 0)

            -- Plain RNN-T has no duration head at all, and every blank advances exactly one frame.
            local skip = 1
            if N_DURATIONS > 0 then
                local dur = loom.get_output('joint', 2)
                local best = 1
                for i = 2, N_DURATIONS do
                    if dur[i] > dur[best] then best = i end
                end
                skip = DURATIONS[best]
            end

            if k ~= BLANK_ID then
                tokens[#tokens + 1] = k
                last_label = k
                top_h = nil          -- last_label moved, so the cached prediction no longer applies
                if N_DURATIONS == 0 then skip = 0 end
            elseif skip == 0 then
                skip = 1             -- a blank must advance, or decoding spins on one frame forever
            end

            symbols = symbols + 1
            t = t + skip
            if skip > 0 then
                advanced = true
                break
            end
        end
        if not advanced then
            -- Defensive bound, not part of the TDT algorithm itself: guards a model that keeps emitting
            -- duration-0 non-blanks forever. Same fallback the C++ decoder carried.
            t = t + 1
        end
    end
