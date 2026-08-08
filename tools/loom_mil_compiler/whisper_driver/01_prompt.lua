    -- The decode prompt, built HERE rather than handed over by the caller: which tokens a Whisper
    -- prompt needs is a property of the checkpoint (a `.en` model has no language or task token at
    -- all), and the ids come from `whisper_export.decoder_prompt_constants`, so a host passes audio and
    -- at most a language -- never a token prefix.
    --
    -- Resolution order for the language, and it is the general rule for anything a model can infer
    -- about its own input: an explicit argument wins; absent, detect if this model can; otherwise fall
    -- back to the default, which here is "no language token", the only correct answer for `.en`.
    local _prompt = { SOT }
    -- What the model has already produced before the decode loop starts. Empty unless the
    -- forced opening timestamp below fills it in.
    local _gen0 = {}
    local _language = inputs.language
    if LANG_HI > 0 then
        if _language == nil then
            -- Detection is one decoder step from SOT alone, argmax restricted to the language block --
            -- unrestricted it would answer with whichever ordinary word scores highest, since the
            -- language tokens live inside the same 51865-wide transcript vocabulary.
            --
            -- It costs no extra encoder pass: `xa` is the encoder output already retained above. It
            -- writes KV cell 0, which the prefill below then overwrites with the identical K/V (same
            -- token SOT, same position 0) before attending to anything.
            loom.run_subgraph_and_retain('decoder', {n_tokens = 1, n_past = 0}, {
                tokens = { SOT },
                position_ids = loom.range(0, 1),
                attention_mask = loom.causal_mask(1, 0),
                xa = {from = 'encoder'},
            })
            _language = loom.argmax_row_range('decoder', 0, LANG_LO, LANG_HI)
        end
        table.insert(_prompt, _language)
        -- `translate` only when asked for it by id; transcribe is the default task.
        local _task = inputs.task
        if _task == nil or _task == 0 then _task = TRANSCRIBE end
        if _task > 0 then table.insert(_prompt, _task) end
    end
    -- Timestamps are opt-in: with this token the model emits plain text, without it the transcript is
    -- interleaved with <|0.00|>-style tokens a caller then has to interpret.
    if NO_TIMESTAMPS > 0 and not inputs.timestamps then
        table.insert(_prompt, NO_TIMESTAMPS)
    elseif TS_HI > 0 then
        -- **Omitting <|notimestamps|> is not enough to get timestamps, and this is the whole reason
        -- this branch exists.** Left to itself the model EMITS <|notimestamps|> as its first token --
        -- it is the highest-scoring one on most audio -- and then produces no timestamps at all,
        -- which is a decision about the decode rather than a transcript. Whisper's own rule is that
        -- the token after the task must be a timestamp, so force that: one prefill of the prompt so
        -- far, then an argmax restricted to the timestamp block, which cannot answer with
        -- <|notimestamps|> or with a word.
        --
        -- The prompt grows by the token the MODEL chose (in practice <|0.00|>, since a window's first
        -- segment starts at its own beginning), and the loop below re-prefills the whole thing -- the
        -- same overwrite-with-identical-K/V as the language detection above.
        loom.run_subgraph_and_retain('decoder', {n_tokens = #_prompt, n_past = 0}, {
            tokens = _prompt,
            position_ids = loom.range(0, #_prompt),
            attention_mask = loom.causal_mask(#_prompt, 0),
            xa = {from = 'encoder'},
        })
        local _first_ts = loom.argmax_row_range('decoder', #_prompt - 1, TS_LO, TS_HI)
        table.insert(_prompt, _first_ts)
        table.insert(_gen0, _first_ts)
    end
