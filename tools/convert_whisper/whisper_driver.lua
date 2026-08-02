-- Lua port of loom::WhisperDriver::transcribe (src/core/whisper_driver.cpp), validating the
-- procedural-generalization architecture (LOOM_PROCEDURAL_GENERALIZATION.md / LOOM_MIL_CONVERSION.md)
-- against the hardest existing driver: a real autoregressive while-loop with a persistent KvCache
-- (bound to the "decoder" module at registration time, on the C++ side) and argmax sampling.
--
-- Expects two modules pre-registered by the host (LoomLuaBridge::register_module): "encoder" (no
-- KvCache, every shape fixed at conversion time) and "decoder" (bound to a persistent KvCache spanning
-- the whole while loop below, matching WhisperDriver's own constructor).
--
-- inputs: waveform (flat f32 array, host-reflect-padded/pad_or_trim'd to 30s -- same convention as
-- WhisperDriver::transcribe's own `waveform_padded` parameter), prompt_tokens (int array), n_audio_ctx
-- (int, 1500 for every real Whisper checkpoint size), max_new_tokens (int), eot_token (int; negative
-- disables the early-stop check, matching WhisperConfig::eot_token's own convention).
--
-- Returns: a flat array of generated token ids (NOT including the prompt), same convention as
-- WhisperDriver::transcribe's own return value.
function infer(inputs)
    local n_audio_ctx = inputs.n_audio_ctx

    -- --- Encoder: one fixed-shape pass (n_tokens/n_past are unused by this topology -- every shape is
    --     a compile-time constant, matching WhisperDriver's own `encoder_builder_->build(0, 0)`). ---
    local enc_mask = loom.zero_mask(n_audio_ctx, n_audio_ctx)
    local xa = loom.run_subgraph("encoder", {n_tokens = 0, n_past = 0}, { waveform = inputs.waveform, enc_attn_mask = enc_mask })

    -- --- Decoder: prefill the prompt in one shot, then greedily decode one token at a time. ---
    local n_past = 0
    local generated = {}

    local n_prompt_tokens = #inputs.prompt_tokens
    local logits, shape = loom.run_subgraph("decoder", {n_tokens = n_prompt_tokens, n_past = n_past}, {
        tokens = inputs.prompt_tokens,
        positions = loom.range(n_past, n_prompt_tokens),
        kq_mask = loom.causal_mask(n_prompt_tokens, n_past),
        xa = xa,
        xa_mask = loom.zero_mask(n_audio_ctx, n_prompt_tokens),
    })
    n_past = n_past + n_prompt_tokens
    local n_vocab = shape[1]
    local next_token = loom.argmax_row(logits, n_vocab, n_prompt_tokens - 1)
    table.insert(generated, next_token)

    if inputs.eot_token >= 0 and next_token == inputs.eot_token then
        return generated
    end

    while #generated < inputs.max_new_tokens do
        local step_logits, step_shape = loom.run_subgraph("decoder", {n_tokens = 1, n_past = n_past}, {
            tokens = { generated[#generated] },
            positions = loom.range(n_past, 1),
            kq_mask = loom.causal_mask(1, n_past),
            xa = xa,
            xa_mask = loom.zero_mask(n_audio_ctx, 1),
        })
        n_past = n_past + 1
        local step_n_vocab = step_shape[1]
        local step_next = loom.argmax_row(step_logits, step_n_vocab, 0)
        table.insert(generated, step_next)

        if inputs.eot_token >= 0 and step_next == inputs.eot_token then
            break
        end
    end

    return generated
end
