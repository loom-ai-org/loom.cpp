
    -- --- Denormalize (real denormalize(decoder_outputs, mel_mean, mel_std)) ---
    local mel = array_affine(z, inputs.mel_std, inputs.mel_mean)
