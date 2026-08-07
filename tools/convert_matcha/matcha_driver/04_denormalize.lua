
    -- --- Denormalize (real denormalize(decoder_outputs, mel_mean, mel_std)) ---
    local mel = array_affine(z, MEL_STD, MEL_MEAN)
