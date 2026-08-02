
    -- --- Denormalize (real denormalize(decoder_outputs, mel_mean, mel_std)) ---
    local mel = z
    for i = 1, #mel do
        mel[i] = mel[i] * inputs.mel_std + inputs.mel_mean
    end
