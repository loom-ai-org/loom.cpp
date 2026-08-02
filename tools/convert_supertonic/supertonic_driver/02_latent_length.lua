    local duration = dur_arr[1]

    -- --- Real get_latent_mask: wav_length=duration*sample_rate; latent_size=base_chunk_size*
    --     compression_factor; T_lat=ceil(wav_length/latent_size) -- matches
    --     src/core/supertonic_driver.cpp's own compute_t_lat exactly. ---
    local wav_length = math.floor(duration * inputs.sample_rate)
    local latent_size = inputs.base_chunk_size * inputs.compression_factor
    local t_lat = math.floor((wav_length + latent_size - 1) / latent_size)
