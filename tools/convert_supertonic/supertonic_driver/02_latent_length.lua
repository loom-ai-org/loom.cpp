    local duration = dur_arr[1]

    -- --- Real get_latent_mask: wav_length=duration*SAMPLE_RATE; latent_size=BASE_CHUNK_SIZE*
    --     COMPRESSION_FACTOR; T_lat=ceil(wav_length/latent_size) -- the same compute_t_lat the
    --     retired C++ SupertonicDriver did, and the frozen waveform it left behind still gates it. ---
    local wav_length = math.floor(duration * SAMPLE_RATE)
    local latent_size = BASE_CHUNK_SIZE * COMPRESSION_FACTOR
    local t_lat = math.floor((wav_length + latent_size - 1) / latent_size)
