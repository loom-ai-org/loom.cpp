
    -- --- decoder_vocoder: ONE MIL-traced call replaces decoder_core+sinegen+stft_forward+generator,
    --     input-for-input identical to kokoro_driver/'s own (StyleTTS2's decoder/generator/sinegen
    --     ARE Kokoro's own istftnet classes, just this checkpoint's own weights). rand_ini/noise draws in
    --     the SAME order as styletts2_driver.lua (uniform then gaussian, against the ONE shared rng_
    --     stream, AFTER the diffusion sampler's own draws) -- index 0 of rand_ini is a placeholder, zeroed
    --     BY THE GRAPH ITSELF, kept here only to preserve the exact draw-count/order the C++ oracle
    --     established. ---
    local dim = HARMONIC_NUM + 1
    local T_f0 = 2 * T_frames
    local L = T_f0 * UPSAMPLE_SCALE
    local rand_ini = {0.0}
    local u = loom.uniform_array(dim - 1)
    for i = 1, dim - 1 do rand_ini[i + 1] = u[i] end
    local noise_in = loom.gaussian_array(dim * L)
    local wsum = compute_wsum(T_frames, GEN_ISTFT_N_FFT, GEN_ISTFT_HOP, UPSAMPLE_SCALE)
