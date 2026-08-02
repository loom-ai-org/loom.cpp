-- Matches export_kokoro_mil.py's own `compute_wsum_np` (== tools/convert_kokoro/kokoro_stft_common.py's
-- `compute_wsum`, driven by T_frames directly rather than a real waveform's own length) -- "decoder_
-- vocoder" needs this as a genuine declared LEAF input (a fixed function of T_frames only, not of any
-- real data, the same "host-precomputed real-valued denominator" convention wsum already used in
-- kokoro_driver.lua, just computed from T_frames up front here instead of from the (no-longer-host-
-- visible, now fully in-graph) SineGen output's own length).
local function compute_wsum(t_frames, n_fft, hop, upsample_scale)
    local t_f0 = 2 * t_frames
    local length = t_f0 * upsample_scale
    local pad = n_fft / 2
    local padded_len = length + 2 * pad
    local t_har = math.floor((padded_len - n_fft) / hop) + 1
    local out_len_full = (t_har - 1) * hop + n_fft
    local window = {}
    for i = 0, n_fft - 1 do window[i + 1] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / n_fft) end
    local wsum = {}
    for i = 1, out_len_full do wsum[i] = 0.0 end
    for t = 0, t_har - 1 do
        for i = 0, n_fft - 1 do
            local idx = t * hop + i + 1
            wsum[idx] = wsum[idx] + window[i + 1] * window[i + 1]
        end
    end
    return wsum
end
