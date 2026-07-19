"""Constructs the constant (checkpoint-independent) tensors needed to express OpenAI Whisper's log-mel
frontend (whisper/audio.py::log_mel_spectrogram) as ordinary loom-engine graph nodes, mirroring
tools/convert_nemo/mel_common.py's role for NeMo's FilterbankFeatures -- same idea (DFT-via-conv trick +
librosa mel filterbank baked as GGUF constants), genuinely different formula in several confirmed spots,
so this is its own module rather than a shared one:

  1. NO preemphasis (NeMo has it, Whisper does not).
  2. Window is torch.hann_window(N_FFT) with `periodic=True` (the torch default) -- a DIFFERENT formula
     from NeMo's `periodic=False` convention (mel_common.py's own hann_window_centered): periodic uses
     denominator N, not N-1 (confirmed directly: torch.hann_window(8,periodic=True)[1]=0.1464466,
     torch.hann_window(8,periodic=False)[1]=0.1882551 -- genuinely different numbers, not a rounding
     artifact). Also win_length == n_fft == 400 always for Whisper, so there's no centered-zero-pad-a-
     shorter-window case to handle at all (unlike NeMo's win_length=400 < n_fft=512).
  3. STFT centering pads with REFLECT, not zero/constant (torch.stft's default pad_mode='reflect' when
     center=True) -- confirmed numerically (see verify_whisper_mel.py scratch check, matched real
     whisper.audio.log_mel_spectrogram to 3.7e-6 using reflect padding; using zero-padding instead does
     NOT match). ggml's native CONV_1D padding (its p0 attr) is zero-only -- there is no reflect-pad
     primitive in this engine and none is being added for this, because Whisper always runs on a FIXED
     30s window (pad_or_trim to N_SAMPLES=480000 samples), so the reflect-padded waveform has a fixed,
     known-at-conversion-time shape -- reflect-padding is therefore done as a host-side preprocessing
     step (numpy, before the waveform ever enters the graph), same "host computes a fixed-shape value,
     feeds it in as a declared input" precedent used throughout this project (VITS's noise injection,
     etc.), not a gap needing new engine code. See pad_reflect() below -- callers must apply it to the
     raw waveform themselves before writing it to the graph's "waveform" input.
  4. Whisper drops the LAST STFT time-frame (`stft[..., :-1]`) before the power calc -- NeMo does not.
  5. mel filterbank: librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2) with librosa's OWN default
     norm ('slaney') -- same call shape as NeMo's, just without an explicit norm= override (which also
     happens to be 'slaney', librosa's default, so behaviorally identical to NeMo's call here).
  6. log10 (not natural log), clamped to a 1e-10 floor, THEN a GLOBAL (whole-spectrogram, not per-frame/
     per-mel-bin) dynamic-range clamp to [global_max-8, +inf), then affine (x+4)/4 -- NO per-feature CMVN
     normalization at all (NeMo's normalize='per_feature' step has no Whisper equivalent).

Confirmed against the real installed `openai-whisper` package's whisper/audio.py directly, and verified
end-to-end numerically (numpy re-derivation vs. real whisper.audio.log_mel_spectrogram on synthetic
noise, max_abs_diff=3.7e-6) before writing anything here.
"""
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 30 * SAMPLE_RATE  # fixed 30s window Whisper always pads/trims to -- N_FRAMES=3000 mel frames


def mel_hparams(n_mels: int) -> dict:
    n_freq = N_FFT // 2 + 1  # 201
    return {
        "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "n_freq": n_freq, "n_mels": n_mels, "n_samples": N_SAMPLES,
        "reflect_pad": N_FFT // 2,  # applied host-side via pad_reflect(), NOT ggml's CONV_1D p0 (zero-only)
    }


def pad_reflect(waveform: np.ndarray, pad: int) -> np.ndarray:
    """Host-side reflect-pad matching torch.stft(center=True)'s default pad_mode='reflect' -- must be
    applied to the raw (already pad_or_trim'd-to-30s) waveform before it's written to the graph's
    "waveform" input; the in-graph CONV_1D nodes use p0=0 (no additional padding) since this is already
    baked in."""
    return np.pad(waveform, (pad, pad), mode="reflect").astype(np.float32)


def periodic_hann(n_fft: int) -> np.ndarray:
    """torch.hann_window(n_fft, periodic=True) -- Whisper's convention, NOT NeMo's periodic=False one
    (mel_common.py::hann_window_centered). Whisper's win_length always equals n_fft (400), so there's no
    separate centered-shorter-window case to handle."""
    n = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)


def build_dft_kernels(n_fft: int) -> tuple:
    """Returns (cos_kernel, sin_kernel), each shaped (n_freq, 1, n_fft) -- same PyTorch Conv1d
    (out_channels, in_channels, kernel_size) convention as mel_common.py::build_dft_kernels, so
    GGUFWriter.add_tensor round-trips to ggml's ne=[n_fft, 1, n_freq] the same way. Whisper's rfft
    imaginary part is the NEGATIVE of the plain sine projection (e^{-i*2*pi*k*n/N} = cos(..) - i*sin(..)),
    confirmed numerically against torch.stft's real output -- callers must negate the sin-kernel's
    convolution result, not bake the negation into the kernel itself (kept symmetric with mel_common.py's
    own convention, where NeMo's usage never needed to negate since only the squared magnitude is used
    there too... same here, in fact: only re**2+im**2 is used, so the sign is actually irrelevant for
    Whisper's magnitude-only power calc. Noted for correctness/clarity but doesn't affect the result.)
    """
    n_freq = n_fft // 2 + 1
    window = periodic_hann(n_fft)
    k = np.arange(n_fft, dtype=np.float64)
    oc = np.arange(n_freq, dtype=np.float64)[:, None]
    angle = 2.0 * np.pi * oc * k[None, :] / n_fft
    cos_kernel = (window[None, :] * np.cos(angle)).astype(np.float32)[:, None, :]
    sin_kernel = (window[None, :] * np.sin(angle)).astype(np.float32)[:, None, :]
    return cos_kernel, sin_kernel


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """(n_mels, n_freq) -- same layout/rationale as mel_common.py::build_mel_filterbank."""
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0,
                                fmax=sample_rate / 2).astype(np.float32)
