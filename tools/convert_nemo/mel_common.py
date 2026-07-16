"""Constructs the constant (checkpoint-independent) tensors needed to express NeMo's
AudioToMelSpectrogramPreprocessor (nemo/collections/asr/parts/preprocessing/features.py's
FilterbankFeatures, default settings: window='hann', mag_power=2.0, mel_norm='slaney',
log_zero_guard_type='add', normalize='per_feature', exact_pad=False i.e. center=True/pad_mode='constant')
as ordinary loom-engine graph nodes, so mel-spectrogram extraction lives inside the GGUF graph instead of
being a caller-supplied precomputed tensor.

Shared by convert_conformer_ctc.py (bakes these as GGUF constant weights) and
reference_forward_conformer.py (uses the identical arrays directly in torch/numpy), so the two can't
silently diverge on window/filterbank construction.

Algorithm confirmed verbatim against NeMo's GitHub source (see BACKLOG.md for the exact quotes):
  1. preemphasis: x[0]'=x[0], x[n]'=x[n]-0.97*x[n-1] for n>=1 (coefficient 0.97, the class default).
  2. STFT (center=True, pad_mode="constant" -- i.e. ordinary ZERO padding, ggml's native CONV_1D padding,
     no reflect-pad primitive needed): frame length n_fft, hop hop_length, an n_fft-length Hann window
     that's zero-padded (centered) from win_length up to n_fft when win_length < n_fft.
  3. power spectrum: mag_power=2.0 and use_grads=False means guard=0, so power = re^2 + im^2 exactly (the
     sqrt-then-square in NeMo's own code cancels algebraically at inference).
  4. mel filterbank: librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm='slaney').
  5. log(power_mel + 2**-24) (log_zero_guard_type='add', log_zero_guard_value=2**-24).
  6. per-feature (per-mel-bin) CMVN over the time axis: unbiased variance (N-1 denominator), epsilon 1e-5
     added to std before dividing (no length masking needed here -- single full utterance, no padding).
"""
import librosa
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SIZE_S = 0.025
WINDOW_STRIDE_S = 0.01
N_FFT = 512
PREEMPH = 0.97
LOG_GUARD = 2.0**-24
NORM_EPS = 1e-5  # NeMo's `CONSTANT`, reused as both the per-feature std epsilon (dither is train-only).


def mel_hparams(n_mels: int) -> dict:
    win_length = round(SAMPLE_RATE * WINDOW_SIZE_S)   # 400
    hop_length = round(SAMPLE_RATE * WINDOW_STRIDE_S)  # 160
    n_freq = N_FFT // 2 + 1                            # 257
    return {
        "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "win_length": win_length, "hop_length": hop_length,
        "n_freq": n_freq, "n_mels": n_mels, "preemph": PREEMPH, "log_guard": LOG_GUARD, "norm_eps": NORM_EPS,
        "stft_pad": N_FFT // 2,  # zero-pad on each side, exactly cancels center=True's n_fft//2 padding.
    }


def hann_window_centered(win_length: int, n_fft: int) -> np.ndarray:
    """torch.hann_window(win_length, periodic=False), zero-padded (centered) to length n_fft -- matches
    torch.stft's own handling of win_length < n_fft exactly."""
    n = np.arange(win_length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (win_length - 1))
    pad_left = (n_fft - win_length) // 2
    full = np.zeros(n_fft, dtype=np.float64)
    full[pad_left : pad_left + win_length] = window
    return full


def build_dft_kernels(n_fft: int, win_length: int) -> tuple:
    """Returns (cos_kernel, sin_kernel), each shaped (n_freq, 1, n_fft) -- PyTorch Conv1d's own
    (out_channels, in_channels, kernel_size) convention, ready for GGUFWriter.add_tensor (gguf-py reverses
    it to ggml's ne=[n_fft, 1, n_freq], exactly what CONV_1D's kernel input expects).

    Real-DFT-via-convolution: cross-correlating a framed+windowed signal against these fixed kernels
    computes the same real/imaginary parts as torch.stft's rfft, since nn.Conv1d/ggml's CONV_1D both
    compute cross-correlation (no kernel flip) -- exactly what a length-n_fft DFT sum already is.
    """
    n_freq = n_fft // 2 + 1
    window = hann_window_centered(win_length, n_fft)
    k = np.arange(n_fft, dtype=np.float64)
    oc = np.arange(n_freq, dtype=np.float64)[:, None]
    angle = 2.0 * np.pi * oc * k[None, :] / n_fft
    cos_kernel = (window[None, :] * np.cos(angle)).astype(np.float32)[:, None, :]
    sin_kernel = (window[None, :] * np.sin(angle)).astype(np.float32)[:, None, :]
    return cos_kernel, sin_kernel


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """(n_mels, n_freq) -- librosa's own shape, which is exactly the numpy layout GGUFWriter needs to
    produce ggml's ne=[n_freq, n_mels] (so MUL_MAT(fb, power_transposed) contracts over n_freq)."""
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sample_rate / 2,
                                norm="slaney").astype(np.float32)
