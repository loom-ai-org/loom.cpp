"""Constant (checkpoint-independent) tensors for Kokoro's Generator STFT/ISTFT (istftnet.py's
TorchSTFT, real config: gen_istft_n_fft=20, gen_istft_hop_size=5, window='hann' with torch.hann_window's
own default `periodic=True` -- SAME window convention as Whisper's mel frontend
(the bespoke Whisper converter's whisper_common.py, retired in P4.1), deliberately duplicated
here rather than cross-imported
(this project's usual per-fixture convention) since the two frontends serve genuinely different models.

Two things Whisper's own STFT never needed, both real, both verified numerically against real
torch.stft/torch.istft (max diffs ~1e-6, see BACKLOG.md) before trusting them here:

1. Real STFT PHASE (torch.angle == atan2(imag,real)), not just magnitude -- needs a real ATAN2 primitive
   (added this milestone, no native ggml equivalent existed). At the DC (k=0) and Nyquist (k=n_fft/2,
   n_fft even) bins, a real signal's imaginary part is mathematically exactly zero, but computing it as
   `-(sin_kernel @ frame)` produces IEEE754 NEGATIVE zero (since sin(0)=0.0 exactly, "-(+0.0)" is -0.0),
   while real torch.stft always returns POSITIVE zero at those bins (confirmed across 200 random seeds) --
   `atan2(-0,neg)=-pi` vs `atan2(+0,neg)=+pi`, a spurious ~2*pi discrepancy at those exact bins if left
   uncorrected. Fixed via a `boundary_mask` (1.0 at k=0/Nyquist, 0.0 elsewhere) and the identity
   `x - x*mask` (exactly 0.0 for any finite x when mask=1, exactly x when mask=0 -- unlike `x*(1-mask)`,
   this is robust to x's sign, since IEEE754 subtraction of a value from itself is always positive zero).
   Even with this fix, comparing raw phase values against ANY independently-computed reference remains
   inherently unstable exactly AT the +-pi branch cut (two numerically-independent float32 pipelines can
   land on either side of a genuine discontinuity from ~1e-7-level rounding noise alone) -- callers
   comparing phase values must use a CIRCULAR distance (`((a-b+pi) % 2*pi) - pi`, absolute value), not a
   plain difference, or a handful of elements will show a spurious ~2*pi "error" that isn't one.
2. Real ISTFT (torch.istft) reconstructs via the standard real-IDFT-then-overlap-add-then-window-sum-
   normalize recipe -- expressible as TWO CONV_TRANSPOSE_1D calls (real-part and imag-part contributions,
   each with the window baked into the synthesis kernel) summed together, divided by the overlap-added
   squared-window normalization ("wsum"). Verified this reduction bit-for-bit (max diff ~2e-8) against
   real torch.istft on random (non-self-consistent) magnitude/phase BEFORE trusting it -- Kokoro's
   Generator feeds ISTFT synthetic exp()/sin() values, never a true forward-transform's own output, so
   this must hold for arbitrary magnitude/phase, not just round-tripped ones.

wsum (the overlap-added squared-window normalization) depends only on n_frames (not on any real graph
values), so it's computed HOST-SIDE at call time (a trivial loop, not a graph node) and fed in as a
declared input -- same "host computes a fixed-shape derived value, feeds it in" precedent as
VITS's z_p noise / this project's kq_mask everywhere else, rather than adding a third CONV_TRANSPOSE_1D
call (over a constant all-ones signal) just to keep it in-graph.
"""
import numpy as np

N_FFT = 20
HOP_LENGTH = 5
N_FREQ = N_FFT // 2 + 1  # 11


def periodic_hann(n_fft: int) -> np.ndarray:
    n = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)


def pad_reflect(waveform: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(waveform, (pad, pad), mode="reflect").astype(np.float32)


def build_forward_dft_kernels(n_fft: int) -> tuple:
    """Returns (cos_kernel, neg_sin_kernel, boundary_mask):
    cos_kernel/neg_sin_kernel: (n_freq, 1, n_fft) -- CONV_1D's own (OC,IC,K) convention (negation for the
    imaginary part's sign baked directly into neg_sin_kernel, not a separate SCALE node).
    boundary_mask: (n_freq, 1) numpy -- GGUFWriter reverses axis order to ne=[1,n_freq], broadcasting
    across the n_frames axis (ne[0]=1) against an [n_frames,n_freq] tensor via `im - im*mask`. 1.0 at
    k=0 and k=n_fft/2 (n_fft even), else 0.0.
    """
    n_freq = n_fft // 2 + 1
    window = periodic_hann(n_fft)
    k = np.arange(n_freq, dtype=np.float64)[:, None]
    n = np.arange(n_fft, dtype=np.float64)[None, :]
    angle = 2.0 * np.pi * k * n / n_fft
    cos_kernel = (window[None, :] * np.cos(angle)).astype(np.float32)[:, None, :]
    neg_sin_kernel = (-(window[None, :] * np.sin(angle))).astype(np.float32)[:, None, :]
    boundary_mask = np.zeros((n_freq, 1), dtype=np.float32)
    boundary_mask[0, 0] = 1.0
    if n_fft % 2 == 0:
        boundary_mask[-1, 0] = 1.0
    return cos_kernel, neg_sin_kernel, boundary_mask


def build_inverse_synth_kernels(n_fft: int) -> tuple:
    """Returns (cos_synth_kernel, neg_sin_synth_kernel), each (n_freq, 1, n_fft) -- CONV_TRANSPOSE_1D's
    own (IC,OC,K) native-PyTorch-ConvTranspose1d convention (here OC=1, so numerically the same shape as
    the forward kernel, but semantically ne[1]=OC/ne[2]=IC after GGUFWriter's axis reversal, the OPPOSITE
    of CONV_1D's ne[1]=IC/ne[2]=OC -- confirmed directly against ggml_conv_transpose_1d's real assertion
    `a->ne[2] == b->ne[1]`, i.e. kernel ne[2] must equal the data's channel count).
    Per-bin weight: 2/n_fft for interior bins, 1/n_fft for k=0 and k=n_fft/2 (n_fft even) -- the standard
    real-IDFT-via-Hermitian-symmetry weighting, confirmed numerically against real torch.istft (not
    assumed from theory alone).
    """
    n_freq = n_fft // 2 + 1
    window = periodic_hann(n_fft)
    w = np.full(n_freq, 2.0 / n_fft, dtype=np.float64)
    w[0] = 1.0 / n_fft
    if n_fft % 2 == 0:
        w[-1] = 1.0 / n_fft
    k = np.arange(n_freq, dtype=np.float64)[:, None]
    n = np.arange(n_fft, dtype=np.float64)[None, :]
    angle = 2.0 * np.pi * k * n / n_fft
    cos_synth = (w[:, None] * window[None, :] * np.cos(angle)).astype(np.float32)[:, None, :]
    neg_sin_synth = (-(w[:, None] * window[None, :] * np.sin(angle))).astype(np.float32)[:, None, :]
    return cos_synth, neg_sin_synth


def compute_wsum(n_frames: int, n_fft: int, hop: int) -> np.ndarray:
    """Host-side overlap-added squared-window normalization (real torch.istft's own NOLA denominator),
    full length (n_frames-1)*hop+n_fft (BEFORE center-crop -- callers crop n_fft//2 off each end
    themselves, same as the graph's own output)."""
    window = periodic_hann(n_fft).astype(np.float32)
    out_len = (n_frames - 1) * hop + n_fft
    wsum = np.zeros(out_len, dtype=np.float32)
    for t in range(n_frames):
        wsum[t * hop:t * hop + n_fft] += window ** 2
    return wsum
