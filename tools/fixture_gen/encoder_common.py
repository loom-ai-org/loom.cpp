"""Numpy building blocks shared across the toy-model reference implementations: RMSNorm, exact-erf GELU
(matching ggml_gelu_erf bit-for-bit, not the tanh/sigmoid approximations), non-causal (bidirectional,
unmasked) multi-head self-attention (used by reference_forward_vision.py/reference_forward_asr.py), and a
plain conv1d (used by reference_forward_asr.py and reference_forward_ode.py).
"""
import math

import numpy as np

_erf = np.vectorize(math.erf)


def rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)).astype(np.float32)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def multi_head_self_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_head: int, scale: float) -> np.ndarray:
    """q, k, v: (n_tokens, n_head*head_dim), already projected. No mask -- every token attends to every
    other token (encoders are non-autoregressive / bidirectional), unlike reference_forward.py's causal
    triangle."""
    n_tokens, dim = q.shape
    head_dim = dim // n_head
    q = q.reshape(n_tokens, n_head, head_dim)
    k = k.reshape(n_tokens, n_head, head_dim)
    v = v.reshape(n_tokens, n_head, head_dim)

    out = np.zeros((n_tokens, n_head, head_dim), dtype=np.float32)
    for h in range(n_head):
        scores = (q[:, h, :] @ k[:, h, :].T) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        out[:, h, :] = probs @ v[:, h, :]
    return out.reshape(n_tokens, n_head * head_dim)


def conv1d(data: np.ndarray, kernel: np.ndarray, stride: int, padding: int) -> np.ndarray:
    """data: (N,IC,IL), kernel: (OC,IC,K) -> (N,OC,OL). Small/naive on purpose -- these toy fixtures are
    a handful of frames, clarity matters far more than speed here."""
    n, ic, il = data.shape
    oc, ic2, k = kernel.shape
    assert ic == ic2
    if padding:
        data = np.pad(data, ((0, 0), (0, 0), (padding, padding)))
    ol = (il + 2 * padding - k) // stride + 1

    out = np.zeros((n, oc, ol), dtype=np.float32)
    for ni in range(n):
        for o in range(oc):
            for x in range(ol):
                patch = data[ni, :, x * stride:x * stride + k]
                out[ni, o, x] = np.sum(patch * kernel[o])
    return out
