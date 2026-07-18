"""Shared helpers for converting a real piper/VITS checkpoint into loom-engine GGUF files."""
import numpy as np
import torch


def load_piper_checkpoint(ckpt_path):
    """Loads a piper training checkpoint (a PyTorch Lightning `.ckpt`, not a bare state dict).

    Real checkpoints store the model under `state_dict`, with every tensor name already
    prefixed `model_g.` (the generator) or `model_d.` (the discriminator, training-only, unused
    for inference). Returns the raw `state_dict` -- callers slice out the `model_g.` prefix
    themselves so tensor-name handling stays visible at the call site.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def fold_weight_norm(weight_g, weight_v):
    """Folds a PyTorch `weight_norm`-reparametrized (`weight_g`, `weight_v`) pair into a plain
    weight tensor: `w = g * v / ||v||`, norm computed per-output-channel (i.e. over every dim
    except dim 0, matching `weight_g`'s own shape -- (C_out, 1, 1, ...) -- which is how every
    weight_norm'd layer in this checkpoint (`WN`'s convs, `ResBlock2`'s convs, `Generator`'s
    `ups`) was constructed, real `dim=0` default confirmed both from `models.py`/`modules.py`'s
    own `weight_norm(...)` calls (no explicit `dim=` override anywhere) and numerically against
    the real checkpoint's own `model_g.dec.ups.0.{weight_g,weight_v}` tensors, matching
    `torch._weight_norm(v, g, 0)` to ~1e-8 (see BACKLOG.md).

    Accepts either torch tensors or numpy arrays; always returns a numpy float32 array.
    """
    g = weight_g.detach().numpy() if torch.is_tensor(weight_g) else np.asarray(weight_g)
    v = weight_v.detach().numpy() if torch.is_tensor(weight_v) else np.asarray(weight_v)
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape([-1] + [1] * (v.ndim - 1))
    return (g * v / norm).astype(np.float32)


def to_f32(tensor):
    """Converts a torch tensor (or anything array-like) to a plain numpy float32 array."""
    arr = tensor.detach().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    return arr.astype(np.float32)


def get_relative_embeddings(table, window_size, length):
    """Real `attentions.py::_get_relative_embeddings` translated to numpy: converts a FIXED
    `(2*window_size+1, k_channels)`-shaped learned relative-position table into the DYNAMIC
    `(2*length-1, k_channels)`-shaped table `REL_POS_ATTENTION_SHAW` actually expects for a
    given sequence length `length` (real phoneme counts routinely exceed `window_size+1`, unlike
    the fixed-size unit test in tests/test_primitive_registry.cpp which sidestepped this).

    Two cases, matching the real code exactly:
    - `length <= window_size + 1`: the learned table is LONGER than needed -- slice out the
      centered `(2*length-1)`-length window.
    - `length > window_size + 1`: the learned table is SHORTER than needed -- zero-pad by
      `length - (window_size + 1)` on each side, then take the centered slice (which after
      padding is the whole table plus the pad, i.e. every real learned position plus zeros).

    `table`: numpy array of shape (2*window_size+1, k_channels) (the `n_heads_rel=1` table with
    its own leading dummy head-dim already squeezed out by the caller). Returns shape
    `(2*length-1, k_channels)`.
    """
    table = np.asarray(table)
    pad_length = max(length - (window_size + 1), 0)
    if pad_length > 0:
        padded = np.pad(table, ((pad_length, pad_length), (0, 0)))
    else:
        padded = table
    total = window_size + 1 - length
    start = max(total, 0)
    end = start + 2 * length - 1
    return padded[start:end]
