"""Shared loading, hyperparameter extraction, and batchnorm-folding logic for converting NVIDIA NeMo's
Conformer-CTC checkpoints into loom-engine GGUF files. Imported by both convert_conformer_ctc.py (writes
the .gguf) and reference_forward_conformer.py (computes the same forward pass in plain PyTorch), so the
two are guaranteed to use identically-folded weights without either one re-deriving the other's math.

A .nemo file is just a gzipped tar archive (NeMo's own `save_to()` format) containing a plain PyTorch
state dict (`model_weights.ckpt`) and a YAML config (`model_config.yaml`) -- no `nemo_toolkit` import is
needed to read either.
"""
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml


def load_nemo(nemo_path: str):
    """Returns (config: dict, state_dict: dict[str, torch.Tensor], tokenizer_model_bytes: bytes | None).
    tokenizer_model_bytes is the raw SentencePiece `.model` protobuf pointed to by
    config["tokenizer"]["model_path"] (an "nemo:<filename>" reference into this same archive), or None if
    the config has no tokenizer block."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(nemo_path, "r:gz") as tar:
            tar.extractall(tmp)  # noqa: S202 -- trusted local file the user explicitly asked to convert
        tmp_path = Path(tmp)
        config = yaml.safe_load((tmp_path / "model_config.yaml").read_text())
        state_dict = torch.load(tmp_path / "model_weights.ckpt", map_location="cpu", weights_only=True)

        tokenizer_model_bytes = None
        model_path = config.get("tokenizer", {}).get("model_path")
        if model_path:
            filename = model_path.split("nemo:", 1)[-1] if model_path.startswith("nemo:") else model_path
            tokenizer_model_bytes = (tmp_path / filename).read_bytes()
    return config, state_dict, tokenizer_model_bytes


def hparams(config: dict) -> dict:
    enc = config["encoder"]
    dec = config["decoder"]
    n_embd = enc["d_model"]
    n_head = enc["n_heads"]
    return {
        "n_layers": enc["n_layers"],
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": n_embd // n_head,
        "ff_hidden": n_embd * enc["ff_expansion_factor"],
        "conv_kernel_size": enc["conv_kernel_size"],
        "conv_padding": (enc["conv_kernel_size"] - 1) // 2,
        "feat_in": enc["feat_in"],
        "subsampling_conv_channels": enc.get("subsampling_conv_channels", n_embd),
        "num_classes": dec["num_classes"] + 1,  # +1 for the CTC blank token
        "ln_eps": 1e-5,   # nn.LayerNorm's default, confirmed against NeMo's conformer_modules.py source
        "bn_eps": 1e-5,   # nn.BatchNorm1d's default, same
    }


def fold_batchnorm(state_dict: dict, prefix: str, eps: float):
    """Eval-mode BatchNorm1d has no batch-statistics dependency: fold to y = x*scale + shift.
    scale = weight / sqrt(running_var + eps); shift = bias - running_mean * scale.
    Returns (scale, shift) as 1D torch tensors, one value per channel."""
    weight = state_dict[f"{prefix}.weight"]
    bias = state_dict[f"{prefix}.bias"]
    running_mean = state_dict[f"{prefix}.running_mean"]
    running_var = state_dict[f"{prefix}.running_var"]
    scale = weight / torch.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale, shift
