"""Shared weights, latent, and JSON graph topology for the Milestone-3 toy VAE-decoder fixture: a small
CONV_TRANSPOSE_1D upsample -> GELU -> CONV_1D (same-padding refine) decoder, a one-shot forward pass (no
control flow -- unlike the ODE-stepper fixture, a VAE decoder needs none).

Generic scope, same precedent as every prior milestone: validates the *upsampling primitive* via a
minimal decoder shape, not a faithful reproduction of any published VAE architecture.

Imported by both make_toy_vae_gguf.py (writes the .gguf) and reference_forward_vae.py (computes the same
forward pass in pure numpy), so the two are guaranteed to agree on weights/latent (same seeds, same RNG
call order) without the reference ever parsing the GGUF binary format back out.
"""
import json

import numpy as np

N_EMBD = 8       # latent and output channels (kept equal for simplicity)
KERNEL_UP = 4    # transpose-conv kernel size
STRIDE_UP = 2    # transpose-conv stride
KERNEL_REFINE = 3
PADDING_REFINE = (KERNEL_REFINE - 1) // 2  # "same" padding

T_LATENT = 4                                       # latent frames
T_OUT = (T_LATENT - 1) * STRIDE_UP + KERNEL_UP      # 10, frames after upsampling

SEED = 5050


def generate_weights() -> dict:
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    return {
        "upsample.weight": rnd(N_EMBD, N_EMBD, KERNEL_UP),        # (IC,OC,K) -> ne=[K,OC,IC]
        "refine.weight": rnd(N_EMBD, N_EMBD, KERNEL_REFINE),      # (OC,IC,K) -> ne=[K,IC,OC]
    }


def generate_latent() -> np.ndarray:
    """(n_embd, t_latent) -- "gguf-ready" layout: gguf-py reverses this (C,T) shape into ggml
    ne=[T,C], the layout CONV_TRANSPOSE_1D's `data` operand (a plain matrix, ne=[IL,IC]) expects."""
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(scale=1.0, size=(N_EMBD, T_LATENT)).astype(np.float32)


def build_topology() -> dict:
    return {
        "version": 1,
        "inputs": [],
        "output": "decoded",
        "nodes": [
            {"op": "CONV_TRANSPOSE_1D", "inputs": ["upsample.weight", "latent.data"], "outputs": ["upsampled"],
             "attrs": {"s0": STRIDE_UP}},
            {"op": "GELU", "inputs": ["upsampled"], "outputs": ["upsampled"]},
            {"op": "CONV_1D", "inputs": ["refine.weight", "upsampled"], "outputs": ["decoded"],
             "attrs": {"s0": 1, "p0": PADDING_REFINE, "d0": 1}},
        ],
    }


def topology_json() -> str:
    return json.dumps(build_topology())
