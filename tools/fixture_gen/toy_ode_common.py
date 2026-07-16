"""Shared hyperparameters, weights, initial noise/conditioning, and JSON graph topology for the
Milestone-3 toy flow-matching vector-field fixture: a small 2-layer CONV_1D network estimating
d(latent)/dt, driven by OdeStepper's forward-Euler integration loop.

Generic scope, same precedent as Milestones 1-2: validates the *ODE-stepping mechanism* (a graph built
once and reused across every integration step, with only "latent"/"timestep" overwritten between
computes) via a minimal network, not a faithful reproduction of any published flow-matching TTS
architecture. The "timestep" input is a plain scalar broadcast across channels, not a learned/sinusoidal
embedding -- see BACKLOG.md.

Imported by both make_toy_ode_gguf.py (writes the .gguf) and reference_forward_ode.py (computes the same
Euler integration in pure numpy), so the two are guaranteed to agree on weights/initial-state (same
seeds, same RNG call order) without the reference ever parsing the GGUF binary format back out.
"""
import json

import numpy as np

N_EMBD = 8   # latent channels
N_FF = 16    # hidden channels inside the vector-field network
KERNEL = 3
PADDING = (KERNEL - 1) // 2  # "same" padding: preserves the frame dimension across each CONV_1D
N_TOKENS = 8  # frames in the latent

SEED = 4040


def hparams() -> dict:
    return {"n_embd": N_EMBD}


def generate_weights() -> dict:
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    return {
        "vf_conv1.weight": rnd(N_FF, N_EMBD, KERNEL),   # (OC,IC,K) -> ne=[K,IC,OC]
        "vf_conv2.weight": rnd(N_EMBD, N_FF, KERNEL),
    }


def generate_initial_latent() -> np.ndarray:
    """(n_embd, n_tokens) -- the starting noise for the ODE integration, in "gguf-ready" layout: gguf-py
    reverses a numpy array's shape into ggml's ne order, so a (C,T)-shaped array here lands as
    ne=[T,C] in the file -- exactly the layout the "latent" topology input declares (T fastest). The
    natural (n_tokens, n_embd) orientation reference_forward_ode.py's own math wants is just this
    array's `.T` (a numpy view, not a copy -- both sides still agree on the underlying byte layout)."""
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(scale=1.0, size=(N_EMBD, N_TOKENS)).astype(np.float32)


def generate_conditioning() -> np.ndarray:
    """(n_embd,) -- a fixed embedding held constant across every integration step."""
    rng = np.random.default_rng(SEED + 2)
    return rng.normal(scale=0.5, size=(N_EMBD,)).astype(np.float32)


def build_topology() -> dict:
    return {
        "version": 1,
        "inputs": [
            {"name": "latent", "dtype": "f32", "shape": ["n_tokens", "n_embd"]},
            {"name": "timestep", "dtype": "f32", "shape": ["1", "n_embd"]},
            {"name": "conditioning", "dtype": "f32", "shape": ["1", "n_embd"]},
        ],
        "output": "velocity",
        "nodes": [
            {"op": "ADD", "inputs": ["latent", "timestep"], "outputs": ["cur"]},
            {"op": "ADD", "inputs": ["cur", "conditioning"], "outputs": ["cur"]},
            {"op": "CONV_1D", "inputs": ["vf_conv1.weight", "cur"], "outputs": ["hidden"],
             "attrs": {"s0": 1, "p0": PADDING, "d0": 1}},
            {"op": "GELU", "inputs": ["hidden"], "outputs": ["hidden"]},
            {"op": "CONV_1D", "inputs": ["vf_conv2.weight", "hidden"], "outputs": ["velocity"],
             "attrs": {"s0": 1, "p0": PADDING, "d0": 1}},
        ],
    }


def topology_json() -> str:
    return json.dumps(build_topology())
