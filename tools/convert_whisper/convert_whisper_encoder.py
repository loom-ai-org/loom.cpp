"""Converts OpenAI Whisper's AudioEncoder into a loom-engine GGUF file: mel frontend (STFT-via-conv +
mel filterbank, see whisper_common.py for the confirmed formula) + conv1/conv2 subsampling + the
self-attention transformer stack + ln_post. Decoder (cross-attention + causal self-attn + tied output
projection) is a separate GGUF/topology -- see convert_whisper_decoder.py -- mirroring VITS's own
multi-GGUF-file precedent (GraphTopology supports exactly one declared output per topology).

Real checkpoint layout confirmed directly against the `tiny.en` checkpoint (torch.load gives a plain
{"dims": {...}, "model_state_dict": {...}} dict -- hand-parsed here directly, no `openai-whisper` package
dependency in THIS script, matching vits_common.load_piper_checkpoint's "no framework dependency"
precedent; the real package IS used, deliberately, in reference_forward_whisper_encoder.py, whose whole
job is being an independent ground truth). encoder.* keys: conv1/conv2 (Conv1d, WITH bias),
positional_embedding (a fixed buffer, not a learned nn.Parameter requiring gradient but still a real
saved tensor -- baked as a GGUF constant either way), blocks.{i}.attn.{query,key,value,out} (key has NO
bias, confirmed in model.py's MultiHeadAttention.__init__), blocks.{i}.{attn_ln,mlp_ln} (LayerNorm),
blocks.{i}.mlp.{0,2} (the two Linears either side of GELU), ln_post.

Whisper's attention has no cross-attention in the encoder (cross_attention=False for AudioEncoder's
ResidualAttentionBlock) and no causal mask (`mask=None`) -- so the encoder is a single
non-autoregressive pass over a FIXED n_audio_ctx=1500 sequence length (always exactly 30s of audio,
padded/trimmed), unlike every previous model in this project (Conformer/Qwen3/VITS all have genuinely
dynamic per-utterance lengths) -- there is no "$n_tokens"-style symbol needed anywhere in this topology,
every shape is a fixed constant. attn uses ATTENTION with kv_cache=false (no persistent cache -- this is
one full-sequence pass, not incremental decoding), plain q/k/v Linear projections (no relative position
concept at all, unlike Conformer/VITS), scale=(head_dim)**-0.5, confirmed mathematically equivalent to
model.py's own (n_state//n_head)**-0.25 applied to BOTH q and k before the matmul (s*s = d**-0.5).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from whisper_common import build_dft_kernels, build_mel_filterbank, mel_hparams


class TopologyBuilder:
    """Same node()-closure/weight-registration idiom as convert_piper_vits/convert_vits.py's
    TopologyBuilder -- duplicated here per this project's "small helper duplicated per tool" convention
    rather than factored into a shared library."""

    def __init__(self):
        self.nodes = []
        self.weights = {}
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t"):
        out = self._fresh(out_hint)
        entry = {"op": op, "inputs": list(inputs), "outputs": [out]}
        if attrs:
            entry["attrs"] = attrs
        self.nodes.append(entry)
        return out

    def weight(self, name, array):
        arr = np.asarray(array)
        if name in self.weights and self.weights[name].shape != arr.shape:
            raise ValueError(f"weight {name!r} already registered with a different shape")
        self.weights[name] = arr
        return name

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


def add_linear(tb, prefix, sd, name, has_bias=True):
    """PyTorch Linear weight is (out, in) -- exactly ggml's MUL_MAT(weight, x) convention already used
    throughout this project (e.g. Qwen3's own Linear layers), no reshaping needed."""
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"])) if has_bias else None
    return w, b


def add_layer_norm(tb, prefix, sd, name):
    g = tb.weight(f"{prefix}.gamma", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.beta", to_f32(sd[f"{name}.bias"]))
    return g, b


def apply_layer_norm(tb, x, prefix, sd, name, eps, out_hint):
    normed = tb.node("LAYER_NORM", [x], {"eps": eps}, f"{out_hint}_normed")
    g, b = add_layer_norm(tb, prefix, sd, name)
    x = tb.node("MUL", [normed, g], None, f"{out_hint}_mul")
    return tb.node("ADD", [x, b], None, out_hint)


def build_encoder(tb, sd, dims):
    """Returns the encoder output name, ne=[n_state, n_audio_ctx] (channel-first, C=ne[0] -- matches
    LAYER_NORM's own C-is-ne[0] convention and every other attention primitive in this project)."""
    n_state = dims["n_audio_state"]
    n_head = dims["n_audio_head"]
    n_layer = dims["n_audio_layer"]
    head_dim = n_state // n_head
    n_mels = dims["n_mels"]
    hp = mel_hparams(n_mels)
    eps = 1e-5  # torch.nn.LayerNorm's own default eps; Whisper's LayerNorm subclass doesn't override it

    # --- mel frontend: STFT-via-conv (cos/sin DFT kernels as CONV_1D weights) + mel filterbank ---
    cos_k, sin_k = build_dft_kernels(hp["n_fft"])
    fb = build_mel_filterbank(hp["sample_rate"], hp["n_fft"], n_mels)
    tb.weight("mel.cos_kernel", cos_k)
    tb.weight("mel.sin_kernel", sin_k)
    tb.weight("mel.filterbank", fb)  # (n_mels, n_freq) -> ggml ne=[n_freq, n_mels]

    stft_attrs = {"s0": hp["hop_length"], "p0": 0, "d0": 1}  # reflect-pad already applied host-side
    # "waveform" (declared input) is the HOST-reflect-padded 30s clip, length n_samples + 2*reflect_pad.
    stft_cos = tb.node("CONV_1D", ["mel.cos_kernel", "waveform"], stft_attrs, "stft_cos")  # [n_frm+1, n_freq, 1]
    stft_sin = tb.node("CONV_1D", ["mel.sin_kernel", "waveform"], stft_attrs, "stft_sin")

    n_frames_full = 1 + (hp["n_samples"] + 2 * hp["reflect_pad"] - hp["n_fft"]) // hp["hop_length"]
    n_frames = n_frames_full - 1  # whisper drops the LAST stft frame (stft[..., :-1])

    def drop_last_frame(x, out_hint):
        v = tb.node("VIEW", [x], {"shape": [n_frames, hp["n_freq"], 1]}, f"{out_hint}_v")
        return tb.node("CONT", [v], None, out_hint)

    stft_cos = drop_last_frame(stft_cos, "stft_cos_trim")
    stft_sin = drop_last_frame(stft_sin, "stft_sin_trim")

    cos_sq = tb.node("SQR", [stft_cos], None, "cos_sq")
    sin_sq = tb.node("SQR", [stft_sin], None, "sin_sq")
    power = tb.node("ADD", [cos_sq, sin_sq], None, "power")  # [n_frames, n_freq, 1]
    power_2d = tb.node("RESHAPE", [power], {"shape": [n_frames, hp["n_freq"]]}, "power_2d")
    power_t_p = tb.node("PERMUTE", [power_2d], {"axes": [1, 0, 2, 3]}, "power_t_p")
    power_t = tb.node("CONT", [power_t_p], None, "power_t")  # [n_freq, n_frames]

    mel_raw = tb.node("MUL_MAT", ["mel.filterbank", power_t], None, "mel_raw")  # [n_mels, n_frames]... see below
    # MUL_MAT(fb[n_freq,n_mels], power_t[n_freq,n_frames]) contracts over n_freq -> ne=[n_mels, n_frames]
    log_spec = tb.node("LOG", [tb.node("CLAMP", [mel_raw], {"min": 1e-10, "max": 3.4e38}, "mel_clamped")],
                        None, "ln_mel")
    # log10(x) = ln(x) / ln(10)
    log_spec = tb.node("SCALE", [log_spec], {"s": 1.0 / float(np.log(10.0))}, "log10_mel")

    # global max over the WHOLE [n_mels, n_frames] spectrogram: flatten to one row, POOL_1D(max) with a
    # kernel spanning the entire length -> a single-element tensor, then max(log_spec, gmax-8) via
    # max(a,c) = relu(a-c)+c (existing primitives only, see primitives_basic.cpp's op history for the
    # "wire onto ggml's own op" principle this follows -- POOL_1D already supports "max" mode natively).
    total = n_mels * n_frames
    flat = tb.node("RESHAPE", [log_spec], {"shape": [total]}, "mel_flat")
    gmax = tb.node("POOL_1D", [flat], {"op": "max", "k0": total, "s0": total, "p0": 0}, "gmax")  # [1]
    floor = tb.node("SCALE", [gmax], {"s": 1.0}, "gmax_copy")
    floor = tb.node("ADD", [floor, tb.weight("mel.neg8", np.array([-8.0], dtype=np.float32))], None, "floor")
    diff = tb.node("SUB", [log_spec, floor], None, "clamp_diff")
    diff = tb.node("RELU", [diff], None, "clamp_diff_relu")
    log_spec = tb.node("ADD", [diff, floor], None, "log_spec_clamped")

    # (x + 4) / 4
    log_spec = tb.node("ADD", [log_spec, tb.weight("mel.plus4", np.array([4.0], dtype=np.float32))], None, "mel_plus4")
    mel = tb.node("SCALE", [log_spec], {"s": 0.25}, "mel_final")  # [n_mels, n_frames]

    # --- conv1 (k=3,p=1,s=1) + GELU, conv2 (k=3,p=1,s=2) + GELU ---
    # CONV_1D's own data convention is [IL, IC, N] (T=ne[0], matching every other CONV_1D consumer in
    # this project -- confirmed against convert_parakeet_tdt.py's own declared "waveform" input shape
    # ["n_tokens","1","1"]). `mel` is channel-first [n_mels, n_frames] (C=ne[0], from the MUL_MAT
    # filterbank contraction above) -- the OPPOSITE convention -- so it needs an explicit transpose
    # (PERMUTE+CONT) before conv1, same "cross the boundary with PERMUTE+CONT" pattern as VITS's own
    # channel-first-attention <-> T-first-CONV_1D boundary crossings.
    mel_tc_p = tb.node("PERMUTE", [mel], {"axes": [1, 0, 2, 3]}, "mel_tc_p")
    mel_tc = tb.node("CONT", [mel_tc_p], None, "mel_tc")  # [n_frames, n_mels]
    conv1_in = tb.node("RESHAPE", [mel_tc], {"shape": [n_frames, n_mels, 1]}, "conv1_in")
    conv1_w = tb.weight("encoder.conv1.weight", to_f32(sd["encoder.conv1.weight"]))  # (n_state,n_mels,3)
    conv1_b = tb.weight("encoder.conv1.bias", to_f32(sd["encoder.conv1.bias"]))
    h = tb.node("CONV_1D", [conv1_w, conv1_in], {"s0": 1, "p0": 1, "d0": 1}, "conv1_out")  # [n_frames, n_state, 1]
    h = tb.node("ADD", [h, tb.node("RESHAPE", [conv1_b], {"shape": [1, n_state, 1]}, "conv1_b_r")], None, "conv1_biased")
    h = tb.node("GELU", [h], None, "conv1_gelu")  # [n_frames, n_state, 1] -- already CONV_1D-ready, no transpose needed

    conv2_w = tb.weight("encoder.conv2.weight", to_f32(sd["encoder.conv2.weight"]))  # (n_state,n_state,3)
    conv2_b = tb.weight("encoder.conv2.bias", to_f32(sd["encoder.conv2.bias"]))
    n_ctx = dims["n_audio_ctx"]
    h = tb.node("CONV_1D", [conv2_w, h], {"s0": 2, "p0": 1, "d0": 1}, "conv2_out")  # [n_ctx, n_state, 1]
    h = tb.node("ADD", [h, tb.node("RESHAPE", [conv2_b], {"shape": [1, n_state, 1]}, "conv2_b_r")], None, "conv2_biased")
    h = tb.node("GELU", [h], None, "conv2_gelu")  # [n_ctx, n_state, 1] == [T, C] (T=ne[0])

    # Cross back over the [T,C]<->[C,T] boundary for the positional-embedding add + attention blocks
    # (this project's own channel-first convention, C=ne[0], matching LAYER_NORM/ATTENTION elsewhere).
    xp = tb.node("PERMUTE", [h], {"axes": [1, 0, 2, 3]}, "x_cf_p")
    x = tb.node("CONT", [xp], None, "x_cf")
    x = tb.node("RESHAPE", [x], {"shape": [n_state, n_ctx]}, "x_cf_2d")  # [n_state, n_ctx] == [C, T]

    # Real PyTorch shape is (n_audio_ctx, n_state) -- row-major, n_state fastest -- so GGUFWriter's own
    # axis reversal already yields ggml ne=[n_state, n_ctx] (channel-first) with NO transpose needed
    # (confirmed against the same reversal rule VITS's emb_rel_k/v tables rely on: numpy (rows, cols) ->
    # ggml ne=[cols, rows]).
    pos_emb = tb.weight("encoder.positional_embedding", to_f32(sd["encoder.positional_embedding"]))
    x = tb.node("ADD", [x, pos_emb], None, "x_pos")

    for i in range(n_layer):
        p = f"encoder.blocks.{i}"
        resid = x
        xn = apply_layer_norm(tb, x, f"{p}.attn_ln", sd, f"{p}.attn_ln", eps, "attn_ln_out")

        qw, qb = add_linear(tb, f"{p}.attn.query", sd, f"{p}.attn.query")
        kw, _ = add_linear(tb, f"{p}.attn.key", sd, f"{p}.attn.key", has_bias=False)
        vw, vb = add_linear(tb, f"{p}.attn.value", sd, f"{p}.attn.value")
        ow, ob = add_linear(tb, f"{p}.attn.out", sd, f"{p}.attn.out")

        q = tb.node("ADD", [tb.node("MUL_MAT", [qw, xn], None, "q"), qb], None, "q_b")
        k = tb.node("MUL_MAT", [kw, xn], None, "k_b")
        v = tb.node("ADD", [tb.node("MUL_MAT", [vw, xn], None, "v"), vb], None, "v_b")
        q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_head, n_ctx]}, "q_r")
        k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_head, n_ctx]}, "k_r")
        v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_head, n_ctx]}, "v_r")

        mask_name = "enc_attn_mask"  # declared input, host-filled with zeros (no causal/padding mask)
        attn = tb.node("ATTENTION", [q, k, v, mask_name],
                       {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))}, "attn_out")
        o = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, "o"), ob], None, "o_b")
        x = tb.node("ADD", [resid, o], None, "res1")

        resid = x
        xn = apply_layer_norm(tb, x, f"{p}.mlp_ln", sd, f"{p}.mlp_ln", eps, "mlp_ln_out")
        w1, b1 = add_linear(tb, f"{p}.mlp.0", sd, f"{p}.mlp.0")
        w2, b2 = add_linear(tb, f"{p}.mlp.2", sd, f"{p}.mlp.2")
        hmlp = tb.node("ADD", [tb.node("MUL_MAT", [w1, xn], None, "mlp1"), b1], None, "mlp1_b")
        hmlp = tb.node("GELU", [hmlp], None, "mlp_gelu")
        hmlp = tb.node("ADD", [tb.node("MUL_MAT", [w2, hmlp], None, "mlp2"), b2], None, "mlp2_b")
        x = tb.node("ADD", [resid, hmlp], None, "res2")

    x = apply_layer_norm(tb, x, "encoder.ln_post", sd, "encoder.ln_post", eps, "ln_post_out")
    return x, n_frames


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    dims = checkpoint["dims"]
    sd = checkpoint["model_state_dict"]

    tb = TopologyBuilder()
    x, n_frames = build_encoder(tb, sd, dims)

    hp = mel_hparams(dims["n_mels"])
    n_samples_padded = hp["n_samples"] + 2 * hp["reflect_pad"]
    inputs = [
        # CONV_1D data convention: [IL, IC, N] -- T=ne[0], channels=ne[1]=1 (mono), batch=ne[2]=1.
        {"name": "waveform", "dtype": "f32", "shape": [str(n_samples_padded), "1", "1"]},
        {"name": "enc_attn_mask", "dtype": "f32", "shape": [str(dims["n_audio_ctx"]), str(dims["n_audio_ctx"])]},
    ]
    topo = tb.topology(inputs, x)

    writer = GGUFWriter(str(out_dir / "whisper_encoder.gguf"), "loom-whisper-encoder")
    writer.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in tb.weights.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_dir / 'whisper_encoder.gguf'}, n_frames={n_frames}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
