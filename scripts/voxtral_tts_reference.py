#!/usr/bin/env python3
"""Regenerate the Voxtral-4B-TTS oracle for family 9's eighth leaf.

Voxtral-4B-TTS-2603 (Mistral) is an autoregressive TTS over FRAMES of 37 codes at 12.5 Hz:

    prompt  [BOS] [BEGIN_AUDIO] [AUDIO] x n_voice [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO],
            with the n_voice [AUDIO] rows REPLACED by a preset voice's embeddings (`voice_embedding/*.pt`)
    per frame  the LM's last (normed) row -> the flow head: argmax over the semantic codebook, then 36
            acoustic values integrated from one unit draw over 7 Euler steps with CFG (alpha 1.2),
            clamped and rounded to 21 levels. The frame's 37 codes, summed through the codebook
            embeddings, are the next LM row. Semantic code [END_AUDIO] ends the loop.
    codec   the audio tokenizer's causal decoder (ALiBi, sliding-window attention) -> 24 kHz.

**Which code is the reference.** The model runs only under vllm-omni (`vllm serve --omni`), which needs
vllm and a GPU. Its two Voxtral-specific modules -- the flow head (`FlowMatchingAudioTransformer`) and the
codec (`VoxtralTTSAudioTokenizer`) -- are plain torch, so this script imports THOSE FILES as they are
(vllm-omni `--src`, pinned in `meta.json`) with the `vllm.*` names they import stubbed out; no line of
either is re-spelled here. The LM is vllm's `MistralForCausalLM`, which vllm-omni does not carry; it is
written out below in the checkpoint's own (Mistral-native) layout -- interleaved-pair RoPE, no Q/K
permutation -- because that is the layout the weights were trained in, and vllm's permutation to
rotate-half is an exact re-indexing of it. The prompt comes from `mistral_common`'s own
`encode_speech_request`, which is what vllm-omni's serving adapter calls.

Only the flow head's draw is random. This script pins it -- one `[36]` unit normal per frame, the
END_AUDIO frame included, as the reference draws one there too -- and writes, all float32 `.npy`:

* `prompt_ids`  the prompt as `encode_speech_request` returns it (voice rows still [AUDIO] ids)
* `voice`       `[n_voice, 3072]`, the preset voice's embeddings as loaded (bf16 -> f32)
* `noise`       `[n_frames, 36]`
* `hidden`      `[n_frames, 3072]`, the LM row each frame was generated from
* `codes`       `[n_frames, 37]`, every frame INCLUDING the last (END_AUDIO) one, with the reference's
                +2 special-token offset (`output_codes + len(AudioSpecialTokens)`)
* `margin`      `[n_frames, 2]`: the semantic top-1 minus top-2 logit, and the smallest distance of any
                acoustic value from a rounding boundary in level units. A tiny one says the frame's codes
                can flip under rounding noise ([[feedback-pinned-draw-margin]]'s check, for rounding)
* `wave`        `[n_samples]` at 24 kHz: the codec on every frame before END_AUDIO, in ONE call

The codec is decoded in one call, not in the serving path's 375-frame chunks (`decode_helper_batch_async`)
nor its 25-frame streaming windows: the decoder is causal, so one call is the function those chunks
approximate, and the texts gated here stay under 375 frames (30 s), where the two are identical.

    ~/loom-moss/venv/bin/python scripts/voxtral_tts_reference.py --model ~/loom-voxtral/model \\
        --src ~/loom-voxtral/vllm_omni_voxtral_tts --out $LOOM_FIXTURES/voxtral_tts_ref --f64

**`--f64` adds `*_f64.npy`**: the loop re-run at float64 in a fresh process from the same prompt and
draws (4B parameters at 8 bytes is 32 GB, so it needs the workstation). Every frame's codes feed the
next, so this says how far two correct implementations drift apart before a tolerance is chosen.
**`--teacher DIR`** re-runs with a previous run's CODES fed back instead of this run's own (Retro-055's
teacher forcing): `hidden` and `codes` are then per-frame comparable to that run's even where a rounding
flipped.

Requires `mistral_common>=1.10`, `einops`, `regex` and `safetensors`; not vllm.
"""
import argparse
import importlib.util
import json
import logging
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TEXT = "Paris is a beautiful city!"
DEFAULT_VOICE = "casual_male"
# The deploy config's (`vllm_omni/deploy/voxtral_tts.yaml`) stage-0 defaults.
CFG_ALPHA = 1.2
MAX_FRAMES = 2048
# vllm-omni's parser: `n_decoding_steps` is absent from params.json, and it defaults to 7.
N_DECODING_STEPS = 7
MODULE = "vllm_omni.model_executor.models.voxtral_tts"


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


# ------------------------------------------------------------------------------ importing upstream --

def _stub_vllm() -> None:
    """Just enough of `vllm.*` (and of vllm_omni's own package) for the two module files to import.
    Everything here is a name the files mention at import time; none of it runs in the paths used."""

    class _Anything:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, *args, **kwargs):
            pass

    def mod(name, **attrs):
        m = sys.modules.get(name) or types.ModuleType(name)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    def default_weight_loader(param, loaded_weight):
        assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
        param.data.copy_(loaded_weight)

    class _Registry:
        def register_processor(self, *args, **kwargs):
            return lambda cls: cls

    anything = lambda *names: {n: type(n, (_Anything,), {}) for n in names}  # noqa: E731
    mod("vllm")
    mod("vllm.config", **anything("VllmConfig"))
    mod("vllm.inputs", MultiModalDataDict=dict)
    mod("vllm.logger", init_logger=logging.getLogger)
    mod("vllm.model_executor")
    mod("vllm.model_executor.model_loader")
    mod("vllm.model_executor.model_loader.weight_utils", default_weight_loader=default_weight_loader)
    mod("vllm.model_executor.models")
    mod("vllm.model_executor.models.interfaces", **anything("SupportsMultiModal"))
    mod("vllm.model_executor.models.utils", flatten_bn=None, init_vllm_registered_model=None,
        maybe_prefix=lambda p, n: f"{p}.{n}" if p else n)
    mod("vllm.multimodal", MULTIMODAL_REGISTRY=_Registry())
    mod("vllm.multimodal.inputs", **anything("MultiModalFieldConfig", "MultiModalKwargsItems"),
        NestedTensors=object)
    mod("vllm.multimodal.parse", **anything("AudioProcessorItems", "MultiModalDataItems",
                                            "MultiModalDataParser"))
    mod("vllm.multimodal.processing", **anything("BaseDummyInputsBuilder", "BaseMultiModalProcessor"))
    mod("vllm.multimodal.processing.processor", **anything("BaseProcessingInfo", "ProcessorInputs",
                                                           "PromptReplacement", "PromptUpdate"))
    mod("vllm.sequence", **anything("IntermediateTensors"))
    mod("vllm.tokenizers", cached_tokenizer_from_config=None)
    mod("vllm.tokenizers.mistral", **anything("MistralTokenizer"))
    for name in ("vllm_omni", "vllm_omni.model_executor", "vllm_omni.model_executor.models", MODULE,
                 "vllm_omni.quantization"):
        mod(name)
    mod("vllm_omni.quantization.component_config", **anything("ComponentQuantizationConfig"))
    mod("vllm_omni.platforms", current_omni_platform=types.SimpleNamespace(device_type="cpu"))


def import_upstream(src: Path):
    """The two vllm-omni files, imported under their own names so they import each other."""
    _stub_vllm()
    loaded = {}
    for name in ("voxtral_tts_audio_generation", "voxtral_tts_audio_tokenizer"):
        spec = importlib.util.spec_from_file_location(f"{MODULE}.{name}", src / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["voxtral_tts_audio_generation"], loaded["voxtral_tts_audio_tokenizer"]


# ------------------------------------------------------------------------------------------ the LM --

class MistralLM(nn.Module):
    """vllm's `MistralForCausalLM` body in the checkpoint's native layout: pre-norm RMSNorm, GQA
    attention with INTERLEAVED-pair RoPE (Mistral's `apply_rotary_emb` over complex pairs), SwiGLU, a
    final norm. Computes in the parameters' dtype throughout, so the f64 arm is f64 everywhere."""

    def __init__(self, p: dict, weights: dict, dtype):
        super().__init__()
        self.n_layers, self.dim = p["n_layers"], p["dim"]
        self.n_heads, self.n_kv, self.head_dim = p["n_heads"], p["n_kv_heads"], p["head_dim"]
        self.eps, self.theta = p["norm_eps"], p["rope_theta"]
        self.w = {k: v.to(dtype) for k, v in weights.items()
                  if k.startswith(("layers.", "norm.")) or k == "mm_audio_embeddings.tok_embeddings.weight"}
        self.cache = [[None, None] for _ in range(self.n_layers)]
        self.n_past = 0

    def embed(self, ids):
        return self.w["mm_audio_embeddings.tok_embeddings.weight"][ids]

    def _norm(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight

    def _rope(self, x, positions):                                   # x [s, h, d]
        d = self.head_dim
        freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=x.dtype) / d))
        angle = positions.to(x.dtype)[:, None] * freqs[None, :]         # [s, d/2]
        cos, sin = torch.cos(angle)[:, None, :], torch.sin(angle)[:, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)

    def forward(self, x):                                            # x [s, dim] -> [s, dim], normed
        s = x.shape[0]
        positions = torch.arange(self.n_past, self.n_past + s)
        for i in range(self.n_layers):
            L = f"layers.{i}."
            h = self._norm(x, self.w[L + "attention_norm.weight"])
            q = (h @ self.w[L + "attention.wq.weight"].T).view(s, self.n_heads, self.head_dim)
            k = (h @ self.w[L + "attention.wk.weight"].T).view(s, self.n_kv, self.head_dim)
            v = (h @ self.w[L + "attention.wv.weight"].T).view(s, self.n_kv, self.head_dim)
            q, k = self._rope(q, positions), self._rope(k, positions)
            past = self.cache[i]
            if past[0] is not None:
                k, v = torch.cat([past[0], k]), torch.cat([past[1], v])
            past[0], past[1] = k, v
            rep = self.n_heads // self.n_kv
            kk = k.repeat_interleave(rep, dim=1).transpose(0, 1)       # [h, t, d]
            vv = v.repeat_interleave(rep, dim=1).transpose(0, 1)
            scores = (q.transpose(0, 1) @ kk.transpose(1, 2)) / math.sqrt(self.head_dim)
            t = kk.shape[1]
            mask = torch.ones(s, t, dtype=torch.bool).tril(diagonal=t - s)
            scores = scores.masked_fill(~mask, float("-inf"))
            att = (torch.softmax(scores, dim=-1) @ vv).transpose(0, 1).reshape(s, -1)
            x = x + att @ self.w[L + "attention.wo.weight"].T
            h = self._norm(x, self.w[L + "ffn_norm.weight"])
            ff = F.silu(h @ self.w[L + "feed_forward.w1.weight"].T) * (h @ self.w[L + "feed_forward.w3.weight"].T)
            x = x + ff @ self.w[L + "feed_forward.w2.weight"].T
        self.n_past += s
        return self._norm(x, self.w["norm.weight"])


# ------------------------------------------------------------------------------------- the model --

def load(model_dir: Path, src: Path, dtype):
    from safetensors.torch import load_file

    gen_mod, tok_mod = import_upstream(src)
    params = json.loads((model_dir / "params.json").read_text())
    weights = {k: v.to(torch.float32) for k, v in load_file(str(model_dir / "consolidated.safetensors")).items()}

    audio_model_args = dict(params["multimodal"]["audio_model_args"])
    audio_model_args["acoustic_transformer_args"] = dict(audio_model_args["acoustic_transformer_args"])
    if audio_model_args["acoustic_transformer_args"].get("n_decoding_steps") is None:
        audio_model_args["acoustic_transformer_args"]["n_decoding_steps"] = N_DECODING_STEPS
    codec_args = params["multimodal"]["audio_tokenizer_args"]

    flow = gen_mod.FlowMatchingAudioTransformer(dict(audio_model_args))
    for name, w in weights.items():
        if name.startswith("acoustic_transformer."):
            got = flow.load_weight((name[len("acoustic_transformer."):], w))
            assert got == name[len("acoustic_transformer."):], f"{name} is not a flow-head parameter"

    hf_config = types.SimpleNamespace(
        audio_config={"codec_args": codec_args, "audio_model_args": audio_model_args},
        text_config=types.SimpleNamespace(hidden_size=params["dim"]))
    vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=hf_config))
    codec = tok_mod.VoxtralTTSAudioTokenizer(vllm_config=vllm_config)
    n_codec = 0
    for name, w in weights.items():
        if name.startswith("audio_tokenizer."):
            codec.load_weight((name[len("audio_tokenizer."):], w))
            n_codec += 1
        elif name.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
            codec.load_weight(("audio_token_embedding.embeddings.weight", w))
            n_codec += 1
    assert not codec.encoder_loaded, "this checkpoint ships an encoder; the oracle does not cover it"

    flow = flow.to(dtype).eval()
    codec = codec.to(dtype).eval()
    # The semantic codebook is `embedding_sum / cluster_usage`, cached on first use: set at this dtype.
    sc = codec.quantizer.semantic_codebook
    sc.embedding_sum, sc.cluster_usage = sc.embedding_sum.to(dtype), sc.cluster_usage.to(dtype)
    sc._embedding = None
    lm = MistralLM(params, weights, dtype)
    return types.SimpleNamespace(params=params, flow=flow, codec=codec, lm=lm, gen_mod=gen_mod, dtype=dtype)


def encode_prompt(model_dir: Path, text: str, voice: str):
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(str(model_dir / "tekken.json"))
    tokenized = tokenizer.instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice))
    audio_id = tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
    return list(tokenized.tokens), int(audio_id)


def run(m, prompt_ids, audio_id, voice_emb, noise, max_frames, teacher=None):
    """vllm-omni's per-step path: `tts_preprocess` (voice rows into the prompt; the frame's codes summed
    through the codebook embeddings as the next row), the LM, then `compute_mm_logits`."""
    flow, codec, lm, dtype = m.flow, m.codec, m.lm, m.dtype
    ids = torch.tensor(prompt_ids)
    rows = lm.embed(ids).clone()
    slots = ids == audio_id
    assert int(slots.sum()) == voice_emb.shape[0], (int(slots.sum()), voice_emb.shape)
    rows[slots] = voice_emb.to(dtype)
    hidden = lm(rows)[-1:]
    record = {"hidden": [], "codes": [], "margin": []}
    draws = iter(noise)
    original_randn = torch.randn

    def pinned_randn(*shape, dtype=None, device=None, **kw):
        # `decode_one_frame`'s one draw, `torch.randn(B, 36, ...)`: the next pinned row instead.
        return next(draws).to(dtype).view(*shape)

    # Teacher-forced, the loop runs the teacher's frame count whatever this run's own END_AUDIO says.
    n_max = max_frames if teacher is None else teacher.shape[0]
    for frame in range(n_max):
        record["hidden"].append(hidden[0].clone())
        # The semantic draw's own mask (`FlowMatchingAudioTransformer.forward`): [EMPTY_AUDIO] and the
        # padding rows are out, [END_AUDIO] is in.
        logits = flow.semantic_codebook_output(hidden).float()[0]
        logits[0] = -float("inf")
        logits[2 + flow.model_args.semantic_codebook_size:] = -float("inf")
        top = torch.topk(logits, 2).values
        torch.randn = pinned_randn
        try:
            # The margin needs the acoustic values before rounding: recompute `decode_one_frame`'s
            # clamp/scale on the same draw by wrapping `round`.
            seen = {}
            original_round = torch.Tensor.round

            def spy_round(t, *a, **k):
                seen["scaled"] = t.detach().clone()
                return original_round(t, *a, **k)

            torch.Tensor.round = spy_round
            try:
                codes = flow(llm_hidden=hidden, cfg_alpha=torch.full((1,), CFG_ALPHA, dtype=dtype))
            finally:
                torch.Tensor.round = original_round
        finally:
            torch.randn = original_randn
        frac = seen["scaled"] - torch.floor(seen["scaled"])
        record["margin"].append(torch.stack([(top[0] - top[1]).to(torch.float64),
                                             (frac - 0.5).abs().min().to(torch.float64)]))
        record["codes"].append(codes[0].clone())
        if (teacher is None and int(codes[0, 0]) == 1) or frame + 1 == n_max:  # END_AUDIO is id 1
            break
        fed = codes if teacher is None else torch.from_numpy(teacher[frame:frame + 1]).long()
        emb = codec.encode_tokens([fed.T.unsqueeze(0)])[0]            # [1, 3072]
        hidden = lm(emb)
    codes = torch.stack(record["codes"])
    n_audio = codes.shape[0] - (1 if int(codes[-1, 0]) == 1 else 0)
    wave = codec.decode((codes[:n_audio] - 2).T.unsqueeze(0), dtype=dtype).reshape(-1)
    return {"hidden": torch.stack(record["hidden"]), "codes": codes,
            "margin": torch.stack(record["margin"]), "wave": wave}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the mistralai/Voxtral-4B-TTS-2603 checkpoint directory")
    ap.add_argument("--src", required=True, help="vllm-omni's model_executor/models/voxtral_tts directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT, help="default: the model card's own example")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="default: the model card's own example")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--teacher", help="a previous --out whose codes.npy is fed back (teacher forcing)")
    ap.add_argument("--f64", action="store_true", help="also re-run the whole loop at float64")
    ap.add_argument("--f64-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out, model_dir, src = Path(args.out), Path(args.model), Path(args.src)
    out.mkdir(parents=True, exist_ok=True)
    prompt_ids, audio_id = encode_prompt(model_dir, args.text, args.voice)
    voice = torch.load(model_dir / "voice_embedding" / f"{args.voice}.pt", map_location="cpu").float()
    teacher = np.load(Path(args.teacher) / "codes.npy") if args.teacher else None

    if args.f64_only:
        m = load(model_dir, src, torch.float64)
        noise = torch.from_numpy(np.load(out / "noise.npy")).double()
        result = run(m, prompt_ids, audio_id, voice, noise, noise.shape[0], teacher)
        for key, value in result.items():
            save(out, f"{key}_f64", value)
        print(f"f64: {result['codes'].shape[0]} frames")
        return

    m = load(model_dir, src, torch.float32)
    gen = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(args.max_frames, m.flow.model_args.n_acoustic_codebook, generator=gen)
    result = run(m, prompt_ids, audio_id, voice, noise, args.max_frames, teacher)
    n = result["codes"].shape[0]
    save(out, "prompt_ids", prompt_ids)
    save(out, "voice", voice)
    save(out, "noise", noise[:n])
    for key, value in result.items():
        save(out, key, value)
    upstream = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    meta = {"text": args.text, "voice": args.voice, "seed": args.seed, "n_prompt": len(prompt_ids),
            "n_voice": int(voice.shape[0]), "n_frames": n, "ended": bool(int(result["codes"][-1, 0]) == 1),
            "n_samples": int(result["wave"].numel()), "sample_rate": 24000, "cfg_alpha": CFG_ALPHA,
            "n_decoding_steps": N_DECODING_STEPS, "teacher": args.teacher,
            "min_semantic_margin": float(result["margin"][:, 0].min()),
            "min_rounding_margin": float(result["margin"][:, 1].min()),
            "vllm_omni": upstream.stdout.strip() or "8a49c65bf9e477338173339540186577086d1cf4 (copy)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    if args.f64:
        del m
        cmd = [sys.executable, __file__, "--model", args.model, "--src", args.src, "--out", args.out,
               "--text", args.text, "--voice", args.voice, "--f64-only"]
        if args.teacher:
            cmd += ["--teacher", args.teacher]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
