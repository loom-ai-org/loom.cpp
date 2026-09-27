#!/usr/bin/env python3
"""Regenerate the CosyVoice3 oracle for family 9's seventh leaf.

Fun-CosyVoice3-0.5B-2512 is three models in sequence, and this runs all three in the reference
implementation with every random draw written out:

* **The LM**, a Qwen2-0.5B that turns `[sos, prompt text + text, task_id, prompt speech tokens]` into
  FSQ speech tokens. It SAMPLES, under `ras_sampling`: a nucleus draw (top-k 25, top-p 0.8 measured
  over the WHOLE softmax), and if that id already appears in the last 10 drawn, a second draw from the
  full distribution with that id banned. Greedy is not a usable oracle here -- an argmax decode stops
  after five tokens and whisper hears "you" -- so the sampler is kept and its DRAWS are pinned: both
  of the reference's `multinomial` calls are replaced by an inverse-CDF walk (probabilities sorted in
  descending order, first running sum >= u * mass) over a recorded uniform. That is a sampler of the
  same distribution, and it is the walk `loom.sample_row` makes, so the engine given the same two
  uniforms per step must draw the same ids. Two uniforms are recorded for EVERY step, whether or not the
  repetition-aware second draw fired.
* **The flow**, which in-fills a mel after the voice's prompt frames with a guided 10-step Euler ODE
  over a 22-layer DiT. Its initial noise is not drawn at inference at all: `CausalConditionalCFM` draws
  one `(1, 80, 15000)` tensor under seed 0 at CONSTRUCTION and slices it, so the "draw" recorded here is
  that slice, frame-major.
* **CausalHiFT**, whose sine source is also fed from construction-time tensors: `SineGen2.sine_waves`
  is a `(1, 7200000, 9)` UNIFORM (not Gaussian) tensor, and the causal path's initial phase
  `rand_ini` never reaches the output (it is added to sample 0, and the 1/480 linear downsample reads
  samples 480i+239 and 480i+240 only -- checked: max|d| 0.0 with it zeroed). The slice used is written
  out harmonic-major, `(9, n_samples)`.

    ~/.venvs/piper/bin/python scripts/cosyvoice3_reference.py \
        --model ~/Dev/models/fun-cosyvoice3-0.5b-2512 --out $LOOM_FIXTURES/cosyvoice3_ref --f64

Everything is float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype), plus
`meta.json`. The gate is `tests/gate/test_e2e_cosyvoice3_lua_driver.cpp`.

**`--f64` adds `mel_f64.npy` and `wave_f64.npy`**: the flow and the vocoder re-run at float64 from the
same tokens and the same draws. The sine source's phase is `2*pi*480 * cumsum(f0*h/sr mod 1)` at the
FRAME rate, which reaches ~1e6 rad within seconds, where an f32 ulp is 0.06 rad -- so the f64 arm is
what says how far two correct f32 implementations can sit apart before a tolerance is chosen.

**Two things the reference needs that are not the model's.** It is built WITHOUT hyperpyyaml (whose
1.2.3 does not load under ruamel.yaml 0.19), by constructing `cosyvoice3.yaml`'s three modules with
its own arguments -- `build_reference` below. And `Qwen2Encoder.forward_one_step` hands transformers a
decode-step mask one column wide while the cache holds the whole prefix: transformers 4.51.3 (the
reference's pin) ignored the missing columns, 4.57 reads them as padding, every step then attends to
itself alone, and the LM runs to its token cap (whisper hears "Beep!"). That mask is all ones in every
call the reference makes, so it is passed as None -- `patch_step_mask`.

Requires the FunAudioLLM/CosyVoice checkout (`--src`) and Matcha-TTS (`--matcha`), neither of which is
a dependency of this repo; `modelscope` is stubbed (it is imported for a downloader never called).
"""
import argparse
import json
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np
import torch

PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
N_MEL = 80
SAMPLES_PER_FRAME = 480


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    # C-order explicitly: `astype` keeps a transposed view's Fortran layout and `np.save` then records
    # `fortran_order: True`, which the engine's test reader refuses rather than reading transposed.
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


def import_reference(src: str, matcha: str) -> None:
    if "modelscope" not in sys.modules:
        stub = types.ModuleType("modelscope")
        stub.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no downloads"))
        sys.modules["modelscope"] = stub
    for path in (matcha, src):
        if path not in sys.path:
            sys.path.insert(0, path)


def build_reference(model_dir: str):
    """`cosyvoice3.yaml`'s `llm`, `flow` and `hift`, constructed with the yaml's own arguments and its
    four seeds, then loaded the way `CosyVoice3Model.load` loads them."""
    import random
    from omegaconf import DictConfig
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    random.seed(1986)
    np.random.seed(1986)
    torch.manual_seed(1986)
    llm = CosyVoice3LM(llm_input_size=896, llm_output_size=896, speech_token_size=6561,
                       length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15],
                       llm=Qwen2Encoder(pretrain_path=f"{model_dir}/CosyVoice-BlankEN"),
                       sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1))
    estimator = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
                    spk_dim=80, out_channels=80, static_chunk_size=25 * 2, num_decoding_left_chunks=-1)
    cfm = CausalConditionalCFM(
        in_channels=240, n_spks=1, spk_emb_dim=80, estimator=estimator,
        cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                               "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7,
                               "reg_loss_type": "l1"}))
    flow = CausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192, output_type="mel", vocab_size=6561,
        input_frame_rate=25, only_mask_loss=True, token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3),
        decoder=cfm)
    hift = CausalHiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=24000, nsf_alpha=0.1,
        nsf_sigma=0.003, nsf_voiced_threshold=10, upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7], istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5]] * 3,
        source_resblock_kernel_sizes=[7, 7, 11], source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512))
    load = partial(torch.load, map_location="cpu", weights_only=True)
    llm.load_state_dict(load(f"{model_dir}/llm.pt"), strict=True)
    flow.load_state_dict(load(f"{model_dir}/flow.pt"), strict=True)
    hift.load_state_dict({k.replace("generator.", ""): v for k, v in load(f"{model_dir}/hift.pt").items()},
                         strict=True)
    return llm.eval(), flow.eval(), hift.eval()


def build_frontend(model_dir: str):
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
    from matcha.utils.audio import mel_spectrogram

    feat = partial(mel_spectrogram, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
                   win_size=1920, fmin=0, fmax=None, center=False)
    tok = partial(get_qwen_tokenizer, token_path=f"{model_dir}/CosyVoice-BlankEN",
                  skip_special_tokens=True, version="cosyvoice3")
    return CosyVoiceFrontEnd(tok, feat, f"{model_dir}/campplus.onnx",
                             f"{model_dir}/speech_tokenizer_v3.onnx", f"{model_dir}/spk2info.pt", "all")


def patch_step_mask() -> None:
    from cosyvoice.llm.llm import Qwen2Encoder

    def forward_one_step(self, xs, masks, cache=None):
        outs = self.model(inputs_embeds=xs, attention_mask=None, output_hidden_states=True,
                          return_dict=True, use_cache=True, past_key_values=cache)
        return outs.hidden_states[-1], outs.past_key_values

    Qwen2Encoder.forward_one_step = forward_one_step


def inverse_cdf(probs_desc: torch.Tensor, u: float) -> int:
    """`loom.sample_row`'s walk: f32, sequential, first running sum >= u * mass."""
    p = probs_desc.float().numpy()
    mass = np.float32(0.0)
    for v in p:
        mass = np.float32(mass + v)
    target = np.float32(np.float32(u) * mass)
    running = np.float32(0.0)
    for i, v in enumerate(p):
        running = np.float32(running + v)
        if running >= target:
            return i
    return len(p) - 1


def draw_margin(probs_desc: torch.Tensor, u: float) -> float:
    """|u * mass - nearest cumulative boundary| / mass, in float64."""
    c = np.cumsum(probs_desc.double().numpy())
    return float(np.min(np.abs(c - u * c[-1])) / c[-1])


class PinnedSampler:
    """`nucleus_sampling` and `random_sampling` with their `multinomial` replaced by `inverse_cdf` over
    uniforms drawn here and recorded, two per LM step."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.draws = []
        self.ras_fired = 0
        # How far each USED uniform landed from the nearest CDF boundary, as a fraction of the mass. An
        # f32 engine resolves a draw only when this beats its own probability error: a JFK-voice run at
        # seed 11 had one at 8.0e-07 (step 24's RAS redraw, over all 6761 ids), and loom took the
        # neighbouring id there after 24 identical tokens -- the reference at f64 did not move. The
        # smallest is written to meta.json so a gate's author can see a run is resolvable.
        self.margins = []

    def install(self) -> None:
        from cosyvoice.utils import common
        sampler = self

        def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
            u1, u2 = (float(np.float32(x)) for x in sampler.rng.random(2))
            sampler.draws.append((u1, u2))
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                if cum_prob < top_p and len(prob) < top_k:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            sampler.margins.append(draw_margin(torch.stack(prob), u1))
            return int(indices[inverse_cdf(torch.stack(prob), u1)])

        def random_sampling(weighted_scores, decoded_tokens, sampling):
            sampler.ras_fired += 1
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            sampler.margins.append(draw_margin(sorted_value, sampler.draws[-1][1]))
            return int(sorted_idx[inverse_cdf(sorted_value, sampler.draws[-1][1])])

        common.nucleus_sampling = nucleus_sampling
        common.random_sampling = random_sampling


def run_lm(llm, mi, out):
    """`Qwen2LM.inference` + `inference_wrapper`, unrolled so the prefill and the first logits are
    captured; the sampling call is the reference's own `sampling_ids`."""
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1)
    n_text = mi["text"].shape[1]
    text_emb = llm.llm.model.model.embed_tokens(text)
    sos = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    prompt = llm.speech_embedding(mi["llm_prompt_speech_token"])
    lm_input = torch.concat([sos, text_emb, task, prompt], dim=1)
    save(out, "lm_prefill", lm_input[0])
    min_len, max_len = int(n_text * 2), int(n_text * 20)
    tokens, cache = [], None
    for i in range(max_len):
        y, cache = llm.llm.forward_one_step(lm_input, masks=None, cache=cache)
        logits = llm.llm_decoder(y[:, -1])
        if i == 0:
            save(out, "lm_logits0", logits[0])
        logp = logits.log_softmax(dim=-1)
        top = llm.sampling_ids(logp.squeeze(dim=0), tokens, 25, ignore_eos=i < min_len)
        if top in llm.stop_token_ids:
            break
        tokens.append(top)
        lm_input = llm.speech_embedding.weight[top].reshape(1, 1, -1)
    return tokens, min_len, max_len


def silence_filter(tokens, silent, max_run=5):
    """`CosyVoice3Model.llm_job`: a run of silent/breath tokens longer than 5 is cut to 5 on the way to
    the flow. The LM itself saw (and the RAS window counted) every one of them."""
    kept, run = [], 0
    for t in tokens:
        if t in silent:
            run += 1
            if run > max_run:
                continue
        else:
            run = 0
        kept.append(t)
    return kept


class VocoderRecorder:
    """Wraps CausalHiFT's source so f0, the unit noise slice and the merged source are captured."""

    def __init__(self, hift):
        self.hift = hift
        self.draws = {}
        recorder = self
        gen = hift.m_source.l_sin_gen
        real = type(gen).forward

        def forward(module, f0):
            n = f0.shape[1]
            recorder.draws["nsf_noise"] = module.sine_waves[0, :n].T.clone()   # (9, n)
            return real(module, f0)

        gen.forward = types.MethodType(forward, gen)

    def __call__(self, mel):
        h = self.hift
        # `CausalHiFTGenerator.inference` runs the F0 predictor at float64 ("f0_predictor precision is
        # crucial for causal inference") whatever the rest of the model runs at.
        h.f0_predictor.to(torch.float64)
        f0 = h.f0_predictor(mel.to(torch.float64), finalize=True).to(mel)
        s = h.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = h.m_source(s)
        s = s.transpose(1, 2)
        self.draws["f0"] = f0[0].clone()
        self.draws["source"] = s[0, 0].clone()
        return h.decode(x=mel, s=s, finalize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the Fun-CosyVoice3-0.5B-2512 directory")
    ap.add_argument("--src", default="/home/flavio/Dev/CosyVoice")
    ap.add_argument("--matcha", default="/home/flavio/Dev/Matcha-TTS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--prompt-wav", default=None, help="default: the checkout's asset/zero_shot_prompt.wav")
    ap.add_argument("--prompt-text", default=PROMPT_TEXT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--f64", action="store_true", help="also re-run flow + vocoder at float64")
    args = ap.parse_args()

    import_reference(args.src, args.matcha)
    patch_step_mask()
    sampler = PinnedSampler(args.seed)
    sampler.install()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    prompt_wav = args.prompt_wav or f"{args.src}/asset/zero_shot_prompt.wav"

    llm, flow, hift = build_reference(args.model)
    fe = build_frontend(args.model)
    # `text_frontend=False`: the text is passed as written. With no wetext/ttsfrd the reference's
    # normaliser would only spell out digits (inflect) and split paragraphs, and the default sentence
    # has neither -- stated rather than relied on.
    mi = fe.frontend_zero_shot(args.text, args.prompt_text, prompt_wav, 24000, "")
    save(out, "text_ids", mi["text"][0])
    save(out, "voice_prompt_text", mi["prompt_text"][0])
    save(out, "voice_prompt_speech_tokens", mi["llm_prompt_speech_token"][0])
    save(out, "voice_prompt_feat", mi["prompt_speech_feat"][0])                 # (n_prompt_frames, 80)
    save(out, "voice_embedding", mi["flow_embedding"][0])

    # `CosyVoice3Model.silent_tokens`, set in its __init__ (the FSQ silence and breath codes).
    silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    with torch.inference_mode():
        tokens, min_len, max_len = run_lm(llm, mi, out)
    save(out, "draws", np.asarray(sampler.draws, dtype=np.float32))       # (n_steps, 2)
    save(out, "speech_tokens_raw", tokens)
    flow_tokens = silence_filter(tokens, set(silent_tokens))
    save(out, "speech_tokens", flow_tokens)
    print(f"LM: {len(tokens)} tokens ({len(sampler.draws)} steps, RAS fired {sampler.ras_fired}x, "
          f"min/max {min_len}/{max_len}), {len(flow_tokens)} after the silence filter")

    token = torch.tensor([flow_tokens], dtype=torch.int32)
    n_prompt_frames = int(mi["prompt_speech_feat"].shape[1])
    n_total_frames = 2 * (mi["flow_prompt_speech_token"].shape[1] + len(flow_tokens))
    noise = flow.decoder.rand_noise[0, :, :n_total_frames].clone()          # (80, n_total_frames)
    save(out, "noise", noise.T)
    rec = VocoderRecorder(hift)

    def synthesize(dtype):
        flow_d = flow.to(dtype)
        flow_d.decoder.rand_noise = flow_d.decoder.rand_noise.to(dtype)
        with torch.inference_mode():
            mel, _ = flow_d.inference(
                token=token, token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
                prompt_token=mi["flow_prompt_speech_token"],
                prompt_token_len=torch.tensor([mi["flow_prompt_speech_token"].shape[1]], dtype=torch.int32),
                prompt_feat=mi["prompt_speech_feat"].to(dtype),
                prompt_feat_len=torch.tensor([n_prompt_frames], dtype=torch.int32),
                embedding=mi["flow_embedding"].to(dtype), streaming=False, finalize=True)
            mel = mel.to(dtype)
            hift_d = hift.to(dtype)
            hift_d.stft_window = hift_d.stft_window.to(dtype)
            sg = hift_d.m_source.l_sin_gen
            sg.sine_waves = sg.sine_waves.to(dtype)
            wave = rec(mel)
        return mel, wave

    mel, wave = synthesize(torch.float32)
    save(out, "mel", mel[0])                                                 # (80, n_gen_frames)
    save(out, "wave", wave[0])
    for name in ("nsf_noise", "f0", "source"):
        save(out, name, rec.draws[name])
    print(f"flow: mel {tuple(mel.shape)} after {n_prompt_frames} prompt frames; "
          f"wave {tuple(wave.shape)} peak {float(wave.abs().max()):.4f}")

    if args.f64:
        mel64, wave64 = synthesize(torch.float64)
        save(out, "mel_f64", mel64[0])
        save(out, "wave_f64", wave64[0])
        save(out, "source_f64", rec.draws["source"])
        print(f"f32 vs f64: mel max|d| {(mel64.float() - mel).abs().max().item():.3e}, "
              f"wave max|d| {(wave64.float() - wave).abs().max().item():.3e}, "
              f"rmse {(wave64.float() - wave).pow(2).mean().sqrt().item():.3e}")

    (out / "meta.json").write_text(json.dumps({
        "text": args.text, "prompt_text": args.prompt_text, "prompt_wav": Path(prompt_wav).name,
        "seed": args.seed, "n_steps": len(sampler.draws), "ras_fired": sampler.ras_fired,
        "min_draw_margin": min(sampler.margins, default=1.0),
        "min_len": min_len, "max_len": max_len, "n_speech_tokens": len(flow_tokens),
        "n_prompt_frames": n_prompt_frames, "sample_rate": 24000, "flow_steps": 10,
        "flow_cfg_rate": 0.7, "silent_tokens": silent_tokens,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
