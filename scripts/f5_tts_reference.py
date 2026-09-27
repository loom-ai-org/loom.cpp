#!/usr/bin/env python3
"""Regenerate the F5-TTS oracle for `tests/gate/test_e2e_f5_tts_lua_driver.cpp`.

**This is family 9's third leaf, end to end, in the reference implementation**: a reference clip plus
its transcript plus the text to speak -> one in-filled mel -> Vocos -> waveform. The loom side runs the
same thing through four traced topologies and the engine's own guided ODE, so what the gate compares is
the whole pipeline rather than any one phase.

    ~/.venvs/piper/bin/python scripts/f5_tts_reference.py \
        --model ~/Dev/models/f5-tts/F5TTS_v1_Base \
        --vocoder ~/Dev/models/vocos-mel-24khz \
        --ref-wav ~/Dev/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
        --out $LOOM_FIXTURES/f5_tts_ref

It writes `ref_audio.npy` (the clip, float32, at 24 kHz), `text_ids.npy` (the character ids, written as
floats because that is the one dtype `tests/support/npy_fixture.h` reads), `mel.npy` (the generated
frames) and `wave.npy` (the waveform), plus `meta.json` carrying the numbers the driver must be handed
to reproduce them. Nothing lands in the repo: these are gate fixtures, which live under
`$LOOM_FIXTURES` by the derived rule.

**`--duration` is written into the meta and passed to the driver**, rather than being left to either
side's estimate. The frame count is a RATE applied to character counts, and the reference measures the
transcripts in UTF-8 bytes where the driver measures them in ids -- equal for ASCII and not in general.
Pinning it is what makes the comparison about the model instead of about that arithmetic.

**The NOISE is written out, not the seed.** The initial state is a Gaussian draw, and torch's RNG and
the engine's are different algorithms -- so handing both sides the same *seed* hands them different
*noise*, and flow matching from a different draw is a different valid sample. Measured before this
script wrote `noise.npy`: max |d| 1.25 on a waveform that is intelligible, correctly voiced and simply
not the same realization. `sample()`'s own draw is reproduced here exactly (`manual_seed(seed)` then
`randn(duration, num_channels)`, which is what its per-item loop does for a batch of one) and the gate
hands it to the driver as `noise`.

Requires the F5-TTS checkout on `sys.path` (it is not a dependency of this repo) and the `vocos`
package.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio

SR, HOP, N_MEL, N_FFT = 24000, 256, 100, 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the F5TTS_v1_Base release directory")
    ap.add_argument("--vocoder", required=True, help="a local charactr/vocos-mel-24khz checkout")
    ap.add_argument("--ref-wav", required=True, help="the reference clip (any rate; resampled)")
    ap.add_argument("--ref-text", default="Some call me nature, others call me mother nature.")
    ap.add_argument("--gen-text", default="I don't really care what you call me.")
    ap.add_argument("--f5-src", default="/home/flavio/Dev/F5-TTS/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=int, default=None,
                    help="total frames; default is the reference's own estimate, and whatever it "
                         "resolves to is written into meta.json for the driver to be handed")
    args = ap.parse_args()

    # `f5_tts.model.__init__` imports Trainer, which imports wandb at module scope.
    sys.path.insert(0, args.f5_src)
    stub = types.ModuleType("f5_tts.model.trainer")
    stub.Trainer = object
    sys.modules["f5_tts.model.trainer"] = stub

    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
    from safetensors.torch import load_file
    from vocos import Vocos

    model_dir = Path(args.model)
    vocab_map, vocab_size = get_tokenizer(str(model_dir / "vocab.txt"), "custom")
    ckpt = sorted(model_dir.glob("*.safetensors"))[0]
    model = CFM(
        transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                        text_mask_padding=True, conv_layers=4, qk_norm=None, pe_attn_head=None,
                        attn_backend="torch", attn_mask_enabled=False, checkpoint_activations=False,
                        mel_dim=N_MEL, text_num_embeds=vocab_size),
        mel_spec_kwargs=dict(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, n_mel_channels=N_MEL,
                             target_sample_rate=SR, mel_spec_type="vocos"),
        odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_map)
    state = load_file(str(ckpt))
    ema = {k.replace("ema_model.", ""): v for k, v in state.items() if k.startswith("ema_model.")}
    model.load_state_dict(ema or state, strict=False)
    model.eval()

    audio, sr = torchaudio.load(args.ref_wav)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    rms = torch.sqrt(torch.mean(audio ** 2))
    # `infer_batch_process`'s own normalisation, which the driver reproduces: scale UP only.
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != SR:
        audio = torchaudio.transforms.Resample(sr, SR)(audio)

    # The reference appends a space to the transcript when its last character is one byte, then
    # concatenates. The driver is handed the ids of the joined string and told how many are the
    # transcript's, which is the same split expressed the only way ids can express it.
    ref_text = args.ref_text if args.ref_text.endswith(" ") else args.ref_text + " "
    chars = convert_char_to_pinyin([ref_text + args.gen_text])[0]
    ref_chars = convert_char_to_pinyin([ref_text])[0]
    ids = [vocab_map.get(c, 0) for c in chars]

    ref_frames = audio.shape[-1] // HOP
    duration = args.duration
    if duration is None:
        ref_len = len(ref_text.encode("utf-8"))
        gen_len = len(args.gen_text.encode("utf-8"))
        duration = ref_frames + int(ref_frames / ref_len * gen_len)

    # `sample()`'s own initial draw, reproduced so it can be handed to the engine. Its loop is
    # `for dur in duration: manual_seed(seed); randn(dur, num_channels)` -- one item here.
    torch.manual_seed(args.seed)
    noise = torch.randn(duration, N_MEL, dtype=torch.float32)

    with torch.inference_mode():
        generated, _ = model.sample(cond=audio, text=[chars], duration=duration, steps=args.steps,
                                     cfg_strength=args.cfg, sway_sampling_coef=args.sway,
                                     seed=args.seed)
        mel = generated.to(torch.float32)[:, ref_frames:, :].permute(0, 2, 1)
        vocoder = Vocos.from_hparams(str(Path(args.vocoder) / "config.yaml"))
        vocoder.load_state_dict(torch.load(str(Path(args.vocoder) / "pytorch_model.bin"),
                                            map_location="cpu", weights_only=True))
        wave = vocoder.eval().decode(mel)
    if rms < 0.1:
        wave = wave * rms / 0.1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ref_audio.npy", audio[0].numpy().astype(np.float32))
    np.save(out / "text_ids.npy", np.asarray(ids, dtype=np.float32))
    np.save(out / "noise.npy", noise.numpy().astype(np.float32))
    np.save(out / "mel.npy", mel[0].numpy().astype(np.float32))
    np.save(out / "wave.npy", wave.squeeze().numpy().astype(np.float32))
    json.dump({
        "ref_text": args.ref_text, "gen_text": args.gen_text,
        "n_ref_text": len(ref_chars), "n_text": len(ids),
        "duration": int(duration), "ref_frames": int(ref_frames),
        "n_steps": args.steps, "cfg_scale": args.cfg, "sway_coef": args.sway, "seed": args.seed,
        "sample_rate": SR,
    }, (out / "meta.json").open("w"), indent=2)
    print(f"wrote {out}: {len(ids)} ids, {duration} frames, {wave.numel()} samples")


if __name__ == "__main__":
    main()
