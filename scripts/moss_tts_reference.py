#!/usr/bin/env python3
"""Regenerate the reference oracle for `tests/gate/test_e2e_moss_tts_composition.cpp`.

The MOSS pair end to end in the reference implementation: text -> MOSS-TTS-Local-Transformer-v1.5 ->
12 codebooks per frame -> MOSS-Audio-Tokenizer-v2 (a 12-codebook PREFIX of its 32) -> 48 kHz stereo.
Two files on the loom side and two models here, chained the same way, so the gate compares the
composition rather than either half alone.

    ~/.venvs/piper/bin/python scripts/moss_tts_reference.py \
        --model ~/Dev/models/moss-tts-local-transformer-v1.5 \
        --codec ~/Dev/models/moss-audio-tokenizer-v2 \
        --out $LOOM_FIXTURES/moss_tts_ref

It writes, all float32 (the one dtype `tests/support/npy_fixture.h` reads):

* `text_ids.npy` and `language.npy`: the caller's side. The text encoded with the checkpoint's own
  tokenizer, plus the language's 1-based position in the contract (English = 3). The prompt template
  around them is the export's (ADR-052), so it is what the gate checks.
* `codes_<N>f.npy`: greedy, N x 12, for each clip length. Both lengths stop short of the frame where
  this sentence ends on its own (38), so each clip is exactly N frames.
* `draws.npy` + `codes_pinned.npy`: the README's sampling (1.7 / top-k 25 / top-p 0.8) with every
  uniform pinned (ADR-047). The reference draws by walking the filtered distribution in
  descending-probability order, which is the walk `loom.sample_row` does with `uniform`.
* `wav_<N>f.npy`: the reference codec's decode of the greedy codes, interleaved `L R L R ...`,
  decoded with `num_quantizers=12`, the prefix decode MOSS-TTS's own processor asks for.

**Voice cloning** (`--clips`, default `samples/jfk.wav` and the codec's own demo clip -- 16 and 24 kHz,
so the resample runs both times):

* `clone_ref_codes_<i>.npy`: clip i through the processor's own `encode_audios_from_path` (channel
  duplication, 48 kHz resample, loudness normalisation, `batch_encode` at F32), all clips in ONE batch
  as a two-reference message encodes them. The script also encodes clip 0 on its own and stops if the
  batch changed its codes, since a one-reference voice file is encoded alone.
* `clone1_codes_<C>f.npy` / `clone2_codes_<C>f.npy`: greedy with clip 0 as the reference, and with
  both. `clone1_codes_pinned.npy`: clip 0, sampled at the README's settings on `draws.npy`.

The gate reads the codes through voice FILES written independently by the exporter's tool --

    python -m loom_exporter.moss_tts_voices <model> -o $OUT --wav samples/jfk.wav --name voice_one ...
    python -m loom_exporter.moss_tts_voices <model> -o $OUT --wav samples/jfk.wav \
        --wav <codec>/demo/demo_gt.wav --name voice_pair ...

-- and checks their inputs equal these arrays before comparing what the driver made of them.

Memory: the models load one after the other (the codec ~8.5 GB while loading, then the LM 18 GB, then
the codec again for its decoder), never two at once.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

TEXT = "The quick brown fox jumps over the lazy dog."
LANGUAGE, LANGUAGE_INDEX = "English", 3
AUDIO_TEMPERATURE, AUDIO_TOP_P, AUDIO_TOP_K = 1.7, 0.8, 25


def pinned_sampler(state):
    def sample(self, logits, do_sample, temperature, top_k, top_p, previous_token_ids=None,
               repetition_penalty=1.0):
        scores = self._apply_repetition_penalty(logits.float(), previous_token_ids, repetition_penalty)
        scores = self._filter_logits(scores / float(temperature), top_k=top_k, top_p=top_p)
        probs = torch.softmax(scores, dim=-1)[0].double()
        order = torch.argsort(probs, descending=True, stable=True)
        kept = probs[order][probs[order] > 0]
        u = float(state["u"][state["i"]])
        state["i"] += 1
        target, running = u * float(kept.sum()), 0.0
        for j, p in enumerate(kept.tolist()):
            running += p
            if running >= target:
                return order[j].view(1)
        return order[len(kept) - 1].view(1)
    return sample


def encode_clips(model_dir, codec_dir, clips):
    """The processor's own encode, batched as a message with every clip as a reference, plus clip 0
    alone -- which must agree, or a one-reference voice file would not be what the batch saw."""
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    codec = transformers.AutoModel.from_pretrained(codec_dir, trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    codec.set_attention_implementation("sdpa")
    codec.set_compute_dtype("fp32")
    codec.decoder = torch.nn.ModuleList()
    Proc = get_class_from_dynamic_module("processing_moss_tts.MossTTSLocalProcessor", model_dir)
    proc = Proc(tokenizer=tok, audio_tokenizer=codec, model_config=config)
    with torch.no_grad():
        batched = proc.encode_audios_from_path([str(c) for c in clips])
        alone = proc.encode_audios_from_path([str(clips[0])])[0]
    if not torch.equal(batched[0], alone):
        raise SystemExit(f"clip 0 encodes differently alone and in a batch "
                         f"({int((batched[0] != alone).sum())} codes differ)")
    del proc, codec
    gc.collect()
    return [c.numpy().astype(np.int64) for c in batched]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--pinned-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--clips", nargs="+", default=None,
                    help="reference clips (default: samples/jfk.wav and <codec>/demo/demo_gt.wav)")
    ap.add_argument("--clone-frames", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    here = Path(__file__).resolve().parent.parent
    clips = args.clips or [here / "samples" / "jfk.wav", Path(args.codec) / "demo" / "demo_gt.wav"]
    ref_codes = encode_clips(args.model, args.codec, clips)
    for i, codes in enumerate(ref_codes):
        np.save(out / f"clone_ref_codes_{i}.npy", codes.astype(np.float32))
        print(f"reference {i} ({clips[i]}): {codes.shape}")

    config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.attn_implementation = config.local_transformer_attn_implementation = "sdpa"
    config.qwen3_config._attn_implementation = "sdpa"
    model = transformers.AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    Proc = get_class_from_dynamic_module("processing_moss_tts.MossTTSLocalProcessor", args.model)
    proc = Proc(tokenizer=tok, audio_tokenizer=None, model_config=config)
    batch = proc([[proc.build_user_message(text=TEXT, language=LANGUAGE)]], mode="generation")
    np.save(out / "text_ids.npy", np.array(tok.encode(TEXT, add_special_tokens=False), np.float32))
    np.save(out / "language.npy", np.array([LANGUAGE_INDEX], np.float32))

    def generate(frames, prompt=None, **kw):
        prompt = batch if prompt is None else prompt
        with torch.no_grad():
            got = model.generate(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"],
                                 max_new_tokens=frames, **kw)
        start, seq = got[0]
        return np.array([r[1:].tolist() for r in seq[start + 1:]
                         if int(r[0]) == config.audio_assistant_slot_token_id], dtype=np.int64)

    greedy = {}
    for frames in args.frames:
        codes = generate(frames, do_sample=False)
        if codes.shape[0] != frames:
            raise SystemExit(f"asked for {frames} frames and the model stopped at {codes.shape[0]}")
        greedy[frames] = codes
        np.save(out / f"codes_{frames}f.npy", codes.astype(np.float32))
        print(f"greedy {frames} frames: {codes.shape}")

    rng = np.random.default_rng(args.seed)
    draws = rng.random((args.pinned_frames + 1) * (config.n_vq + 1)).astype(np.float32)
    state = {"u": draws, "i": 0}
    model._sample_next_token = pinned_sampler(state).__get__(model)
    pinned = generate(args.pinned_frames, do_sample=True, audio_temperature=AUDIO_TEMPERATURE,
                      audio_top_p=AUDIO_TOP_P, audio_top_k=AUDIO_TOP_K, audio_repetition_penalty=1.0)
    np.save(out / "draws.npy", draws)
    np.save(out / "codes_pinned.npy", pinned.astype(np.float32))
    print(f"pinned: {pinned.shape}, {state['i']} draws used")

    # Voice cloning: the references as `[frames, 12]` code tensors, which `_resolve_audio_items` takes
    # as they are -- the codes the voice files hold, not the clips again.
    refs = [torch.from_numpy(c) for c in ref_codes]
    for n_refs in (1, 2):
        prompt = proc([[proc.build_user_message(text=TEXT, reference=refs[:n_refs], language=LANGUAGE)]],
                      mode="generation")
        model.__dict__.pop("_sample_next_token", None)          # greedy: the class's own sampler
        codes = generate(args.clone_frames, prompt, do_sample=False)
        if codes.shape[0] != args.clone_frames:
            raise SystemExit(f"clone with {n_refs} reference(s) stopped at {codes.shape[0]} frames")
        np.save(out / f"clone{n_refs}_codes_{args.clone_frames}f.npy", codes.astype(np.float32))
        print(f"clone, {n_refs} reference(s), greedy {args.clone_frames} frames: prompt "
              f"{prompt['input_ids'].shape[1]} rows")
        if n_refs == 1:
            state = {"u": draws, "i": 0}
            model._sample_next_token = pinned_sampler(state).__get__(model)
            pinned = generate(args.pinned_frames, prompt, do_sample=True,
                              audio_temperature=AUDIO_TEMPERATURE, audio_top_p=AUDIO_TOP_P,
                              audio_top_k=AUDIO_TOP_K, audio_repetition_penalty=1.0)
            np.save(out / "clone1_codes_pinned.npy", pinned.astype(np.float32))
            print(f"clone pinned: {pinned.shape}, {state['i']} draws used")

    del model
    gc.collect()
    codec = transformers.AutoModel.from_pretrained(args.codec, trust_remote_code=True,
                                                   dtype=torch.float32).eval()
    codec.set_attention_implementation("sdpa")
    codec.set_compute_dtype("fp32")     # the checkpoint's bf16 would autocast on CPU
    codec.encoder = torch.nn.ModuleList()
    for frames, codes in greedy.items():
        with torch.no_grad():
            wav = codec.decode(torch.from_numpy(codes.T.copy()).unsqueeze(1), return_dict=True,
                               num_quantizers=codes.shape[1]).audio[0]
        wav = wav.numpy().T.reshape(-1).astype(np.float32)          # interleave L R
        np.save(out / f"wav_{frames}f.npy", wav)
        print(f"wav {frames} frames: {wav.size} floats, peak {np.abs(wav).max():.4f}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
