"""Ground truth for Whisper streamed as SimulStreaming does it (`test_e2e_whisper_simul_stream.cpp`, ADR-128): ufal's
SimulStreaming (github.com/ufal/SimulStreaming, `PaddedAlignAttWhisper` + `SimulWhisperOnline`), run here, not
re-derived, in its computation-unaware simulation (`--comp_unaware`: the clip in `min_chunk_size` pieces, one
`process_iter` per piece, `finish` at the end).

SimulStreaming loads OpenAI-format `.pt` checkpoints; the zoo's whisper-small is the HF one. The two hold the same
tensors under other names, so the HF state dict is renamed into OpenAI's `Whisper` and SimulStreaming's
`load_model` is pointed at it; the alignment heads are OpenAI's own table for the model (`_ALIGNMENT_HEADS`).

Its defaults, as its CLI sets them: greedy, `frame_threshold` 25, `audio_max_len` 30, `audio_min_len` 0,
`rewind_threshold` 200, no CIF model and `never_fire` off -- which `load_cif` turns into ALWAYS firing, so the
last word of an update is never held back -- language given, no prompt.

Per clip, language and chunk size, in `<out>/<clip>.<lang>.<chunk>/`:
    steps.txt   -- per `process_iter`: `step <i> <samples so far> <is_last> <buffer samples> <buffer offset s>`,
                   then `ctx <ids>` (the context + forced tokens the decode started from),
                   `new <ids>` (the update's new tokens) and `frames <f...>` (the most-attended frame after each
                   decoded token, the last one included)
    text.txt    -- the stream's text, the updates joined

    python scripts/whisper_simulstreaming_reference.py <SimulStreaming dir> <hf whisper dir> <openai name> <out> \\
        --lang en --chunk 1.2[,...] <clip.npy> [...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _openai_state_dict(hf_dir):
    from transformers import WhisperForConditionalGeneration

    hf = WhisperForConditionalGeneration.from_pretrained(hf_dir, dtype=torch.float32).eval()
    sd = hf.model.state_dict()
    out = {}
    names = {"self_attn.q_proj": "attn.query", "self_attn.k_proj": "attn.key", "self_attn.v_proj": "attn.value",
             "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "attn_ln",
             "encoder_attn.q_proj": "cross_attn.query", "encoder_attn.k_proj": "cross_attn.key",
             "encoder_attn.v_proj": "cross_attn.value", "encoder_attn.out_proj": "cross_attn.out",
             "encoder_attn_layer_norm": "cross_attn_ln", "fc1": "mlp.0", "fc2": "mlp.2",
             "final_layer_norm": "mlp_ln"}
    for key, value in sd.items():
        part, rest = key.split(".", 1)
        if rest.startswith("layers."):
            _, i, sub = rest.split(".", 2)
            module, param = sub.rsplit(".", 1)
            out[f"{part}.blocks.{i}.{names[module]}.{param}"] = value
        elif rest == "embed_positions.weight":
            out[f"{part}.positional_embedding"] = value
        elif rest == "embed_tokens.weight":
            out["decoder.token_embedding.weight"] = value
        elif rest.startswith("layer_norm."):
            out[f"{part}.{'ln_post' if part == 'encoder' else 'ln'}.{rest.split('.')[1]}"] = value
        else:  # conv1, conv2
            out[key] = value
    c = hf.config
    dims = dict(n_mels=c.num_mel_bins, n_audio_ctx=c.max_source_positions, n_audio_state=c.d_model,
                n_audio_head=c.encoder_attention_heads, n_audio_layer=c.encoder_layers, n_vocab=c.vocab_size,
                n_text_ctx=c.max_target_positions, n_text_state=c.d_model, n_text_head=c.decoder_attention_heads,
                n_text_layer=c.decoder_layers)
    return dims, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("simul_dir")
    ap.add_argument("hf_dir")
    ap.add_argument("openai_name", help="the OpenAI name whose alignment heads apply, e.g. small")
    ap.add_argument("out")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--chunk", default="1.2")
    ap.add_argument("--frame-threshold", type=int, default=25)
    ap.add_argument("--rewind", type=int, default=200,
                    help="AlignAttConfig.rewind_threshold; written to the case's rewind.txt when not 200")
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.simul_dir)
    from simulstreaming.whisper.simul_whisper import simul_whisper as sw
    from simulstreaming.whisper.simul_whisper.whisper import _ALIGNMENT_HEADS
    from simulstreaming.whisper.simul_whisper.whisper.model import ModelDimensions, Whisper
    import simulstreaming_whisper as entry

    dims, state = _openai_state_dict(a.hf_dir)

    def fresh_model(name=None, download_root=None):
        # A new model per `PaddedAlignAttWhisper`: it registers its attention and cache hooks on the model it
        # loads, so a second one on the same model would run every hook twice.
        model = Whisper(ModelDimensions(**dims))
        model.load_state_dict(state)
        model.set_alignment_heads(_ALIGNMENT_HEADS[a.openai_name])
        return model.eval()

    sw.load_model = fresh_model

    torch.set_grad_enabled(False)
    for chunk_s in (float(c) for c in a.chunk.split(",")):
        asr = entry.SimulWhisperASR(language=a.lang, model_path=f"./{a.openai_name}.pt", cif_ckpt_path=None,
                                    frame_threshold=a.frame_threshold, audio_max_len=30.0, audio_min_len=0.0,
                                    segment_length=chunk_s, beams=1, task="transcribe", decoder_type="greedy",
                                    never_fire=False, init_prompt=None, static_init_prompt=None,
                                    max_context_tokens=None, logdir=None)
        for clip in a.clips:
            audio = np.load(clip).reshape(-1).astype(np.float32)
            # Before the online processor, whose `init` -> `refresh_segment` reads it.
            asr.model.cfg.rewind_threshold = a.rewind
            online = entry.SimulWhisperOnline(asr)
            # Each call's inputs and outputs, read off the model without changing it.
            record = []
            infer = asr.model.infer

            def hooked(is_last=False):
                m = asr.model
                buffer = int(sum(s.shape[0] for s in m.segments))
                tokens, generation = infer(is_last=is_last)
                ctx = generation["starting_tokens"].tokens.tolist() if generation else []
                frames = [p["most_attended_frames"][0] for p in generation.get("progress", [])] if generation else []
                record.append((buffer, list(map(int, ctx)), list(map(int, tokens)), frames, is_last))
                return tokens, generation

            asr.model.infer = hooked
            chunk = int(chunk_s * 16000)
            texts, steps = [], []
            beg = 0
            # `--comp_unaware`: pieces of `min_chunk_size`, the last one short.
            while True:
                end = min(beg + chunk, len(audio))
                online.insert_audio_chunk(audio[beg:end])
                o = online.process_iter()
                steps.append((end, False, online.audio_bufer_offset))
                if o:
                    texts.append(o["text"])
                if end >= len(audio):
                    break
                beg = end
            o = online.finish()
            steps.append((len(audio), True, online.audio_bufer_offset))
            if o:
                texts.append(o["text"])
            asr.model.infer = infer
            d = Path(a.out) / f"{Path(clip).stem}.{a.lang}.{chunk_s:g}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "steps.txt", "w") as f:
                for i, ((buffer, ctx, new, frames, is_last), (pushed, last, offset)) in enumerate(zip(record, steps)):
                    assert is_last == last
                    f.write(f"step {i} {pushed} {int(is_last)} {buffer} {offset:.6f}\n")
                    f.write("ctx " + " ".join(map(str, ctx)) + "\n")
                    f.write("new " + " ".join(map(str, new)) + "\n")
                    f.write("frames " + " ".join(map(str, frames)) + "\n")
            (d / "text.txt").write_text("".join(texts))
            if a.rewind != 200:
                (d / "rewind.txt").write_text(f"{a.rewind}\n")
            print(f"{d.name}: {len(record)} steps\n  {''.join(texts)!r}")


if __name__ == "__main__":
    main()
