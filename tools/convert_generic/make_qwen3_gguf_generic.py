#!/usr/bin/env python3
"""Same real weights/hparams/tokenizer as tools/convert_qwen3/convert_qwen3.py, but the JSON graph
topology comes from walking a real torch.export() ATen graph of tools/convert_generic/qwen3_module.py's
Qwen3LLM through aten_to_loom's generic converter -- the exact same converter/op-mapping table the toy LLM
POC used, unmodified. See BACKLOG.md's "generic converter" section for what carried over vs. what didn't.

Usage: python3 make_qwen3_gguf_generic.py <hf_checkpoint_dir> <out.gguf>
Requires: pip install torch gguf numpy safetensors
"""
import json
import sys
from pathlib import Path

import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_qwen3"))
import qwen3_tokenizer
from aten_to_loom import Converter, _qualname_to_gguf_name
from qwen3_module import Qwen3LLM, hparams


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <hf_checkpoint_dir> <out.gguf>", file=sys.stderr)
        sys.exit(1)
    hf_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads((hf_dir / "config.json").read_text())
    tokenizer_json = json.loads((hf_dir / "tokenizer.json").read_text())
    hp = hparams(config)

    model = Qwen3LLM(hf_dir).eval()

    example_n_tokens = 3
    tokens = torch.zeros(example_n_tokens, dtype=torch.long)
    positions = torch.arange(example_n_tokens, dtype=torch.long)
    kq_mask = torch.zeros(example_n_tokens, example_n_tokens)

    ep = torch.export.export(model, (tokens, positions, kq_mask))

    input_specs = {
        "tokens": ("i32", ["n_tokens"]),
        "positions": ("i32", ["n_tokens"]),
        "kq_mask": ("f32", ["n_kv", "n_tokens"]),
    }
    topo = Converter(example_n_tokens=example_n_tokens).convert(ep, input_specs)

    w = GGUFWriter(str(out_path), "loom-qwen3-generic")
    w.add_string("loom.architecture", "qwen3")
    for key in ("n_vocab", "n_embd", "n_layer", "n_head", "n_head_kv", "n_embd_head_k", "n_ff", "rope_dims"):
        w.add_uint32(f"loom.{key}", hp[key])
    # n_embd_head_v isn't a separate config field for Qwen3 (== n_embd_head_k, same as convert_qwen3.py).
    w.add_uint32("loom.n_embd_head_v", hp["n_embd_head_k"])
    w.add_uint32("loom.n_ctx_train", config["max_position_embeddings"])
    for key in ("rope_freq_base", "rope_freq_scale", "rms_norm_eps"):
        w.add_float32(f"loom.{key}", hp[key])
    w.add_string("model.graph_topology", json.dumps(topo))

    qwen3_tokenizer.write_bpe_vocab(
        w, tokenizer_json, vocab_size=hp["n_vocab"],
        bos_token_id=config["bos_token_id"], eos_token_id=config["eos_token_id"],
    )

    for name, param in model.state_dict().items():
        # nn.Parameter names already match the GGUF key convention 1:1 by construction (Qwen3LLM's
        # attributes were named to match) except for the module-path prefixes .weight/layers.->blk.,
        # which the converter's own _qualname_to_gguf_name rule already normalized when it wrote the
        # topology's weight references -- reuse it here so the tensor names written to the GGUF and the
        # names the topology JSON references can never drift apart.
        gguf_name = _qualname_to_gguf_name(name)
        w.add_tensor(gguf_name, param.detach().numpy())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {out_path} ({len(topo['nodes'])} topology nodes, {hp['n_layer']} layers)")


if __name__ == "__main__":
    main()
