#!/usr/bin/env python3
"""Same weights/hparams as tools/fixture_gen/make_toy_llm_gguf.py, but the JSON graph topology comes from
walking a real torch.export() of tools/convert_generic/toy_llm_module.ToyLLM through aten_to_loom's
generic converter, instead of toy_llm_common.build_topology()'s hand-written topology. Proves the
converter reproduces a topology numerically equivalent to the known-good hand-written one (checked by
tests/test_e2e_toy_llm_generic.cpp against the *same* reference fixtures test_e2e_toy_llm.cpp uses).

Requires: pip install torch gguf numpy (torch is why this isn't wired into the default ctest fixture
generation the way every other toy-model GGUF is -- see tests/CMakeLists.txt's comment on this test).
"""
import json
import sys
from pathlib import Path

import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixture_gen"))
import toy_llm_common as common
from aten_to_loom import Converter
from toy_llm_module import ToyLLM


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_llm_generic.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = ToyLLM().eval()

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

    w = GGUFWriter(str(out_path), "loom-toy-llm-generic")
    w.add_string("loom.architecture", "toy_llm")
    hp = common.hparams()
    for key in ("n_vocab", "n_embd", "n_layer", "n_head", "n_head_kv", "n_embd_head_k",
                "n_embd_head_v", "n_ff", "n_ctx_train", "rope_dims"):
        w.add_uint32(f"loom.{key}", hp[key])
    for key in ("rope_freq_base", "rope_freq_scale", "rms_norm_eps"):
        w.add_float32(f"loom.{key}", hp[key])
    w.add_string("model.graph_topology", json.dumps(topo))

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {out_path} ({len(topo['nodes'])} topology nodes)")


if __name__ == "__main__":
    main()
