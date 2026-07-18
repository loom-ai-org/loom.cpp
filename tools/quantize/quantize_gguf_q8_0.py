#!/usr/bin/env python3
"""Quantizes a loom GGUF's matmul-weight tensors to Q8_0, copying every other KV/tensor through
byte-identical.

Which tensors count as "matmul weights" is derived from the model's own embedded topology JSON --
recursively walking every node (expanding "repeat_for" blocks using the GGUF's own "loom.n_layer" KV) and
collecting each MUL_MAT node's *first* input, the weight-first argument per loom's convention (confirmed
in src/ops/primitives_attention.cpp's ATTENTION doc-comment and src/ops/primitives_basic.cpp's op_mul_mat,
a bare ggml_mul_mat(a, b) wrap where ggml_mul_mat computes b @ a.T with `a` as the weight) -- NOT tensor-
name pattern matching. See BACKLOG.md's "quantized weight support" milestone for why: a name-based
deny-list (the approach in a reference C++ quantizer checked against, femelo/qwen3-asr.cpp's
src/quantize.cpp) doesn't generalize across architectures with different naming conventions, and has a
real, confirmed bug where excluded tensors slip through unprotected if stored in a non-F32 dtype on input.
This design is self-adapting instead: on a tied-embeddings model (Qwen3), "token_embd.weight" is *also* a
real MUL_MAT input (the logits projection) and gets quantized automatically; on a model where the
embedding table only ever feeds GET_ROWS (the toy LLM), the exact same code leaves it untouched -- no
special-casing either way.

No C++ engine change is required for this to work at all: GgufModel::load (src/core/gguf_model.cpp) loads
tensors as whatever type the GGUF declares, and ggml_mul_mat/ggml_get_rows have no type assertion beyond
shape compatibility -- ggml's CPU backend already implements the standard quantized-weights/F32-activations
dot product. This script only needs to produce a well-formed Q8_0 GGUF for that existing runtime path to
exercise.

Usage: python3 quantize_gguf_q8_0.py <in.gguf> <out.gguf>
Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quantize

Q8_0_BLOCK_SIZE = 32


def collect_mul_mat_weight_names(nodes: list, n_layer: int) -> set:
    """Recursively walks a topology node list (possibly containing "repeat_for" blocks, expanded here for
    i in range(n_layer) with "{i}" substituted into MUL_MAT's first input) and returns every MUL_MAT node's
    weight-tensor name. Single-level "$n_layer" repeat_for only, matching every topology in this repo
    today (toy LLM, Qwen3) -- multiple simultaneously-nested repeat_for blocks with independent index
    variables would need a fuller substitution pass than this POC needs."""
    names = set()

    def walk(node_list, i):
        for node in node_list:
            if "repeat_for" in node:
                for j in range(n_layer):
                    walk(node["nodes"], j)
            elif node.get("op") == "MUL_MAT":
                name = node["inputs"][0]
                if i is not None:
                    name = name.replace("{i}", str(i))
                names.add(name)

    walk(nodes, None)
    return names


def copy_kv(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copies every real KV pair from the input GGUF to the output writer, verbatim. Skips the reader's
    own pseudo-fields ("GGUF.version" etc., derived from the file header, not real KVs) and
    "general.architecture" (already set by the GGUFWriter constructor's own `arch` argument)."""
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(key, field.contents(), vtype)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.gguf> <out.gguf>", file=sys.stderr)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(in_path))

    arch = reader.get_field("general.architecture").contents()
    n_layer = reader.get_field("loom.n_layer").contents()
    topology = json.loads(reader.get_field("model.graph_topology").contents())
    quantizable = collect_mul_mat_weight_names(topology["nodes"], n_layer)

    writer = GGUFWriter(str(out_path), arch)
    copy_kv(reader, writer)

    n_quantized, n_skipped_unaligned, n_passthrough = 0, 0, 0
    for tensor in reader.tensors:
        if tensor.name in quantizable and tensor.tensor_type == GGMLQuantizationType.F32 and \
                tensor.data.shape[-1] % Q8_0_BLOCK_SIZE == 0:
            # `quantize()` returns a uint8 array shaped in *bytes* (last dim = packed row size in bytes,
            # not element count) -- add_tensor's `raw_shape` (when given) is fed straight into
            # quant_shape_from_byte_shape, so it must be a byte-shape too, not the pre-quantization
            # logical shape. Omitting it lets add_tensor default to `q.shape` itself (already the correct
            # byte-shape), which add_tensor_info then converts back to the logical element-shape
            # internally -- confirmed against a standalone repro before trusting it here.
            q = quantize(np.ascontiguousarray(tensor.data), GGMLQuantizationType.Q8_0)
            writer.add_tensor(tensor.name, q, raw_dtype=GGMLQuantizationType.Q8_0)
            n_quantized += 1
        else:
            if tensor.name in quantizable:
                n_skipped_unaligned += 1
            else:
                n_passthrough += 1
            # Explicit raw_dtype (rather than relying on add_tensor_info's numpy-dtype inference, which
            # only covers plain float/int types) so this also round-trips correctly if the input GGUF ever
            # already contains a quantized tensor, not just this script's own F32 inputs.
            writer.add_tensor(tensor.name, np.ascontiguousarray(tensor.data), raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {out_path}: {n_quantized} tensors -> Q8_0, {n_skipped_unaligned} MUL_MAT weights left "
          f"F32 (not block-aligned to {Q8_0_BLOCK_SIZE}), {n_passthrough} other tensors left F32")


if __name__ == "__main__":
    main()
