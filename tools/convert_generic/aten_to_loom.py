"""Generic torch.export()-ATen-graph -> loom topology-JSON converter (proof of concept).

Walks torch.export()'s ExportedProgram.graph node-by-node and maps each node 1:1 onto a loom
PrimitiveRegistry op via a small, fixed table (OP_MAP below) -- no pattern-matching or subgraph fusion.
Where loom has a fused primitive with no ATen equivalent, the source nn.Module is expected to have called
a custom op (see toy_llm_module.py's loom::rope_neox / loom::attention) so it survives export as a single
opaque node just like any other ATen op.

Deliberately NOT attempted here (see BACKLOG.md's generic-converter section for the full list): dynamic
shapes, op/subgraph fusion, weight-name auto-derivation (still needs one small explicit qualname -> GGUF
-key rule per model family), anything multi-graph.
"""
import re

import torch


def _qualname_to_gguf_name(qualname: str) -> str:
    """ToyLLM-specific rule: 'layers.N.xxx' -> 'blk.N.xxx', and every weight ends in '.weight' whether or
    not the nn.Module attribute itself was a bare nn.Parameter (attn_norm/ffn_norm/token_embd/output_norm)
    or a Linear's .weight. This one small rule is the "still needs an explicit weight-name mapping" half
    of the design documented in BACKLOG.md -- not auto-derived from the graph."""
    name = qualname.replace("layers.", "blk.", 1) if qualname.startswith("layers.") else qualname
    if not name.endswith(".weight"):
        name += ".weight"
    return name


def _layer_index(node: torch.fx.Node):
    stack = node.meta.get("nn_module_stack") or {}
    for qualname in stack:
        m = re.search(r"layers\.(\d+)", qualname)
        if m:
            return int(m.group(1))
    return None


def _arg_ref(a, node_symbol):
    """Resolve one ATen node argument to either a loom symbol name (graph-reference) or a plain literal,
    via node_symbol[node.name] for real torch.fx.Node args."""
    if isinstance(a, torch.fx.Node):
        return node_symbol[a.name]
    return a


class Converter:
    def __init__(self, example_n_tokens: int):
        self.example_n_tokens = example_n_tokens

    def _shape_attr(self, shape_list):
        """RESHAPE's "shape" attr is fed straight into ggml_reshape_* (src/ops/primitives_basic.cpp:147-
        174), i.e. ggml's `ne` convention: fastest-varying dim first -- the REVERSE of ATen's view()/
        reshape() arg, which is plain numpy/PyTorch order (slowest-varying/outermost first). Also
        substitutes the literal that equals the traced example's n_tokens with the symbol "n_tokens" --
        required for GraphBuilder::build(n_tokens, n_past) to rebuild the graph fresh at a different
        length on every generation step (see src/core/graph_builder.cpp:109-112)."""
        reversed_shape = list(reversed(shape_list))
        return [("n_tokens" if d == self.example_n_tokens else d) for d in reversed_shape]

    def convert(self, ep: torch.export.ExportedProgram, input_specs: dict) -> dict:
        node_symbol = {}  # fx node name -> loom topology symbol name
        nodes = []

        params = ep.graph_signature.inputs_to_parameters
        for n in ep.graph.nodes:
            if n.op == "placeholder" and n.name in params:
                node_symbol[n.name] = _qualname_to_gguf_name(params[n.name])
            elif n.op == "placeholder":
                node_symbol[n.name] = n.name  # a declared top-level graph input (tokens/positions/...)

        for n in ep.graph.nodes:
            if n.op != "call_function":
                continue

            target = str(n.target)
            out_symbol = n.name
            node_symbol[n.name] = out_symbol

            if target == "aten.embedding.default":
                weight, indices = (_arg_ref(a, node_symbol) for a in n.args[:2])
                nodes.append({"op": "GET_ROWS", "inputs": [weight, indices], "outputs": [out_symbol]})

            elif target == "aten.rms_norm.default":
                inp, _shape, weight, eps = n.args
                assert weight is None, "this POC always keeps the affine as a separate MUL node"
                nodes.append({"op": "RMS_NORM", "inputs": [_arg_ref(inp, node_symbol)], "outputs": [out_symbol],
                              "attrs": {"eps": float(eps)}})

            elif target == "aten.mul.Tensor":
                a, b = (_arg_ref(x, node_symbol) for x in n.args)
                nodes.append({"op": "MUL", "inputs": [a, b], "outputs": [out_symbol]})

            elif target == "aten.add.Tensor":
                a, b = (_arg_ref(x, node_symbol) for x in n.args)
                nodes.append({"op": "ADD", "inputs": [a, b], "outputs": [out_symbol]})

            elif target == "aten.linear.default":
                inp = _arg_ref(n.args[0], node_symbol)
                weight = _arg_ref(n.args[1], node_symbol)
                # ggml_mul_mat(A, B) convention: weight first (src/ops/primitives_basic.cpp:24-27).
                nodes.append({"op": "MUL_MAT", "inputs": [weight, inp], "outputs": [out_symbol]})
                if len(n.args) > 2 and n.args[2] is not None:
                    raise NotImplementedError("bias not needed/handled by this POC's toy LLM")

            elif target in ("aten.view.default", "aten.reshape.default"):
                inp = _arg_ref(n.args[0], node_symbol)
                shape = self._shape_attr(list(n.args[1]))
                nodes.append({"op": "RESHAPE", "inputs": [inp], "outputs": [out_symbol], "attrs": {"shape": shape}})

            elif target == "loom.rope_neox.default":
                x, positions, n_dims, freq_base, freq_scale = n.args
                nodes.append({"op": "ROPE", "inputs": [_arg_ref(x, node_symbol), _arg_ref(positions, node_symbol)],
                              "outputs": [out_symbol], "attrs": {
                                  "n_dims": int(n_dims), "mode": 2,
                                  # n_ctx_orig only feeds ggml_rope_ext's YaRN extrapolation correction,
                                  # which is a no-op whenever ext_factor == 0.0 (no YaRN here) -- safe to
                                  # hardcode rather than thread a 6th arg through loom::rope_neox.
                                  "n_ctx_orig": 0,
                                  "freq_base": float(freq_base), "freq_scale": float(freq_scale),
                                  "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
                              }})

            elif target == "loom.attention.default":
                q, k, v, mask, scale = n.args
                layer = _layer_index(n)
                assert layer is not None, "couldn't recover layer index from nn_module_stack"
                nodes.append({"op": "ATTENTION", "inputs": [_arg_ref(q, node_symbol), _arg_ref(k, node_symbol),
                              _arg_ref(v, node_symbol), _arg_ref(mask, node_symbol)], "outputs": [out_symbol],
                              "attrs": {"layer": layer, "scale": float(scale), "kv_cache": True}})

            elif target == "aten.silu.default":
                nodes.append({"op": "SILU", "inputs": [_arg_ref(n.args[0], node_symbol)], "outputs": [out_symbol]})

            else:
                raise NotImplementedError(f"no op-mapping registered for ATen target '{target}'")

        output_node = next(n for n in ep.graph.nodes if n.op == "output")
        (final_ref,) = output_node.args[0]
        output_symbol = node_symbol[final_ref.name]

        topo_inputs = []
        for name in ep.graph_signature.user_inputs:
            dtype, shape = input_specs[name]
            topo_inputs.append({"name": name, "dtype": dtype, "shape": shape})

        return {"version": 1, "inputs": topo_inputs, "output": output_symbol, "nodes": nodes}
