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
    of the design documented in BACKLOG.md -- not auto-derived from the graph. Extended for Conformer-CTC
    (first model in this POC with biased Linears): a qualname already ending in '.bias' (or '.weight')
    is left alone -- only bare-Parameter names (no suffix at all) get '.weight' appended."""
    name = qualname.replace("layers.", "blk.", 1) if qualname.startswith("layers.") else qualname
    if not (name.endswith(".weight") or name.endswith(".bias")):
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

    def _permute_axes(self, dims):
        """Translates ATen's aten.permute.default `dims` arg (source-encoding: output torch-axis i comes
        from input torch-axis dims[i], normal torch axis numbering) into ggml_permute's own convention
        (destination-encoding: ne_out[axes[k]] = ne_in[k], REVERSED/ne-order axis numbering) --
        confirmed these are genuinely different conventions by reading ggml_permute's C source directly
        (ggml.c), then derived axes[k] = ndim-1 - dims.index(ndim-1-k) and verified it numerically against
        3 independent permutations (including a non-involution 3-cycle) via a standalone numpy repro
        before trusting it here -- same rigor as the RESHAPE shape-order bug this project already caught
        once. PERMUTE always needs exactly 4 entries (ggml's ne is always 4D); axes beyond len(dims) are
        forced to be identity (axes[k] = k) since the derived sub-permutation only ever produces values in
        0..ndim-1 for those, and a valid 4-permutation must place the leftover indices somewhere."""
        ndim = len(dims)
        axes = [0, 1, 2, 3]
        for k in range(ndim):
            src_torch_axis = ndim - 1 - k
            out_torch_axis = dims.index(src_torch_axis)
            axes[k] = ndim - 1 - out_torch_axis
        return axes

    def _shape_attr(self, shape_list):
        """RESHAPE's "shape" attr is fed straight into ggml_reshape_* (src/ops/primitives_basic.cpp:147-
        174), i.e. ggml's `ne` convention: fastest-varying dim first -- the REVERSE of ATen's view()/
        reshape() arg, which is plain numpy/PyTorch order (slowest-varying/outermost first). Also
        substitutes the literal that equals the traced example's n_tokens with the symbol "n_tokens" --
        required for GraphBuilder::build(n_tokens, n_past) to rebuild the graph fresh at a different
        length on every generation step (see src/core/graph_builder.cpp:109-112).

        Real bug caught here (Conformer-CTC POC, 2026-07-18): a model with no genuine n_tokens dimension
        was converted with example_n_tokens=-1 (meant as an inert "never matches" sentinel) -- but -1 is
        ALSO PyTorch's own reshape()/view() "infer this dimension" sentinel, so `d == self.example_n_tokens`
        fired on every legitimate ATen -1 entry, replacing it with the string "n_tokens" and silently
        defeating RESHAPE's own -1-inference logic (op_reshape's infer-branch never triggered since the
        entry was no longer the literal -1 by the time it got there) -- corrupted the reshape into a
        0-sized target and crashed deep inside ggml_reshape_2d's own nelements assertion, not at any
        obviously-related call site. -1 is excluded here unconditionally: it must never be treated as a
        substitutable literal regardless of what example_n_tokens is set to, since ATen itself reserves it."""
        reversed_shape = list(reversed(shape_list))
        return [("n_tokens" if (d == self.example_n_tokens and d != -1) else d) for d in reversed_shape]

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
                if len(n.args) > 2 and n.args[2] is not None:
                    mm_symbol = out_symbol + "_mm"
                    nodes.append({"op": "MUL_MAT", "inputs": [weight, inp], "outputs": [mm_symbol]})
                    bias = _arg_ref(n.args[2], node_symbol)
                    nodes.append({"op": "ADD", "inputs": [mm_symbol, bias], "outputs": [out_symbol]})
                else:
                    nodes.append({"op": "MUL_MAT", "inputs": [weight, inp], "outputs": [out_symbol]})

            elif target == "aten.view.default":
                # .view() is ATen-guaranteed to never copy (raises instead if the input's strides don't
                # support a pure reinterpretation) -- always already contiguous-compatible, matching
                # ggml_reshape_*'s own hard requirement directly, no CONT needed.
                inp = _arg_ref(n.args[0], node_symbol)
                shape = self._shape_attr(list(n.args[1]))
                nodes.append({"op": "RESHAPE", "inputs": [inp], "outputs": [out_symbol], "attrs": {"shape": shape}})

            elif target == "aten.reshape.default":
                # Unlike .view(), .reshape() is copy-capable -- it silently inserts a copy for
                # non-contiguous input instead of raising (confirmed the hard way: a real
                # permute().reshape() chain hit ggml_reshape_3d's GGML_ASSERT(ggml_is_contiguous(a)) at
                # runtime, since no separate aten.contiguous.default node exists for reshape() to signal
                # this the way it does for an explicit .permute().contiguous() call). Always emit an
                # explicit CONT first to match reshape()'s real (copy-if-needed) semantics -- same
                # unconditional-CONT-after-PERMUTE precedent convert_conformer_ctc.py's own hand-written
                # topology already uses.
                inp = _arg_ref(n.args[0], node_symbol)
                cont_symbol = out_symbol + "_cont"
                nodes.append({"op": "CONT", "inputs": [inp], "outputs": [cont_symbol]})
                shape = self._shape_attr(list(n.args[1]))
                nodes.append({"op": "RESHAPE", "inputs": [cont_symbol], "outputs": [out_symbol], "attrs": {"shape": shape}})

            elif target in ("aten.squeeze.dim", "aten.unsqueeze.default"):
                # No explicit target shape arg (unlike view/reshape) -- read the concrete shape torch.export
                # itself already computed for this node (FakeTensor propagation), reversed into ggml's
                # ne-order same as every other RESHAPE-family conversion. Same CONT-first treatment as
                # aten.reshape.default and for the same real reason: squeeze/unsqueeze only ever touch a
                # size-1 dim so ATen never needs to copy for them, but ggml_reshape_* still requires full
                # contiguity regardless -- hit this for real when unsqueeze followed a PERMUTE output.
                inp = _arg_ref(n.args[0], node_symbol)
                cont_symbol = out_symbol + "_cont"
                nodes.append({"op": "CONT", "inputs": [inp], "outputs": [cont_symbol]})
                shape = list(reversed(list(n.meta["val"].shape)))
                nodes.append({"op": "RESHAPE", "inputs": [cont_symbol], "outputs": [out_symbol], "attrs": {"shape": shape}})

            elif target == "aten.permute.default":
                inp = _arg_ref(n.args[0], node_symbol)
                axes = self._permute_axes(list(n.args[1]))
                nodes.append({"op": "PERMUTE", "inputs": [inp], "outputs": [out_symbol], "attrs": {"axes": axes}})

            elif target == "aten.layer_norm.default":
                # Variable-arity: torch.export omits trailing args that match the ATen schema's own
                # defaults (weight=None, bias=None, eps=1e-05) -- confirmed empirically, not assumed, since
                # this differs from aten.rms_norm.default's default (eps=None), which never gets omitted
                # for this project's own eps values.
                inp = _arg_ref(n.args[0], node_symbol)
                weight = n.args[2] if len(n.args) > 2 else None
                bias = n.args[3] if len(n.args) > 3 else None
                eps = n.args[4] if len(n.args) > 4 else 1e-05
                assert weight is None and bias is None, "this POC always keeps the affine as separate MUL/ADD"
                nodes.append({"op": "LAYER_NORM", "inputs": [inp], "outputs": [out_symbol], "attrs": {"eps": float(eps)}})

            elif target == "aten.glu.default":
                inp = _arg_ref(n.args[0], node_symbol)
                dim = n.args[1] if len(n.args) > 1 else -1
                assert dim == 1, "loom's GLU primitive always splits on ne[1] (channels-first convention)"
                nodes.append({"op": "GLU", "inputs": [inp], "outputs": [out_symbol]})

            elif target == "aten.relu.default":
                nodes.append({"op": "RELU", "inputs": [_arg_ref(n.args[0], node_symbol)], "outputs": [out_symbol]})

            elif target in ("aten.conv1d.default", "aten.conv2d.default"):
                is_2d = target == "aten.conv2d.default"
                inp, weight = _arg_ref(n.args[0], node_symbol), _arg_ref(n.args[1], node_symbol)
                bias = _arg_ref(n.args[2], node_symbol) if len(n.args) > 2 and n.args[2] is not None else None
                stride = list(n.args[3]) if len(n.args) > 3 else ([1, 1] if is_2d else [1])
                padding = list(n.args[4]) if len(n.args) > 4 else ([0, 0] if is_2d else [0])
                dilation = list(n.args[5]) if len(n.args) > 5 else ([1, 1] if is_2d else [1])
                groups = n.args[6] if len(n.args) > 6 else 1

                conv_symbol = out_symbol + "_conv" if bias is not None else out_symbol
                if is_2d:
                    nodes.append({"op": "CONV_2D", "inputs": [weight, inp], "outputs": [conv_symbol], "attrs": {
                        "s0": stride[0], "s1": stride[1], "p0": padding[0], "p1": padding[1],
                        "d0": dilation[0], "d1": dilation[1]}})
                    # CONV_2D output ne=[OW,OH,OC,N] -> bias broadcasts on ne[2] (matches
                    # convert_conformer_ctc.py's broadcast_bias_reshape_2d, same reasoning: op_conv_2d's
                    # final PERMUTE lands channels at ne[2], not ne[1] like CONV_1D).
                    bias_shape = [1, 1, -1, 1]
                else:
                    op = "CONV_1D_DW" if groups != 1 else "CONV_1D"
                    nodes.append({"op": op, "inputs": [weight, inp], "outputs": [conv_symbol], "attrs": {
                        "s0": stride[0], "p0": padding[0], "d0": dilation[0]}})
                    # CONV_1D output ne=[T,OC,N] -> bias broadcasts on ne[1] (matches broadcast_bias_reshape).
                    bias_shape = [1, -1, 1]

                if bias is not None:
                    bias_reshaped = out_symbol + "_bias_r"
                    channels = int(n.meta["val"].shape[1])  # channel axis is always torch-dim 1 (NCHW/NCL)
                    shape = [c if c != -1 else channels for c in bias_shape]
                    nodes.append({"op": "RESHAPE", "inputs": [bias], "outputs": [bias_reshaped], "attrs": {"shape": shape}})
                    nodes.append({"op": "ADD", "inputs": [conv_symbol, bias_reshaped], "outputs": [out_symbol]})

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

            elif target == "loom.rel_pos_attention.default":
                q, k, v, p, pos_bias_u, pos_bias_v, mask, scale = n.args
                nodes.append({"op": "REL_POS_ATTENTION",
                              "inputs": [_arg_ref(x, node_symbol) for x in (q, k, v, p, pos_bias_u, pos_bias_v, mask)],
                              "outputs": [out_symbol], "attrs": {"scale": float(scale)}})

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
