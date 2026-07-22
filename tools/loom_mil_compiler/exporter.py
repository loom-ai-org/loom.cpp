import json
import re
import numpy as np
from coremltools.converters.mil.mil import Block, Function, Operation, Var

from .driver_ir import (
    Argmax, Assign, BinOp, Break, Call, FieldAccess, If, Index, Len, Lit, Local, LocalDecl,
    LuaCodegen, RawBlock, RawExpr, Return, SubgraphCall, UnaryOp, Var as IRVar, While, check_subgraph_calls,
    validate,
)
from .driver_ir import Function as IRFunction

# Traced-model input names auto-computed by the driver (via loom.range) rather than unpacked from the
# caller's `inputs` table -- see apply_monolithic_export/apply_atomic_export's own comment.
_POSITION_INPUT_NAMES = {"cache_position", "position_ids"}

# Traced-model input names auto-computed by the driver via loom.causal_mask (an already-prepared 4D
# additive mask, the same "pass it explicitly so the traced model skips computing it internally" fix as
# _POSITION_INPUT_NAMES -- see export_lfm2_*.py's own _causal_mask() comment).
_CAUSAL_MASK_INPUT_NAMES = {"attention_mask"}

# CoreML's own naming convention for a symbolic (dynamic) shape dimension -- always "is" followed by
# digits (e.g. "is0", "is936"). Matched with \b so it only substitutes whole symbol tokens, never a
# coincidental "is" substring inside a longer identifier.
_DYNAMIC_SYMBOL_RE = re.compile(r"\bis\d+\b")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class LoomGGUFExporter:
    # A mapping from standard/custom MIL op_types to Loom's C++ register_op primitives.
    OP_MAP = {
        "add": "ADD",
        "sub": "SUB",
        "mul": "MUL",
        "div": "DIV",
        "real_div": "DIV",
        "floor_div": "DIV",
        "matmul": "MUL_MAT",
        "relu": "RELU",
        "gelu": "GELU",
        "silu": "SILU",
        "softmax": "SOFTMAX",
        "reshape": "RESHAPE",
        "transpose": "PERMUTE",
        "concat": "CONCAT",
        "gather": "GET_ROWS",
        "reduce_mean": "MEAN",
        "layer_norm": "LAYER_NORM",
        "rms_norm": "RMS_NORM",
        "sigmoid": "SIGMOID",
        "tanh": "TANH",
        "exp": "EXP",
        "sin": "SIN",
        "cos": "COS",
        "floor": "FLOOR",
        "clamp": "CLAMP",
        "pow": "POW",
        "rsqrt": "RSQRT",
        "shape": "SHAPE",
        "range_1d": "RANGE_1D",
        "expand_dims": "RESHAPE",
        "squeeze": "RESHAPE",
        "less_equal": "LESS_EQUAL",
        "greater_equal": "GREATER_EQUAL",
        "less": "LESS",
        "greater": "GREATER",
        "equal": "EQUAL",
        "not_equal": "NOT_EQUAL",
        "select": "SELECT",
        "abs": "ABS",
        "neg": "NEG",
        "sign": "SIGN",
        "minimum": "MINIMUM",
        "maximum": "MAXIMUM",
        "reduce_sum": "REDUCE_SUM",
        "identity": "IDENTITY",
        # Specialized Loom dialect ops:
        "loom_fused_attention": "ATTENTION",
        "loom_spline": "RQ_SPLINE_INVERSE",
        "loom_rope": "ROPE",
    }

    def __init__(self, program, **kwargs):
        import os
        self.program = program
        self.kwargs = kwargs
        self.weights = {}
        self.topologies = {}
        self.ir_function = None
        self.profile = kwargs.get("profile") or os.environ.get("LOOM_PROFILE", None)
        self.output_path = kwargs.get("output_path") or os.environ.get("LOOM_OUTPUT_PATH", "model.gguf")
        self.quantize = kwargs.get("quantize") or os.environ.get("LOOM_QUANTIZE", None)

    def safe_name(self, name: str) -> str:
        """
        Sanitizes MIL SSA variable/op names to be safe for Lua identifiers
        by replacing characters like %, ., / and prepending _ if starts with a digit.
        """
        for c in "%./-+$#@!&*()[]{}|<>?;:":
            name = name.replace(c, "_")
        if name and name[0].isdigit():
            name = "_" + name
        return name

    def get_var_info(self, var):
        """
        Extracts dtype ('f32' or 'i32') and shape in fastest-varying (ne-order) reversed list.
        """
        name = self.safe_name(var.name)
        dtype_str = str(var.dtype)
        if "fp" in dtype_str or "float" in dtype_str or "double" in dtype_str:
            dtype = "f32"
        elif "int" in dtype_str:
            dtype = "i32"
        else:
            dtype = "f32"

        shape = []
        if hasattr(var, "shape") and var.shape is not None:
            for dim in var.shape:
                # All shape entries in Loom topologies must be serialized as string expressions for C++
                # parsing. Every symbolic dim (CoreML stringifies these as "isN") collapses to the single
                # "n_tokens" symbol GraphBuilder/SymbolEnv resolve at build time -- the engine's
                # dynamic-shape support is genuinely single-axis (EXPORT-BACKLOG.md item 3), so this
                # exporter only ever targets models with exactly one true dynamic quantity (sequence
                # length), matching every model on the current roadmap (batch/hidden/heads are always
                # architecturally static). A topology routinely contains SEVERAL distinct "isN" names for
                # that one same quantity, not just one: CoreML's shape algebra mints a fresh opaque symbol
                # at any derivation step it can't simplify back to the original input symbol (confirmed
                # empirically -- an LFM2 ShortConv layer's causal pad+conv+slice, which provably preserves
                # sequence length, produces 4 distinct "isN" names downstream of its one "is0" input; the
                # atomic-export path's own inter-slice boundaries surface even more of these as separate
                # declared inputs of a single slice). There is no cheap, reliable way to tell "several
                # names, one true quantity" apart from "two genuinely independent dynamic axes" from the
                # dim strings alone -- CoreML doesn't expose symbol-equality at this level, and an
                # input-count-based heuristic tried here produced false positives on real, correct atomic
                # slices. If a future model genuinely needs a second independent dynamic axis, that would
                # surface as a numerical mismatch against the reference model, not a syntactic error here.
                #
                # Substituting each symbol OCCURRENCE (not the whole dim string) matters: a dim isn't
                # always bare "is936" -- coremltools' own shape inference can report a full ARITHMETIC
                # EXPRESSION over several such symbols, e.g. a reshape's "-1"-inferred axis surfacing as
                # "is936*is937*is938*is939*is940/1024" (confirmed on LFM2's GQA key/value reshape, which
                # crashed ggml_reshape_4d's element-count assertion once this was collapsed to the bare
                # string "n_tokens", discarding the "/1024" entirely). SymbolEnv's own expression evaluator
                # (src/core/symbol_env.cpp) supports `+ - * / ()`, so substituting every symbol occurrence
                # with "n_tokens" and keeping the surrounding arithmetic intact lets GraphBuilder evaluate
                # the whole expression correctly at build time instead.
                dim_str = str(dim)
                if _DYNAMIC_SYMBOL_RE.search(dim_str):
                    shape.append(_DYNAMIC_SYMBOL_RE.sub("n_tokens", dim_str))
                else:
                    shape.append(dim_str)
        
        # Loom expects fast-varying dimension first, so reverse standard shape order
        reversed_shape = list(reversed(shape))
        return {"name": name, "dtype": dtype, "shape": reversed_shape}

    def export(self):
        """
        Traverses the MIL program:
          - 'main' function becomes the embedded Lua driver script.
          - Other functions represent heavy submodules and become static topologies.
          - Weights and assets are serialized to GGUF.
        """
        is_bespoke = len(self.program.functions) > 1 and "main" in self.program.functions

        if is_bespoke and self.profile is None:
            # 1. Advanced / Bespoke Exporting Workflow
            print("Exporting via Advanced/Bespoke workflow...")
            for func_name, func in self.program.functions.items():
                if func_name == "main":
                    self.transpile_to_lua(func, name="main")
                else:
                    self.topologies[func_name] = self.generate_graph_topology(func, func_name)
            driver_script = self._finalize_driver()
        else:
            profile = self.profile or "monolithic"
            if profile == "atomic":
                try:
                    # Validation/codegen must run INSIDE this try block, not after it: atomic
                    # partitioning is a best-effort heuristic (scope-boundary guessing), and an IR that
                    # fails validation (e.g. a spurious/undefined subgraph input the heuristic
                    # mis-attributed to the wrong slice) is exactly the same class of "atomic partitioning
                    # didn't actually work" failure as an exception raised during partitioning itself --
                    # both should fall back to the monolithic profile rather than crashing the export.
                    self.apply_atomic_export()
                    driver_script = self._finalize_driver()
                except Exception as e:
                    print(f"Warning: Automated atomic partitioning failed: {e}. Falling back to monolithic profile.")
                    self.topologies = {}
                    self.ir_function = None
                    self.apply_monolithic_export()
                    driver_script = self._finalize_driver()
            else:
                self.apply_monolithic_export()
                driver_script = self._finalize_driver()

        # 3. Serialization Phase
        self.write_gguf(driver_script)
        return self.output_path

    def _finalize_driver(self) -> str:
        """Validates the built driver IR and codegens it to Lua source text."""
        validate(self.ir_function)
        check_subgraph_calls(self.ir_function, self.topologies)
        return "\n".join(LuaCodegen().emit_function(self.ir_function))

    def apply_monolithic_export(self):
        print("Exporting via Automatic Monolithic path...")
        main_func = self.program.functions["main"]
        self.topologies["main_topo"] = self.generate_graph_topology(main_func, "main_topo")

        first_input = "tokens"
        feature_scale = 1
        if main_func.inputs:
            first_input_var = list(main_func.inputs.values())[0]
            first_input = self.safe_name(list(main_func.inputs.keys())[0])
            if hasattr(first_input_var, "shape") and len(first_input_var.shape) == 3:
                # For 3D shapes [batch, seq, feature], scale tokens by the last dimension (feature size)
                try:
                    feature_scale = int(first_input_var.shape[2])
                except (ValueError, TypeError):
                    pass

        n_tokens_expr = Len(first_input)
        if feature_scale > 1:
            n_tokens_expr = BinOp("floordiv", Len(first_input), Lit(feature_scale))

        body = []
        for name in main_func.inputs.keys():
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                # A traced model's own "cache_position"/"position_ids" input (see export_lfm2_*.py's own
                # comment: passing this explicitly rather than letting the model derive it internally
                # from a Python-level `.shape[1]` query is what keeps it genuinely dynamic under
                # torch.jit.trace) is host-computed here, not unpacked from the caller's `inputs` table --
                # the driver already knows n_tokens/n_past, and callers shouldn't need to know this is an
                # LFM2-specific implementation detail of the traced graph.
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            elif name in _CAUSAL_MASK_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))
            else:
                body.append(Local(safe_inp, BinOp("or", FieldAccess("inputs", safe_inp), FieldAccess("inputs", "tokens"))))

        inputs_tbl = {self.safe_name(k): IRVar(self.safe_name(k)) for k in main_func.inputs.keys()}

        body.append(SubgraphCall(
            outputs=["_mono_out"],
            extra_outputs=["_mono_shape"],
            module="main_topo",
            n_tokens=n_tokens_expr,
            n_past=Lit(0),
            inputs=inputs_tbl,
        ))

        # Argmax the logits row for the active (last real) token rather than returning the raw output
        # array -- _mono_shape[1] is the output's ne0 (vocab size), the same convention
        # transpile_operation's own "argmax" case and apply_atomic_export's final slice rely on.
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar("_mono_out")]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar("_mono_out"), Index(IRVar("_mono_shape"), 1), row_expr])])],
            else_=[Return([IRVar("_mono_out")])],
        ))

        self.ir_function = IRFunction("main", ["inputs"], body)

    def apply_atomic_export(self):
        print("Exporting via Automatic Atomic path...")
        import re
        main_func = self.program.functions["main"]
        operations = list(main_func.operations)
        
        # 1. Scope-Based Partitioning.
        #
        # Preferred signal: coremltools attaches the ORIGINAL PyTorch module hierarchy to every op
        # converted from a traced/exported model, under ScopeSource.TORCHSCRIPT_MODULE_NAME (e.g.
        # ('model', 'model', '5', 'self_attn') for `model.model.layers[5].self_attn`). A digit segment
        # that is NOT the tuple's last element marks indexing into a repeated submodule
        # (nn.ModuleList/Sequential) -- exactly the atomic layer boundary we want. Trailing digits are
        # NOT a boundary signal: they are just auto-generated per-op SSA name suffixes coremltools
        # invents when the original PyTorch value had no better name (e.g. a bare intermediate named
        # "1823"), and treating those as boundaries mis-partitions the graph into one slice per op
        # instead of one slice per real decoder layer.
        #
        # Fallback signal: hand-built MIL programs (e.g. this module's own unit tests) have no torch
        # scope metadata at all, so we fall back to the previous heuristic of regex-matching the op's
        # own output name for "layer_N"/"embed"/"output_head"-shaped names.
        try:
            from coremltools.converters.mil.mil.scope import ScopeSource
        except ImportError:
            ScopeSource = None

        def torch_scope_key(op):
            if ScopeSource is None or not hasattr(op, "scopes"):
                return None
            mn = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, ())
            if not mn:
                return None
            for i in range(len(mn) - 1):
                if mn[i].isdigit():
                    return tuple(mn[:i + 1])
            return tuple(mn[:-1]) if len(mn) > 1 else tuple(mn)

        def name_regex_key(op):
            op_name = op.outputs[0].name if op.outputs else op.name
            match = re.search(r'(layers?|blk|blocks?|modules?|linear|dense|fc|conv)_(\d+)', op_name, re.IGNORECASE)
            if match:
                return f"layer_{match.group(2)}"
            if any(x in op_name.lower() for x in ("embed", "emb", "wte")):
                return "embedding"
            if any(x in op_name.lower() for x in ("lm_head", "output_head", "logits", "pred", "output")):
                return "output_head"
            return None

        def boundary_key(op):
            return torch_scope_key(op) or name_regex_key(op)

        def label_for(key):
            if isinstance(key, str):
                return key
            if key[-1].isdigit():
                return f"layer_{key[-1]}"
            # Match on the LEAF scope segment only (the immediate submodule attribute name), and
            # exactly rather than by substring: a broad "'embed' in ..." match on the whole path
            # also matches LFM2's final-output RMSNorm, which the model confusingly calls
            # "embedding_norm" despite having nothing to do with the token-embedding lookup --
            # colliding both onto the same "embedding" label and silently discarding one topology.
            leaf = key[-1].lower()
            if leaf in ("embed_tokens", "embedding", "wte", "tok_embeddings"):
                return "embedding"
            if leaf in ("lm_head", "output_head", "logits"):
                return "output_head"
            return self.safe_name("_".join(key))

        # Seed the initial slice with the first identifiable boundary (ops before it, e.g. leading
        # const/cast setup with no scope opinion of their own, join that first slice).
        initial_key = None
        for op in operations:
            k = boundary_key(op)
            if k is not None:
                initial_key = k
                break

        slices = [] # list of tuples: (slice_name, ops_list)
        if initial_key is not None:
            current_key = initial_key
            current_ops = []
            for op in operations:
                k = boundary_key(op)
                if k is not None and k != current_key:
                    if current_ops:
                        slices.append((current_key, current_ops))
                    current_ops = []
                    current_key = k
                current_ops.append(op)
            if current_ops:
                slices.append((current_key, current_ops))

            # Fold metadata-only slices (e.g. precomputed rotary-embedding tables, which show up as
            # their own scope but are pure const/cast) forward into the next slice that actually
            # consumes them -- they have no compute/output of their own to serve as a standalone topology.
            merged = []
            pending = []
            for key, ops in slices:
                if all(op.op_type in ("const", "cast") for op in ops):
                    pending.extend(ops)
                    continue
                merged.append((key, pending + ops))
                pending = []
            if pending:
                if merged:
                    merged[-1] = (merged[-1][0], merged[-1][1] + pending)
                else:
                    merged = []
            slices = [(label_for(key), ops) for key, ops in merged]

        if len(slices) <= 1:
            raise ValueError("No distinct scope-based layer boundaries could be identified in the graph.")

        # 2. Extract inputs/outputs interfaces for each sliced topology
        # Replicate consumed constants locally in each slice to decouple them,
        # then extract only non-constant variable inputs.
        for idx, (name, ops) in enumerate(slices):
            local_ops = list(ops)
            consumed_consts = []
            for op in ops:
                for k, v in op.inputs.items():
                    from coremltools.converters.mil.mil import Var
                    if isinstance(v, Var) and v.op and v.op.op_type == "const" and v not in op.outputs:
                        if v.op not in local_ops and v.op not in consumed_consts:
                            consumed_consts.append(v.op)
            if consumed_consts:
                local_ops = consumed_consts + local_ops
            slices[idx] = (name, local_ops)

        slice_inputs = {}

        for name, ops in slices:
            # A var is an external input of THIS slice iff the op that produced it is not itself
            # part of this slice's own op list -- i.e. it's either a top-level function input (its
            # producing op is a placeholder, not in `ops`) or another slice's output. Checking
            # membership against a running "seen so far" set (as a previous version of this code did)
            # is wrong: it also catches purely-internal intermediates produced earlier in the SAME
            # slice, misclassifying them as external inputs (observed on LFM2: 30-60 bogus "inputs"
            # per layer, some with >4 MIL dims, crashing ggml_new_tensor at runtime).
            ops_set = set(ops)
            slice_in = {}
            for op in ops:
                for k, v in op.inputs.items():
                    from coremltools.converters.mil.mil import Var
                    if isinstance(v, Var) and v not in op.outputs and v.op not in ops_set:
                        if v.op and v.op.op_type == "const":
                            continue
                        slice_in[self.safe_name(v.name)] = v
            slice_inputs[name] = slice_in
            
        # 3. Generate topologies for all sliced sub-graphs
        for name, ops in slices:
            inputs_dict = slice_inputs[name]
            self.topologies[name] = self.generate_graph_topology(None, name, ops_list=ops, inputs_dict=inputs_dict)
            
        # 4. Synthesize the automatic looping driver IR
        first_input = "tokens"
        feature_scale = 1
        if main_func.inputs:
            first_input_var = list(main_func.inputs.values())[0]
            first_input = self.safe_name(list(main_func.inputs.keys())[0])
            if hasattr(first_input_var, "shape") and len(first_input_var.shape) == 3:
                try:
                    feature_scale = int(first_input_var.shape[2])
                except (ValueError, TypeError):
                    pass

        n_tokens_expr = Len(first_input)
        if feature_scale > 1:
            n_tokens_expr = BinOp("floordiv", Len(first_input), Lit(feature_scale))

        body = []
        for name in main_func.inputs.keys():
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                # See apply_monolithic_export's identical case: host-computed, not caller-supplied.
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            elif name in _CAUSAL_MASK_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))
            else:
                body.append(Local(safe_inp, BinOp("or", FieldAccess("inputs", safe_inp), FieldAccess("inputs", "tokens"))))

        for idx, (name, ops) in enumerate(slices):
            inputs_dict = slice_inputs[name]

            # Map input keys, standardizing the first input name to "hidden_states" for decoder layers
            is_layer = name.startswith("layer_")
            first_key = list(inputs_dict.keys())[0] if inputs_dict else None

            slice_inputs_tbl = {}
            for k in inputs_dict.keys():
                safe_k = self.safe_name(k)
                if is_layer and k == first_key:
                    slice_inputs_tbl["hidden_states"] = IRVar(safe_k)
                else:
                    slice_inputs_tbl[safe_k] = IRVar(safe_k)

            last_op = ops[-1]
            output_names = [self.safe_name(v.name) for v in last_op.outputs]

            if idx == len(slices) - 1:
                # Final slice: also capture the output shape so the driver can argmax the last
                # sequence position's logits row instead of returning a raw output value (matching
                # the "argmax" convention transpile_operation's bespoke path and
                # apply_monolithic_export both use for causal-LM next-token generation).
                body.append(SubgraphCall(
                    outputs=output_names, extra_outputs=["_atomic_final_shape"],
                    module=name, n_tokens=n_tokens_expr, n_past=Lit(0), inputs=slice_inputs_tbl,
                ))
            else:
                body.append(SubgraphCall(
                    outputs=output_names, module=name, n_tokens=n_tokens_expr, n_past=Lit(0), inputs=slice_inputs_tbl,
                ))

        final_last_op = slices[-1][1][-1]
        final_output_names = [self.safe_name(v.name) for v in final_last_op.outputs]
        final_out = final_output_names[0]
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar(final_out)]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar(final_out), Index(IRVar("_atomic_final_shape"), 1), row_expr])])],
            else_=[Return([IRVar(final_out)])],
        ))

        self.ir_function = IRFunction("main", ["inputs"], body)

    def transpile_to_lua(self, func: Function, name="main"):
        """
        Transpiles the main MIL orchestration function to a Lua JIT driver script.
        """
        # Track the first input variable name to dynamically derive n_tokens
        self.first_input = "tokens"
        if func.inputs:
            self.first_input = self.safe_name(list(func.inputs.keys())[0])

        body = []
        # Unpack incoming inputs
        for inp_name in func.inputs.keys():
            safe_inp = self.safe_name(inp_name)
            body.append(Local(safe_inp, FieldAccess("inputs", safe_inp)))

        # Transpile operations inside function block
        body.extend(self.transpile_block(func))

        # Unpack and return outputs
        output_names = [self.safe_name(v.name) for v in func.outputs]
        body.append(Return([IRVar(n) for n in output_names]))

        self.ir_function = IRFunction(name, ["inputs"], body)

    def transpile_block(self, block: Block) -> list:
        stmts = []
        for op in block.operations:
            stmts.extend(self.transpile_operation(op))
        return stmts

    def transpile_operation(self, op: Operation) -> list:
        op_type = op.op_type
        output_names = [self.safe_name(v.name) for v in op.outputs]

        # A. Constant Serialization
        if op_type == "const":
            val = op.val.val
            if isinstance(val, np.ndarray) and val.size > 100:
                weight_name = self.safe_name(op.outputs[0].name)
                self.weights[weight_name] = val
                return [RawBlock([f"-- Weight {weight_name} packaged in GGUF"])]
            return [Local(output_names[0], RawExpr(self.format_lua_val(val)))]

        # B. Autoregressive / Loop Control Flow
        if op_type == "while_loop":
            cond_block = op.blocks[0]
            body_block = op.blocks[1]

            cond_stmts = self.transpile_block(cond_block)
            cond_var = self.safe_name(cond_block.outputs[0].name)
            body_stmts = self.transpile_block(body_block)
            loop_body = cond_stmts + [If(UnaryOp("not", IRVar(cond_var)), [Break()])] + body_stmts
            return [While(Lit(True), loop_body)]

        # C. Conditional Branches
        if op_type == "cond":
            true_block = op.blocks[0]
            false_block = op.blocks[1]
            pred_var = self.safe_name(op.inputs["pred"].name)

            true_stmts = self.transpile_block(true_block)
            false_stmts = self.transpile_block(false_block)

            # `cond`'s own output(s) are each block's own returned var (true_block.outputs[i] /
            # false_block.outputs[i] map positionally to op.outputs[i]) -- neither branch's own `local`
            # declarations survive past the if/else in Lua (block-scoped), so the result name is declared
            # OUTSIDE the if/else via LocalDecl and plain-assigned (Assign, no `local`) from inside each
            # arm. Previously this case didn't bind op.outputs at all, silently leaving any use of the
            # cond's result reading an undeclared global (nil) -- caught by validate() once the IR
            # rewrite made "read before defined" a mechanical, export-time check instead of a runtime bug.
            decls = []
            for i, out_var in enumerate(op.outputs):
                out_name = self.safe_name(out_var.name)
                decls.append(LocalDecl(out_name))
                true_stmts = true_stmts + [Assign(out_name, IRVar(self.safe_name(true_block.outputs[i].name)))]
                false_stmts = false_stmts + [Assign(out_name, IRVar(self.safe_name(false_block.outputs[i].name)))]

            return decls + [If(IRVar(pred_var), true_stmts, false_stmts)]

        # D. Submodule Dispatch
        if op_type in self.program.functions:
            inputs_tbl = {k: IRVar(self.safe_name(v.name)) for k, v in op.inputs.items() if hasattr(v, "name")}
            n_tokens_expr = Len(self.first_input)
            n_past_expr = Lit(0)
            if "n_tokens" in op.inputs:
                n_tokens_expr = IRVar(self.safe_name(op.inputs["n_tokens"].name))
            if "n_past" in op.inputs:
                n_past_expr = IRVar(self.safe_name(op.inputs["n_past"].name))
            return [SubgraphCall(outputs=output_names, module=op_type, n_tokens=n_tokens_expr, n_past=n_past_expr,
                                  inputs=inputs_tbl)]

        # E. Fast Host Math Mapping
        if op_type == "argmax":
            x_name = self.safe_name(op.inputs["x"].name)
            # Retrieve shape to get vocabulary size (ne0 dimension)
            x_info = self.get_var_info(op.inputs["x"])
            n_vocab = int(x_info["shape"][0])
            # Row index is the last row (seq_len - 1), which is #first_input - 1
            row_expr = BinOp("-", Len(self.first_input), Lit(1))
            return [Argmax(output_names[0], x_name, Lit(n_vocab), row_expr)]

        if op_type == "range":
            start = self.safe_name(op.inputs["start"].name)
            end = self.safe_name(op.inputs["end"].name)
            return [Local(output_names[0], Call("loom.range", [IRVar(start), IRVar(end)]))]

        if op_type == "causal_mask":
            n_tokens = self.safe_name(op.inputs["n_tokens"].name)
            n_past = self.safe_name(op.inputs["n_past"].name)
            return [Local(output_names[0], Call("loom.causal_mask", [IRVar(n_tokens), IRVar(n_past)]))]

        # F. Fallback for generic SSA arithmetic
        inputs = [self.safe_name(v.name) for k, v in op.inputs.items() if hasattr(v, "name")]
        if len(inputs) == 2:
            op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(op_type)
            if op_symbol:
                return [Local(output_names[0], BinOp(op_symbol, IRVar(inputs[0]), IRVar(inputs[1])))]
        return [RawBlock([f"-- Fallback: host math implementation for {op_type}"])]

    def format_lua_val(self, val):
        if isinstance(val, (int, float, bool)):
            return str(val).lower()
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, np.ndarray):
            return "{" + ", ".join(map(str, val.flatten())) + "}"
        return "nil"

    def _try_fuse_gqa_repeat_kv(self, tile_op, reshape_op, resolve):
        """
        Detects whether `tile_op` (whose consumer is `reshape_op`) matches HF's standard `repeat_kv()`
        idiom -- a tile of a size-1 axis immediately merged back into an adjacent axis by a reshape --
        and if so returns a list of composed node dicts fusing the pair. Returns None if the shapes
        don't match this specific pattern (caller falls back to the generic per-op handling).

        Derivation, per axis: prefer `reshape_op`'s own declared output dim IF it's "reliable" (doesn't
        reference more than one distinct dynamic symbol -- see `_DYNAMIC_SYMBOL_RE`); this correctly
        picks up the one axis that actually changes (the merged head count, always a clean static int).
        Every OTHER axis -- confirmed empirically to be unreliable too, not just the merged one, once
        `reps` isn't a compile-time constant (coremltools' shape algebra loses track of the seq-length
        axis here as well, reporting it as an unresolvable multi-symbol product) -- falls back to
        `tile_op`'s own pre-expand input shape instead, which is unaffected by any of this and reliable by
        construction (repeat_kv()'s reshape never actually touches seq/head_dim/batch numerically).

        IMPORTANT correctness note (found the hard way -- see EXPORT-BACKLOG.md item 3): a single native
        REPEAT (`ggml_repeat`) is NOT equivalent to HF's `repeat_kv()`. `ggml_repeat` block-tiles an axis
        (`dst[i] = src[i % ne_src]`, i.e. concatenating whole copies: kv0,kv1,...,kv7,kv0,kv1,...,kv7),
        while `repeat_kv()`'s unsqueeze->expand->reshape-merge idiom produces an *interleaved* repeat
        (`dst[i] = src[i // n_rep]`, i.e. kv0,kv0,kv1,kv1,...,kv7,kv7) -- the standard GQA head-group
        convention every model on this roadmap uses. These only agree when n_rep==1. The fix composes
        three primitives that reproduce the interleaved semantics exactly, mirroring what HF's own ops
        actually do: (1) RESHAPE the pre-tile tensor to insert a genuine size-1 axis in the position the
        real (un-collapsed) unsqueeze put it -- a pure relabeling, moves no data since the axis being
        vacated is already size 1 (batch); (2) REPEAT that size-1 axis up to `n_rep` -- always safe
        regardless of block-tile-vs-interleave semantics, since repeating a *single* source element by
        any tiling scheme yields the same output; (3) RESHAPE again to merge the now-`n_rep`-sized axis
        into the adjacent kv-heads axis, with `n_rep` as the faster-varying component of the pair --
        exactly reproducing `dst[i] = src[i // n_rep]` via a plain contiguous axis-merge.
        """
        pre_tile_x = tile_op.inputs.get("x")
        if pre_tile_x is None:
            return None
        # `tile_op`'s own "x" is itself the output of the preceding unsqueeze (HF's repeat_kv() inserts
        # the new axis via `[:, :, None, :, :]`, which traces as a MIL "expand_dims"), NOT the genuine
        # pre-expand tensor -- walk back through it to find the tensor whose rank actually matches the
        # reshape's own (merged) output rank.
        producer = getattr(pre_tile_x, "op", None)
        if producer is not None and producer.op_type == "expand_dims":
            inner_x = producer.inputs.get("x") or producer.inputs.get("data")
            if inner_x is not None:
                pre_tile_x = inner_x
        if not hasattr(pre_tile_x, "shape") or pre_tile_x.shape is None:
            return None
        pre_rank = len(pre_tile_x.shape)

        out_var = reshape_op.outputs[0]
        if not hasattr(out_var, "shape") or out_var.shape is None or len(out_var.shape) != pre_rank:
            return None
        out_shape = self.get_var_info(out_var)["shape"]
        raw_out_dims = list(reversed(out_var.shape))
        pre_shape = self.get_var_info(pre_tile_x)["shape"]

        target_shape = []
        changed_axis = None
        for i in range(pre_rank):
            raw_str = str(raw_out_dims[i])
            if len(set(_DYNAMIC_SYMBOL_RE.findall(raw_str))) > 1:
                target_shape.append(pre_shape[i])
            else:
                target_shape.append(out_shape[i])
                if str(out_shape[i]) != str(pre_shape[i]):
                    changed_axis = i

        # This fusion only knows how to handle the exact repeat_kv() shape (one axis grows by an integer
        # ratio, the rest unchanged) -- anything else falls back to the generic per-op handling.
        if changed_axis is None or changed_axis == 0:
            return None
        try:
            kv_count = int(pre_shape[changed_axis])
            out_count = int(target_shape[changed_axis])
        except (TypeError, ValueError):
            return None
        if kv_count <= 0 or out_count % kv_count != 0:
            return None
        ratio = out_count // kv_count
        if ratio == 1:
            return None

        # ggml caps tensors at 4 dims -- making room for a genuine new axis at `changed_axis` (shifting
        # kv-heads outward by one slot) only works if the outermost axis it displaces into is free, i.e.
        # size 1. That's always batch here (every model on this roadmap runs batch=1), but verify rather
        # than assume: this fusion doesn't apply otherwise.
        if pre_rank != 4 or str(pre_shape[-1]) != "1":
            return None

        pre_name = resolve(self.safe_name(pre_tile_x.name))
        out_name = self.safe_name(out_var.name)
        reshape1_name = out_name + "_gqa_unsqueeze"
        repeat_name = out_name + "_gqa_repeat"

        # (1) insert a genuine size-1 axis at `changed_axis`, pushing kv-heads out one slot into the
        # (always size-1) batch axis -- a pure relabeling of the same flat data, since the axis being
        # vacated contributes nothing to the strides.
        reshape1_shape = pre_shape[:changed_axis] + ["1", pre_shape[changed_axis]]
        # (2) grow that new size-1 axis to `ratio` -- always safe (repeating a single source element by
        # any tiling scheme gives the same result regardless of block-tile-vs-interleave semantics).
        repeat_shape = pre_shape[:changed_axis] + [str(ratio), pre_shape[changed_axis]]
        # (3) merge (ratio, kv_count) back into one axis of size `ratio*kv_count`, with `ratio` as the
        # faster-varying component of the pair -- a plain contiguous axis-merge that reproduces
        # `dst[i] = src[i // ratio]` exactly, matching HF's own reshape-after-expand.
        final_shape = list(target_shape)

        return [
            {
                "op": "RESHAPE",
                "inputs": [pre_name],
                "outputs": [reshape1_name],
                "attrs": {"shape": reshape1_shape},
            },
            {
                "op": "REPEAT",
                "inputs": [reshape1_name],
                "outputs": [repeat_name],
                "attrs": {"shape": repeat_shape},
            },
            {
                "op": "RESHAPE",
                "inputs": [repeat_name],
                "outputs": [out_name],
                "attrs": {"shape": final_shape},
            },
        ]

    def generate_graph_topology(self, func: Function, func_name: str, ops_list=None, inputs_dict=None) -> dict:
        """
        Walks a heavy submodule MIL graph and serializes it to a static graph topology.
        """
        nodes = []
        topo_inputs = []
        aliases = {}
        
        inputs = inputs_dict if inputs_dict is not None else (func.inputs if func else {})
        operations = ops_list if ops_list is not None else (func.operations if func else [])

        def resolve(name):
            while name in aliases:
                name = aliases[name]
            return name
        
        # Track inputs to the submodule and standardize the first input name to "hidden_states" for decoder layers
        first_input_var = None
        for name, var in inputs.items():
            if first_input_var is None:
                first_input_var = var
                
        if first_input_var is not None:
            orig_name = self.safe_name(first_input_var.name)
            
            if func_name.startswith("layer_"):
                aliases[orig_name] = "hidden_states"
                var_info = self.get_var_info(first_input_var)
                var_info["name"] = "hidden_states"
                topo_inputs.append(var_info)
            else:
                topo_inputs.append(self.get_var_info(first_input_var))
            
            for name, var in inputs.items():
                if var != first_input_var:
                    topo_inputs.append(self.get_var_info(var))
        else:
            for name, var in inputs.items():
                topo_inputs.append(self.get_var_info(var))

        # Populated by the "tile" branch below when it fuses a tile+reshape GQA repeat_kv() pair into a
        # single REPEAT node -- the "reshape"/"expand_dims"/"squeeze" branch checks this to skip
        # re-emitting a (redundant, incorrect) node for a reshape op already handled that way.
        fused_reshape_op_ids = set()

        for op in operations:
            op_type = op.op_type
            if op_type == "const":
                val = op.val.val
                weight_name = self.safe_name(op.outputs[0].name)
                
                # For monolithic profiles, skip namespace prefixing
                if func_name == "main_topo" or self.profile == "monolithic":
                    namespaced_name = weight_name
                else:
                    namespaced_name = f"{func_name}.{weight_name}"
                    
                # Safe compaction to satisfy GGUF's GGML_MAX_NAME (64 chars) limit
                if len(namespaced_name) >= 64:
                    import hashlib
                    h = hashlib.md5(namespaced_name.encode("utf-8")).hexdigest()[:6]
                    namespaced_name = f"{namespaced_name[:30]}_{h}_{namespaced_name[-20:]}"
                    
                self.weights[namespaced_name] = np.array(val)
                if weight_name != namespaced_name:
                    aliases[weight_name] = namespaced_name
                continue

            if op_type == "cast":
                input_name = self.safe_name(op.inputs["x"].name)
                output_name = self.safe_name(op.outputs[0].name)
                # MIL casts between float precisions (fp16<->fp32) are pure no-ops for this engine, since
                # every ggml op here computes in f32 internally regardless of a tensor's storage dtype --
                # aliasing them away is correct and avoids emitting a mountain of redundant CAST nodes.
                # But a cast that actually changes numeric *kind* (e.g. HF rotary embedding's
                # `position_ids.float()`, MIL's int32->fp32) is a real value reinterpretation: skipping it
                # leaves an integer-typed ggml tensor flowing into float-only ops downstream (MUL_MAT's
                # vec_dot has no integer kernel and null-derefs). Emit a real CAST node for that case.
                in_dtype = self.get_var_info(op.inputs["x"])["dtype"]
                out_dtype = self.get_var_info(op.outputs[0])["dtype"]
                if in_dtype != out_dtype:
                    nodes.append({
                        "op": "CAST",
                        "inputs": [resolve(input_name)],
                        "outputs": [output_name],
                        "attrs": {"dtype": out_dtype},
                    })
                else:
                    aliases[output_name] = resolve(input_name)
                continue

            if op_type == "linear":
                # Compose linear as MUL_MAT + optional ADD
                x_var_obj = op.inputs.get("x") or op.inputs.get("input")
                x_var = self.safe_name(x_var_obj.name)
                weight_var = self.safe_name(op.inputs["weight"].name)
                bias_var = self.safe_name(op.inputs["bias"].name) if "bias" in op.inputs and hasattr(op.inputs["bias"], "name") else None
                output_var = self.safe_name(op.outputs[0].name)
                
                # In Loom, MUL_MAT expects [weight, x]
                if bias_var:
                    inter_var = output_var + "_matmul"
                    nodes.append({
                        "op": "MUL_MAT",
                        "inputs": [resolve(weight_var), resolve(x_var)],
                        "outputs": [inter_var]
                    })
                    nodes.append({
                        "op": "ADD",
                        "inputs": [inter_var, resolve(bias_var)],
                        "outputs": [output_var]
                    })
                else:
                    nodes.append({
                        "op": "MUL_MAT",
                        "inputs": [resolve(weight_var), resolve(x_var)],
                        "outputs": [output_var]
                    })
                continue

            if op_type == "matmul":
                # MIL's matmul(x, y, transpose_x, transpose_y) computes X @ Y where X = x^T if
                # transpose_x else x, Y = y^T if transpose_y else y (batched over leading dims). This is
                # NOT the same op as ggml_mul_mat(A, B), which always contracts over ne0 of both operands
                # and returns ne=[A.ne1, B.ne1, ...] -- i.e. it computes B_mat @ A_mat^T, not A_mat @
                # B_mat. Forwarding MIL's x/y straight through as ggml_mul_mat(x, y) silently produces a
                # transposed-but-same-shape (for square attention scores) or outright wrong-axis result,
                # exactly the numerical-correctness bug tracked in EXPORT-BACKLOG.md item 1 -- confirmed
                # by bisecting the real attention-score matmul (transpose_y=True) and the
                # scores@value matmul (transpose_x=transpose_y=False) against HF's own SDPA inputs.
                #
                # Both combinations used by scaled_dot_product_attention's decomposition are handled
                # explicitly below (derived from ggml_mul_mat's result.ne=[A.ne1,B.ne1,B.ne2,B.ne3]
                # formula); any other combination is intentionally unsupported rather than silently wrong.
                x_var_obj = op.inputs["x"]
                y_var_obj = op.inputs["y"]
                x_var = self.safe_name(x_var_obj.name)
                y_var = self.safe_name(y_var_obj.name)
                output_var = self.safe_name(op.outputs[0].name)

                tx_var = op.inputs.get("transpose_x")
                ty_var = op.inputs.get("transpose_y")
                tx = bool(tx_var.val) if tx_var is not None and hasattr(tx_var, "val") else False
                ty = bool(ty_var.val) if ty_var is not None and hasattr(ty_var, "val") else False

                if not tx and ty:
                    # X @ Y^T: both operands already share ne0 (the contracted/embedding axis) in their
                    # natural layout, so this is a straight ggml_mul_mat(y, x) -- key-first, matching the
                    # llama.cpp attention-score convention.
                    nodes.append({
                        "op": "MUL_MAT",
                        "inputs": [resolve(y_var), resolve(x_var)],
                        "outputs": [output_var]
                    })
                elif not tx and not ty:
                    # X @ Y: Y needs its leading two ne axes swapped (and made contiguous) before it can
                    # be used as ggml_mul_mat's first ("A") operand -- see the derivation in the comment
                    # above. Composed as PERMUTE + CONT so the C++ side never has to guess this from
                    # shapes alone.
                    perm_var = output_var + "_mm_y_perm"
                    cont_var = output_var + "_mm_y_cont"
                    nodes.append({
                        "op": "PERMUTE",
                        "inputs": [resolve(y_var)],
                        "outputs": [perm_var],
                        "attrs": {"axes": [1, 0, 2, 3]}
                    })
                    nodes.append({
                        "op": "CONT",
                        "inputs": [perm_var],
                        "outputs": [cont_var]
                    })
                    nodes.append({
                        "op": "MUL_MAT",
                        "inputs": [cont_var, resolve(x_var)],
                        "outputs": [output_var]
                    })
                else:
                    raise NotImplementedError(
                        f"matmul op '{op.name}' has transpose_x={tx}, transpose_y={ty}, which no "
                        "exporter composition handles yet (only transpose_x=False has been needed so far)."
                    )
                continue

            if op_type == "split":
                # Compose split as multiple zero-copy VIEW slices
                x_var = self.safe_name(op.inputs["x"].name)
                axis = op.inputs["axis"].val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
                
                # Retrieve input shape info (ne-reversed shape)
                x_info = self.get_var_info(op.inputs["x"])
                ne_shape = x_info["shape"]
                rank = len(ne_shape)
                
                # Normalize negative axis relative to the tensor rank
                if axis < 0:
                    axis = rank + axis
                
                # Map MIL standard axis to Loom ne-reversed axis
                ne_axis = rank - 1 - axis
                num_splits = len(op.outputs)
                dim_to_split = ne_shape[ne_axis]
                
                if isinstance(dim_to_split, int):
                    split_dim_size = dim_to_split // num_splits
                else:
                    split_dim_size = f"({dim_to_split} / {num_splits})"
                    
                # Create a VIEW node for each split output
                for idx, out_var in enumerate(op.outputs):
                    out_name = self.safe_name(out_var.name)
                    
                    slice_shape = list(ne_shape)
                    slice_shape[ne_axis] = split_dim_size
                    
                    # Calculate byte offset rule
                    offset_elements = f"{idx} * {split_dim_size}"
                    for prev_ax in range(ne_axis):
                        offset_elements = f"({offset_elements} * {ne_shape[prev_ax]})"
                    offset_bytes = f"({offset_elements} * 4)" # 4 bytes per float element
                    
                    nodes.append({
                        "op": "VIEW",
                        "inputs": [resolve(x_var)],
                        "outputs": [out_name],
                        "attrs": {
                            "shape": slice_shape,
                            "offset": offset_bytes
                        }
                    })
                continue

            if op_type == "slice_by_index":
                # Compose slice_by_index as an optimized zero-copy VIEW node
                x_var = self.safe_name(op.inputs["x"].name)
                output_var = self.safe_name(op.outputs[0].name)
                
                begin = op.inputs["begin"].val if "begin" in op.inputs and hasattr(op.inputs["begin"], "val") else [0]
                end = op.inputs["end"].val if "end" in op.inputs and hasattr(op.inputs["end"], "val") else [1]
                
                x_info = self.get_var_info(op.inputs["x"])
                ne_shape = x_info["shape"]
                rank = len(ne_shape)
                
                begin_list = list(begin) if isinstance(begin, (list, tuple, np.ndarray)) else [begin]
                end_list = list(end) if isinstance(end, (list, tuple, np.ndarray)) else [end]
                
                while len(begin_list) < rank:
                    begin_list.append(0)
                while len(end_list) < rank:
                    end_list.append(ne_shape[rank - 1 - len(end_list)])
                    
                slice_shape = []
                for i in range(rank):
                    mil_axis = rank - 1 - i
                    dim_size = ne_shape[i]
                    b_val = begin_list[mil_axis]
                    e_val = end_list[mil_axis]
                    
                    if b_val is None:
                        b_val = 0
                    if e_val is None:
                        e_val = dim_size
                    
                    # For axis 0 (the head dimension being split), compute slice size.
                    # For all other dimensions, preserve the parent's shape exactly to avoid layout swaps!
                    if i == 0:
                        if isinstance(dim_size, int):
                            if e_val < 0:
                                e_val = dim_size + e_val
                            b_val = max(0, min(dim_size, b_val))
                            e_val = max(0, min(dim_size, e_val))
                            slice_shape.append(e_val - b_val)
                        else:
                            slice_shape.append(f"({e_val} - {b_val})")
                    else:
                        slice_shape.append(dim_size)
                        
                # Calculate byte offset in C-major MIL layout mapping to ne_shape strides:
                offset_elements = "0"
                for i in range(rank):
                    b_val = begin_list[i]
                    if b_val != 0:
                        stride_product = "1"
                        ne_limit = rank - 1 - i
                        for prev_ax in range(ne_limit):
                            stride_product = f"({stride_product} * {ne_shape[prev_ax]})"
                        offset_elements = f"({offset_elements} + ({b_val} * {stride_product}))"
                        
                offset_bytes = f"({offset_elements} * 4)"
                
                nodes.append({
                    "op": "VIEW",
                    "inputs": [resolve(x_var)],
                    "outputs": [output_var],
                    "attrs": {
                        "shape": slice_shape,
                        "offset": offset_bytes
                    }
                })
                continue

            if op_type == "fill":
                # Compile-time evaluation of constant fill tensors
                shape_val = op.inputs["shape"].val if "shape" in op.inputs and hasattr(op.inputs["shape"], "val") else None
                value_val = op.inputs["value"].val if "value" in op.inputs and hasattr(op.inputs["value"], "val") else 0.0
                
                if shape_val is not None:
                    shape_list = list(shape_val) if isinstance(shape_val, (list, tuple, np.ndarray)) else [shape_val]
                    ne_shape = list(reversed(shape_list))
                    
                    array = np.full(ne_shape, value_val, dtype=np.float32)
                    weight_name = self.safe_name(op.outputs[0].name)
                    self.weights[weight_name] = array
                    continue
                else:
                    # Dynamic fill: pre-allocate a safe max-size static constant weight and slice via VIEW
                    rank = len(self.get_var_info(op.outputs[0])["shape"])
                    prealloc_shape = [4096] * rank
                    
                    val_weight_name = self.safe_name(op.outputs[0].name) + "_prealloc"
                    namespaced_val_name = f"{func_name}.{val_weight_name}"
                    self.weights[namespaced_val_name] = np.full(prealloc_shape, value_val, dtype=np.float32)
                    
                    # Sliced shape is dynamic "n_tokens" along the dynamic dimensions
                    slice_shape = ["n_tokens"] * rank
                    nodes.append({
                        "op": "VIEW",
                        "inputs": [namespaced_val_name],
                        "outputs": [self.safe_name(op.outputs[0].name)],
                        "attrs": {
                            "shape": slice_shape,
                            "offset": 0
                        }
                    })
                    continue

            if op_type == "band_part":
                # Map band_part (with lower=-1, upper=0) to DIAG_MASK_ZERO. MIL's actual input keys
                # are "lower"/"upper" (see tensor_operation.py's band_part InputSpec) -- NOT
                # "num_lower"/"num_upper", which never matched, so this always silently used the
                # (coincidentally causal-shaped) -1/0 defaults regardless of the op's real attrs. Same
                # bug class as the transpose/"perm" mismatch above; not currently exercised by LFM2
                # (its causal mask gets constant-folded at trace time rather than computed via a live
                # band_part op), but a real latent bug for any model that reaches this path.
                num_lower = op.inputs["lower"].val if "lower" in op.inputs and hasattr(op.inputs["lower"], "val") else -1
                num_upper = op.inputs["upper"].val if "upper" in op.inputs and hasattr(op.inputs["upper"], "val") else 0
                
                x_var = self.safe_name(op.inputs["x"].name)
                output_var = self.safe_name(op.outputs[0].name)
                
                if num_lower == -1 and num_upper == 0:
                    # It's a lower-triangle zero mask (causal zero mask)
                    nodes.append({
                        "op": "DIAG_MASK_ZERO",
                        "inputs": [resolve(x_var)],
                        "outputs": [output_var],
                        "attrs": {"n_past": 0}
                    })
                else:
                    # Keep all (no-op / alias)
                    aliases[output_var] = resolve(x_var)
                continue

            if op_type == "transpose":
                # Map transpose to PERMUTE with ne-reversed axes
                x_var = self.safe_name(op.inputs["x"].name)
                output_var = self.safe_name(op.outputs[0].name)
                
                x_info = self.get_var_info(op.inputs["x"])
                rank = len(x_info["shape"])
                
                # MIL's transpose op names its (required) permutation input "perm", not "axes" --
                # checking for "axes" here always missed, silently falling back to an identity
                # permutation for EVERY transpose in the model (confirmed on LFM2: every single
                # PERMUTE node in both the ShortConv and attention layers was emitted as a no-op
                # [0,1,2,3], which is what caused the numerical mismatch against the real model).
                perm_var = op.inputs.get("perm") or op.inputs.get("axes")
                if perm_var is None or not hasattr(perm_var, "val"):
                    raise ValueError(f"transpose op '{op.name}' has no resolvable 'perm' constant")
                # perm entries may be negative (confirmed on LFM2: e.g. [0, -1, -2] for .transpose(-1,-2)).
                raw_perm = [int(p) for p in perm_var.val]
                norm_perm = [(p + rank) if p < 0 else p for p in raw_perm]

                # MIL's semantics: output.shape[i] = input.shape[norm_perm[i]] -- destination axis i
                # PULLS FROM source axis norm_perm[i]. ggml_permute's signature is the opposite
                # direction: ggml_permute(x, axis0..axis3) means "source ne[k] MOVES TO dest ne[axis_k]"
                # (see ggml.c: result->ne[axis_k] = a->ne[k]). Converting one to the other needs the
                # INVERSE permutation, not a direct pass-through -- on top of the usual ne-order axis
                # reversal (MIL axis a <-> ne-axis rank-1-a). Passing norm_perm straight through
                # (reversed only) silently permuted every transpose in the model incorrectly.
                inv_perm = [0] * rank
                for i, p in enumerate(norm_perm):
                    inv_perm[p] = i
                ne_axes = []
                for k in range(rank):
                    a = rank - 1 - k       # MIL input axis feeding ggml source axis k
                    b = inv_perm[a]        # MIL output axis that input axis a lands at
                    ne_axes.append(rank - 1 - b)
                while len(ne_axes) < 4:
                    ne_axes.append(len(ne_axes))
                    
                nodes.append({
                    "op": "PERMUTE",
                    "inputs": [resolve(x_var)],
                    "outputs": [output_var],
                    "attrs": {"axes": ne_axes}
                })
                continue

            if op_type == "tile":
                # GQA repeat_kv() fusion: HF's standard `repeat_kv()` idiom (used to tile K/V from
                # num_key_value_heads up to num_attention_heads) traces as
                # unsqueeze(new_axis) -> tile(new_axis, n_rep) -> reshape(merge new_axis into the
                # adjacent one). When this tile's own `reps` isn't a compile-time constant (confirmed:
                # coremltools reports `reps.val is None` here, because n_rep -- architecturally a FIXED
                # hyperparameter -- gets computed via a runtime shape query during tracing anyway), the
                # generic "tile" handling below silently treats every unreliable axis as rep_factor=1
                # (a no-op), and the tile's own OUTPUT shape is symbolic in every axis too. Detect the
                # fusable pattern up front (via the tile's single consumer) and compose the interleaved
                # RESHAPE->REPEAT->RESHAPE sequence directly from RELIABLE information -- the tile's own
                # pre-expand input shape for every unchanged axis, and the downstream reshape's own
                # reliably-inferred output shape for the one axis that actually changes -- entirely
                # bypassing the poisoned intermediate.
                child_ops = list(op.outputs[0].child_ops) if op.outputs else []
                if len(child_ops) == 1 and child_ops[0].op_type == "reshape":
                    fused_nodes = self._try_fuse_gqa_repeat_kv(op, child_ops[0], resolve)
                    if fused_nodes is not None:
                        nodes.extend(fused_nodes)
                        fused_reshape_op_ids.add(id(child_ops[0]))
                        continue

                # Map tile to REPEAT by calculating the target shape
                x_var = self.safe_name(op.inputs["x"].name)
                reps = op.inputs["reps"].val if "reps" in op.inputs and hasattr(op.inputs["reps"], "val") else [1]
                
                # Retrieve input shape info (ne-reversed shape)
                x_info = self.get_var_info(op.inputs["x"])
                ne_shape = x_info["shape"]
                rank = len(ne_shape)
                
                reps_list = list(reps) if isinstance(reps, (list, tuple, np.ndarray)) else [reps]
                if len(reps_list) > rank:
                    reps_list = reps_list[-rank:]
                while len(reps_list) < rank:
                    reps_list.insert(0, 1)
                
                # Target shape in ne-order
                target_shape = []
                for i in range(rank):
                    mil_axis = rank - 1 - i
                    dim_size = ne_shape[i]
                    rep_factor = reps_list[mil_axis]
                    if rep_factor is None:
                        rep_factor = 1
                    
                    try:
                        dim_int = int(dim_size)
                        target_shape.append(str(dim_int * rep_factor))
                    except (ValueError, TypeError):
                        target_shape.append(f"({dim_size} * {rep_factor})")
                        
                # Limit target shape strictly to 4D to satisfy GGML's maximum dimension limits
                while len(target_shape) > 4 and target_shape[-1] == "1":
                    target_shape.pop()
                if len(target_shape) > 4:
                    target_shape = target_shape[:4]
                        
                nodes.append({
                    "op": "REPEAT",
                    "inputs": [resolve(x_var)],
                    "outputs": [self.safe_name(op.outputs[0].name)],
                    "attrs": {"shape": target_shape}
                })
                continue

            if op_type in ["reshape", "expand_dims", "squeeze"]:
                if id(op) in fused_reshape_op_ids:
                    # Already emitted as a single fused REPEAT by its producing "tile" op above.
                    continue

                # Map reshape/expand_dims/squeeze to RESHAPE with 1 input and a shape attribute derived
                # from the op's own declared output shape.
                x_var_obj = op.inputs.get("x") or op.inputs.get("data")
                x_var = self.safe_name(x_var_obj.name)

                out_var = op.outputs[0]
                out_info = self.get_var_info(out_var)
                # Same ne-order reversal get_var_info applies, kept aligned 1:1 with out_info["shape"] so
                # raw_dims[i] is the exact MIL-side (pre-substitution) source of out_info["shape"][i].
                raw_dims = list(reversed(out_var.shape)) if hasattr(out_var, "shape") and out_var.shape is not None else []

                x_info = self.get_var_info(x_var_obj) if hasattr(x_var_obj, "shape") and x_var_obj.shape is not None else None
                out_rank = len(out_info["shape"])
                in_rank = len(x_info["shape"]) if x_info is not None else -1

                if x_info is not None and in_rank > out_rank:
                    # A rank-REDUCING reshape (merging several trailing MIL axes -- i.e. several LEADING
                    # ne-order axes -- into one). Coremltools' own output-shape inference for the merged
                    # axis is fundamentally untrustworthy here: it isn't just that a single symbol might
                    # be a genuinely-different static quantity reported symbolically (confirmed on LFM2's
                    # attention-output reshape merging (heads=16, head_dim=64) into hidden_size=1024,
                    # reported as a lone symbol that is NOT n_tokens); coremltools also renumbers even
                    # perfectly-unchanged pass-through axes (confirmed: this reshape's own INPUT seq axis
                    # and OUTPUT seq axis are two DIFFERENT symbol objects, despite being the identical
                    # unchanged quantity) -- so there is no symbol-identity check that can tell "reliable"
                    # apart from "unreliable" here, from the output's own shape alone. Instead, derive the
                    # target POSITIONALLY from the input's OWN (reliable) shape: every ne-order axis
                    # except the merged one is a direct, unchanged carry-over from the input at the
                    # corresponding position; the merged axis (always ne-order axis 0, since merging
                    # collapses the trailing MIL axes = leading ne-order axes) is a literal -1, delegating
                    # to op_reshape's own numpy/PyTorch-style inference (src/ops/primitives_basic.cpp) --
                    # correct by construction from the input's real total element count at build time.
                    merge_count = in_rank - out_rank + 1
                    target_shape = [-1] + list(x_info["shape"][merge_count:])
                    inferred_count = 1
                else:
                    # Rank-preserving (or rank-increasing/split) reshape: a BARE pass-through symbol
                    # (exactly one distinct symbol occupying the whole dim, no arithmetic) has
                    # consistently been the genuine n_tokens quantity in every such case seen so far
                    # (position_ids/cache_position reshapes, RoPE cos/sin, Q/K/V head-splits, etc.) --
                    # unlike the rank-REDUCING merge case above, a split/pass-through doesn't fabricate a
                    # NEW computed quantity the way a merge does, so there's no reason to distrust it.
                    # Only a genuine multi-symbol ARITHMETIC EXPRESSION (2+ distinct symbols) is treated
                    # as unreliable here, exactly as originally established.
                    target_shape = []
                    inferred_count = 0
                    for i, d in enumerate(out_info["shape"]):
                        raw_str = str(raw_dims[i]) if i < len(raw_dims) else d
                        if len(set(_DYNAMIC_SYMBOL_RE.findall(raw_str))) > 1:
                            target_shape.append(-1)
                            inferred_count += 1
                        else:
                            target_shape.append(d)
                if inferred_count > 1:
                    raise NotImplementedError(
                        f"reshape op '{op.name}' needs more than one inferred (multi-symbol) dimension "
                        f"in target shape {out_info['shape']!r} -- RESHAPE only supports a single -1 entry."
                    )

                # Limit target shape strictly to 4D to satisfy GGML's maximum dimension limits
                while len(target_shape) > 4 and target_shape[-1] == "1":
                    target_shape.pop()
                if len(target_shape) > 4:
                    target_shape = target_shape[:4]

                nodes.append({
                    "op": "RESHAPE",
                    "inputs": [resolve(x_var)],
                    "outputs": [self.safe_name(out_var.name)],
                    "attrs": {"shape": target_shape}
                })
                continue

            if op_type == "concat":
                # CONCAT in Loom expects strictly 2 inputs.
                # For 3+ inputs, we chain multiple 2-input CONCAT nodes sequentially!
                values_obj = op.inputs.get("values")
                if values_obj:
                    inputs = []
                    for item in values_obj:
                        if isinstance(item, Var):
                            inputs.append(resolve(self.safe_name(item.name)))
                    
                    if len(inputs) > 2:
                        prev_output = inputs[0]
                        output_var = self.safe_name(op.outputs[0].name)
                        
                        axis = op.inputs.get("axis").val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
                        rank = len(self.get_var_info(op.outputs[0])["shape"])
                        if axis < 0:
                            axis = rank + axis
                        ne_axis = rank - 1 - axis
                        
                        for i in range(1, len(inputs) - 1):
                            inter_output = f"{output_var}_concat_temp_{i}"
                            nodes.append({
                                "op": "CONCAT",
                                "inputs": [prev_output, inputs[i]],
                                "outputs": [inter_output],
                                "attrs": {"dim": ne_axis}
                            })
                            prev_output = inter_output
                            
                        nodes.append({
                            "op": "CONCAT",
                            "inputs": [prev_output, inputs[-1]],
                            "outputs": [output_var],
                            "attrs": {"dim": ne_axis}
                        })
                        continue
                    elif len(inputs) == 2:
                        axis = op.inputs.get("axis").val if "axis" in op.inputs and hasattr(op.inputs["axis"], "val") else 0
                        rank = len(self.get_var_info(op.outputs[0])["shape"])
                        if axis < 0:
                            axis = rank + axis
                        ne_axis = rank - 1 - axis
                        nodes.append({
                            "op": "CONCAT",
                            "inputs": inputs,
                            "outputs": [self.safe_name(op.outputs[0].name)],
                            "attrs": {"dim": ne_axis}
                        })
                        continue
                continue

            if op_type == "conv":
                # Map conv to CONV_1D or CONV_2D and extract static attributes
                strides = op.inputs["strides"].val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else [1]
                pad = op.inputs["pad"].val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else [0]
                dilations = op.inputs["dilations"].val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else [1]
                groups = op.inputs["groups"].val if "groups" in op.inputs and hasattr(op.inputs["groups"], "val") else 1
                
                # Format to standard integers
                s0 = int(strides[0]) if isinstance(strides, (list, tuple, np.ndarray)) else int(strides)
                p0 = int(pad[0]) if isinstance(pad, (list, tuple, np.ndarray)) else int(pad)
                d0 = int(dilations[0]) if isinstance(dilations, (list, tuple, np.ndarray)) else int(dilations)
                g_val = int(groups[0]) if isinstance(groups, (list, tuple, np.ndarray)) else int(groups)
                
                # Check if it is a depthwise convolution (groups > 1)
                is_dw = (g_val > 1)
                if is_dw:
                    mapped_op = "CONV_2D_DW" if isinstance(strides, (list, tuple, np.ndarray)) and len(strides) == 2 else "CONV_1D_DW"
                else:
                    mapped_op = "CONV_2D" if isinstance(strides, (list, tuple, np.ndarray)) and len(strides) == 2 else "CONV_1D"
                
                # Extract main inputs [x, weight]
                x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                x_var = self.safe_name(x_var_obj.name)
                weight_var = self.safe_name(op.inputs["weight"].name)
                
                nodes.append({
                    "op": mapped_op,
                    "inputs": [resolve(weight_var), resolve(x_var)],
                    "outputs": [self.safe_name(op.outputs[0].name)],
                    "attrs": {
                        "s0": s0,
                        "p0": p0,
                        "d0": d0,
                        "groups": g_val
                    }
                })
                continue

            mapped_op = self.OP_MAP.get(op_type)
            if mapped_op is None:
                raise NotImplementedError(f"MIL op '{op_type}' is missing a ggml mapping.")

            inputs = []
            if mapped_op == "GET_ROWS":
                # GET_ROWS in Loom C++ strictly expects 2 inputs: [weights, indices]
                # Any 3rd input (like axis) must be pruned.
                x_val_obj = op.inputs.get("x") or op.inputs.get("params")
                indices_val_obj = op.inputs.get("indices")
                if x_val_obj and indices_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name)), resolve(self.safe_name(indices_val_obj.name))]
            elif mapped_op == "MUL_MAT":
                # MUL_MAT strictly expects exactly 2 inputs: [x, y]
                # Any other trailing transpose variables must be pruned.
                x_val_obj = op.inputs.get("x")
                y_val_obj = op.inputs.get("y")
                if x_val_obj and y_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name)), resolve(self.safe_name(y_val_obj.name))]
            elif mapped_op in ("MEAN", "PERMUTE", "SOFTMAX", "CLAMP", "RSQRT", "RESHAPE", "VIEW"):
                # Unary reduction/metadata operations in Loom C++ strictly expect exactly 1 input tensor
                x_val_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                if x_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name))]
            elif mapped_op in ("CONV_1D", "CONV_2D"):
                # Convolutions strictly expect exactly 2 inputs: [x, weight]
                # Strides, padding, dilation, groups are passed as JSON attributes.
                x_val_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                weight_val_obj = op.inputs.get("weight")
                if x_val_obj and weight_val_obj:
                    inputs = [resolve(self.safe_name(x_val_obj.name)), resolve(self.safe_name(weight_val_obj.name))]
            elif op_type in ("add", "mul"):
                # Swap commutative inputs to ensure the larger/dynamic tensor is first,
                # preventing GGML broadcast repetition failures.
                inp1 = resolve(self.safe_name(op.inputs.get("x").name)) if "x" in op.inputs and hasattr(op.inputs["x"], "name") else None
                inp2 = resolve(self.safe_name(op.inputs.get("y").name)) if "y" in op.inputs and hasattr(op.inputs["y"], "name") else None
                if inp1 and inp2:
                    if inp1 in self.weights and inp2 not in self.weights:
                        inputs = [inp2, inp1]
                    else:
                        inputs = [inp1, inp2]
                elif inp1:
                    inputs = [inp1]
                elif inp2:
                    inputs = [inp2]
            else:
                for k, v in op.inputs.items():
                    if isinstance(v, Var):
                        inputs.append(resolve(self.safe_name(v.name)))
                    elif isinstance(v, (list, tuple)):
                        for item in v:
                            if isinstance(item, Var):
                                inputs.append(resolve(self.safe_name(item.name)))

            outputs = [self.safe_name(v.name) for v in op.outputs]

            attrs = {}
            for k, v in op.inputs.items():
                if not isinstance(v, Var) and not isinstance(v, (list, tuple)):
                    if hasattr(v, "val"):
                        attrs[k] = v.val
                    else:
                        attrs[k] = v

            node = {
                "op": mapped_op,
                "inputs": inputs,
                "outputs": outputs
            }
            if attrs:
                # Filter out complex objects
                serializable_attrs = {}
                for ak, av in attrs.items():
                    if isinstance(av, (int, float, str, bool, list, dict)):
                        serializable_attrs[ak] = av
                if serializable_attrs:
                    node["attrs"] = serializable_attrs
            nodes.append(node)

        func_outputs = func.outputs if func else (ops_list[-1].outputs if ops_list else [])
        output_symbol = resolve(self.safe_name(func_outputs[0].name)) if func_outputs else "output"

        return {
            "version": 1,
            "inputs": topo_inputs,
            "output": output_symbol,
            "nodes": self._prune_dead_nodes(nodes, output_symbol)
        }

    def _prune_dead_nodes(self, nodes: list, output_symbol: str) -> list:
        """
        Removes topology nodes whose output is never consumed, directly or transitively, by the
        topology's own declared output. GraphBuilder builds and COMPUTES every node unconditionally
        regardless of whether anything uses its result -- so a subgraph orphaned by a fused/bypassed
        composition (e.g. the GQA repeat_kv() fusion above, which computes the correct REPEAT directly
        from reliable inputs and leaves the original tile's own "reps"-computation subgraph --
        gather/concat/equal/select/div, all now unused -- entirely orphaned) isn't just wasted compute:
        it can still crash. Confirmed empirically: that exact orphaned chain segfaulted during
        ggml_backend_graph_compute despite having zero real consumers, because nothing ever validates an
        unused node's own shapes/values are sane. Keeps only nodes reachable backward from the output.
        """
        needed = {output_symbol}
        live = []
        for node in reversed(nodes):
            if any(o in needed for o in node["outputs"]):
                live.append(node)
                needed.update(node["inputs"])
        live.reverse()
        return live

    def _collect_mul_mat_weight_names(self) -> set:
        """Every MUL_MAT node's *first* input, across all topologies -- the weight-first argument per
        loom's convention (src/ops/primitives_basic.cpp's op_mul_mat is a bare ggml_mul_mat(a, b) wrap
        with `a` as the weight). Mirrors tools/quantize/quantize_gguf_q8_0.py's topology-driven tensor
        selection (proven end-to-end against a real quantized model, see BACKLOG.md's "quantized weight
        support" milestone) rather than tensor-name pattern matching -- this exporter never emits
        "repeat_for" blocks (unlike that script's GGUF-KV-driven input), so no expansion pass is needed."""
        names = set()
        for topo in self.topologies.values():
            for node in topo.get("nodes", []):
                if node.get("op") == "MUL_MAT" and node.get("inputs"):
                    names.add(node["inputs"][0])
        return names

    def _prune_dead_weights(self) -> None:
        """
        Drops any `self.weights` entry never referenced as an input by any surviving topology node.
        `generate_graph_topology`'s "const" handling unconditionally serializes EVERY MIL const op
        (below the >100-element inline threshold) as a GGUF weight tensor -- including incidental
        attribute-only constants (e.g. a "matmul" op's own `transpose_x`/`transpose_y` boolean flags,
        already consumed directly via `.val` in Python by the dedicated "matmul" composition and never
        emitted as a node input reference) that are not real model weights at all and have no consumer
        anywhere in the topology. Confirmed one such orphan (`_80_transpose_x_0`, GGUF-declared with an
        empty `shape=[]`, i.e. a genuine zero-rank scalar tensor) sitting immediately before a real,
        actually-used weight (`const_2_to_fp16`, LFM2's RoPE inverse-frequency table) in tensor-declaration
        order -- and a real, reproducible crash computing a MUL_MAT against that exact real weight,
        consistent with a degenerate zero-byte allocation for the dead scalar corrupting the next
        tensor's backing memory. Pruning these (same principle as `_prune_dead_nodes`) is correct
        regardless of whether it's this crash's full root cause: a weight nothing reads should never be
        written to the GGUF at all.
        """
        referenced = set()
        for topo in self.topologies.values():
            for node in topo.get("nodes", []):
                referenced.update(node.get("inputs", []))
        dead = [name for name in self.weights if name not in referenced]
        for name in dead:
            del self.weights[name]

    def write_gguf(self, driver_script: str):
        from gguf import GGUFWriter
        import os

        self._prune_dead_weights()

        arch = self.kwargs.get("architecture") or os.environ.get("LOOM_ARCH", "mil_model")
        w = GGUFWriter(self.output_path, f"loom-{arch}")
        w.add_string("loom.architecture", arch)

        # Embed the Lua driver orchestration script
        w.add_string("model.driver_script", driver_script)

        # Embed each static submodule topology JSON string
        for submodule_name, topo in self.topologies.items():
            w.add_string(f"model.graph_topology.{submodule_name}", json.dumps(topo, cls=NumpyEncoder))

        qtype = None
        quantizable = set()
        block_size = 1
        if self.quantize:
            from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
            qtype = GGMLQuantizationType[self.quantize]
            block_size, _ = GGML_QUANT_SIZES[qtype]
            quantizable = self._collect_mul_mat_weight_names()
        n_quantized = 0

        # Quantize & write weights / tensors
        for name, array in self.weights.items():
            if array.dtype == bool or array.dtype == np.bool_:
                array = array.astype(np.int32)
            elif np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
            elif array.dtype == np.int64:
                array = array.astype(np.int32)
            elif not np.issubdtype(array.dtype, np.number):
                continue

            # Only MUL_MAT weight tensors are quantized (matching llama.cpp's own convention: norm/bias
            # 1D tensors have negligible size benefit and real accuracy cost). Tensors whose last
            # (fastest-varying) dimension isn't block-aligned are left F32 rather than erroring, same
            # graceful behavior as the standalone quantize_gguf_q8_0.py POC.
            if (qtype is not None and name in quantizable and array.ndim >= 2 and array.dtype == np.float32
                    and array.shape[-1] % block_size == 0):
                from gguf import quants
                q = quants.quantize(np.ascontiguousarray(array), qtype)
                # No `raw_shape` -- add_tensor's raw_shape (when given) is a *byte*-shape fed straight
                # into quant_shape_from_byte_shape, not the pre-quantization logical shape; omitting it
                # lets it default to the quantized array's own (correct) byte-shape.
                w.add_tensor(name, q, raw_dtype=qtype)
                n_quantized += 1
            else:
                w.add_tensor(name, array)

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        suffix = f", {n_quantized} tensor(s) quantized to {self.quantize}" if self.quantize else ""
        print(f"wrote GGUF with driver_script and {len(self.topologies)} topologies to {self.output_path}{suffix}")
