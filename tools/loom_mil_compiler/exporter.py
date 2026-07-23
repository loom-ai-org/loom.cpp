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
from .passes import apply_loom_mil_passes

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

        if not (is_bespoke and self.profile is None):
            # EXPORT-IMPROVEMENT-BACKLOG.md item 3: graph rewrites (currently just GQA repeat_kv()
            # fusion) run as real MIL->MIL passes over the pymil graph here, before any of the workflows
            # below ever walk it -- not interleaved into generate_graph_topology's own translation walk.
            # Skipped for the bespoke/advanced workflow below: that path exists specifically to accept
            # hand-built Programs (see test_compiler.py's MockOperation) that were never traced through
            # ct.convert()'s standard pipeline and may contain synthetic ops standing in for ones MIL
            # itself doesn't have -- a real MIL pass (dead_code_elimination in particular, which insists
            # on internally-consistent var/op child-tracking) isn't meaningful there and isn't safe to run
            # over a graph that was assembled by directly splicing Python op lists rather than through
            # MIL's own block-mutation API.
            apply_loom_mil_passes(self.program)

        if is_bespoke and self.profile is None:
            # 1. Advanced / Bespoke Exporting Workflow
            print("Exporting via Advanced/Bespoke workflow...")
            for func_name, func in self.program.functions.items():
                if func_name == "main":
                    self.transpile_to_lua(func, name="main")
                else:
                    self.topologies[func_name] = self.generate_graph_topology(func, func_name)
            driver_script = self._finalize_driver()
        elif self.kwargs.get("submodule_layout") is not None:
            # Submodule-export blueprint (EXPORT-IMPROVEMENT-BACKLOG.md item 2): `self.program` here is
            # NOT a single flattened trace -- it has no "main" function at all, just one Function per
            # independently-traced submodule (prefix/aux/layer_i/suffix_i) -- so there is no monolithic
            # fallback to degrade to on failure the way "atomic" has; a bug here should fail loudly, not
            # silently produce a working-looking but wrong export.
            self.apply_submodule_export()
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
        #
        # A SubgraphCall only ever exposes ONE slice's output(s) as `last_op.outputs` (see
        # `output_names = ... last_op.outputs` below) -- the single-output-per-topology convention
        # the driver/engine actually supports. So a var is only reachable by a LATER slice via
        # legitimate SubgraphCall input-wiring if its producer op is the designated LAST op of
        # whichever slice originally owns it (e.g. layer_(N-1)'s final hidden_states, threaded into
        # layer_N). Any op that is NOT its own slice's last op -- whether because it's genuinely
        # ungoverned (boundary_key is None, swept into whichever slice happened to be "current" in
        # iteration order purely by happenstance) or because it's a real but non-final interior op
        # of a shared multi-output slice (e.g. a RoPE cos/sin precompute under its own
        # "model_model_pos_emb" scope, where only ONE of the two sibling cos/sin ops can ever be the
        # slice's single exposed output) -- can NEVER be read this way. Any later slice that
        # references such a var's name directly sees it as an "external input" nothing upstream ever
        # actually provides: the mis-attribution bug, previously only caught (not fixed) by
        # validate()'s atomic->monolithic fallback, surfacing as a SubgraphCall reading an input no
        # earlier statement ever defined.
        #
        # Fix: recursively pull each such not-properly-exposed producer (and its own transitive
        # dependencies, stopping at consts or at another op that IS its slice's legitimate exposed
        # output) into EVERY slice that consumes it, instead of leaving it live in only the one slice
        # that happened to inherit it. This is safe to duplicate freely: every such op is a pure
        # function of consts/already-available inputs (that's exactly why it carries no real
        # cross-slice state of its own), so recomputing it per consuming slice is redundant compute,
        # never a correctness change. Any resulting now-unused copy left behind in the original
        # "accidental host" slice is harmless: item 3's `_prune_dead_nodes` already drops any node
        # unreachable from that topology's own declared output.
        exposed_ops = {ops[-1] for _, ops in slices if ops}

        def _is_legitimate_external_ref(op):
            return op in exposed_ops

        # Some MIL ops (e.g. "concat"/"stack") take a LIST of Vars under one input key (e.g.
        # `values`) rather than one Var per key -- generate_graph_topology's own input-extraction
        # already knows to flatten these (see its "elif isinstance(v, (list, tuple))" branch further
        # below); this partitioning code must walk the exact same shape or it silently never visits
        # a list-valued input's real producer op at all. Confirmed as a second, real bug this round:
        # without this, replicating the pos_emb cos/sin closure for an attention layer pulled in the
        # `concat((freqs, freqs))` node (reached via `cos`'s own bare-Var "x" input) but never its
        # OWN `freqs` producer (only reachable through `concat`'s list-valued `values` input), leaving
        # a node referencing a never-defined `freqs` var in the emitted topology.
        def _iter_input_vars(op):
            for v in op.inputs.values():
                if isinstance(v, Var):
                    yield v
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, Var):
                            yield item

        def _collect_replica_closure(op, ops_set, acc, visited):
            if op in ops_set or op in visited:
                return
            visited.add(op)
            for v in _iter_input_vars(op):
                if v.op is None or v in op.outputs:
                    continue
                producer = v.op
                if producer in ops_set or producer in visited:
                    continue
                if producer.op_type == "const":
                    if producer not in acc:
                        acc.append(producer)
                elif not _is_legitimate_external_ref(producer):
                    _collect_replica_closure(producer, ops_set, acc, visited)
                    if producer not in acc:
                        acc.append(producer)

        for idx, (name, ops) in enumerate(slices):
            ops_set = set(ops)
            extra = []
            visited = set()
            for op in ops:
                for v in _iter_input_vars(op):
                    if v.op and v not in op.outputs:
                        producer = v.op
                        if producer in ops_set:
                            continue
                        if producer.op_type == "const":
                            if producer not in extra:
                                extra.append(producer)
                        elif not _is_legitimate_external_ref(producer):
                            _collect_replica_closure(producer, ops_set, extra, visited)
                            if producer not in extra:
                                extra.append(producer)
            local_ops = (extra + list(ops)) if extra else list(ops)
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
                for v in _iter_input_vars(op):
                    if v not in op.outputs and v.op not in ops_set:
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

    def apply_submodule_export(self):
        """
        Synthesizes the driver for a submodule-export blueprint (`kwargs["submodule_layout"]`, a
        `SubmoduleExportResult` from submodule_export.py): one real, independently-traced Function per
        prefix/aux/layer_i/suffix_i submodule. Every function is self-contained by construction (no
        cross-slice variable leakage to detect, unlike apply_atomic_export's scope-partitioned single
        trace), so this only has to generate each function's topology directly
        (`generate_graph_topology(func, name)`, no ops_list/inputs_dict reconstruction) and chain
        SubgraphCalls prefix -> [aux] -> layer_0..N-1 -> suffix_0..M-1 -> argmax.
        """
        print("Exporting via Submodule-Blueprint path...")
        layout = self.kwargs["submodule_layout"]
        functions = self.program.functions
        special_names = set(_POSITION_INPUT_NAMES) | set(_CAUSAL_MASK_INPUT_NAMES)

        def is_aux_input(name):
            return layout.aux_kwarg and (name == layout.aux_kwarg or name.startswith(layout.aux_kwarg + "_"))

        has_aux = layout.aux_output_names is not None
        stage_names = ["prefix"] + (["aux"] if has_aux else []) \
            + [f"layer_{i}" for i in range(layout.num_layers)] + layout.suffix_names

        # 1. Generate topologies for every submodule function. generate_graph_topology drops any
        # declared input no node actually reads (post dead-node-pruning) -- e.g. LFM2's conv-type
        # layers never touch `position_embeddings`, only its attention-type layers do, so which inputs
        # survive can differ PER LAYER even though they were traced with an identical call signature.
        # Wiring below must therefore consult each stage's own post-filter declared inputs, never a
        # single shared name list.
        for name in stage_names:
            self.topologies[name] = self.generate_graph_topology(functions[name], name)

        def declared_inputs(name):
            return [inp["name"] for inp in self.topologies[name]["inputs"]]

        prefix_input_names = declared_inputs("prefix")
        chain_in_names = [n for n in prefix_input_names if n not in special_names]
        if len(chain_in_names) != 1:
            raise ValueError(f"prefix submodule must declare exactly one non-special input, got {prefix_input_names}")
        first_input = self.safe_name(chain_in_names[0])
        n_tokens_expr = Len(first_input)

        # 2. `first_input` (the caller-supplied token-ids input) must be defined before anything below
        # reads n_tokens_expr, mirroring apply_monolithic_export/apply_atomic_export's own ordering.
        body = [Local(first_input, BinOp("or", FieldAccess("inputs", first_input), FieldAccess("inputs", "tokens")))]

        special_needed = set()
        for name in stage_names:
            special_needed.update(n for n in declared_inputs(name) if n in special_names)
        for name in sorted(special_needed):
            safe_inp = self.safe_name(name)
            if name in _POSITION_INPUT_NAMES:
                body.append(Local(safe_inp, Call("loom.range", [Lit(0), n_tokens_expr])))
            else:
                body.append(Local(safe_inp, Call("loom.causal_mask", [n_tokens_expr, Lit(0)])))

        # 3. Prefix.
        chain_var = "_sub_chain_0"
        body.append(SubgraphCall(
            outputs=[chain_var], module="prefix", n_tokens=n_tokens_expr, n_past=Lit(0),
            inputs={self.safe_name(n): IRVar(self.safe_name(n)) for n in prefix_input_names},
        ))

        # 4. Auxiliary submodule (computed once, shared across every repeated-block call below) -- e.g.
        # LFM2's rotary-embedding table, computed once in Lfm2Model.forward and threaded into every
        # decoder layer as `position_embeddings=(cos, sin)`.
        aux_out_vars = None
        if has_aux:
            aux_input_names = declared_inputs("aux")
            aux_chain_names = [n for n in aux_input_names if n not in special_names]
            if len(aux_chain_names) > 1:
                raise ValueError(f"aux submodule must declare at most one non-special input, got {aux_input_names}")
            aux_inputs_tbl = {}
            for n in aux_input_names:
                safe_n = self.safe_name(n)
                # The aux submodule's own non-special input (if it has one -- it may not, e.g. LFM2's
                # pos_emb is called with the real hidden_states value purely for its dtype/device, which
                # the traced graph never ends up actually depending on) plays the same "current chain
                # tensor" role a repeated-block call's primary input does -- feed it the same value.
                aux_inputs_tbl[safe_n] = IRVar(chain_var) if n in aux_chain_names else IRVar(safe_n)
            aux_out_vars = [f"_sub_aux_{i}" for i in range(len(layout.aux_output_names))]
            body.append(SubgraphCall(
                outputs=aux_out_vars, module="aux", n_tokens=n_tokens_expr, n_past=Lit(0), inputs=aux_inputs_tbl,
            ))

        # 5. Repeated block, threading `chain_var` (hidden_states) from one layer's output into the
        # next's input, exactly like apply_atomic_export's own layer chain. Each layer's OWN declared
        # inputs are consulted independently (see the comment on step 1).
        for i in range(layout.num_layers):
            layer_name = f"layer_{i}"
            layer_inputs = declared_inputs(layer_name)
            chain_names_i = [n for n in layer_inputs if n not in special_names and not is_aux_input(n)]
            if len(chain_names_i) != 1:
                raise ValueError(f"repeated block must declare exactly one non-special, non-aux input, got {layer_inputs}")
            chain_name_i = chain_names_i[0]

            inputs_tbl = {}
            for n in layer_inputs:
                safe_n = self.safe_name(n)
                if n == chain_name_i:
                    inputs_tbl[safe_n] = IRVar(chain_var)
                elif is_aux_input(n):
                    idx = 0 if n == layout.aux_kwarg else int(n[len(layout.aux_kwarg) + 1:])
                    inputs_tbl[safe_n] = IRVar(aux_out_vars[idx])
                else:
                    inputs_tbl[safe_n] = IRVar(safe_n)

            next_chain_var = f"_sub_chain_{i + 1}"
            body.append(SubgraphCall(
                outputs=[next_chain_var], module=layer_name, n_tokens=n_tokens_expr, n_past=Lit(0),
                inputs=inputs_tbl,
            ))
            chain_var = next_chain_var

        # 6. Suffix chain (e.g. final norm + lm_head).
        for idx, name in enumerate(layout.suffix_names):
            in_names = declared_inputs(name)
            if len(in_names) != 1:
                raise ValueError(f"suffix submodule '{name}' must declare exactly one input, got {in_names}")
            is_last = idx == len(layout.suffix_names) - 1
            next_chain_var = f"_sub_suffix_{idx}"
            call_kwargs = dict(
                outputs=[next_chain_var], module=name, n_tokens=n_tokens_expr, n_past=Lit(0),
                inputs={self.safe_name(in_names[0]): IRVar(chain_var)},
            )
            if is_last:
                call_kwargs["extra_outputs"] = ["_submodule_final_shape"]
            body.append(SubgraphCall(**call_kwargs))
            chain_var = next_chain_var

        # 7. Argmax the logits row for the active (last real) token -- same convention
        # apply_monolithic_export/apply_atomic_export use for causal-LM next-token generation.
        row_expr = BinOp("-", n_tokens_expr, Lit(1))
        body.append(If(
            cond=BinOp("==", Call("type", [IRVar(chain_var)]), Lit("table")),
            then=[Return([Call("loom.argmax_row", [IRVar(chain_var), Index(IRVar("_submodule_final_shape"), 1), row_expr])])],
            else_=[Return([IRVar(chain_var)])],
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

        if op_type in ("random_normal", "random_uniform"):
            # Unlike generate_graph_topology's own static-topology walk (which can NEVER satisfy a
            # random op -- see that function's own comment), this driver-level walk CAN: `main`'s body
            # is genuine Lua, already calling host functions mid-script (range/causal_mask above), so a
            # random op here maps directly onto the existing loom.gaussian_array/loom.uniform_array host
            # RNG functions (src/core/lua_bridge.cpp) real hand-written drivers already use.
            shape_var = op.inputs.get("shape")
            if shape_var is None or not hasattr(shape_var, "val") or shape_var.val is None:
                raise NotImplementedError(
                    f"{op_type} op '{op.name}' has a non-constant 'shape' input, which this exporter "
                    "doesn't support (the sample count passed to loom.gaussian_array/uniform_array must "
                    "be a compile-time constant)."
                )
            n = 1
            for d in shape_var.val:
                n *= int(d)

            if op_type == "random_normal":
                mean_var, stddev_var = op.inputs.get("mean"), op.inputs.get("stddev")
                mean = float(mean_var.val) if mean_var is not None and hasattr(mean_var, "val") and mean_var.val is not None else 0.0
                stddev = float(stddev_var.val) if stddev_var is not None and hasattr(stddev_var, "val") and stddev_var.val is not None else 1.0
                if mean != 0.0 or stddev != 1.0:
                    raise NotImplementedError(
                        f"random_normal op '{op.name}' has mean={mean}/stddev={stddev} -- this exporter "
                        "only supports the standard N(0,1) case loom.gaussian_array itself draws "
                        "(compose an explicit MUL/ADD after it for any other mean/stddev)."
                    )
                return [Local(output_names[0], Call("loom.gaussian_array", [Lit(n)]))]

            low_var, high_var = op.inputs.get("low"), op.inputs.get("high")
            low = float(low_var.val) if low_var is not None and hasattr(low_var, "val") and low_var.val is not None else 0.0
            high = float(high_var.val) if high_var is not None and hasattr(high_var, "val") and high_var.val is not None else 1.0
            if low != 0.0 or high != 1.0:
                raise NotImplementedError(
                    f"random_uniform op '{op.name}' has low={low}/high={high} -- this exporter only "
                    "supports the standard U(0,1) case loom.uniform_array itself draws (compose an "
                    "explicit MUL/ADD after it for any other range)."
                )
            return [Local(output_names[0], Call("loom.uniform_array", [Lit(n)]))]

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

    def generate_graph_topology(self, func: Function, func_name: str, ops_list=None, inputs_dict=None) -> dict:
        """
        Walks a heavy submodule MIL graph and serializes it to a static graph topology.
        """
        nodes = []
        topo_inputs = []
        aliases = {}
        
        inputs = inputs_dict if inputs_dict is not None else (func.inputs if func else {})
        operations = ops_list if ops_list is not None else (func.operations if func else [])

        # EXPORT-IMPROVEMENT-BACKLOG.md item 4: an "lstm" op can't be translated into ordinary static
        # topology nodes the way every other op_type below is -- ggml has no native LSTM/GRU op, and
        # unlike e.g. `linear`/`matmul`, correct recurrence needs a genuine host-side per-timestep loop
        # (see recurrent.py's own module docstring and tools/loom_mil_compiler/test_recurrent.py's
        # verified topology-generation logic), not a fixed sequence of graph nodes. `generate_graph_topology`
        # only ever returns ONE static topology for the whole `operations` list it's given; splitting a
        # function at an "lstm" op boundary into "pre-LSTM topology -> recurrent stepper call ->
        # post-LSTM topology" driver segments -- the way apply_atomic_export splits at torch-scope
        # boundaries -- is real, unimplemented follow-up work, not something this call can do safely.
        # Fail loudly and specifically here rather than either silently mistranslating (there's no OP_MAP
        # entry for "lstm" so it would otherwise hit the generic "missing a ggml mapping" message, which
        # doesn't explain why or point at the real fix) or leaving a caller to guess.
        if any(op.op_type in ("lstm", "gru") for op in operations):
            raise NotImplementedError(
                f"generate_graph_topology('{func_name}'): contains an '{next(op.op_type for op in operations if op.op_type in ('lstm', 'gru'))}' "
                "op. ggml has no native LSTM/GRU op -- correct recurrence needs a real per-timestep "
                "stepper (see tools/loom_mil_compiler/recurrent.py and the new "
                "LoomLuaBridge::l_run_recurrent C++ binding, which generalizes the existing "
                "BiLstmStepper), not a static topology. Auto-wiring this into the generic export "
                "profiles (monolithic/atomic/submodule-blueprint driver synthesis) is unimplemented "
                "follow-up work -- for now, build the per-timestep topologies directly via "
                "recurrent.build_lstm_cell_topologies() and drive them with loom.run_recurrent() in a "
                "hand-written driver script, the same way tools/convert_kokoro/kokoro_driver.lua does "
                "today via BiLstmStepper."
            )

        def resolve(name):
            seen = set()
            while name in aliases and name not in seen:
                seen.add(name)
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
                # A submodule traced standalone with its real parameter name already literally
                # "hidden_states" (e.g. the submodule-export blueprint, EXPORT-IMPROVEMENT-BACKLOG.md
                # item 2) would otherwise alias it to itself here -- `resolve()` walks `aliases` in a
                # `while name in aliases` loop, so a self-referential entry spins forever.
                if orig_name != "hidden_states":
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

            if op_type == "pad":
                # EXPORT-IMPROVEMENT-BACKLOG.md item 4: the one op type STFT's own center-framing
                # (torch.stft(..., center=True)) decomposes into, via coremltools' own
                # common::lower_complex_dialect_ops pass, that this exporter didn't already handle --
                # confirmed by directly tracing a small torch.stft module through the standard
                # ct.convert(..., convert_to="milinternal") pipeline (with thread 1's
                # apply_torch_frontend_patches() active) and inspecting the resulting MIL ops: the whole
                # decomposition is const/cast/reshape/conv/expand_dims/pad, every one of which this
                # exporter already covers except this. Only PAD_1D/PAD_1D_REFLECT (ne[0]-only, i.e. MIL's
                # LAST axis) exist in C++ -- anything padding a different axis, or a non-zero constant
                # fill, or a mode other than constant/reflect, has no ggml-side implementation yet.
                x_var_obj = op.inputs.get("x") or op.inputs.get("data")
                x_var = self.safe_name(x_var_obj.name)
                output_var = self.safe_name(op.outputs[0].name)

                pad_var = op.inputs.get("pad")
                if pad_var is None or not hasattr(pad_var, "val") or pad_var.val is None:
                    raise NotImplementedError(
                        f"pad op '{op.name}' has a non-constant 'pad' input, which this exporter doesn't support."
                    )
                pad_vals = [int(v) for v in pad_var.val]
                if len(pad_vals) % 2 != 0:
                    raise NotImplementedError(f"pad op '{op.name}' has an odd-length 'pad' array {pad_vals!r}.")
                n_padded = len(pad_vals) // 2

                if not hasattr(x_var_obj, "shape") or x_var_obj.shape is None:
                    raise NotImplementedError(f"pad op '{op.name}' has an input with no known rank.")
                rank = len(x_var_obj.shape)

                mode_var = op.inputs.get("mode")
                mode = (mode_var.val if mode_var is not None and hasattr(mode_var, "val")
                        and mode_var.val is not None else "constant")

                # MIL's "pad" only pads the LAST n_padded dims of x: pad[2*i]/pad[2*i+1] apply to MIL
                # axis (rank - n_padded + i). Only a non-zero pad on the FASTEST-varying axis (ne[0],
                # i.e. MIL's last axis) is supported -- exactly what PAD_1D/PAD_1D_REFLECT's ggml kernels
                # (and every real case seen so far, STFT included) operate on.
                lp0 = rp0 = 0
                for i in range(n_padded):
                    mil_axis = rank - n_padded + i
                    lp, rp = pad_vals[2 * i], pad_vals[2 * i + 1]
                    if lp == 0 and rp == 0:
                        continue
                    if mil_axis != rank - 1:
                        raise NotImplementedError(
                            f"pad op '{op.name}' pads MIL axis {mil_axis} (non-zero {lp}/{rp}), but this "
                            "exporter only supports padding the fastest-varying axis (ne[0]/MIL's last "
                            "axis) -- padding any other axis needs a new C++ primitive first."
                        )
                    lp0, rp0 = lp, rp

                if mode == "constant":
                    constant_val_var = op.inputs.get("constant_val")
                    constant_val = (float(constant_val_var.val)
                                     if constant_val_var is not None and hasattr(constant_val_var, "val")
                                     and constant_val_var.val is not None else 0.0)
                    if constant_val != 0.0:
                        raise NotImplementedError(
                            f"pad op '{op.name}' has mode='constant' with a non-zero constant_val="
                            f"{constant_val} -- PAD_1D only supports zero-fill."
                        )
                    mapped_op = "PAD_1D"
                elif mode == "reflect":
                    mapped_op = "PAD_1D_REFLECT"
                else:
                    raise NotImplementedError(
                        f"pad op '{op.name}' has mode='{mode}', which this exporter doesn't support "
                        "(only 'constant' and 'reflect' are)."
                    )

                nodes.append({
                    "op": mapped_op,
                    "inputs": [resolve(x_var)],
                    "outputs": [output_var],
                    "attrs": {"lp0": lp0, "rp0": rp0},
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
                # GQA repeat_kv() fusion (the case where this "tile"'s own `reps` isn't a compile-time
                # constant, which otherwise poisons this node's -- and the following reshape's -- shape
                # inference) is now handled as a real MIL->MIL pass in passes.py, run once over the whole
                # program before this walk ever starts (EXPORT-IMPROVEMENT-BACKLOG.md item 3). By the time
                # a "tile" op reaches this generic per-op handling, either it was never part of that
                # pattern, or it already got rewritten into a plain reshape(shape, 1)->tile(ratio)->reshape
                # sequence with fully reliable (concrete-int-or-single-dynamic-symbol) shapes -- so no
                # special-casing is needed here anymore.
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
                    
                    if rep_factor == 1:
                        # A rep factor of 1 is a no-op regardless of dim_size's type -- skip the
                        # multiplication wrapper entirely rather than emitting a redundant "(dim * 1)"
                        # expression string for a dynamic dim_size (harmless once evaluated by
                        # SymbolEnv, but needlessly opaque, and the GQA repeat_kv() fusion pass in
                        # passes.py routinely produces exactly this shape -- reps=1 on every
                        # unchanged axis, ratio only on the newly-inserted one).
                        target_shape.append(dim_size)
                        continue
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

            if op_type == "conv_transpose":
                # Map conv_transpose to CONV_TRANSPOSE_1D/2D. Loom's C++ primitives
                # (src/ops/primitives_conv.cpp's op_conv_transpose_1d/2d, backing ggml_conv_transpose_1d /
                # ggml_conv_transpose_2d_p0) only ever support stride + zero padding + no dilation + no
                # grouping -- the plain "valid, groups=1" case every real use so far needs (this exporter's
                # own ISTFT module, tools/loom_mil_compiler/istft.py, and Kokoro's hand-built ISTFT/Generator
                # upsampling both only ever call it this way). Anything else raises rather than silently
                # dropping the extra configuration, matching this file's own "conv"/"matmul" convention.
                strides = op.inputs["strides"].val if "strides" in op.inputs and hasattr(op.inputs["strides"], "val") else [1]
                pad = op.inputs["pad"].val if "pad" in op.inputs and hasattr(op.inputs["pad"], "val") else [0]
                dilations = op.inputs["dilations"].val if "dilations" in op.inputs and hasattr(op.inputs["dilations"], "val") else [1]
                groups = op.inputs["groups"].val if "groups" in op.inputs and hasattr(op.inputs["groups"], "val") else 1
                output_shape = op.inputs.get("output_shape")
                bias_var = op.inputs.get("bias")
                pad_type_var = op.inputs.get("pad_type")
                pad_type = (pad_type_var.val if pad_type_var is not None and hasattr(pad_type_var, "val")
                            and pad_type_var.val is not None else "valid")

                if bias_var is not None and getattr(bias_var, "val", None) is not None and np.any(bias_var.val):
                    raise NotImplementedError(
                        f"conv_transpose op '{op.name}' has a non-zero 'bias', which this exporter "
                        "doesn't support (no bias-add exists in ggml_conv_transpose_1d/2d) -- compose an "
                        "explicit ADD node after CONV_TRANSPOSE instead."
                    )
                if pad_type != "valid":
                    raise NotImplementedError(
                        f"conv_transpose op '{op.name}' has pad_type='{pad_type}', which this exporter "
                        "doesn't support (ggml_conv_transpose_1d/2d only implement 'valid'-style padding)."
                    )
                # Torch-traced conv_transpose always carries an explicit 'output_shape' even for the
                # ordinary case (confirmed directly) -- it's redundant with pad_type='valid''s own formula
                # (Dout = (D_in-1)*stride + K), not a real extra constraint, so it's fine to ignore rather
                # than reject; just sanity-check it agrees rather than trusting it blindly.
                if output_shape is not None and getattr(output_shape, "val", None) is not None:
                    x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                    weight_var_obj = op.inputs["weight"]
                    strides_list = list(strides) if isinstance(strides, (list, tuple, np.ndarray)) else [strides]
                    n_spatial = len(strides_list)
                    in_spatial = list(x_var_obj.shape[-n_spatial:])
                    kernel_spatial = list(weight_var_obj.shape[-n_spatial:])
                    expected = [(int(d) - 1) * int(s) + int(k) for d, s, k in zip(in_spatial, strides_list, kernel_spatial)]
                    actual = [int(v) for v in output_shape.val[-n_spatial:]]
                    if expected != actual:
                        raise NotImplementedError(
                            f"conv_transpose op '{op.name}' declares output_shape={list(output_shape.val)!r}, "
                            f"which doesn't match the 'valid'-padding formula's own {expected!r} -- this "
                            "combination isn't supported."
                        )
                if any(int(p) != 0 for p in (pad if isinstance(pad, (list, tuple, np.ndarray)) else [pad])):
                    raise NotImplementedError(
                        f"conv_transpose op '{op.name}' has non-zero 'pad' {pad!r}, which this exporter "
                        "doesn't support (ggml_conv_transpose_1d/2d are zero-padding only)."
                    )
                if any(int(d) != 1 for d in (dilations if isinstance(dilations, (list, tuple, np.ndarray)) else [dilations])):
                    raise NotImplementedError(
                        f"conv_transpose op '{op.name}' has non-unit 'dilations' {dilations!r}, which "
                        "this exporter doesn't support."
                    )
                g_val = int(groups[0]) if isinstance(groups, (list, tuple, np.ndarray)) else int(groups)
                if g_val != 1:
                    raise NotImplementedError(
                        f"conv_transpose op '{op.name}' has groups={g_val}, which this exporter doesn't "
                        "support (no depthwise/grouped CONV_TRANSPOSE primitive exists)."
                    )

                s0 = int(strides[0]) if isinstance(strides, (list, tuple, np.ndarray)) else int(strides)
                mapped_op = "CONV_TRANSPOSE_2D" if isinstance(strides, (list, tuple, np.ndarray)) and len(strides) == 2 else "CONV_TRANSPOSE_1D"

                x_var_obj = op.inputs.get("x") or op.inputs.get("data") or op.inputs.get("input")
                x_var = self.safe_name(x_var_obj.name)
                weight_var = self.safe_name(op.inputs["weight"].name)

                nodes.append({
                    "op": mapped_op,
                    "inputs": [resolve(weight_var), resolve(x_var)],
                    "outputs": [self.safe_name(op.outputs[0].name)],
                    "attrs": {"s0": s0}
                })
                continue

            if op_type in ("random_normal", "random_uniform", "random_bernoulli", "random_categorical"):
                # EXPORT-IMPROVEMENT-BACKLOG.md item 4: ggml has no RNG-capable compute op at all (no
                # ggml_rand* of any kind exists) -- this isn't a missing-ggml-mapping gap the way most
                # other NotImplementedError cases in this file are, it's a hard architectural boundary:
                # a static per-submodule topology (what this function produces) is a pure, deterministic
                # dataflow graph GraphBuilder builds once and reuses; there is no way to make one of its
                # nodes produce fresh randomness on each call. Fail with a message pointing at the actual
                # fix (already-existing infrastructure, not something new to build) rather than the
                # generic "missing a ggml mapping" message below, which would suggest this is just an
                # unimplemented translation rather than something this function can never do.
                raise NotImplementedError(
                    f"MIL op '{op_type}' (from '{op.name}') can't be translated into a static topology "
                    "node -- ggml has no RNG-capable compute op. The fix is to hoist whatever "
                    "torch.randn()/torch.rand()-style call produced this op OUTSIDE this submodule's own "
                    "trace boundary (e.g. via a 'prefix'/'aux' split in a SubmoduleExportSpec, see "
                    "submodule_export.py) and feed the pre-sampled noise in as an explicit input instead "
                    "-- sampled at runtime via the existing loom.gaussian_array/loom.uniform_array/"
                    "loom.seed_rng Lua host functions (src/core/lua_bridge.cpp), the same host-side-RNG "
                    "pattern every hand-written driver (VitsDriver/SupertonicDriver/MatchaDriver/...) "
                    "already uses."
                )

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

        pruned_nodes = self._prune_dead_nodes(nodes, output_symbol)

        # A declared input with no node (post-pruning) that actually reads it is unreachable from this
        # topology's own output -- GraphBuilder's ggml_gallocr_alloc_graph only allocates a backend
        # buffer for tensors reachable from the declared output, so an orphan input tensor is created
        # but never given a buffer. If the driver ever supplied a value for it anyway (e.g. a
        # submodule-export blueprint's per-layer function still nominally "declaring" a
        # cache_position/position_ids input that its own real call never ends up depending on once
        # past_key_values is forced to None -- see submodule_export.py's _CACHE_KWARG_NAMES comment),
        # setting data into that unallocated tensor is a ggml hard-crash ("tensor buffer not set"), not
        # a graceful error. Dropping it from the declared list here means check_subgraph_calls treats
        # the driver still supplying it as an undeclared-input validation error instead -- catching the
        # mismatch at export time rather than as a runtime crash.
        referenced = {name for node in pruned_nodes for name in node["inputs"]}
        topo_inputs = [inp for inp in topo_inputs if inp["name"] in referenced]

        return {
            "version": 1,
            "inputs": topo_inputs,
            "output": output_symbol,
            "nodes": pruned_nodes
        }

    def _prune_dead_nodes(self, nodes: list, output_symbol: str) -> list:
        """
        Removes topology nodes whose output is never consumed, directly or transitively, by the
        topology's own declared output. GraphBuilder builds and COMPUTES every node unconditionally
        regardless of whether anything uses its result -- so an orphaned subgraph isn't just wasted
        compute, it can still crash (confirmed empirically on an orphaned chain that segfaulted during
        ggml_backend_graph_compute despite having zero real consumers, because nothing ever validates an
        unused node's own shapes/values are sane). Keeps only nodes reachable backward from the output.

        The GQA repeat_kv() fusion's own orphaned dependency chain (the original tile's now-unused
        "reps"-computation subgraph -- gather/concat/equal/select/div) no longer needs this: that fusion
        now runs as a real MIL->MIL pass (passes.py) with `common::dead_code_elimination` run right after
        it, so the orphan never survives into this walk at all (EXPORT-IMPROVEMENT-BACKLOG.md item 3).
        What this still exists for: `apply_atomic_export`'s own `_collect_replica_closure` deliberately
        replicates a producer op (and its transitive dependencies) into EVERY slice that consumes it,
        which leaves a now-unused copy behind in whichever slice originally "accidentally hosted" it --
        see that method's own comment. That's a Python-level list-slicing artifact of partitioning a
        single flattened trace after the fact, not something any MIL-level pass over the pre-partitioned
        `main` function could see or clean up, so this backward-reachability walk is still required for
        the atomic profile specifically.
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

    def _write_tokenizer(self, w, tokenizer_dir: str):
        """Dispatches to the right vocab writer for `tokenizer_dir`'s real HF tokenizer family, auto-
        detecting both the family ("bpe"/"wordpiece"/"sentencepiece_proto") and, for "bpe", the
        pretokenizer regex shape (`tokenizer.ggml.pre`) unless explicitly overridden via
        `tokenizer_family`/`tokenizer_pre` kwargs -- see tokenizer_detect.py's own module docstring for
        the detection recipes."""
        from .tokenizer_detect import detect_vocab_family, detect_loom_pre_type

        family = self.kwargs.get("tokenizer_family") or detect_vocab_family(tokenizer_dir)

        if family == "bpe":
            from .bpe_tokenizer_export import write_bpe_vocab
            pre_type = self.kwargs.get("tokenizer_pre")
            if pre_type is None:
                from transformers import AutoTokenizer
                pre_type = detect_loom_pre_type(AutoTokenizer.from_pretrained(tokenizer_dir))
            write_bpe_vocab(w, tokenizer_dir, pre_type=pre_type)
        elif family == "wordpiece":
            from .wordpiece_tokenizer_export import write_wordpiece_vocab
            write_wordpiece_vocab(w, tokenizer_dir)
        elif family == "sentencepiece_proto":
            from pathlib import Path
            # Every entry point that imports loom_mil_compiler (export_hf_causal_lm.py,
            # export_lfm2_*.py) inserts tools/ itself (not its parent) onto sys.path, so convert_nemo/ is
            # importable as a top-level package the same way loom_mil_compiler is -- not "tools.convert_nemo".
            from convert_nemo.tokenizer_common import write_sentencepiece_vocab
            proto_path = next(p for p in (Path(tokenizer_dir) / "tokenizer.model",
                                           Path(tokenizer_dir) / "spiece.model") if p.exists())
            write_sentencepiece_vocab(w, proto_path.read_bytes())
        elif family == "byte":
            from .byt5_tokenizer_export import write_byt5_vocab
            write_byt5_vocab(w, tokenizer_dir)
        else:
            raise NotImplementedError(f"_write_tokenizer: no vocab writer for tokenizer family {family!r}")

    def write_gguf(self, driver_script: str):
        from gguf import GGUFWriter
        import os

        self._prune_dead_weights()

        arch = self.kwargs.get("architecture") or os.environ.get("LOOM_ARCH", "mil_model")
        w = GGUFWriter(self.output_path, f"loom-{arch}")
        w.add_string("loom.architecture", arch)

        tokenizer_dir = self.kwargs.get("tokenizer_dir") or os.environ.get("LOOM_TOKENIZER_DIR")
        if tokenizer_dir:
            self._write_tokenizer(w, tokenizer_dir)

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
