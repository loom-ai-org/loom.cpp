"""A small intermediate representation for the embedded Lua driver script.

Exporters build a `Function` out of the node types below instead of concatenating raw Lua text while
deciding what the driver should do. `validate()` walks the IR mechanically checking that every symbol is
defined before it's read, and `check_subgraph_calls()` cross-checks each `loom.run_subgraph()` call's
declared inputs against the target topology's own declared inputs -- both classes of bug (undefined-symbol
use, spurious/mismatched subgraph inputs) that previously only surfaced as a runtime crash or silently
wrong output. `LuaCodegen` is the only place that knows Lua's concrete syntax.
"""
from __future__ import annotations

import dataclasses
from typing import Any


class DriverIRError(Exception):
    pass


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class Expr:
    def reads(self) -> list[str]:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


@dataclasses.dataclass
class Var(Expr):
    name: str

    def reads(self) -> list[str]:
        return [self.name]

    def render(self) -> str:
        return self.name


@dataclasses.dataclass
class Lit(Expr):
    value: Any  # int/float/bool/str

    def reads(self) -> list[str]:
        return []

    def render(self) -> str:
        v = self.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)


@dataclasses.dataclass
class RawExpr(Expr):
    """Escape hatch for Lua literal text this IR doesn't model further (e.g. array constructors)."""
    text: str

    def reads(self) -> list[str]:
        return []

    def render(self) -> str:
        return self.text


@dataclasses.dataclass
class Len(Expr):
    var: str

    def reads(self) -> list[str]:
        return [self.var]

    def render(self) -> str:
        return f"#{self.var}"


@dataclasses.dataclass
class BinOp(Expr):
    op: str  # '+', '-', '*', '/', 'floordiv', '==', 'or', 'and', ...
    left: Expr
    right: Expr

    def reads(self) -> list[str]:
        return self.left.reads() + self.right.reads()

    def render(self) -> str:
        if self.op == "floordiv":
            return f"math.floor({self.left.render()} / {self.right.render()})"
        return f"({self.left.render()} {self.op} {self.right.render()})"


@dataclasses.dataclass
class UnaryOp(Expr):
    op: str  # 'not', '-'
    operand: Expr

    def reads(self) -> list[str]:
        return self.operand.reads()

    def render(self) -> str:
        return f"{self.op} {self.operand.render()}"


@dataclasses.dataclass
class FieldAccess(Expr):
    table: str
    field: str

    def reads(self) -> list[str]:
        return [self.table]

    def render(self) -> str:
        return f"{self.table}.{self.field}"


@dataclasses.dataclass
class Index(Expr):
    table: Expr
    idx: int  # 1-based Lua index

    def reads(self) -> list[str]:
        return self.table.reads()

    def render(self) -> str:
        return f"{self.table.render()}[{self.idx}]"


@dataclasses.dataclass
class Call(Expr):
    fn: str
    args: list  # list[Expr]

    def reads(self) -> list[str]:
        out: list[str] = []
        for a in self.args:
            out.extend(a.reads())
        return out

    def render(self) -> str:
        return f"{self.fn}({', '.join(a.render() for a in self.args)})"


@dataclasses.dataclass
class TableLit(Expr):
    items: dict  # str -> Expr

    def reads(self) -> list[str]:
        out: list[str] = []
        for v in self.items.values():
            out.extend(v.reads())
        return out

    def render(self) -> str:
        parts = [f"{k} = {v.render()}" for k, v in self.items.items()]
        return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

class Stmt:
    def defines(self) -> list[str]:
        return []

    def reads(self) -> list[str]:
        return []


@dataclasses.dataclass
class Local(Stmt):
    name: str
    expr: Expr

    def defines(self) -> list[str]:
        return [self.name]

    def reads(self) -> list[str]:
        return self.expr.reads()


@dataclasses.dataclass
class LocalDecl(Stmt):
    """`local name` with no initializer -- declares a name in the ENCLOSING scope so a later Assign
    inside an If/While branch can write to it and have the value survive past that branch's own Lua
    scope (a bare `local` inside an if/else arm would not: Lua locals are block-scoped)."""
    name: str

    def defines(self) -> list[str]:
        return [self.name]


@dataclasses.dataclass
class Assign(Stmt):
    """`name = expr` (no `local`) -- assigns an ALREADY-declared variable (see LocalDecl). Reads `name`
    itself (in addition to expr's own reads) purely so validate() enforces it was declared by an earlier
    LocalDecl/Local -- assigning to an undeclared name would silently create a stray Lua global."""
    name: str
    expr: Expr

    def reads(self) -> list[str]:
        return [self.name] + self.expr.reads()


@dataclasses.dataclass
class SubgraphCall(Stmt):
    """`local out1, out2 = loom.run_subgraph(module, {axis = expr, ...}, {k = v, ...})`.

    `axes` replaces the old fixed `n_tokens`/`n_past` positional pair (EXPORT-ROADMAP.md R1): a
    topology declares its own axis names now (`axes.py`'s N_SAMPLES/N_ENC_FRAMES/N_TOKENS/...), so the
    driver call binds whatever that specific topology actually needs -- `{n_tokens=..., n_past=...}`
    for the ordinary LLM/token-sequence case, `{n_samples=...}` for Conformer-CTC/Parakeet, etc."""
    outputs: list
    module: str
    axes: dict  # str (axis name) -> Expr
    inputs: dict  # str -> Expr
    extra_outputs: list = dataclasses.field(default_factory=list)

    def defines(self) -> list[str]:
        return list(self.outputs) + list(self.extra_outputs)

    def reads(self) -> list[str]:
        out: list[str] = []
        for e in self.axes.values():
            out.extend(e.reads())
        for v in self.inputs.values():
            out.extend(v.reads())
        return out


@dataclasses.dataclass
class Argmax(Stmt):
    """`local result = loom.argmax_row(tensor, n_vocab, row)`."""
    result: str
    tensor: str
    n_vocab: Expr
    row: Expr

    def defines(self) -> list[str]:
        return [self.result]

    def reads(self) -> list[str]:
        return [self.tensor] + self.n_vocab.reads() + self.row.reads()


@dataclasses.dataclass
class Return(Stmt):
    exprs: list  # list[Expr]

    def reads(self) -> list[str]:
        out: list[str] = []
        for e in self.exprs:
            out.extend(e.reads())
        return out


@dataclasses.dataclass
class If(Stmt):
    cond: Expr
    then: list  # list[Stmt]
    else_: list = dataclasses.field(default_factory=list)

    def reads(self) -> list[str]:
        return self.cond.reads()


@dataclasses.dataclass
class While(Stmt):
    cond: Expr
    body: list  # list[Stmt]

    def reads(self) -> list[str]:
        return self.cond.reads()


@dataclasses.dataclass
class Break(Stmt):
    pass


@dataclasses.dataclass
class RawBlock(Stmt):
    """Escape hatch for lines that don't (yet) deserve their own node type.

    `verbatim` emits the lines with no indentation added at all, which is what adopting an existing
    hand-written driver requires (P4.0.6/C.3): its lines already carry their own indentation, and
    re-indenting them would move every line of the embedded `model.driver_script` -- failing the
    byte-identity gate for a purely cosmetic reason and hiding whatever real change the same commit
    made. Off by default: a block the exporter itself synthesizes (a `-- Weight ... packaged in GGUF`
    marker, say) has no indentation of its own and wants the enclosing block's.
    """
    lines: list
    defines_: list = dataclasses.field(default_factory=list)
    reads_: list = dataclasses.field(default_factory=list)
    verbatim: bool = False

    def defines(self) -> list[str]:
        return self.defines_

    def reads(self) -> list[str]:
        return self.reads_


@dataclasses.dataclass
class Function:
    name: str
    params: list
    body: list  # list[Stmt]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BUILTINS = {"loom", "math", "table", "string", "nil", "true", "false"}


def validate(function: Function) -> None:
    """Checks that every symbol read by a statement was defined by an earlier one (or is a function
    param/known builtin). Raises DriverIRError with the offending node/symbol on failure -- catches the
    "used a var from a slice that isn't the last op"/spurious-input class of bug at export time instead of
    a runtime crash or silently wrong Lua."""
    defined = set(function.params) | _BUILTINS
    _validate_block(function.body, defined, function.name)


def _validate_block(stmts: list, defined: set, fn_name: str) -> None:
    for stmt in stmts:
        for sym in stmt.reads():
            if sym not in defined:
                raise DriverIRError(
                    f"driver IR validation failed in function '{fn_name}': symbol '{sym}' is read by "
                    f"{stmt!r} before being defined by any earlier statement"
                )
        if isinstance(stmt, If):
            _validate_block(stmt.then, set(defined), fn_name)
            _validate_block(stmt.else_, set(defined), fn_name)
        elif isinstance(stmt, While):
            _validate_block(stmt.body, set(defined), fn_name)
        defined.update(stmt.defines())


def _walk_subgraph_calls(stmts: list):
    for stmt in stmts:
        if isinstance(stmt, SubgraphCall):
            yield stmt
        elif isinstance(stmt, If):
            yield from _walk_subgraph_calls(stmt.then)
            yield from _walk_subgraph_calls(stmt.else_)
        elif isinstance(stmt, While):
            yield from _walk_subgraph_calls(stmt.body)


def _topology_output_names(topo: dict) -> list:
    """A topology declares its output(s) as either "outputs" (plural array, P2 multi-output) or "output"
    (singular string, every model on the roadmap before P2 and every single-output topology since) --
    see graph_topology.h's own comment on this same distinction. Normalizes both into a plain list."""
    if "outputs" in topo:
        return list(topo["outputs"])
    if "output" in topo:
        return [topo["output"]]
    return []


def check_subgraph_calls(function: Function, topologies: dict) -> None:
    """For every SubgraphCall, confirms its `inputs` dict keys are all inputs the target topology actually
    declares, and that its `outputs`/`extra_outputs` don't request more than the topology actually
    produces. Topologies not present in `topologies` (e.g. synthesized/registered elsewhere) are skipped,
    not treated as an error."""
    for call in _walk_subgraph_calls(function.body):
        topo = topologies.get(call.module)
        if topo is None:
            continue
        declared = {inp["name"] for inp in topo.get("inputs", [])}
        provided = set(call.inputs.keys())
        extra = provided - declared
        if extra:
            raise DriverIRError(
                f"driver IR: loom.run_subgraph('{call.module}', ...) passes undeclared input(s) "
                f"{sorted(extra)}; topology '{call.module}' only declares inputs {sorted(declared)}"
            )

        # loom.run_subgraph returns every declared output's DATA first (in declared order), THEN every
        # declared output's SHAPE in that same order (lua_bridge.cpp's l_run_subgraph) -- so `outputs`
        # (the data locals) can request anywhere from 0 up to N of them (Lua silently discards
        # uncaptured trailing return values), but `extra_outputs` (shape locals) only line up correctly
        # if EVERY data output was captured first; a partial `outputs` list would make `extra_outputs`
        # silently capture data values instead of shapes.
        declared_outputs = _topology_output_names(topo)
        n_declared = len(declared_outputs)
        if len(call.outputs) > n_declared:
            raise DriverIRError(
                f"driver IR: loom.run_subgraph('{call.module}', ...) captures {len(call.outputs)} data "
                f"output(s) but topology '{call.module}' only declares {n_declared}: {declared_outputs}"
            )
        if call.extra_outputs and len(call.outputs) != n_declared:
            raise DriverIRError(
                f"driver IR: loom.run_subgraph('{call.module}', ...) requests {len(call.extra_outputs)} "
                f"shape output(s) via extra_outputs but only captures {len(call.outputs)}/{n_declared} "
                f"data outputs first; capturing a shape requires capturing every data output first "
                f"(topology '{call.module}' declares {declared_outputs})"
            )


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------

class LuaCodegen:
    def __init__(self, indent: str = "    "):
        self.indent = indent

    def emit_function(self, fn: Function) -> list:
        lines = [f"function {fn.name}({', '.join(fn.params)})"]
        lines.extend(self._emit_block(fn.body, 1))
        lines.append("end")
        return lines

    def _emit_block(self, stmts: list, depth: int) -> list:
        lines: list = []
        for stmt in stmts:
            lines.extend(self._emit_stmt(stmt, depth))
        return lines

    def _emit_stmt(self, stmt: Stmt, depth: int) -> list:
        pad = self.indent * depth
        if isinstance(stmt, Local):
            return [f"{pad}local {stmt.name} = {stmt.expr.render()}"]
        if isinstance(stmt, LocalDecl):
            return [f"{pad}local {stmt.name}"]
        if isinstance(stmt, Assign):
            return [f"{pad}{stmt.name} = {stmt.expr.render()}"]
        if isinstance(stmt, SubgraphCall):
            targets = ", ".join(list(stmt.outputs) + list(stmt.extra_outputs))
            call = Call("loom.run_subgraph",
                        [Lit(stmt.module), TableLit(stmt.axes), TableLit(stmt.inputs)])
            return [f"{pad}local {targets} = {call.render()}"]
        if isinstance(stmt, Argmax):
            call = Call("loom.argmax_row", [Var(stmt.tensor), stmt.n_vocab, stmt.row])
            return [f"{pad}local {stmt.result} = {call.render()}"]
        if isinstance(stmt, Return):
            return [f"{pad}return {', '.join(e.render() for e in stmt.exprs)}"]
        if isinstance(stmt, If):
            lines = [f"{pad}if {stmt.cond.render()} then"]
            lines.extend(self._emit_block(stmt.then, depth + 1))
            if stmt.else_:
                lines.append(f"{pad}else")
                lines.extend(self._emit_block(stmt.else_, depth + 1))
            lines.append(f"{pad}end")
            return lines
        if isinstance(stmt, While):
            lines = [f"{pad}while {stmt.cond.render()} do"]
            lines.extend(self._emit_block(stmt.body, depth + 1))
            lines.append(f"{pad}end")
            return lines
        if isinstance(stmt, Break):
            return [f"{pad}break"]
        if isinstance(stmt, RawBlock):
            if stmt.verbatim:
                return list(stmt.lines)
            return [f"{pad}{line}" for line in stmt.lines]
        raise DriverIRError(f"LuaCodegen: unhandled IR node {stmt!r}")
