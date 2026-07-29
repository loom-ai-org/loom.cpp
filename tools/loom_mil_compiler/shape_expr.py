"""Symbolic shape expressions as **sympy objects** rather than strings.

The exporter derives an expression for every dynamic tensor dimension it emits (see
`exporter._infer_dynamic_dim_expr_uncached` and `value_facts.py`), and until this module existed it did
so by f-string concatenation: `f"(floor((({in_expr}) + {pad} - {kernel}) / {stride}) + 1)"`. That is
strictly worse algebra than the exporter is handed, for a reason that is easy to miss --
**`coremltools.converters.mil.mil.Symbol` subclasses `sympy.core.symbol.Symbol`**, and sympy is a
declared coremltools dependency. MIL's own shape inference propagates real algebra where it can (a
`reshape`'s output comes back as `shape=(1, 4*is2)`, a genuine compound sympy expression, not an opaque
symbol); the exporter was stringifying that and rebuilding expressions on top of the strings. The
result was correct but unreadable and enormous -- StyleTTS2's diffusion `reduce_mean` axis arrived as
`(floor(((1) * ((floor(((1) * (n_tokens) * (512)) / ...` for an expression that is algebraically just
`n_tokens`.

Three things this module provides, and one constraint that shapes all of them:

* `symbol()` / `as_expr()` / `parse()` build expressions, declaring shape symbols **positive integers**.
  That is not cosmetic: it is what lets sympy collapse `floor(512*n_tokens/512)` to `n_tokens` at all
  (`floor` of a symbol with no integer assumption cannot be simplified away).
* `sub_dynamic_symbols()` replaces MIL's opaque `isN` symbols by substitution on the expression tree
  rather than by regex over its printed form.
* `render()` prints back into the engine's expression language -- and this is the constraint:
  **`src/core/symbol_env.cpp` is a small recursive-descent evaluator accepting only `+ - * /`, unary
  minus, parentheses, identifiers, numbers, `floor(...)` and `sqrt(...)`**, evaluated in `double`.
  Sympy's default printer will happily emit `**`, `ceiling`, `Min`/`Max` and `Piecewise`, none of which
  parse. So `render` is a restricted printer that maps onto exactly that grammar and **raises**
  `UnsupportedShapeExpression` on anything outside it, rather than emitting silently-unparseable text
  that would only fail at model-load time. `parse()` implements the same grammar, so
  `parse(render(e))` round-trips by construction -- which is what makes it safe for a call site holding
  an already-rendered dim string to lift it back into algebra and keep composing.

One deliberate non-obviousness in `render`: `floor`'s argument is passed through `sympy.together` first.
Sympy automatically distributes a rational coefficient over a sum, so `floor((n_tokens - 512)/160)`
becomes `floor(n_tokens/160 - 16/5)` on construction. Both are the same real number, but the engine
evaluates in floating point, where a *single* division is one rounding and the distributed form is
three -- enough to flip a `floor` at an exact boundary. `together` puts it back over one denominator.
"""
import re

import sympy

from .symbols import DYNAMIC_SYMBOL_RE

# The engine's identifier rule (symbol_env.cpp's parse_ident): alnum + underscore, not starting with a
# digit. Anything a rendered expression names has to satisfy it or the C++ parser cannot read it back.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Rendering precedence levels, matching symbol_env.cpp's own expr/term/factor grammar.
_ADD, _MUL, _ATOM = 1, 2, 3

# Highest integer exponent `render` will expand into repeated multiplication. Shape expressions are
# linear-ish by nature; a large power almost certainly means something upstream went wrong, and
# emitting a 40-factor product would be worse than raising.
_MAX_POW = 8


class UnsupportedShapeExpression(ValueError):
    """Raised when an expression cannot be written in `symbol_env.cpp`'s grammar.

    Deliberately loud: the alternative is emitting a shape attribute that every consumer of the GGUF
    will fail to parse, at model-load time, with no pointer back to the op that produced it.
    """


_SYMBOL_CACHE = {}


def symbol(name: str) -> sympy.Symbol:
    """A shape symbol, declared a **positive integer**.

    Every symbol must come from here, not from bare `sympy.Symbol(name)`: sympy's structural equality
    includes assumptions, so `Symbol("n_tokens", integer=True) != Symbol("n_tokens")`, and two call
    sites that disagree would silently stop cancelling against each other. The assumptions are also
    load-bearing for simplification -- a tensor dimension really is a positive integer, and saying so is
    what lets `floor(512*n_tokens/512)` reduce to `n_tokens`.
    """
    if name not in _SYMBOL_CACHE:
        if not _IDENT_RE.match(name):
            raise UnsupportedShapeExpression(
                f"{name!r} is not a valid symbol name for symbol_env.cpp (identifiers are "
                f"[A-Za-z_][A-Za-z0-9_]*)."
            )
        _SYMBOL_CACHE[name] = sympy.Symbol(name, integer=True, positive=True)
    return _SYMBOL_CACHE[name]


# The one true dynamic quantity every topology is built at -- see `get_var_info`'s own docstring for why
# the engine's dynamic-shape support is deliberately single-axis.
N_TOKENS = symbol("n_tokens")


def as_expr(value):
    """Anything the exporter carries as a dimension -- a Python int/float, a rendered string, a MIL
    shape entry, or an expression built here -- as one sympy expression.

    Exact-integer floats become `Integer`, not `Float`: sympy will not simplify `floor(1.0*n_tokens)`
    (a `Float` coefficient makes the product non-integer as far as the assumption system is concerned),
    while `floor(n_tokens)` collapses. Since every float reaching here that happens to be integral came
    from a shape or an index, this loses nothing and recovers most simplification opportunities.
    """
    if isinstance(value, str):
        return parse(value)
    if isinstance(value, float) and value.is_integer():
        return sympy.Integer(int(value))
    if isinstance(value, bool):  # bool is an int subclass; nothing here means "True" as a dimension
        raise UnsupportedShapeExpression(f"{value!r} is not a shape expression")
    expr = sympy.sympify(value)
    if not isinstance(expr, sympy.Basic):
        raise UnsupportedShapeExpression(f"{value!r} is not a shape expression")
    return expr


def floor_div(x, y):
    """Integer division as the exporter means it: `floor(x / y)`, matching both MIL's `floor_div` and
    the `floor(...)` the engine's evaluator provides."""
    return sympy.floor(as_expr(x) / as_expr(y))


def to_number(expr):
    """`expr` as a plain Python `int`/`float` if it is a literal, else None.

    Keeps the exporter's existing distinction between the two: several JSON attributes are emitted as
    real numbers when the value is known and as an expression string only when it isn't.
    """
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic) and expr.is_number:
        if expr.is_Integer:
            return int(expr)
        return float(expr)
    return None


def sub_dynamic_symbols(expr, overrides=None, default="n_tokens"):
    """Replaces every opaque MIL shape symbol (`is0`, `is531`, ...) in `expr`.

    `overrides` maps a raw MIL symbol name to a replacement *expression* (Kokoro passes e.g.
    `{"is42": "600*n_tokens+20"}` -- see `LoomGGUFExporter.symbol_overrides`); anything not overridden
    becomes `default`. Substitution happens on the expression tree, so a compound MIL dim like
    `4*is2 + 20` keeps its own algebra instead of being rebuilt from its printed form.

    `DYNAMIC_SYMBOL_RE` is matched with `fullmatch` here, not `search`: it exists to find `isN` tokens
    inside a printed expression, and what is being tested here is a symbol's whole name.
    """
    expr = as_expr(expr)
    subs = {}
    for free in expr.free_symbols:
        if DYNAMIC_SYMBOL_RE.fullmatch(free.name):
            subs[free] = as_expr((overrides or {}).get(free.name, default))
    return expr.subs(subs) if subs else expr


def has_dynamic_symbol(expr) -> bool:
    """True iff `expr` still mentions an unresolved MIL `isN` symbol."""
    return any(DYNAMIC_SYMBOL_RE.fullmatch(s.name) for s in as_expr(expr).free_symbols)


# -- rendering -------------------------------------------------------------------------------------


def render(expr) -> str:
    """`expr` in `symbol_env.cpp`'s expression language, or raise `UnsupportedShapeExpression`."""
    text, _ = _render(as_expr(expr))
    return text


def _wrap(expr, min_prec):
    text, prec = _render(expr)
    return f"({text})" if prec < min_prec else text


def _render(e):
    """(text, precedence) for one node. Precedence is the level at which the text can be dropped into a
    parent without parentheses, using symbol_env.cpp's own expr/term/factor levels."""
    if e.is_Symbol:
        if not _IDENT_RE.match(e.name):
            raise UnsupportedShapeExpression(f"symbol {e.name!r} is not a valid identifier")
        return e.name, _ATOM

    if e.is_Integer:
        n = int(e)
        # A negative literal is rendered as a unary minus, which symbol_env.cpp parses at factor level
        # only -- so it must parenthesize inside any product ("a*(-5)"), hence the _ADD level.
        return (str(n), _ATOM if n >= 0 else _ADD)

    if e.is_Rational:  # a genuine p/q that is not an integer
        num, den = e.p, e.q
        return (f"{num}/{den}", _MUL if num >= 0 else _ADD)

    if e.is_Float:
        value = float(e)
        if value != value or value in (float("inf"), float("-inf")):
            raise UnsupportedShapeExpression(f"{value!r} is not a finite shape expression")
        # repr gives the shortest round-tripping form ("1.5", not sympy's own "1.50000000000000").
        # std::stod reads exponent notation too, so no special case is needed for very small floats.
        return (repr(value), _ATOM if value >= 0 else _ADD)

    if isinstance(e, sympy.floor):
        # `together` first -- see this module's docstring: sympy distributes rational coefficients over
        # sums on construction, and the distributed form is more floating-point roundings inside a
        # floor than the engine needs to take.
        return f"floor({_wrap(sympy.together(e.args[0]), 0)})", _ATOM

    if e.is_Pow:
        return _render_pow(e)

    if e.is_Mul:
        return _render_mul(e)

    if e.is_Add:
        return _render_add(e)

    raise UnsupportedShapeExpression(
        f"{type(e).__name__} has no equivalent in symbol_env.cpp's grammar "
        f"(+ - * /, unary minus, parentheses, identifiers, numbers, floor, sqrt): {e!r}"
    )


def _render_pow(e):
    base, exponent = e.args
    if exponent == sympy.Rational(1, 2):
        return f"sqrt({_wrap(base, 0)})", _ATOM
    if exponent.is_Integer:
        n = int(exponent)
        if n < 0:
            return f"1/{_wrap(sympy.Pow(base, -n), _ATOM)}", _MUL
        if 1 <= n <= _MAX_POW:
            # No exponentiation operator exists in the target grammar, so a small integer power becomes
            # the repeated product it stands for.
            return "*".join(_wrap(base, _MUL) for _ in range(n)), _MUL if n > 1 else _ATOM
    raise UnsupportedShapeExpression(
        f"exponent {exponent} is not expressible in symbol_env.cpp's grammar (it has no '**'; only "
        f"sqrt and integer powers up to {_MAX_POW} are expanded): {e!r}"
    )


def _render_mul(e):
    if e.could_extract_minus_sign():
        return f"-{_wrap(-e, _MUL)}", _ADD
    numerator, denominator = sympy.fraction(e)
    num_text = _render_product(numerator)
    if denominator == 1:
        return num_text, _MUL
    # The divisor goes in at atom level so a product denominator keeps its parentheses -- "a/(b*c)",
    # never "a/b*c", which the left-to-right term rule would evaluate differently.
    return f"{num_text}/{_wrap(denominator, _ATOM)}", _MUL


def _render_product(expr):
    if expr.is_Mul:
        return "*".join(_wrap(f, _MUL) for f in expr.as_ordered_factors())
    return _wrap(expr, _MUL)


def _render_add(e):
    parts = []
    for i, term in enumerate(e.as_ordered_terms()):
        negated = term.could_extract_minus_sign()
        text = _wrap(-term if negated else term, _MUL)
        if i == 0:
            parts.append(f"-{text}" if negated else text)
        else:
            parts.append(f"- {text}" if negated else f"+ {text}")
    return " ".join(parts), _ADD


# -- parsing ---------------------------------------------------------------------------------------


def parse(text: str):
    """The inverse of `render`: exactly `symbol_env.cpp`'s grammar, and nothing else.

    Deliberately hand-written rather than `sympy.sympify`, for two reasons. It accepts precisely what
    the C++ evaluator accepts, so a string that parses here is one the engine can read (and `parse` is
    therefore also the grammar's test oracle); and it builds every identifier through `symbol()`, so
    parsed expressions compare and cancel against constructed ones instead of silently differing by
    their assumptions.
    """
    parser = _Parser(text)
    expr = parser.parse_expr()
    parser.skip_ws()
    if parser.pos != len(parser.src):
        raise UnsupportedShapeExpression(
            f"unexpected trailing input in shape expression {text!r} at {parser.pos}"
        )
    return expr


class _Parser:
    def __init__(self, src):
        self.src = src
        self.pos = 0

    def skip_ws(self):
        while self.pos < len(self.src) and self.src[self.pos].isspace():
            self.pos += 1

    def peek(self):
        self.skip_ws()
        return self.src[self.pos] if self.pos < len(self.src) else ""

    def accept(self, ch):
        if self.peek() == ch:
            self.pos += 1
            return True
        return False

    def expect(self, ch):
        if not self.accept(ch):
            raise UnsupportedShapeExpression(
                f"expected {ch!r} in shape expression {self.src!r} at {self.pos}"
            )

    def parse_expr(self):
        value = self.parse_term()
        while True:
            if self.accept("+"):
                value = value + self.parse_term()
            elif self.accept("-"):
                value = value - self.parse_term()
            else:
                return value

    def parse_term(self):
        value = self.parse_factor()
        while True:
            if self.accept("*"):
                value = value * self.parse_factor()
            elif self.accept("/"):
                value = value / self.parse_factor()
            else:
                return value

    def parse_factor(self):
        if self.accept("-"):
            return -self.parse_factor()
        if self.accept("("):
            value = self.parse_expr()
            self.expect(")")
            return value
        # The engine allows a '$' sigil before a symbol reference; accepted here for symmetry.
        self.accept("$")
        self.skip_ws()
        if self.pos < len(self.src) and (self.src[self.pos].isalpha() or self.src[self.pos] == "_"):
            name = self.parse_ident()
            if name in ("floor", "sqrt"):
                self.expect("(")
                arg = self.parse_expr()
                self.expect(")")
                return sympy.floor(arg) if name == "floor" else sympy.sqrt(arg)
            return symbol(name)
        return self.parse_number()

    def parse_ident(self):
        self.skip_ws()
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == "_"):
            self.pos += 1
        if self.pos == start:
            raise UnsupportedShapeExpression(
                f"expected an identifier in shape expression {self.src!r} at {self.pos}"
            )
        return self.src[start:self.pos]

    def parse_number(self):
        self.skip_ws()
        match = re.compile(r"\d+(\.\d*)?([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?").match(self.src, self.pos)
        if match is None:
            raise UnsupportedShapeExpression(
                f"expected a number in shape expression {self.src!r} at {self.pos}"
            )
        self.pos = match.end()
        text = match.group(0)
        if re.fullmatch(r"\d+", text):
            return sympy.Integer(int(text))
        return as_expr(float(text))
