"""Covers `shape_expr.py`: the grammar boundary between the exporter's algebra and the engine.

The properties worth pinning are the ones a future change could silently break -- that `render` never
emits something `src/core/symbol_env.cpp` cannot parse (and raises loudly instead), that `parse` accepts
exactly what `render` produces, and that the assumptions on shape symbols (positive integers) are what
make the simplifications real rather than cosmetic.

Run: ~/.venvs/piper/bin/python3 -m pytest tools/loom_mil_compiler/test_shape_expr.py
"""
import re
import sys
from pathlib import Path

import pytest
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loom_mil_compiler.shape_expr import (  # noqa: E402
    N_TOKENS,
    UnsupportedShapeExpression,
    as_expr,
    floor_div,
    has_dynamic_symbol,
    parse,
    render,
    sub_dynamic_symbols,
    symbol,
    to_number,
)

n = N_TOKENS

# Everything symbol_env.cpp's tokenizer can encounter: identifiers, digits, the four operators,
# parentheses, decimal points and spaces. Anything else (notably '**') means an unparseable attribute.
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/(). ]*$")

EXPRESSIONS = [
    n,
    as_expr(64),
    as_expr(0),
    as_expr(-1),
    n / 160,
    sympy.floor((n - 512) / 160) + 1,
    (n - 1) * 8 - 8 + 16,
    600 * n + 20,
    4 * n * 64,
    (n + 2) * (n - 3),
    sympy.sqrt(n),
    n ** 2,
    1 / n,
    sympy.Float(1.5) * n,
    -n,
    floor_div(n * 512, 512),
    sympy.floor(n / 2) * 3,
]


@pytest.mark.parametrize("expr", EXPRESSIONS, ids=lambda e: str(e))
def test_render_round_trips_through_parse(expr):
    text = render(expr)
    assert _ALLOWED_CHARS.match(text), f"{text!r} contains a character symbol_env.cpp cannot tokenize"
    assert "**" not in text
    assert sympy.simplify(parse(text) - expr) == 0


def test_render_matches_the_engines_arithmetic_at_concrete_lengths():
    """The real contract: whatever `render` prints must evaluate to the same number the expression
    means, for every sequence length the model can be built at."""
    expr = sympy.floor((n - 512) / 160) + 1
    text = render(expr)
    for probe in (1600, 16000, 16001, 31999, 320000):
        assert float(parse(text).subs(n, probe)) == float(expr.subs(n, probe))


# -- what makes the simplification real ---------------------------------------------------------

def test_shape_symbols_are_positive_integers():
    """Not cosmetic: `floor` of a symbol only collapses when the symbol is known to be an integer, and
    that single fact is what turns StyleTTS2's nested-floor monster back into `n_tokens`."""
    assert n.is_integer and n.is_positive
    assert sympy.floor(n) == n
    assert floor_div(512 * n, 512) == n


def test_symbols_are_interned_so_assumptions_cannot_diverge():
    assert symbol("n_tokens") is N_TOKENS
    assert parse("n_tokens") is N_TOKENS


def test_integral_floats_become_integers():
    """A `Float` coefficient blocks the integer assumption, so `floor(1.0*n)` would not collapse."""
    assert as_expr(2.0) == sympy.Integer(2)
    assert sympy.floor(as_expr(1.0) * n) == n
    assert as_expr(1.5) != sympy.Integer(1)


def test_floor_arguments_are_recombined_over_one_denominator():
    """Sympy distributes rational coefficients over sums on construction; the engine evaluates in
    doubles, where one division rounds once and the distributed form rounds three times."""
    assert str(sympy.floor((n - 512) / 160)) == "floor(n_tokens/160 - 16/5)"
    assert render(sympy.floor((n - 512) / 160)) == "floor((n_tokens - 512)/160)"


# -- the grammar's edges ------------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    sympy.ceiling(n / 3),
    sympy.Max(n, 8),
    sympy.Min(n, 8),
    n ** sympy.Rational(1, 3),
    sympy.Piecewise((n, n > 2), (1, True)),
    sympy.log(n),
])
def test_inexpressible_constructs_raise_rather_than_emitting_bad_text(expr):
    with pytest.raises(UnsupportedShapeExpression):
        render(expr)


def test_small_integer_powers_expand_into_products():
    assert render(n ** 3) == "n_tokens*n_tokens*n_tokens"


def test_large_integer_powers_raise():
    with pytest.raises(UnsupportedShapeExpression):
        render(n ** 40)


def test_division_by_a_product_keeps_its_parentheses():
    """`a/(b*c)` and `a/b*c` are different numbers under the engine's left-to-right term rule."""
    a, b, c = symbol("a"), symbol("b"), symbol("c")
    text = render(a / (b * c))
    assert float(parse(text).subs({a: 12, b: 3, c: 2})) == 2.0


@pytest.mark.parametrize("text", ["n_tokens **", "floor(n_tokens", "n_tokens 3", "", "@"])
def test_parse_rejects_what_the_engine_would_reject(text):
    with pytest.raises(UnsupportedShapeExpression):
        parse(text)


def test_parse_accepts_the_engines_dollar_sigil():
    assert parse("$n_tokens + 1") == n + 1


# -- MIL symbol substitution --------------------------------------------------------------------

def test_opaque_mil_symbols_become_n_tokens():
    # `isN` is coremltools' own naming for a symbolic dim; MIL hands these over as sympy symbols
    # inside real expressions, e.g. a reshape's `4*is2`.
    assert sub_dynamic_symbols(4 * sympy.Symbol("is2")) == 4 * n
    assert has_dynamic_symbol(4 * sympy.Symbol("is2"))
    assert not has_dynamic_symbol(4 * n)


def test_overrides_replace_a_named_symbol_with_a_whole_expression():
    """Kokoro's decoder_vocoder declares several independent dynamic leaf inputs whose real lengths are
    fixed multiples of the one true quantity -- supplied by raw MIL symbol name."""
    expr = sub_dynamic_symbols(sympy.Symbol("is42") * 4 + 20, {"is42": "600*n_tokens+20"})
    assert expr == 2400 * n + 100
    assert render(expr) == "2400*n_tokens + 100"


def test_to_number_separates_literals_from_expressions():
    assert to_number(as_expr(3)) == 3
    assert isinstance(to_number(as_expr(3)), int)
    assert to_number(sympy.Rational(3, 2)) == 1.5
    assert to_number(n) is None
