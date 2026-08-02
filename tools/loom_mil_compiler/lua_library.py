"""`loom_lua` -- the driver-side standard library, and the component that emits it.

**Why this exists.** Stage C peeled five hand-written TTS drivers into `.lua` fragments, which named the
blocks but left them heterogeneous: measured across the five families, **11 functions totalling 112 lines
were shipped byte-identical in two of them** (`bilstm_run`, `run_resblk_stack`, `run_proj1x1`, the four
layout converters, `sigmoid`, `round_half_to_even`, `predict_durations`, `compute_wsum`), and their own
comments said so -- "identical to kokoro_driver.lua's own". The duplication was documented rather than
removed. Beyond those, the same handful of array operations kept reappearing as inline loops: sum a flat
array, slice a packed vector in half, apply an affine map, turn log-durations into integer ones, repeat
frames by duration.

So the fragments become **atomic Lua functions in `lua/`, one per file**, and a driver declares which it
uses instead of carrying its own copy. That is the same move `driver_components.py` made for statements,
applied to the half that stayed hand-written.

**What is deliberately NOT here.** VITS's frame expansion fuses the Gaussian reparameterisation into the
repeat loop; Kokoro's and StyleTTS2's duration-encoder loops interleave a subgraph call with per-timestep
row surgery; StyleTTS2's ADPM2 sampler is here only because it is a whole coherent sampler, not because
it is shared. Generalising the first two would produce a function with a callback per inner statement --
a worse thing to read than the loop, which is the same argument `flow_matching_export.py` makes about
why ADPM2 is not codegen'd. The rule this module follows: **a library function names one operation; a
family's own control flow stays in the family.**

**Honesty about call counts.** Some entries here have exactly one caller today -- `repeat_by_duration_
tfast`, `pad_last_to_multiple`. They are not deduplication, and saying so matters: they are here because
they name an operation the next family will reach for, and because a named function with a docstring is
worth more than a bare loop even at one call site. The `callers` count in `catalogue()` reports it, so
nobody has to guess which is which.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .driver_builder import DriverComponent
from .spec_protocol import ConfigDerived, Unchecked

LUA_DIR = Path(__file__).resolve().parent / "lua"


@dataclass(frozen=True)
class LuaFunction:
    """One library function: its source file, and the other library functions it calls.

    `requires` is checked against the body in both directions (`undeclared_calls` /
    `unused_requires`), so a dependency added to the Lua and not to the manifest -- which would emit a
    driver calling an undefined function -- fails at export time rather than inside LuaJIT.
    """

    name: str
    requires: Tuple[str, ...] = ()

    __links__ = {
        "name": ConfigDerived(
            claim=lambda spec, ctx: True,
            measured=lambda spec, ctx: spec.path.is_file(),
            detail=lambda spec, ctx: str(spec.path),
            message="{label} has no source file at {detail}. Every loom_lua entry is one .lua file "
                    "holding one function, named after it.",
            needs=(),
        ),
    }
    __unchecked__ = {
        "requires": Unchecked(
            "checked, but across the whole library rather than per entry: `undeclared_calls` and "
            "`unused_requires` compare every declaration against the body that declares it, in both "
            "directions, and the enforcing test walks all of them. A per-field link would re-read the "
            "same file once per entry and report one offender at a time"
        ),
    }

    @property
    def path(self) -> Path:
        return LUA_DIR / f"{self.name}.lua"

    def source(self) -> str:
        return self.path.read_text().rstrip("\n")

    def calls(self, library) -> set:
        """Library functions this one's body actually calls.

        Comments are stripped first, and that is not a nicety: several of these functions' docstrings
        name their siblings in prose, which a match over the raw text reads as a dependency. The check
        found that on its first run -- `round_half_to_even`'s comment mentions `predict_durations`,
        which calls *it*, so believing the comment would have inverted the dependency."""
        import re

        body = "\n".join(line for line in self.source().split("\n")
                         if not line.lstrip().startswith("--"))
        return {other for other in library
                if other != self.name and re.search(rf"\b{re.escape(other)}\s*\(", body)}


# name -> what it calls. Ordered by category rather than alphabetically, because the order is how a
# reader learns what the library covers.
_FUNCTIONS = (
    # -- flat-array primitives -------------------------------------------------------------------
    LuaFunction("array_sum"),
    LuaFunction("array_slice"),
    LuaFunction("array_affine"),
    # -- scalar math -----------------------------------------------------------------------------
    LuaFunction("sigmoid"),
    LuaFunction("round_half_to_even"),
    # -- layout conversion, between the two conventions this project names ------------------------
    LuaFunction("to_row_major"),
    LuaFunction("from_row_major"),
    LuaFunction("to_layout_a"),
    LuaFunction("from_layout_a"),
    # -- duration prediction and frame expansion --------------------------------------------------
    LuaFunction("durations_from_logw"),
    LuaFunction("pad_last_to_multiple"),
    LuaFunction("repeat_by_duration_tfast"),
    LuaFunction("predict_durations", requires=("sigmoid", "round_half_to_even")),
    # -- topology-driving helpers -----------------------------------------------------------------
    LuaFunction("bilstm_run"),
    LuaFunction("run_resblk_stack", requires=("to_layout_a", "from_layout_a")),
    LuaFunction("run_proj1x1", requires=("to_layout_a",)),
    # -- vocoder-side host precomputation ----------------------------------------------------------
    LuaFunction("compute_wsum"),
    # -- StyleTTS2's ADPM2 sampler, whole rather than shared ---------------------------------------
    LuaFunction("karras_schedule"),
    LuaFunction("adpm2_step"),
    LuaFunction("adpm2_sample", requires=("adpm2_step",)),
)

LIBRARY: Dict[str, LuaFunction] = {fn.name: fn for fn in _FUNCTIONS}


def resolve(names) -> List[LuaFunction]:
    """`names` plus everything they transitively require, in definition order.

    Definition order rather than requested order: Lua resolves a `local function` by lexical position,
    so a helper must be emitted before its caller. Walking `_FUNCTIONS` and keeping whatever is reachable
    gives that for free, and makes the emitted order stable no matter what order a family declares in.
    """
    unknown = [n for n in names if n not in LIBRARY]
    if unknown:
        raise KeyError(
            f"no such loom_lua function(s): {sorted(unknown)}. Known: {sorted(LIBRARY)}"
        )
    wanted, frontier = set(), list(names)
    while frontier:
        name = frontier.pop()
        if name in wanted:
            continue
        wanted.add(name)
        frontier.extend(LIBRARY[name].requires)
    return [fn for fn in _FUNCTIONS if fn.name in wanted]


def undeclared_calls() -> Dict[str, list]:
    """`{function: library functions it calls but does not declare}` -- the direction that produces a
    driver calling something never emitted."""
    out = {}
    for fn in _FUNCTIONS:
        missing = sorted(fn.calls(LIBRARY) - set(fn.requires))
        if missing:
            out[fn.name] = missing
    return out


def unused_requires() -> Dict[str, list]:
    """`{function: declared requirements its body never calls}` -- the direction that bloats every
    driver depending on it with functions it does not use."""
    out = {}
    for fn in _FUNCTIONS:
        extra = sorted(set(fn.requires) - fn.calls(LIBRARY))
        if extra:
            out[fn.name] = extra
    return out


@dataclass
class LuaLibrary(DriverComponent):
    """Emits the loom_lua functions a driver uses, and only those.

    "And only those" is the part worth enforcing: emitting the whole library into every driver would put
    StyleTTS2's ADPM2 sampler in Matcha's GGUF, and a reader of an embedded `driver_script` should be
    able to assume every function in it is reachable. `uses` is therefore a declaration, and
    `unreferenced()` checks it from the other side -- a name declared but never called is dead code that
    ships.
    """

    uses: Tuple[str, ...]

    __links__ = {
        "uses": ConfigDerived(
            claim=lambda spec, ctx: (),
            measured=lambda spec, ctx: tuple(sorted(set(spec.uses) - set(LIBRARY))),
            detail=lambda spec, ctx: sorted(set(spec.uses) - set(LIBRARY)),
            message=(
                "{label} declares loom_lua function(s) {detail}, which do not exist. The library is "
                "tools/loom_mil_compiler/lua/, one file per function."
            ),
            needs=(),
        ),
    }

    def link_label(self) -> str:
        return "LuaLibrary"

    def emitted(self) -> List[LuaFunction]:
        return resolve(self.uses)

    def unreferenced(self, driver_text: str) -> List[str]:
        """Declared functions the rest of the driver never calls.

        Takes the *rendered* driver rather than a fragment list, because a function can be called from
        another library function, from a fragment, or from a component's emitted statement -- and only
        the finished text sees all three."""
        import re

        names = {fn.name for fn in self.emitted()}
        body = driver_text
        dead = []
        for name in sorted(names):
            # Definitions do not count as references, so drop them before counting.
            without_defs = re.sub(rf"^local function {re.escape(name)}\b.*$", "", body,
                                  flags=re.MULTILINE)
            if not re.search(rf"\b{re.escape(name)}\s*\(", without_defs):
                dead.append(name)
        return dead

    def prelude(self, ctx) -> List[str]:
        lines = []
        for fn in self.emitted():
            lines.extend(fn.source().split("\n"))
            lines.append("")
        return lines


def catalogue() -> str:
    """One table: every library function, what it requires, and how many families call it.

    Generated from the manifest and the shipped drivers rather than transcribed, so it cannot drift --
    the same property P4.0.7's component catalogue is meant to have."""
    import re

    families = {}
    for driver in sorted(LUA_DIR.parent.parent.glob("convert_*/*_driver")):
        text = "\n".join(p.read_text() for p in sorted(driver.glob("*.lua")))
        families[driver.name.replace("_driver", "")] = text
    rows = ["| function | requires | called by |", "|---|---|---|"]
    for fn in _FUNCTIONS:
        callers = [fam for fam, text in families.items()
                   if re.search(rf"\b{re.escape(fn.name)}\s*\(", text)]
        # A function with no *fragment* caller is not unused -- `sigmoid` and `round_half_to_even` are
        # reached through `predict_durations`, `to_layout_a` through `run_proj1x1`. Saying "no caller"
        # would be the sort of half-true summary this project's checks exist to avoid.
        via = sorted(other.name for other in _FUNCTIONS if fn.name in other.requires)
        if not callers:
            callers = [f"*{', '.join(f'`{v}`' for v in via)}* only"] if via else ["**nothing** ⚠"]
        rows.append(f"| `{fn.name}` | {', '.join(f'`{r}`' for r in fn.requires) or '—'} | "
                    f"{', '.join(callers)} |")
    return "\n".join(rows)
