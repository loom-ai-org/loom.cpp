"""The driver-component registry -- the shelf the components sit on (BACKLOG.md P4.0.7, stage D.1).

**Why a registry at all, when the components already share a calling convention.** Stage C put every
driver contribution behind one `DriverComponent` API, which is what made the five hand-written TTS
drivers composable. What it did not give is an *account* of them: which components exist, what each one
emits, which models use it, and what its declarations would say if they failed. P4.0.7's deliverable is
that account -- and the trap the author's stage-D review named is that **a name is not a mechanism**.
Peeling produced named blocks that were still duplicated; an inventory of components can just as easily
be a directory rather than a shelf.

So the entries here are not a transcription. Everything in them is checked, in both directions, the way
`loom_lua`'s manifest is:

* **A component with no entry fails an export.** `DriverBuilder.build` looks every component up as it
  assembles, so a class added to this package and forgotten here cannot ship: it would be a piece of a
  driver that no catalogue accounts for. (`unregistered_component_classes()` is the same question asked
  statically, and the enforcing test asks it that way so the answer does not depend on which family
  happens to be exported.)
* **An entry that claims the wrong emission fails an export.** `emits` says which of the three slots a
  component contributes to; `DriverBuilder.build` compares that against what it actually emitted. A
  stale claim -- "prelude" on a component that has become statement-only -- is exactly the kind of drift
  the catalogue would otherwise report as fact.
* **An entry nothing uses is reported.** `usage()` derives the model list by building every registered
  family's real component list, not by reading a `used_by` field someone maintained by hand.

**Why a module and not the `driver_components/` directory the plan wrote.** The plan's `/` was shorthand
for the shelf, and the shelf is the registry plus the checks, not the file layout. Splitting the module
into a package would also silently weaken the standing rule: `test_spec_protocol`'s scan walks
`pkgutil.iter_modules(package.__path__)` and imports each module, so a dataclass defined in
`driver_components/foo.py` has `__module__ == "loom_mil_compiler.driver_components.foo"`, which the scan
neither reaches nor reports as unimportable. A directory would therefore have cost a real check to buy a
cosmetic one, which is the trade this roadmap keeps refusing.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, get_args

from .driver_builder import DriverComponent
from .spec_protocol import Unchecked

# The three slots a component can contribute to, in emission order. `DriverScript` is prelude lines,
# the entry function's body, then postlude lines -- see `driver_builder.DriverScript` for why a driver
# is a Lua module rather than one function.
PRELUDE, STATEMENTS, POSTLUDE = "prelude", "statements", "postlude"
SLOTS = (PRELUDE, STATEMENTS, POSTLUDE)


@dataclass(frozen=True)
class ComponentEntry:
    """One registered component: what it is, what it emits, and where it came from.

    `emits` is the set of slots the component may contribute to -- *may*, not *does*, because two
    entries legitimately vary per instance: `LuaFragment` emits a prelude when `top_level` is set and
    statements otherwise, and `RawLuaDriver`'s postlude is empty for a driver whose entry function is
    the last thing in the file. `DriverBuilder.build` therefore checks the observed slots are a subset
    of the declared ones, which catches a claim that is too narrow (a real drift) without failing on a
    claim that is merely not exercised by one model.

    The other direction -- a slot declared and never once observed across all eleven models -- is what
    `usage()` and the enforcing test cover, since it is only answerable from more than one export.
    """

    name: str
    cls: type
    emits: Tuple[str, ...]
    summary: str
    # Why an entry no model uses is still on the shelf. Required for exactly those entries (the
    # enforcing test derives which from `usage()`), and the same rule `lua_library` applies to a
    # library function with no caller: a component nobody uses is either dead code or deliberately
    # kept, and the two must not look alike.
    no_user_reason: Optional[str] = None

    # Every field here is checked, and none of them by a link -- the same shape as `LuaFunction.
    # requires`, and for the same reason: what these declare is checked across the whole manifest or
    # against a built driver, neither of which a per-field link can see when the manifest is read.
    __unchecked__ = {
        "name": Unchecked(
            "the shelf name. It refers to nothing -- it CREATES the reference the catalogue and "
            "`get()` resolve against -- and `registry()` raises on a duplicate, which is the only way "
            "to get it wrong"
        ),
        "cls": Unchecked(
            "checked, in both directions, but not per entry: `__post_init__` rejects a class that is "
            "not a DriverComponent, and `unregistered_component_classes()` walks the package for the "
            "converse -- a shipped component class with no entry. A per-field link has no second "
            "authority to appeal to here, since the class IS the component"
        ),
        "emits": Unchecked(
            "checked at export time by `check_emission`, against what the component really emitted. "
            "Not answerable when the manifest is read: it needs a built driver and one concrete "
            "instance, and two entries legitimately vary per instance (see the class docstring)"
        ),
        "summary": Unchecked("the catalogue's prose. Cosmetic to every check and load-bearing to the "
                             "reader P4.1 is meant to be, which is the same category as a component's "
                             "own `note`"),
        "no_user_reason": Unchecked(
            "required exactly when `usage()` finds no user for this entry, and forbidden when it finds "
            "one -- both directions enforced by test_component_registry, because the question is about "
            "the whole registry and every family's component list rather than about this field"
        ),
    }

    def __post_init__(self):
        unknown = [slot for slot in self.emits if slot not in SLOTS]
        if unknown:
            raise ValueError(f"component {self.name!r} declares unknown emission slot(s) {unknown}; "
                             f"known slots: {list(SLOTS)}")
        if not issubclass(self.cls, DriverComponent):
            raise TypeError(f"component {self.name!r} registers {self.cls.__name__}, which is not a "
                            f"DriverComponent subclass")

    def links(self) -> Dict[str, list]:
        """`{field: [Link]}` this component's class declares, for the catalogue. Read off the class,
        so it is the same declaration `DriverBuilder.build` hands to the checker."""
        from .spec_protocol import declared_links_for

        return declared_links_for(self.cls)

    def unchecked(self) -> Dict[str, object]:
        from .spec_protocol import declared_unchecked

        return declared_unchecked(self.cls)


def _entries() -> Tuple[ComponentEntry, ...]:
    """The manifest. A function rather than a module-level tuple because every component class lives in
    a module that imports this one's `DriverComponent` -- importing them at module scope would make the
    registry and the components a cycle."""
    from .driver_components import (
        ArgmaxEpilogue, DriverInputs, DriverReturn, FlowMatchingSampler, LuaFragment, ModularChain,
        MonolithicCall, PrefillDecodeLoop, RawLuaDriver, SubgraphCallComponent,
    )
    from .lua_library import LuaLibrary

    return (
        # -- the synthesized paths' components (C.2) -------------------------------------------------
        ComponentEntry(
            "driver_inputs", DriverInputs, (STATEMENTS,),
            "Binds every name the topologies below are called with: read from the caller's `inputs` "
            "table, or computed host-side (`cache_position` via loom.range, `attention_mask` via "
            "loom.causal_mask).",
        ),
        ComponentEntry(
            "monolithic_call", MonolithicCall, (STATEMENTS,),
            "The single `run_subgraph` call a flattened export's driver makes, capturing the output's "
            "shape alongside its data so the epilogue knows the vocab size -- or, for a KV-cached "
            "topology, retaining the output engine-side and binding nothing, so the logits never "
            "become a Lua table at all.",
        ),
        ComponentEntry(
            "modular_chain", ModularChain, (STATEMENTS,),
            "Threads one tensor through an independently-traced submodule chain: prefix -> [aux] -> "
            "layer_0..N -> suffix_0..M, each stage carrying its own resolved input map.",
        ),
        ComponentEntry(
            "prefill_decode_loop", PrefillDecodeLoop, (STATEMENTS,),
            "The `infer_with_past` generation loop: prefill, then decode one token at a time against "
            "the KV cache until max_new_tokens or eos_token. One loop rather than a prefill plus a "
            "decode loop, because a cached ATTENTION node makes the prefill its first iteration. "
            "**The `used by` column over-states this one**, and it is the only entry where that is "
            "true: it is a field of every flattened causal-LM builder, but the exporter sets it only "
            "for a topology whose cross-step state is ENTIRELY the KV cache. LFM2-monolithic's ten "
            "ShortConv layers are not, so it carries the field and exports `infer` alone.",
        ),
        ComponentEntry(
            "argmax_epilogue", ArgmaxEpilogue, (STATEMENTS,),
            "Returns the next token rather than the raw logits: argmax over the active row, read out "
            "of the producing module's retained output by name, or -- for a topology that marshalled "
            "its tensor -- over the returned table, guarded for an output that is not an array.",
        ),
        # -- adopting and peeling a hand-written driver (C.3-C.8) ------------------------------------
        ComponentEntry(
            "raw_lua_driver", RawLuaDriver, (PRELUDE, STATEMENTS, POSTLUDE),
            "A hand-written `.lua` adopted whole -- prelude, one verbatim body block, postlude -- with "
            "its own `loom.run_subgraph` call sites parsed out and declared. The step every TTS family "
            "moved onto the builder through; no family is on it today.",
            no_user_reason=(
                "the adoption step's component (C.3). All five TTS families passed through it and all "
                "five are now peeled, so it ships with no user by design -- it is what the *next* "
                "hand-written driver is adopted by, in a commit whose gate is byte-identity. Deleting "
                "it would mean the next family's first step has to be written again"
            ),
        ),
        ComponentEntry(
            "lua_fragment", LuaFragment, (PRELUDE, STATEMENTS),
            "One hand-written block of a peeled driver, kept as its own `.lua` file, declaring what it "
            "reads and defines (and, since D.2, which topologies its computed call sites drive).",
        ),
        ComponentEntry(
            "subgraph_call", SubgraphCallComponent, (STATEMENTS,),
            "One `loom.run_subgraph` as IR rather than text, so `check_subgraph_calls` covers its "
            "output arity too -- what a peel buys structurally.",
        ),
        ComponentEntry(
            "flow_matching_sampler", FlowMatchingSampler, (PRELUDE, STATEMENTS),
            "A `FlowMatchingSpec`'s generated Euler-CFM sampler function, plus the line that calls it.",
        ),
        ComponentEntry(
            "driver_return", DriverReturn, (STATEMENTS,),
            "What the entry function hands back to the host.",
        ),
        # -- the driver-side standard library (P4.0.7 step 1) ----------------------------------------
        ComponentEntry(
            "lua_library", LuaLibrary, (PRELUDE,),
            "Emits the `loom_lua` functions a driver declares, and only those -- the transitive "
            "closure of `uses`, in definition order.",
        ),
    )


_BY_NAME: Optional[Dict[str, ComponentEntry]] = None
_BY_CLASS: Optional[dict] = None


def registry() -> Dict[str, ComponentEntry]:
    """`{name: entry}`, in manifest order."""
    global _BY_NAME, _BY_CLASS
    if _BY_NAME is None:
        entries = _entries()
        _BY_NAME = {}
        _BY_CLASS = {}
        for entry in entries:
            if entry.name in _BY_NAME:
                raise ValueError(f"two components registered as {entry.name!r}")
            if entry.cls in _BY_CLASS:
                raise ValueError(f"{entry.cls.__name__} is registered twice: "
                                 f"{_BY_CLASS[entry.cls].name!r} and {entry.name!r}")
            _BY_NAME[entry.name] = entry
            _BY_CLASS[entry.cls] = entry
    return _BY_NAME


def get(name: str) -> ComponentEntry:
    try:
        return registry()[name]
    except KeyError:
        raise KeyError(f"no such driver component: {name!r}. Registered: "
                       f"{sorted(registry())}") from None


# A component class defined in one of this package's own modules is a *shipped* component and must be
# registered. One defined anywhere else is not: a test's stand-in subclass exercises the builder API
# rather than shipping in a GGUF, and an out-of-tree family's component is not in this catalogue by
# definition. `test_` modules are excluded on the same rule the standing-rule scan uses.
_PACKAGE = __name__.rsplit(".", 1)[0]


def is_shipped(cls) -> bool:
    module = getattr(cls, "__module__", "")
    if not module.startswith(f"{_PACKAGE}."):
        return False
    return not module.rsplit(".", 1)[-1].startswith("test_")


def entry_for(component) -> Optional[ComponentEntry]:
    """The entry for `component`'s class, or `None` for a component this registry does not govern.

    Raises for a *shipped* component with no entry -- the direction that keeps the catalogue complete.
    A driver assembled from a component nobody registered is a driver the catalogue under-reports, and
    since the catalogue is what P4.1 is meant to reuse from without reading the source, an omission
    there is worse than a missing feature.
    """
    registry()
    cls = type(component)
    entry = _BY_CLASS.get(cls)
    if entry is not None:
        return entry
    if is_shipped(cls):
        raise KeyError(
            f"{cls.__module__}.{cls.__name__} is a driver component but is not in the component "
            f"registry, so it would appear in a shipped driver and in no catalogue. Add a "
            f"ComponentEntry to component_registry._entries(). Registered: {sorted(registry())}."
        )
    return None


def check_emission(component, emitted) -> None:
    """`emitted` is the set of slots `component` really contributed to. Raises if the registry claims
    a narrower set.

    Subset, not equality -- see `ComponentEntry.emits` for the two components that legitimately vary
    per instance."""
    entry = entry_for(component)
    if entry is None:
        return
    extra = [slot for slot in SLOTS if slot in emitted and slot not in entry.emits]
    if extra:
        raise ValueError(
            f"component {entry.name!r} ({entry.cls.__name__}) emitted {extra}, which its registry "
            f"entry does not declare (declares {list(entry.emits)}). The catalogue's 'emits' column "
            f"is generated from that declaration, so an out-of-date one publishes a false account of "
            f"what the component contributes to a driver."
        )


def unregistered_component_classes() -> List[str]:
    """Shipped `DriverComponent` subclasses with no registry entry, by `module.ClassName`.

    The static form of the check `entry_for` makes at export time. Both exist because they fail at
    different moments and only one of them is cheap: this one needs no checkpoint and covers components
    no family currently uses, while `entry_for` covers a component reached by a path this scan's import
    of the package somehow did not define.
    """
    import importlib
    import inspect
    import pkgutil

    # By `__name__`, not by a literal: this package is imported as `loom_mil_compiler` from the tests
    # (which put `tools/` on the path) and as `tools.loom_mil_compiler` from the CLI, and a scan that
    # silently found nothing under one of those spellings would report a clean result for the wrong
    # reason.
    package = importlib.import_module(_PACKAGE)
    registry()
    found = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("test_"):
            continue
        try:
            module = importlib.import_module(f"{_PACKAGE}.{module_info.name}")
        except Exception:
            # Reported by test_spec_protocol's own scan, which owns the "a module that will not import
            # hides whatever is in it" question; duplicating the report here would give two failures
            # for one cause.
            continue
        for obj in vars(module).values():
            if (inspect.isclass(obj) and issubclass(obj, DriverComponent) and obj is not DriverComponent
                    and obj.__module__ == module.__name__ and obj not in _BY_CLASS):
                found.append(f"{module_info.name}.{obj.__name__}")
    return sorted(found)


# -- usage: which models really use which component --------------------------------------------------
#
# Derived rather than declared, for the same reason `lua_library.catalogue()` counts its callers by
# reading the shipped fragments: a `used_by` field is a second copy of a fact, and the copy is the one
# that rots. The two halves come from different places because the two paths genuinely differ -- a TTS
# family builds its component list itself, while the synthesized paths' lists are assembled inside the
# exporter from a traced program, and a builder's own dataclass fields are the authority on which
# components that list holds (`DriverBuilder.components` returns exactly them).


def builder_components(builder_class) -> List[str]:
    """The registered names of the components a `DriverBuilder` subclass is made of, read off its
    dataclass fields."""
    registry()
    names = []
    for field in fields(builder_class):
        cls = field.type if isinstance(field.type, type) else None
        if cls is None:
            # `Optional[X]` -- an entry function beside the main one is optional by construction
            # (PrefillArgmaxBuilder.decode, KV-CACHE.md 3.3), and a component that only some models
            # carry still belongs in the catalogue for the ones that do.
            cls = next((arg for arg in get_args(field.type)
                        if isinstance(arg, type) and arg in _BY_CLASS), None)
        if cls is None:
            # A string annotation (`from __future__ import annotations`) or a container; resolved by
            # name against the registry instead of by identity.
            cls = next((entry.cls for entry in registry().values()
                        if entry.cls.__name__ == str(field.type)), None)
        entry = _BY_CLASS.get(cls)
        if entry is not None:
            names.append(entry.name)
    return names


def usage(model_paths: Optional[Dict[str, str]] = None):
    """`({component name: models that use it}, {model: why it could not be read})`.

    The TTS half is real: every registered `text-to-speech` recognizer's config is built and its
    `driver_components()` called, which is the same list the export uses. That works without a
    checkpoint because a peeled family's component list is paths and IR expressions -- nothing in it
    reads the model -- and if that ever stops being true this function raises rather than reporting a
    shorter list.

    The synthesized half is read off `PrefillArgmaxBuilder`/`ModularChainBuilder`'s fields and
    attributed to models by the decomposition each config declares, through
    `driver_components.SYNTHESIZED_BUILDERS` -- which `LoomGGUFExporter` itself constructs from, so the
    attribution cannot drift from what really runs.

    **The second return value is not an implementation detail.** One registered recognizer -- P4.0.4's
    generic `hf-causal-lm` fallback -- reads a real `config.json` to build its config at all, since
    inferring the architecture from the checkpoint is the whole point of it. Pass its path in
    `model_paths` to include it; otherwise it is reported as unread rather than dropped, because a
    catalogue that quietly omitted a model would understate exactly what it exists to account for.
    """
    from .driver_components import SYNTHESIZED_BUILDERS
    from .registry import default_registry

    paths = dict(model_paths or {})
    used: Dict[str, List[str]] = {name: [] for name in registry()}
    unavailable: Dict[str, str] = {}
    for task, entry in sorted(default_registry()._entries.items()):
        for recognizer in entry.recognizers:
            try:
                # A `Path`, not a string: the generic `hf-causal-lm` recognizer reads a real
                # `config.json` to infer what it is building, and a path that does not exist is a
                # perfectly good answer to that (no overrides) while a `str` is a TypeError.
                config = recognizer.build_config(
                    Path(paths.get(recognizer.name, "<unused>")), "<unused>.gguf")
                components = getattr(config, "driver_components", lambda: None)()
                if components is not None:
                    names = [entry_for(component).name for component in components]
                else:
                    builder = SYNTHESIZED_BUILDERS.get(type(config.decomposition).__name__)
                    names = builder_components(builder) if builder is not None else []
            except Exception as exc:
                unavailable[recognizer.name] = f"{type(exc).__name__}: {exc}"
                continue
            for name in dict.fromkeys(names):
                used[name].append(recognizer.name)
    return {name: sorted(models) for name, models in used.items()}, unavailable


# -- the catalogue (D.3) -----------------------------------------------------------------------------


def _link_summary(entry: ComponentEntry) -> str:
    """Each declared link as `field -> describe()`, read off the class.

    `describe()` rather than a sentence written here: it is the same string `LinkChecker.finish`
    prints for a link that never became checkable, so a reader who meets one in a failure can find it
    in this table by matching the text.
    """
    from .spec_protocol import CoveredBy, NestedSpec, declared_raw

    parts = []
    for field, links in entry.links().items():
        for link in links:
            parts.append(f"* `{field}` — {link.describe()}")
            says = _failure_wording(link)
            if says:
                parts.append(f"  <br>*says:* {says}")
    # Declaration-only entries carry no check of their own and are exactly the ones a reader would
    # otherwise misread as "unchecked": `ModularChain.stages` and `FlowMatchingSampler.spec` hold the
    # specs that do the checking. Reporting them as absent is how a catalogue understates coverage.
    for field, value in declared_raw(entry.cls).items():
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(item, NestedSpec):
                parts.append(f"* `{field}` — holds spec(s) with links of their own, checked in "
                             f"{item.where}")
            elif isinstance(item, CoveredBy):
                parts.append(f"* `{field}` — checked as part of `{item.by}`'s link")
    return "\n".join(parts) if parts else "* nothing — every field is `__unchecked__`, with its reason"


def _failure_wording(link) -> str:
    """What this link says when it fails, taken from the link itself rather than restated.

    `ConfigDerived` carries a `str.format` template (the field that keeps `EncoderOutput`'s messages
    verbatim), so the catalogue can quote the real wording; the fixed link kinds phrase their own
    message inside `check`, and quoting those would mean maintaining a second copy -- which is what
    the `Probes` section of `DRIVER-COMPONENTS.md` records instead, from real failing exports.
    """
    template = getattr(link, "message", None)
    if not isinstance(template, str):
        return ""
    first = template.split(". ")[0].strip()
    return first if first.endswith(".") else first + "."


def catalogue() -> str:
    """The component catalogue: per component, what it emits, what it declares, and which models use
    it (BACKLOG.md P4.0.7 / stage D.3).

    Generated from the registry, the classes' own declarations and the families' real component lists,
    so it cannot drift -- the same property `lua_library.catalogue()` has and the reason both exist.
    `DRIVER-COMPONENTS.md` carries the output between generated-block markers, and a test regenerates
    and compares.
    """
    used, unavailable = usage()
    rows = ["| component | class | emits | links | unchecked | used by |", "|---|---|---|---|---|---|"]
    for name, entry in registry().items():
        models = used[name] or ["*nobody* (see below)"]
        rows.append(
            f"| `{name}` | `{entry.cls.__name__}` | {', '.join(entry.emits)} | "
            f"{sum(len(links) for links in entry.links().values())} | {len(entry.unchecked())} | "
            f"{', '.join(models)} |"
        )
    lines = ["\n".join(rows), ""]
    for name, entry in registry().items():
        lines.append(f"### `{name}` — `{entry.cls.__name__}`")
        lines.append("")
        lines.append(entry.summary)
        lines.append("")
        lines.append(f"*Emits:* {', '.join(entry.emits)}. *Used by:* "
                     f"{', '.join(used[name]) or '**no model** — see below'}.")
        lines.append("")
        lines.append(_link_summary(entry))
        if entry.no_user_reason:
            lines.append("")
            lines.append(f"> No model uses it today: {entry.no_user_reason}.")
        lines.append("")
    if unavailable:
        lines.append("Recognizers whose config could not be built without a checkpoint, and are "
                     "therefore not counted in *used by*:")
        for model, reason in sorted(unavailable.items()):
            lines.append(f"* `{model}` — {reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# The catalogue lives in `DRIVER-COMPONENTS.md` between these markers, and the enforcing test
# regenerates and compares -- the doc is a rendering of the code, not a second copy of it.
DOC = Path(__file__).resolve().parents[2] / "DRIVER-COMPONENTS.md"
_BLOCKS = {"component catalogue": lambda: catalogue()}


def _lua_catalogue() -> str:
    from .lua_library import catalogue as lua

    return lua() + "\n"


_BLOCKS["loom_lua catalogue"] = _lua_catalogue


def rendered_doc(current: str) -> str:
    """`current` with every generated block replaced by freshly generated content."""
    out = current
    for name, generate in _BLOCKS.items():
        start = f"<!-- generated: {name} -->"
        end = "<!-- /generated -->"
        head, _, rest = out.partition(start)
        body, _, tail = rest.partition(end)
        if not body and not tail:
            raise ValueError(f"{DOC} has no generated block for {name!r}")
        out = f"{head}{start}\n\n{generate().rstrip()}\n\n{end}{tail}"
    return out


if __name__ == "__main__":  # `python -m loom_mil_compiler.component_registry` regenerates the doc
    DOC.write_text(rendered_doc(DOC.read_text()))
    print(f"regenerated the generated blocks of {DOC}")
