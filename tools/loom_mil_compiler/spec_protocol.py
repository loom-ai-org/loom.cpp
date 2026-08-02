"""The spec protocol (BACKLOG.md P4.0.5, `EXPORT-PREPARATION.md` §2/§6 stage B).

Every spec in this tree earns its existence by being **checked against the real model**, and before this
module each spec class hand-wrote its own `validate()` to do it. Four of them existed, all good, none
alike: `EncoderOutput.validate` (a family claim vs. the checkpoint's own `d_model`),
`EstimatorSpec.validate_against_topology` (declared inputs vs. supplied inputs), `ModularExportSpec`'s
attribute paths (an incidental `AttributeError` from `get_by_path`), and `LoomGGUFExporter.
_validate_input_axes` (one root axis, everything else declared). A fifth family author would have
written a fifth, of whatever quality they happened to reach for.

The observation this module is built on, which is the author's and is the whole design:

    All spec'ing classes validate and check generally analogous dependencies, links and
    correspondences; the specificities are informed as data.

So the *relationship kinds* become a shared, library-side vocabulary with the predicate and the raising
in one place, and a spec class declares `field -> link kind` once. What stays per-model is data: which
topology, which attribute path, which config field to read. Three consequences, in the order they
matter:

1. **A new family gets validation for free**, in the shape reviewers already trust.
2. **A text (JSON/YAML/TOML) front-end becomes cheap later rather than duplicative** -- the values are
   data and the predicates are library-side, so `asdict`/`fromdict` plus a schema is all it needs. That
   is deliberately *deferred* until families 2/6/10/11 have shown their shapes, so the schema is not
   frozen against four unknown ones.
3. **It composes with the driver builder** (P4.0.6): a `DriverComponent` declares its links and they are
   checked before it emits anything.

**The acceptance criterion, and it is the one that shaped this file.** Each of the four existing
validators had to be re-expressible with *no loss of error-message quality*. This tree's errors name the
offending input, the expected channel count and the config field it came from; a generic checker that
degrades those into "validation failed" is a regression, not a refactor. That is why `ConfigDerived`
takes a message template with `{spec.<attr>}` access rather than formatting a canned sentence, and why
`TopologyInput` reproduces `EstimatorSpec`'s bidirectional missing/unsupplied wording verbatim rather
than reporting the first offender it finds. The retrofit tests assert on message *content*.

## Declaring links

A spec class declares a class-level `__links__` dict, and (for the standing rule below) `__unchecked__`
for fields that genuinely cannot be checked:

    @dataclass
    class EstimatorSpec:
        topology: str
        inputs: list

        __links__ = {
            "topology": TopologyName(),
            "inputs": TopologyInput(FieldRef("topology"), exact=True),
        }

Keys are usually field names -- the link is handed that field's value. A key that names no field is a
*named check* instead: `ConfigDerived` reads through its own `claim`/`measured` callables and ignores
the field value entirely, which is what lets `EncoderOutput` (an `Enum`, with no fields at all) declare
three of them.

**The standing rule this enables**, and the reason `__unchecked__` is a real declaration rather than a
default: *every spec field must be either checkable against the real model/topology, or explicitly
documented as unchecked.* `test_spec_protocol.py` walks every registered spec class and fails on any
field that is neither. That is what makes a component trustworthy enough to reuse without reading its
implementation -- the actual requirement behind P4.0.7's "marketplace".

## Deferral: the detail that must not be skipped

The context a link needs arrives at different times -- the loaded `nn.Module` after `load_model()`, the
topologies only after tracing, the merged weights after that, the driver IR last. A link whose context
is not available yet therefore cannot be checked yet.

**A link that is never checked is reported, not silently skipped.** `LinkChecker` keeps every deferred
link, retries them each time `provide()` brings a new slot, and `finish()` raises listing whatever is
still outstanding. Without that, "validated" quietly comes to mean "validated where convenient", which
is precisely the failure mode this protocol exists to prevent -- and it would be invisible, because a
skipped check and a passing check look identical from outside.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Callable, Optional, Tuple

from . import axes
from .shape_expr import UnsupportedShapeExpression, parse


class LinkError(ValueError):
    """A declared link that does not hold against the real model/topology/driver.

    Subclasses `ValueError` deliberately: every validator this protocol generalizes raised `ValueError`,
    and callers (tests, `main_export`) catch that. The type is narrower; the contract is unchanged.
    """


# -- the context ------------------------------------------------------------------------------------

# The slots a link can require. Named constants rather than bare strings so a typo in a `requires`
# tuple is an ImportError at module load instead of a link that defers forever and only surfaces as a
# `finish()` failure.
TOPOLOGIES = "topologies"
MODEL = "model"
OUTPUTS = "outputs"
WEIGHTS = "weights"
DRIVER = "driver"

_SLOTS = (TOPOLOGIES, MODEL, OUTPUTS, WEIGHTS, DRIVER)


class LinkCheckContext:
    """The real objects links are checked against, populated as the export produces them.

    Mutable and accumulating rather than a frozen value, because that is what the export actually does:
    `load_model()` yields `model`, tracing yields `topologies`, the phase merge yields `weights`, the
    driver builder yields `driver`. Every slot is `None` until it exists, and `has()` is the only thing
    that reads that -- a link never sees a half-populated context, it is deferred instead.
    """

    def __init__(self, **slots):
        for name in _SLOTS:
            setattr(self, name, None)
        self.provide(**slots)

    def provide(self, **slots) -> None:
        """Adds real objects to the context. Unknown slot names raise rather than being stored, so a
        caller cannot quietly populate something no link will ever read."""
        for name, value in slots.items():
            if name not in _SLOTS:
                raise LinkError(f"unknown link-check context slot {name!r}; known slots: {list(_SLOTS)}")
            setattr(self, name, value)

    def has(self, slot: str) -> bool:
        return getattr(self, slot) is not None

    def missing(self, requires) -> Tuple[str, ...]:
        return tuple(slot for slot in requires if not self.has(slot))


@dataclass(frozen=True)
class LinkSite:
    """Where a link came from, for the message: which spec (as it should be *named* to a reader) and
    which field. `label` is the spec's own `link_label()` when it has one -- `FlowMatchingSpec` names
    itself by its `func_name`, because "FlowMatchingSpec" alone does not tell a reader which of a
    model's samplers failed."""

    label: str
    field: str


@dataclass(frozen=True)
class FieldRef:
    """A reference to a *sibling field* of the same spec, for links whose subject depends on another
    declared value -- `TopologyInput(FieldRef("topology"))` means "the topology this spec's `topology`
    field names", which is what keeps `EstimatorSpec`'s two fields from having to be checked together
    by hand."""

    field: str

    def resolve(self, spec):
        return getattr(spec, self.field)


def _deref(ref, spec):
    return ref.resolve(spec) if isinstance(ref, FieldRef) else ref


# -- the link kinds ---------------------------------------------------------------------------------


class Link:
    """One checkable relationship between a spec's declared value and the real thing.

    `requires` names the context slots `check` reads; the checker defers rather than calling `check`
    until they are all present. `check` raises `LinkError` or returns.
    """

    requires: Tuple[str, ...] = ()

    def check(self, spec, value, ctx: LinkCheckContext, site: LinkSite) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        """One line naming what this link would have checked, for `LinkChecker.finish`'s report on
        links that were never checkable."""
        return type(self).__name__


@dataclass(frozen=True)
class TopologyName(Link):
    """The value must name one of the exported topologies.

    Generalizes `flow_matching_export._check`'s first half, whose message is reproduced verbatim: a
    spec that names a topology the export did not produce is a typo or a stale rename, and listing the
    names that *do* exist is what turns it into a one-line fix.
    """

    requires = (TOPOLOGIES,)

    def check(self, spec, value, ctx, site):
        if value not in ctx.topologies:
            raise LinkError(
                f"{site.label} names topology {value!r}, which is not among the exported topologies "
                f"{sorted(ctx.topologies)}."
            )


@dataclass(frozen=True)
class TopologyInput(Link):
    """The value is the list of topology input names the spec supplies; checked against what the
    topology really declares.

    Generalizes `EstimatorSpec.validate_against_topology`, message and all. `exact=True` is the real
    contract for a `run_subgraph` call -- an input the topology does not declare is rejected by the
    engine, and one it declares but nobody supplies is an uninitialised tensor -- so the check runs
    *both* ways and the message says which offenders fall on which side. `exact=False` exists for a
    spec that legitimately supplies a subset (a caller filling the rest).
    """

    topology: object  # str | FieldRef
    exact: bool = True

    requires = (TOPOLOGIES,)

    def check(self, spec, value, ctx, site):
        name = _deref(self.topology, spec)
        topology = ctx.topologies.get(name)
        if topology is None:
            # TopologyName is the link that reports this, with the list of real names; if the spec did
            # not also declare one, say so rather than raising an opaque AttributeError here.
            raise LinkError(
                f"{site.label} names topology {name!r}, which is not among the exported topologies "
                f"{sorted(ctx.topologies)}."
            )
        declared = [inp["name"] for inp in topology.get("inputs", [])]
        # A field naming ONE input is the same declaration as a field naming a list of one; iterating a
        # bare string a character at a time is never what a caller meant.
        supplied = [value] if isinstance(value, str) else list(value)
        missing = [n for n in supplied if n not in declared]
        unsupplied = [n for n in declared if n not in supplied] if self.exact else []
        if missing or unsupplied:
            raise LinkError(
                f"{site.label} does not match topology {name!r}: "
                + (f"supplies input(s) it does not declare: {missing}; " if missing else "")
                + (f"leaves declared input(s) unsupplied: {unsupplied}; " if unsupplied else "")
                + f"topology declares {declared}, spec supplies {supplied}."
            )

    def describe(self):
        return f"TopologyInput({self.topology!r}, exact={self.exact})"


@dataclass(frozen=True)
class TopologyOutputArity(Link):
    """The named topology must declare exactly `count` outputs.

    Generalizes `driver_ir.check_subgraph_calls`' output half, and reaches somewhere that check cannot:
    `check_subgraph_calls` sees a *driver* whose call captures N outputs, but a spec that *generates*
    that call (`FlowMatchingSpec`) knows the arity it is about to hard-code before any driver text
    exists. Emitting `local v = loom.run_subgraph(...)` and then indexing `v[i]` is only correct for a
    single-output estimator -- with two, `v` silently becomes the first output's data and the loop
    integrates the wrong tensor.

    A topology declares its outputs as either `outputs` (plural array, P2 multi-output) or `output`
    (singular string) -- the same normalization `driver_ir._topology_output_names` performs.
    """

    topology: object  # str | FieldRef
    count: int = 1

    requires = (TOPOLOGIES,)

    def check(self, spec, value, ctx, site):
        name = _deref(self.topology, spec)
        topology = ctx.topologies.get(name)
        if topology is None:
            raise LinkError(
                f"{site.label} names topology {name!r}, which is not among the exported topologies "
                f"{sorted(ctx.topologies)}."
            )
        declared = _topology_output_names(topology)
        if len(declared) != self.count:
            raise LinkError(
                f"{site.label} is built for a topology declaring {self.count} output(s), but "
                f"{name!r} declares {len(declared)}: {declared}."
            )

    def describe(self):
        return f"TopologyOutputArity({self.topology!r}, count={self.count})"


def _topology_output_names(topo: dict) -> list:
    """Both spellings of a topology's declared outputs, normalized -- see `driver_ir` for the same
    function and `graph_topology.h` for why the distinction exists at all."""
    if "outputs" in topo:
        return list(topo["outputs"])
    single = topo.get("output")
    return [single] if single is not None else []


@dataclass(frozen=True)
class ModuleAttrPath(Link):
    """The value is a dotted attribute path that must resolve on the loaded `nn.Module`.

    Generalizes `ModularExportSpec`'s paths, and **upgrades the behaviour rather than reproducing it**.
    Today a wrong path raises whatever Python raises (`AttributeError`), wherever `get_by_path`'s
    traversal happens to reach it, which for `suffix_attrs` is after the prefix has already been traced.
    A link check raises up front, naming the path, the component that failed, and the type of the
    module it failed on -- the last being the thing that actually tells a caller whether they typo'd the
    name or targeted the wrong nesting level.

    `expect` optionally names a type the resolved object must be an instance of.
    """

    expect: Optional[type] = None

    requires = (MODEL,)

    def check(self, spec, value, ctx, site):
        obj = ctx.model
        walked = []
        for part in str(value).split("."):
            if not hasattr(obj, part):
                where = ".".join(walked) or type(ctx.model).__name__
                raise LinkError(
                    f"{site.label}.{site.field} = {value!r} does not resolve on the loaded model: "
                    f"{type(obj).__name__} (at {where}) has no attribute {part!r}."
                )
            walked.append(part)
            obj = getattr(obj, part)
        if self.expect is not None and not isinstance(obj, self.expect):
            raise LinkError(
                f"{site.label}.{site.field} = {value!r} resolves to {type(obj).__name__}, "
                f"but this field must name a {self.expect.__name__}."
            )

    def describe(self):
        return "ModuleAttrPath()" if self.expect is None else f"ModuleAttrPath(expect={self.expect.__name__})"


@dataclass(frozen=True)
class Axis(Link):
    """The value names axes from `axes.py`'s vocabulary, and nothing else.

    Generalizes the *declaration* side of `LoomGGUFExporter._validate_input_axes` (P4.0.2). The
    exporter's own two raises stay exactly where they are -- they operate on the traced program, which
    no spec can see -- but the question "is `root_axis` a real axis name?" is answerable from the
    declaration alone, and today is not asked at all: a typo'd `"n_token"` is a perfectly good dict key
    and substitutes silently, producing shape expressions over a symbol nothing else in the model uses.

    `form` says how to read axis names out of the field, because the two declaration sites carry
    different shapes -- and that is the specificity-as-data the protocol is built on rather than two
    link kinds:

    * `"name"` -- the value is one axis name (`ExportPhase.root_axis`).
    * `"declaration_table"` -- the value is `{input name: {axis index: expression}}`
      (`ExportPhase.declared_axes`), whose expressions are `symbol_env.cpp` grammar over axis names,
      e.g. Kokoro's `"600*n_enc_frames+20"`. Every free symbol must be in the vocabulary.
    """

    form: str = "name"

    requires = ()

    def check(self, spec, value, ctx, site):
        for where, text in self._occurrences(value, site):
            self._check_one(where, text, site)

    def _occurrences(self, value, site):
        if self.form == "name":
            yield site.field, value
            return
        if self.form != "declaration_table":
            raise LinkError(f"unknown Axis form {self.form!r}; known forms: ['name', 'declaration_table']")
        for input_name, per_axis in (value or {}).items():
            for axis_index, expr in (per_axis or {}).items():
                yield f"{site.field}[{input_name!r}][{axis_index}]", expr

    @staticmethod
    def _check_one(where, text, site):
        known = sorted(getattr(axes, name).name for name in axes.__all__)
        try:
            expr = parse(str(text))
        except UnsupportedShapeExpression as exc:
            raise LinkError(
                f"{site.label}.{where} = {text!r} is not a valid shape expression: {exc}"
            ) from exc
        unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in known)
        if unknown:
            raise LinkError(
                f"{site.label}.{where} = {text!r} names {unknown}, which is not in the axis vocabulary "
                f"{known} (axes.py). Declaring an axis means naming one of those, not inventing a "
                f"per-model symbol."
            )

    def describe(self):
        return f"Axis(form={self.form!r})"


@dataclass(frozen=True)
class ConfigDerived(Link):
    """A claim the spec makes about the checkpoint, which must equal a property measured off the real
    object.

    Generalizes `EncoderOutput.validate` -- all three of its checks, which look different but are one
    shape: read a number the spec's declared family implies, measure the same number on what the model
    actually did, compare. The two readers are the data (`claim` and `measured` are per-spec callables
    over `(spec, ctx)`); the comparison, the deferral and the raising are library-side.

    `message` is a `str.format` template rather than a canned sentence, and that is the acceptance
    criterion of the whole protocol in one field: this tree's ASR errors name the checkpoint's own
    `cfg.encoder.d_model` and the axis it was measured on, and degrading them to "expected 512, got 176"
    would be a regression. Available placeholders:

    * `{spec}` -- the spec object, with attribute access (`{spec.channel_axis}`, `{spec.name}`)
    * `{claimed}` / `{actual}` -- the two compared values
    * `{detail}` -- whatever `detail(spec, ctx)` returns, for context neither of those carries (the
      offending tensor's full shape, say)
    * `{model_type}` -- `type(ctx.model).__name__`, which every message of this shape wants: the point
      of a config-derived check is that the checkpoint is not what the spec claimed, and naming the
      class that was actually loaded is how a reader sees that at a glance
    * `{label}` -- `site.label`
    """

    claim: Callable
    measured: Callable
    message: str
    detail: Optional[Callable] = None
    # Slots beyond MODEL that the readers need. `EncoderOutput`'s three all read the traced forward's
    # return value, which only exists inside the trace.
    needs: Tuple[str, ...] = (MODEL,)

    @property
    def requires(self):
        return self.needs

    def check(self, spec, value, ctx, site):
        claimed = self.claim(spec, ctx)
        actual = self.measured(spec, ctx)
        if claimed == actual:
            return
        detail = self.detail(spec, ctx) if self.detail is not None else None
        raise LinkError(self.message.format(
            spec=spec, claimed=claimed, actual=actual, detail=detail, label=site.label,
            model_type=type(ctx.model).__name__ if ctx.model is not None else None,
        ))

    def describe(self):
        return f"ConfigDerived(needs={list(self.needs)})"


@dataclass(frozen=True)
class WeightName(Link):
    """The value must name a tensor present in the export's merged weight dict.

    Generalizes the half of `multi_phase_export.merge_phase_weights` that is about names rather than
    collisions: a spec that reaches for a weight by name (a driver component wanting a lookup table, a
    phase declaring a shared embedding) is only correct if the name survived the merge, and per-phase
    prefixing conventions make that easy to get subtly wrong. Checkable only after every phase has been
    traced and merged, so it defers until then.
    """

    requires = (WEIGHTS,)

    def check(self, spec, value, ctx, site):
        names = list(value) if isinstance(value, (list, tuple)) else [value]
        unknown = [n for n in names if n not in ctx.weights]
        if unknown:
            raise LinkError(
                f"{site.label}.{site.field} names weight(s) {unknown}, which are not in the merged "
                f"weight dict ({len(ctx.weights)} tensors). Weight names carry each phase's own prefix "
                f"after the merge -- see merge_phase_weights."
            )


@dataclass(frozen=True)
class DriverSymbol(Link):
    """The value must be a symbol the emitted driver really defines before it is read.

    Generalizes `driver_ir.validate`, and exists for the same reason `TopologyOutputArity` does: that
    check runs on a finished `Function`, while a spec that *contributes* statements to one (P4.0.6's
    `DriverComponent`) declares the symbols it expects the surrounding driver to have already bound.
    Deferred until the driver IR exists, which is the last thing an export builds.
    """

    requires = (DRIVER,)

    def check(self, spec, value, ctx, site):
        from .driver_ir import Function

        driver = ctx.driver
        # Either the entry function alone, or the whole built script -- a driver is a Lua *module*, so
        # a top-level `local function` in the prelude (every generated sampler is one) is a symbol the
        # driver really defines, and a link that could not see those would fail exactly the components
        # P4.0.6 introduced it for.
        function = getattr(driver, "entry", driver)
        if not isinstance(function, Function):
            raise LinkError(
                f"{site.label}.{site.field}: the driver context slot holds {type(driver).__name__}, "
                f"not a driver_ir.Function."
            )
        names = list(value) if isinstance(value, (list, tuple)) else [value]
        defined = set(function.params)
        for stmt in function.body:
            defined.update(stmt.defines())
        defined.update(_top_level_function_names(getattr(driver, "prelude", ())))
        unknown = [n for n in names if n not in defined]
        if unknown:
            raise LinkError(
                f"{site.label}.{site.field} reads driver symbol(s) {unknown}, which the emitted driver "
                f"'{function.name}' never defines (defines: {sorted(defined)})."
            )


_TOP_LEVEL_FUNCTION = None  # compiled on first use; see _top_level_function_names


def _top_level_function_names(prelude_lines) -> set:
    """Names bound by a top-level `function f(...)` / `local function f(...)` in a driver's prelude."""
    global _TOP_LEVEL_FUNCTION
    if _TOP_LEVEL_FUNCTION is None:
        import re
        _TOP_LEVEL_FUNCTION = re.compile(r"^\s*(?:local\s+)?function\s+([A-Za-z_]\w*)\s*\(")
    names = set()
    for line in prelude_lines or ():
        match = _TOP_LEVEL_FUNCTION.match(line)
        if match:
            names.add(match.group(1))
    return names


# -- combinators ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WhenSet(Link):
    """`inner`, but a `None` value means the field is simply not in use -- not a failure and not a
    deferral. This is how an optional declaration (`ModularExportSpec.aux_attr`) stays link-declared
    rather than being pushed into `__unchecked__`, which would lose the check for every family that
    *does* set it."""

    inner: Link

    @property
    def requires(self):
        return self.inner.requires

    def check(self, spec, value, ctx, site):
        if value is None:
            return
        self.inner.check(spec, value, ctx, site)

    def describe(self):
        return f"WhenSet({self.inner.describe()})"


@dataclass(frozen=True)
class EachOf(Link):
    """`inner`, applied to every element of a list-valued field (`ModularExportSpec.suffix_attrs`), with
    the element's index folded into the reported field name so a failure names *which* one."""

    inner: Link

    @property
    def requires(self):
        return self.inner.requires

    def check(self, spec, value, ctx, site):
        for i, item in enumerate(value or []):
            self.inner.check(spec, item, ctx, LinkSite(site.label, f"{site.field}[{i}]"))

    def describe(self):
        return f"EachOf({self.inner.describe()})"


# -- declaration + the standing rule ----------------------------------------------------------------


@dataclass(frozen=True)
class Unchecked:
    """An explicit "this field cannot be checked against anything real, and here is why".

    Required rather than implied: a field with no link and no `Unchecked` entry fails the standing-rule
    test. The reason is the point -- "cosmetic, rendered into a comment" and "nobody got around to it"
    are different statements, and only one of them should survive review.
    """

    reason: str


@dataclass(frozen=True)
class CoveredBy:
    """"This field IS checked, but as part of another field's link."

    Declared in `__links__` alongside real links, and emits no check of its own. The case that forced it
    is `FlowMatchingSpec`: `carried_input`, `time_input` and `fixed_inputs` are three fields that only
    mean anything as the one per-step argument table they compose into, and that table is what the
    estimator's declared inputs are compared against -- in a single message naming both the inputs
    supplied-but-undeclared and the inputs declared-but-unsupplied. Splitting that into three per-field
    links would report one offender at a time and lose the half of the message that says what is
    *missing*.

    Marking those three `Unchecked` would be false -- they are checked, thoroughly -- and leaving them
    undeclared would fail the standing rule for the wrong reason. `CoveredBy` says which link does it,
    and `undeclared_fields` verifies the named field really is link-declared, so a rename cannot quietly
    turn this into an exemption.
    """

    by: str


@dataclass(frozen=True)
class NestedSpec:
    """"This field holds another spec, which declares and runs its own links."

    The fourth declaration kind, alongside a real `Link`, `Unchecked` and `CoveredBy`, and like
    `CoveredBy` it emits no check of its own. Configs hold specs -- `ASRNemoEncoderExportConfig.output`
    is an `EncoderOutput`
    with three `ConfigDerived` links of its own, `Modular.spec` is a `ModularExportSpec` -- and the
    standing rule has to say something about those fields. `Unchecked` would be false and a link would
    be a duplicate.

    **Deliberately not auto-recursing.** The checker does not walk into a nested spec, because the
    nested spec's links are frequently checkable only at a site the outer checker cannot reach:
    `EncoderOutput`'s three need the traced forward's real return value, which exists for exactly one
    instant inside the wrapper's `forward` during tracing and nowhere else (running them out of band
    would consume RNG a traced constant could depend on). `where` records that site in prose, so a
    reader of the declaration can find the check rather than assuming it happens here.
    """

    where: str


# Entries that satisfy the standing rule without emitting a check of their own.
_DECLARATION_ONLY = (CoveredBy, NestedSpec)


def _entries(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _merged(spec_class, attr: str) -> dict:
    """`attr` merged along the MRO, base-first, so a subclass adds to its base's declarations rather
    than shadowing them.

    Load-bearing rather than a nicety: `architecture`, `output_path` and `decomposition` are
    `LoomExportConfig`'s fields and belong to every family. Plain attribute lookup would find only the
    subclass's dict, so each of the five family configs would have to restate all three -- and the
    standing rule would be enforcing copy-paste."""
    merged = {}
    for klass in reversed(getattr(spec_class, "__mro__", [spec_class])):
        merged.update(getattr(klass, attr, {}) or {})
    return merged


def declared_raw(spec_class) -> dict:
    """Every `__links__` entry for `spec_class`, declarations included, merged along the MRO."""
    return _merged(spec_class, "__links__")


def declared_links(spec) -> dict:
    """`{field or check name: [Link]}` for `spec`'s class, with single links normalized to lists.

    `CoveredBy`/`NestedSpec` entries are declarations, not checks, so they are filtered out here and
    only read by `undeclared_fields`."""
    out = {}
    for name, value in declared_raw(type(spec)).items():
        links = [v for v in _entries(value) if not isinstance(v, _DECLARATION_ONLY)]
        if links:
            out[name] = links
    return out


def declared_unchecked(spec_class) -> dict:
    return _merged(spec_class, "__unchecked__")


def undeclared_fields(spec_class) -> list:
    """Dataclass fields of `spec_class` that are neither link-declared, `CoveredBy` another field, nor
    explicitly `__unchecked__`.

    The standing rule, as a function so the enforcing test is three lines and a new family author can
    run the same check on their own class before opening a review.
    """
    if not is_dataclass(spec_class):
        return []
    links = declared_raw(spec_class)
    unchecked = declared_unchecked(spec_class)
    return [f.name for f in fields(spec_class) if f.name not in links and f.name not in unchecked]


def dangling_coverage(spec_class) -> list:
    """`CoveredBy` entries whose named field is not itself link-declared -- i.e. a claim that some other
    link does the checking, where no such link exists. Rename-proofing: without this, deleting the link
    a field defers to silently converts `CoveredBy` into an exemption."""
    raw = declared_raw(spec_class)
    real = {name for name, value in raw.items()
            if any(not isinstance(v, _DECLARATION_ONLY) for v in _entries(value))}
    dangling = []
    for name, value in raw.items():
        for entry in _entries(value):
            if isinstance(entry, CoveredBy) and entry.by not in real:
                dangling.append(f"{name} -> {entry.by}")
    return dangling


def spec_label(spec) -> str:
    """How a spec should be named in an error. Defaults to the class name; a spec with more than one
    instance per model overrides `link_label()` so a reader can tell which one failed -- Matcha declares
    one `FlowMatchingSpec` but nothing stops a model declaring three."""
    label = getattr(spec, "link_label", None)
    return label() if callable(label) else type(spec).__name__


# -- checking ---------------------------------------------------------------------------------------


@dataclass
class _Pending:
    spec: object
    site: LinkSite
    link: Link


class LinkChecker:
    """Runs declared links as the context fills in, and refuses to let an unchecked one pass silently.

    Typical use, from a `Decomposition.export`:

        checker = LinkChecker()
        checker.check(config)                       # nothing to check against yet -- all deferred
        checker.provide(model=model)                # ModuleAttrPath/ConfigDerived links run now
        ...trace...
        checker.provide(topologies=topologies)      # TopologyName/TopologyInput links run now
        checker.finish()                            # raises if anything never became checkable

    `finish()` is the load-bearing call. Deferral is not a way to opt out: a link whose slot never
    arrives is a link the author believed was running and that never ran.
    """

    def __init__(self, ctx: Optional[LinkCheckContext] = None):
        self.ctx = ctx if ctx is not None else LinkCheckContext()
        self._pending: list = []

    def check(self, spec, label: Optional[str] = None) -> None:
        """Registers every link `spec` declares, running the ones whose context is already present."""
        resolved = label or spec_label(spec)
        for name, links in declared_links(spec).items():
            for link in links:
                self._run_or_defer(_Pending(spec, LinkSite(resolved, name), link))

    def check_all(self, specs) -> None:
        """`check` over an iterable of `(spec, label)` pairs or bare specs."""
        for item in specs:
            if isinstance(item, tuple):
                self.check(item[0], item[1])
            else:
                self.check(item)

    def provide(self, **slots) -> None:
        """Adds context, then retries every deferred link that just became checkable."""
        self.ctx.provide(**slots)
        still_pending, ready = [], []
        for pending in self._pending:
            (ready if not self.ctx.missing(pending.link.requires) else still_pending).append(pending)
        self._pending = still_pending
        for pending in ready:
            self._run(pending)

    def finish(self) -> None:
        """Raises if any link never became checkable. Call this once, at the end of an export."""
        if not self._pending:
            return
        lines = [
            f"  {p.site.label}.{p.site.field}: {p.link.describe()} needs "
            f"{list(self.ctx.missing(p.link.requires))}"
            for p in sorted(self._pending, key=lambda p: (p.site.label, p.site.field))
        ]
        raise LinkError(
            f"{len(self._pending)} declared link(s) were never checked, because the export never "
            f"produced what they check against:\n" + "\n".join(lines) + "\n"
            "A declared link that never runs is worse than no declaration: it reads as validated. "
            "Either populate the missing context (LinkChecker.provide) or mark the field __unchecked__ "
            "with the reason."
        )

    @property
    def deferred(self) -> list:
        """The still-unchecked links, for a caller that wants to inspect rather than raise."""
        return list(self._pending)

    def _run_or_defer(self, pending: _Pending) -> None:
        if self.ctx.missing(pending.link.requires):
            self._pending.append(pending)
        else:
            self._run(pending)

    @staticmethod
    def _run_link(pending: _Pending, ctx: LinkCheckContext) -> None:
        value = getattr(pending.spec, pending.site.field, None)
        pending.link.check(pending.spec, value, ctx, pending.site)

    def _run(self, pending: _Pending) -> None:
        self._run_link(pending, self.ctx)


def check_links(spec, ctx: Optional[LinkCheckContext] = None, label: Optional[str] = None,
                strict: bool = True, **slots) -> list:
    """Checks `spec`'s links against a context built here, and returns the deferred ones.

    The one-shot form, for a validator with a single natural call site and everything it needs already
    in hand -- `EncoderOutput`'s three links, checked inside the traced wrapper's `forward` where the
    model's real outputs exist and nowhere else. `strict=True` (the default) raises on anything that
    could not be checked, which is the same rule `LinkChecker.finish` applies; pass `strict=False` only
    when the caller is itself tracking the deferral.
    """
    checker = LinkChecker(ctx if ctx is not None else LinkCheckContext(**slots))
    if ctx is not None and slots:
        checker.ctx.provide(**slots)
    checker.check(spec, label)
    if strict:
        checker.finish()
    return checker.deferred
