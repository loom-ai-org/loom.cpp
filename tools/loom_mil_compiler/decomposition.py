"""`Decomposition` -- how a model becomes exported topologies (BACKLOG.md P4.0.3).

This is the axis `profile` was introduced to be and never became. It answers one question, separately
from *which* family a config belongs to: does this model export as one traced graph, as independently
traced submodules assembled per a `ModularExportSpec`, or as N independently traced phases merged into
one GGUF? Before this module that answer was encoded in the config's CLASS
(`LMMonolithicCausalModelExportConfig` vs `LMModularCausalModelExportConfig`), which is why exporting
the same LFM2 checkpoint two ways needed two classes and two registry entries built from two different
types rather than one type with a field set differently.

**Why a strategy object rather than a mode string on one config.** The three forms need genuinely
different data -- `Modular` needs a `ModularExportSpec` and a dummy sequence length chosen not to
collide with any static dim; `Flattened` needs a trace length and (for quantizable families) a quantize
mode; `MultiPhase` needs the phase list itself. A single config carrying every field with a string
selecting which subset is live makes invalid states representable and pushes the checking into
`export()`. Each decomposition instead carries its own fields, and a config that names one must supply
that decomposition's own hooks.

**The hooks are a protocol, not a base class.** Each `Decomposition` documents what it reads off the
config it is handed; families implement only what their own decomposition asks for. That is what lets a
family template stay family-shaped (how to load THIS checkpoint, what wrapper THIS model needs) while
the decomposition stays decomposition-shaped (trace once vs. trace per submodule vs. trace per phase).

**What is genuinely a choice, and what is not.** Only the causal-LM family currently has two
decompositions available for the same checkpoint -- LFM2 exports either way, which is a caller decision
(`--model lfm2-monolithic` / `--model lfm2-modular`), not a property of the checkpoint. Kokoro cannot be
exported flattened and Qwen3 has no phases: for those families the decomposition is a structural fact
about the model, declared once in the family's own config rather than chosen per run. The field is
universal; the *choice* is not, and a family that only ever has one answer says so by defaulting it.
"""
from dataclasses import dataclass, field
from typing import Optional

from .modular_export import ModularExportSpec
from .spec_protocol import NestedSpec, Unchecked


class Decomposition:
    """Base for the three real shapes. Subclasses implement `export(config)` and document which hooks
    they read off `config`."""

    def export(self, config) -> str:
        raise NotImplementedError

    def driver_builder(self, config, **context):
        """The `driver_builder.DriverBuilder` that assembles this decomposition's driver for `config`,
        or `None` if this decomposition does not build one.

        **The builder is selected by the decomposition, not owned by the family**
        (`EXPORT-PREPARATION.md` §5 decision 2, BACKLOG.md P4.0.6). The orchestration shape a driver has
        is a property of how the model was decomposed -- one traced graph means prefill-then-argmax, a
        submodule chain means thread the hidden state through it, N phases means whatever that family's
        phases compose into -- so a fourth decomposition (the cross-attention AR decode shape, families
        2 + 6) arrives bringing its own builder rather than every family in it declaring one.

        `config` is passed because the *contents* still are family-specific: `MultiPhase` reads which
        phases and samplers this family declared. What the decomposition fixes is the shape.

        `**context` is whatever the specific decomposition documents needing beyond the config, in the
        same "hooks are a protocol, not a base class" spirit as `export()` itself -- `MultiPhase` takes
        `source=`, the driver text after its generated samplers have been substituted in, because that
        text only exists once the phases have been traced.

        Returning `None` is a real answer, not a stub: `Flattened` covers both the synthesized
        prefill path and the bespoke hand-built-Program workflow, and the latter transpiles a MIL `main`
        function op by op rather than assembling components (`LoomGGUFExporter.transpile_to_lua`).
        """
        return None


@dataclass
class Flattened(Decomposition):
    """One traced forward pass -> one `main_topo` topology. Qwen3, LFM2-monolithic, and all three NeMo
    ASR encoders.

    Config hooks: `prepare_environment()` (optional, defaults to a no-op via `LoomExportConfig`),
    `load_model()`, `build_trace(model)` -> `(wrapper, dummy_inputs, mil_inputs)`, `backend_kwargs()`,
    and `export_architecture()`."""

    def export(self, config) -> str:
        import coremltools as ct
        import torch

        from .register import LoomGGUFBackend
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        model = config.load_model()
        # The config's own links (P4.0.5), before the trace. Nothing walks into `config.output`-style
        # NestedSpec fields: those are checked where their context exists -- for the ASR family, inside
        # the traced wrapper's forward, which is the only moment the real outputs exist.
        checker = LinkChecker()
        checker.check(config)
        checker.provide(model=model)
        checker.finish()
        wrapper, dummy_inputs, mil_inputs = config.build_trace(model)

        traced = torch.jit.trace(wrapper, dummy_inputs)
        program = ct.convert(
            traced,
            inputs=mil_inputs,
            convert_to="milinternal",
            # Load-bearing and NOT a per-model choice: ct.convert()'s default FP16-casts every constant
            # weight even for convert_to="milinternal" -- root-caused via Conformer-CTC's CONV_2D
            # subsampling stage, but it applies to every model this exporter has ever produced. See
            # nemo_asr_export.py's module docstring.
            compute_precision=ct.precision.FLOAT32,
        )

        LoomGGUFBackend()(
            program,
            output_path=config.output_path,
            architecture=config.export_architecture(),
            **config.backend_kwargs(),
        )
        print(f"SUCCESS! Flattened model exported cleanly to: {config.output_path}")
        return config.output_path


@dataclass
class Modular(Decomposition):
    """Independently traced submodules (embedding, rotary table, each decoder layer, final norm, head)
    assembled into one multi-Function Program per `ModularExportSpec` -- LFM2's modular form, and the
    only split mechanism left after the `"atomic"` profile's retirement (EXPORT-ROADMAP.md R7). See
    `modular_export.py`'s own module docstring.

    Config hooks: `prepare_environment()`, `load_model()`, `modular_dummy_inputs()`, `backend_kwargs()`,
    `export_architecture()`, and `max_seq_len`."""

    spec: ModularExportSpec
    # A dummy sequence length deliberately NOT equal to any of the model's own static dims (e.g. LFM2's
    # batch=1, hidden_size=1024, num_attention_heads=16, head_dim=64, vocab_size=65536) -- export_modular
    # marks an axis dynamic when its captured size equals this value, so a collision would wrongly mark a
    # static axis dynamic (or vice versa).
    dummy_seq_len: int = 37

    __links__ = {
        "spec": NestedSpec(
            where="export_modular, which checks every declared attribute path against the real "
                  "nn.Module before it traces anything"
        ),
    }
    __unchecked__ = {
        "dummy_seq_len": Unchecked(
            "a sentinel length, and the one field here where a link would be actively misleading. Its "
            "correctness condition is a NON-collision with any of the model's own static dims -- "
            "checkable in principle, but only against the specific checkpoint, and a wrong value does "
            "not fail: it marks a static axis dynamic (or the reverse) and exports something plausible. "
            "The real guard is the per-model reference test, not a declaration."
        ),
    }

    def export(self, config) -> str:
        from .modular_export import export_modular
        from .register import LoomGGUFBackend
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        model = config.load_model()
        # The config's own links (P4.0.5). `decomposition` is a NestedSpec, so this deliberately does
        # not walk into `self.spec` -- `export_modular` checks that against the real module itself,
        # before it traces anything.
        checker = LinkChecker()
        checker.check(config)
        checker.provide(model=model)
        checker.finish()
        dummy_inputs = config.modular_dummy_inputs(self.dummy_seq_len)

        print("Tracing each submodule standalone...")
        result = export_modular(
            model, self.spec, dummy_inputs,
            seq_len=self.dummy_seq_len, max_seq_len=config.max_seq_len,
        )

        print("Compiling to GGUF (modular blueprint)...")
        # `flat_namespace` is deliberately NOT among backend_kwargs() for this decomposition, and that
        # is load-bearing rather than an omission: a modular Program's per-submodule functions each need
        # their own `{func_name}.{weight}` prefix, which is exactly what leaving this False preserves
        # (topology_ops.py reads it in 8 places). See BACKLOG.md P4.0.3.
        LoomGGUFBackend()(
            result.program,
            output_path=config.output_path,
            architecture=config.export_architecture(),
            modular_layout=result,
            **config.backend_kwargs(),
        )
        print(f"SUCCESS! Modular-blueprint model exported cleanly to: {config.output_path}")
        return config.output_path


@dataclass
class MultiPhase(Decomposition):
    """N independently traced phases, weights merged, written as one GGUF -- every TTS family
    (Kokoro 2 phases, VITS 3, Matcha 4, Supertonic 4, StyleTTS2 3).

    Not an alternative to `Flattened` for any model that uses it: these phases exist because the model
    genuinely cannot be traced as one graph (separate checkpoints, a Python-side sampler between stages,
    or a submodule that is not the model's own `forward`). Declared, not chosen.

    Config hooks: `phases()`, `samplers()`, `estimators()`, `driver_script_path`, `architecture`."""

    def driver_builder(self, config, **context):
        """A `MultiPhaseDriverBuilder` around this family's hand-written driver (P4.0.6/C.3).

        Takes `source=` -- the driver text *after* `render_driver` has substituted the generated
        samplers -- because that text does not exist until every phase has been traced, which is also
        the moment the topologies it will be checked against come into being.
        """
        from .driver_components import MultiPhaseDriverBuilder, RawLuaDriver

        peeled = config.driver_components()
        if peeled is not None:
            # A peeled family (C.4-C.8) builds its driver from components and never reads the whole
            # `.lua`, so `source` -- and with it `render_driver`'s marker substitution -- is unused:
            # its sampler is a `FlowMatchingSampler` component that emits both the function and the
            # line calling it.
            return MultiPhaseDriverBuilder(peeled=peeled)
        return MultiPhaseDriverBuilder(driver=RawLuaDriver(
            source=context["source"], origin=config.driver_script_path.name,
            external=config.external_topologies(),
        ))

    def export(self, config) -> str:
        import coremltools as ct
        import torch

        from .driver_builder import DriverContext
        from .exporter import LoomGGUFExporter
        from .flow_matching_export import render_driver
        from .multi_phase_export import merge_phase_weights
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        # One checker for the whole export (P4.0.5): every spec this config declares shares a single
        # deferral ledger, so `finish()` below is a real statement about the export rather than about
        # whichever call site happened to have context in hand.
        checker = LinkChecker()
        checker.check(config)
        phase_topologies = {}
        named_weights = []
        phases = config.phases()
        # Every phase's axis declarations, checked before the first (slow) trace: an axis name outside
        # axes.py's vocabulary, or a declared_axes entry naming an input this phase does not declare.
        # Both are answerable from the declaration alone, which is why they run here and not after.
        for phase in phases:
            checker.check(phase, f"ExportPhase({phase.name!r})")
        for phase in phases:
            traced = torch.jit.trace(phase.wrapper, phase.dummy_inputs)
            mil_prog = ct.convert(
                traced, inputs=phase.mil_inputs, convert_to="milinternal",
                compute_precision=ct.precision.FLOAT32,
            )
            exporter = LoomGGUFExporter(
                mil_prog, root_axis=phase.root_axis, declared_axes=phase.declared_axes,
            )
            main_func = mil_prog.functions["main"]
            topo = exporter.generate_graph_topology(main_func, phase.name)
            print(f"  {phase.name}: {len(topo['nodes'])} nodes, {len(exporter.weights)} weights")
            phase_topologies[phase.name] = topo
            named_weights.append((phase.name, exporter.weights))

        out_exporter = LoomGGUFExporter(
            None, output_path=config.output_path, architecture=config.architecture,
        )
        out_exporter.topologies = phase_topologies
        out_exporter.weights = merge_phase_weights(named_weights)

        peeled = config.driver_components()
        # `render_driver`'s marker substitution is the unpeeled path only: a peeled family's sampler is
        # a `FlowMatchingSampler` component that emits both the generated function and the line calling
        # it, and its specs are checked through the builder like every other component's.
        driver_source = None if peeled is not None else render_driver(
            config.driver_script_path.read_text(), config.samplers(),
            topologies=out_exporter.topologies, estimators=config.estimators(),
            checker=checker,
        )
        # The driver goes through a DriverBuilder as of P4.0.6/C.3, which for now holds one component:
        # the hand-written source, adopted whole. That is not cosmetic even while the emitted text is
        # byte-identical -- the adoption parses the driver's own `loom.run_subgraph` call sites and
        # declares each against the topologies just traced, which is the first time these five drivers
        # have been cross-checked against the model they ship with at all.
        driver_source = self.driver_builder(config, source=driver_source).render(
            DriverContext(
                topologies=out_exporter.topologies,
                axes={phase.name: phase.root_axis for phase in phases},
                weights=out_exporter.weights,
            ),
            checker=checker,
        )
        # Nothing may be written until every declared link has actually run. A link that deferred and
        # never became checkable reads as validated and is not -- see spec_protocol's module docstring.
        checker.finish()
        out_exporter.write_gguf(driver_source)
        print(f"wrote {config.output_path}")
        return config.output_path
