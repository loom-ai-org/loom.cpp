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


class Decomposition:
    """Base for the three real shapes. Subclasses implement `export(config)` and document which hooks
    they read off `config`."""

    def export(self, config) -> str:
        raise NotImplementedError


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

        config.prepare_environment()
        model = config.load_model()
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
    assembled into one multi-Function Program per `ModularExportSpec` -- LFM2's modular profile, and the
    only split mechanism left after `profile="atomic"`'s retirement (EXPORT-ROADMAP.md R7). See
    `modular_export.py`'s own module docstring.

    Config hooks: `prepare_environment()`, `load_model()`, `modular_dummy_inputs()`, `backend_kwargs()`,
    `export_architecture()`, and `max_seq_len`."""

    spec: ModularExportSpec
    # A dummy sequence length deliberately NOT equal to any of the model's own static dims (e.g. LFM2's
    # batch=1, hidden_size=1024, num_attention_heads=16, head_dim=64, vocab_size=65536) -- export_modular
    # marks an axis dynamic when its captured size equals this value, so a collision would wrongly mark a
    # static axis dynamic (or vice versa).
    dummy_seq_len: int = 37

    def export(self, config) -> str:
        from .modular_export import export_modular
        from .register import LoomGGUFBackend

        config.prepare_environment()
        model = config.load_model()
        dummy_inputs = config.modular_dummy_inputs(self.dummy_seq_len)

        print("Tracing each submodule standalone...")
        result = export_modular(
            model, self.spec, dummy_inputs,
            seq_len=self.dummy_seq_len, max_seq_len=config.max_seq_len,
        )

        print("Compiling to GGUF (modular-blueprint profile)...")
        # `profile` is deliberately NOT among backend_kwargs() for this decomposition, and that is
        # load-bearing rather than an omission: exporter.export() dispatches the modular path on
        # `modular_layout is not None`, but topology_ops.py ALSO reads `profile == "monolithic"` as a
        # weight-namespacing switch (`{func_name}.{weight}` unless flattened). A modular Program's
        # per-submodule functions need those namespaces, so this path must leave profile unset. See
        # BACKLOG.md P4.0.3 for the full accounting of what that string really controls.
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

    def export(self, config) -> str:
        import coremltools as ct
        import torch

        from .exporter import LoomGGUFExporter
        from .flow_matching_export import render_driver
        from .multi_phase_export import merge_phase_weights

        config.prepare_environment()
        phase_topologies = {}
        named_weights = []
        for phase in config.phases():
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

        driver_source = render_driver(
            config.driver_script_path.read_text(), config.samplers(),
            topologies=out_exporter.topologies, estimators=config.estimators(),
        )
        out_exporter.write_gguf(driver_source)
        print(f"wrote {config.output_path}")
        return config.output_path
