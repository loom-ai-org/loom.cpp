"""
Structural discovery for the modular-export blueprint (EXPORT-IMPROVEMENT-BACKLOG.md item 2).

Replaces `apply_atomic_export`'s scope-based partitioning of one flattened trace -- which has to
*reconstruct* each slice's inputs/outputs after the fact from op scope metadata, the source of two
separately-fixed mis-attribution bugs per EXPORT-BACKLOG.md -- with tracing each real submodule
standalone. Each submodule's MIL graph is then self-contained by construction: there is no cross-slice
variable leakage to detect, because nothing was ever flattened into one function to begin with.
"""
import torch
import torch.nn as nn


def get_by_path(root, path: str):
    """Resolves a dotted attribute path (e.g. "model.layers") against `root`."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def find_repeated_blocks(model: nn.Module):
    """Structurally locates every nn.ModuleList/nn.Sequential with more than one child, keyed by its
    dotted attribute path. Doesn't assume any particular attribute name -- `model.layers`,
    `transformer.h`, `model.decoder.layers` all show up the same way to this check -- and naturally
    handles hybrid architectures (mixed child classes within one ModuleList, e.g. LFM2's
    attention/conv layer mix), since each child is only ever inspected/traced as its own concrete
    class rather than forced through one shared pattern.
    """
    blocks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 1:
            blocks[name] = list(module)
    return blocks


def capture_calls(model: nn.Module, dummy_inputs: dict, targets: dict):
    """Runs ONE real eager forward pass of `model` with `dummy_inputs`, recording the exact
    (args, kwargs) each of `targets` (name -> nn.Module instance) was actually called with.

    This is ground truth for however the model's real forward() invokes each submodule -- no shape,
    dtype, or call convention is ever guessed or hand-derived (the historical alternative this
    replaces: tools/convert_lfm/make_lfm2_gguf.py's LayerSubmodule hand-fabricated dummy
    position-embedding tensors of a hardcoded shape).
    """
    captured = {}
    handles = []

    def _make_hook(name):
        def hook(module, args, kwargs):
            captured[name] = (args, dict(kwargs))
        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=True))

    try:
        with torch.no_grad():
            model(**dummy_inputs)
    finally:
        for h in handles:
            h.remove()

    missing = [name for name in targets if name not in captured]
    if missing:
        raise RuntimeError(
            f"submodule(s) {missing} were never invoked during the dummy forward pass -- dummy_inputs "
            "don't exercise this model's real forward path for the declared ModularExportSpec"
        )
    return captured
