# Loom MIL Compiler - A coremltools backend plugin for Loom
from .torch_patches import apply_torch_frontend_patches
from .register import LoomGGUFBackend
from . import vits_spline_op  # noqa: F401  registers torch.ops.loom.spline_inverse + its MIL frontend hook

apply_torch_frontend_patches()
