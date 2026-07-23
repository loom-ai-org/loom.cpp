# Loom MIL Compiler - A coremltools backend plugin for Loom
from .torch_patches import apply_torch_frontend_patches
from .register import LoomGGUFBackend

apply_torch_frontend_patches()
