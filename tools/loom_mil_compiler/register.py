import coremltools as ct
from coremltools.converters.mil.converter import ConverterRegistry
from coremltools.converters.mil.mil import Program

from .exporter import LoomGGUFExporter

@ConverterRegistry.backend
class LoomGGUFBackend:
    name = "loom"
    alias_names = ["gguf"]

    def __call__(self, program: Program, **kwargs):
        """
        Compiler Backend Entry Point.
        Ingests the highly optimized MIL Program and lowers it to Loom assets.
        """
        exporter = LoomGGUFExporter(program, **kwargs)
        return exporter.export()
