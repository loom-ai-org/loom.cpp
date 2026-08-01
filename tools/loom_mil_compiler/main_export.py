"""Single entry point for exporting any model this project knows how to export -- BACKLOG.md P3.2's
`main_export()` + `loom-export` CLI, this project's `optimum-cli export onnx --model <id>` equivalent.

    loom-export <model-path> -o <out.gguf>                              # fully automatic
    loom-export <model-path> -o <out.gguf> \\
        --task automatic-speech-recognition --model parakeet-tdt        # explicit

See `registry.py` for how a model is recognized (task, then model within it) and BACKLOG.md's P3.2
entry for why detection is two-axis rather than one flat per-model key; `tasks.py` for the canonical
task vocabulary `--task` accepts.
"""
import argparse
from pathlib import Path

from .registry import default_registry
from .tasks import known_tasks


def main_export(model_path: str, output_path: str, task: str = None, model: str = None) -> str:
    """Exports whatever `model_path` names to `output_path`. `task`/`model` are optional overrides --
    with neither, both axes are auto-detected; with `task` alone, detection is restricted to that task's
    recognizers; `model` requires `task` (it names one specific recognizer within it). Returns
    `output_path`."""
    if model is not None and task is None:
        raise ValueError("--model requires --task (which family's recognizer to look up)")

    registry = default_registry()
    path = Path(model_path)
    recognizer = registry.get(task, model) if model is not None else registry.detect(path, task)
    config = recognizer.build_config(path, output_path)
    return config.export()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Path to a checkpoint directory or file (HF dir, .nemo archive, ...)")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF path")
    parser.add_argument(
        "--task", default=None, choices=known_tasks(),
        help="Restrict/override task detection (the canonical vocabulary, see tasks.py)",
    )
    parser.add_argument("--model", default=None, help="Explicit model override within --task, e.g. 'qwen3'")
    args = parser.parse_args()

    output_path = main_export(args.model_path, args.output, task=args.task, model=args.model)
    print(f"SUCCESS! Exported to: {output_path}")


if __name__ == "__main__":
    main()
