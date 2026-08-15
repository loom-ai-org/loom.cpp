#!/usr/bin/env python3
"""Two GGUFs for the model-contract test: one that DECLARES its contract and one that does not.

The pair is the point. `ModelContract` has to serve both a file from a current exporter and every file
exported before the contract existed, and the second case is not hypothetical -- it is the whole fleet
on disk today. A test with only the declared file would pass while the fallback rotted.

`contract_declared.gguf` is shaped like a Whisper export (fixed clip, timestamped ASR) but carries no
weights or topology beyond the minimum, because nothing here runs a graph: this exercises what the file
SAYS about itself, which is metadata all the way down.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

TOPOLOGY_JSON = '{"version": 1, "nodes": []}'


def _base(path: Path, arch: str) -> GGUFWriter:
    w = GGUFWriter(str(path), "loom-contract-fixture")
    w.add_string("loom.architecture", arch)
    w.add_string("model.graph_topology", TOPOLOGY_JSON)
    return w


def _finish(w: GGUFWriter) -> None:
    w.add_tensor("dummy.weight", np.zeros((2, 2), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_declared(path: Path) -> None:
    w = _base(path, "contract_declared_test")
    w.add_string("loom.task", "automatic-speech-recognition")
    w.add_string("loom.input.kind", "audio")
    w.add_string("loom.output.kind", "token_ids")
    w.add_uint32("loom.sample_rate", 16000)
    w.add_uint32("loom.n_samples", 480000)
    w.add_array("loom.entry_points", ["infer"])
    w.add_string("loom.text.frontend", "vocab")
    w.add_array("loom.text.languages", ["en", "de"])

    # The ASR decode table: the whole reason `transcribe` stopped spelling Whisper's tokens. The ids are
    # arbitrary here -- what is under test is that they are read from the file rather than looked up by
    # name in a vocabulary this fixture does not even have.
    w.add_int32("loom.asr.timestamp_first_id", 50364)
    w.add_float32("loom.asr.timestamp_step_sec", 0.02)
    w.add_array("loom.asr.control_ids", [50363, 50257])
    w.add_array("loom.asr.language_names", ["en", "de"])
    w.add_array("loom.asr.language_ids", [50259, 50261])
    w.add_array("loom.asr.task_names", ["transcribe", "translate"])
    w.add_array("loom.asr.task_ids", [50359, 50358])
    w.add_uint32("loom.asr.prev_context", 448)
    _finish(w)


def write_legacy(path: Path) -> None:
    """A pre-contract file: `loom.architecture` and hparams, and nothing that states what it IS."""
    w = _base(path, "contract_legacy_test")
    w.add_uint32("loom.sample_rate", 16000)
    w.add_uint32("loom.n_samples", 480000)
    w.add_uint32("loom.n_audio_ctx", 1500)
    w.add_uint32("loom.n_text_ctx", 448)
    _finish(w)


# A driver with both LM shapes in one file, selected by an input. `loom::text::generate` tells them
# apart by what comes BACK -- a list means the driver generated internally, a number means one token per
# call -- so a fixture that can produce either is what pins that branch.
#
# `#inputs.tokens + 100` rather than a constant: the returned token then encodes how long the prompt was
# when it was produced, which is how the test sees that the prompt actually grows between calls. A
# constant would pass whether the loop re-fed the prompt or not.
GENERATE_DRIVER = """
function infer(inputs)
    if inputs.mode == 1 then
        return {7, 8, 9}
    end
    return #inputs.tokens + 100
end
"""


def write_generate(path: Path) -> None:
    w = _base(path, "generate_test")
    w.add_string("model.driver_script", GENERATE_DRIVER)
    # The stop token the engine substitutes when a caller names none. 103 is the token this driver
    # produces on its second step from a two-token prompt, so "stops where the file says" is testable
    # without the test naming an eos itself.
    w.add_uint32("tokenizer.ggml.eos_token_id", 103)
    _finish(w)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_declared(out_dir / "contract_declared.gguf")
    write_legacy(out_dir / "contract_legacy.gguf")
    write_generate(out_dir / "generate_driver.gguf")


if __name__ == "__main__":
    main()
