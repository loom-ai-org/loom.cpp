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


# A fixed-clip ASR model that emits timestamp tokens, for the long-form seek.
#
# THE BUG THIS EXISTS FOR. A segment end is a float number of seconds, so the sample index it maps to is
# almost never exact: with the step read from the file as f32, 550 steps of 0.02 s come to 10.99999975
# rather than 11, and 11 s of audio at 16 kHz TRUNCATED to 175999 -- one sample short of the end. The
# loop then ran a second window over four samples of real audio and 30 s of zero padding, and Whisper
# transcribed the silence. Rounding fixed it, and nothing hermetic could have caught it: the only ASR
# fixture that emits timestamps is a real 970 MB Whisper export.
#
# The driver returns one closed segment covering the whole clip -- `<|0.00|> hello <|11.00|>` in ids --
# so a correct seek advances to exactly the end of the audio and stops. An off-by-one in the other
# direction is visible as a second window, which is what the test asserts on.
TIMESTAMP_DRIVER = """
function infer(inputs)
    -- Ignores the waveform: what is under test is the loop AROUND this call, not a decode. Returning
    -- the same span whatever it is handed is what makes a second window unambiguous evidence of a seek
    -- that did not reach the end, rather than of the model finding something in the padding.
    return {TS_FIRST, HELLO_ID, TS_FIRST + STEPS_PER_CLIP}
end
"""


def write_timestamped_asr(path: Path) -> None:
    from gguf import GGUFWriter

    # 11 s at 16 kHz, one timestamp step per 0.02 s -- Whisper's own geometry, at Whisper's own ids, so
    # the arithmetic under test is the arithmetic that failed.
    sample_rate, clip_seconds, step_sec = 16000, 30.0, 0.02
    ts_first = 50364
    steps_for_11s = int(11.0 / step_sec)  # 550

    w = _base(path, "timestamped_asr_test")
    w.add_string("loom.task", "automatic-speech-recognition")
    w.add_string("loom.input.kind", "audio")
    w.add_string("loom.output.kind", "token_ids")
    w.add_uint32("loom.sample_rate", sample_rate)
    w.add_uint32("loom.n_samples", int(sample_rate * clip_seconds))
    w.add_int32("loom.asr.timestamp_first_id", ts_first)
    # f32 ON PURPOSE. Storing this as f64 would sidestep the precision loss and the test would pass
    # against the truncating code it exists to catch -- the fixture has to reproduce the file format's
    # own rounding, not work around it.
    w.add_float32("loom.asr.timestamp_step_sec", step_sec)

    # A vocabulary, because `transcribe` detokenizes: one real word at a known id, and filler up to the
    # timestamp block so the ids the driver returns are inside it.
    hello_id = 5
    tokens = ["<pad>", "<eos>", "!", "?", ".", "hello", "!?"]
    tokens += [f"<unused{i}>" for i in range(len(tokens), ts_first + steps_for_11s + 1)]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    # One merge, because `BpeVocab::load` requires the array to exist -- an empty list writes no key
    # at all and the model fails to load. Nothing here encodes text (the driver returns ids directly and
    # `transcribe` only decodes), so any well-formed merge whose result is in the vocabulary will do.
    w.add_token_merges(["! ?"])
    w.add_eos_token_id(1)

    driver = (TIMESTAMP_DRIVER
              .replace("TS_FIRST", str(ts_first))
              .replace("HELLO_ID", str(hello_id))
              .replace("STEPS_PER_CLIP", str(steps_for_11s)))
    w.add_string("model.driver_script", driver)
    _finish(w)


# A DYNAMIC-LENGTH ASR model: no `loom.n_samples`, no timestamp ids, no language or task table -- the
# shape of the NeMo CTC and transducer families, which is every ASR export here except Whisper.
#
# Two behaviours need it and neither is reachable with the fixed-clip fixture above.
#
# 1. THE SEGMENT'S EXTENT. This family emits no timestamp tokens, so `transcribe` returns one segment
#    for the whole file. It used to report `{0.0, 0.0}` -- which reads as a zero-length span at the
#    start of the audio, a measurement, and a wrong one. It reports `{0.0, duration}` now: no boundary
#    the model chose (`timestamped` stays false, `closed` stays false), but a true statement of what
#    the transcript covers.
#
# 2. AN ARGUMENT THAT SELECTS NOTHING. This driver is called with the waveform and its length and
#    nothing else, so `language=` cannot reach it however multilingual the checkpoint. That used to
#    throw. It warns and proceeds now, and the warning is returned rather than printed because a
#    library has no logger.
DYNAMIC_DRIVER = """
function infer(inputs)
    -- Ignores its input, like the fixture above: what is under test is what `transcribe` does around
    -- the call -- the segment it synthesises and the arguments it accepts -- not a decode.
    return {HELLO_ID}
end
"""


def write_dynamic_asr(path: Path) -> None:
    from gguf import GGUFWriter

    w = _base(path, "dynamic_asr_test")
    w.add_string("loom.task", "automatic-speech-recognition")
    w.add_string("loom.input.kind", "audio")
    w.add_string("loom.output.kind", "token_ids")
    w.add_uint32("loom.sample_rate", 16000)
    # Deliberately NO `loom.n_samples`: that absence is what makes the length dynamic, and it is the
    # condition both behaviours above key on.

    hello_id = 5
    tokens = ["<pad>", "<eos>", "!", "?", ".", "hello", "!?"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_merges(["! ?"])
    w.add_eos_token_id(1)
    w.add_string("model.driver_script", DYNAMIC_DRIVER.replace("HELLO_ID", str(hello_id)))
    _finish(w)


# A DYNAMIC-LENGTH ASR model that DOES select a language and a task: Canary's shape. Its prompt carries
# a source and a target language, so unlike the file above a `language=` CAN reach the model -- and the
# declared table is how `transcribe` knows. The driver echoes what it was handed, as tokens, which is
# the only way the test sees that the arguments arrived (and, for `task`, that 0 arrived as 0).
#
# Task ids are TARGET-LANGUAGE ids, as Canary's are: `transcribe` is 0 ("the source"), `translate` is
# the id of `en`.
DYNAMIC_LANGUAGES_DRIVER = """
function infer(inputs)
    local out = {HELLO_ID}
    if inputs.language ~= nil then table.insert(out, inputs.language) end
    if inputs.task ~= nil then
        if inputs.task == 0 then table.insert(out, ZERO_ID) else table.insert(out, inputs.task) end
    end
    if inputs.target_language ~= nil then
        table.insert(out, TARGET_ID)
        table.insert(out, inputs.target_language)
    end
    return out
end
"""


def write_dynamic_asr_languages(path: Path) -> None:
    w = _base(path, "dynamic_asr_languages_test")
    w.add_string("loom.task", "automatic-speech-recognition")
    w.add_string("loom.input.kind", "audio")
    w.add_string("loom.output.kind", "token_ids")
    w.add_uint32("loom.sample_rate", 16000)

    tokens = ["<pad>", "<eos>", "!", "?", ".", "hello", "!?", "en", "de", "t0", ">"]
    ids = {t: i for i, t in enumerate(tokens)}
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_merges(["! ?"])
    w.add_eos_token_id(1)
    w.add_array("loom.asr.language_names", ["de", "en"])
    w.add_array("loom.asr.language_ids", [ids["de"], ids["en"]])
    w.add_array("loom.asr.task_names", ["transcribe", "translate"])
    w.add_array("loom.asr.task_ids", [0, ids["en"]])
    # The languages it can WRITE -- here a subset of what it hears, so the test can tell the two tables
    # apart: `de` is a source language and not a target one.
    w.add_array("loom.asr.target_language_names", ["en"])
    w.add_array("loom.asr.target_language_ids", [ids["en"]])
    w.add_string("model.driver_script", DYNAMIC_LANGUAGES_DRIVER
                 .replace("HELLO_ID", str(ids["hello"])).replace("ZERO_ID", str(ids["t0"]))
                 .replace("TARGET_ID", str(ids[">"])))
    _finish(w)


# A token classifier, for `loom::text::classify`. One class per ROW of the model's output, which is
# what `token_labels_epilogue` returns from `loom.argmax_rows` in a real export -- spelled here as a
# table so the fixture needs no graph.
#
# It labels by POSITION rather than by token value (row i gets class i % 3), which is what makes the
# framing-token strip observable: a driver returning a constant would look identical whether the [CLS]
# and [SEP] rows were dropped or not.
CLASSIFY_DRIVER = """
function infer(inputs)
    local out = {}
    for i = 1, #inputs.tokens do
        out[i] = (i - 1) % 3
    end
    return out
end
"""

# A driver that returns one class for the whole sentence rather than one per token. Not a degenerate
# case of the contract above -- a pooled sequence classifier is a different task -- and the fixture
# exists so the length check is tested against something real rather than by inspection.
POOLED_DRIVER = """
function infer(inputs)
    return {2}
end
"""


def write_classifier(path: Path) -> None:
    w = _base(path, "token_classifier_test")
    w.add_string("loom.task", "token-classification")
    w.add_string("loom.input.kind", "text")
    w.add_string("loom.output.kind", "class")
    # Id-indexed, which is the whole convention: the driver returns 0/1/2 and these are their names.
    # Three names for a head the fixture pretends has four classes, so the "an id past the table keeps
    # its number" branch has something to hit.
    w.add_array("loom.labels", ["O", "B-PER", "I-PER"])
    # The framing ids a WordPiece encode adds, under the generic KVs the vocabulary writer uses -- CLS
    # reuses BOS, SEP has its own. `classify` strips on these DECLARED ids rather than on a spelling,
    # so the fixture needs no vocabulary at all to exercise it.
    w.add_bos_token_id(101)
    w.add_sep_token_id(102)
    w.add_string("model.driver_script", CLASSIFY_DRIVER)
    _finish(w)


def write_spm_classifier(path: Path) -> None:
    """The same door, framed the way a SentencePiece encode frames it: `<s> ... </s>`, no SEP.

    A second fixture rather than a second KV on the first, because what it pins is that the framing is
    PER-TOKENIZER: family 12's WordPiece checkpoints wrap in CLS/SEP and its XLM-R one wraps in BOS/EOS
    (P5). `framing_ids` reading only the first pair left a trailing `</s>` in the answer wearing
    whatever label the head gave it -- one extra entry, at the end, which is exactly where a caller
    comparing lengths against its own word count would find it.
    """
    w = _base(path, "spm_token_classifier_test")
    w.add_string("loom.task", "token-classification")
    w.add_string("loom.input.kind", "text")
    w.add_string("loom.output.kind", "class")
    w.add_array("loom.labels", ["O", "B-PER", "I-PER"])
    # XLM-R's own ids, and no separator token at all -- so this fixture also covers the "a file naming
    # only some of the four strips only those" half of `framing_ids`.
    w.add_bos_token_id(0)
    w.add_eos_token_id(2)
    w.add_pad_token_id(1)
    w.add_string("model.driver_script", CLASSIFY_DRIVER)
    _finish(w)


def write_pooled_classifier(path: Path) -> None:
    w = _base(path, "pooled_classifier_test")
    w.add_string("loom.task", "token-classification")
    w.add_string("loom.input.kind", "text")
    w.add_string("loom.output.kind", "class")
    w.add_string("model.driver_script", POOLED_DRIVER)
    _finish(w)


# Family 13's doors (ADR-062): `loom::audio::classify` and `loom::audio::embed`. The drivers answer from
# the SAMPLE COUNT alone, so the fixtures need no graph: what is under test is how a flat answer is cut
# into rows by the declared contract, and that cut is metadata plus arithmetic.
#
# The frame driver returns one row per four samples, alternating [0.25, 0.75] / [0.75, 0.25] -- so a
# reshape that read the rows the wrong way round, or one row short, is a different table.
FRAME_CLASSIFY_DRIVER = """
function infer(inputs)
    local out = {}
    for r = 0, math.floor(#inputs.waveform / 4) - 1 do
        local speech = (r % 2 == 0) and 0.75 or 0.25
        out[#out + 1] = 1 - speech
        out[#out + 1] = speech
    end
    return out
end
"""

CLIP_CLASSIFY_DRIVER = """
function infer(inputs)
    return {0.1, 0.7, 0.2}
end
"""

# Echoes what the door handed over, so the test can see the waveform AND the `length` beside it.
EMBED_DRIVER = """
function infer(inputs)
    return {#inputs.waveform, inputs.length[1], inputs.waveform[1]}
end
"""

# Three numbers for a two-label head: not a whole number of rows, which the door must refuse.
SKEWED_DRIVER = """
function infer(inputs)
    return {0.5, 0.5, 0.5}
end
"""


def _audio_classifier(path: Path, arch: str, task: str, kind: str, granularity: str, driver: str,
                      labels=None, frame_rate=None, frame_offset=None) -> None:
    w = _base(path, arch)
    w.add_string("loom.task", task)
    w.add_string("loom.input.kind", "audio")
    w.add_string("loom.output.kind", kind)
    w.add_string("loom.output.granularity", granularity)
    # i32, as the exporter's contract writer spells every int -- the reader must accept it.
    w.add_int32("loom.sample_rate", 16000)
    if labels:
        w.add_array("loom.labels", labels)
    if frame_rate is not None:
        w.add_float32("loom.output.frame_rate", frame_rate)
    if frame_offset is not None:
        w.add_float32("loom.output.frame_offset", frame_offset)
    w.add_string("model.driver_script", driver)
    _finish(w)


def write_audio_classifiers(out_dir: Path) -> None:
    _audio_classifier(out_dir / "audio_frame_classifier.gguf", "frame_classifier_test",
                      "audio-classification", "class", "frame", FRAME_CLASSIFY_DRIVER,
                      labels=["non_speech", "speech"], frame_rate=4.0, frame_offset=0.5)
    _audio_classifier(out_dir / "audio_clip_classifier.gguf", "clip_classifier_test",
                      "audio-classification", "class", "clip", CLIP_CLASSIFY_DRIVER,
                      labels=["en", "de", "fr"])
    _audio_classifier(out_dir / "audio_embedder.gguf", "embedder_test",
                      "audio-embedding", "embeddings", "clip", EMBED_DRIVER)
    _audio_classifier(out_dir / "audio_skewed_classifier.gguf", "skewed_classifier_test",
                      "audio-classification", "class", "frame", SKEWED_DRIVER,
                      labels=["non_speech", "speech"], frame_rate=50.0)


def write_codec(path: Path) -> None:
    """A codec decoder: `audio_codes` in, `audio` out (ADR-020).

    No driver and no graph -- what is under test is the modality FOLD, which is metadata all the way
    down. It exists so `token_ids` -> "text" and `audio_codes` -> "codes" are pinned by two fixtures
    that differ in one key, which is the only way the asymmetry survives someone tidying it.
    """
    w = _base(path, "codec_test")
    w.add_string("loom.task", "audio-codec")
    w.add_string("loom.input.kind", "audio_codes")
    w.add_string("loom.output.kind", "audio")
    w.add_uint32("loom.sample_rate", 44100)
    # What a caller needs in order to build the input at all: the matrix is n_codebooks wide, and the
    # ids in each column are bounded by the codebook size.
    w.add_uint32("loom.codec.n_codebooks", 9)
    w.add_uint32("loom.codec.codebook_size", 1024)
    w.add_float32("loom.codec.frame_rate", 86.1328125)
    _finish(w)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_declared(out_dir / "contract_declared.gguf")
    write_legacy(out_dir / "contract_legacy.gguf")
    write_generate(out_dir / "generate_driver.gguf")
    write_timestamped_asr(out_dir / "timestamped_asr.gguf")
    write_dynamic_asr(out_dir / "dynamic_asr.gguf")
    write_dynamic_asr_languages(out_dir / "dynamic_asr_languages.gguf")
    write_codec(out_dir / "contract_codec.gguf")
    write_classifier(out_dir / "classify_driver.gguf")
    write_spm_classifier(out_dir / "classify_driver_spm.gguf")
    write_pooled_classifier(out_dir / "pooled_classifier.gguf")
    write_audio_classifiers(out_dir)


if __name__ == "__main__":
    main()
