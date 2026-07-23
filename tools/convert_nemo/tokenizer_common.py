"""Extracts SentencePiece vocab data from a `.model` protobuf and writes it into a GGUF file using
llama.cpp's own "tokenizer.ggml.*" KV schema (confirmed directly against gguf-py's `GGUFWriter`,
gguf-py's own `SentencePieceVocab` class in `gguf/vocab.py`, and llama.cpp's `include/llama.h`):
`tokenizer.ggml.model` is `"t5"` for a genuine SentencePiece *unigram* model, or `"llama"` for real
SentencePiece **BPE** (llama.cpp's own tag -- the original LLaMA/Mistral tokenizers are themselves
SentencePiece BPE models, confirmed via `gguf-py`'s `SentencePieceVocab` writing this exact tag) --
selected automatically here from the real `.model` protobuf's own `trainer_spec.model_type`
(`UNIGRAM=1`, `BPE=2`; `WORD`/`CHAR` are not implemented on the C++ side, see `loom::Vocab`), not
assumed or hardcoded. `.tokens`/`.scores`/`.token_type` arrays, `unknown_token_id`, `add_space_prefix`,
`remove_extra_whitespaces`, and the raw `precompiled_charsmap` blob (needed for Unicode normalization
during encode -- see `Vocab` in the C++ engine, which mirrors llama.cpp's XCDA-based normalizer
exactly) are identical between both types, confirmed by inspecting a real BPE model's protobuf directly.

SentencePiece's own per-piece `Type` enum (`NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5,
BYTE=6`, confirmed via direct protobuf inspection) is numerically identical to llama.cpp's
`llama_token_type` (confirmed from `include/llama.h`), so piece types are copied straight through with
no remapping.

Uses the `sentencepiece` package's bundled protobuf definitions directly (`sentencepiece_model_pb2`),
not the `SentencePieceProcessor` wrapper -- the wrapper doesn't expose `precompiled_charsmap` or the
normalizer flags needed here.

Optional `bos_token_id`/`eos_token_id`/`add_bos_token`/`add_eos_token` kwargs close the gap that
otherwise blocks ALBERT/XLNet-style Unigram models (which wrap sequences via SentencePiece's own BOS/EOS
convention rather than a separate CLS/SEP concept, unlike T5's own tokenizer) -- default to today's no-op
(nothing written), so every existing NeMo T5/BPE call site is byte-for-byte unaffected.

Requires: pip install sentencepiece gguf
"""
from gguf import GGUFWriter
from sentencepiece import sentencepiece_model_pb2 as spm_pb2


def write_sentencepiece_vocab(writer: GGUFWriter, tokenizer_model_bytes: bytes, *,
                               bos_token_id: int | None = None, eos_token_id: int | None = None,
                               add_bos_token: bool = False, add_eos_token: bool = False) -> None:
    m = spm_pb2.ModelProto()
    m.ParseFromString(tokenizer_model_bytes)

    pieces = [p.piece for p in m.pieces]
    scores = [p.score for p in m.pieces]
    types = [int(p.type) for p in m.pieces]
    unk_id = next((i for i, p in enumerate(m.pieces) if p.type == p.UNKNOWN), 0)

    model_type = m.trainer_spec.model_type
    if model_type == m.trainer_spec.UNIGRAM:
        tokenizer_model = "t5"
    elif model_type == m.trainer_spec.BPE:
        tokenizer_model = "llama"
    else:
        raise NotImplementedError(f"SentencePiece model_type {model_type} (WORD/CHAR) is not implemented "
                                   "on the C++ side (loom::Vocab only supports UNIGRAM and BPE)")

    writer.add_tokenizer_model(tokenizer_model)
    writer.add_token_list(pieces)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_unk_token_id(unk_id)
    writer.add_add_space_prefix(bool(m.normalizer_spec.add_dummy_prefix))
    writer.add_remove_extra_whitespaces(bool(m.normalizer_spec.remove_extra_whitespaces))
    if m.normalizer_spec.precompiled_charsmap:
        writer.add_precompiled_charsmap(m.normalizer_spec.precompiled_charsmap)

    if bos_token_id is not None:
        writer.add_bos_token_id(bos_token_id)
    if eos_token_id is not None:
        writer.add_eos_token_id(eos_token_id)
    # Only written when true -- matches the C++ side's own "absent KV defaults to false" convention, so
    # every existing call site that never passes these kwargs writes byte-for-byte the same GGUF as before.
    if add_bos_token:
        writer.add_add_bos_token(True)
    if add_eos_token:
        writer.add_add_eos_token(True)
