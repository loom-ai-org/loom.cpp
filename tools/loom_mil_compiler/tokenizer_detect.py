"""Auto-detects a HF tokenizer's BPE-family pretokenizer regex "shape" and the tokenizer's overall vocab
family, so `export_hf_causal_lm.py` no longer needs an explicit `--tokenizer-pre`/`--tokenizer-family` flag
for the common case.

Two independent responsibilities:

1. `detect_vocab_family()` -- reads `tokenizer.json`'s own `model.type` field directly ("BPE"/"WordPiece"/
   "Unigram" map to "bpe"/"wordpiece"/"sentencepiece_proto"), with no `tokenizers`/`AutoTokenizer`
   dependency for this specific check (matches `bpe_tokenizer_export.py`'s own no-extra-dependency
   convention). This is simpler and more reliable here than porting llama.cpp's own per-architecture
   dispatch (one Python class per model architecture, each hardcoding which vocab-writing path to call) --
   this project's exporter is architecture-agnostic (any `AutoModelForCausalLM`), so there's no per-arch
   class to hang that dispatch off of; sniffing the tokenizer's own declared type directly is the natural
   generic-exporter equivalent. "Unigram" only resolves to "sentencepiece_proto" if a sibling
   tokenizer.model/spiece.model SentencePiece protobuf exists -- loom's Vocab (C++) needs the real
   protobuf for precompiled_charsmap, a tokenizer.json-only Unigram model isn't supported yet. ByT5-family
   tokenizers ("byte") have no tokenizer.json/tokenizer.model at all (no Rust "fast" backend exists for
   them) -- detected instead via tokenizer_config.json's own `tokenizer_class=="ByT5Tokenizer"` field.

2. `detect_loom_pre_type()` -- for the "bpe" family, ports llama.cpp's own auto-detection recipe from
   `conversion/base.py`'s `get_vocab_base_pre` verbatim: encode a fixed test string (`_CHKTXT`, byte-for-
   byte identical to llama.cpp's own `chktxt`) with the real HF tokenizer, sha256-hash the resulting
   token-id list's string repr, and look the hash up in `_CHKHSH_TO_LLAMA_PRE` (transcribed directly from
   `conversion/base.py`'s own `if chkhsh == ...: res = ...` cascade -- see that file, or llama.cpp's own
   `convert_hf_to_gguf_update.py`, if this table ever needs re-deriving/extending) to get llama.cpp's
   canonical `tokenizer.ggml.pre` name. That name is then mapped, via `_LLAMA_PRE_TO_LOOM_PRE_TYPE`, onto
   one of the pretokenizer "shape" keys `include/loom/core/bpe_vocab.cpp`'s own `pre_spec_table()` actually
   implements -- kept in sync with that table by hand (both are transcribed from the same llama.cpp
   source, see bpe_vocab.cpp's own comment for the full implemented-vs-not-yet list). A name llama.cpp
   recognizes but loom doesn't implement resolves to `None` and raises `NotImplementedError` (actionable:
   pass `--tokenizer-pre` explicitly with a supported shape key, or extend `pre_spec_table()`) rather than
   silently mis-tokenizing; a hash not present in `_CHKHSH_TO_LLAMA_PRE` at all (a tokenizer newer than
   this transcription) falls back to a warning + the "qwen2" default, matching this project's pre-existing
   default before this auto-detection existed.
"""
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Optional

# Verbatim copy of llama.cpp's own chktxt (conversion/base.py) -- deliberately not reworded/reformatted,
# every character (including the exact emoji/CJK/Khmer/Cyrillic/punctuation runs) is part of what makes
# different tokenizers' encodings of it hash differently.
_CHKTXT = (
    '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
)

# chkhsh -> llama.cpp's own canonical `tokenizer.ggml.pre` name. Transcribed directly from
# conversion/base.py's `get_vocab_base_pre` if-cascade (92 entries covering 81 unique names, current as of
# this transcription -- see that file's own `convert_hf_to_gguf_update.py` regenerator if this needs
# updating for a newer llama.cpp).
_CHKHSH_TO_LLAMA_PRE: dict[str, str] = {
    "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b": "chatglm-bpe",
    "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516": "chatglm-bpe",
    "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2": "glm4",
    "9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902": "glm4",
    "cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267": "glm4",
    "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35": "minerva-7b",
    "7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664": "hunyuan",
    "bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6": "hunyuan-dense",
    "a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6": "falcon-h1",
    "60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86": "falcon-h1",
    "3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896": "falcon-h1",
    "48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b": "falcon-h1",
    "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890": "kimi-k2",
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c": "qwen2",
    "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f": "qwen35",
    "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273": "grok-2",
    "b3d1dd861f1d4c5c0d2569ce36baf3f90fe8a102db3de50dd71ff860d91be3df": "jina-v2-de",
    "0fe1cf6eda062318a1af7270f3331a85c539a01778ff948e24388e949c5282f4": "gpt-2",
    "9e454714343b69b99b71795c1d27a68c2a1d15dab111f4d353109f966af29da7": "lfm2",
    "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5": "llama-bpe",
    "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754": "deepseek-llm",
    "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821": "deepseek-coder",
    "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed": "falcon",
    "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f": "bert-bge",
    "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e": "falcon3",
    "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7": "bert-bge-large",
    "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166": "mpt",
    "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34": "starcoder",
    "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454": "gpt-2",
    "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3": "stablelm2",
    "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff": "refact",
    "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8": "command-r",
    "d772b220ace2baec124bed8cfafce0ead7d6c38a4b65ef11261cf9d5d62246d1": "tiny_aya",
    "52df12b4c8d4176e7481aab4b6e8454d1fd0a210a04a574f6d4e067d10e23c3e": "cohere2moe",
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea": "qwen2",
    "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166": "olmo",
    "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e": "dbrx",
    "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448": "jina-v1-en",
    "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f": "jina-v2-en",
    "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643": "jina-v2-es",
    "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6": "jina-v2-de",
    "a023e9fdc5a11f034d3ef515b92350e56fb2af1f66c6b6811a4444ea9bf8763d": "jina-v5-nano",
    "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d": "smaug-bpe",
    "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360": "poro-chat",
    "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a": "jina-v2-code",
    "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee": "viking",
    "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901": "jais",
    "bc5108ee1eb6a3d600cadd065f63190fbd0554dbc9e4bbd6a0d977970afc8d2a": "jais-2",
    "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f": "codeshell",
    "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e": "tekken",
    "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249": "smollm",
    "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7": "bloom",
    "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21": "gpt3-finnish",
    "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae": "exaone",
    "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085": "phi-2",
    "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450": "chameleon",
    "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65": "roberta-bpe",
    "ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb": "gigachat",
    "d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1": "megrez",
    "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5": "deepseek-v3",
    "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5": "deepseek-r1-qwen",
    "ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e": "gpt-4o",
    "7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f": "superbpe",
    "1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15": "trillion",
    "96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224": "bailingmoe",
    "d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406": "llama4",
    "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3": "pixtral",
    "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec": "seed-coder",
    "b0a6b1c0bd5998ebd9df08611efde34a4ff03faed45ae09c43e6b31ebd4b94cf": "a.x-4.0",
    "f6791d196f87ce6b56a7d234be618e0d58f8cda3549416635b2bebcd22cd95c4": "midm-2.0",
    "169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51": "lfm2",
    "2085e1638f6c377a0aa4ead21b27bb4cb941bf800df86ed391011769c1758dfb": "exaone4",
    "a1e163ecab2e718a4c829d1148b6e86824ec36163bb71941c3dca9cd5ac25756": "mellum",
    "a0b64b4385f123663873756336c085744376d015ff328bb1d901598f63c44152": "modern-bert",
    "49fc0303c9e0d2c2c565c510f64b2d9b271276acdcdadff733249eda9f7d59df": "afmoe",
    "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206": "bailingmoe2",
    "53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e": "granite-docling",
    "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95": "minimax-m2",
    "4a2e2abae11ca2b86d570fc5b44be4d5eb5e72cc8f22dd136a94b37da83ab665": "kormo",
    "9d70134b369a70e5735009b6de918f7581b5211f7c074d1f89f753aea8248af1": "youtu",
    "16389f0a1f51ee53e562ffd51c371dc508639ab0e4261502071836e50e223e91": "solar-open",
    "6c81ce329e0802883b22eabab0d3fa48357337ef1ecb45443828bf1f6254833f": "exaone-moe",
    "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4": "qwen35",
    "b4b8ca1f9769494fbd956ebc4c249de6131fb277a4a3345a7a92c7dd7a55808d": "joyai-llm",
    "e4d54df1ebc1f2b91acd986c5b51aa50837d5faf7c7398e73c1f9e9ee5d19869": "kanana2",
    "862f827721df956049dff5ca81a57f29e575280bc622e290d3bf4e35eca29015": "f2llmv2",
    "62f6fb0a6fd5098caeabb19b07a5c1099cafc8b9c40eab6ea89ece4ec02fbc57": "sarvam-moe",
    "f728162c1315c26e40249849799b4ba3fe584c32084b4795b03eb295e63cb5af": "talkie",
    "36f3066e97b7f3994b379aaacde306c1444c6ae84e81a5ae3cd2b7ed3b8c42d4": "minicpm5",
    "f241072145675bf8322086f115aebad05e9f869557a238bf2150a2a417d1bf60": "granite-embed-multi-97m",
    "789696f5946cc0fc59371f39f6097cafed196b3acded6140432f26bbb1ae1669": "granite-embed-multi-311m",
    "9dcf830ee9990cdbf78cc523a5f7bd9ad8f3f9890c2d3581d2785ad10f07049d": "mellum2",
}

# llama.cpp `tokenizer.ggml.pre` name -> loom::BpeVocab shape key (bpe_vocab.cpp's `pre_spec_table()`),
# for the ~40 names this project has actually implemented so far (see that table's own comment for the
# grouping/rationale). Every other name llama.cpp recognizes maps to `None` here deliberately -- a real,
# not-yet-implemented pretokenizer family, not an omission to silently paper over.
_LLAMA_PRE_TO_LOOM_PRE_TYPE: dict[str, Optional[str]] = {
    # kQwenLlama3, single-digit
    "qwen2": "qwen2", "deepseek-r1-qwen": "qwen2", "kormo": "qwen2", "f2llmv2": "qwen2",
    "megrez": "qwen2", "stablelm2": "qwen2", "hunyuan": "qwen2", "solar-open": "qwen2", "grok-2": "qwen2",
    # kQwenLlama3 + marks (qwen35 only)
    "qwen35": "qwen35",
    # kQwenLlama3, grouped-digit \p{N}{1,3}
    "llama3": "llama3", "llama-v3": "llama3", "llama-bpe": "llama3", "falcon3": "llama3",
    "falcon-h1": "llama3", "pixtral": "llama3", "midm-2.0": "llama3", "lfm2": "llama3",
    "jina-v5-nano": "llama3", "dbrx": "llama3", "smaug-bpe": "llama3", "glm4": "llama3",
    "chatglm-bpe": "llama3",
    # kGpt2Classic, unbounded
    "gpt-2": "gpt-2", "phi-2": "gpt-2", "jina-v2-es": "gpt-2", "jina-v2-de": "gpt-2",
    "gigachat": "gpt-2", "a.x-4.0": "gpt-2", "mellum": "gpt-2", "modern-bert": "gpt-2",
    "exaone4": "gpt-2", "mpt": "gpt-2", "olmo": "gpt-2", "jais": "gpt-2", "trillion": "gpt-2",
    "granite-docling": "gpt-2", "roberta-bpe": "gpt-2", "jina-v1-en": "gpt-2", "jina-v2-code": "gpt-2",
    # kGpt2Classic, single-digit
    "starcoder": "starcoder", "refact": "starcoder", "command-r": "starcoder", "smollm": "starcoder",
    "codeshell": "starcoder", "exaone": "starcoder", "minerva-7b": "starcoder", "mellum2": "starcoder",
    # kWhitespacePunctExclude
    "poro-chat": "poro-chat", "bloom": "poro-chat", "gpt3-finnish": "poro-chat", "viking": "viking",
    # explicitly NOT YET implemented on the C++ side (see bpe_vocab.cpp's pre_spec_table() for why each is
    # harder than a new regex -- CJK-script splitters, case-transition/camelCase shapes, cascading
    # whitespace, or "byte_encode=false" SPM-style-BPE, none of which fit a bounded regex-shape addition):
    "jais-2": None, "deepseek-llm": None, "deepseek-coder": None, "deepseek-v3": None, "youtu": None,
    "falcon": None, "hunyuan-dense": None, "joyai-llm": None, "tekken": None, "gpt-4o": None,
    "llama4": None, "kanana2": None, "talkie": None, "minimax-m2": None,
    "granite-embed-multi-97m": None, "granite-embed-multi-311m": None, "tiny_aya": None,
    "cohere2moe": None, "superbpe": None, "bailingmoe": None, "bailingmoe2": None, "seed-coder": None,
    "kimi-k2": None, "afmoe": None, "exaone-moe": None, "sarvam-moe": None, "minicpm5": None,
    "chameleon": None, "jina-v2-en": None,
    # These two hash-collide with jina-v2-en/bert-bge in llama.cpp's own table (both real BERT/WordPiece
    # models, whose tokenizer.ggml.pre is written but functionally unused by WPM tokenization) -- never
    # consulted by this module's own detect_loom_pre_type() anyway, since that's only called for the "bpe"
    # vocab family (see detect_vocab_family()).
    "bert-bge": None, "bert-bge-large": None,
}


def compute_chkhsh(tokenizer) -> str:
    """`tokenizer` is any object with an `.encode(str) -> list[int]` method (a real HF
    `PreTrainedTokenizerFast`, or a stub for testing)."""
    chktok = tokenizer.encode(_CHKTXT)
    return sha256(str(chktok).encode()).hexdigest()


def detect_llama_pre_name(tokenizer) -> Optional[str]:
    return _CHKHSH_TO_LLAMA_PRE.get(compute_chkhsh(tokenizer))


def detect_loom_pre_type(tokenizer, *, fallback: str = "qwen2") -> str:
    llama_pre = detect_llama_pre_name(tokenizer)
    if llama_pre is None:
        print(f"warning: tokenizer hash not found in llama.cpp's own chkhsh table (a tokenizer newer than "
              f"this transcription?) -- falling back to '{fallback}'; pass --tokenizer-pre explicitly if "
              f"this is wrong", file=sys.stderr)
        return fallback
    loom_pre = _LLAMA_PRE_TO_LOOM_PRE_TYPE.get(llama_pre)
    if loom_pre is None:
        raise NotImplementedError(
            f"tokenizer auto-detected as llama.cpp pretokenizer family '{llama_pre}', which "
            "loom::BpeVocab does not implement yet -- pass --tokenizer-pre explicitly with a supported "
            "shape key, or extend bpe_vocab.cpp's pre_spec_table() for this family "
            "(see EXPORT-BACKLOG.md item 4)")
    return loom_pre


def detect_vocab_family(tokenizer_dir: str) -> str:
    """Returns "bpe" / "wordpiece" / "sentencepiece_proto" / "byte". See module docstring."""
    tok_dir = Path(tokenizer_dir)
    tokenizer_json_path = tok_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        model_type = json.loads(tokenizer_json_path.read_text())["model"]["type"]
        if model_type == "BPE":
            return "bpe"
        if model_type == "WordPiece":
            return "wordpiece"
        if model_type == "Unigram":
            if not any((tok_dir / n).exists() for n in ("tokenizer.model", "spiece.model")):
                raise NotImplementedError(
                    "tokenizer.json model.type=='Unigram' but no sibling tokenizer.model/spiece.model "
                    "SentencePiece protobuf found -- loom's Unigram support (loom::Vocab) needs the real "
                    "protobuf for precompiled_charsmap; a tokenizer.json-only Unigram model is not "
                    "supported yet (see EXPORT-BACKLOG.md item 4)")
            return "sentencepiece_proto"
        raise NotImplementedError(f"tokenizer.json model.type={model_type!r} has no loom vocab-writer yet")
    if any((tok_dir / n).exists() for n in ("tokenizer.model", "spiece.model")):
        return "sentencepiece_proto"
    # ByT5-family tokenizers (e.g. google/byt5-small) have NO tokenizer.json/tokenizer.model at all --
    # ByT5Tokenizer has no Rust "fast" backend, so no fast-tokenizer file is ever distributed for it. The
    # only on-disk marker is tokenizer_config.json's own tokenizer_class field.
    config_path = tok_dir / "tokenizer_config.json"
    if config_path.exists():
        tokenizer_class = json.loads(config_path.read_text()).get("tokenizer_class")
        if tokenizer_class == "ByT5Tokenizer":
            return "byte"
    raise NotImplementedError(f"no recognized tokenizer file (tokenizer.json/tokenizer.model/spiece.model/"
                               f"a ByT5Tokenizer tokenizer_config.json) found in {tokenizer_dir}")
