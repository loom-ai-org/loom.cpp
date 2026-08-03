"""The exporter's axis vocabulary (EXPORT-ROADMAP.md R1).

Before this module, every dynamic dimension in every exported model rendered as the same bare symbol,
`shape_expr.N_TOKENS` -- correct for an LLM's subword-token axis, but also standing in for at least four
unrelated quantities (see EXPORT-ROADMAP.md's R1 table): Conformer-CTC/Parakeet's raw audio sample count,
Kokoro's post-encoder acoustic frame count, VITS/Matcha's phoneme count (a genuine token axis, just not
subwords), and StyleTTS2's style-vector length. Two concrete costs came from that: `symbol_overrides`
existed only to declare a leaf input's axis by hand (there being no vocabulary to declare it in), and the
derivation walk's own "I don't know" fallback was spelled identically to a real derivation that happened
to equal the same bare symbol (`value_facts.scalar_expr_is_guess`'s own docstring).

Every name here is a `shape_expr.symbol()` -- a positive-integer sympy symbol, not a bare string -- so an
axis declared here composes with the rest of the derivation algebra (`floor_div`, `as_expr`, `render`)
exactly like `N_TOKENS` always has. Declaring a model input's axis (`LoomGGUFExporter`'s `root_axis`/
`declared_axes` kwargs) means naming ONE of these, not inventing a new ad hoc symbol per model.

This is deliberately just the vocabulary, not a schema (that's R3/`LoomExportConfig`, out of scope for
this item) -- a small, fixed enum of names with a docstring per name saying what it counts and at what
rate, extended only when a genuinely new kind of quantity shows up.
"""
from .shape_expr import N_TOKENS, symbol

N_SAMPLES = symbol("n_samples")
"""Raw waveform samples at the model's native sample rate -- Conformer-CTC/Parakeet TDT/RNNT's own
"waveform" input (see nemo_asr_export.py's `trace_and_convert`). Every STFT-frame and encoder-frame
count downstream of it is a DERIVED expression over this axis (the conv/STFT formulas in
`exporter.py`'s `_infer_dynamic_dim_expr_uncached` already compute those correctly); it is the raw
sample count itself that used to render as the bare, misleading "n_tokens"."""

N_ENC_FRAMES = symbol("n_enc_frames")
"""Encoder output frames after subsampling -- the acoustic/ASR frame count a downstream decoder
consumes. Kokoro's `decoder_vocoder` phase declares its "asr" input's own axis this way: unlike
N_SAMPLES's downstream frame counts, the four sibling leaf inputs in that phase (f0_curve/n_curve/
noise_in/wsum) have NO data-flow path back to "asr" at all -- they are independently-traced leaves,
not derived from it by any op this exporter's derivation walk could follow -- so their own axes are
declared as explicit multiples of this one (see `export_kokoro_mil.py`'s `declared_axes` table,
replacing the old `symbol_overrides` dict of raw MIL symbol names)."""

N_LATENT = symbol("n_latent")
"""A style/latent vector's own length -- StyleTTS2's diffusion sampler, which derives nothing from any
other axis (EXPORT-ROADMAP.md's R1 table: "what other axes get derived from it: --")."""

N_KV = symbol("n_kv")
"""The KV-cache extent a cached attention step attends over -- `n_past + n_tokens` (KV-CACHE.md stage 3).

**The one axis here that never comes from a `ct.RangeDim`, and that is a property of the trace rather
than an omission.** With no cache in the traced graph, HF computes scores of shape `[1, h, s, s]`, and
declaring a second independent range dim for the mask's last axis fails coremltools' own type inference
at conversion time -- two independent `RangeDim`s over one attention block cannot be traced at all
(KV-CACHE.md §2). So this axis arrives on the ONE input that needs it, the fused causal mask, by being
declared at export time after `fuse_loom_attention` has made it sound: post-fusion the mask var's only
consumers are `ATTENTION` nodes, so no other node's shape derives from it (see `_retype_fused_mask_input`,
which checks that property rather than assuming it).

The engine already knows the name: `GraphBuilder` derives `n_kv = n_tokens + n_past` when a driver's axis
table does not bind it (`graph_builder.cpp:129`), which is exactly how the bespoke converters' hand-written
`{"name": "kq_mask", "shape": ["n_kv", "n_tokens"]}` has always resolved (`convert_qwen3.py:65`).

Consequences worth stating because they are easy to misread as bugs:

* `_validate_input_axes` never sees this axis. It reads the *traced* program's input symbols, and the
  retyping happens after the trace, on the emitted topology -- so a fused causal LM still has exactly one
  traced dynamic symbol and passes that check trivially. The check is not weakened; the case simply does
  not reach it.
* It is equally not declarable through `declared_axes`, because the mask shares its traced symbol with
  `tokens`/`cache_position` by design, and an override keyed on that symbol would rewrite all three.
  `_resolve_declared_axes` raises on exactly that rather than silently doing it."""

N_CODES = symbol("n_codes")
"""Neural audio codec token count -- not used by any model exported as of this roadmap phase (family
11 in EXPORT-ROADMAP.md's R5 table is still unimplemented). Declared here so a future codec-decoder
export has a real name to reach for instead of reusing "n_tokens" out of convenience."""

BATCH = symbol("batch")
"""The batch axis -- always a literal 1 for every model this exporter targets (see
`value_facts.gather_shape_value`'s own torch_axis==0 shortcut, and `_infer_dynamic_dim_expr_uncached`'s
matching fallback). Declared here for completeness/documentation only: no exported model's batch axis
is ever actually dynamic, so nothing renders this symbol today.
"""

__all__ = ["N_TOKENS", "N_SAMPLES", "N_ENC_FRAMES", "N_LATENT", "N_KV", "N_CODES", "BATCH"]
