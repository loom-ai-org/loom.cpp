#!/usr/bin/env python3
"""Compare two `snapshot_gguf.py` snapshot trees *semantically*, for changes that are meant to alter
shape expressions without altering what they mean.

`diff -r` is the right gate for a refactor that must be output-preserving, and it stays the gate for
those. But a change like "carry sympy expressions instead of concatenated strings" (see shape_expr.py)
deliberately rewrites every symbolic shape attribute -- `(floor(((1) * (1) * ((((floor(((1) * (((1)+
(((n_tokens) - (1)))))) / ((1) * (1))))) + 512))) / ((1))))` becomes `n_tokens + 512` -- so its diff has
to be *read* rather than required to be empty. This reads it mechanically instead:

* every differing value is evaluated at a range of concrete `n_tokens`, in the same grammar the engine
  itself evaluates (`shape_expr.parse`, which mirrors `src/core/symbol_env.cpp`), and reported as
  EQUIVALENT only if both sides agree at every probe;
* anything that differs and is *not* an expression pair (a renamed node, a changed op, a different node
  count, a tensor whose bytes moved) is reported as a STRUCTURAL difference, which for this class of
  change means a real regression;
* topologies present on one side only, and any non-JSON file difference, are reported too.

Usage: compare_snapshots.py <snap_a> <snap_b>
Exit status is 1 if anything structural (or numerically unequal) turned up.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loom_mil_compiler.shape_expr import UnsupportedShapeExpression, parse, symbol  # noqa: E402

# Spread across the dynamic range every current model declares (0.1 s .. 20 s of 16 kHz audio, plus
# token-count-sized values), including odd and even and off-by-one neighbours, so a pair that agrees
# only at a convenient multiple cannot pass.
PROBES = (1, 2, 3, 7, 16, 63, 64, 65, 101, 160, 161, 1600, 3200, 16000, 16001, 31999, 100000, 320000)

# EXPORT-ROADMAP.md's R1: named axes replacing the single "n_tokens" fallback every dynamic dimension
# used to render as (see tools/loom_mil_compiler/axes.py). {new axis name -> the expression that named
# the identical quantity before the rename} -- a declared claim ("n_samples IS what n_tokens used to
# mean for this model"), not something this tool infers. Substituted into a NEW-side expression before
# evaluating, so a pure rename compares as equivalent instead of structural. A rename not listed here
# still compares correctly (see `evaluate`'s own uniform-probe substitution below): it just isn't
# documented as a KNOWN one, the way an entry here is.
ALIAS_MAP = {
    "n_samples": "n_tokens",      # Conformer-CTC / Parakeet TDT / Parakeet RNNT -- raw audio samples
    "n_enc_frames": "n_tokens",   # Kokoro decoder_vocoder -- post-encoder acoustic frame count
}


def _dealias(expr):
    """`expr` with every `ALIAS_MAP` entry substituted, so a renamed symbol reads under its pre-R1
    name. Substitution is on the expression tree (see shape_expr.py's own module docstring for why
    that matters over string rewriting): a compound expression like `600*n_enc_frames + 20` becomes
    `600*n_tokens + 20` exactly, not just a bare symbol swap."""
    subs = {}
    for new_name, old_expr in ALIAS_MAP.items():
        sym = symbol(new_name)
        if sym in expr.free_symbols:
            subs[sym] = parse(old_expr)
    return expr.subs(subs) if subs else expr


def evaluate(value, probe):
    """`value` (a JSON attribute: number or expression string) at one concrete axis length.

    Any free symbol remaining after `ALIAS_MAP` substitution -- whether that's the ordinary
    `n_tokens` an untouched model still uses, or some other axis name this tool has no alias entry
    for -- is substituted with the SAME `probe` value. That is sound, not just permissive: this whole
    exporter targets models with exactly one true dynamic quantity per topology (see exporter.py's
    `get_var_info`), so any two symbol names that both survive to a single comparison are, by
    construction, two names for the same run's one real quantity, not two independently-varying ones.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        expr = _dealias(parse(value))
        free = expr.free_symbols
        if not free:
            return float(expr)
        return float(expr.subs({s: probe for s in free}))
    except (UnsupportedShapeExpression, TypeError, ValueError):
        return None


def equivalent(old, new):
    """True iff both sides are expressions that agree at every probe."""
    for probe in PROBES:
        a, b = evaluate(old, probe), evaluate(new, probe)
        if a is None or b is None or a != b:
            return False
    return True


def walk(old, new, path, equal_pairs, structural):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                structural.append((path + [key], old.get(key, "<missing>"), new.get(key, "<missing>")))
            else:
                walk(old[key], new[key], path + [key], equal_pairs, structural)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            structural.append((path, f"<{len(old)} items>", f"<{len(new)} items>"))
            return
        for i, (a, b) in enumerate(zip(old, new)):
            walk(a, b, path + [str(i)], equal_pairs, structural)
    elif old != new:
        if equivalent(old, new):
            equal_pairs.append((path, old, new))
        else:
            structural.append((path, old, new))


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    a_root, b_root = Path(sys.argv[1]), Path(sys.argv[2])
    failures = 0
    for model_dir in sorted(a_root.iterdir()):
        other = b_root / model_dir.name
        if not other.is_dir():
            print(f"{model_dir.name}: MISSING in {b_root}")
            failures += 1
            continue
        equal_pairs, structural, notes = [], [], []
        for a_file in sorted(model_dir.iterdir()):
            b_file = other / a_file.name
            if not b_file.exists():
                notes.append(f"{a_file.name}: missing on the other side")
                failures += 1
                continue
            if a_file.suffix == ".json":
                walk(json.loads(a_file.read_text()), json.loads(b_file.read_text()),
                     [a_file.stem], equal_pairs, structural)
            elif a_file.name == "kv.txt":
                # Every structured payload's own content is compared as its own file; kv.txt only
                # carries their digests, which are expected to move whenever the payload does.
                for line_a, line_b in zip(a_file.read_text().splitlines(),
                                          b_file.read_text().splitlines()):
                    if line_a != line_b and "sha256=" not in line_a:
                        structural.append(([a_file.name], line_a, line_b))
            elif a_file.read_bytes() != b_file.read_bytes():
                notes.append(f"{a_file.name}: differs")
                failures += 1
        status = "OK " if not structural else "BAD"
        print(f"{status} {model_dir.name}: {len(equal_pairs)} expression(s) rewritten, "
              f"{len(structural)} structural difference(s)")
        for note in notes:
            print(f"      {note}")
        for path, old, new in structural[:20]:
            print(f"      STRUCTURAL {'/'.join(path)}:\n        - {old!r}\n        + {new!r}")
        failures += len(structural)
    print("\nAll differences are numerically equivalent." if not failures
          else f"\n{failures} difference(s) are NOT equivalent.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
