"""Naming conventions shared by the exporter and its op-handler table.

Kept in its own module purely so `topology_ops.py` and `exporter.py` can both use it without an import
cycle (`exporter` imports `topology_ops`, never the other way around).
"""
import re

# CoreML's own naming convention for a symbolic (dynamic) shape dimension -- always "is" followed by
# digits (e.g. "is0", "is936"). Matched with \b so it only substitutes whole symbol tokens, never a
# coincidental "is" substring inside a longer identifier.
DYNAMIC_SYMBOL_RE = re.compile(r"\bis\d+\b")
