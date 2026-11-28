#!/usr/bin/env python3
"""GGUF fixtures for test_rule_based_phonemizer.cpp: a toy phonemizer script and its data, no weights.

  rule_phonemizer_test.gguf      two languages, `x` with data and `y` without. The script maps through its
                                 language's data (a JSON-free "from=to;" table, so the test needs no JSON
                                 reader), reports what it cannot map in `dropped`, and has `set_lexicon`.
                                 It also records whether `io`, `os`, `dofile` and `loadfile` are reachable.
  rule_phonemizer_bad_kind.gguf  `loom.phonemizer.kind = "python"`, which load must refuse.
  rule_phonemizer_no_entry.gguf  a script with no `phonemize`, which load must refuse.

Requires: pip install gguf
"""
import sys

import numpy as np
from gguf import GGUFWriter

SCRIPT = r'''
local tables, lexicons = {}, {}
local reachable = (io ~= nil) or (os ~= nil) or (dofile ~= nil) or (loadfile ~= nil)

local function table_of(lang)
  local t = tables[lang]
  if t == nil then
    t = {}
    local data = phonemizer_data(lang)
    if data then
      for from, to in data:gmatch("(%d+)=(%d+);") do t[tonumber(from)] = tonumber(to) end
    end
    tables[lang] = t
  end
  return t
end

function phonemize(cps, lang)
  if reachable then return {33} end   -- "!" : the sandbox leaked
  local t, ipa, dropped = table_of(lang), {}, {}
  local word = lexicons[lang] and lexicons[lang][table.concat(cps, ",")]
  if word then return word, {} end
  for _, cp in ipairs(cps) do
    if t[cp] then ipa[#ipa + 1] = t[cp] else dropped[#dropped + 1] = cp end
  end
  return ipa, dropped
end

function set_lexicon(lang, entries)
  lexicons[lang] = lexicons[lang] or {}
  for _, e in ipairs(entries) do lexicons[lang][table.concat(e[1], ",")] = e[2] end
end
'''

# x: a -> ɑ (U+0251), b -> β (U+03B2). Digits and anything else are dropped.
DATA_X = "97=593;98=946;"


def write(path, kind, script, data=True):
    w = GGUFWriter(path, "phonemizer_test")
    w.add_string("loom.phonemizer.kind", kind)
    w.add_string("loom.phonemizer.script", script)
    w.add_string("loom.phonemizer.ruleset", "toy 1; x=sha256:00")
    w.add_array("loom.text.languages", ["x", "y"])
    if data:
        w.add_string("loom.phonemizer.data.x", DATA_X)
    w.add_tensor("marker", np.zeros(1, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


test, bad_kind, no_entry = sys.argv[1:4]
write(test, "lua", SCRIPT)
write(bad_kind, "python", SCRIPT)
write(no_entry, "lua", "function other() end")
