#pragma once

// The pre-MIL, per-model C++ drivers -- deliberately NOT part of `loom/loom.h`.
//
// Each of these predates the Lua driver becoming the orchestration device. A driver hard-codes one
// model family's orchestration in C++: its phase ordering, its sampler, its per-token loop. That is
// exactly the work EXPORT-PREPARATION.md 1.3 records as belonging in the exporter instead, because
// adding a family costs a Python config there and a specialized C++ translation unit here, and the
// engine's stated purpose is to stay small enough for edge devices.
//
// They are separated out (P4.0.8, E.2) for one reason: while `loom/loom.h` re-exported all six, every
// consumer of the umbrella header depended on them transitively, a naive grep for consumers reported
// none, and new code could accrete against a driver without anyone noticing. Anything that needs one
// now says so at its own include site, so the remaining dependency set is the include list of this
// file plus a grep for it -- auditable rather than transitive.
//
// RETIREMENT POLICY (BACKEND.md's R6 rule, extended to these by P4.0.8). A driver may be deleted only
// in the commit that re-points the last test consuming it. The non-obvious precondition is numerical:
// several of these drivers are the ground truth their model's MIL/Lua test compares against, so the
// Lua test must first carry its own reference fixture. That is the real cost, and the actual reason
// all six were still alive when this header was written.
//
// Nothing new should be added here. This file only shrinks.

#include "loom/core/vits_driver.h"
#include "loom/core/whisper_driver.h"
#include "loom/core/kokoro_driver.h"
#include "loom/core/styletts2_driver.h"
#include "loom/core/supertonic_driver.h"
#include "loom/core/matcha_driver.h"
