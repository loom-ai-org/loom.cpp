#pragma once

#include <stdexcept>
#include <string>

// EXPORTED TYPEINFO, WHICH IS WHAT MAKES `catch (const loom::Error&)` WORK ACROSS A LIBRARY
// BOUNDARY. These classes are header-only, so their typeinfo is emitted as a weak symbol into every
// binary that includes this file -- the engine, which throws, and the Python extension module, which
// catches. Whether those two weak copies are merged into one is a platform question, and the two
// platforms answer differently:
//
//   * ELF resolves through a flat namespace, so the copies coalesce and the typeinfo addresses are
//     equal at run time. This has always worked and needed nothing.
//   * Mach-O uses a two-level namespace, and pybind11 builds every extension module with
//     `-fvisibility=hidden` (`pybind11_add_module` sets CXX_VISIBILITY_PRESET). A hidden weak symbol
//     is not a candidate for coalescing, so `_loom.so` gets its own private `typeinfo for
//     loom::Error` -- and Apple's libc++ compares type_info BY ADDRESS. Two addresses, no match: the
//     `catch` clause below is simply skipped and the exception continues as its std:: base.
//
// The symptom is not a crash and not a lost error. It is a `loom::LoadError` arriving in Python as a
// plain `RuntimeError` instead of `loom.LoomError`, so a caller who wrote `except loom.LoomError`
// -- the documented way to tell "your GGUF is wrong" from "this binding is wrong" -- stops catching
// anything. `tests/ci/test_vocab_dispatch.py` in loom-py is what noticed, on the first macOS build.
//
// Default visibility on the class attaches to its typeinfo and vtable, which is exactly the subset
// that has to be shared. MSVC has no equivalent spelling and no equivalent problem here, since it
// compares typeinfo by name.
#if defined(_WIN32)
#  define LOOM_ERROR_API
#else
#  define LOOM_ERROR_API __attribute__((visibility("default")))
#endif

namespace loom {

// Base class for all loom-engine errors, so callers can catch broadly with `catch (const loom::Error&)`
// or narrowly by the specific subtype below.
class LOOM_ERROR_API Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

// Thrown by GgufModel::load() when a .gguf file is missing, malformed, or missing a required tensor/KV.
class LOOM_ERROR_API LoadError : public Error {
public:
    explicit LoadError(const std::string& what) : Error(what) {}
};

// Thrown by GraphTopology::parse() when the embedded JSON graph definition is malformed or uses an
// unsupported schema version/construct.
class LOOM_ERROR_API SchemaError : public Error {
public:
    explicit SchemaError(const std::string& what) : Error(what) {}
};

// Thrown by PrimitiveRegistry::get() when a JSON node references an "op" with no registered primitive.
class LOOM_ERROR_API UnknownOpError : public Error {
public:
    explicit UnknownOpError(const std::string& what) : Error(what) {}
};

} // namespace loom
