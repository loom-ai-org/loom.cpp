#pragma once

#include <stdexcept>
#include <string>

namespace loom {

// Base class for all loom-engine errors, so callers can catch broadly with `catch (const loom::Error&)`
// or narrowly by the specific subtype below.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

// Thrown by GgufModel::load() when a .gguf file is missing, malformed, or missing a required tensor/KV.
class LoadError : public Error {
public:
    explicit LoadError(const std::string& what) : Error(what) {}
};

// Thrown by GraphTopology::parse() when the embedded JSON graph definition is malformed or uses an
// unsupported schema version/construct.
class SchemaError : public Error {
public:
    explicit SchemaError(const std::string& what) : Error(what) {}
};

// Thrown by PrimitiveRegistry::get() when a JSON node references an "op" with no registered primitive.
class UnknownOpError : public Error {
public:
    explicit UnknownOpError(const std::string& what) : Error(what) {}
};

} // namespace loom
