#include "loom/core/symbol_table.h"
#include "loom/loom_errors.h"

#include <cctype>
#include <cmath>

namespace loom {

void SymbolEnv::set(const std::string& name, double value) {
    values_[name] = value;
}

bool SymbolEnv::has(const std::string& name) const {
    return values_.find(name) != values_.end();
}

double SymbolEnv::get(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        throw SchemaError("SymbolEnv: unbound symbol '" + name + "'");
    }
    return it->second;
}

namespace {

// Minimal recursive-descent evaluator over `expr`'s grammar (see symbol_table.h). Operates directly on
// the string with a cursor rather than a separate tokenizer pass -- the grammar is small enough that
// this stays readable.
class ExprParser {
public:
    ExprParser(const std::string& src, const SymbolEnv& env) : src_(src), env_(env) {}

    double parse() {
        double v = parse_expr();
        skip_ws();
        if (pos_ != src_.size()) {
            throw SchemaError("SymbolEnv: unexpected trailing input in expression '" + src_ + "' at " + std::to_string(pos_));
        }
        return v;
    }

private:
    const std::string& src_;
    const SymbolEnv& env_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) ++pos_;
    }

    char peek() {
        skip_ws();
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }

    bool consume(char c) {
        if (peek() == c) { ++pos_; return true; }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) {
            throw SchemaError(std::string("SymbolEnv: expected '") + c + "' in expression '" + src_ + "' at " + std::to_string(pos_));
        }
    }

    double parse_expr() {
        double v = parse_term();
        for (;;) {
            if (consume('+')) v += parse_term();
            else if (consume('-')) v -= parse_term();
            else break;
        }
        return v;
    }

    double parse_term() {
        double v = parse_factor();
        for (;;) {
            if (consume('*')) v *= parse_factor();
            else if (consume('/')) v /= parse_factor();
            else break;
        }
        return v;
    }

    double parse_factor() {
        if (consume('-')) return -parse_factor();
        if (consume('(')) {
            double v = parse_expr();
            expect(')');
            return v;
        }
        if (consume('$')) {
            return env_.get(parse_ident());
        }
        skip_ws();
        if (pos_ < src_.size() && (std::isalpha(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
            std::string ident = parse_ident();
            if (ident == "sqrt") {
                expect('(');
                double v = parse_expr();
                expect(')');
                return std::sqrt(v);
            }
            if (ident == "floor") {
                expect('(');
                double v = parse_expr();
                expect(')');
                return std::floor(v);
            }
            // Bare identifier with no '$' sigil: still treated as a symbol reference for convenience
            // (e.g. a lone "n_layer" attribute value with no arithmetic around it).
            return env_.get(ident);
        }
        return parse_number();
    }

    std::string parse_ident() {
        skip_ws();
        size_t start = pos_;
        while (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) ++pos_;
        if (pos_ == start) {
            throw SchemaError("SymbolEnv: expected identifier in expression '" + src_ + "' at " + std::to_string(pos_));
        }
        return src_.substr(start, pos_ - start);
    }

    double parse_number() {
        skip_ws();
        size_t start = pos_;
        size_t consumed = 0;
        double v = 0.0;
        try {
            v = std::stod(src_.substr(pos_), &consumed);
        } catch (const std::exception&) {
            throw SchemaError("SymbolEnv: expected number in expression '" + src_ + "' at " + std::to_string(pos_));
        }
        if (consumed == 0) {
            throw SchemaError("SymbolEnv: expected number in expression '" + src_ + "' at " + std::to_string(pos_));
        }
        pos_ = start + consumed;
        return v;
    }
};

} // namespace

double SymbolEnv::eval(const std::string& expr) const {
    // A leading '$' is just the usual sigil for "this whole attribute value is a symbol reference";
    // ExprParser's parse_factor() already consumes '$' wherever it appears (including mid-expression,
    // e.g. "1/sqrt($n_embd_head_k)"), so no special-casing is needed here.
    ExprParser parser(expr, *this);
    return parser.parse();
}

} // namespace loom
