#include "loom/core/rule_based_phonemizer.h"

#include "loom/core/unicode.h"
#include "loom/loom_errors.h"

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

#ifdef LOOM_LUA_PUC
extern "C" int luaopen_bit(lua_State* L);
#endif

namespace loom {

namespace {

// The libraries a pure function needs. No `io`, `os`, `package` or `debug`, and the base library's two
// file loaders are removed below.
void open_pure_libs(lua_State* L) {
    const struct {
        const char* name;
        lua_CFunction open;
    } libs[] = {
        {"", luaopen_base},          {LUA_TABLIBNAME, luaopen_table}, {LUA_STRLIBNAME, luaopen_string},
        {LUA_MATHLIBNAME, luaopen_math}, {"bit", luaopen_bit},
    };
    for (const auto& lib : libs) {
        lua_pushcfunction(L, lib.open);
        lua_pushstring(L, lib.name);
        lua_call(L, 1, 0);
    }
    for (const char* name : {"dofile", "loadfile"}) {
        lua_pushnil(L);
        lua_setglobal(L, name);
    }
}

void push_codepoints(lua_State* L, const std::string& utf8) {
    const std::vector<char32_t> cps = utf8_decode(utf8);
    lua_createtable(L, static_cast<int>(cps.size()), 0);
    for (size_t i = 0; i < cps.size(); ++i) {
        lua_pushnumber(L, static_cast<lua_Number>(cps[i]));
        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
}

// An array of codepoints at `index` as UTF-8; nil reads as empty, which is how an optional result is absent.
std::string read_codepoints(lua_State* L, int index, const char* what) {
    if (lua_isnil(L, index)) return {};
    if (!lua_istable(L, index)) throw Error(std::string("phonemizer: ") + what + " is not an array of codepoints");
    std::vector<char32_t> cps(lua_objlen(L, index));
    for (size_t i = 0; i < cps.size(); ++i) {
        lua_rawgeti(L, index, static_cast<int>(i + 1));
        cps[i] = static_cast<char32_t>(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    return utf8_encode(cps);
}

std::string pop_error(lua_State* L) {
    std::string message = lua_tostring(L, -1) ? lua_tostring(L, -1) : "(no message)";
    lua_pop(L, 1);
    return message;
}

} // namespace

std::unique_ptr<RuleBasedPhonemizer> RuleBasedPhonemizer::load(const GgufModel& model) {
    if (!model.has_kv("loom.phonemizer.kind")) return nullptr;
    const std::string kind = model.kv_str("loom.phonemizer.kind");
    if (kind != "lua") throw LoadError("RuleBasedPhonemizer: loom.phonemizer.kind is '" + kind + "', not 'lua'");

    std::unique_ptr<RuleBasedPhonemizer> self(new RuleBasedPhonemizer());
    self->languages_ = model.kv_arr_str("loom.text.languages");
    for (const auto& code : self->languages_) {
        const std::string key = "loom.phonemizer.data." + code;
        self->data_.push_back(model.has_kv(key) ? model.kv_str(key) : std::string());
    }
    if (model.has_kv("loom.phonemizer.ruleset")) self->ruleset_ = model.kv_str("loom.phonemizer.ruleset");

    self->L_ = luaL_newstate();
    if (self->L_ == nullptr) throw LoadError("RuleBasedPhonemizer: luaL_newstate() failed (out of memory)");
    lua_State* L = self->L_;
    open_pure_libs(L);
    lua_pushlightuserdata(L, self.get());
    lua_pushcclosure(L, &RuleBasedPhonemizer::l_phonemizer_data, 1);
    lua_setglobal(L, "phonemizer_data");

    const std::string script = model.kv_str("loom.phonemizer.script");
    if (luaL_loadbuffer(L, script.data(), script.size(), "=loom.phonemizer.script") != 0 ||
        lua_pcall(L, 0, 0, 0) != 0) {
        throw LoadError("RuleBasedPhonemizer: the script failed to load: " + pop_error(L));
    }
    lua_getglobal(L, "phonemize");
    const bool has_phonemize = lua_isfunction(L, -1);
    lua_pop(L, 1);
    if (!has_phonemize) throw LoadError("RuleBasedPhonemizer: the script defines no global function 'phonemize'");
    return self;
}

RuleBasedPhonemizer::~RuleBasedPhonemizer() {
    if (L_ != nullptr) lua_close(L_);
}

int RuleBasedPhonemizer::l_phonemizer_data(lua_State* L) {
    const auto* self = static_cast<const RuleBasedPhonemizer*>(lua_touserdata(L, lua_upvalueindex(1)));
    const auto lang = static_cast<long long>(luaL_checknumber(L, 1));
    if (lang < 0 || lang >= static_cast<long long>(self->data_.size()) || self->data_[lang].empty()) {
        lua_pushnil(L);
    } else {
        lua_pushlstring(L, self->data_[lang].data(), self->data_[lang].size());
    }
    return 1;
}

void RuleBasedPhonemizer::check_lang(int lang) const {
    if (lang < 0 || lang >= static_cast<int>(languages_.size())) {
        throw Error("RuleBasedPhonemizer: language index " + std::to_string(lang) + " is outside the " +
                    std::to_string(languages_.size()) + " languages of loom.text.languages");
    }
}

RuleBasedPhonemizer::Result RuleBasedPhonemizer::phonemize(const std::string& text, int lang) {
    check_lang(lang);
    lua_State* L = L_;
    lua_getglobal(L, "phonemize");
    push_codepoints(L, text);
    lua_pushnumber(L, lang);
    if (lua_pcall(L, 2, 2, 0) != 0) throw Error("RuleBasedPhonemizer: phonemize: " + pop_error(L));
    Result result;
    try {
        result.ipa = read_codepoints(L, -2, "phonemize's first result");
        result.dropped = read_codepoints(L, -1, "phonemize's second result");
    } catch (...) {
        lua_pop(L, 2);
        throw;
    }
    lua_pop(L, 2);
    return result;
}

void RuleBasedPhonemizer::set_lexicon(int lang, const std::vector<std::pair<std::string, std::string>>& entries) {
    check_lang(lang);
    lua_State* L = L_;
    lua_getglobal(L, "set_lexicon");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        throw Error("RuleBasedPhonemizer: the script defines no 'set_lexicon'");
    }
    lua_pushnumber(L, lang);
    lua_createtable(L, static_cast<int>(entries.size()), 0);
    for (size_t i = 0; i < entries.size(); ++i) {
        lua_createtable(L, 2, 0);
        push_codepoints(L, entries[i].first);
        lua_rawseti(L, -2, 1);
        push_codepoints(L, entries[i].second);
        lua_rawseti(L, -2, 2);
        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
    if (lua_pcall(L, 2, 0, 0) != 0) throw Error("RuleBasedPhonemizer: set_lexicon: " + pop_error(L));
}

} // namespace loom
