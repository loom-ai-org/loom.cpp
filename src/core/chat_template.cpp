#include "loom/core/chat_template.h"

#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <algorithm>

namespace loom {
namespace {

// One space of leading and trailing whitespace removed, matching Jinja's own `| trim` (which strips
// the ASCII whitespace set, not the Unicode one).
std::string trim(const std::string& text) {
    const char* ws = " \t\n\r\f\v";
    const size_t begin = text.find_first_not_of(ws);
    if (begin == std::string::npos) return "";
    return text.substr(begin, text.find_last_not_of(ws) - begin + 1);
}

} // namespace

std::unique_ptr<ChatTemplate> ChatTemplate::load(const GgufModel& model) {
    if (!model.has_kv("tokenizer.chat_template.roles")) {
        return nullptr;
    }
    auto tmpl = std::unique_ptr<ChatTemplate>(new ChatTemplate());
    tmpl->roles_ = model.kv_arr_str("tokenizer.chat_template.roles");
    tmpl->prefixes_ = model.kv_arr_str("tokenizer.chat_template.prefixes");
    tmpl->suffixes_ = model.kv_arr_str("tokenizer.chat_template.suffixes");
    if (tmpl->roles_.empty() || tmpl->prefixes_.size() != tmpl->roles_.size() ||
        tmpl->suffixes_.size() != tmpl->roles_.size()) {
        throw LoadError("ChatTemplate::load: tokenizer.chat_template.roles/prefixes/suffixes are " +
                         std::to_string(tmpl->roles_.size()) + "/" + std::to_string(tmpl->prefixes_.size()) +
                         "/" + std::to_string(tmpl->suffixes_.size()) + " entries; the three are parallel "
                         "by definition and a role with no tags is not a role");
    }
    // Every string but the roles may legitimately be empty -- a ChatML template has no prologue at all
    // -- so these read with defaults rather than requiring the KV.
    tmpl->prologue_ = model.has_kv("tokenizer.chat_template.prologue")
        ? model.kv_str("tokenizer.chat_template.prologue") : "";
    tmpl->system_prologue_ = model.has_kv("tokenizer.chat_template.system_prologue")
        ? model.kv_str("tokenizer.chat_template.system_prologue") : "";
    tmpl->generation_prefix_ = model.has_kv("tokenizer.chat_template.generation_prefix")
        ? model.kv_str("tokenizer.chat_template.generation_prefix") : "";
    tmpl->trim_content_ = model.kv_bool("tokenizer.chat_template.trim_content", false);
    return tmpl;
}

bool ChatTemplate::has_role(const std::string& role) const {
    return std::find(roles_.begin(), roles_.end(), role) != roles_.end();
}

std::string ChatTemplate::apply(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt) const {
    if (messages.empty()) {
        throw Error("ChatTemplate::apply: no messages. A chat prompt is a conversation, and an empty "
                    "one asks the model nothing.");
    }
    // Which prologue is a property of the CONVERSATION rather than of the file: a template that
    // injects a default system turn injects it only when the caller supplied none.
    std::string out = messages.front().role == "system" ? system_prologue_ : prologue_;
    for (const ChatMessage& message : messages) {
        const auto it = std::find(roles_.begin(), roles_.end(), message.role);
        if (it == roles_.end()) {
            std::string known;
            for (const std::string& role : roles_) known += (known.empty() ? "" : ", ") + role;
            throw Error("ChatTemplate::apply: this checkpoint's template has no '" + message.role +
                        "' role -- it declares " + known + ". Gemma 3 is the usual case: its template "
                        "folds a system message into the first user turn rather than emitting a block "
                        "for it, so the system text belongs at the top of that turn's own content.");
        }
        const size_t i = static_cast<size_t>(it - roles_.begin());
        out += prefixes_[i] + (trim_content_ ? trim(message.content) : message.content) + suffixes_[i];
    }
    if (add_generation_prompt) out += generation_prefix_;
    return out;
}

} // namespace loom
