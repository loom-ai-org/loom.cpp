#pragma once

// A checkpoint's chat template, as role tags the engine concatenates (P4.23, ADR-018).
//
// WHY THERE IS NO JINJA HERE. The template a checkpoint ships is a Jinja program in
// `tokenizer_config.json`, and it is per-MODEL -- which is the one kind of complexity this engine does
// not carry (ADR-003). The exporter, which already has a Jinja evaluator and the real template, reduces
// it to the role tags it actually emits and VERIFIES that reduction against `apply_chat_template` on
// real conversations before writing it. What lands here is data: two parallel arrays and three strings.
//
// The consequence worth stating: a template whose output is not
// `prologue + sum(prefix[role] + content + suffix[role]) + generation_prefix` is not exported at all,
// and the model then has no chat door rather than a wrong one. `loom-exporter`'s
// `chat_template_export.py` is where a checkpoint is rejected and why.
//
// A ROLE THE FILE DOES NOT DECLARE IS AN ERROR, not a dropped argument. Gemma 3 is the live case: its
// template folds a system message into the text of the first user turn instead of emitting a block for
// it, so its export declares `user` and `assistant` only, and asking it for a system prompt says so.

#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

struct ChatMessage {
    std::string role;
    std::string content;
};

class ChatTemplate {
public:
    // Returns nullptr when `model` carries no "tokenizer.chat_template.*" KVs -- a base model, a
    // checkpoint whose template did not decompose, and every GGUF exported before P4.23.
    static std::unique_ptr<ChatTemplate> load(const GgufModel& model);

    // The prompt text for `messages`. `add_generation_prompt` appends the opening of the reply the
    // model is being asked for (`<start_of_turn>model\n`), which is what turns a transcript into a
    // question -- pass false only when building a training-shaped transcript.
    //
    // Throws loom::Error naming the role when `messages` uses one this checkpoint does not declare.
    //
    // The result still has to be TOKENIZED, and by this model's own vocabulary: the markers it
    // contains are added tokens, which `BpeVocab::encode` emits atomically only for a file carrying
    // `tokenizer.ggml.token_type`. The two halves of P4.23 land together or not at all.
    std::string apply(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true) const;

    // Which roles this checkpoint's template can express, in the order the export verified them.
    const std::vector<std::string>& roles() const { return roles_; }
    bool has_role(const std::string& role) const;

    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;

private:
    ChatTemplate() = default;

    std::vector<std::string> roles_;
    std::vector<std::string> prefixes_;
    std::vector<std::string> suffixes_;
    // What precedes the first message. Two of them because a template may inject a DEFAULT system turn
    // when the conversation does not open with one -- SmolLM2 injects "You are a helpful AI assistant
    // named SmolLM, trained by Hugging Face" -- so which prologue applies is a property of the
    // conversation, not of the file.
    std::string prologue_;
    std::string system_prologue_;
    std::string generation_prefix_;
    // Whether the template trims each message's content. Gemma 3's does (`{{ content | trim }}`), and
    // an engine that did not would disagree with `transformers` by whitespace on any multi-line prompt.
    bool trim_content_ = false;
};

} // namespace loom
