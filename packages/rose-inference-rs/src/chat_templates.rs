use crate::types::Message;

#[derive(Debug, Clone)]
pub enum ChatTemplate {
    Qwen3,
}

impl ChatTemplate {
    pub fn from_string(template: &str) -> Self {
        match template.to_lowercase().as_str() {
            "qwen3" => Self::Qwen3,
            _ => Self::Qwen3,
        }
    }

    pub fn format_messages(&self, messages: &[Message], enable_thinking: Option<bool>) -> String {
        match self {
            Self::Qwen3 => self.format_qwen_style(messages, enable_thinking),
        }
    }

    fn format_qwen_style(&self, messages: &[Message], enable_thinking: Option<bool>) -> String {
        let mut out = Vec::new();
        for m in messages {
            out.push(format!("<|im_start|>{}\n{}<|im_end|>", m.role, m.content));
        }
        out.push("<|im_start|>assistant\n".to_string());

        if enable_thinking == Some(false) {
            out.push("<think>\n\n</think>\n\n".to_string());
        }

        out.join("\n")
    }

    pub fn format_reranker_prompt(query: &str, document: &str) -> String {
        let prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        let user_content = format!(
            "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {}\n<Document>: {}",
            query, document
        );

        format!("{}{}{}", prefix, user_content, suffix)
    }
}
