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
}
