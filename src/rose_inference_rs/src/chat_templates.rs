use crate::types::Message;

#[derive(Debug, Clone)]
pub enum ChatTemplate {
    Qwen2,
    Qwen3,
}

impl ChatTemplate {
    pub fn from_string(template: &str) -> Self {
        match template.to_lowercase().as_str() {
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            _ => Self::Qwen3, // Default to Qwen3
        }
    }

    pub fn format_messages(&self, messages: &[Message]) -> String {
        match self {
            Self::Qwen2 | Self::Qwen3 => {
                // Both use the same <|im_start|> format
                self.format_qwen_style(messages)
            }
        }
    }

    fn format_qwen_style(&self, messages: &[Message]) -> String {
        let mut out = Vec::new();
        for m in messages {
            out.push(format!("<|im_start|>{}\n{}<|im_end|>", m.role, m.content));
        }
        out.push("<|im_start|>assistant\n".to_string());
        out.join("\n")
    }
}
