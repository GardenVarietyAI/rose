use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationKwargs {
    pub model_path: String,
    pub tokenizer_path: String,
    pub model_kind: String,
    pub max_input_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub top_p: Option<f64>,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub stop: Option<Vec<String>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub chat_template: Option<String>,
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub generation_kwargs: GenerationKwargs,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InferenceResponse {
    InputTokensCounted {
        input_tokens: u32,
    },
    Token {
        token: String,
        position: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_logprobs: Option<Vec<TopLogProb>>,
    },
    Complete {
        input_tokens: u32,
        output_tokens: u32,
        total_tokens: u32,
        finish_reason: FinishReason,
    },
    Error {
        error: String,
    },
}
