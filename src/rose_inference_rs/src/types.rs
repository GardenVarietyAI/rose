use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub model_path: String,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationKwargs {
    pub max_input_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub top_p: Option<f64>,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub config: InferenceConfig,
    pub generation_kwargs: GenerationKwargs,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
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
        finish_reason: String,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
}
