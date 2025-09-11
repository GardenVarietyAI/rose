use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub tool_call_id: Option<String>,
}

#[pyclass]
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationKwargs {
    #[pyo3(get, set)]
    pub model_path: String,
    #[pyo3(get, set)]
    pub tokenizer_path: String,
    #[pyo3(get, set)]
    pub model_kind: String,
    #[pyo3(get, set)]
    pub response_chain_id: Option<String>,
    #[pyo3(get, set)]
    pub max_input_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub max_output_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub top_p: Option<f64>,
    #[pyo3(get, set)]
    pub temperature: Option<f64>,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub repeat_penalty: Option<f32>,
    #[pyo3(get, set)]
    pub repeat_last_n: Option<usize>,
    #[pyo3(get, set)]
    pub stop: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub logprobs: Option<bool>,
    #[pyo3(get, set)]
    pub top_logprobs: Option<usize>,
    #[pyo3(get, set)]
    pub chat_template: Option<String>,
    #[pyo3(get, set)]
    pub enable_thinking: Option<bool>,
}

#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature = (role, content, tool_call_id=None))]
    fn new(role: String, content: String, tool_call_id: Option<String>) -> Self {
        Self { role, content, tool_call_id }
    }
}

#[pymethods]
impl GenerationKwargs {
    #[new]
    #[pyo3(signature = (
        model_path,
        tokenizer_path,
        model_kind,
        response_chain_id=None,
        max_input_tokens=None,
        max_output_tokens=None,
        top_p=None,
        temperature=None,
        seed=None,
        repeat_penalty=None,
        repeat_last_n=None,
        stop=None,
        logprobs=None,
        top_logprobs=None,
        chat_template=None,
        enable_thinking=None,
    ))]
    fn new(
        model_path: String,
        tokenizer_path: String,
        model_kind: String,
        response_chain_id: Option<String>,
        max_input_tokens: Option<usize>,
        max_output_tokens: Option<usize>,
        top_p: Option<f64>,
        temperature: Option<f64>,
        seed: Option<u64>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        stop: Option<Vec<String>>,
        logprobs: Option<bool>,
        top_logprobs: Option<usize>,
        chat_template: Option<String>,
        enable_thinking: Option<bool>,
    ) -> Self {
        Self {
            model_path,
            tokenizer_path,
            model_kind,
            response_chain_id,
            max_input_tokens,
            max_output_tokens,
            top_p,
            temperature,
            seed,
            repeat_penalty,
            repeat_last_n,
            stop,
            logprobs,
            top_logprobs,
            chat_template,
            enable_thinking,
        }
    }
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

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopLogProb {
    #[pyo3(get)]
    pub token: String,
    #[pyo3(get)]
    pub logprob: f32,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct TokenEvent {
    #[pyo3(get)]
    pub token: String,
    #[pyo3(get)]
    pub token_id: u32,
    #[pyo3(get)]
    pub position: u32,
    #[pyo3(get)]
    pub logprob: Option<f32>,
    #[pyo3(get)]
    pub top_logprobs: Option<Vec<TopLogProb>>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct CompleteEvent {
    #[pyo3(get)]
    pub input_tokens: u32,
    #[pyo3(get)]
    pub output_tokens: u32,
    #[pyo3(get)]
    pub total_tokens: u32,
    #[pyo3(get)]
    pub finish_reason: String,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct InputTokensCountedEvent {
    #[pyo3(get)]
    pub input_tokens: u32,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ErrorEvent {
    #[pyo3(get)]
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InferenceResponse {
    InputTokensCounted {
        input_tokens: u32,
    },
    Token {
        token: String,
        token_id: u32,
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
