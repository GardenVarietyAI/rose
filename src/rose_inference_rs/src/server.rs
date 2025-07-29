use anyhow::Result;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use futures_util::stream::SplitSink;
use futures_util::SinkExt;
use tracing::{info, debug, warn};

use crate::models;
use crate::error::InferenceError;

#[derive(Clone)]
pub struct InferenceServer {
    device: Device,
}

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub config: ModelConfig,
    pub generation_kwargs: GenerationKwargs,
    pub messages: Option<Vec<Message>>,
    pub prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct GenerationKwargs {
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum InferenceResponse {
    #[serde(rename = "input_tokens_counted")]
    InputTokensCounted { input_tokens: u32 },
    #[serde(rename = "token")]
    Token {
        token: String,
        position: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_logprobs: Option<Vec<LogProb>>,
    },
    #[serde(rename = "complete")]
    Complete {
        input_tokens: u32,
        output_tokens: u32,
        total_tokens: u32,
    },
    #[serde(rename = "error")]
    Error { error: String },
}

#[derive(Debug, Serialize)]
pub struct LogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

impl InferenceServer {
    pub async fn new(device_str: String) -> Result<Self> {
        let device = match device_str.as_str() {
            "cuda" => Device::new_cuda(0)?,
            "cpu" => Device::Cpu,
            "metal" => Device::new_metal(0)?,
            _ => Device::Cpu,
        };

        info!("Initialized inference server with device: {:?}", device);

        Ok(Self {
            device,
        })
    }

    pub async fn process_streaming_request<S>(&self, request_text: &str, ws_sender: &mut SplitSink<S, WsMessage>) -> Result<()>
    where
        S: futures_util::Stream<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + futures_util::Sink<WsMessage>,
        <S as futures_util::Sink<WsMessage>>::Error: std::error::Error + Send + Sync + 'static,
    {
        debug!("Processing streaming request: {}", request_text);

        let request: InferenceRequest = serde_json::from_str(request_text)
            .map_err(|e| InferenceError::InvalidRequest(e.to_string()))?;

        let prompt = if let Some(prompt) = request.prompt {
            prompt
        } else if let Some(messages) = request.messages {
            // Format messages using simple chat format for now
            // TODO: Implement proper chat template support when available in tokenizers crate
            self.format_messages(&messages)
        } else {
            return Err(InferenceError::InvalidRequest("Either prompt or messages must be provided".to_string()).into());
        };

        let (stream_tx, mut stream_rx) = mpsc::channel::<InferenceResponse>(100);

        let max_tokens = request.generation_kwargs.max_new_tokens.unwrap_or(512);
        let temperature = request.generation_kwargs.temperature.or(request.config.temperature).unwrap_or(0.7);
        let top_p = request.generation_kwargs.top_p;

        let generation_task = {
            let model_path = request.config.model_path.clone();
            let device = self.device.clone();
            let prompt = prompt.clone();
            tokio::spawn(async move {
                models::generate_streaming(&model_path, device, &prompt, max_tokens, temperature, top_p, stream_tx).await
            })
        };

        while let Some(response) = stream_rx.recv().await {
            let response_json = serde_json::to_string(&response)?;
            if let Err(e) = ws_sender.send(WsMessage::Text(response_json)).await {
                warn!("Failed to send streaming response: {}", e);
                break;
            }
        }

        let _ = generation_task.await;

        Ok(())
    }

    fn format_messages(&self, messages: &[Message]) -> String {
        // Use Qwen2.5 chat template format
        let mut prompt_parts = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt_parts.push(format!("<|im_start|>system\n{}<|im_end|>", msg.content));
                }
                "user" => {
                    prompt_parts.push(format!("<|im_start|>user\n{}<|im_end|>", msg.content));
                }
                "assistant" => {
                    prompt_parts.push(format!("<|im_start|>assistant\n{}<|im_end|>", msg.content));
                }
                _ => {
                    prompt_parts.push(format!("<|im_start|>{}\n{}<|im_end|>", msg.role, msg.content));
                }
            }
        }

        prompt_parts.push("<|im_start|>assistant\n".to_string());
        prompt_parts.join("\n")
    }
}
