use anyhow::Result;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::models::CausalLM;
use candle_core::Device;

pub async fn stream(
    _model: &mut dyn CausalLM,
    _tokenizer: &Tokenizer,
    _device: Device,
    _prompt: &str,
    _max_input_tokens: usize,
    _max_output_tokens: usize,
    _temperature: f32,
    _top_p: Option<f32>,
    stream_tx: mpsc::Sender<crate::types::InferenceResponse>,
    _seed: u64,
    _repeat_penalty: f32,
    _repeat_last_n: usize,
) -> Result<()> {
    let _ = stream_tx
        .send(crate::types::InferenceResponse::InputTokensCounted { input_tokens: 0 })
        .await;

    let _ = stream_tx
        .send(crate::types::InferenceResponse::Token {
            token: "hello".to_string(),
            position: 0,
            logprob: None,
            top_logprobs: None,
        })
        .await;

    let _ = stream_tx
        .send(crate::types::InferenceResponse::Complete {
            input_tokens: 0,
            output_tokens: 1,
            total_tokens: 1,
            finish_reason: "stop".to_string(),
        })
        .await;

    Ok(())
}
