use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::error::InferenceError;
use crate::models::CausalLM;

pub async fn stream(
    model: &mut dyn CausalLM,
    tokenizer: &Tokenizer,
    device: Device,
    prompt: &str,
    max_input_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    stream_tx: mpsc::Sender<crate::server::InferenceResponse>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
) -> Result<()> {
    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

    let mut tokens = encoding.get_ids().to_vec();

    // Sliding context window
    if tokens.len() > max_input_tokens {
        tokens = tokens[tokens.len() - max_input_tokens..].to_vec();
    }

    let prompt_tokens = tokens.len() as u32;

    // Notify input token count
    let input_tokens_msg = crate::server::InferenceResponse::InputTokensCounted {
        input_tokens: prompt_tokens,
    };
    if stream_tx.send(input_tokens_msg).await.is_err() {
        return Ok(());
    }

    // Setup sampling
    let sampling = if temperature <= 0.0 {
        candle_transformers::generation::Sampling::ArgMax
    } else {
        match top_p {
            Some(p) => candle_transformers::generation::Sampling::TopP {
                p: p as f64,
                temperature: temperature as f64,
            },
            None => candle_transformers::generation::Sampling::All {
                temperature: temperature as f64,
            },
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    // Initial forward pass
    let prompt_tensor = Tensor::from_slice(&tokens, (1, tokens.len()), &device)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;

    let mut all_tokens = tokens.clone();
    all_tokens.push(next_token);

    // Stream first token
    let token_text = tokenizer
        .decode(&[next_token], true)
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
    let token_msg = crate::server::InferenceResponse::Token {
        token: token_text,
        position: 0,
        logprob: None,
        top_logprobs: None,
    };
    if stream_tx.send(token_msg).await.is_err() {
        return Ok(());
    }

    for index in 0..max_output_tokens {
        let next_input_tensor = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let logits = model.forward(&next_input_tensor, all_tokens.len() - 1)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        // Apply repeat penalty
        let logits = if repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            )?
        };

        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        // Decode token
        let token_text = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

        // Stream token
        let token_msg = crate::server::InferenceResponse::Token {
            token: token_text,
            position: index + 1,
            logprob: None,
            top_logprobs: None,
        };
        if stream_tx.send(token_msg).await.is_err() {
            break;
        }

        // EOS detection (Qwen2.5 uses 151645 as EOS, <|im_end|> fallback)
        let eos_token = model.eos_token_id();
        let im_end_token = model.im_end_token_id(tokenizer);
        if next_token == eos_token || next_token == im_end_token {
            break;
        }

        // Prevent infinite token repetition
        if index > 10 {
            let recent_tokens = &all_tokens[all_tokens.len().saturating_sub(8)..];
            if recent_tokens.len() >= 8 && recent_tokens.iter().all(|&t| t == next_token) {
                break;
            }
        }
    }

    // Notify completion
    let output_tokens = all_tokens.len() as u32;
    let complete_msg = crate::server::InferenceResponse::Complete {
        input_tokens: prompt_tokens,
        output_tokens,
        total_tokens: prompt_tokens + output_tokens,
    };
    let _ = stream_tx.send(complete_msg).await;

    Ok(())
}
