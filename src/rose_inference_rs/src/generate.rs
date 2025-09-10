use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::error::InferenceError;
use crate::logprobs;
use crate::models::CausalLM;
use crate::types::{FinishReason, InferenceResponse, TopLogProb};

pub async fn stream(
    model: &mut dyn CausalLM,
    tokenizer: &Tokenizer,
    device: Device,
    prompt: &str,
    max_input_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    stop: Option<&[String]>,
    stream_tx: mpsc::Sender<InferenceResponse>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
) -> Result<()> {
    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

    let mut all_tokens = encoding.get_ids().to_vec();

    // Sliding context window
    if all_tokens.len() > max_input_tokens {
        tracing::warn!(
            "Truncating input from {} to {} tokens",
            all_tokens.len(),
            max_input_tokens
        );
        all_tokens = all_tokens[all_tokens.len() - max_input_tokens..].to_vec();
    }

    let input_tokens = all_tokens.len() as u32;

    // Notify input token count
    if stream_tx
        .send(InferenceResponse::InputTokensCounted { input_tokens })
        .await
        .is_err()
    {
        // Consumer gone
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

    let mut next_token: Option<u32> = None;
    let mut decoded_so_far = String::new();
    let mut finish_reason = FinishReason::Length;

    let mut single_token_buf = vec![0u32; 1];

    for index in 0..max_output_tokens {
        let (input_slice, input_len) = match next_token {
            Some(tok) => {
                single_token_buf[0] = tok;
                (single_token_buf.as_slice(), 1)
            }
            None => (all_tokens.as_slice(), all_tokens.len()),
        };
        let input_tensor = Tensor::from_slice(input_slice, (1, input_len), &device)?;
        let past_length = if next_token.is_some() {
            all_tokens.len() - 1
        } else {
            0
        };
        let logits = model.forward(&input_tensor, past_length)?;

        tokio::task::yield_now().await;

        let logits = logits.squeeze(0)?.squeeze(0)?;
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

        let sampled_token = logits_processor.sample(&logits)?;
        all_tokens.push(sampled_token);

        let token_text = tokenizer
            .decode(&[sampled_token], true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?
            .replace('\u{FFFD}', ""); // Remove replacement characters

        // Calculate logprobs if requested
        let (logprob, top_logprobs_data) = if logprobs.unwrap_or(false) {
            let logprobs_result =
                logprobs::calculate_logprobs(&logits, sampled_token, tokenizer, top_logprobs)?;
            let converted_top_logprobs = logprobs_result.top_logprobs.map(|tlps|
                tlps.into_iter().map(|tlp| TopLogProb {
                    token: tlp.token,
                    logprob: tlp.logprob,
                }).collect()
            );
            (Some(logprobs_result.logprob), converted_top_logprobs)
        } else {
            (None, None)
        };

        // Check for stop sequences
        decoded_so_far.push_str(&token_text);
        if let Some(stops) = stop {
            if stops.iter().any(|s| decoded_so_far.ends_with(s)) {
                // don't emit this stop piece; keep counts consistent by removing it
                decoded_so_far.truncate(decoded_so_far.len() - token_text.len());
                let _ = all_tokens.pop();
                finish_reason = FinishReason::Stop;
                break;
            }
        }

        // Emit token
        let token_msg = InferenceResponse::Token {
            token: token_text,
            token_id: sampled_token,
            position: index as u32,
            logprob,
            top_logprobs: top_logprobs_data,
        };
        if stream_tx.send(token_msg).await.is_err() {
            finish_reason = FinishReason::Stop;
            break;
        }

        tokio::time::sleep(Duration::from_millis(1)).await;

        if sampled_token == model.eos_token_id()
            || sampled_token == model.im_end_token_id(tokenizer)
        {
            finish_reason = FinishReason::Stop;
            break;
        }

        // Repetition guard
        if index > 10 {
            let recent = &all_tokens[all_tokens.len().saturating_sub(8)..];
            if recent.len() >= 8 && recent.iter().all(|&t| t == sampled_token) {
                finish_reason = FinishReason::Stop;
                break;
            }
        }

        next_token = Some(sampled_token);
    }

    let output_tokens = all_tokens.len() as u32 - input_tokens;

    let complete_msg = InferenceResponse::Complete {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        finish_reason,
    };
    let _ = stream_tx.send(complete_msg).await;

    Ok(())
}
