use std::time::Duration;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use crate::error::InferenceError;
use crate::models::CausalLM;
use crate::tensor_pool::return_to_pool;
use crate::types::{FinishReason, InferenceResponse};

// Qwen3 special tokens for tool calling
const TOOL_CALL_START_TOKEN: u32 = 151657;
const TOOL_CALL_END_TOKEN: u32 = 151658;

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

    // Tool call detection state
    let mut in_tool_call = false;
    let mut tool_call_tokens: Vec<u32> = Vec::new();
    let mut consecutive_tool_starts = 0;

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
        let logits = {
            let result = model.forward(&input_tensor, past_length)?;
            // Pool intermediate tensors for reuse
            if input_tensor.shape().dims() == &[1, 1] {
                return_to_pool(input_tensor);
            }
            result
        };

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

        // Check for tool call tokens
        if sampled_token == TOOL_CALL_START_TOKEN {
            // Only increment if we're already in a tool call (shouldn't happen)
            if in_tool_call {
                consecutive_tool_starts += 1;
                tracing::warn!("Detected nested <tool_call> token ({}) - occurrence {}", sampled_token, consecutive_tool_starts);

                // Break if we see too many consecutive tool call starts (model is stuck)
                if consecutive_tool_starts > 3 {
                    tracing::warn!("Model stuck generating tool_call tokens, stopping");
                    finish_reason = FinishReason::Stop;
                    break;
                }
            } else {
                tracing::info!("Detected <tool_call> token ({})", sampled_token);
                consecutive_tool_starts = 0; // Reset counter for fresh tool call
            }

            in_tool_call = true;
            tool_call_tokens.clear();
            // Set next token before continuing
            next_token = Some(sampled_token);
            continue;
        } else if sampled_token == TOOL_CALL_END_TOKEN && in_tool_call {
            // Parse and emit tool call

            if !tool_call_tokens.is_empty() {
                // Decode the tokens between <tool_call> and </tool_call>
                let tool_json = tokenizer
                    .decode(&tool_call_tokens, false)
                    .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

                tracing::info!("Tool call content: {}", tool_json);

                // Try to parse as JSON
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&tool_json) {
                    if let Some(name) = json_value.get("name").and_then(|v| v.as_str()) {
                        let arguments = json_value
                            .get("arguments")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));

                        // TODO: Generate call_id at response formatting stage, not during inference
                        let call_id = uuid::Uuid::new_v4()
                            .simple()
                            .to_string()
                            .chars()
                            .take(8)
                            .collect::<String>();

                        let tool_call_msg = InferenceResponse::ToolCall {
                            name: name.to_string(),
                            arguments,
                            call_id: format!("call_{}", call_id),
                        };

                        if stream_tx.send(tool_call_msg).await.is_err() {
                            finish_reason = FinishReason::Stop;
                            break;
                        }

                        // Mark finish reason as tool_calls and end generation immediately
                        finish_reason = FinishReason::ToolCalls;
                        break;
                    }
                }
            }
            // We broke out above after emitting the tool call
            break;
        } else if in_tool_call {
            // Collect tokens for tool call
            tool_call_tokens.push(sampled_token);
            consecutive_tool_starts = 0;  // Reset counter when we get non-tool_call tokens
            // Set next token and continue collecting tool call content
            next_token = Some(sampled_token);
            continue;
        }

        let token_text = tokenizer
            .decode(&[sampled_token], true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?
            .replace('\u{FFFD}', ""); // Remove replacement characters

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

        // Emit token (only if not in tool call)
        let token_msg = InferenceResponse::Token {
            token: token_text,
            token_id: sampled_token,
            position: index as u32,
            logprob: None,
            top_logprobs: None,
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

        // Set next token for generation
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
