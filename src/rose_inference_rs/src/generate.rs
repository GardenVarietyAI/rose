use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
// use tokio::sync::mpsc;

use crate::error::InferenceError;
use crate::models::CausalLM;
use crate::types::{FinishReason, InferenceResponse};

pub async fn stream(
    model: &mut dyn CausalLM,
    tokenizer: &Tokenizer,
    device: Device,
    prompt: &str,
    max_input_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    _stop: Option<&[String]>, // custom stop token unused for now
    // stream_tx: mpsc::Sender<InferenceResponse>,
    stream_tx: UnboundedSender<InferenceResponse>,
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
        all_tokens = all_tokens[all_tokens.len() - max_input_tokens..].to_vec();
    }

    let input_tokens = all_tokens.len() as u32;

    // Notify input token count
    // if stream_tx
    //     .send(InferenceResponse::InputTokensCounted { input_tokens })
    //     .await
    //     .is_err()
    // {
    //     // Consumer gone
    //     return Ok(());
    // }
    if stream_tx
        .send(InferenceResponse::InputTokensCounted { input_tokens })
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
    // let mut decoded_so_far = String::new();
    let mut finish_reason = FinishReason::Length;

    for index in 0..max_output_tokens {
        let input = match next_token {
            Some(tok) => vec![tok],     // step with last token
            None => all_tokens.clone(), // first step: use prompt
        };
        let input_tensor = Tensor::from_slice(&input, (1, input.len()), &device)?;
        let logits = model.forward(&input_tensor, all_tokens.len() - input.len())?;
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
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

        // Check for stop sequences
        // decoded_so_far.push_str(&token_text);
        // if let Some(stops) = stop {
        //     if stops.iter().any(|s| decoded_so_far.ends_with(s)) {
        //         // don't emit this stop piece; keep counts consistent by removing it
        //         decoded_so_far.truncate(decoded_so_far.len() - token_text.len());
        //         let _ = all_tokens.pop();
        //         finish_reason = FinishReason::Stop;
        //         break;
        //     }
        // }

        // Emit token
        let token_msg = InferenceResponse::Token {
            token: token_text,
            position: index as u32,
            logprob: None,
            top_logprobs: None,
        };
        // if stream_tx.send(token_msg).await.is_err() {
        //     finish_reason = FinishReason::Stop;
        //     break;
        // }
        if stream_tx.send(token_msg).is_err() {
            finish_reason = FinishReason::Stop;
            break;
        }

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
    // let _ = stream_tx.send(complete_msg).await;
    let _ = stream_tx.send(complete_msg);

    Ok(())
}
