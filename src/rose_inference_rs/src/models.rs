use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::models::qwen2::{ModelForCausalLM, Config};
use candle_transformers::generation::LogitsProcessor;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::info;
use std::path::Path;

use crate::error::InferenceError;

pub async fn generate_streaming(
    model_path: &str,
    device: Device,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    stream_tx: mpsc::Sender<crate::server::InferenceResponse>,
) -> Result<()> {
    // Load tokenizer directly
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    info!("Loading tokenizer from: {}", tokenizer_path);

    let tokenizer = if Path::new(&tokenizer_path).exists() {
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| InferenceError::TokenizerError(format!("Failed to load tokenizer: {}", e)))?
    } else {
        return Err(InferenceError::TokenizerError(format!("Tokenizer not found at: {}", tokenizer_path)).into());
    };

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
    let ids = encoding.get_ids();
    let prompt_tokens = ids.len() as u32;
    let tokens = ids.to_vec();

    info!("Starting generation with {} prompt tokens, max_tokens: {}, temperature: {}",
          prompt_tokens, max_tokens, temperature);
    info!("Raw prompt received: {:?}", prompt);

    // Load model directly
    info!("Loading Qwen2 model from: {:?}", model_path);
    let config_path = Path::new(model_path).join("config.json");
    let config_json = std::fs::read_to_string(&config_path)?;
    let qwen_config: Config = serde_json::from_str(&config_json)?;

    let weights_path = Path::new(model_path).join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)? };
    let mut model = ModelForCausalLM::new(&qwen_config, vb)?;

    info!("Model loaded successfully");

    // Send input tokens counted
    let input_tokens_msg = crate::server::InferenceResponse::InputTokensCounted {
        input_tokens: prompt_tokens,
    };
    if stream_tx.send(input_tokens_msg).await.is_err() {
        info!("Stream receiver dropped during input tokens, stopping");
        return Ok(());
    }
    info!("Sent input tokens count: {}", prompt_tokens);

    // Setup sampling with API parameters - use lower temperature for better quality
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
            }
        }
    };

    let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

    // Process initial prompt
    info!("Starting initial forward pass with {} tokens", tokens.len());
    let mut next_token = {
        let input = Tensor::from_slice(&tokens, (1, tokens.len()), &device)?;
        info!("Created input tensor, running initial forward");
        let logits = model.forward(&input, 0)?;
        info!("Initial forward completed, logits shape: {:?}", logits.shape());
        // ModelForCausalLM returns [1, 1, vocab_size] - just squeeze the first two dims
        let logits = logits.squeeze(0)?.squeeze(0)?;
        info!("Logits after squeeze: {:?}", logits.shape());
        logits_processor.sample(&logits)?
    };

    let mut all_tokens = tokens.clone();
    all_tokens.push(next_token);

    // Send first token
    let token_text = tokenizer
        .decode(&[next_token], true)
        .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;

    let token_msg = crate::server::InferenceResponse::Token {
        token: token_text.clone(),
        position: 0,
        logprob: None,
        top_logprobs: None,
    };
    if stream_tx.send(token_msg).await.is_err() {
        return Ok(());
    }

    // Generate remaining tokens
    let repeat_penalty = 1.1f32;
    let repeat_last_n = 64;
    let to_sample = max_tokens.saturating_sub(1);

    info!("Starting generation loop for {} tokens", to_sample);
    for index in 0..to_sample {
        let input = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let logits = model.forward(&input, all_tokens.len() - 1)?;
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

        // Stream token immediately after decoding
        let token_msg = crate::server::InferenceResponse::Token {
            token: token_text.clone(),
            position: index + 1,
            logprob: None,
            top_logprobs: None,
        };
        if stream_tx.send(token_msg).await.is_err() {
            break;
        }

        // Check for EOS token - Qwen2.5 uses 151645 as EOS
        let eos_token = 151645u32;
        let im_end_token = tokenizer.get_vocab(true).get("<|im_end|>").copied().unwrap_or(151643);

        if next_token == eos_token || next_token == im_end_token {
            info!("Generation stopped at token {} (EOS: {}, im_end: {})", index + 1, eos_token, im_end_token);
            break;
        }

        // Break on repetitive tokens (prevent infinite loops)
        if index > 10 {
            let recent_tokens = &all_tokens[all_tokens.len().saturating_sub(8)..];
            if recent_tokens.len() >= 8 && recent_tokens.iter().all(|&t| t == next_token) {
                info!("Breaking repetitive token loop at token {} (repeated token: {})", index + 1, next_token);
                break;
            }
        }

        // Hard limit to prevent infinite loops
        if index >= max_tokens {
            info!("Max tokens reached, stopping at {}", index + 1);
            break;
        }
    }

    // Send completion
    let output_tokens = all_tokens.len() as u32;
    let complete_msg = crate::server::InferenceResponse::Complete {
        input_tokens: prompt_tokens,
        output_tokens,
        total_tokens: prompt_tokens + output_tokens,
    };
    let _ = stream_tx.send(complete_msg).await;

    Ok(())
}
