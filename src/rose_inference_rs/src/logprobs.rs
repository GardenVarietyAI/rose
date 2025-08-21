use anyhow::Result;
use candle_core::Tensor;
use candle_nn::ops;
use tokenizers::Tokenizer;
use crate::types::TopLogProb;

#[derive(Debug, Clone)]
pub struct LogprobsData {
    pub logprob: f32,
    pub top_logprobs: Option<Vec<TopLogProb>>,
}

pub fn calculate_logprobs(
    logits: &Tensor,
    sampled_token: u32,
    tokenizer: &Tokenizer,
    top_k: Option<usize>,
) -> Result<LogprobsData> {
    // Apply softmax to get probabilities
    let softmax_logits = ops::softmax(logits, 0)?;

    // Get logprob for the sampled token
    let sampled_prob = softmax_logits.get(sampled_token as usize)?.to_scalar::<f32>()?;
    let sampled_logprob = sampled_prob.ln();

    // Calculate top-k logprobs if requested
    let top_logprobs = if let Some(k) = top_k {
        Some(calculate_top_logprobs(&softmax_logits, k, tokenizer)?)
    } else {
        None
    };

    Ok(LogprobsData {
        logprob: sampled_logprob,
        top_logprobs,
    })
}

fn calculate_top_logprobs(
    softmax_logits: &Tensor,
    k: usize,
    tokenizer: &Tokenizer,
) -> Result<Vec<TopLogProb>> {
    // Get vocabulary size
    let vocab_size = softmax_logits.dim(0)?;
    let k = k.min(vocab_size); // Ensure k doesn't exceed vocab size

    // Convert to Vec for sorting
    let probs: Vec<f32> = softmax_logits.to_vec1()?;

    // Create (index, prob) pairs and sort by probability descending
    let mut indexed_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k and convert to tokens
    let mut top_logprobs = Vec::with_capacity(k);
    for (token_id, prob) in indexed_probs.into_iter().take(k) {
        let token = tokenizer
            .decode(&[token_id as u32], false)
            .unwrap_or_else(|_| format!("<unk_{}>", token_id));

        top_logprobs.push(TopLogProb {
            token,
            logprob: prob.ln(),
        });
    }

    Ok(top_logprobs)
}
