use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use std::fs::File;

use super::Reranker;

pub struct Qwen3Reranker {
    model: Qwen3,
    device: Device,
    yes_token_id: u32,
    no_token_id: u32,
}

impl Qwen3Reranker {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let mut file = File::open(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file {}: {}", model_path, e))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;

        let model = Qwen3::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 reranker from GGUF: {}", e))?;

        let yes_token_id = 9693u32;
        let no_token_id = 2152u32;

        Ok(Self {
            model,
            device: device.clone(),
            yes_token_id,
            no_token_id,
        })
    }
}

impl Reranker for Qwen3Reranker {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.model.forward(input, 0).map_err(Into::into)
    }

    fn score(&mut self, query_tokens: &[u32], _doc_tokens: &[u32]) -> Result<f32> {
        const MAX_LENGTH: usize = 512;
        let truncated_tokens = if query_tokens.len() > MAX_LENGTH {
            &query_tokens[..MAX_LENGTH]
        } else {
            query_tokens
        };

        let input_tensor =
            Tensor::from_slice(truncated_tokens, (1, truncated_tokens.len()), &self.device)?;

        let logits = self.forward(&input_tensor)?;

        // The logits shape is [seq_len, vocab_size] for single batch
        // We want the last token's logits
        let dims = logits.dims();
        let seq_len = dims[0];
        let last_logits = logits.get(seq_len - 1)?;

        let yes_logit = last_logits
            .get(self.yes_token_id as usize)?
            .to_scalar::<f32>()?;
        let no_logit = last_logits
            .get(self.no_token_id as usize)?
            .to_scalar::<f32>()?;

        let max_logit = yes_logit.max(no_logit);
        let yes_exp = (yes_logit - max_logit).exp();
        let no_exp = (no_logit - max_logit).exp();
        let sum_exp = yes_exp + no_exp;

        let score = yes_exp / sum_exp;

        Ok(score)
    }
}
