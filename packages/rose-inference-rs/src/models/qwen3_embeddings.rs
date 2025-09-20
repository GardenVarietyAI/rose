use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use std::fs::File;

use super::Embeddings;

pub struct Qwen3Embeddings {
    model: Qwen3,
    device: Device,
}

impl Qwen3Embeddings {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let mut file = File::open(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file {}: {}", model_path, e))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;

        let model = Qwen3::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 embeddings from GGUF: {}", e))?;

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.model.forward(input, 0).map_err(Into::into)
    }
}

impl Embeddings for Qwen3Embeddings {
    fn encode(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Maximum context length for Qwen3 embeddings model
        // Matches the model's training context window to prevent OOM errors
        const MAX_LENGTH: usize = 8192;

        // Small epsilon to prevent division by zero during L2 normalization
        const NORMALIZATION_EPSILON: f32 = 1e-12;

        let truncated_tokens = if tokens.len() > MAX_LENGTH {
            &tokens[..MAX_LENGTH]
        } else {
            tokens
        };

        let input_tensor =
            Tensor::from_slice(truncated_tokens, (1, truncated_tokens.len()), &self.device)?;

        let logits = self.forward(&input_tensor)?;

        // The output is [seq_len, hidden_dim] for a single sequence
        let dims = logits.dims();
        let seq_len = dims[0];
        let last_token_embedding = logits.get(seq_len - 1)?;

        let embedding_vec = last_token_embedding.to_vec1::<f32>()?;

        let norm = embedding_vec
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(NORMALIZATION_EPSILON);
        let normalized = embedding_vec.iter().map(|x| x / norm).collect();

        Ok(normalized)
    }

    fn encode_batch(&mut self, batch: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(batch.len());

        for tokens in batch {
            embeddings.push(self.encode(&tokens)?);
        }

        Ok(embeddings)
    }
}
