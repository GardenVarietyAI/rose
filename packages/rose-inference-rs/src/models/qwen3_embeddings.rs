use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use std::fs::File;

use super::Embeddings;

pub struct Qwen3Embeddings {
    model: Qwen3,
    device: Device,
    hidden_dim: usize,  // Model's native hidden dimension
    output_dims: Option<usize>,  // For Matryoshka truncation
}

impl Qwen3Embeddings {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dims(model_path, device, None)
    }

    pub fn load_with_dims(model_path: &str, device: &Device, output_dims: Option<usize>) -> Result<Self> {
        let mut file = File::open(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file {}: {}", model_path, e))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;

        let hidden_dim = content
            .metadata
            .get("qwen3.embedding_length")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                gguf_file::Value::U64(n) => Some(*n as usize),
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Cannot find qwen3.embedding_length in GGUF metadata")
            })?;

        let model = Qwen3::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 embeddings from GGUF: {}", e))?;

        if let Some(dims) = output_dims {
            if dims > hidden_dim {
                anyhow::bail!(
                    "Requested output dimensions {} exceed model's hidden dimension {}",
                    dims, hidden_dim
                );
            }
        }

        eprintln!("Loaded embedding model with hidden_dim={}, output_dims={:?}", hidden_dim, output_dims);

        Ok(Self {
            model,
            device: device.clone(),
            hidden_dim,
            output_dims,
        })
    }

    fn forward_embeddings(&mut self, input: &Tensor, seq_lengths: Option<&[usize]>) -> Result<Tensor> {
        self.model.forward_embeddings(input, 0, seq_lengths).map_err(Into::into)
    }
}

impl Embeddings for Qwen3Embeddings {
    fn encode(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Use batch implementation for consistency
        let batch_result = self.encode_batch(vec![tokens.to_vec()])?;
        batch_result.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("Unexpected empty batch result"))
    }

    fn encode_batch(&mut self, batch: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }

        const MAX_LENGTH: usize = 8192;
        const NORMALIZATION_EPSILON: f32 = 1e-12;
        const PAD_TOKEN: u32 = 0;  // Assuming 0 is pad token

        // Find max length and prepare for padding
        let batch_size = batch.len();
        let max_len = batch.iter()
            .map(|tokens| tokens.len().min(MAX_LENGTH))
            .max()
            .unwrap_or(0);

        // Build padded batch tensor
        let mut padded_batch = vec![PAD_TOKEN; batch_size * max_len];

        for (i, tokens) in batch.iter().enumerate() {
            let truncated_len = tokens.len().min(MAX_LENGTH);
            let truncated = &tokens[..truncated_len];

            let row_offset = i * max_len;
            padded_batch[row_offset..row_offset + truncated_len].copy_from_slice(truncated);
        }

        // Create batch tensor [batch_size, max_len]
        let input_tensor = Tensor::from_slice(
            &padded_batch,
            (batch_size, max_len),
            &self.device
        )?;

        // Get actual sequence lengths for each item in batch
        let seq_lengths: Vec<usize> = batch.iter()
            .map(|tokens| tokens.len().min(MAX_LENGTH))
            .collect();

        // Single batched forward pass with sequence lengths
        let hidden_states = self.forward_embeddings(&input_tensor, Some(&seq_lengths))?;

        // hidden_states is [batch_size, hidden_dim]
        // Convert to vec of vecs and normalize each
        let embeddings_matrix = hidden_states.to_vec2::<f32>()?;

        let mut embeddings = Vec::with_capacity(batch_size);
        for mut embedding_vec in embeddings_matrix {

            // Validate dimension
            if embedding_vec.len() != self.hidden_dim {
                anyhow::bail!(
                    "Model output dimension mismatch: expected {}, got {}",
                    self.hidden_dim,
                    embedding_vec.len()
                );
            }

            // Apply Matryoshka truncation if specified
            if let Some(dims) = self.output_dims {
                if dims < embedding_vec.len() {
                    embedding_vec.truncate(dims);
                }
            }

            // L2 normalize
            let norm = embedding_vec
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(NORMALIZATION_EPSILON);

            let normalized = embedding_vec.iter().map(|x| x / norm).collect();
            embeddings.push(normalized);
        }

        Ok(embeddings)
    }
}
