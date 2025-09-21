use anyhow::{bail, Context, Result};
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
    max_length: usize,
}

impl Qwen3Embeddings {
    pub fn load(model_path: &str, device: &Device, output_dims: Option<usize>) -> Result<Self> {
        let mut file = File::open(model_path)
            .with_context(|| format!("Failed to open GGUF file {}", model_path))?;

        let content = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF content")?;

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

        let max_length = content
            .metadata
            .get("qwen3.context_length")
            .and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                gguf_file::Value::U64(n) => Some(*n as usize),
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Cannot find qwen3.context_length in GGUF metadata")
            })?;

        let model = Qwen3::from_gguf(content, &mut file, device)
            .context("Failed to create Qwen3 embeddings from GGUF")?;

        if let Some(dims) = output_dims {
            if dims == 0 {
                bail!("Requested output dimensions must be > 0");
            }
            if dims > hidden_dim {
                bail!(
                    "Requested output dimensions {} exceed model's hidden dimension {}",
                    dims, hidden_dim
                );
            }
        }

        eprintln!("Loaded embedding model with hidden_dim={}, max_length={}, output_dims={:?}", hidden_dim, max_length, output_dims);

        Ok(Self {
            model,
            device: device.clone(),
            hidden_dim,
            output_dims,
            max_length,
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
            .context("Unexpected empty batch result")
    }

    fn encode_batch(&mut self, batch: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(vec![]);
        }

        const NORMALIZATION_EPSILON: f32 = 1e-12;

        // Find max length and prepare for padding
        let batch_size = batch.len();
        let max_len = batch.iter()
            .map(|tokens| tokens.len().min(self.max_length))
            .max()
            .unwrap_or(0);

        // Edge case: when all inputs are empty return zero vectors with proper dims
        if max_len == 0 {
            let out_dim = self.output_dims.unwrap_or(self.hidden_dim);
            return Ok(vec![vec![0.0; out_dim]; batch_size]);
        }

        // Build padded batch tensor (pad with 0 to make rectangular)
        let mut padded_batch = vec![0u32; batch_size * max_len];

        for (i, tokens) in batch.iter().enumerate() {
            let t = tokens.len().min(self.max_length);
            let row = i * max_len;
            padded_batch[row..row + t].copy_from_slice(&tokens[..t]);
        }

        // Create batch tensor [batch_size, max_len]
        let input_tensor = Tensor::from_slice(
            &padded_batch,
            (batch_size, max_len),
            &self.device
        )?;

        // Get actual sequence lengths for each item in batch
        let seq_lengths: Vec<usize> = batch.iter()
            .map(|tokens| tokens.len().min(self.max_length))
            .collect();

        let hidden_states = self.forward_embeddings(&input_tensor, Some(&seq_lengths))?;

        // hidden_states is [batch_size, hidden_dim]
        // Convert to vec of vecs and normalize each
        let embeddings_matrix = hidden_states.to_vec2::<f32>()?;

        let mut embeddings = Vec::with_capacity(batch_size);
        for mut embedding_vec in embeddings_matrix {

            // Safety check
            if embedding_vec.len() != self.hidden_dim {
                bail!(
                    "Model output dimension mismatch: expected {}, got {}",
                    self.hidden_dim,
                    embedding_vec.len()
                );
            }

            if let Some(dims) = self.output_dims {
                if dims < embedding_vec.len() {
                    embedding_vec.truncate(dims);
                }
            }

            // L2 normalize
            let mut ss = 0.0f32;
            for &x in embedding_vec.iter() {
                ss += x * x;
            }
            let norm = ss.sqrt().max(NORMALIZATION_EPSILON);
            for x in embedding_vec.iter_mut() {
                *x /= norm;
            }

            embeddings.push(embedding_vec);
        }

        Ok(embeddings)
    }
}
