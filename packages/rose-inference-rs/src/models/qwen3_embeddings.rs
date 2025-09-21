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

    fn forward_embeddings(&mut self, input: &Tensor) -> Result<Tensor> {
        self.model.forward_embeddings(input, 0).map_err(Into::into)
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

        let hidden_states = self.forward_embeddings(&input_tensor)?;

        // The output is [batch, hidden_dim] for the last token
        // Since batch size is 1, flatten to get a 1D vector
        let mut embedding_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;

        if embedding_vec.len() != self.hidden_dim {
            anyhow::bail!(
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

        // L2 normalize (required after truncation for Matryoshka)
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
