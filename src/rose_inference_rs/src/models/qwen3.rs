use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

use super::CausalLM;

pub struct Qwen3CausalLM {
    model: Qwen3,
    eos_token: u32,
}

impl Qwen3CausalLM {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let path = Path::new(model_path);

        // Determine actual GGUF file path
        let gguf_file_path = if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            // Direct path to GGUF file
            path.to_path_buf()
        } else {
            // Directory path - find GGUF file inside
            let mut found_gguf = None;
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let file_path = entry.path();
                    if file_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                        found_gguf = Some(file_path);
                        break;
                    }
                }
            }
            found_gguf.ok_or_else(|| anyhow::anyhow!("No GGUF file found in directory: {}", model_path))?
        };

        let mut file = File::open(&gguf_file_path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file {}: {}", gguf_file_path.display(), e))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;

        let model = Qwen3::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 from GGUF: {}", e))?;

        let eos_token = 151643u32;
        Ok(Self { model, eos_token })
    }
}

impl CausalLM for Qwen3CausalLM {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor> {
        self.model.forward(input, past_length).map_err(Into::into)
    }

    #[allow(unused_variables)]
    fn sample_logits(&self, _logits: &Tensor) -> Result<u32> {
        unimplemented!()
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token
    }

    fn im_end_token_id(&self, tokenizer: &Tokenizer) -> u32 {
        tokenizer
            .get_vocab(true)
            .get("<|im_end|>")
            .copied()
            .unwrap_or(self.eos_token)
    }
}
