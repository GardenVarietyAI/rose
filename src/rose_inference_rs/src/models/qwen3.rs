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
        eprintln!("Loading Qwen3 model from: {}", model_path);
        let path = Path::new(model_path);

        // Determine actual GGUF file path
        let gguf_file_path = if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            eprintln!("Direct GGUF file path detected");
            // Direct path to GGUF file
            path.to_path_buf()
        } else {
            eprintln!("Directory path detected, searching for GGUF files");
            // Directory path - find GGUF file inside
            let mut found_gguf = None;
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let file_path = entry.path();
                    eprintln!("Checking file: {:?}", file_path);
                    if file_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                        eprintln!("Found GGUF file: {:?}", file_path);
                        found_gguf = Some(file_path);
                        break;
                    }
                }
            }
            found_gguf.ok_or_else(|| anyhow::anyhow!("No GGUF file found in directory: {}", model_path))?
        };

        eprintln!("Opening GGUF file: {:?}", gguf_file_path);
        let mut file = File::open(&gguf_file_path)
            .map_err(|e| anyhow::anyhow!("Failed to open GGUF file {}: {}", gguf_file_path.display(), e))?;

        eprintln!("Reading GGUF content...");
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;

        eprintln!("Creating Qwen3 model from GGUF...");
        let model = Qwen3::from_gguf(content, &mut file, device)
            .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 from GGUF: {}", e))?;

        let eos_token = 151643u32;
        eprintln!("Qwen3 model loaded successfully");
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
