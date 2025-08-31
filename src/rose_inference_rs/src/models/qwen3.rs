use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::models::qwen3::{ModelForCausalLM, Config};
use candle_nn::VarBuilder;
use std::path::Path;
use tokenizers::Tokenizer;

use super::CausalLM;
use crate::dtype_config::DTypeConfig;

pub struct Qwen3UnquantizedCausalLM {
    model: ModelForCausalLM,
    eos_token: u32,
}

impl Qwen3UnquantizedCausalLM {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let dtype_config = DTypeConfig::auto_detect(device);
        Self::load_with_config(model_path, device, dtype_config)
    }

    pub fn load_with_config(model_path: &str, device: &Device, dtype_config: DTypeConfig) -> Result<Self> {
        // Validate dtype config for this device first (fail fast)
        dtype_config.validate(device).map_err(|e| anyhow::anyhow!("DType config validation failed: {}", e))?;

        let model_dir = Path::new(model_path);

        // Load config to get model parameters
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

        // Get EOS token from config
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let eos_token = config_json
            .get("eos_token_id")
            .and_then(|v| match v {
                serde_json::Value::Number(n) => n.as_u64(),
                serde_json::Value::Array(arr) => arr.first().and_then(|v| v.as_u64()),
                _ => None,
            })
            .unwrap_or(151645) as u32;

        // Load safetensors using VarBuilder
        let safetensors_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();

        if safetensors_files.is_empty() {
            return Err(anyhow::anyhow!("No safetensors files found in {}", model_dir.display()));
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype_config.weights_dtype, device)? };
        let model = ModelForCausalLM::new(&config, vb)
            .map_err(|e| anyhow::anyhow!("Failed to load Qwen3 from safetensors: {}", e))?;

        Ok(Self {
            model,
            eos_token,
        })
    }
}

impl CausalLM for Qwen3UnquantizedCausalLM {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor> {
        // Input tokens should remain as integers for embedding lookup - do not cast
        self.model.forward(input, past_length).map_err(Into::into)
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
