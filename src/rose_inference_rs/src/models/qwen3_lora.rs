use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_lora::Qwen3LoraForCausalLM;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use std::path::Path;
use tokenizers::Tokenizer;

use super::CausalLM;
use crate::device::DeviceConfig;
use crate::lora::{load_lora_weights, LoraConfig};

/// Instead of replacing all layers, this intercepts specific linear operations and applies LoRA
pub struct Qwen3LoraModel {
    base_model: Qwen3LoraForCausalLM,
    eos_token: u32,
}

impl Qwen3LoraModel {
    pub fn load(
        model_path: &str,
        lora_adapter_path: Option<&str>,
        device: &Device,
    ) -> Result<Self> {
        let device_config = DeviceConfig::detect_dtypes(device);
        let model_dir = Path::new(model_path);

        // Load model config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Get EOS token
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let eos_token = config_json
            .get("eos_token_id")
            .and_then(|v| match v {
                serde_json::Value::Number(n) => n.as_u64(),
                serde_json::Value::Array(arr) => arr.first().and_then(|v| v.as_u64()),
                _ => None,
            })
            .unwrap_or(151645) as u32;

        // Load model weights
        let safetensors_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &safetensors_files,
                device_config.weights_dtype,
                &device_config.device,
            )?
        };

        if safetensors_files.is_empty() {
            return Err(anyhow::anyhow!(
                "No safetensors files found in {}",
                model_dir.display()
            ));
        }

        // Create base model with LoRA-aware layers
        let mut base_model = Qwen3LoraForCausalLM::new(
            &config,
            device_config.compute_dtype,
            device_config.kv_cache_dtype,
            vb,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Qwen3 (LoRA-aware): {}", e))?;

        // Load LoRA adapter if specified
        if let Some(adapter_path) = lora_adapter_path {
            tracing::info!("🎯 Loading LoRA adapter from: {}", adapter_path);

            let adapter_dir = Path::new(adapter_path);

            // Load LoRA config
            let config_path = adapter_dir.join("adapter_config.json");
            let lora_config = if config_path.exists() {
                let path_str = config_path.to_string_lossy();
                Some(LoraConfig::from_json_file(&path_str)?)
            } else {
                tracing::warn!("No adapter_config.json found, using default LoRA config");
                Some(LoraConfig::default())
            };

            // Load LoRA weights
            let adapter_weights_path = adapter_dir.join("adapter_model.safetensors");
            if adapter_weights_path.exists() {
                let weights_path = adapter_weights_path.to_string_lossy();
                let weights = load_lora_weights(&weights_path, device)?;
                tracing::info!("Loaded {} LoRA weight pairs", weights.len());

                // Apply immediately to the model
                if let Some(cfg) = &lora_config {
                    let scale = (cfg.alpha as f64) / (cfg.r as f64);
                    base_model.apply_adapter(&weights, scale);
                    tracing::info!(
                        "Applied LoRA adapter: scale alpha/r = {}/{}",
                        cfg.alpha,
                        cfg.r
                    );
                }
            } else {
                tracing::error!("No adapter_model.safetensors found in adapter directory");
            }
        } else {
            tracing::info!("🔄 No LoRA adapter specified, using base model only");
        }

        Ok(Self {
            base_model,
            eos_token,
        })
    }

    /// Switch LoRA adapter at runtime
    #[allow(dead_code)]
    pub fn switch_adapter(&mut self, adapter_path: Option<&str>, device: &Device) -> Result<()> {
        if let Some(adapter_path) = adapter_path {
            tracing::info!("Switching to LoRA adapter: {}", adapter_path);

            let adapter_dir = Path::new(adapter_path);
            let config_path = adapter_dir.join("adapter_config.json");
            let lora_config = if config_path.exists() {
                let path_str = config_path.to_string_lossy();
                Some(LoraConfig::from_json_file(&path_str)?)
            } else {
                Some(LoraConfig::default())
            };

            let adapter_weights_path = adapter_dir.join("adapter_model.safetensors");
            if adapter_weights_path.exists() {
                let weights_path = adapter_weights_path.to_string_lossy();
                let weights = load_lora_weights(&weights_path, device)?;
                tracing::info!(
                    "Loaded {} LoRA weight pairs for adapter switch",
                    weights.len()
                );
                let scale = lora_config
                    .as_ref()
                    .map(|c| (c.alpha as f64) / (c.r as f64))
                    .unwrap_or(1.0);
                self.base_model.apply_adapter(&weights, scale);
            } else {
                return Err(anyhow::anyhow!(
                    "No adapter_model.safetensors found in {}",
                    adapter_path
                ));
            }
        } else {
            tracing::info!("Removing LoRA adapter, using base model only");
            self.base_model.clear_adapter();
        }

        Ok(())
    }
}

impl CausalLM for Qwen3LoraModel {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor> {
        // Forward through LoRA-aware base model (LoRA applied inside if set)
        let logits = self
            .base_model
            .forward(input, past_length)
            .map_err(|e| anyhow::anyhow!("Base model forward failed: {}", e))?;
        Ok(logits)
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

    fn reset_state(&mut self) {
        self.base_model.clear_kv_cache();
    }
}
