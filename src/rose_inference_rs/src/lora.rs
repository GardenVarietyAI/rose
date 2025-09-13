use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_core::safetensors::Load;
use serde_json::Value;
use std::collections::HashMap;

/// LoRA configuration
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub r: usize,           // rank
    pub alpha: f32,         // alpha parameter
    #[allow(dead_code)]
    pub dropout: f32,       // dropout (not used in inference)
    pub target_modules: Vec<String>, // which modules to apply LoRA to
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

impl LoraConfig {
    pub fn from_json_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;

        let r = json.get("r").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        let alpha = json.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32;
        let dropout = json.get("lora_dropout").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;

        let target_modules = if let Some(modules) = json.get("target_modules").and_then(|v| v.as_array()) {
            modules.iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect()
        } else {
            LoraConfig::default().target_modules
        };

        Ok(Self {
            r,
            alpha,
            dropout,
            target_modules,
        })
    }
}

/// Loads LoRA adapter weights from safetensors file
pub fn load_lora_weights(adapter_path: &str, device: &Device) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let mut weights = HashMap::new();

    // Load the safetensors file
    let file_data = std::fs::read(adapter_path)?;
    let tensors = safetensors::SafeTensors::deserialize(&file_data)?;

    // Group LoRA weights by module name
    let mut lora_a_weights = HashMap::new();
    let mut lora_b_weights = HashMap::new();

    for (name, _) in tensors.tensors() {
        if name.ends_with(".lora_A.weight") {
            let mut module_name = name.strip_suffix(".lora_A.weight").unwrap();
            // Strip HuggingFace prefix (base_model.model.)
            if module_name.starts_with("base_model.model.") {
                module_name = module_name.strip_prefix("base_model.model.").unwrap();
            }
            let tensor = tensors.tensor(&name)?.load(device)?;
            lora_a_weights.insert(module_name.to_string(), tensor);
        } else if name.ends_with(".lora_B.weight") {
            let mut module_name = name.strip_suffix(".lora_B.weight").unwrap();
            // Strip HuggingFace prefix (base_model.model.)
            if module_name.starts_with("base_model.model.") {
                module_name = module_name.strip_prefix("base_model.model.").unwrap();
            }
            let tensor = tensors.tensor(&name)?.load(device)?;
            lora_b_weights.insert(module_name.to_string(), tensor);
        }
    }

    // Pair up A and B weights for each module
    for (module_name, lora_a) in lora_a_weights {
        if let Some(lora_b) = lora_b_weights.get(&module_name) {
            tracing::info!("LoRA pair {}: A={:?}, B={:?}", module_name, lora_a.dims(), lora_b.dims());
            weights.insert(module_name.clone(), (lora_a, lora_b.clone()));
        } else {
            tracing::warn!("Missing LoRA B weight for: {}", module_name);
        }
    }

    // Show first few loaded keys for debugging
    tracing::info!("Sample LoRA keys loaded:");
    for (key, _) in weights.iter().take(5) {
        tracing::info!("  {}", key);
    }

    Ok(weights)
}
