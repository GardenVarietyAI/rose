use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use std::path::Path;
use tokenizers::Tokenizer;

use super::CausalLM;

pub struct Qwen2CausalLM {
    model: ModelForCausalLM,
    eos_token: u32,
}

impl Qwen2CausalLM {
    pub fn load(model_path: &str, device: &Device) -> Result<Self> {
        let config_path = Path::new(model_path).join("config.json");
        let config_json = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_json)?;
        let weights_path = Path::new(model_path).join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, device)?
        };
        let model = ModelForCausalLM::new(&config, vb)?;
        let eos_token = 151645u32;
        Ok(Self { model, eos_token })
    }
}

impl CausalLM for Qwen2CausalLM {
    fn forward(&mut self, input: &Tensor, past_length: usize) -> Result<Tensor> {
        self.model.forward(input, past_length).map_err(Into::into)
    }

    #[allow(unused_variables)]
    fn sample_logits(&self, _logits: &Tensor) -> Result<u32> {
        // TODO: Implement proper sampling (use LogitsProcessor or similar)
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
